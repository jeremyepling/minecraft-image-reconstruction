"""Replay Surface Finder boundaries, crop coverage and parallel-plane separation on the frozen pictures."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_round import FINDER, cached_regions
from benchmark.count_metrics import count_totals,reference_planes
from benchmark.mask_metrics import PROTOCOL, aggregate, distances, footprint, measure, outline
from benchmark.normal_fields import cached_fields, fields
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


COUNT=ROOT/'benchmark/reviews/count-01'
METHODS=('regions-first','color-edges','depth-split')
RULE=('Choose on shopfront and wing, weighting each layout equally. Require either a 0.10 pixel reduction '
      'in mean edge error or a 0.01 gain in surface F1. Allow at most 0.10 pixels worse edge error, '
      '0.002 lower F1, 0.002 lower assigned pixel fraction and 0.003 higher opening fill. '
      'Among qualifying candidates prioritize F1, then lower boundary error.')


def summarize(rows):
    return {name:aggregate([r['scores'][name] for r in rows]) for name in METHODS}


def decision(rows):
    layouts=[summarize([r for r in rows if r['layout']==layout]) for layout in sorted({r['layout'] for r in rows})]
    ratings={m:{k:float(np.mean([g[m][k] for g in layouts])) for k in
                ('edge_error_px','f1','assigned_fraction','opening_fill')} for m in METHODS}
    base=ratings[METHODS[0]]
    eligible=[METHODS[0]]
    for m in METHODS[1:]:
        v=ratings[m]
        edge=base['edge_error_px']-v['edge_error_px']
        f1=v['f1']-base['f1']
        if ((edge>=.10 or f1>=.01) and edge>=-.10 and f1>=-.002
            and v['assigned_fraction']>=base['assigned_fraction']-.002 and v['opening_fill']<=base['opening_fill']+.003):
            eligible.append(m)
    return dict(selected=max(eligible,key=lambda m:(ratings[m]['f1'],-ratings[m]['edge_error_px'])),
                rule=RULE,ratings=ratings)


def overlap(mask, target):
    intersection=int((mask & target).sum())
    a,b=distances(outline(mask),outline(target))
    return dict(iou=intersection/max(1,int((mask|target).sum())),correct_samples=intersection,
                mean_border_error_px=float((a.sum()+b.sum())/(len(a)+len(b))) if len(a)+len(b) else None)


def check_crop(image,part):
    """Verify every source pixel, including interior holes, without reference labels."""
    from engine.surface_perimeter import masked_crop, outline as rings
    support=part['support']
    w,h=support['size']
    mask=np.zeros((h,w),bool)
    for y,a,b in support['runs']:
        mask[y,a:b]=True
    # Rasterize rings at cell centers by crossing each vertical edge from left to right.
    flips=np.zeros((h,w+1),bool)
    shape=rings(support)
    for ring in shape['rings']:
        points=ring['points']
        for (x,y),(xx,yy) in zip(points,points[1:]+points[:1]):
            if x==xx:
                flips[min(y,yy):max(y,yy),x]^=True
            elif y!=yy:
                raise ValueError('A traced ring cuts diagonally through a mask cell')
    if not np.array_equal(np.logical_xor.accumulate(flips,axis=1)[:,:-1],mask):
        raise ValueError('The perimeter changed the covered pixels')
    cropped=masked_crop(image,part)
    mx=np.floor((np.arange(image.width)+.5)*w/image.width).astype(int)
    my=np.floor((np.arange(image.height)+.5)*h/image.height).astype(int)
    native=mask[np.ix_(my,mx)]
    if cropped is None:
        if native.any():
            raise ValueError('A nonempty region has no crop')
        return 0,None
    crop,box=cropped
    x,y,xx,yy=box
    expected=native[y:yy,x:xx]
    if native.sum()!=expected.sum() or not np.array_equal(np.asarray(crop.getchannel('A')),expected.astype(np.uint8)*255):
        raise ValueError('The native cutout changed the covered pixels')
    if not np.array_equal(np.asarray(crop)[:,:,:3][expected],np.asarray(image.crop(box))[expected]):
        raise ValueError('The surface cutout changed source RGB')
    return len(shape['rings']),box


MARGINS=(0,4,8,16)


def context_diagnostic(parts,boxes,reference,matches):
    """Count available edge evidence in the RGB crop independently of its ownership mask."""
    stride=PROTOCOL['sample_stride']
    ids=reference['id_map']
    y=(np.arange(ids.shape[0])+.5)*stride
    x=(np.arange(ids.shape[1])+.5)*stride
    counts={p:Counter() for p in MARGINS}
    for i,pid in matches:
        if boxes[i] is None:
            continue
        a,b,c,d=boxes[i]
        mask=footprint(parts[i],ids.shape,reference['image_size'],True)
        missed=(ids==pid)&~mask
        for pad in MARGINS:
            inside=np.ix_((y>=b-pad)&(y<d+pad),(x>=a-pad)&(x<c+pad))
            context=~mask[inside]
            counts[pad].update(matched_surfaces=1,missed_target_samples=int(missed.sum()),
                               missed_target_visible_in_context=int(missed[inside].sum()),
                               other_surface_or_background_in_context=int((context&(ids[inside]!=pid)).sum()))
    return counts


def chart_diagnostic(parts,count_row,reference,matches):
    ids=reference['id_map']
    targets=dict(matches)
    charts={p['id']:p['quad'] for p in count_row['surfaces']}
    stats=Counter()
    for i,p in enumerate(parts):
        mask=footprint(p,ids.shape,reference['image_size'],True)
        if i not in targets:
            continue
        true=ids==targets[i]
        chart=charts[p['id']]
        window=footprint({'quad':chart},ids.shape,reference['image_size'],False) if chart is not None else np.zeros_like(mask)
        correct=mask&true
        stats.update(matched_regions=1,correct_samples_in_surface=int(correct.sum()),
                     correct_samples_in_old_chart=int((correct&window).sum()))
    return dict(stats)


def downstream_counts(image,planes):
    from engine.masked_count import count
    from engine.surface_dimensions import dimensions
    from engine.surface_sample import sample_surface
    rows=[]
    for part in planes:
        sample=sample_surface(image,part)
        reading=count(sample)
        q=sample['quad'] if sample is not None else None
        reading['dimensions']=dimensions(part,q,image.size,reading['counts'])
        rows.append(dict(id=part['id'],quad=q,expected_chart=None,measures={'masked-repeat':reading}))
    return rows


def run(out, field_cache):
    from engine.surface_boundary_refine import SETTINGS as color_settings, refine
    from engine.surface_depth_split import SETTINGS as depth_settings, split
    from benchmark.perimeter_sheet import make_sheet
    frozen=read(FINDER/'manifest.json')
    counted=read(COUNT/'manifest.json')
    cohorts,suite,fixtures=groups()
    if fixtures!=frozen['fixtures'] or digest(ARCHIVE)!=frozen['reference_sha256']:
        raise ValueError('The recorded fixture changed')
    for path,sha in [(FINDER/'cases.json',frozen['artifact_sha256']['cases.json']),
                     *[(COUNT/file,counted['artifacts'][file]) for file in ('cases.json','review-regions.json')]]:
        if digest(path)!=sha:
            raise ValueError(f'Frozen input changed: {path}')
    out.mkdir(parents=True,exist_ok=False)
    stamp=revision()
    lookup={f'{case.floor}/{case.id}':case for group in cohorts.values() for case in group}
    originals=read(FINDER/'cases.json')
    counts={r['key']:r for r in read(COUNT/'cases.json')}
    views={r['frame']:r for r in read(COUNT/'review-regions.json')}
    results=[]
    prepared=False
    for cohort in ('development','previously_reviewed_regression'):
        for row in [r for r in originals if r['cohort']==cohort]:
            key=row['key']
            case=lookup[key]
            if digest(case.path('image'))!=row['sha256'] or row['sha256']!=counts[key]['sha256']:
                raise ValueError(f'Input screenshot changed: {key}')
            with Image.open(case.path('image')) as src:
                image=src.convert('RGB')
            parts=cached_regions(row,frozen)
            cached=cached_fields(row['sha256'],field_cache,kinds=('depth',))
            if cached is None:
                if not prepared:
                    from benchmark.normal_assets import prepare
                    prepare(download=True)
                    prepared=True
                _,depth=fields(case.path('image'),row['sha256'],out.parent/(out.name+'-fields'))
            else:
                depth=cached['depth']
            predictions={METHODS[0]:parts}
            timing={}
            for name,fn in [(METHODS[1],lambda:refine(image,parts,version='26.3')),
                            (METHODS[2],lambda:split(image,parts,depth,version='26.3'))]:
                t=perf_counter()
                predictions[name]=fn()
                timing[name]=perf_counter()-t
            # Store changed predictions before reading reference geometry or choosing an illustration.
            path=out/'predictions'/(key+'.json.gz')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps({m:predictions[m] for m in METHODS[1:]},
                                   separators=(',',':'),allow_nan=False).encode(),mtime=0))
            downstream={m:downstream_counts(image,predictions[m]) for m in METHODS[1:]}
            _,reference=recorded(case)
            scores={METHODS[0]:row['scores']['mask-first']}
            indices={p['id']:i for i,p in enumerate(parts)}
            pairs={METHODS[0]:[(indices[x['proposal']],x['recorded_surface']) for x in row['matched_regions']['mask-first']]}
            for name in METHODS[1:]:
                scores[name],pairs[name]=measure(predictions[name],reference,use_mask=True)
            plane_sizes=reference_planes(reference,read(case.path('labels'))['faces'])
            dims={METHODS[0]:count_totals([counts[key]],'masked-repeat')}
            for name,readings in downstream.items():
                targets=dict(pairs[name])
                for i,reading in enumerate(readings):
                    reading['expected']=plane_sizes[targets[i]]['size'] if i in targets else None
                dims[name]=count_totals([dict(eligible=scores[name]['eligible'],surfaces=readings)],'masked-repeat')
            audits={}
            margins={}
            for m,planes in predictions.items():
                validated=[check_crop(image,p) for p in planes]
                audits[m]=dict(regions=len(planes),rings=sum(n for n,_ in validated))
                margins[m]=context_diagnostic(planes,[box for _,box in validated],reference,pairs[m])
            diag=chart_diagnostic(parts,counts[key],reference,pairs[METHODS[0]])
            result=dict(key=key,sha256=row['sha256'],layout=row['layout'],cohort=cohort,scores=scores,crop_audit=audits,
                        timing_seconds=timing,cache='cached' if cached is not None else 'generated',
                        numerical_depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                        chart_diagnostic=diag,context_margin=margins,block_counts=dims,predictions=path.relative_to(out).as_posix())
            if key in views:
                view=views[key]
                target=dict(pairs[METHODS[0]])[indices[view['proposal']]]
                ids=reference['id_map']
                true=ids==target
                zoom=dict(region=view['proposal'],recorded_surface=target,expected=plane_sizes[target]['size'],scene_box=view['region_xyxy'],
                          old_chart=view['chart_xy'],methods={})
                quad=footprint({'quad':view['chart_xy']},ids.shape,image.size,False)
                zoom['quad_as_outline']=overlap(quad,true)
                for m,planes in predictions.items():
                    by_reference={pid:i for i,pid in pairs[m]}
                    chosen=by_reference.get(target)
                    if chosen is None:
                        # For illustrations only, follow the best-overlapping descendant even when it lost the match.
                        descendants=[i for i,p in enumerate(planes) if p.get('parent_id',p['id'])==view['proposal']]
                        chosen=max(descendants,key=lambda i:int((footprint(planes[i],ids.shape,image.size,True)&true).sum()),default=None)
                    if chosen is not None:
                        mask=footprint(planes[chosen],ids.shape,image.size,True)
                        zoom['methods'][m]=dict(id=planes[chosen]['id'],matched=target in by_reference,
                                                fraction_in_old_chart=int((mask&quad).sum())/max(1,int(mask.sum())),**overlap(mask,true))
                result['review']=zoom
            results.append(result)
            if len(results)%8==0:
                print(f'{len(results)}/56 pictures',flush=True)
        if cohort=='development':
            selected=decision(results)
            save(out/'selection.json',selected)
    save(out/'cases.json',results)
    summary=dict(all=summarize(results))
    for k in ('cohort','layout'):
        summary[k]={v:summarize([r for r in results if r[k]==v]) for v in sorted({r[k] for r in results})}
    margins={k:dict(sum((r['context_margin'][selected['selected']][k] for r in results),Counter())) for k in MARGINS}
    save(out/'context-margins.json',dict(native_pixel_margins=margins,selected=selected['selected'],
        unit='One scoring sample per 4 by 4 native pixels. Counts pooled over matched surface appearances.',
        comparison='Extra RGB is available for inspection. Only supported pixels count as a surface.'))
    for cohort,rows in [('all',results),('development',[r for r in results if r['cohort']=='development']),
                       ('regression',[r for r in results if r['cohort']!='development'])]:
        summary.setdefault('block_counts',{})[cohort]={m:{k:sum(r['block_counts'][m][k] for r in rows)
            for k in ('eligible','correct','incorrect','missing')} for m in METHODS}
    save(out/'scores.json',summary)
    from benchmark.perimeter_context_count import run as run_context_counts
    # An independent image-only regeneration also checks that the margin experiment starts from the same masks.
    run_context_counts(out/'count-context')
    context_scores=read(out/'count-context/scores.json')['all']
    control=context_scores['tight']
    if any(control[k]!=summary['block_counts']['all']['color-edges'][k] for k in ('eligible','correct','incorrect','missing')):
        raise ValueError('The tight-crop control differs from the Surface Finder comparison')
    make_sheet(out,results,lookup,originals,frozen,selected['selected'])
    manifest=dict(run_id=out.name,code=stamp,stage='Surface Finder',version=suite['version'],fixtures=fixtures,
        models=frozen['shared_models'],capture=frozen['capture'],reference_sha256=digest(ARCHIVE),
        baseline_manifest_sha256=digest(FINDER/'manifest.json'),review_source_manifest_sha256=digest(COUNT/'manifest.json'),
        methods=list(METHODS),settings={'color-edges':color_settings,'depth-split':depth_settings,'scoring':PROTOCOL},
        selection=selected,aggregate=summary['all'],frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in results],
        artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()},
        reproduce=r'.\Invoke-SurfaceRound.ps1 -Round perimeters',
        runtime='Frozen MoGe-2 and Depth Anything V2 predictions. Per-picture timings cover refinement only.')
    save(out/'manifest.json',manifest)
    print(out,flush=True)


@image_only()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--fields',type=Path,help='Read-only compatible numerical model cache')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out=(args.out or ROOT/'runs'/f'perimeters-{stamp}-{uuid4().hex[:6]}').resolve()
    options=[ROOT/'runs/normal-fields',ROOT.parent/'normal-round/runs/normal-fields']
    cache=args.fields or next((p for p in options if p.is_dir()),options[0])
    run(out,cache)


if __name__=='__main__':
    main()
