"""Compare named ECT boundary causes and ordinary RGB contrast on the same saved outlines."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageFilter

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import footprint, outline
from benchmark.surface_metrics import _one_to_one
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import BASE, SAM, SOURCE_METHOD, capture_index, checked, read_gzip
from engine.normal_regions import _patch
from engine.surface_model import canonical
from runtime.mode import image_only


COMMON = dict(min_area=24, min_quality=.62, min_stability=.62, min_anchor_share=.18,
              max_scene_area_share=.86, max_existing_iou=.72, max_selected_iou=.48,
              max_selected_containment=.87, min_novel_boundary=6, edge_tolerance_stride4=1)
# Identical budgets give each kind of boundary evidence the same opportunity to add shapes.
PRESETS = {f'{kind}-{budget}': dict(kind=kind, budget=budget)
           for kind in ('rgb','geometry','cause-margin') for budget in (1,3,6)}


def maximum3(array):
    return np.asarray(Image.fromarray(array).filter(ImageFilter.MaxFilter(3)))


def edge_fields(image, causes):
    h,w = causes['depth'].shape
    rgb = np.asarray(image.resize((w,h),Image.Resampling.BOX),np.float32)
    dx = np.abs(np.diff(rgb,axis=1)).mean(axis=2)
    dy = np.abs(np.diff(rgb,axis=0)).mean(axis=2)
    rgb_edge = np.zeros((h,w),np.float32)
    rgb_edge[:,1:] = dx
    rgb_edge[:,:-1] = np.maximum(rgb_edge[:,:-1],dx)
    rgb_edge[1:] = np.maximum(rgb_edge[1:],dy)
    rgb_edge[:-1] = np.maximum(rgb_edge[:-1],dy)
    # A soft saturation retains RGB contrasts of different strengths on one bounded scale.
    rgb_edge = np.rint(255*(1-np.exp(-rgb_edge/24))).astype('uint8')
    geo = np.maximum(causes['depth'],causes['normal'])
    appearance = np.maximum(causes['reflectance'],causes['illumination'])
    return [maximum3(a).astype(np.float32)/255 for a in (rgb_edge,geo,appearance)]


def inventory(image, parts, masks, records, causes):
    """Do not open reference data. Each further method receives exactly these same masks."""
    shape = masks.shape[1:]
    fields = edge_fields(image,causes)
    prior = np.stack([footprint(p,shape,image.size,True) for p in parts])
    areas = prior.sum(axis=(1,2))
    building = prior.any(axis=0)
    scene_area = building.sum()
    old_edges = np.zeros(shape,'uint8')
    for plane in prior:
        points = outline(plane)
        old_edges[points[:,0],points[:,1]] = 1
    old_edges = maximum3(old_edges).astype(bool)
    rejected, events, seen = Counter(), [], set()
    # Evaluate the strongest observation of each identical mask only once.
    order = sorted(range(len(records)),key=lambda i:(-records[i]['quality'],-records[i]['stability'],i))
    for i in order:
        r = records[i]
        area = r['area']
        if (area<COMMON['min_area'] or area>scene_area*COMMON['max_scene_area_share'] or
                r['quality']<COMMON['min_quality'] or r['stability']<COMMON['min_stability']):
            rejected['size_or_quality'] += 1
            continue
        x0,y0,x1,y1 = r['bbox']
        # The MobileSAM archive stores half-open boxes on the stride-four image grid.
        cropped = masks[i,y0:y1,x0:x1]
        ys,xs = np.nonzero(cropped)
        ys,xs = ys+y0,xs+x0
        if len(xs)!=area:
            raise ValueError('MobileSAM bounding box or archived area disagrees with its mask')
        identity = (x0,y0,x1,y1,np.packbits(cropped,axis=None).tobytes())
        if identity in seen:
            rejected['duplicate'] += 1
            continue
        seen.add(identity)
        anchor_share = float(building[ys,xs].mean())
        if anchor_share<COMMON['min_anchor_share']:
            rejected['weak_anchor'] += 1
            continue
        hits = prior[:,ys,xs].sum(axis=1)
        iou = hits/(areas+area-hits)
        if iou.max()>=COMMON['max_existing_iou']:
            rejected['already_delineated'] += 1
            continue
        by,bx = outline(masks[i]).T
        novel = ~old_edges[by,bx]
        if int(novel.sum())<COMMON['min_novel_boundary']:
            rejected['no_new_boundary'] += 1
            continue
        # Existing silhouettes can make an incorrect new dividing line look convincing.
        # Give the boundary pixels unique to the proposal most of the vote.
        raw,geo,appearance = [v[by,bx] for v in fields]
        def strength(values):
            return float(.70*values[novel].mean()+.30*values.mean())
        scores = dict(rgb=strength(raw), geometry=strength(geo))
        scores['cause-margin'] = strength(geo-.55*appearance)
        events.append(dict(proposal=i, parent=int(np.argmax(hits)), area=area, anchor_share=anchor_share,
                           best_prior_iou=float(iou.max()), novel_rim=int(novel.sum()), rim=len(by),
                           quality=r['quality'], stability=r['stability'], scores=scores))
    return events, prior, rejected


def select(events, masks, settings):
    chosen = []
    for e in sorted(events, key=lambda e:(-e['scores'][settings['kind']],-e['quality'],-e['stability'],e['proposal'])):
        mask = masks[e['proposal']]
        for previous in chosen:
            shared = int(np.count_nonzero(mask & masks[previous['proposal']]))
            if (shared/(e['area']+previous['area']-shared)>=COMMON['max_selected_iou'] or
                    shared/min(e['area'],previous['area'])>=COMMON['max_selected_containment']):
                break
        else:
            chosen.append(e)
            if len(chosen)>=settings['budget']:
                break
    return chosen


def part_from_mask(image, gray, parts, masks, event):
    mask = masks[event['proposal']]
    y,edge = np.nonzero(np.diff(np.pad(mask.astype('int8'),((0,0),(1,1))),axis=1))
    runs = np.column_stack((y[::2],edge[::2],edge[1::2])).tolist()
    size = list(mask.shape[::-1])
    anchor = parts[event['parent']]
    normal = np.asarray(anchor['normal_camera'])
    result = dict(id=f"outline-afterpass-{event['proposal']}", support=dict(runs=runs,size=size),
                  support_area=event['area'], normal_camera=normal.tolist(), relative_inverse_depth=None,
                  geometry_status='mask only', orientation_source=anchor['id'], mobile_sam_proposal=event['proposal'])
    chart = _patch(runs,gray,np.asarray(image.size)/size,size,normal)
    if chart:
        result.update(quad=canonical(chart['quad']),geometry_status='mask with sampling chart')
    return result


def load_inputs(case, row, source, fields, meta):
    def path(name, rel):
        return checked(source/'benchmark/reviews'/name/rel,meta[name]['artifacts'][rel])
    sha = row['sha256']
    with Image.open(checked(case.path('image'),sha)) as img:
        image = img.convert('RGB')
    parts = read_gzip(path(BASE,row['predictions']))[SOURCE_METHOD]
    name = 'proposals/'+sha
    with np.load(path(SAM,name+'.npz'),allow_pickle=False) as data:
        masks = np.unpackbits(data['packed'],axis=2,count=int(data['shape'][2])).astype(bool)
    records = read_gzip(path(SAM,name+'.json.gz'))['candidates']
    rel = 'fields/'+sha+'.npz'
    with np.load(checked(fields/rel,meta['ect']['artifacts'][rel]),allow_pickle=False) as data:
        causes = {k:data[k] for k in ('depth','normal','reflectance','illumination')}
    if (len(records)!=len(masks) or masks.shape[1:]!=(image.height//4,image.width//4) or
            any(a.dtype!=np.uint8 or a.shape!=masks.shape[1:] for a in causes.values())):
        raise ValueError('Model and source pixel grids disagree')
    return image,parts,masks,records,causes


def possible(mask, ids, areas, eligible):
    hits = np.bincount(ids[mask],minlength=len(areas))[eligible]
    iou = hits/(int(mask.sum())+areas[eligible]-hits)
    return [(float(v),int(r)) for r,v in zip(eligible,iou) if v>=.50]


def grade(ref, prior, masks, candidates, choices, *, diagnose):
    """Scoring only. Same IoU, matching and overlap ownership as mask_metrics.measure."""
    ids = ref['id_map']
    areas = np.bincount(ids.ravel())
    eligible = np.array([r for r,patch in ref['patches'].items() if patch['eligible']],dtype=int)
    if prior.shape[1:]!=ids.shape:
        raise ValueError('Unexpected scoring resolution')
    baseline = [(iou,i,r) for i,m in enumerate(prior) for iou,r in possible(m,ids,areas,eligible)]
    previous = {r for _,r in _one_to_one(baseline)}
    results = {}
    selected = {e['proposal'] for edits in choices.values() for e in edits}
    hits = {i:possible(masks[i],ids,areas,eligible) for i in selected}
    for name, edits in choices.items():
        tests = list(prior)+[masks[e['proposal']] for e in edits]
        pairs = baseline+[(iou,len(prior)+j,r) for j,e in enumerate(edits) for iou,r in hits[e['proposal']]]
        matches = _one_to_one(pairs)
        mapped = np.full(ids.shape,-1,dtype='int32')
        for i in sorted(range(len(tests)),key=lambda i:-int(tests[i].sum())):
            mapped[tests[i]]=i
        correct = sum(int(np.count_nonzero((mapped==i)&(ids==r))) for i,r in matches)
        results[name]=dict(matched=len(matches),proposed=len(tests),eligible=len(eligible),correct_pixels=correct,
                           gained=sorted({r for _,r in matches}-previous),lost=sorted(previous-{r for _,r in matches}))
    diagnostic = {}
    if diagnose:
        raw=set()
        shortlisted=set()
        good={e['proposal'] for e in candidates}
        for i,m in enumerate(masks):
            possible_ids={r for _,r in possible(m,ids,areas,eligible)}-previous
            raw.update(possible_ids)
            if i in good:
                shortlisted.update(possible_ids)
        diagnostic=dict(raw_missing_with_match=sorted(raw), filtered_missing_with_match=sorted(shortlisted),
                        raw_count=len(masks), filtered_count=len(candidates))
    return results, diagnostic


def tally(rows):
    return {key:sum(r[key] for r in rows) for key in ('eligible','matched','proposed','correct_pixels')}


def gzip_result(out, path, value, artifacts):
    target = out/path
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    artifacts[path]=digest(target)


@image_only()
def run(source,fields,out):
    reviews, data = source/'benchmark/reviews', source/'benchmark/data'
    meta = {k:read(reviews/k/'manifest.json') for k in (BASE,SAM)}
    meta['ect'] = read(fields/'manifest.json')
    base = meta[BASE]
    captures = capture_index(data)
    roster = read(checked(reviews/BASE/'cases.json',base['artifacts']['cases.json']))
    archive = checked(data/'visible-geometry-v1.zip',base['reference_sha256'])
    if (len(roster)!=56 or set(captures)!={r['key'] for r in roster} or
            any(m['fixtures']!=base['fixtures'] for m in meta.values()) or
            meta['ect']['roster_sha256']!=base['artifacts']['cases.json']):
        raise ValueError('Incorrect fixtures or roster')
    out.mkdir(parents=True,exist_ok=False)
    dev=[r for r in roster if r['cohort']=='development']
    if len(dev)!=20 or {r['layout'] for r in dev}!={'wing','shopfront'}:
        raise ValueError('Wrong development split')
    grades=[]
    for row in dev:
        case=captures[row['key']]
        image,parts,masks,records,causes=load_inputs(case,row,source,fields,meta)
        events,prior,_=inventory(image,parts,masks,records,causes)
        choices={name:select(events,masks,config) for name,config in PRESETS.items()}
        scores,_=grade(recorded(case,archive)[1],prior,masks,events,choices,diagnose=False)
        grades.append(dict(key=row['key'],scores=scores))
    dev_scores={name:tally([r['scores'][name] for r in grades]) for name in PRESETS}
    def priority(name):
        s=dev_scores[name]
        return (s['matched'],2*s['matched']/(s['proposed']+s['eligible']),s['correct_pixels'],-PRESETS[name]['budget'])
    selected={label:max((k for k in PRESETS if PRESETS[k]['kind'] in kinds),key=priority)
              for label,kinds in [('ect',('geometry','cause-margin')),('rgb',('rgb',))]}
    save(out/'development-freeze.json',dict(selected=selected, settings=PRESETS, scores=dev_scores, frames=grades,
         rule='Most one-to-one development matches, then surface F1, then assigned reference pixels, then smaller proposal budget.'))
    print('Development freeze:',selected,dev_scores,flush=True)
    rows,artifacts=[],{}
    for row in roster:
        case=captures[row['key']]
        image,parts,masks,records,causes=load_inputs(case,row,source,fields,meta)
        start=perf_counter()
        events,prior,rejected=inventory(image,parts,masks,records,causes)
        inv_s=perf_counter()-start
        gray=np.asarray(image.convert('L').resize(masks.shape[1:][::-1],Image.Resampling.BOX)
                        .filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
        choices={}
        record={k:row[k] for k in ('key','sha256','layout','cohort')}
        for label,preset in selected.items():
            start=perf_counter()
            choices[label]=select(events,masks,PRESETS[preset])
            extra=[part_from_mask(image,gray,parts,masks,e) for e in choices[label]]
            seconds=inv_s+perf_counter()-start
            rel=('predictions/' if label=='ect' else 'rgb-predictions/')+row['key']+'.json.gz'
            gzip_result(out,rel,parts+extra,artifacts)
            prefix='' if label=='ect' else 'rgb_'
            record.update({prefix+'predictions':rel,prefix+'prediction_sha256':artifacts[rel],
                           prefix+'cached_processing_s':seconds,prefix+'selected':[e['proposal'] for e in choices[label]]})
        choices['baseline']=[]
        score,diagnostic=grade(recorded(case,archive)[1],prior,masks,events,choices,diagnose=True)
        rel='evidence/'+row['key']+'.json.gz'
        gzip_result(out,rel,dict(candidates=events,rejected=dict(rejected)),artifacts)
        rows.append(dict(**record,scores=score,diagnostics=diagnostic,evidence=rel))
        if len(rows)%10==0:
            print('Graded',len(rows),flush=True)
    save(out/'cases.json',rows)
    summary={cohort:{method:tally([r['scores'][method] for r in rows if cohort=='all' or r['cohort']==cohort])
                     for method in ('baseline','rgb','ect')} for cohort in ['all',*dict.fromkeys(r['cohort'] for r in rows)]}
    for rel in ('cases.json','development-freeze.json'):
        artifacts[rel]=digest(out/rel)
    save(out/'manifest.json',dict(run_id=out.name,code=revision(),version=base['version'], fixtures=base['fixtures'],
         reference_sha256=base['reference_sha256'], frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in rows],
         sources={k:digest(reviews/k/'manifest.json') for k in (BASE,SAM)},fields_manifest_sha256=digest(fields/'manifest.json'),
         model={k:meta['ect'][k] for k in ('upstream','code_revision','timm_revision','weights_url','checkpoint_sha256','device','settings')},
         sources_sha256={name:digest(Path(__file__).with_name(name)) for name in ('ect_afterpass.py','ect_fields.py')},
         settings=dict(common=COMMON,presets=PRESETS,selected=selected,proposal_score='0.70 mean on novel rim + 0.30 mean on full rim',
                       rgb='four-neighbor mean absolute RGB difference after BOX resize, soft saturation 1-exp(-difference/24)',
                       geometry='max(depth, normal)',appearance='max(reflectance, illumination)',
                       cause_margin='geometry - 0.55 * appearance; 3x3 local maximum on each field before subtraction',
                       mode='Retain ordinary Surface Finder and add whole MobileSAM masks. Overlapping proposals are graded.'),
         inference_s=sum(meta['ect']['inference_s'].values()),model_load_s=meta['ect']['load_s'],
         cached_processing_s=sum(r['cached_processing_s'] for r in rows),scores=summary, artifacts=artifacts,
         reproduce='python -m benchmark.ect_afterpass --source-root REPO --fields ECT_FIELDS --out FRESH_OUTPUT'))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--fields',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.source_root.resolve(),args.fields.resolve(),args.out.resolve())
