"""Compare local and whole-build effects of merging masks on the frozen images."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.coplanar_fragment_audit import CONTROL, REVIEW, load_case
from benchmark.coplanar_fragment_join import join_regions
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import read,save,checked,digest
from benchmark.texture_camera_sheet import font
from benchmark.wall_texture_evaluate import metrics
from engine.pillar_seams import native_mask


def index(folder):
    meta=read(folder/'summary.json')
    return {r['id']:r for r in read(checked(folder/'cases.json',meta['cases_sha256']))}


def detail(job):
    split,fr,h,control,interventions,folder=job
    key=fr['id']
    with_groups={name:r for name,r in interventions.items() if r['status']['groups']}
    if not with_groups:
        return dict(split=split,id=key,methods={})
    photo,parts,labels,ids,missed=load_case((split,fr,h,control))
    by_id={p['id']:p for p in parts}
    lut=np.zeros(len(labels['faces'])+1,int)
    blocks={}
    for f in labels['faces']:
        lut[f['id']]=f['surface_id']
        blocks[f['id']]=tuple(f['xyz_world'])
    truth=lut[ids]
    base=metrics(parts,truth,photo.size)
    area={v['id']:v for v in base['parts']}
    old=read(CONTROL/split/key/'cells.json')['world']
    prior={tuple(p+d for p,d in zip(xyz,control['translation'])) for xyz in old}
    methods={}
    for name,r in with_groups.items():
        groups=r['status']['groups']
        joined=join_regions(photo,parts,groups)
        m=metrics(joined,truth,photo.size)
        voted_blocks=set()
        labels_by_group=[]
        for group in groups:
            votes=[]
            for k in group:
                a=area[k]
                votes.append(a['dominant'] if a['correct_if_dominant']>=.8*a['pixels'] else 0)
            labels_by_group.append(votes)
            if 0 not in votes and len(set(votes))==1:
                voted_blocks.update(blocks[i] for i in blocks if lut[i]==votes[0])
        next_world=read(folder/name/split/key/'cells.json')['world']
        placed={tuple(p+d for p,d in zip(xyz,r['translation'])) for xyz in next_world}
        true=set(blocks.values())
        restored=missed&placed
        lost=(prior&true)-placed
        assert len(restored)==r['relative_geometry']['gained'] and len(lost)==r['relative_geometry']['lost']
        methods[name]=dict(groups=groups,surface_votes=labels_by_group,matched_gained=sorted(set(m['matched'])-set(base['matched'])),
            matched_lost=sorted(set(base['matched'])-set(m['matched'])),unmatched_delta=m['unmatched_predictions']-base['unmatched_predictions'],
            paired_pixel_delta=m['paired_pixels']-base['paired_pixels'],on_merged_surface_gained=len(restored&voted_blocks),
            on_merged_surface_lost=len(lost&voted_blocks),other_gained=len(restored-voted_blocks),other_lost=len(lost-voted_blocks))
    return dict(split=split,id=key,methods=methods)


def sheet(args,audit,control):
    choices={'heldout-facade-oblique-right':'Facade diagonal',
             'legacy-light_gray_concrete-holdout-portico-masonry-high-right':'Portico masonry',
             'legacy-light_gray_concrete-dev-shopfront-plaster-low-left':'Shopfront plaster'}
    page=Image.new('RGB',(1350,730),'#172c33')
    pen=ImageDraw.Draw(page);pen.text((16,10),'Adjoining predicted masks on one recorded surface',fill='white',font=font(27))
    x=0
    for split,fr,h in framed.cases(framed.EVALUATION):
        key=fr['id']
        if key not in choices:
            continue
        im,parts,_,_,_,labels,ids,*_=read_view(
            CONTROL,split,fr,h,control[key])
        info=max(audit[key]['thresholds']['standard']['groups'],key=lambda g:g['on_face'])
        surf=info['surface']
        masks=[native_mask(next(p for p in parts if p['id']==k),im.size) for k in info['mask_ids']]
        lut=np.zeros(len(labels['faces'])+1,int)
        for f in labels['faces']:lut[f['id']]=f['surface_id']
        reference=lut[ids]==surf
        y,xs=np.nonzero(reference|np.logical_or.reduce(masks))
        crop=max(0,int(xs.min())-8),max(0,int(y.min())-8),min(im.width,int(xs.max())+9),min(im.height,int(y.max())+9)
        pen.text((x+12,54),choices[key],fill='white',font=font(21))
        pen.text((x+12,80),f"{len(masks)} fragments; {info['on_face']} missed blocks on this surface",fill='white',font=font(16))
        thumb=ImageOps.contain(im,(425,207));page.paste(thumb,(x+12,110))
        def overlay(groups,colors):
            rgb=np.asarray(im).copy()
            for mask,c in zip(groups,colors):
                rgb[mask]=.52*rgb[mask]+.48*np.array(c)
            return ImageOps.contain(Image.fromarray(rgb).crop(crop),(425,170))
        pen.text((x+12,327),'Predicted fragments',fill='white',font=font(16))
        page.paste(overlay(masks,[(245,146,52),(99,188,249),(217,104,224),(255,220,105)]),(x+12,353))
        pen.text((x+12,529),'Recorded continuous plane',fill='white',font=font(16))
        page.paste(overlay([reference],[(70,224,174)]),(x+12,552))
        x+=450
    assert x==1350
    page.save(args.out/'split-surfaces.jpg',quality=93)


def shelter_gate(folder, methods):
    from training.visible_labels import visible_labels
    inputs=Path('../framed-depth-transfer/runs/transfer-input')
    frames=read(inputs/'screen.json')['selected']
    records={name:index(folder/('transfer-'+name)) for name in methods}
    results=[]
    for frame in frames:
        key=frame['id']
        files=frame['files']
        src=inputs/frame['directory']/'captures'/frame['source_id']
        pose=read(checked(src.with_suffix('.json'),files['camera.json']))
        blocks=read(checked(src.with_name(src.name+'-blocks.json'),files['world-readback.json']))
        labels,ids,_=visible_labels(pose,{tuple(b[k] for k in 'xyz'):b for b in blocks},(0,0,0))
        lut=np.zeros(len(labels['faces'])+1,int)
        for face in labels['faces']:lut[face['id']]=face['surface_id']
        truth=lut[ids]
        scene={}
        with Image.open(checked(src.with_suffix('.png'),files['rgb.png'])) as original:
            photo=original.convert('RGB')
        previous=records[methods[0]][key]
        parts=read(checked(inputs/'upstream'/key/'surfaces.json',previous['surface_sha256']))['planes']
        base=metrics(parts,truth,photo.size)
        area={p['id']:p for p in base['parts']}
        for name in methods:
            groups=records[name][key]['status']['groups']
            if not groups:continue
            joined=metrics(join_regions(photo,parts,groups),truth,photo.size)
            votes=[[area[k]['dominant'] if area[k]['correct_if_dominant']>=.8*area[k]['pixels'] else 0
                    for k in group] for group in groups]
            scene[name]=dict(groups=groups,surface_votes=votes,
                matched_gained=sorted(set(joined['matched'])-set(base['matched'])),
                matched_lost=sorted(set(base['matched'])-set(joined['matched'])),
                unmatched_delta=joined['unmatched_predictions']-base['unmatched_predictions'],
                paired_pixel_delta=joined['paired_pixels']-base['paired_pixels'])
        results.append(dict(split='shelter_gate',id=key,methods=scene))
    return results,records


def run(args):
    control=index(CONTROL)
    names=['photo','ideal','transition','diagonal']
    variants={name:index(args.runs/name) for name in names}
    jobs=[]
    for s,f,h in framed.cases(framed.SIZES):
        key=f['id']
        prior=control[key]
        current={m:idx[key] for m,idx in variants.items()}
        for r in current.values():
            assert r['control_pack_sha256']==prior['pack_sha256']
            assert all(r[k]==prior[k] for k in ('rgb_sha256','input_hashes','reference_hashes'))
        jobs.append((s,f,h,prior,current,args.runs))
    with ProcessPoolExecutor(args.workers) as pool:
        detail_rows=list(pool.map(detail,jobs))
    independent=[name for name in names if name!='ideal']
    extra,extra_records=shelter_gate(args.runs,independent)
    detail_rows.extend(extra)
    audit={r['id']:r for r in read(checked(args.out/'cases.json',read(args.out/'summary.json')['cases_sha256']))}
    totals={}
    group_sets={'fixed':framed.EVALUATION,'development':['train'],'alternate':['extra'],'new':['new'],'shelter_gate':['shelter_gate']}
    for cohort,splits in group_sets.items():
        totals[cohort]={}
        for name in names:
            if cohort=='shelter_gate' and name=='ideal':continue
            c=Counter()
            for line in detail_rows:
                if line['split'] not in splits or name not in line['methods']:
                    continue
                key=line['id'];r=line['methods'][name]
                trial=(extra_records if cohort=='shelter_gate' else variants)[name][key]
                score=trial['relative_geometry']
                if cohort!='shelter_gate' and trial['f1']>trial['previous_f1']:
                    c.update(per_view_better=1,best_view_net_correct=score['gained']-score['lost'],
                             best_view_net_extras=score['extra_added']-score['extra_removed'])
                for k,v in r.items():
                    if isinstance(v,int):c[k]+=v
                c.update(views_with_merge=1,gained_matches=len(r['matched_gained']),lost_matches=len(r['matched_lost']))
                for votes in r['surface_votes']:
                    c.update(groups=1)
                    if 0 in votes:c.update(mixed_or_unresolved=1)
                    elif len(set(votes))==1:c.update(same_reference_plane=1)
                    else:c.update(distinct_reference_planes=1)
            totals[cohort][name]=c
    save(args.out/'merge-detail.json',dict(totals=totals,frames=detail_rows))
    with ZipFile(args.out/'exports.zip','w',ZIP_DEFLATED) as z:
        for name in names:
            folder=args.runs/name
            for f in ['summary.json','cases.json']:z.write(folder/f,name+'/'+f)
            for r in variants[name].values():
                if r['changed']:
                    sub=folder/r['split']/r['id']
                    for f in [checked(sub/'reconstruction.zip',r['pack_sha256']),sub/'cells.json']:
                        z.write(f,name+'/'+r['split']+'/'+r['id']+'/'+f.name)
        for name in independent:
            run='transfer-'+name
            folder=args.runs/run
            for f in ['summary.json','cases.json']:z.write(folder/f,run+'/'+f)
            for key,r in extra_records[name].items():
                checked(folder/key/'reconstruction.zip',r['pack_sha256'])
                for f in ['reconstruction.zip','cells.json']:z.write(folder/key/f,run+'/'+key+'/'+f)
    with ZipFile(args.out/'exports.zip') as z:assert z.testzip() is None
    seconds=[r['status']['surface_seconds'] for dataset in (variants,extra_records) for r in dataset['diagonal'].values()]
    save(args.out/'placement.json',dict(export_archive_sha256=digest(args.out/'exports.zip'),methods={
        name:read(args.runs/name/'summary.json')['groups'] for name in names},shelter_gate={
        name:read(args.runs/('transfer-'+name)/'summary.json')['relative_geometry'] for name in independent},
        diagonal_stage_seconds=dict(samples=len(seconds),median=np.median(seconds),p95=np.percentile(seconds,95),maximum=max(seconds)),
        merge_audit=digest(args.out/'merge-detail.json')))
    sheet(args,audit,control)
    print(totals)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--runs',type=Path,default=Path('runs'))
    p.add_argument('--out',type=Path,default=REVIEW)
    run(p.parse_args())
