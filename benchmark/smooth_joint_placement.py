"""Compare common wall votes to the shipped single-owner outline and placement on 204 views."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from benchmark import smooth_local_release as local
from benchmark import facade_inset_round as suite, pavement_opening_eval as replay
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import read, save, digest, checked
from benchmark.thin_face_placement_run import ref_inputs
from benchmark.wall_texture_evaluate import metrics
from engine import atlas_count,eave_pinholes,inset_returns,jagged_texture_join,pavement_opening
from engine import smooth_atlas_outline as outline,smooth_extent_placement as placer
from engine.pillar_seams import native_mask

OUT=suite.ROOT/'benchmark/reviews/smooth-joint-placement-01'
RUN=suite.ROOT/'runs/smooth-joint-placement-01'


@lru_cache(maxsize=1)
def atlas():
    return atlas_count.load_atlas(suite.ATLAS,'26.3')


@lru_cache(maxsize=1)
def old_inputs():
    summary=read(local.OUT/'summary.json')
    rows={r['id']:r for r in read(checked(local.OUT/'cases.json',summary['cases_sha256']))}
    packs=ZipFile(checked(local.OUT/'default-exports.zip',summary['archive_sha256']))
    return rows,packs,read(local.PREVIOUS/'summary.json')


def old_pack(job):
    key=job[1]['id']
    rows,packs,prev=old_inputs()
    row=rows[key]
    if row['added']:
        data=packs.read(key+'/reconstruction.zip')
        assert sha256(data).hexdigest()==row['pack_sha256']
        return row['pack_sha256'],set(decoded_pack(BytesIO(data)))
    h,c=local.previous(job,prev)
    assert h==row['previous_pack_sha256']
    return h,c


def grade(job):
    split,frame,_,_=job
    key=frame['id']
    rows,_,_=old_inputs()
    old=rows[key]
    h,cells=old_pack(job)
    assert old['source_hash']==frame['files']['rgb.png']
    row=dict(id=key,split=split,source_hash=old['source_hash'],previous_pack_sha256=h,
             baseline=old['result'],result=old['result'],added=[],outline=[],old_accepted=len(old['added']))
    image,base,counts,camera=suite.ordinary_inputs(*job)
    parts,rims,phase,insets,_=inset_returns.detect(image,base,'26.3',camera=camera)
    parts,eaves=eave_pinholes.refine(image,parts)
    if not (rims or phase or insets):
        parts,_,_=jagged_texture_join.refine(image,parts,'26.3',camera=camera)
    candidates=placer.proposals(image,parts,common_wall=True)
    row['raw_components']=len(candidates)
    if not candidates:
        return row
    before=outline.checked_regions(image,parts,placer.proposals(image,parts),atlas())
    extra=outline.extra_interior(image,outline.checked_regions(image,parts,candidates,atlas()),before)
    if not extra:
        return row
    # Fix the candidates before loading scoring labels.
    truth=suite.visible_map(split,frame)
    old_parts=outline.apply_regions(image,parts,before)
    corrected=outline.apply_regions(image,parts,before+extra)
    after=metrics(corrected,truth,image.size)
    base_metrics=metrics(old_parts,truth,image.size)
    predicted={p['id']:p for p in base_metrics['parts']}
    row['surface_changes']=dict(gained=sorted(set(after['matched'])-set(base_metrics['matched'])),
                                lost=sorted(set(base_metrics['matched'])-set(after['matched'])),
                                unpaired=after['unmatched_predictions']-base_metrics['unmatched_predictions'],
                                paired_pixels=after['paired_pixels']-base_metrics['paired_pixels'])
    for region in extra:
        vals=np.concatenate([truth[y,l:r] for y,l,r in region['runs']])
        target=predicted[region['surface']]['dominant']
        hist=Counter(vals.tolist())
        row['outline'].append(dict(surface=region['surface'],box=region['box'],target=int(target),
                                   pixels=len(vals),correct=hist[target],background=hist[0],
                                   neighbor=len(vals)-hist[0]-hist[target],reference_ids=dict(hist)))
    measured=eave_pinholes.measurement_parts(parts) if eaves else parts
    if rims or phase or insets or not cells or camera.get('P') is None or pavement_opening.detect(image,measured,camera):
        return row
    additions=placer.recover_from_regions(image,'26.3',measured,cells,camera,suite.ATLAS,extra)
    assert cells.isdisjoint(additions)
    row['added']=sorted(additions)
    if not additions:
        return row
    path=RUN/key/'reconstruction.zip'
    path.parent.mkdir(parents=True,exist_ok=True)
    packed=_write_read(path,cells|additions)
    truth,occupied,T,_=ref_inputs(job)
    labels,ids=replay.visible_faces(split,frame)
    optional=horizontal_slivers(labels,ids)
    now,prior=registered(packed,T),registered(cells,T)
    scores=placement_scores(now,prior,dict.fromkeys(truth),occupied)
    shifted=suite.framed.spacing.shift(now,scores['translation'])
    prev=suite.framed.spacing.shift(prior,scores['baseline_translation'])
    assert replay.priority(prev,truth,optional)==row['baseline']['facade']
    row['block_deltas']={k:scores['relative_geometry'][k] for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')}
    row.update(pack_sha256=digest(path),result=dict(strict={k:scores['relative_geometry'][k] for k in replay.FIELDS},
               facade=replay.priority(shifted,truth,optional),registered={k:scores['metrics'][k] for k in replay.FIELDS},
               translation=scores['translation']))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--public',action='store_true')
    args=p.parse_args()
    jobs=list(suite.jobs([*suite.framed.SIZES,'shelter']))
    if args.public:
        from benchmark.eave_patch_release import dimensions
        checks=[]
        rows=read(OUT/'cases.json')
        byid={j[1]['id']:j for j in jobs}
        for row in rows:
            key=row['id']
            if not row['outline'] and not row['old_accepted']:
                continue
            path=RUN/'public'/key
            m=read(path/'build-manifest.json')
            ordinary=decoded_pack(path/'reconstruction.zip')
            assert m['through']=='placement' and m['image_sha256']==row['source_hash']
            _,prior=old_pack(byid[key])
            expected=prior|set(map(tuple,row['added']))
            assert ordinary==dict.fromkeys(expected,'minecraft:smooth_stone'),key
            stats=m['surface_finder']['boundary_settings']['smooth_atlas_outline']
            current=[(x['surface'],x['box'],x['pixels']) for x in stats['events'] if x.get('common_wall')]
            assert current==[(x['surface'],x['box'],x['pixels']) for x in row['outline']],key
            counts=read(path/'counts.json')
            original=suite.ordinary_inputs(*byid[key])[2]
            assert dimensions(counts)==dimensions(original),key
            checks.append(dict(id=key,source_hash=row['source_hash'],added=len(row['added']),verified_cells=len(expected),
                               public_pack_sha256=digest(path/'reconstruction.zip'),dimensions_unchanged=True))
        save(OUT/'ordinary.json',checks)
        return
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(grade,jobs))
    assert len(rows)==204
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'cases.json',rows)
    baseline=local.summarize(rows,'baseline')
    assert baseline==read(local.OUT/'summary.json')['cohorts']
    result=local.summarize(rows,'result')
    changed=[r for r in rows if r['added']]
    archive=OUT/'default-exports.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
        for r in changed:
            z.write(RUN/r['id']/'reconstruction.zip',r['id']+'/reconstruction.zip')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for r in changed:
            assert sha256(z.read(r['id']+'/reconstruction.zip')).hexdigest()==r['pack_sha256']
    fixed=result['fixed']['strict']
    hist=2*fixed['correct']/(2*fixed['correct']+fixed['missed']+fixed['extras']+384)
    save(OUT/'summary.json',dict(baseline='f3df5f7',frames=len(rows),changed=changed,cohorts=result,
                                 previous_cohorts=baseline,archive_sha256=digest(archive),
                                 cases_sha256=digest(OUT/'cases.json'),historical_112_strict_f1=hist))
    print('Changed',[(r['id'],r['block_deltas']) for r in changed])
    print('New outline',[(r['id'],r['outline']) for r in rows if r['outline']])


if __name__=='__main__':
    main()
