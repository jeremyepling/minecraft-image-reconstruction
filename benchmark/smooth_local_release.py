"""Replay smooth-patch placement on all frozen source images and verify ordinary exports."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark import facade_inset_round as suite, pavement_opening_eval as replay
from benchmark.edge_vote_outline_eval import baseline_pack
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import COHORTS, ref_inputs
from engine import eave_pinholes, inset_returns, jagged_texture_join, pavement_opening, smooth_extent_placement
from engine.surface_perimeter import with_outline

REVIEW=suite.ROOT/'benchmark/reviews'
OUT=REVIEW/'smooth-local-placement-01'
PREVIOUS=REVIEW/'pavement-local-01'
RUN=suite.ROOT/'runs/smooth-local-release'
CONTROL=REVIEW/'pavement-opening-round-01/cases.json'
CLI=('legacy-light_gray_concrete-holdout-registry-plaster-high-right',
     'legacy-light_gray_concrete-dev-shopfront-plaster-low-left',
     'legacy-light_gray_concrete-holdout-portico-plaster-high-right',
     'legacy-smooth_stone-dev-wing-plaster-low-left')


def previous(job,release):
    key=job[1]['id']
    published=next((r for r in release['changed'] if r['id']==key),None)
    if published:
        with ZipFile(checked(PREVIOUS/'default-exports.zip',release['archive_sha256'])) as z:
            data=z.read(key+'/reconstruction.zip')
        assert sha256(data).hexdigest()==published['pack_sha256']
        return published['pack_sha256'],set(decoded_pack(BytesIO(data)))
    return baseline_pack(job)


def grade(task):
    job,base,previous_release=task
    split,frame,_,_=job
    key=frame['id']
    image,parts,counts,camera=suite.ordinary_inputs(*job)
    old_hash,original=previous(job,previous_release)
    source=frame['files']['rgb.png']
    assert source==base['rgb_sha256']
    row=dict(id=key,split=split,source_hash=source,previous_pack_sha256=old_hash,
             baseline={k:base[k] for k in ('strict','facade','registered','translation')})
    row['result']=row['baseline'].copy()
    revised,returns,phase,outlines,_=inset_returns.detect(image,parts,'26.3',camera=camera)
    if returns or phase or outlines or not original:
        return dict(row,added=[],skip='inset or empty baseline')
    revised,eaves=eave_pinholes.refine(image,revised)
    revised,_,_=jagged_texture_join.refine(image,revised,'26.3',camera=camera)
    if eaves:
        revised=eave_pinholes.measurement_parts(revised)
    if pavement_opening.detect(image,revised,camera):
        return dict(row,added=[],skip='pavement opening')
    revised=[with_outline(p) for p in revised]
    started=perf_counter()
    added,regions=smooth_extent_placement.recover(image,'26.3',revised,original,camera,suite.ATLAS)
    row.update(elapsed_seconds=round(perf_counter()-started,4),added=sorted(added),
               proposed_regions=len(regions),proposed_pixels=sum(r['pixels'] for r in regions))
    if not regions:
        return row
    # The upstream stage and existing pack must still agree on every affected image.
    before,_,_=replay.build(image,revised,counts,camera)
    assert before==original,key
    if not added:
        return row
    path=RUN/key/'reconstruction.zip'
    path.parent.mkdir(parents=True,exist_ok=True)
    points=_write_read(path,original|added)
    assert added.isdisjoint(original)
    truth,occupied,transform,_=ref_inputs(job)
    labels,ids=replay.visible_faces(split,frame)
    optional=horizontal_slivers(labels,ids)
    raw=registered(points,transform)
    prior=registered(original,transform)
    scores=placement_scores(raw,prior,dict.fromkeys(truth),occupied)
    placed=suite.framed.spacing.shift(raw,scores['translation'])
    old=suite.framed.spacing.shift(prior,scores['baseline_translation'])
    assert replay.priority(old,truth,optional)==row['baseline']['facade']
    assert list(scores['baseline_translation'])==row['baseline']['translation']
    row.update(pack_sha256=digest(path),result=dict(strict={k:scores['relative_geometry'][k] for k in replay.FIELDS},
               registered={k:scores['metrics'][k] for k in replay.FIELDS},
               facade=replay.priority(placed,truth,optional),translation=scores['translation']))
    return row


def summarize(rows, member):
    result={}
    for name,splits in COHORTS.items():
        result[name]={}
        selected=[r for r in rows if r['split'] in splits]
        for metric in ('strict','facade','registered'):
            total=Counter()
            for r in selected:
                total.update(r[member][metric])
            result[name][metric]=dict(total,f1=2*total['correct']/(2*total['correct']+total['missed']+total['extras']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cli',action='store_true')
    args=p.parse_args()
    if args.cli:
        published=read(OUT/'summary.json')
        changed={r['id']:r for r in published['changed']}
        checks=[]
        for key in CLI:
            path=suite.ROOT/'runs/smooth-cli'/key
            build=read(path/'build-manifest.json')
            row=changed[key]
            assert build['image_sha256']==row['source_hash']
            ordinary=decoded_pack(path/'reconstruction.zip')
            expected=decoded_pack(RUN/key/'reconstruction.zip')
            assert ordinary==expected,(key,len(ordinary.keys()-expected.keys()),len(expected.keys()-ordinary.keys()))
            check=dict(id=key,source_hash=row['source_hash'],blocks=len(ordinary),
                       ordinary_pack_sha256=digest(path/'reconstruction.zip'),candidate_pack_sha256=row['pack_sha256'],
                       detail=build['placement']['smooth_extent_placement'])
            checks.append(check)
        save(OUT/'ordinary.json',checks)
        return
    previous_release=read(PREVIOUS/'summary.json')
    controls={r['id']:r['baseline'] for r in read(CONTROL)}
    for r in previous_release['changed']:
        controls[r['id']].update(r['result'])
    with ProcessPoolExecutor(args.workers) as pool:
        results=list(pool.map(grade,((j,controls[j[1]['id']],previous_release) for j in suite.jobs([*suite.framed.SIZES,'shelter']))))
    assert len(results)==204
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'cases.json',results)
    before=summarize(results,'baseline')
    after=summarize(results,'result')
    assert before==previous_release['cohorts']
    changed=[r for r in results if r['added']]
    archive=OUT/'default-exports.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
        for r in changed:
            z.write(RUN/r['id']/'reconstruction.zip',r['id']+'/reconstruction.zip')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for r in changed:
            assert sha256(z.read(r['id']+'/reconstruction.zip')).hexdigest()==r['pack_sha256']
    fixed=after['fixed']['strict']
    historical=2*fixed['correct']/(2*fixed['correct']+fixed['missed']+fixed['extras']+384)
    save(OUT/'summary.json',dict(baseline='a5a88ad',frames=len(results),changed=changed,cohorts=after,
         previous_cohorts=before,archive_sha256=digest(archive),cases_sha256=digest(OUT/'cases.json'),
         historical_112_strict_f1=historical))
    print('changed',len(changed),'fixed',after['fixed'])


if __name__=='__main__':
    main()
