"""Screen the source-only ground detector and re-export placements on matched views."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark import facade_inset_round as suite, smooth_joint_placement as previous
from benchmark import pavement_opening_eval as replay
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.smooth_local_release import summarize
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine import eave_pinholes, inset_returns, jagged_texture_join, pavement_opening, visible_placement
from engine.pavement_obstructions import window_obstructions


RUN = suite.ROOT/'runs/opening-candidate-vote'


@lru_cache(maxsize=1)
def reference():
    summary = read(previous.OUT/'summary.json')
    return {r['id']: r for r in read(checked(previous.OUT/'cases.json', summary['cases_sha256']))}


def score(job):
    split, frame, _, _ = job
    key = frame['id']
    baseline = reference()[key]
    pack_hash = baseline.get('pack_sha256', baseline['previous_pack_sha256'])
    assert baseline['source_hash'] == frame['files']['rgb.png']
    image, base, counts, camera = suite.ordinary_inputs(*job)
    refined, returns, phase, outlines, _ = inset_returns.detect(image,base,'26.3',camera=camera)
    refined, eaves = eave_pinholes.refine(image,refined)
    screen = base
    joins = []
    if not (returns or phase or outlines):
        screen, joins, _ = jagged_texture_join.refine(image,refined,'26.3',camera=camera)
    matches = pavement_opening.detect(image,screen,camera)
    row = dict(id=key, split=split, source_hash=baseline['source_hash'], previous_pack_sha256=pack_hash,
               matched=matches, baseline=baseline['result'], result=baseline['result'])
    if not matches:
        return row
    # Current matched views have unchanged placement/counting inputs. Fail closed if replaying
    # a future fixture that requires re-running one of these upstream refinements.
    assert not (returns or phase or outlines or joins or eaves), key
    regions = [p for p in screen if p['id'] in matches]
    def predict(early):
        cells, status = visible_placement.place(image,'26.3',base,counts=counts,camera=camera,
            atlas_path=suite.ATLAS, opening_parts=screen if early else None)
        if early:
            assert status['pavement_surface_ids'] == matches
        return cells - window_obstructions(image.size,status,regions,cells)
    if 'pack_sha256' in baseline:
        with ZipFile(previous.OUT/'default-exports.zip') as archive:
            before = set(decoded_pack(BytesIO(archive.read(key+'/reconstruction.zip'))))
    else:
        h, before = previous.old_pack(job)
        assert h == pack_hash
    assert predict(False) == before, key
    after = predict(True)
    added = after-before
    removed = before-after
    row.update(added=sorted(added), removed=sorted(removed))
    if not (added or removed):
        return row
    path = RUN/key/'reconstruction.zip'
    path.parent.mkdir(parents=True,exist_ok=True)
    assert set(_write_read(path,after)) == after
    truth, occupied, transform, _ = ref_inputs(job)
    now, prior = registered(after,transform),registered(before,transform)
    scores = placement_scores(now,prior,dict.fromkeys(truth),occupied)
    optional = horizontal_slivers(*replay.visible_faces(split,frame))
    shifted = suite.framed.spacing.shift(now,scores['translation'])
    old = suite.framed.spacing.shift(prior,scores['baseline_translation'])
    assert replay.priority(old,truth,optional) == row['baseline']['facade']
    row.update(pack_sha256=digest(path),result=dict(strict={k:scores['relative_geometry'][k] for k in replay.FIELDS},
        facade=replay.priority(shifted,truth,optional),registered={k:scores['metrics'][k] for k in replay.FIELDS},
        translation=scores['translation']))
    row['block_deltas'] = {k:scores['relative_geometry'][k] for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')}
    return row


def main():
    p=ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    jobs=list(suite.jobs([*suite.framed.SIZES,'shelter']))
    assert len(jobs)==204
    if args.smoke:
        jobs=[j for j in jobs if 'test-colonnade-01-oblique' in j[1]['id']]
    RUN.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(score,jobs))
    for r in rows:
        if r['matched']:
            print(r['id'],r.get('block_deltas'),r['result'], flush=True)
    save(RUN/('smoke.json' if args.smoke else 'cases.json'),rows)
    if not args.smoke:
        before=summarize(rows,'baseline')
        assert before==read(previous.OUT/'summary.json')['cohorts']
        changed = [r for r in rows if 'pack_sha256' in r]
        archive = RUN/'default-exports.zip'
        with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
            for row in changed:
                name = row['id']+'/reconstruction.zip'
                z.write(RUN/name,name)
        with ZipFile(archive) as z:
            assert z.testzip() is None
        cohorts = summarize(rows,'result')
        fixed = cohorts['fixed']['strict']
        hist = 2*fixed['correct']/(2*fixed['correct']+fixed['missed']+fixed['extras']+384)
        save(RUN/'summary.json',dict(baseline='e043ac2',frames=len(rows),previous_cohorts=before,cohorts=cohorts,
             cases_sha256=digest(RUN/'cases.json'),archive_sha256=digest(archive),changed=changed,historical_112_strict_f1=hist))


if __name__=='__main__':
    main()
