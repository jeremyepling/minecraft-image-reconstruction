"""Score rechecking visible wall candidates after pavement cleanup on all 204 fixed inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark import facade_inset_round as suite, pavement_opening_eval as replay
from benchmark import smooth_joint_placement as previous, smooth_local_release as release
from benchmark.mask_blocks import _write_read, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine import eave_pinholes, inset_returns, jagged_texture_join, pavement_obstructions, pavement_opening
from engine import visible_placement, window_wall_recheck


OUT = suite.ROOT/'benchmark/reviews/window-wall-01'
RUN = suite.ROOT/'runs/window-recheck'


@lru_cache(maxsize=1)
def baseline():
    summary = read(previous.OUT/'summary.json')
    records = read(checked(previous.OUT/'cases.json', summary['cases_sha256']))
    return {r['id']:r for r in records}


def grade(job):
    split, frame, _, _ = job
    key = frame['id']
    old = baseline()[key]
    assert frame['files']['rgb.png'] == old['source_hash']
    pack_hash = old.get('pack_sha256', old['previous_pack_sha256'])
    row = dict(id=key, split=split, source_hash=old['source_hash'], previous_pack_sha256=pack_hash,
               baseline=old['result'], result=old['result'], added=[])
    image, base, counts, camera = suite.ordinary_inputs(*job)
    parts, rim, phase, outlines, _ = inset_returns.detect(image, base, '26.3', camera=camera)
    detector_parts = base
    if not (rim or phase or outlines):
        parts, eaves = eave_pinholes.refine(image, parts)
        detector_parts, joins, _ = jagged_texture_join.refine(image, parts, '26.3', camera=camera)
    matched = set(pavement_opening.detect(image, detector_parts, camera))
    row['openings'] = sorted(matched)
    if not matched:
        return row
    # No new counting or mask stages intervene on the positively matched images.
    assert not (rim or phase or outlines or eaves or joins)
    cells, status = visible_placement.place(image, '26.3', base, counts=counts, camera=camera, atlas_path=suite.ATLAS)
    regions = [part for part in detector_parts if part['id'] in matched]
    cells -= pavement_obstructions.window_obstructions(image.size, status, regions, cells)
    # Cross-check the complete ordinary starting export including its materials.
    previous_hash, archived = previous.old_pack(job)
    assert previous_hash == pack_hash and archived == cells, key
    added = window_wall_recheck.recover(image, base, status, matched, cells)
    assert not (added & cells)
    row['added'] = sorted(added)
    if not added:
        return row
    path = RUN/key/'reconstruction.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = _write_read(path, cells | added)
    truth, occupied, transform, _ = ref_inputs(job)
    labels, ids = replay.visible_faces(split, frame)
    optional = horizontal_slivers(labels, ids)
    now, prior = registered(packed, transform), registered(cells, transform)
    scores = placement_scores(now, prior, dict.fromkeys(truth), occupied)
    shifted = suite.framed.spacing.shift(now, scores['translation'])
    original = suite.framed.spacing.shift(prior, scores['baseline_translation'])
    assert replay.priority(original, truth, optional) == row['baseline']['facade']
    relative = scores['relative_geometry']
    row['block_deltas'] = {k:relative[k] for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')}
    row.update(pack_sha256=digest(path), result=dict(strict={k:relative[k] for k in replay.FIELDS},
               facade=replay.priority(shifted, truth, optional), registered={k:scores['metrics'][k] for k in replay.FIELDS},
               translation=scores['translation']))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    jobs = list(suite.jobs([*suite.framed.SIZES, 'shelter']))
    assert len(jobs) == 204
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(grade, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/'cases.json', rows)
    before, after = (release.summarize(rows, member) for member in ('baseline','result'))
    assert before == read(previous.OUT/'summary.json')['cohorts']
    changed = [r for r in rows if r['added']]
    archive = OUT/'default-exports.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for row in changed:
            key = row['id']
            z.write(RUN/key/'reconstruction.zip', key+'/reconstruction.zip')
    with ZipFile(archive) as z:
        assert z.testzip() is None
    fixed = after['fixed']['strict']
    hist = 2*fixed['correct']/(2*fixed['correct']+fixed['missed']+fixed['extras']+384)
    save(OUT/'summary.json', dict(baseline='e043ac2',frames=len(rows),changed=changed,cohorts=after,
        previous_cohorts=before,archive_sha256=digest(archive),cases_sha256=digest(OUT/'cases.json'),
        historical_112_strict_f1=hist))
    print('Openings', [(r['id'], r['added']) for r in rows if r['openings']])
    print('Fixed', after['fixed'])


if __name__ == '__main__':
    main()
