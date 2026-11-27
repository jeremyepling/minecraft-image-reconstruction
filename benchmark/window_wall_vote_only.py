"""Compare early wall-only visibility votes on the frozen 204-image geometry suite."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

from benchmark import facade_inset_round as suite, smooth_joint_placement as prior, smooth_local_release as pool_scores
from benchmark.mask_blocks import _write_read, registered
from benchmark.pavement_opening_eval import FIELDS, priority, visible_faces
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine import pavement_obstructions, pavement_opening, visible_placement


OUT = suite.ROOT / 'benchmark/reviews/window-wall-vote-only-01'
RUN = suite.ROOT / 'runs/window-wall-vote-only'


@lru_cache(maxsize=1)
def control():
    release = read(prior.OUT / 'summary.json')
    return {r['id']: r for r in read(checked(prior.OUT / 'cases.json', release['cases_sha256']))}


def grade(job):
    split, frame, _, _ = job
    key = frame['id']
    source = frame['files']['rgb.png']
    old = control()[key]
    assert source == old['source_hash']
    common = dict(id=key, split=split, source_hash=source, baseline=old['result'], result=old['result'])
    image, parts, counts, camera = suite.ordinary_inputs(*job)
    regions = set(pavement_opening.detect(image, parts, camera))
    if not regions:
        return dict(common, matched=[])
    cells, status = visible_placement.place(image, '26.3', parts, counts=counts, camera=camera, atlas_path=suite.ATLAS)
    opening = [p for p in parts if p['id'] in regions]
    cells -= pavement_obstructions.window_obstructions(image.size, status, opening, cells)
    # Both matching views predate the later smooth-wall updates.
    prior_hash, original = prior.old_pack(job)
    assert prior_hash == old.get('pack_sha256', old['previous_pack_sha256'])
    path = RUN / key / 'reconstruction.zip'
    path.parent.mkdir(exist_ok=True, parents=True)
    decoded = _write_read(path, cells)
    truth, occupied, transform, _ = ref_inputs(job)
    now, before = registered(decoded, transform), registered(original, transform)
    # After the exported coordinates are fixed, open scoring labels.
    scores = placement_scores(now, before, dict.fromkeys(truth), occupied)
    labels, ids = visible_faces(split, frame)
    optional = horizontal_slivers(labels, ids)
    assert priority(suite.framed.spacing.shift(before, scores['baseline_translation']), truth, optional) == old['result']['facade']
    shifted = suite.framed.spacing.shift(now, scores['translation'])
    strict = scores['relative_geometry']
    common['result'] = dict(strict={k:strict[k] for k in FIELDS}, registered={k:scores['metrics'][k] for k in FIELDS},
                            facade=priority(shifted, truth, optional), translation=scores['translation'])
    result = dict(common, matched=sorted(regions), gained_cells=sorted(cells-original), lost_cells=sorted(original-cells),
                  deltas={k:strict[k] for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')})
    print(key, result['deltas'], result['gained_cells'], result['lost_cells'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    jobs = sorted(suite.jobs([*suite.framed.SIZES, 'shelter']), key=lambda j: 'test-colonnade-01' not in j[1]['id'])
    assert len(jobs) == 204
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(grade, jobs))
    baseline = pool_scores.summarize(rows, 'baseline')
    assert baseline == read(prior.OUT / 'summary.json')['cohorts']
    after = pool_scores.summarize(rows, 'result')
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'summary.json', dict(baseline='e043ac2', cohorts=after, previous_cohorts=baseline, frames=len(rows),
                                    affected=[r for r in rows if r['matched']],
                                    frames_and_source_hashes={r['id']:r['source_hash'] for r in rows}))
    print('Fixed', after['fixed'])


if __name__ == '__main__':
    main()
