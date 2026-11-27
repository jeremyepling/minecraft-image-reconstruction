"""Replay local obstruction removal with a single immutable, source-only opening detector."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import facade_inset_round as suite
from benchmark.edge_vote_outline_eval import baseline_pack
from benchmark.mask_blocks import registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import save
from benchmark.thin_face_placement_run import ref_inputs
from engine.pavement_obstructions import window_obstructions


def inspect(job):
    from engine.pavement_opening import inspect_openings
    split, frame, prior, control = job
    image, parts, _, camera = suite.ordinary_inputs(*job)
    records = inspect_openings(image, parts, camera)
    rejected = {r['id'] for r in records if r['reject']}
    pack_hash, before = baseline_pack(job)
    remove = window_obstructions(image.size, camera, [p for p in parts if p['id'] in rejected], before)
    row = dict(id=frame['id'], split=split, source_hash=frame['files']['rgb.png'], pack_hash=pack_hash,
               proposed=records, removed=sorted(remove))
    if remove:
        truth, occupied, transform, _ = ref_inputs(job)
        score = placement_scores(registered(before-remove, transform), registered(before, transform),
                                 dict.fromkeys(truth), occupied)
        row['relative'] = score['relative_geometry']
        row['registered'] = score['metrics']
        row['translation'] = score['translation']
        row['baseline_translation'] = score['baseline_translation']
    return row


def init(directory):
    if directory:
        import engine
        engine.__path__.insert(0, str(directory.resolve()/'engine'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector-root', type=Path)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    jobs = list(suite.jobs([*suite.framed.SIZES, 'shelter']))
    assert len(jobs) == 204
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.detector_root,)) as pool:
        rows = list(pool.map(inspect, jobs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, dict(base='a6e0195', frames=len(rows), cases=rows))
    for row in rows:
        if any(r['reject'] for r in row['proposed']):
            print(row['id'], len(row['removed']), row.get('relative', {}))


if __name__ == '__main__':
    main()
