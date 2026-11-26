"""Run a placement method on a prepared native suite and decode the resulting packs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess

from benchmark.relative_geometry import PRIMARY_METRIC
from benchmark.texture_camera_eval import digest, read, save
from benchmark.visible_depth_score import aggregate, init_candidate
from benchmark.visible_depth_transfer import score_case


def run(inputs, out, root, adapter, baseline, workers):
    if baseline and out.resolve() == baseline.resolve():
        raise ValueError('Candidate and reference outputs need separate directories')
    manifest = inputs / 'screen.json'
    selected = read(manifest)['selected']
    jobs = [(f,inputs,'comparison',out,baseline) for f in selected]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_candidate,
                             initargs=(root, adapter)) as pool:
        rows = list(pool.map(score_case, jobs))
    if baseline:
        previous = {row['id']: row for row in read(baseline/'cases.json')}
        if previous.keys() != {row['id'] for row in rows}:
            raise ValueError('The candidate and baseline contain different frames')
        for row in rows:
            if any(row[k] != previous[row['id']][k] for k in ('rgb_sha256','surface_sha256','counts_sha256')):
                raise ValueError(f"Placement inputs differ for {row['id']}")
    save(out/'cases.json',rows)
    summary = aggregate(rows)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    module = root.joinpath(*adapter.split(':')[0].split('.')).with_suffix('.py')
    save(out/'manifest.json',dict(primary_metric=PRIMARY_METRIC, adapter=adapter, revision=revision, adapter_sha256=digest(module),
                                  selection_sha256=digest(manifest), cases_sha256=digest(out/'cases.json'),
                                  summary=summary))
    print(summary,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,help='Directory with per-image decoded reference coordinates.')
    parser.add_argument('--candidate-root',type=Path,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    run(args.inputs,args.out,args.candidate_root.resolve(),args.candidate,args.baseline,args.workers)
