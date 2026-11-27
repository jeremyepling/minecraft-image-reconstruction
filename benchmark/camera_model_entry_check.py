"""Check archived expanded exports against the normal image-only CLI."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from benchmark.camera_model_controls import DEFAULT_BUILDS, REPEAT_BUILDS, check, frozen_frames, inventory_hash
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_scores import FIXED_DATASET, score_build
from benchmark.mask_blocks import decoded_pack


SAMPLE = ('train-pillars-01-marble-high-right', 'test-castle-01-limestone-low-left',
          'validation-hillside-chapel-2-warm-low-left', 'test-castle-01-ashlar-high-right')
MATCHED = WORKTREES / 'camera-geocalib/runs/camera-geocalib-01/matched-control/builds'

CLI = '''
from pathlib import Path
import runpy
import sys
from benchmark.expanded_baseline import replay_source_fields
from contextlib import nullcontext
source, baseline, output, mode = sys.argv[1:]
fields = replay_source_fields(Path(source), Path(baseline)) if mode == 'cached-fields' else nullcontext()
with fields:
    sys.argv = ['reconstruct.py', 'build', '--through', 'placement', '--version', '26.3', source, '--out', output]
    runpy.run_path('reconstruct.py', run_name='__main__')
'''


def run(task):
    entry, output, checkout, policy, live_fields = task
    key = entry['id']
    ordinary = DEFAULT_BUILDS / entry['dataset'] / key
    repeat_rescue = not entry['ordinary_supported'] and entry['repeat_supported']
    expected = REPEAT_BUILDS / key / 'repeat' if policy in ('repeat', 'budget') and repeat_rescue else ordinary
    matched = MATCHED / entry['dataset'] / key
    if policy == 'budget' and not entry['repeat_supported'] and (matched / 'build-manifest.json').is_file():
        expected = matched
    case = output / entry['dataset'] / key
    case.mkdir(parents=True, exist_ok=True)
    with (case / 'execution.log').open('w', encoding='utf-8') as transcript:
        result = subprocess.run([sys.executable, '-c', CLI, str(entry['rgb']), str(ordinary), str(case),
                                 'live-fields' if live_fields else 'cached-fields'], cwd=checkout,
                                env={**os.environ, 'MCIR_IMAGE_ONLY': '1'}, stdout=transcript,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'Normal build failed for {key}. See {case / "execution.log"}')
    actual, archived = score_build(entry, case), score_build(entry, expected)
    actual_pack, archived_pack = (folder / 'reconstruction.zip' for folder in (case, expected))
    actual_inventory, archived_inventory = (inventory_hash(decoded_pack(p)) for p in (actual_pack, archived_pack))
    check(actual_inventory, archived_inventory, f'{key} decoded export')
    projection = [read(folder / 'build-manifest.json')['placement']['P'] for folder in (case, expected)]
    check(projection[0], projection[1], f'{key} camera projection')
    for name in ('camera_usable', 'geometry', 'scores'):
        check(actual[name], archived[name], f'{key} {name}')
    print(f'{key}: camera, decoded export and scores match', flush=True)
    return dict(id=key, dataset=entry['dataset'], split=entry['split'], source_sha256=entry['rgb_sha256'],
                expected=('ordinary' if expected == ordinary else 'matched' if expected == matched else 'repeat'),
                camera_usable=actual['camera_usable'], exported=actual['exported'],
                camera_projection_exact=True, decoded_sha256=actual_inventory,
                normal_zip_sha256=digest(actual_pack), archived_zip_sha256=digest(archived_pack))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('policy', choices=('ordinary', 'repeat', 'budget'))
    parser.add_argument('--checkout', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=SAMPLE)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--live-fields', action='store_true')
    args = parser.parse_args()
    entries = {e['id']: e for e in frozen_frames() if e['dataset'] != FIXED_DATASET}
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, ((entries[key], output, checkout, args.policy, args.live_fields) for key in args.ids)))
    code = ('reconstruct.py', 'engine/surface_model.py', 'engine/visible_placement.py', 'engine/camera_repeat_floor.py')
    save(output / 'entry-check.json', dict(policy=args.policy, live_fields=args.live_fields,
         revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
         code_sha256={filename: digest(checkout / filename) for filename in code if (checkout / filename).exists()},
         cases=rows))


if __name__ == '__main__':
    main()
