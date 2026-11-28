"""Run the published floating wall and horizontal-cell arms on six new views."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import os
from unittest.mock import patch

from benchmark import floating_fresh_validation as fresh
from benchmark.expanded_baseline import ROOT, WORKTREES, save


OUT = ROOT / 'runs/floating-patch-validation'
HOLDOUT2_OUT = ROOT / 'runs/floating-patch-validation-holdout2'
REVIEW = ROOT / 'benchmark/reviews/floating-patch-validation-01'
FROZEN = REVIEW / 'frames.json'
HOLDOUT2_FROZEN = REVIEW / 'holdout2-frames.json'
BRIDGE = WORKTREES / 'floating-horizontal-bridges'
BRIDGE_REVISION = '6f5df52'
BRIDGE_SHA256 = '03239c244f622bf0d4d7859dfa4c93ba567660f02ea9e4db178f77de5db5e78a'


def bridge_runner():
    source = BRIDGE / 'benchmark/floating_horizontal_bridges.py'
    if sha256(source.read_text(encoding='utf-8').encode()).hexdigest() != BRIDGE_SHA256:
        raise ValueError(f'Horizontal-cell runner differs from {BRIDGE_REVISION}: {source}')
    spec = spec_from_file_location('published_horizontal_bridges', source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(case, *, output):
    with patch.object(fresh, 'OUT', output):
        fresh.run(case)
    bridge = bridge_runner()
    with patch.object(bridge, 'FRESH', output), patch.object(bridge, 'OUT', output / 'bridges'):
        bridge.run((case, True, True))
    return case['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--holdout2', action='store_true')
    args = parser.parse_args()
    os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    frozen, output = (HOLDOUT2_FROZEN, HOLDOUT2_OUT) if args.holdout2 else (FROZEN, OUT)
    entries = list(fresh.cases(frozen))
    save(output / 'manifest.json', dict(frozen_manifest_sha256=fresh.fixture_hash(frozen), cases=entries))
    if args.manifest_only:
        return
    with ProcessPoolExecutor(args.workers) as pool:
        produce = partial(run, output=output)
        results = [pool.submit(produce, entry) for entry in entries]
        for result in as_completed(results):
            print('COMPLETE', result.result(), flush=True)


if __name__ == '__main__':
    main()
