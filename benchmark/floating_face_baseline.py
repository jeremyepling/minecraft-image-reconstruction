"""Verify the archived horizontal exports and optionally produce the six new controls.

Run as a script so the pinned validation checkout supplies the baseline modules.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from subprocess import check_output
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT.with_name('floating-patch-validation')
VALIDATION_REVISION = '7958722'
REVIEW = ROOT / 'benchmark/reviews/floating-face-audit-01'
sys.path.insert(0, str(VALIDATION))

from benchmark.expanded_baseline import digest, read, save
from benchmark.floating_patch_scores import (
    BRIDGE, HOLDOUT2_FROZEN, HOLDOUT2_OUT, PUBLISHED_SCORES_SHA256,
    run_panels, summarize, verify_case, wall_roots,
)
from benchmark.floating_patch_validation import REVIEW as ARCHIVE, fresh, run


def panel_fixture(group):
    return REVIEW / f'{group}-frames.json'


def panel_output(group):
    return ROOT / 'runs' / f'floating-{group}'


def panel_roots(root):
    return wall_roots(root, root / 'ordinary', root / 'horizontal-cells/assembly')


def previous_results():
    bridge_path = BRIDGE / 'benchmark/reviews/floating-horizontal-bridges-01/results.json.gz'
    if digest(bridge_path) != PUBLISHED_SCORES_SHA256:
        raise ValueError(f'Published bridge score hash changed: {bridge_path}')
    results = {panel: {row['id']: row['score']['horizontal-cells'] for row in records['rows']}
               for panel, records in read(bridge_path).items()}
    for panel, file in (('new', 'results.json.gz'), ('holdout2', 'holdout2-results.json.gz')):
        rows = read(ARCHIVE / file)['panels'][panel]['rows']
        results[panel] = {row['id']: row['arms']['horizontal-cells']['score'] for row in rows}
    return results


def verify(pool, new_panels):
    expected = previous_results()
    panels = list(run_panels())
    panels.append(('holdout2', list(fresh.cases(HOLDOUT2_FROZEN)), panel_roots(HOLDOUT2_OUT)))
    for group in new_panels:
        panels.append((group, list(fresh.cases(panel_fixture(group))), panel_roots(panel_output(group))))
    names = {'previous': 'original', 'new': 'holdout'}
    results = {}
    for panel, cases, roots in panels:
        rows = list(pool.map(verify_case, ((case, roots) for case in cases)))
        if panel in expected:
            selected = {row['id']: row['arms']['horizontal-cells']['score'] for row in rows}
            if selected != expected[panel]:
                raise ValueError(f'Decoded horizontal-cell scores changed: {panel}')
        cohorts = summarize(rows)
        label = names.get(panel, panel)
        results[label] = dict(rows=rows, cohorts=cohorts, matched_published_scores=panel in expected)
        print(label, cohorts['all']['views'], cohorts['all']['arms']['horizontal-cells'], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--produce-holdout', nargs='+', choices=('holdout3', 'holdout4'), default=[])
    parser.add_argument('--verify-holdout', nargs='*', choices=('holdout3', 'holdout4'), default=['holdout3'])
    args = parser.parse_args()
    actual = check_output(['git', '-C', str(VALIDATION), 'rev-parse', 'HEAD'], text=True).strip()
    if not actual.startswith(VALIDATION_REVISION):
        raise ValueError(f'Expected baseline revision {VALIDATION_REVISION}, found {actual}')
    os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with ProcessPoolExecutor(args.workers) as pool:
        for group in args.produce_holdout:
            fixture = panel_fixture(group)
            output = panel_output(group)
            entries = list(fresh.cases(fixture))
            save(output / 'manifest.json', dict(frozen_manifest_sha256=fresh.fixture_hash(fixture), cases=entries))
            for ident in pool.map(partial(run, output=output), entries):
                print('COMPLETE', ident, flush=True)
        results = verify(pool, args.verify_holdout)
    save(REVIEW / 'horizontal-baseline.json.gz', dict(validation_revision=actual,
         fixture_sha256={g: fresh.fixture_hash(panel_fixture(g)) for g in args.verify_holdout}, panels=results))


if __name__ == '__main__':
    main()
