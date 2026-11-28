"""Measure the post-expansion chart partition bypass with ordinary Floating consumers."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import os

import cv2
import numpy as np

from benchmark import floating_v1_expanded as expanded
from benchmark.expanded_baseline import digest, read, save
from benchmark.expanded_scores import load_reference
from benchmark.floating_expanded_run import run_case, summarize
from benchmark.mask_blocks import decoded_pack
from engine import floating
from training.floating_shared_inputs import cases

IDS = [f'front-{i:03}-{family}-full-noon' for i, family in (
    (13, 'windmill'), (14, 'vault'), (55, 'windmill'), (56, 'vault'), (97, 'windmill'), (98, 'vault'))]


@contextmanager
def intercept(receipt, bypass):
    packet = expanded.expanded_packet
    partition = expanded.pink.partition_charts

    def measure(charts, flags, fields):
        assert set(flags) == set(fields)
        split, details = partition(charts, flags, fields)
        result = (charts, {}) if bypass else (split, details)
        cells = lambda items: sum(len(chart['cells']) for chart in items)
        receipt.append(dict(requested=sorted(flags), fields=sorted(fields),
                            input_charts=len(charts), split_charts=len(split), output_charts=len(result[0]),
                            input_cells=cells(charts), split_cells=cells(split), output_cells=cells(result[0]),
                            split_details=details))
        return result

    def wrapped(*args, **kwargs):
        with patch.object(expanded.pink, 'partition_charts', measure):
            return packet(*args, **kwargs)

    with patch.object(expanded, 'expanded_packet', wrapped):
        yield


def test_wrapper():
    charts = [dict(id='sample', H=np.eye(3).tolist(), cells=[dict(u=0, v=0), dict(u=1, v=0)])]
    flags = {'sample'}
    fields = {'sample': np.array([[1, 2]])}
    partition = expanded.pink.partition_charts
    stock = partition(charts, flags, fields)
    def fixture():
        return expanded.pink.partition_charts(charts, flags, fields)
    with patch.object(expanded, 'expanded_packet', fixture):
        for bypass in (False, True):
            audit = []
            with intercept(audit, bypass):
                result = expanded.expanded_packet()
            assert result == ((charts, {}) if bypass else stock)
            assert len(audit) == 1 and audit[0]['split_charts'] == 2
            assert expanded.pink.partition_charts is partition


def run_control(job):
    entry, out, archive, stock = job
    audit = []
    with intercept(audit, True):
        row = run_case((entry, out, archive))
    assert len(audit) == 1
    folder = out / 'original' / entry['id']
    save(folder / 'partition-audit.json', audit[0])
    def aligned(root):
        return {tuple(r[:3]): r[3] for r in read(root / 'aligned-placements.json.gz')}
    base = stock / 'original' / entry['id']
    left, right = aligned(base), aligned(folder)
    truth = load_reference(entry)
    added, removed = right.keys() - left.keys(), left.keys() - right.keys()
    delta = dict(true_added=len(added & truth.truth.keys()), true_removed=len(removed & truth.truth.keys()),
                 extra_added=len(added - truth.truth.keys()), extra_removed=len(removed - truth.truth.keys()),
                 material_changes=sum(left[p] != right[p] for p in left.keys() & right.keys()),
                 same_decoded_pack=decoded_pack(base/'reconstruction.zip') == decoded_pack(folder/'reconstruction.zip'))
    save(folder / 'paired-delta.json', delta)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['tower', 'controls'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--fields', type=Path, required=True)
    p.add_argument('--stock', type=Path)
    p.add_argument('--case', action='append')
    args = p.parse_args()
    os.environ['MCIR_PREPARED_ATLAS_26_3'] = str(args.fixture.resolve() / 'atlas')
    test_wrapper()
    cv2.setNumThreads(1)
    if args.mode == 'tower':
        target = args.out / 'tower'
        audit = []
        with intercept(audit, True):
            assembly = floating.build(args.source, target, field_cache=args.fields)
        blocks = decoded_pack(target/'reconstruction.zip')
        assert len(blocks) == assembly['size'] and len(audit) == 1
        save(target/'partition-audit.json', dict(audit[0], exported=len(blocks), source_sha256=digest(args.source),
              materials=dict(Counter(blocks.values())), pack_sha256=digest(target/'reconstruction.zip')))
        print(target, len(blocks), audit, flush=True)
    else:
        requested = args.case or IDS
        entries = cases(args.fixture.resolve(), cohort='front-full', ids=requested, lighting='noon')
        assert sorted(row['id'] for row in entries) == sorted(requested)
        save(args.out/'fixtures.json', dict(cases=entries))
        with ProcessPoolExecutor(3) as pool:
            rows = list(pool.map(run_control, [(e, args.out, args.fields, args.stock) for e in entries]))
        save(args.out/'summary.json', dict(method='Bypass post-expansion depth split', requested=requested,
             all_arms=summarize(rows), baseline=summarize([read(args.stock/'original'/i/'case.json') for i in requested])))


if __name__ == '__main__':
    main()
