"""Replay a second-edge gate and check accepted joins against withheld native origins."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.expanded_wall_grids import NATIVE, OUT, cases, evaluate
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import axis_vector
from benchmark.sky_crop_guard import pooled
from engine.geometry import project


def replay(args):
    case, arm = args
    frame = Path(case['frame'])
    case = dict(case, variants={'sky': dict(image=str(frame / 'rgb.png'), source_sha256=case['rgb_sha256'])})
    settings = dict(edges=2) if arm.endswith('two-edges') else dict(edge_px=1e-4, coplanar_px=1e-4)
    signed = arm.startswith('signed')
    tree = 'signed/charts' if signed else 'source-normal/charts'
    evaluate(case, OUT / tree, OUT / arm, signed=signed, **settings)


def true_wall_origins(charts, labels):
    """Scoring only. Verify each supplied local cell maps to exactly one recorded visible face."""
    selected = {c['id']: c for c in charts if c['method'] == 'independently-rebased-visible-wall-grid'}
    faces = defaultdict(list)
    for f in labels['faces']:
        pid = f"plane-{f['surface_id']}"
        if pid in selected:
            faces[pid].append(f)
    origins = {}
    all_positions = set()
    for pid, chart in selected.items():
        side, up = (axis_vector(i) for i in chart['axes'])
        normal = axis_vector(chart['normal_axis'])
        inverse = np.linalg.inv(chart['H'])
        first = faces[pid][0]
        uv = project(inverse, np.array(first['corners_px'][0]))
        assert np.max(np.abs(uv - np.rint(uv))) < 1e-5
        u, v = np.rint(uv).astype(int)
        origin = np.array(first['corners_world'][0]).astype(int) - u * side - v * up
        offset = np.minimum(0, side) + np.minimum(0, up) - np.maximum(0, normal)
        positions = {tuple(origin + offset + c['u'] * side + c['v'] * up) for c in chart['cells']}
        expected = {tuple(f['xyz_world']) for f in faces[pid]}
        assert positions == expected
        all_positions.update(positions)
        origins[pid] = origin
    return origins, len(all_positions)


def inspect(case):
    name = case['id']
    tail = Path('assembly/connected-atlas/sky') / name
    native = read(NATIVE / tail / 'score.json')
    labels = read(Path(case['frame']) / 'labels.json')
    signed_path = OUT / 'signed/charts/sky' / name / 'charts.json'
    origins, npositions = true_wall_origins(read(signed_path)['charts'], labels)
    check = read(OUT / 'checks' / (name + '.json'))
    row = dict(case=check, visible_wall_blocks=npositions, native_score=native['score']['fixed'] or native['score']['relaxed_yaw'])
    for arm in ('source-normal', 'signed', 'signed-two-edges', 'signed-tight-edges', 'source-normal-two-edges'):
        folder = OUT / arm / tail
        r = read(folder / 'score.json')
        path = OUT / 'source-normal/charts/sky' / name / 'charts.json' if arm.startswith('source-normal') else signed_path
        assert digest(path) == r['chart_sha256'] and digest(folder / 'reconstruction.zip') == r['pack_sha256']
        p = read(folder / 'placements.json')
        assert decoded_pack(folder / 'reconstruction.zip') == {tuple(x): material for x, material in p['local']}
        joins = Counter()
        if arm.startswith('signed'):
            for edge in r['joins']:
                a, b = edge['chart_ids']
                if a not in origins or b not in origins:
                    joins['nonwall'] += 1
                else:
                    correct = np.array_equal(origins[b] - origins[a], edge['shift'])
                    joins[edge['kind'] + ('_correct' if correct else '_wrong')] += 1
        row[arm] = dict(score=r['score']['fixed'] or r['score']['relaxed_yaw'], export_sha256=r['pack_sha256'],
                        packet_sha256=r['chart_sha256'], charts=r['charts_seen'], oriented=r['oriented_charts'],
                        selected=r['kept_charts'], components=r['components'], accepted_joins=len(r['joins']),
                        rejected_joins=len(r['conflicts']), join_audit=joins, fixed_yaw=r['score']['fixed'] is not None)
    return row


def report():
    rows = [inspect(case) for case in cases()]
    cohorts = {}
    for ds in ('all', 'visible-builds-v2', 'visible-builds-v3', 'decorative-builds-v1'):
        batch = [r for r in rows if ds == 'all' or r['case']['dataset'] == ds]
        stats = {}
        for arm in ('native', 'source-normal', 'signed', 'signed-two-edges', 'signed-tight-edges', 'source-normal-two-edges'):
            scores = [r['native_score'] if arm == 'native' else r[arm]['score'] for r in batch]
            stats[arm] = pooled(scores)
            details = stats[arm]
            details['materials'] = {k:sum(s['scores'][k] for s in scores) for k in ('assigned','fallback','assigned_correct','assigned_positions')}
            details['materials'].update(correct=sum(s['scores']['material']['correct'] for s in scores),
                                        positions=sum(s['scores']['material']['positions'] for s in scores))
            if arm != 'native':
                details['joins'] = dict(sum((Counter(r[arm]['join_audit']) for r in batch), Counter()))
                details['counts'] = {k:sum(r[arm][k] for r in batch) for k in ('charts','oriented','selected','components','accepted_joins','rejected_joins','fixed_yaw')}
        cohorts[ds] = dict(views=len(batch), visible_wall_blocks=sum(r['visible_wall_blocks'] for r in batch), arms=stats)
        print(ds, len(batch), {k:round(v['geometry_strict']['f1'] * 100, 2) for k,v in stats.items()}, flush=True)
    save(ROOT / 'benchmark/reviews/expanded-wall-grids-01/scored.json.gz', dict(cohorts=cohorts, cases=rows))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--replay', nargs='+', choices=['signed-two-edges', 'signed-tight-edges', 'source-normal-two-edges'])
    cli.add_argument('--workers', type=int, default=4)
    cfg = cli.parse_args()
    if cfg.replay:
        with ProcessPoolExecutor(cfg.workers) as pool:
            list(pool.map(replay, ((case, arm) for case in cases() for arm in cfg.replay)))
    report()
