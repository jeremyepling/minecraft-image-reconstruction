"""Reconcile a shared corner with its receiver's independent horizontal repeat measurement."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.floating_bridges_report import summarize
from benchmark.floating_corner_overwrite import jobs, regenerate, scan, sources
from benchmark.floating_corner_overwrite_report import choose, counts
from benchmark.floating_extra_budget import LOCATION, combined
from benchmark.floating_horizontal_bridges import horizontal_charts, solver
from benchmark.shared_corner_rows import confident_count
from engine.geometry import project

OUT = ROOT / 'runs/pair-repair'
REVIEW = ROOT / 'benchmark/reviews/floating-corner-pair-01'


def component(charts, planes, rgb, include_horizontal):
    """Use the same source-only face assembly as the exporter, including visible horizontal faces."""
    walls, _ = solver.oriented(charts)
    measured = walls + horizontal_charts(walls, planes, rgb.shape[1::-1], rgb) if include_horizontal else walls
    result = solver.assemble(measured, min_edges=1, edge_pixels=2.5, coplanar_pixels=.5)
    selected = set(result['groups'][0]['origins']) if result['groups'] else set()
    joins = {tuple(sorted(edge['chart_ids'])) for edge in result['joins'] if set(edge['chart_ids']) <= selected}
    return selected, joins


def widths(grid, quad):
    uv = project(np.linalg.inv(grid['H']), quad)
    return np.abs(uv[[1, 2], 0] - uv[[0, 3], 0])


def witnesses(seed, grown):
    """Count repeated atlas matches outside the cells used to establish this earlier grid."""
    def observations(chart):
        return {(p['u'], p['v']):p['block_id'].removeprefix('minecraft:') for p in chart['cells']
                if not p.get('fallback') and p.get('confidence', 0) >= .5}
    original = observations(seed)
    seeds = {(p['u'], p['v']) for p in seed['cells']}
    by_material = Counter(original.values())
    evidence = {}
    for material, n in by_material.items():
        held = [uv for uv, m in observations(grown).items() if m == material and uv not in seeds]
        if n >= 2 and len({uv[0] for uv in held}) >= 2 and len({uv[1] for uv in held}) >= 2:
            evidence[material] = len(held)
    return evidence


def select(options, main, edges):
    candidates = []
    for row in options:
        proposed_edges = {tuple(e) for e in row['edges']}
        if (row['width_ok'] and row['atlas_holdout'] and set(row['pair']) <= main
                and main <= set(row['selected']) and edges <= proposed_edges
                and tuple(sorted(row['pair'])) in proposed_edges):
            candidates.append(row)
    return max(candidates, key=lambda r:(r['width_improvement'], r['arm']))['arm'] if candidates else 'keep'


def alternatives(packet, dense, eligible, planes, measurements, rgb, *, include_horizontal=True):
    """Propose both intact earlier grids at each surviving corner. All inputs come from the source image."""
    charts = {c['id']:c for c in packet['charts']}
    shapes = {s['id']:s for s in planes}
    main, edges = component(packet['charts'], planes, rgb, include_horizontal)
    options, proposals = [], {}
    for seam in eligible:
        donor, receiver = seam['donor_id'], seam['receiver_id']
        pair = [donor, receiver]
        replacements = {cid:regenerate(dense[cid], shapes[cid], rgb) for cid in pair}
        proposal = [replacements.get(c['id'], c) for c in packet['charts']]
        selected, joined = component(proposal, planes, rgb, include_horizontal)
        row = measurements.get(receiver, {})
        expected = confident_count(row, 0)
        before = after = None
        width_ok = False
        improvement = 0.
        if expected is not None and row.get('quad'):
            before = widths(charts[receiver], row['quad'])
            after = widths(replacements[receiver], row['quad'])
            before_error = abs(before - expected)
            after_error = abs(after - expected)
            width_ok = bool(min(before_error) >= 1 and max(after_error) <= .5)
            improvement = float(min(before_error) - max(after_error))
        arm = f'p{len(options)}'
        options.append(dict(arm=arm, pair=pair, expected=expected, width_ok=width_ok,
                            current_widths=before.tolist() if before is not None else None,
                            previous_widths=after.tolist() if after is not None else None,
                            width_improvement=improvement, atlas_holdout=witnesses(dense[receiver],replacements[receiver]),
                            selected=sorted(selected), edges=sorted(joined)))
        proposals[arm] = proposal
    return options, proposals, main, edges


def execute(condition):
    cv2.setNumThreads(1)
    receipt, case = scan(condition, False)
    name, panel = case['id'], condition['panel']
    base = Path(receipt['base'])
    rel = Path('sky')/name
    src = sources(panel)[0]
    packet = read(Path(receipt['files']['keep']))
    eligible = [s for s in receipt['accepted_seams']
                if all(receipt['charts'][s[role]]['status']=='offered' for role in ('donor_id','receiver_id'))]
    arms = {'keep':base/'assembly'}
    options, proposals, main, edges = [], {}, set(), set()
    if eligible:
        rgb = np.array(Image.open(case['variants']['sky']['image']).convert('RGB'))
        planes = read(src/rel/'source-handoff/surfaces.json')['planes']
        measurements = {s['id']:s for s in read(src/rel/'source-handoff/counts.json')['surfaces']}
        dense = {c['id']:c for c in read(Path(receipt['files']['dense']))['charts']}
        options, proposals, main, edges = alternatives(packet,dense,eligible,planes,measurements,rgb)
    arm = select(options, main, edges)
    for option in options:
        key = option['arm']
        root = OUT/panel/key
        save(root/'charts'/rel/'charts.json', dict(packet,charts=proposals[key]))
        with patch.object(combined, 'inputs', sources):
            combined.run((panel,case,root/'charts',root/'assembly'))
        actual = read(root/'assembly'/LOCATION/name/'score.json')
        assert set(option['selected']) == set(actual['selected_origins'])
        arms[key] = root/'assembly'
    # Native geometry and material labels are used by the scorer after selecting the arm.
    scored = summarize([case],arms,LOCATION)['rows'][0]
    result = dict(scored,panel=panel,choice=arm,options=options,selected_before=sorted(main),edges_before=sorted(edges),
                  source_sha256=receipt['source_sha256'],hashes=receipt['sha256'],reference_sha256=receipt['reference_sha256'],
                  pack_sha256={k:digest(r/LOCATION/name/'reconstruction.zip') for k,r in arms.items()})
    save(OUT/panel/'rows'/(name+'.json'),result)
    return result


def summary(rows):
    def summarize_choices(choices):
        return dict(placement={k:counts(rows,choices,'geometry',k) for k in ('strict','facade')},
                    joint={k:counts(rows,choices,'scores',k) for k in ('strict','facade')},
                    paired={k:sum(r['paired'][arm][k] for r,arm in zip(rows,choices)) for k in ('gained','lost','added','removed')})
    return dict(views=len(rows),eligible=sum(bool(r['options']) for r in rows),
                pairs=sum(len(r['options']) for r in rows),selected=sum(r['choice']!='keep' for r in rows),
                baseline=summarize_choices(['keep']*len(rows)),candidate=summarize_choices([r['choice'] for r in rows]),
                reference_chooser={f'{field}_{kind}':summarize_choices(choose(rows,field,kind))
                                   for field in ('geometry','scores') for kind in ('strict','facade')})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    conditions = [r for r,special in jobs() if not special]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(execute, conditions))
    cohorts = dict(all=rows, fixed_sky=[r for r in rows if r['panel']=='sky'])
    for dataset in sorted({r['dataset'] for r in rows}):
        cohorts[dataset] = [r for r in rows if r['dataset']==dataset]
    save(REVIEW/'rows.json.gz',rows)
    save(REVIEW/'summary.json',{k:summary(v) for k,v in cohorts.items()})
    print([(r['id'],r['options'][int(r['choice'][1:])]['pair']) for r in rows if r['choice']!='keep'])
