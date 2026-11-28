"""Serialize saved Floating masks and block-grid attempts in source-image coordinates."""
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from benchmark.floating_stage_report import hash_file, read
from engine.geometry import UNIT, project


PIPELINE = (
    ('Counter chart', 'dense', 'counter-charts.json'),
    ('Dense atlas', 'dense', 'dense-charts.json'),
    ('Shared rows', 'shared', 'charts.json'),
    ('Native atlas', 'native', 'charts.json'),
    ('Registered seams', 'native/registered-seam', 'charts.json'),
    ('Connected seed', 'seeds/connected', 'charts.json'),
    ('Expanded chart', 'package/charts', 'charts.json'),
    ('Baseline input', 'split-border/charts', 'charts.json'),
    ('Inherited depth split', 'pregrid/v1/split-only/charts', 'charts.json'),
    ('Full refit arm', 'pregrid/v1/complete/charts', 'charts.json'),
    ('Partial refit arm', 'pregrid/v1/partial/charts', 'charts.json'),
    ('Fresh-only arm', 'pregrid/v1/fresh-only/charts', 'charts.json'),
)


def parent(chart_id):
    return re.sub(r'~p\d+$', '', chart_id.removeprefix('horizontal-'))


def rings_path(outline, size):
    scale = np.asarray(size) / outline['size']
    return ''.join(
        'M' + 'L'.join(f'{x:.1f} {y:.1f}' for x, y in np.asarray(ring['points']) * scale) + 'Z'
        for ring in outline['rings'] if ring['points']
    )


def mask_path(mask):
    # Merge identical runs in adjacent rows. Coordinates remain on native pixel edges.
    spans = defaultdict(list)
    for y, row in enumerate(mask):
        corners = np.flatnonzero(np.diff(row, prepend=False, append=False))
        for left, right in corners.reshape(-1, 2):
            rectangles = spans[left, right]
            if rectangles and rectangles[-1][1] == y:
                rectangles[-1][1] += 1
            else:
                rectangles.append([y, y + 1])
    return ''.join(f'M{left} {top}h{right-left}v{bottom-top}H{left}Z'
                   for (left, right), rectangles in spans.items() for top, bottom in rectangles)


def grid_path(quads):
    edges = {}
    for quad in quads:
        if not np.isfinite(quad).all():
            raise ValueError('Grid has a nonfinite source projection.')
        vertices = [tuple(round(float(value), 1) for value in point) for point in quad]
        for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True):
            if a != b:
                edges.setdefault(tuple(sorted((a, b))), None)
    return ''.join(f'M{a[0]} {a[1]}L{b[0]} {b[1]}' for a, b in edges)


def packet_grids(path, source, receipts):
    packet = read(path)
    counts = source['counts']
    surface_sha = source['surfaceSha']
    if (packet['source_sha256'] != source['sourceSha'] or packet['surfaces_sha256'] != surface_sha or
            packet['counts_sha256'] != counts):
        raise ValueError(f'Grid packet belongs to another source or handoff: {path}')
    digest = hash_file(path)
    final = [label for label, receipt in receipts.items() if digest == receipt['chart_sha256']]
    return packet['charts'], final


def final_receipts(root, row, case, source):
    from pathlib import PureWindowsPath

    results = {}
    for label, arm, proof in [('Baseline', 'adopted', 'baseline'), ('Candidate', 'nonshrinking', 'candidate')]:
        location = PureWindowsPath(row['roots'][arm])
        stage = location.parts[location.parts.index(root.name) + 1:]
        assembly = root.joinpath(*stage)
        folder = assembly / 'connected-atlas' / 'sky' / case['id']
        receipt = read(folder / 'score.json')
        expected = case[proof]['pack_sha256']
        if (receipt['pack_sha256'] != expected or receipt['source_sha256'] != source['sourceSha'] or
                receipt['surfaces_sha256'] != source['surfaceSha'] or receipt['counts_sha256'] != source['counts']):
            raise ValueError(f'Selected grid packet disagrees with the published score: {folder}')
        results[label] = receipt
    return results


def add_grid(surface, label, chart, status=None):
    cells = chart['cells']
    if not cells:
        return
    uv = np.array([(item['u'], item['v']) for item in cells])
    quads = project(np.asarray(chart['H']), uv[:, None, :] + UNIT)
    path = grid_path(quads)
    chart_id = chart['id']
    layers = surface['layers']
    for item in layers:
        if (item.get('path') == path and item.get('chart') == chart_id and
                item.get('cellCount') == len(cells)):
            item['stages'].append(label)
            item['status'].extend(value for value in status or [] if value not in item['status'])
            return
    layers.append(dict(label=label, stages=[label], chart=chart_id, path=path, cellCount=len(cells),
                       part=chart_id if '~p' in chart_id else None, status=status or []))


def horizontal_replay(charts, planes, image, final, receipts):
    from benchmark.floating_horizontal_bridges import horizontal_charts, solver

    walls, _ = solver.oriented(charts)
    found = horizontal_charts(walls, planes, image.size, np.array(image))
    ids = {chart['id'] for chart in found}
    for label in final:
        receipt = receipts[label]
        selected = {key for key in receipt['selected_origins'] if key.startswith('horizontal-')}
        if not selected <= ids or len(walls) + len(found) != receipt['oriented_charts']:
            raise ValueError(f'Horizontal replay did not reproduce the {label} assembly receipt.')
    return found


def build(root, case, image, out):
    """Read the original V1 benchmark, including archived model proposals and skipped regions."""
    marker = case['id']
    rel = Path('sky') / marker
    handoff = root / 'source' / rel / 'source-handoff'
    planes_path = handoff / 'surfaces.json'
    counts_path = handoff / 'counts.json'
    planes_doc = read(planes_path)
    planes = planes_doc['planes']
    if tuple(planes_doc['image_size']) != image.size:
        raise ValueError(f'Native source size differs: {marker}')
    row = read(root / 'rows' / f'{marker}.json.gz')
    source = dict(sourceSha=case['source_sha256'], surfaceSha=hash_file(planes_path), counts=hash_file(counts_path))
    if row['source_sha256'] != source['sourceSha'] or row['reference_sha256'] != case['reference_sha256']:
        raise ValueError(f'Surface evidence differs from the archived comparison: {marker}')
    receipts = final_receipts(root, row, case, source)
    by_id = {}
    for plane in planes:
        marker_id = plane['id']
        by_id[marker_id] = dict(id=marker_id, mask=rings_path(plane['outline'], image.size),
                                bounds=plane.get('source_crop', {}).get('xyxy'),
                                pixels=plane['support_area'], geometry=plane.get('geometry_status'), layers=[],
                                parts={}, depth=plane.get('relative_inverse_depth'))
    for count in read(counts_path)['surfaces']:
        evidence = by_id[count['id']]
        quads = [cell['quad'] for cell in count.get('visible_cells', [])]
        evidence['count'] = dict(method=count['method'], blocks=count['dimensions']['blocks'], cellCount=len(quads))
        if quads:
            evidence['layers'].append(dict(label='Block Counter', stages=['Block Counter'], chart=count['id'],
                                           path=grid_path(quads), cellCount=len(quads), status=[]))

    for marker_id, info in row['audit'].items():
        if 'parts' not in info:
            continue
        ids_path = root / 'pregrid/v1/source' / marker / f'{marker_id}-ids.png'
        if hash_file(ids_path) != info['mask_sha256']:
            raise ValueError(f'Depth partitions differ from the scored run: {ids_path}')
        ids = np.asarray(Image.open(ids_path))
        if ids.shape != (image.height, image.width):
            raise ValueError(f'Depth mask is out of register: {ids_path}')
        for part, detail in info['parts'].items():
            member = ids == int(part)
            if int(member.sum()) != detail['pixels']:
                raise ValueError(f'Depth region differs from its receipt: {ids_path}/{part}')
            by_id[marker_id]['parts'][f'{marker_id}~p{part}'] = dict(mask=mask_path(member),
                pixels=detail['pixels'], inherited=detail['inherited'], fits=list(detail['grids']))

    loaded = set()
    for label, channel, file in PIPELINE:
        packet_path = root / channel / rel / file
        if not packet_path.is_file():
            continue
        charts, final = packet_grids(packet_path, source, receipts)
        for chart in charts:
            status = [f'{name} · ' + ('selected' if chart['id'] in receipts[name]['selected_origins'] else 'outside selected assembly')
                      for name in final]
            add_grid(by_id[parent(chart['id'])], label, chart, status)
        if final and any(name not in loaded for name in final):
            for chart in horizontal_replay(charts, planes, image, final, receipts):
                status = [f'{name} · ' + ('selected' if chart['id'] in receipts[name]['selected_origins'] else 'outside selected assembly')
                          for name in final]
                add_grid(by_id[parent(chart['id'])], 'Assembly roof/underside (replay)', chart, status)
        loaded.update(final)
    if loaded != set(receipts):
        raise ValueError(f'Missing exact assembly chart packet: {case["id"]}')
    data = dict(schema=1, id=case['id'], sourceSha256=source['sourceSha'], size=image.size,
                surfaces=list(by_id.values()))
    out.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf8')
