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
        # Archived roots can be moved or renamed. The assembly's relative path is
        # defined by its stage, independent of the old machine's run-root name.
        stage = next((location.parts[i:] for i, name in enumerate(location.parts)
                      if name in ('split-border', 'pregrid') and location.parts[-1] == 'assembly'), None)
        if stage is None:
            raise ValueError(f'Unrecognized V1 assembly location: {location}')
        assembly = root.joinpath(*stage)
        folder = assembly / 'connected-atlas' / 'sky' / case['id']
        receipt = read(folder / 'score.json')
        expected = case[proof]['pack_sha256']
        if (receipt['pack_sha256'] != expected or receipt['source_sha256'] != source['sourceSha'] or
                receipt['surfaces_sha256'] != source['surfaceSha'] or receipt['counts_sha256'] != source['counts']):
            raise ValueError(f'Selected grid packet disagrees with the published score: {folder}')
        pack = folder / 'reconstruction.zip'
        if hash_file(pack) != expected:
            raise ValueError(f'Export differs from score receipt: {pack}')
        results[label] = dict(receipt, folder=folder, scoreHash=hash_file(folder / 'score.json'))
    return results


def add_grid(surface, label, chart, status=None):
    cells = chart['cells']
    if not cells:
        surface['decisions'].append(dict(arm=label, kind='empty-chart', detail=chart['id']))
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
            item.setdefault('methods', {})[label] = chart.get('method')
            item['status'].extend(value for value in status or [] if value not in item['status'])
            return
    layers.append(dict(label=label, stages=[label], chart=chart_id, path=path, cellCount=len(cells),
                       part=chart_id if '~p' in chart_id else None, status=status or [],
                       methods={label: chart.get('method')}))


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
    return walls, found


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
    by_id = source_surfaces(planes, read(counts_path), image)
    provenance = dict(surfaceSha256=source['surfaceSha'], countsSha256=source['counts'],
                      rowSha256=hash_file(root / 'rows' / f'{marker}.json.gz'), packets=[], arms={})
    manifest = read(handoff / 'build-manifest.json')
    if manifest['image_sha256'] != source['sourceSha']:
        raise ValueError(f'Source handoff belongs to a different image: {marker}')
    provenance.update(handoffSha256=hash_file(handoff / 'build-manifest.json'),
                      models=manifest.get('model_sha256'), fields=row.get('source_fields', {}).get('fields_sha256'))
    sky = manifest.get('surface_finder', {}).get('boundary_settings', {}).get('background_filter', {})
    gaps = []
    if sky.get('rejected_ids'):
        gaps.append(dict(stage='Sky filter', reason='Rejected mask outlines were not retained in the source handoff.',
                         ids=sky['rejected_ids']))
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
                pixels=detail['pixels'], inherited=detail['inherited'], fits=list(detail['grids']),
                fitEvidence=detail.get('evidence'), maskSha256=info['mask_sha256'])

    loaded = set()
    for label, channel, file in PIPELINE:
        packet_path = root / channel / rel / file
        if not packet_path.is_file():
            gaps.append(dict(stage=label, reason='No chart packet saved for this view.'))
            continue
        charts, final = packet_grids(packet_path, source, receipts)
        provenance['packets'].append(dict(stage=label, path=packet_path.relative_to(root).as_posix(),
                                          sha256=hash_file(packet_path)))
        for chart in charts:
            status = [f'{name} · ' + ('selected' if chart['id'] in receipts[name]['selected_origins'] else 'outside selected assembly')
                      for name in final]
            add_grid(by_id[parent(chart['id'])], label, chart, status)
        if final and any(name not in loaded for name in final):
            walls, horizontal = horizontal_replay(charts, planes, image, final, receipts)
            for arm in set(final) - loaded:
                receipt = receipts[arm]
                export_support(by_id, walls + horizontal, arm, receipt)
                provenance['arms'][arm] = dict(scoreSha256=receipt['scoreHash'], packSha256=receipt['pack_sha256'],
                                               chartSha256=receipt['chart_sha256'],
                                               score=(receipt['folder'] / 'score.json').relative_to(root).as_posix(),
                                               depthVetoCount=receipt['depth_vetoes'])
            for chart in horizontal:
                status = [f'{name} · ' + ('selected' if chart['id'] in receipts[name]['selected_origins'] else 'outside selected assembly')
                          for name in final]
                add_grid(by_id[parent(chart['id'])], 'Assembly roof/underside (replay)', chart, status)
        loaded.update(final)
    if loaded != set(receipts):
        raise ValueError(f'Missing exact assembly chart packet: {case["id"]}')
    for field, kind in [('contracted', 'wall-contraction'), ('edge', 'edge-refit'),
                        ('pair', 'shared-corner-repair'), ('split', 'baseline-depth-split')]:
        for cid in row.get(field, []):
            if parent(cid) in by_id:
                by_id[parent(cid)]['decisions'].append(dict(arm='Baseline input', kind=kind, detail=cid))
    for cid, removed in row.get('border_dropped', {}).items():
        if parent(cid) in by_id:
            by_id[parent(cid)]['decisions'].append(dict(arm='Baseline input', kind='border-filter',
                                                         detail=dict(chart=cid, dropped=removed)))
    decision = row['audit'].get('decision', {})
    if not decision:
        gaps.append(dict(stage='Depth refit', reason='No refit decision recorded for this view.'))
    for option, detail in decision.get('arms', {}).items():
        for pair in detail['blocked_coplanar']:
            for pid in set(map(parent, pair)):
                by_id[pid]['decisions'].append(dict(arm=option, kind='sibling-coplanar-veto', detail=pair))
    for surface in by_id.values():
        if not surface['layers']:
            surface['decisions'].append(dict(arm='Source', kind='no-saved-grid',
                detail='Surface retained by the Finder. No nonempty projected grid was saved.'))
    write_evidence(out, marker, image.size, source['sourceSha'], by_id, provenance, gaps,
                   sourceChoices=row.get('source_choices'), rimCorner=decision.get('rim_corner'))


def source_surfaces(planes, counts, image):
    by_id = {}
    for plane in planes:
        marker_id = plane['id']
        by_id[marker_id] = dict(id=marker_id, mask=rings_path(plane['outline'], image.size),
                                bounds=plane.get('source_crop', {}).get('xyxy'),
                                pixels=plane['support_area'], geometry=plane.get('geometry_status'), layers=[],
                                parts={}, decisions=[], exports={}, depth=plane.get('relative_inverse_depth'))
    for count in counts['surfaces']:
        evidence = by_id[count['id']]
        quads = [cell['quad'] for cell in count.get('visible_cells', [])]
        evidence['count'] = dict(method=count['method'], blocks=count['dimensions']['blocks'], cellCount=len(quads))
        if quads:
            evidence['layers'].append(dict(label='Block Counter', stages=['Block Counter'], chart=count['id'],
                                           path=grid_path(quads), cellCount=len(quads), status=[]))

    return by_id


def write_evidence(out, marker, size, sha, surfaces, provenance, gaps, **details):
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dict(schema=2, id=marker, sourceSha256=sha, size=size, surfaces=list(surfaces.values()),
                provenance=provenance, gaps=gaps, **details)
    temp = out.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf8')
    temp.replace(out)


def export_support(surfaces, oriented, arm, receipt):
    """Recover the writer's first contributing source cell and verify every XYZ and ID."""
    from benchmark.mask_blocks import decoded_pack
    from benchmark.relative_face_assembly import axis_vector

    charts = {chart['id']: chart for chart in oriented}
    origins = receipt['selected_origins']
    owners = {}
    for surface in surfaces.values():
        surface['exports'][arm] = dict(cells=[], overlaps=[], selectedCharts=[])
    for chart_id in sorted(origins, key=lambda key: -charts[key].get('confidence', 0)):
        chart = charts[chart_id]
        export = surfaces[parent(chart_id)]['exports'][arm]
        export['selectedCharts'].append(chart_id)
        a, b = map(axis_vector, chart['axes'])
        offset = (np.asarray(origins[chart_id]) + np.minimum(0, a) + np.minimum(0, b)
                  - np.maximum(0, axis_vector(chart['normal_axis'])))
        allowed = set(map(tuple, chart['export_uv'])) if 'export_uv' in chart else None
        for cell in chart['cells']:
            uv = cell['u'], cell['v']
            if allowed is not None and uv not in allowed:
                continue
            xyz = tuple((offset + uv[0]*a + uv[1]*b).tolist())
            block = 'minecraft:' + cell['block_id'].removeprefix('minecraft:')
            item = dict(chart=chart_id, uv=uv, xyz=xyz, block=block)
            if xyz in owners:
                winner = owners[xyz]
                export['overlaps'].append(dict(item, winnerChart=winner['chart'], winnerBlock=winner['block']))
            else:
                quad = project(np.asarray(chart['H']), UNIT + uv)
                item.update(quad=quad.round(3).tolist(), fallback=bool(cell.get('fallback')))
                export['cells'].append(item)
                owners[xyz] = item
    if {p: item['block'] for p, item in owners.items()} != decoded_pack(receipt['folder'] / 'reconstruction.zip'):
        raise ValueError(f'Replayed source cells differ from the decoded {arm} pack.')
    for kind, records in [('join', receipt['joins']), ('conflict', receipt['conflicts'])]:
        for record in records:
            for pid in set(map(parent, record['chart_ids'])):
                surfaces[pid]['decisions'].append(dict(arm=arm, kind=kind, detail=record))


def build_v1_case(root, case, output=None):
    marker = case['id']
    row = read(root / 'rows' / f'{marker}.json.gz')
    src = Path(case['variants']['sky']['image'])
    if hash_file(src) != row['source_sha256']:
        raise ValueError(f'Source differs from V1 receipts: {src}')
    proof = dict(id=marker, source_sha256=row['source_sha256'], reference_sha256=row['reference_sha256'],
                 baseline=dict(pack_sha256=row['packs']['adopted']), candidate=dict(pack_sha256=row['packs']['nonshrinking']))
    with Image.open(src) as image:
        build(root, proof, image.convert('RGB'), (output or root / 'evidence') / marker / 'surfaces.json')


def build_expanded(folder, source, record, assembly):
    """Build portable evidence from the ordinary Expanded Split-Border receipts."""
    handoff = folder / 'source-handoff'
    planes_path, counts_path = handoff / 'surfaces.json', handoff / 'counts.json'
    planes = read(planes_path)
    with Image.open(source) as image:
        if tuple(planes['image_size']) != image.size:
            raise ValueError(f"Expanded surface evidence size differs: {record['id']}")
        surfaces = source_surfaces(planes['planes'], read(counts_path), image)
        size = image.size
    source_sha = record['source_sha256']
    proof = dict(sourceSha=source_sha, surfaceSha=hash_file(planes_path), counts=hash_file(counts_path))
    charts, _ = packet_grids(folder / 'charts.json', proof, {})
    for chart in charts:
        add_grid(surfaces[parent(chart['id'])], 'Split-border grid', chart)
    oriented = read(folder / 'oriented-charts.json')['charts']
    for chart in oriented:
        label = 'Horizontal grid' if chart['id'].startswith('horizontal-') else 'Oriented wall grid'
        status = ['Candidate · selected'] if chart['id'] in assembly['selected_origins'] else []
        add_grid(surfaces[parent(chart['id'])], label, chart, status)
    export_support(surfaces, oriented, 'Candidate', dict(assembly, folder=folder))
    provenance = dict(surfaceSha256=proof['surfaceSha'], countsSha256=proof['counts'],
                      handoffSha256=hash_file(handoff / 'build-manifest.json'), fields=record['field_sha256'],
                      arms={'Candidate': dict(packSha256=record['score']['pack_sha256'],
                                              chartSha256=record['charts_sha256'])})
    write_evidence(folder / 'surfaces.json', record['id'], size, source_sha, surfaces, provenance, [])


def build_local(handoff, folder, record, image, attempts=None):
    """Package the Combined runner's own candidate sheets and decoded local export."""
    from benchmark.mask_blocks import decoded_pack

    evidence, score = record['evidence'], record['score']
    manifest = read(handoff / 'build-manifest.json')
    planes = read(handoff / 'surfaces.json')
    if (manifest['image_sha256'] != evidence['source_sha256'] or
            score['source_sha256'] != evidence['source_sha256'] or
            tuple(planes['image_size']) != image.size):
        raise ValueError('Mismatched source or source dimensions in local-grid handoff.')
    surfaces = source_surfaces(planes['planes'], read(handoff / 'counts.json'), image)
    owners = {}
    pack = folder / 'reconstruction.zip'
    if hash_file(pack) != score['pack_sha256']:
        raise ValueError('Local-grid pack differs from score receipt.')
    for surface in surfaces.values():
        surface['exports']['Candidate'] = dict(cells=[], overlaps=[], selectedCharts=[])
    for anchor in evidence['anchors']:
        key = tuple(anchor['cell'])
        item = dict(xyz=key, chart=anchor['surface'], quad=anchor['quad'],
                    block='minecraft:' + anchor['block'].removeprefix('minecraft:'), correlation=anchor['corr'])
        owners[key] = item
    if {p: row['block'] for p, row in owners.items()} != decoded_pack(pack):
        raise ValueError('Local-grid source cells differ from decoded pack.')
    for row in owners.values():
        export = surfaces[row['chart']]['exports']['Candidate']
        export['cells'].append(row)
    seen = set()
    for index in reversed(range(len(evidence['anchors']))):
        anchor = evidence['anchors'][index]
        key = tuple(anchor['cell'])
        surface = surfaces[anchor['surface']]
        if key in seen:
            winner = owners[key]
            surface['exports']['Candidate']['overlaps'].append(dict(xyz=key, chart=surface['id'],
                block='minecraft:'+anchor['block'].removeprefix('minecraft:'), winnerChart=winner['chart'],
                winnerBlock=winner['block'], quad=anchor['quad']))
        seen.add(key)
    for pid, label, cells in attempts or []:
        if cells:
            surfaces[pid]['layers'].append(dict(label=label, stages=[label], chart=pid, cellCount=len(cells),
                                               path=grid_path([c['quad'] for c in cells]), status=[]))
    for pid, surface in surfaces.items():
        selected = [anchor for anchor in evidence['anchors'] if anchor['surface'] == pid]
        if selected:
            surface['exports']['Candidate']['selectedCharts'].append(pid)
            surface['layers'].append(dict(label='Selected local grid', stages=['Selected local grid'], chart=pid,
                                          cellCount=len(selected), path=grid_path([a['quad'] for a in selected]),
                                          status=['Candidate · selected']))
            surface['decisions'].append(dict(arm='Candidate', kind='selection', detail=evidence['reason']))
    gaps = [dict(stage='Relative joins', reason='Local-grid runner records a sheet or a consensus, without a join graph.')]
    if attempts is not None:
        gaps.append(dict(stage='Grid search', reason='Saved alternatives are verified grid or sprite witnesses; internal rejected search positions have no saved polygons.'))
    if attempts is None:
        gaps.append(dict(stage='Grid alternatives', reason='This archived case retained only the selected anchors.'))
    provenance = dict(surfaceSha256=hash_file(handoff/'surfaces.json'), countsSha256=hash_file(handoff/'counts.json'),
                      handoffSha256=hash_file(handoff/'build-manifest.json'), fields=evidence['field_sha256'],
                      arms=dict(Candidate=dict(packSha256=score['pack_sha256'])))
    write_evidence(folder/'surfaces.json', record['id'], image.size, evidence['source_sha256'], surfaces,
                   provenance, gaps, sourceChoices=dict(status=evidence['status'], reason=evidence['reason']))


def reuse_evidence(source, target, case, size):
    """Copy a runner package after binding its source and both exported arms to the review."""
    from shutil import copyfile
    evidence = read(source)
    expected = {arm: case[field]['pack_sha256'] for arm, field in
                [('Candidate', 'candidate'), ('Baseline', 'baseline')]}
    if (evidence.get('schema') != 2 or evidence['id'] != case['id'] or
            evidence['sourceSha256'] != case['source_sha256'] or tuple(evidence['size']) != size or
            any(evidence['provenance']['arms'].get(arm, {}).get('packSha256') != digest
                for arm, digest in expected.items())):
        raise ValueError(f'Surface evidence differs from review source or packs: {source}')
    if source.resolve() != target.resolve():
        copyfile(source, target)


def pair_evidence(baseline, candidate, target, case, size):
    """Compare independently saved local-grid packages; each arm retains its own mask IDs."""
    inputs = {}
    for arm, path in [('Baseline', baseline), ('Candidate', candidate)]:
        data = read(path)
        proof = data['provenance']['arms'].get('Candidate', {})
        if (data.get('schema') != 2 or data['id'] != case['id'] or data['sourceSha256'] != case['source_sha256']
                or tuple(data['size']) != size or proof.get('packSha256') != case[arm.lower()]['pack_sha256']):
            raise ValueError(f'Local-grid evidence disagrees with the {arm} source or pack: {path}')
        inputs[arm] = data
    surfaces, gaps, arms = {}, [], {}
    for arm, data in inputs.items():
        arms[arm] = dict(data['provenance']['arms']['Candidate'], surfaceSha256=data['provenance']['surfaceSha256'],
                         countsSha256=data['provenance']['countsSha256'])
        gaps.extend(dict(g, arm=arm) for g in data['gaps'])
        for surface in data['surfaces']:
            # Masks with the same ID may describe different polygons after a Finder change.
            # Keep whole evidence per arm to preserve even different count and mask handoffs.
            key = f"{arm} / {surface['id']}"
            surfaces[key] = dict(surface, id=key, sourceSurface=surface['id'], arm=arm,
                exports={arm: surface['exports']['Candidate']},
                decisions=[dict(d, arm=arm) for d in surface['decisions']],
                layers=[dict(layer, status=[status.replace('Candidate ·', arm+' ·') for status in layer['status']])
                        for layer in surface['layers']])
    write_evidence(target, case['id'], size, case['source_sha256'], surfaces,
                   dict(arms=arms, packages={a: hash_file(p) for a,p in [('Baseline', baseline), ('Candidate', candidate)]}),
                   gaps)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate portable surface evidence from saved V1 rows and source PNGs.')
    parser.add_argument('run_root', type=Path)
    parser.add_argument('frames', type=Path, help='Directory containing <split>/frames/<id>/rgb.png')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', action='append', help='Exact saved case ID (repeatable); defaults to every saved row.')
    args = parser.parse_args()
    lookup = {p.name.removesuffix('.json.gz'): p for p in (args.run_root/'rows').glob('*.json.gz')}
    selected = args.case or sorted(lookup)
    if not selected or set(selected) - lookup.keys():
        parser.error('No matching saved V1 rows.')
    for marker in selected:
        sources = list(args.frames.glob(f'*/frames/{marker}/rgb.png'))
        if len(sources) != 1:
            parser.error(f'Expected one frozen source for {marker}, found {len(sources)}.')
        build_v1_case(args.run_root, dict(id=marker, variants=dict(sky=dict(image=sources[0]))), args.out)
    print(f'Pack-verified surface evidence for {len(selected)} views in {args.out}')


if __name__ == '__main__':
    main()
