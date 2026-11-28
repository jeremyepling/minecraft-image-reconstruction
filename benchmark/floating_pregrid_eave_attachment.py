"""Bound attachment headroom for Stable Yard's independently fitted visible eave."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.floating_expanded_wall_attachment import (
    PREFIX, ROOT, WORKTREES, baseline, combined, compare, data, digest, permitted, read, save, workload,
)
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import axis_vector, oriented
from benchmark.relative_face_scoring import _yaw
from engine.geometry import project


CASE = 'train-stable-yard-1-warm-high-right'
PART = 'mask-0~p24'
OUT = ROOT / 'runs/pregrid-eave-attachment'
REVIEW = ROOT / 'benchmark/reviews/floating-pregrid-eave-attachment-01'


def cell_sites(chart):
    u, v = (axis_vector(i) for i in chart['axes'])
    normal = axis_vector(chart['normal_axis'])
    corner = np.minimum(0, u) + np.minimum(0, v) - np.maximum(0, normal)
    uv = np.array([(cell['u'], cell['v']) for cell in chart['cells']])
    xyz = corner + uv @ np.stack([u, v])
    xy = np.floor(project(np.asarray(chart['H']), uv + .5)).astype(int)
    return xyz, xy, normal


def archive_eave(panel):
    root = WORKTREES / 'floating-depth-before-atlas/runs/depth-before-atlas' / panel
    packet = root / 'complete/charts/sky' / CASE / 'charts.json'
    bitmap = root / 'source' / CASE / 'mask-0-ids.png'
    meta = read(bitmap.with_name('mask-0.json'))
    assert digest(bitmap) == meta['ids_sha256']
    eave = next(c for c in read(packet)['charts'] if c['id'] == PART)
    return eave, (np.asarray(Image.open(bitmap)) == 24), meta, digest(packet)


def rigid_attachment(chart, pose, owners, faces):
    local, xy, normal = cell_sites(chart)
    yaw = pose['yaw_quarters']
    orientation = tuple(_yaw(normal, yaw))
    truth_faces = [faces[int(owners[y, x])] for x, y in xy]
    assert all(tuple(f['normal_world']) == orientation for f in truth_faces)
    eave_points = {tuple(f['xyz_world']) for f in truth_faces}
    assert len(eave_points) == len(local) == 21
    offsets = Counter(tuple(np.asarray(f['xyz_world']) - _yaw(xyz, yaw) - pose['translation'])
                      for xyz, f in zip(local, truth_faces, strict=True))
    # One rigid position for all cells. The reference supplies no cell, grid or material.
    assert len(offsets) == 1
    world = next(iter(offsets))
    shift = np.asarray(_yaw(world, (-yaw) % 4))
    blocks = {tuple((xyz + shift).tolist()): c['block_id'] for xyz, c in zip(local, chart['cells'], strict=True)}
    assert all(not c.get('fallback') for c in chart['cells'])
    return blocks, eave_points, shift.tolist()


def superseded(charts, origins, region):
    sites = set()
    for c in charts:
        if c['id'] not in origins or not c['id'].startswith('mask-0~'):
            continue
        selected = permitted(c)
        xyz, xy, _ = cell_sites(selected)
        for (x, y), point in zip(xy, xyz, strict=True):
            if 0 <= y < region.shape[0] and 0 <= x < region.shape[1] and region[y, x]:
                sites.add(tuple(point + origins[c['id']]))
    return sites


def export_case(panel, case, baseline_path, origins, name, additions, removed):
    assembler = combined.assembly.assemble

    def intervene(charts, **kwargs):
        result = assembler(charts, **kwargs)
        group = result['groups'][0]
        assert group['origins'] == origins
        for pos in removed:
            group['blocks'].pop(pos, None)
            group['assigned'].discard(pos)
        for pos, material in additions.items():
            if pos not in group['blocks']:
                group['blocks'][pos] = material
                group['assigned'].add(pos)
        return result

    output = OUT / name
    with patch.object(combined, 'inputs', data), patch.object(combined.assembly, 'assemble', intervene):
        combined.run((panel, case, baseline_path / 'charts', output))
    folder = output / PREFIX / CASE
    coords = read(folder / 'placements.json')
    score = read(folder / 'score.json')
    pack = folder / 'reconstruction.zip'
    local = decoded_pack(pack)
    assert local == {tuple(xyz): block for xyz, block in coords['local']}
    return score, local, {tuple(xyz): material for xyz, material in coords['aligned']}


def main():
    panel, case, prior, _ = next(row for row in workload() if row[1]['id'] == CASE)
    path, incumbent = baseline(panel, CASE, prior)
    filename = path / 'charts/sky' / CASE / 'charts.json'
    packet = read(filename)
    eave, region, metadata, eave_packet_sha = archive_eave(panel)
    image = Path(case['variants']['sky']['image'])
    assert digest(image) == packet['source_sha256'] == incumbent['source_sha256'] == metadata['source_sha256']
    named = dict(eave, id='independently-counted-eave')
    charts, _ = oriented(packet['charts'] + [named])
    child = next(c for c in charts if c['id'] == named['id'])
    frame = Path(case['frame'])
    assert all(digest(frame / k) == v for k, v in incumbent['reference_sha256'].items())
    labels = read(frame / 'labels.json')
    with np.load(frame / 'labels.npz') as channels:
        owners = channels['face_id']
    faces = {f['id']: f for f in labels['faces']}
    truth = {tuple(b['xyz_world']): b['block_id'] for b in labels['blocks']}
    original = incumbent['score']['fixed']
    origins = incumbent['selected_origins']
    additions, target, shift = rigid_attachment(child, original, owners, faces)
    legacy = superseded(charts, origins, region)

    parent = next(c for c in charts if c['id'] == PART)
    flat_shift = np.array(shift)
    normal = axis_vector(parent['normal_axis'])
    dimension = int(np.flatnonzero(normal)[0])
    flat_shift[dimension] = origins[PART][dimension]
    planar_cells, _, _ = cell_sites(child)
    flat = {tuple((xyz + flat_shift).tolist()): cell['block_id']
            for xyz, cell in zip(planar_cells, child['cells'], strict=True)}

    saved = read(path / 'assembly' / PREFIX / CASE / 'placements.json')
    before = {tuple(xyz): material for xyz, material in saved['aligned']}
    arms = {}
    for label, added, removed in (('detached', {}, set()), ('same-plane', flat, set()),
                                  ('reference-attach', additions, set()), ('reference-replace', additions, legacy)):
        score, local, aligned = export_case(panel, case, path, origins, label, added, removed)
        fixed = score['score']['fixed']
        prior = {tuple(a + b for a, b in zip(_yaw(xyz, original['yaw_quarters']), original['translation'])): material
                 for xyz, material in local.items()}
        if label == 'detached':
            assert original == fixed and aligned == before
        arms[label] = dict(score=fixed, paired=compare(before, aligned, truth),
                           eave_correct=len(target & aligned.keys()),
                           eave_exact=sum(aligned.get(pos) == truth[pos] for pos in target),
                           incumbent_registration=dict(paired=compare(before, prior, truth),
                                                       eave_correct=len(target & prior.keys())),
                           pack_sha256=score['pack_sha256'])
    best = max(arms, key=lambda label: arms[label]['score']['geometry']['strict']['f1'])
    report = dict(id=CASE, chart=PART, source_sha256=digest(image),
                  benchmark_chart_sha256=digest(filename), reference_sha256=incumbent['reference_sha256'],
                  eave_packet_sha256=eave_packet_sha, eave_mask_sha256=metadata['ids_sha256'],
                  eave_pixels=int(region.sum()), legacy_candidates=len(legacy), translation_local=shift,
                  parent_plane_local=origins[PART][dimension], child_plane_local=shift[dimension],
                  normal=normal.tolist(), arms=arms, chooser=best)
    save(REVIEW / 'results.json', report)
    rgb = np.asarray(Image.open(image).convert('RGB')).copy()
    visible = np.isin(owners, [f['id'] for f in faces.values() if tuple(f['xyz_world']) in target])
    rgb[region] = .45 * rgb[region] + .55 * np.array([243, 75, 173])
    rgb[visible & region] = .35 * rgb[visible & region] + .65 * np.array([80, 245, 167])
    Image.fromarray(rgb).save(REVIEW / 'verified-eave.png')
    for label, result in arms.items():
        print(label, result['score']['geometry']['strict'], result['paired'], result['incumbent_registration'])


if __name__ == '__main__':
    main()
