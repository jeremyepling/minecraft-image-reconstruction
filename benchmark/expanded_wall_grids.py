"""Replay complete independently rebased full-cube wall grids on frozen original test views."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from random import Random
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay, relative_face_round
from benchmark.camera_labels import screen_matrix
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from benchmark.expanded_crop_ceiling import DATASETS
from benchmark.full_block_lattice import atlas
from benchmark.relative_face_assembly import axis_vector, oriented
from benchmark.relative_face_scoring import _yaw, yaw_from_anchor_normal
from benchmark.relative_oracle_charts import UNIT, fit_homography
from engine.atlas_count import rank_tile
from engine.geometry import project
from engine.surface_perimeter import _native_alpha


NATIVE = WORKTREES / 'crop-ceiling-expanded/runs/native-crop-ceiling/native'
OUT = ROOT / 'runs/wall-grids'
ARMS = ('source-normal', 'signed')
TEXELS = np.stack(np.meshgrid((np.arange(16) + .5) / 16, (np.arange(16) + .5) / 16), axis=-1)


def axis_code(vector):
    i = int(np.argmax(abs(vector)))
    return (i + 1) * int(vector[i])


def material(rgb, owners, fid, pixels):
    # Nearest native pixels avoid mixing a neighboring face into a partial tile.
    h = fit_homography((0, 0), pixels)
    xy = np.floor(project(h, TEXELS)).astype(int)
    x, y = xy[..., 0], xy[..., 1]
    ih, iw = owners.shape
    inside = (x >= 0) & (x < iw) & (y >= 0) & (y < ih)
    x = x.clip(0, iw - 1)
    y = y.clip(0, ih - 1)
    result = rank_tile(dict(rgb=rgb[y, x], mask=inside & (owners[y, x] == fid)), atlas())
    if not result['block_id']:
        return dict(block_id='smooth_stone', confidence=0., fallback=True)
    return dict(block_id=result['block_id'], confidence=max(0., result['evidence']['correlation']), fallback=False)


def charts_for_frame(case, packet, handoff):
    frame = Path(case['frame'])
    labels = read(frame / 'labels.json')
    camera = read(frame / 'camera.json')
    matrix = screen_matrix(camera)
    eye = np.array(camera['eye_world'])
    rgb = np.asarray(Image.open(frame / 'rgb.png').convert('RGB'))
    with np.load(frame / 'labels.npz') as data:
        owners = data['face_id']
    counts = np.bincount(owners.ravel().astype(int))
    lookup = np.zeros(len(counts), dtype=int)
    kinds = {b['id']: b.get('geometry_kind', 'full_cube') for b in labels['blocks']}
    walls = defaultdict(list)
    for face in labels['faces']:
        fid = face['id']
        assert counts[fid] == face['visible_pixels'] > 0
        lookup[fid] = face['surface_id']
        if kinds[face['block']] == 'full_cube' and face['normal_world'][1] == 0:
            walls[face['surface_id']].append(face)
    owner_counts = np.bincount(lookup, weights=counts)
    planes = {p['id']: p for p in read(handoff / 'surfaces.json')['planes']}
    assert len(planes) == len(set(lookup) - {0})
    for pid, plane in planes.items():
        sid = int(pid.removeprefix('plane-'))
        alpha, (x, y, x1, y1) = _native_alpha(plane['support'], camera['image_size'])
        mask = np.asarray(alpha) > 0
        assert mask.sum() == plane['support_area'] == owner_counts[sid]
        assert np.all(lookup[owners[y:y1, x:x1][mask]] == sid)

    rng = Random('expanded-wall-grids/' + case['id'])
    details = Counter()
    reproject_error = chart_error = 0.
    charts, exact_axes = [], {}
    for sid, faces in walls.items():
        pid = f'plane-{sid}'
        normal = np.asarray(faces[0]['normal_world'], int)
        up = np.array([0, 1, 0])
        right = np.cross(up, normal)
        assert np.array_equal(np.cross(right, up), normal)
        basis = np.stack((right, up))
        data = []
        for face in faces:
            assert np.array_equal(face['normal_world'], normal)
            xyz = np.asarray(face['corners_world'])
            uv = (xyz @ basis.T).astype(int)
            low = uv.min(0)
            relative = uv - low
            vertices = dict(zip(map(tuple, relative), face['corners_px']))
            assert set(vertices) == set(map(tuple, UNIT))
            px = np.asarray([vertices[tuple(p)] for p in UNIT])
            assert (px[1, 0] + px[2, 0]) > (px[0, 0] + px[3, 0])
            assert (px[2, 1] + px[3, 1]) < (px[0, 1] + px[1, 1])
            projected = (xyz - eye) @ matrix[:, :3].T + matrix[:, 3]
            err = np.linalg.norm(projected[:, :2] / projected[:, 2:] - face['corners_px'], axis=1)
            reproject_error = max(reproject_error, float(err.max()))
            data.append((face, low, px))
        shift = np.min([low for _, low, _ in data], axis=0) - [rng.randrange(-12, 13), rng.randrange(-12, 13)]
        _, anchor, corners = max(data, key=lambda item: item[0]['visible_pixels'])
        h = fit_homography(anchor - shift, corners)
        cells = []
        for face, low, pixels in data:
            u, v = low - shift
            err = np.linalg.norm(project(h, UNIT + (u, v)) - pixels, axis=1)
            chart_error = max(chart_error, float(err.max()))
            match = material(rgb, owners, face['id'], pixels)
            cells.append(dict(u=int(u), v=int(v), **match))
        assert len(set((c['u'], c['v']) for c in cells)) == len(cells)
        rng.shuffle(cells)
        charts.append(dict(id=pid, H=h.tolist(), normal_camera=planes[pid]['normal_camera'],
                           cells=cells, confidence=float(np.mean([c['confidence'] for c in cells])),
                           method='independently-rebased-visible-wall-grid'))
        exact_axes[pid] = dict(axes=[axis_code(right), 2], normal_axis=axis_code(normal))
    assert max(reproject_error, chart_error) < 1e-4
    rng.shuffle(charts)
    retained = [c for c in packet['charts'] if c['id'] not in exact_axes]
    count_cells = sum(len(c['cells']) for c in charts)
    details.update(wall_charts=len(charts), wall_cells=count_cells, retained_other_charts=len(retained),
                   retained_other_cells=sum(len(c['cells']) for c in retained), verified_masks=len(planes),
                   native_mask_pixels=int(counts[1:].sum()))
    return charts + retained, exact_axes, dict(details, max_chart_corner_px=chart_error,
                                               max_camera_corner_px=reproject_error)


def signed_packet(charts, truth, camera):
    measured, hint = oriented(charts)
    by_id = {c['id']: c for c in measured}
    # Express any retained source-oriented decorative charts in the signed arm's
    # global compass. This rotation never positions a chart or assigns it depth.
    yaw = yaw_from_anchor_normal(hint['normal_camera'], axis_vector(hint['local_normal_axis']), camera) if hint else None
    quarters = yaw['yaw_quarters'] if yaw else None
    def rotate(code):
        return axis_code(np.array(_yaw(axis_vector(code), quarters)))
    result = []
    audit = Counter()
    for chart in charts:
        pid = chart['id']
        if pid in truth:
            result.append(dict(chart, **truth[pid]))
            if pid not in by_id:
                audit['source_dropped_walls'] += 1
            elif quarters is None:
                audit['source_unregistered_walls'] += 1
            else:
                source = by_id[pid]
                correct = [rotate(c) for c in source['axes']] == truth[pid]['axes'] and rotate(source['normal_axis']) == truth[pid]['normal_axis']
                audit['source_correct_axes' if correct else 'source_wrong_axes'] += 1
        elif pid in by_id:
            if quarters is None:
                raise ValueError('Retained source-oriented nonwall chart lacks a fixed compass')
            source = by_id[pid]
            result.append(dict(chart, axes=[rotate(c) for c in source['axes']], normal_axis=rotate(source['normal_axis'])))
        else:
            audit['source_dropped_nonwall'] += 1
    return result, dict(audit)


def evaluate(case, tree, root, *, signed, edges=1, edge_px=2.5, coplanar_px=.5):
    frame = Path(case['frame'])
    name = case['id']
    with ExitStack() as stack:
        stack.enter_context(patch.object(floating_cloud_replay, 'RUN', NATIVE))
        mode = (tree, 'charts.json', signed, edges, edge_px, coplanar_px, False)
        stack.enter_context(patch.multiple(relative_face_round, ASSEMBLY=root / 'assembly',
            MODES=dict(relative_face_round.MODES, **{'connected-atlas': mode})))
        if case['dataset'] == 'visible-builds-v2':
            proof = next(x for x in read(ROOT / 'training/framing-audit-v2.json')['frames'] if x['id'] == name)
            assert proof['accepted'] and proof['rgb_sha256'] == case['rgb_sha256']
            def frame_read(path):
                data = read(path)
                return dict(data, framing=proof) if path == frame / 'frame.json' else data
            stack.enter_context(patch.object(expanded_scores, 'read', frame_read))
        relative_face_round.evaluate((case, 'sky', 'connected-atlas'))


def run(case):
    frame = Path(case['frame'])
    name = case['id']
    source = frame / 'rgb.png'
    sha = digest(source)
    assert sha == case['rgb_sha256']
    metadata = read(frame / 'frame.json')
    assert all(digest(frame / k) == metadata['files'][k] for k in ('rgb.png', 'labels.npz', 'labels.json', 'camera.json'))
    original = NATIVE / 'atlas/connected/sky' / name / 'charts.json'
    handoff = NATIVE / 'sky' / name / 'source-handoff'
    packet = read(original)
    floating_cloud_replay.check_handoff(packet, handoff, sha)
    assert atlas().provenance['atlas_sha256'] == packet['atlas_sha256']
    charts, axes, diagnostics = charts_for_frame(case, packet, handoff)
    signed, audit = signed_packet(charts, axes, read(frame / 'camera.json'))
    save(OUT / 'checks' / (name + '.json'), dict(id=name, dataset=case['dataset'], **diagnostics, **audit,
         source_sha256=sha, native_packet_sha256=digest(original), reference_sha256={k:metadata['files'][k] for k in ('labels.json','labels.npz','camera.json')}))
    case = dict(case, variants={'sky': dict(image=str(source), source_sha256=sha)})
    for arm, inputs in zip(ARMS, (charts, signed)):
        root = OUT / arm
        tree = root / 'charts'
        save(tree / 'sky' / name / 'charts.json', dict(packet, charts=inputs))
        evaluate(case, tree, root, signed=arm == 'signed')
    print('Verified', name, diagnostics['wall_cells'], flush=True)


def cases():
    for dataset in DATASETS:
        root = SOURCE / dataset
        for row in read(root / 'manifest.json')['frames']:
            if row['split'] == 'test':
                yield dict(row, dataset=dataset, frame=str(root / 'data/test/frames' / row['id']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--workers', type=int, default=4)
    cli.add_argument('--id', nargs='*')
    cfg = cli.parse_args()
    selected = [c for c in cases() if not cfg.id or c['id'] in cfg.id]
    assert len(selected) == (len(cfg.id) if cfg.id else 38)
    with ProcessPoolExecutor(cfg.workers) as pool:
        list(pool.map(run, selected))
