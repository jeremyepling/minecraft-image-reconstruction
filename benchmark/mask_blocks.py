"""Frozen geometry-to-pack experiment. Reference data enters only after prediction.

Ordinary surface output stops at camera-space charts. This adapter tests whether
changing their supported footprints changes the visible blocks the exporter writes.
"""
from collections import Counter
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark.camera_metrics import registration, truth_camera
from benchmark.normal_sheet import quad_coverage
from benchmark.surface_metrics import DOWNSAMPLE, observed_support
from engine.camera_grid import predict as predict_camera
from engine.export import write_pack
from engine.nbt import read_nbt
from engine.surface_model import assemble, intrinsic, pitch


# Read from the suite's pinned 26.3 client via version.json and catalog.template_format.
PACK = dict(data_version=5023, palette_key='id', pack_version=dict(data_major=121, data_minor=0))
SETTINGS = dict(camera='seam-fit', pitch='repeat', lift='shared-edges-v1',
                minimum_carrier_iou=.5, maximum_plane_corner_residual_blocks=.5,
                minimum_face_samples=4, required_footprint_coverage=1., sample_stride_px=DOWNSAMPLE,
                material='smooth_stone', pack=PACK)


def _footprints(parts, size, masks):
    shape = tuple((n+DOWNSAMPLE-1)//DOWNSAMPLE for n in size[::-1])
    footprints = []
    for part in parts:
        if masks and 'support' in part:
            footprints.append(observed_support(part['support'], shape, size))
        elif not masks and part.get('quad') is not None:
            footprints.append(quad_coverage([part], shape))
        else:
            footprints.append(np.zeros(shape, bool))
    return footprints


def _plane_from_chart(row, extrinsics):
    if row.get('units') != 'blocks':
        return None, 'unresolved_scale'
    rotation = extrinsics[:, :3]
    corners = np.linalg.solve(rotation, (np.asarray(row['corners_camera']) - extrinsics[:, 3]).T).T
    normal = rotation.T @ np.asarray(row['normal_camera'])
    axis = int(np.argmax(abs(normal)))
    level = int(np.rint(np.median(corners[:, axis])))
    if abs(corners[:, axis]-level).max() > SETTINGS['maximum_plane_corner_residual_blocks']:
        return None, 'unresolved_depth'
    return (axis, level, int(np.sign(normal[axis]))), 'supported'


def _carriers(image, parts, masks, P):
    if P is None:
        return []
    scale = np.linalg.norm(P[2, :3])
    focal = np.linalg.norm(P[0, :3]-image.width/2*P[2, :3]) / scale
    K = intrinsic(image.size, focal)
    extrinsics = np.linalg.solve(K, P/scale)
    charts, pitches, supports = [], [], []
    for part, mask in zip(parts, masks, strict=True):
        if part.get('quad') is None or not mask.any():
            continue
        charts.append(dict(id=part['id'], quad=part['quad']))
        pitches.append(part.get('pitch') or pitch(image, part['quad'], '26.3', 'repeat'))
        supports.append(mask)
    lifted = assemble(charts, pitches, image.size, K)['planes']
    return [(mask, *_plane_from_chart(row, extrinsics)) for mask, row in zip(supports, lifted, strict=True)]


def _carrier(mask, carriers):
    best, selected, reason = 0., None, 'unresolved_chart'
    for prior, plane, status in carriers:
        intersection = np.count_nonzero(mask & prior)
        if not intersection:
            continue
        iou = intersection / np.count_nonzero(mask | prior)
        if iou >= SETTINGS['minimum_carrier_iou'] and iou > best:
            best, selected, reason = iou, plane, status
    return selected, reason


@dataclass
class CellMap:
    axis: int
    level: int
    front: int
    valid_pixels: np.ndarray
    cells: np.ndarray
    inverse: np.ndarray
    counts: np.ndarray

    def occupied(self, mask):
        covered = np.bincount(self.inverse, weights=mask.ravel()[self.valid_pixels], minlength=len(self.cells))
        complete = self.cells[(covered == self.counts) & (self.counts >= SETTINGS['minimum_face_samples'])]
        xyz = np.empty((len(complete), 3), dtype=int)
        xyz[:, self.axis] = self.level - int(self.front > 0)
        xyz[:, [i for i in range(3) if i != self.axis]] = complete
        return set(map(tuple, xyz.tolist()))


def _cell_map(P, shape, plane):
    """Assign every visible pixel center a unit square. A hole vetoes the whole block.

    Grouping all viewport pixels also handles cells clipped by the image boundary.
    Pixels past a plane's horizon cannot vote for cells behind the observer.
    """
    axis, level, front = plane
    free = [i for i in range(3) if i != axis]
    H = np.column_stack((P[:, free], P[:, 3] + level*P[:, axis]))
    y, x = np.indices(shape, dtype=float)
    rays = np.stack(((x.ravel()+.5)*DOWNSAMPLE, (y.ravel()+.5)*DOWNSAMPLE, np.ones(x.size)))
    uvw = np.linalg.solve(H, rays)
    valid = np.flatnonzero(uvw[2] > 1e-8)
    uv = np.floor((uvw[:2, valid]/uvw[2, valid]).T).astype(np.int64)
    unique, inverse, counts = np.unique(uv, axis=0, return_inverse=True, return_counts=True)
    return CellMap(axis, level, front, valid, unique, inverse.ravel(), counts)


def _blocks(masks, carriers, P, cache):
    blocks, resolved = set(), Counter()
    details = []
    for index, mask in enumerate(masks):
        plane, status = _carrier(mask, carriers) if P is not None else (None, 'unresolved_camera')
        cells = set()
        if plane is not None:
            if plane not in cache:
                try:
                    cache[plane] = _cell_map(P, mask.shape, plane)
                except np.linalg.LinAlgError:
                    cache[plane] = None
            if cache[plane] is not None:
                cells = cache[plane].occupied(mask)
            status = 'emitted' if cells else 'no_fully_supported_cell'
        blocks.update(cells)
        resolved[status] += 1
        details.append(dict(index=index, status=status, plane=plane, cells=len(cells)))
    return blocks, dict(resolved), details


def decoded_pack(path):
    """Read actual template placement commands, tile origins, block indices and states."""
    result = {}
    with ZipFile(path) as bundle:
        script = bundle.read('data/reconstruction/function/build.mcfunction').decode().splitlines()
        for line in filter(None, script):
            place, template, target, *origin = line.split()
            if (place, template) != ('place', 'template') or len(origin) != 3:
                raise ValueError(f'Invalid placement command {line}')
            namespace, part = target.split(':')
            offset = [int(n.removeprefix('~') or '0') for n in origin]
            name = f'data/{namespace}/structure/{part}.nbt'
            data = read_nbt(gzip.decompress(bundle.read(name)))
            if data['DataVersion'] != PACK['data_version']:
                raise ValueError('Wrong structure version')
            for block in data['blocks']:
                pos = tuple(v+b for v, b in zip(offset, block['pos'], strict=True))
                if pos in result:
                    raise ValueError(f'Overlapping structure tile at {pos}')
                result[pos] = data['palette'][block['state']][PACK['palette_key']]
    return result


def _write_read(path, blocks):
    material = SETTINGS['material']
    write_pack(path, {p: material for p in blocks}, PACK)
    read = decoded_pack(path)
    if read != {p: 'minecraft:'+material for p in blocks}:
        raise ValueError('Decoded structure differs from the proposed coordinates/states')
    return set(read)


def registered(cells, T):
    if not cells:
        return set()
    # Reflection exchanges the lower and upper block corner. Register centers.
    centers = np.asarray(sorted(cells), float) + .5
    xyz = np.floor(centers @ T[:3, :3].T + T[:3, 3]).astype(int)
    return set(map(tuple, xyz.tolist()))


def _score(read, target):
    return dict(serialized=len(read), visible_truth=len(target), correct_visible=len(read & target),
                extra=len(read-target), missed=len(target-read))


def evaluate_case(case, image, baseline_parts, selected_parts, output_dir, *, selected_uses_masks=True):
    """Score a frame offline. ``output_dir`` must be private to that frame.

    Metric predicted depth comes exclusively from the same B charts/repeat counts.
    The recorded-camera/plane diagnostic conditions on a >=.5-IoU association: it
    cannot measure the ability to choose a depth or reject an unmatched proposal.
    """
    from benchmark.surface_reference import recorded

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    camera = predict_camera(image, '26.3', 'seam-fit')
    P = None if camera['P'] is None else np.asarray(camera['P'])
    methods = dict(baseline=_footprints(baseline_parts, image.size, False),
                   selected=_footprints(selected_parts, image.size, selected_uses_masks))
    carriers = _carriers(image, baseline_parts, methods['baseline'], P)
    cache, reports, decoded = {}, {}, {}
    for name, footprints in methods.items():
        blocks, counts, details = _blocks(footprints, carriers, P, cache)
        filename = name+'-visible.zip'
        decoded[name] = _write_read(out/filename, blocks)
        reports[name] = dict(regions=counts, details=details, pack=filename)

    # Scoring starts here, after both image-only structure packs were written.
    pose, truth = recorded(case)
    labels = json.loads(case.path('labels').read_text(encoding='utf8'))['faces']
    target = {tuple(f['xyz']) for f in labels}
    T = registration(P, pose) if P is not None else np.eye(4)
    reference_P = truth_camera(pose)
    diagnostic = [(truth['id_map']==pid, (p['axis'], p['offset'], p['normal'][p['axis']]), 'supported')
                  for pid, p in truth['patches'].items() if p['support_px']]
    cache = {}
    for name, footprints in methods.items():
        reports[name]['score'] = _score(registered(decoded[name], T), target)
        cells, counts, _ = _blocks(footprints, diagnostic, reference_P, cache)
        filename = name+'-known-plane-diagnostic.zip'
        known_read = _write_read(out/filename, cells)
        reports[name]['known_camera_and_plane_diagnostic'] = dict(regions=counts, pack=filename,
                                                                 score=_score(known_read, target))
    result = dict(image=case.id, floor=case.floor, layout=case.frame['layout'], camera_status=camera['scale_status'],
                  P=camera['P'], in_game_placement_verified=False, **reports)
    (out/'blocks.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result
