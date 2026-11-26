"""Frozen geometry-to-pack experiment. Reference data enters only after prediction.

Ordinary surface output stops at camera-space charts. This adapter tests whether
changing their supported footprints changes the visible blocks the exporter writes.
"""
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark.camera_metrics import registration, truth_camera
from benchmark.normal_sheet import quad_coverage
from benchmark.surface_metrics import DOWNSAMPLE, observed_support
from engine.camera_grid import predict as predict_camera
from engine.export import JAVA_26_3_FORMAT, write_pack
from engine.nbt import read_nbt
from engine.visible_grid import SETTINGS as GRID_SETTINGS, CellMap, _blocks, _carriers, _carrier, _cell_map, _plane_from_chart


# Read from the suite's pinned 26.3 client via version.json and catalog.template_format.
PACK = JAVA_26_3_FORMAT
SETTINGS = dict(camera='seam-fit', pitch='repeat', lift='shared-edges-v1',
                **GRID_SETTINGS, required_footprint_coverage=1., sample_stride_px=DOWNSAMPLE,
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

    Each method sends its own sampling charts through the same texture count and lift.
    A second selected-method control holds B's original charts and counts fixed.
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
    baseline_carriers = _carriers(image, baseline_parts, methods['baseline'], P)
    selected_carriers = (_carriers(image, selected_parts, methods['selected'], P)
                         if selected_uses_masks else baseline_carriers)
    cache, reports, decoded = {}, {}, {}
    for name, footprints, carriers in [('baseline',methods['baseline'],baseline_carriers),
                                       ('selected',methods['selected'],selected_carriers),
                                       ('selected_with_B_charts',methods['selected'],baseline_carriers)]:
        blocks, counts, details = _blocks(footprints, carriers, P, cache)
        filename = name+'-visible.zip'
        decoded[name] = _write_read(out/filename, blocks)
        reports[name] = dict(regions=counts, details=details, pack=filename)

    # Scoring starts here, after both image-only structure packs were written.
    pose, truth = recorded(case)
    labels = json.loads(case.path('labels').read_text(encoding='utf8'))['faces']
    target = {tuple(f['xyz']) for f in labels}
    T = registration(P, pose) if P is not None else np.eye(4)
    reports['selected_with_B_charts']['score'] = _score(registered(decoded['selected_with_B_charts'], T), target)
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
