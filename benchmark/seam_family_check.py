"""Run frozen block seam estimators on four actual Surface Finder outputs from Twin Spire."""
import argparse
from collections import Counter
from importlib import import_module
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.block_seam_round import MODELS
from benchmark.camera_metrics import registration
from benchmark.camera_round import ROOT, digest, read, save
from benchmark.count_materials import true_faces
from benchmark.count_metrics import reference_planes
from benchmark.grid_placement_check import grade_shapes, tile_shapes
from benchmark.mask_blocks import SETTINGS, _blocks, _carriers, _footprints, _score, _write_read, registered
from benchmark.mask_metrics import measure
from benchmark.surface_metrics import truth
from calibration.world import GROUND
from engine.camera_grid import predict as predict_camera
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only


FRAMES = tuple('test-twinspire1-'+palette+'-'+angle
               for palette in ('cool', 'earth') for angle in ('high-right', 'low-left'))
RELEASE = ROOT/'training/visible-builds-v1.json'


def read_grid(image, parts, samples, methods):
    rows = []
    for part, sample in zip(parts, samples, strict=True):
        baseline = count(sample)
        quad = sample['quad'] if sample else None
        readings = {}
        for name, method in methods.items():
            reading = baseline if method is None else method.predict(sample, baseline)
            counts = reading['counts']
            readings[name] = dict(counts=counts, phase=reading['phase'],
                                  blocks=dimensions(part, quad, image.size, counts)['blocks'])
        rows.append(dict(id=part['id'], quad=quad, readings=readings))
    return rows


def export_coordinates(image, parts, rows, methods, folder):
    """Decode all geometry-only packs while the answer files are still unopened."""
    camera = predict_camera(image, '26.3', SETTINGS['camera'])
    projection = None if camera['P'] is None else np.asarray(camera['P'])
    masks = _footprints(parts, image.size, True)
    folder.mkdir(parents=True)
    decoded, cache = {}, {}
    for name in methods:
        charts = []
        for row in rows:
            counts = row['readings'][name]['counts']
            charts.append(dict(id=row['id'], quad=row['quad'], pitch={'counts': counts if None not in counts else None}))
        carriers = _carriers(image, charts, masks, projection)
        blocks, _, _ = _blocks(masks, carriers, projection, cache)
        decoded[name] = _write_read(folder/(name+'.zip'), blocks)
    return camera, decoded


def evaluate(archive, frame, image_sha, surfaces, methods, out):
    prefix = 'frames/'+frame+'/'
    source_dir = surfaces/frame
    build = read(source_dir/'build-manifest.json')
    if (build['image_sha256'] != image_sha or build['version'] != '26.3' or build['through'] != 'surfaces'
            or build['surface_finder']['method'] != 'Regions First'):
        raise ValueError('Unexpected Surface Finder input or settings for '+frame)
    with Image.open(BytesIO(archive.read(prefix+'rgb.png'))) as src:
        image = Image.fromarray(np.asarray(src.convert('RGB')))
    model = read(source_dir/'surfaces.json')
    if model['image_size'] != list(image.size):
        raise ValueError('Surface Finder coordinate system differs from screenshot')
    parts = model['planes']
    samples = [sample_surface(image, part) for part in parts]
    rows = read_grid(image, parts, samples, methods)
    camera, decoded = export_coordinates(image, parts, rows, methods, out/'packs'/frame)

    pose = json.loads(archive.read(prefix+'camera.json'))
    if json.loads(archive.read(prefix+'frame.json'))['ground_y'] != GROUND:
        raise ValueError('Reference registration needs the capture ground height')
    faces = json.loads(archive.read(prefix+'reference-faces.json'))['faces']
    ref = truth(pose, faces)
    planes = reference_planes(ref, faces)
    visible = true_faces(planes, image.size)
    _, pairs = measure(parts, ref, use_mask=True)
    matched = dict(pairs)
    eligible = sum(p['eligible'] for p in ref['patches'].values())
    totals = {name: Counter(correct=0, incorrect=0, unmatched_returned=0,
                            proposed_faces=0, iou_matches=0, aligned=0) for name in methods}
    for i, (row, sample) in enumerate(zip(rows, samples, strict=True)):
        rid = matched.get(i)
        expected = planes[rid]['size'] if rid is not None else None
        true_cells = {cell for pid, cell in visible if pid == rid}
        inverse = np.linalg.inv(ref['charts'][rid]) if rid is not None else None
        row.update(reference=rid, expected=expected)
        for name, reading in row['readings'].items():
            counts = totals[name]
            if None not in reading['blocks']:
                if expected is None:
                    counts['unmatched_returned'] += 1
                else:
                    counts['correct' if expected == reading['blocks'] else 'incorrect'] += 1
            grade = grade_shapes(tile_shapes(sample, reading), true_cells, inverse)
            reading['footprints'] = grade
            counts.update(dict(proposed_faces=grade['proposed'], aligned=grade['aligned'], iou_matches=grade['iou_matches']))
    coords = {tuple(f['xyz']) for f in faces}
    transform = np.eye(4) if camera['P'] is None else registration(np.asarray(camera['P']), pose)
    for name, counts in totals.items():
        counts['missing'] = eligible - counts['correct'] - counts['incorrect']
        counts['unaligned_faces'] = counts['proposed_faces'] - counts['aligned']
        counts.update(_score(registered(decoded[name], transform), coords))
    return dict(frame=frame, image_sha256=image_sha, surfaces_sha256=digest(source_dir/'surfaces.json'),
                build_manifest_sha256=digest(source_dir/'build-manifest.json'), models=build['model_sha256'],
                stages=build['stages'], proposed_surfaces=len(parts), matched_surfaces=len(pairs),
                eligible_surfaces=eligible, fully_visible_faces=len(visible), camera=camera['scale_status'],
                rows=rows, totals=totals)


@image_only()
def run(shard, surfaces, selected, out):
    release = read(RELEASE)
    expected = release['shards']['test']['sha256']
    if digest(shard) != expected:
        raise ValueError('Twin Spire test shard does not match Visible Builds v1')
    roster = {r['id']:r['rgb_sha256'] for r in release['frames'] if r['layout']=='twinspire1' and r['split']=='test'}
    if set(roster) != set(FRAMES):
        raise ValueError('Expected the four prespecified Twin Spire variant 1 views')
    methods = {name: None if name == 'baseline' else import_module('benchmark.'+MODELS[name]) for name in selected}
    out.mkdir(parents=True, exist_ok=False)
    with ZipFile(shard) as archive:
        rows = [evaluate(archive, frame, roster[frame], surfaces, methods, out) for frame in FRAMES]
    scores = {k:sum(r[k] for r in rows) for k in ('proposed_surfaces','matched_surfaces','eligible_surfaces','fully_visible_faces')}
    scores['methods'] = {}
    for name in methods:
        total = Counter()
        for row in rows:
            total.update(row['totals'][name])
        scores['methods'][name] = dict(total)
    save(out/'scores.json', scores)
    save(out/'cases.json', rows)
    save(out/'manifest.json', dict(dataset=release['dataset'], shard_sha256=expected, version='26.3', frames=roster,
         selection='Prespecified Twin Spire variant 1, two palettes at two angles, held out as a family.',
         geometry='Image-only Surface Finder proposals, current rectifier, fully supported predicted footprints. '
                  'Scoring pairs surface masks and visible faces at IoU >=0.5 and requires four corners within 0.25 block.',
         export='Decoded smooth-stone geometry packs. The lift uses the independently estimated camera origin.',
         module_sha256={name:digest(Path(module.__file__)) for name,module in methods.items() if module},
         inputs={r['frame']:r['surfaces_sha256'] for r in rows}, evaluator_sha256=digest(Path(__file__)),
         artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(scores))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--shard', type=Path, required=True)
    cli.add_argument('--surfaces', type=Path, required=True)
    cli.add_argument('--methods', nargs='+', choices=['baseline', *MODELS], default=['baseline', *MODELS])
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    run(args.shard, args.surfaces, args.methods, args.out)
