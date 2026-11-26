"""Score frozen grid choices on visible face footprints and decoded Minecraft structures."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_export import _archive_pack
from benchmark.count_materials import true_faces
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.mask_blocks import SETTINGS, _blocks, _carriers, _footprints, _score, _write_read, registered
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.textures import match_cells
from benchmark.verify_default_strip import SIGNATURES, region_signature
from benchmark.wall_join_test import checked, read_gzip
from engine.camera_grid import predict
from engine.geometry import UNIT, project
from engine.strip_neighbor import clean_surfaces
from engine.surface_sample import sample_surface
from engine.surface_tiles import complete_footprints
from runtime.mode import image_only, mark_image_only


CORNER_TOLERANCE = .25  # Measured on the recorded plane in Minecraft blocks.
EMPTY = dict(counts=[None, None], phase=[None, None], blocks=[None, None])
METHODS = ('baseline', 'ranker')


def choose(surface, index):
    if index is None:
        return surface['baseline']
    if index == -1:
        return EMPTY
    if not 0 <= index < len(surface['candidates']):
        raise ValueError(f"Invalid choice for {surface['id']}: {index}")
    return surface['candidates'][index]


def corner_error(quad, cell, inverse):
    """Worst corner distance using the best cyclic correspondence, either winding."""
    xy = project(inverse, quad)
    truth = np.asarray(cell) + UNIT
    return float(min(np.linalg.norm(xy - np.roll(order, offset, axis=0), axis=1).max()
                     for order in (truth, truth[::-1]) for offset in range(4)))


def tile_shapes(sample, reading):
    return [q for _, q in complete_footprints(sample, reading)
            if np.min(np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)) >= 4.]


def grade_shapes(shapes, eligible, inverse):
    matches = match_cells(shapes, eligible, inverse)[0] if shapes and eligible else {}
    errors = [corner_error(q, cell, inverse) for cell, (q, _, _) in matches.items()]
    return dict(proposed=len(shapes), iou_matches=len(matches),
                aligned=sum(e <= CORNER_TOLERANCE for e in errors))


def score_grids(samples, inputs, selections, surfaces, reference, planes, size):
    valid = true_faces(planes, size)
    ordinary = {name: Counter() for name in METHODS}
    oracle = Counter()
    cases = []
    for sample, row, choice in zip(samples, inputs, selections, strict=True):
        rid = surfaces.get(row['id'])
        eligible = {cell for p, cell in valid if p == rid}
        inverse = np.linalg.inv(reference['charts'][rid]) if rid is not None else None
        expected = planes[rid]['size'] if rid is not None else None
        # Reuse identical candidate geometries, independently of confidence or dimensional correctness.
        cache = {}
        def grade(reading):
            key = tuple(reading['counts']), tuple(reading['phase'])
            if key not in cache:
                cache[key] = grade_shapes(tile_shapes(sample, reading), eligible, inverse)
            return cache[key]

        rows = {}
        for method, reading in (('baseline', row['baseline']), ('ranker', choose(row, choice['index']))):
            rec = grade(reading)
            counts = ordinary[method]
            counts.update(rec)
            resolved = sum(v is not None for v in reading['phase'])
            counts[('no_phase', 'one_axis_phase', 'both_axes_phase')[resolved]] += 1
            counts['surfaces_with_aligned_face'] += int(rec['aligned'] > 0)
            counts['exact_size_and_aligned_face'] += int(rec['aligned'] > 0 and reading['blocks'] == expected)
            rows[method] = rec
        # The perfect chooser can pick ONE measured grid per surface with access to the answers.
        # These two ceilings have distinct objectives; their best candidate can differ.
        best = both = 0
        had_exact_size = False
        if rid is not None:
            for cand in [row['baseline'], *row['candidates']]:
                exact = cand['blocks'] == expected
                had_exact_size |= exact
                if eligible and None not in cand['phase']:
                    aligned = grade(cand)['aligned']
                    best = max(best, aligned)
                    if exact:
                        both = max(both, aligned)
        oracle.update(aligned=best, surfaces_with_aligned_face=int(best > 0),
                      exact_size_and_aligned_face=int(both > 0), surfaces_with_exact_size=int(had_exact_size))
        cases.append(dict(id=row['id'], reference_id=rid, eligible=len(eligible),
                          oracle_aligned=best, oracle_aligned_with_exact_dimensions=both, **rows))
    return (dict(eligible_visible_faces=len(valid), methods=ordinary, perfect_grid_chooser=oracle),
            cases)


def export_readings(image, parts, inputs, selections, directory):
    """Write and decode both geometry-only packs before opening any reference pose or labels."""
    camera = predict(image, '26.3', SETTINGS['camera'])
    P = None if camera['P'] is None else np.asarray(camera['P'])
    masks = _footprints(parts, image.size, True)
    outputs, cache = {}, {}
    for method in METHODS:
        charts = []
        for part, data, choice in zip(parts, inputs, selections, strict=True):
            reading = data['baseline'] if method == 'baseline' else choose(data, choice['index'])
            counts = reading['counts']
            charts.append(dict(id=part['id'], quad=data['quad'],
                               pitch={'counts': counts if None not in counts else None}))
        carriers = _carriers(image, charts, masks, P)
        blocks, outcomes, _ = _blocks(masks, carriers, P, cache)
        outputs[method] = (_write_read(directory/(method+'.zip'), blocks), outcomes)
    return camera, outputs


def evaluate(job, *, out):
    capture, source, pred_sha, signature, candidates, choices = job
    key = source['key']
    with Image.open(checked(capture.path('image'), source['sha256'])) as opened:
        image = Image.fromarray(np.asarray(opened.convert('RGB')))
    earlier = read_gzip(checked(BASE/source['predictions'], pred_sha))['gap']
    parts, _ = clean_surfaces(image, earlier)
    if region_signature(parts) != signature:
        raise ValueError(f'Approved proposal set changed on {key}')
    samples = [sample_surface(image, part) for part in parts]
    inputs = candidates['surfaces']
    selections = choices['surfaces']
    for sample, part, data, selected in zip(samples, parts, inputs, selections, strict=True):
        if not (part['id'] == data['id'] == selected['id']):
            raise ValueError(f'Unexpected surface identity on {key}')
        if (sample['quad'] if sample else None) != data['quad']:
            raise ValueError(f'Sampling chart changed for {key}/{part["id"]}')
    folder = out / 'packs' / key
    folder.mkdir(parents=True)
    camera, decoded = export_readings(image, parts, inputs, selections, folder)

    pose, ref = recorded(capture)
    faces = read(capture.path('labels'))['faces']
    planes = reference_planes(ref, faces)
    _, pairs = measure(parts, ref, use_mask=True)
    matched = {parts[i]['id']: rid for i, rid in pairs}
    grids, details = score_grids(samples, inputs, selections, matched, ref, planes, image.size)
    target = {tuple(face['xyz']) for face in faces}
    transform = np.eye(4) if camera['P'] is None else registration(np.asarray(camera['P']), pose)
    exports = {}
    for method, (blocks, outcomes) in decoded.items():
        exports[method] = dict(coordinates=_score(registered(blocks, transform), target),
                               regions=outcomes, pack=f'{key}/{method}.zip')
    return dict(key=key, sha256=source['sha256'], cohort=candidates['cohort'], layout=source['layout'],
                camera_status=camera['scale_status'], grids=grids, surfaces=details, exports=exports)


def totals(frames):
    def combine(rows):
        values = Counter()
        for row in rows:
            values.update(row)
        return dict(values)
    result = dict(frames=len(frames), eligible_visible_faces=sum(f['grids']['eligible_visible_faces'] for f in frames),
                  perfect_grid_chooser=combine(f['grids']['perfect_grid_chooser'] for f in frames),
                  supported_cameras=sum(f['camera_status'] == 'supported' for f in frames))
    for method in METHODS:
        result[method] = dict(grids=combine(f['grids']['methods'][method] for f in frames),
                              coordinates=combine(f['exports'][method]['coordinates'] for f in frames),
                              nonempty_packs=sum(f['exports'][method]['coordinates']['serialized'] > 0 for f in frames))
    return result


@image_only()
def run(candidates_file, selections_file, out, workers):
    source = read(checked(BASE/'manifest.json', BASE_SHA))
    rows = read(checked(BASE/'cases.json', source['artifacts']['cases.json']))
    checked(ARCHIVE, source['reference_sha256'])
    signatures = read(SIGNATURES)['frames']
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}': c for cases in cohorts.values() for c in cases}
    if fixtures != source['fixtures'] or suite['version'] != source['version']:
        raise ValueError('Frozen capture release changed')
    expected = {r['key']: r['sha256'] for r in rows}
    if len(expected) != 56 or signatures.keys() != expected.keys() or captures.keys() != expected.keys():
        raise ValueError('Unexpected capture inventory')
    def index(path):
        frames = read(path)
        if len(frames) != len(expected) or {f['key']: f['sha256'] for f in frames} != expected:
            raise ValueError(f'Screenshot roster differs in {path}')
        return {f['key']: f for f in frames}
    candidates, choices = index(candidates_file), index(selections_file)
    jobs = []
    for row in rows:
        key = row['key']
        predicted = candidates[key]
        cohort = 'development' if row['layout'] in suite['layouts']['dev'] else 'regression'
        if cohort != predicted['cohort'] or row['layout'] != predicted['layout']:
            raise ValueError(f'Unexpected layout or split for {key}')
        jobs.append((captures[key], row, source['artifacts'][row['predictions']], signatures[key],
                     predicted, choices[key]))
    out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        frames = list(pool.map(partial(evaluate, out=out), jobs))
    scores = {name: totals([f for f in frames if name in ('all', f['cohort'])])
              for name in ('all', 'development', 'regression')}
    save(out/'cases.json', frames)
    save(out/'scores.json', scores)
    with ZipFile(out/'geometry-packs.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for frame in frames:
            for rec in frame['exports'].values():
                member = rec['pack']
                _archive_pack(archive, member, out/'packs'/member)
    save(out/'manifest.json', dict(code=revision(), fixtures=fixtures, frames=source['frames'],
         source_manifest_sha256=BASE_SHA, approved_outlines_sha256=digest(SIGNATURES),
         reference_sha256=source['reference_sha256'], candidates_sha256=digest(candidates_file),
         selections_sha256=digest(selections_file), pictures=56, splits=source['splits'],
         geometry_settings=SETTINGS, corners_max_error_blocks=CORNER_TOLERANCE,
         local_scoring='Fully supported predicted footprints and fully visible truth faces, minimum four-pixel edge. '
            'On matched surfaces pair one-to-one at plane IoU >=0.5. Aligned requires every corner within '
            'the stated Euclidean tolerance in block-plane coordinates. Oracle picks one candidate per surface.',
         coordinate_scoring='Image-derived seam-fit camera supplies global origin. Only the continuous counts enter '
            'the geometry lift. Write and decode smooth-stone Minecraft structure packs, then register and grade '
            'coordinates against visible truth. Material identity and placement inside a running game are unmeasured.',
         artifacts={name: digest(out/name) for name in ('scores.json', 'cases.json', 'geometry-packs.zip')}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--candidates', type=Path, required=True)
    cli.add_argument('--selections', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--workers', type=int, default=4)
    args = cli.parse_args()
    run(args.candidates.resolve(), args.selections.resolve(), args.out.resolve(), args.workers)
