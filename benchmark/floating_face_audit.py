"""Audit the original source-only seed quads against native visible block faces."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

from benchmark.camera_labels import screen_matrix
from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.floating_horizontal_bridges import horizontal_charts
from benchmark.floating_patch_controls import UNIT, fit_homography, inputs, native_centers, paths, project
from benchmark.floating_patch_report import baseline
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import axis_vector, oriented
from benchmark.relative_face_scoring import _yaw
from engine.surface_perimeter import _native_alpha
from engine.visible_raster import rasterize


COHORTS = ('original', 'fresh', 'supplemental', 'sky', 'holdout', 'holdout2')
REVIEW = ROOT / 'benchmark/reviews/floating-face-audit-01'
NEW_COHORTS = ('holdout3', 'holdout4')
THRESHOLDS = dict(iou=.65, visible_fraction=.70, corner_rmse_blocks=.20, proposal_native_fraction=.55)
CORNER_ORDERS = [np.roll(order, shift, axis=0) for order in (UNIT, UNIT[::-1]) for shift in range(4)]


def cohort_inputs(group):
    if group in COHORTS:
        return inputs(group)
    if group not in NEW_COHORTS:
        raise ValueError(f'Unexpected cohort {group}')
    holdout = ROOT / 'runs' / f'floating-{group}'
    ordinary = holdout / 'ordinary'
    return ordinary, ordinary / 'atlas/connected', holdout / 'source_material/charts', read(holdout / 'manifest.json')['cases']


def block_id(value):
    return value.removeprefix('minecraft:') if value is not None else None


class NativeFaces:
    def __init__(self, frame):
        labels = read(frame / 'labels.json')
        self.faces = {f['id']: f for f in labels['faces']}
        self.blocks = {b['id']: b for b in labels['blocks']}
        self.cubes = [f for f in self.faces.values() if self.blocks[f['block']].get('geometry_kind', 'full_cube') == 'full_cube']
        self.corners = np.asarray([face['corners_px'] for face in self.cubes])
        self.low, self.high = self.corners.min(axis=1), self.corners.max(axis=1)
        self.horizontal = np.asarray([bool(f['normal_world'][1]) for f in self.cubes])
        self.areas = np.asarray([cv2.contourArea(quad) for quad in self.corners.astype('float32')])
        camera = read(frame / 'camera.json')
        self.projection = screen_matrix(camera)
        self.eye = np.asarray(camera['eye_world'])
        self.size = tuple(camera['image_size'])
        with np.load(frame / 'labels.npz') as arrays:
            self.owners = arrays['face_id']
        counts = np.bincount(self.owners.ravel())
        if self.owners.shape[::-1] != self.size or not all(counts[i] == f['visible_pixels'] > 0 for i, f in self.faces.items()):
            raise ValueError(f'Native face index and raster disagree: {frame}')
        self.visibility_by_index = {}

    def visibility(self, index):
        if index in self.visibility_by_index:
            return self.visibility_by_index[index]
        face = self.cubes[index]
        corners = (np.asarray(face['corners_world']) - self.eye) @ self.projection[:, :3].T + self.projection[:, 3]
        info = rasterize(corners, *self.size)
        if info is None or np.max(abs(info[-1] - self.corners[index])) >= 1e-4:
            raise ValueError(f'Projected corners disagree for native face {face["id"]}')
        declared = face['visible_pixels']
        if np.count_nonzero(info[1] & (self.owners[info[0]] == face['id'])) != declared:
            raise ValueError(f'Raster ownership extends outside native face {face["id"]}')
        self.visibility_by_index[index] = min(1., float(declared / info[3]))
        return self.visibility_by_index[index]

    def overlap(self, quad, horizontal):
        candidate = ((self.low < quad.max(0)) & (self.high > quad.min(0))).all(axis=1)
        if horizontal is not None:
            candidate &= self.horizontal == horizontal
        indices = np.flatnonzero(candidate)
        seed_area = cv2.contourArea(quad.astype('float32'))
        best = (0., None)
        for index in indices:
            # Local pixel coordinates keep the convex intersection stable for small faces.
            low = np.minimum(quad.min(0), self.low[index])
            intersection, _ = cv2.intersectConvexConvex((quad - low).astype('float32'),
                                                       (self.corners[index] - low).astype('float32'))
            union = seed_area + self.areas[index] - intersection
            iou = min(1., intersection / union) if union > 0 else 0.
            if iou > best[0]:
                best = (iou, int(index))
        return best

    def seed(self, chart, cell, horizontal):
        seed = dict(u=cell['u'], v=cell['v'], proposed_id=block_id(cell.get('block_id')),
                    confidence=cell.get('confidence'), method=chart.get('method'))
        quad = project(np.asarray(chart['H']), np.asarray([cell['u'], cell['v']]) + UNIT)
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype('float32')):
            return dict(seed, classification='invalid_quad')
        iou, match = self.overlap(quad, horizontal)
        info = rasterize(np.column_stack([quad, np.ones(4)]), *self.size)
        counts = Counter(map(int, self.owners[info[0]][info[1]])) if info else Counter()
        native_pixels = sum(counts.values())
        fractions = {i: n / native_pixels for i, n in counts.items()}
        seed.update(proposal_quad_px=quad.tolist(), iou=iou, native_pixels=native_pixels,
                    background_fraction=fractions.get(0, 0.), native_face_count=len(counts.keys() - {0}))
        if match is None:
            return dict(seed, classification='no_eligible_face')

        face = self.cubes[match]
        visible = self.visibility(match)
        pix = fractions.get(face['id'], 0.)
        h = fit_homography([0, 0], self.corners[match])
        mapped = project(np.linalg.inv(h), quad)
        if not np.isfinite(mapped).all():
            return dict(seed, classification='invalid_face_projection', face_id=face['id'])
        errors = [np.linalg.norm(mapped - order, axis=1) for order in CORNER_ORDERS]
        corner_errors = min(errors, key=lambda e: np.mean(e**2))
        rms = float(np.sqrt(np.mean(corner_errors**2)))
        lengths = np.linalg.norm(mapped - np.roll(mapped, -1, axis=0), axis=1)
        lengths = (lengths[:2] + lengths[2:]) / 2
        ratio = cv2.contourArea(mapped.astype('float32'))
        native_id = block_id(self.blocks[face['block']]['block_id'])
        if iou >= THRESHOLDS['iou'] and rms <= THRESHOLDS['corner_rmse_blocks']:
            if visible < THRESHOLDS['visible_fraction']:
                result = 'partly_occluded'
            elif pix < THRESHOLDS['proposal_native_fraction']:
                result = 'weak_native_support'
            else:
                result = 'aligned'
        elif pix == 0:
            result = 'no_visible_overlap'
        elif max(lengths) < .80:
            result = 'undersized_both_axes'
        elif min(lengths) > 1.25:
            result = 'oversized_both_axes'
        elif min(lengths) < .80 or max(lengths) > 1.25:
            result = 'axis_scale_error'
        else:
            result = 'offset_or_distorted'
        return dict(seed, classification=result, face_id=face['id'], surface_id=face['surface_id'],
                    face_direction=face['direction'], xyz_world=face['xyz_world'], native_id=native_id,
                    material_exact=seed['proposed_id'] == native_id, visible_fraction=visible,
                    native_quad_px=self.corners[match].tolist(), proposal_native_fraction=pix,
                    corner_rmse_blocks=rms, corner_max_blocks=float(corner_errors.max()),
                    area_ratio=ratio, lengths_in_true_blocks=lengths.tolist())


def audit(job):
    group, case = job
    cv2.setNumThreads(1)
    source, ordinary, whole, _ = cohort_inputs(group)
    name = case['id']
    source_image = case['variants']['sky']
    image = ROOT / source_image['image']
    sha = digest(image)
    if sha != source_image['source_sha256']:
        raise ValueError(f'Frozen source changed: {image}')
    rel = f'sky/{name}/charts.json'
    packets = [read(root / rel) for root in (ordinary, whole)]
    handoff = source / 'sky' / name / 'source-handoff'
    for packet in packets:
        if packet['source_sha256'] != sha:
            raise ValueError(f'Chart image mismatch: {name}')
        paths.check_handoff(packet, handoff, sha)
    planes = {p['id']: p for p in read(handoff / 'surfaces.json')['planes']}
    frame = ROOT / case['frame']
    native = NativeFaces(frame)
    grown = {r['id']: r for r in packets[1]['charts']}
    if set(grown) - {r['id'] for r in packets[0]['charts']}:
        raise ValueError(f'Whole-wall packet has charts without a source seed packet: {name}')
    records = []
    for chart in packets[0]['charts']:
        ident = chart['id']
        normal = chart['normal_camera']
        horizontal = abs(normal[1]) >= .67 if normal is not None else None
        expanded = grown.get(ident)
        if expanded and expanded['H'] != chart['H']:
            raise ValueError(f'Whole-wall chart changed the seed lattice: {name}/{ident}')
        seeds = [native.seed(chart, c, horizontal) for c in chart['cells']]
        expanded_count = len(expanded['cells']) if expanded else 0
        row = dict(id=ident, horizontal=horizontal, seeds=seeds, grown_cells=expanded_count)
        if expanded_count and ident in planes:
            _, centers = native_centers(expanded, native.owners)
            alpha, (x, y, x1, y1) = _native_alpha(planes[ident]['support'], native.size)
            mask_ids = set(map(int, np.unique(native.owners[y:y1, x:x1][np.asarray(alpha) > 0]))) - {0}
            center_ids = set(map(int, centers)) - {0}
            seed_surfaces = {s['surface_id'] for s in seeds if s.get('proposal_native_fraction', 0) > 0}
            row.update(center_faces=len(center_ids), mask_faces=len(mask_ids),
                       center_blocks=len({tuple(native.faces[i]['xyz_world']) for i in center_ids}),
                       mask_blocks=len({tuple(native.faces[i]['xyz_world']) for i in mask_ids}),
                       center_background=int((centers == 0).sum()),
                       center_off_seed_surfaces=sum(bool(i and native.faces[i]['surface_id'] not in seed_surfaces) for i in centers),
                       center_surfaces=len({native.faces[i]['surface_id'] for i in center_ids}))
        records.append(row)
    export = attribute_export(group, name, image, planes, list(grown.values()), native, records)
    return dict(group=group, id=name, dataset=case['dataset'], split=case['split'], layout=case['layout'],
                source_sha256=sha, reference_sha256={k: digest(frame / k) for k in ('camera.json', 'labels.json', 'labels.npz')},
                seed_chart_sha256=digest(ordinary / rel), whole_chart_sha256=digest(whole / rel), charts=records,
                export=export)


def attribute_export(group, name, image, planes, grown, native, records):
    root = (ROOT / 'runs' / f'floating-{group}' / 'horizontal-cells/assembly' if group in NEW_COHORTS
            else baseline(group, horizontal=True))
    folder = root / 'connected-atlas/sky' / name
    receipt = read(folder / 'score.json')
    truth = {tuple(b['xyz_world']): b['block_id'] for b in native.blocks.values()}
    selected = receipt['selected_origins']
    charts, _ = oriented(grown)
    if set(selected) - {c['id'] for c in charts}:
        with Image.open(image) as photo:
            charts.extend(horizontal_charts(charts, list(planes.values()), native.size, np.asarray(photo.convert('RGB'))))
    by_id = {c['id']: c for c in charts}
    seed_records = {r['id']: r for r in records}
    local = {}
    sources = {}
    for ident in sorted(selected, key=lambda i: by_id[i].get('confidence', 0.), reverse=True):
        chart = by_id[ident]
        u, v = map(axis_vector, chart['axes'])
        normal = axis_vector(chart['normal_axis'])
        base = selected[ident] + np.minimum(0, u) + np.minimum(0, v) - np.maximum(0, normal)
        for c in chart['cells']:
            xyz = tuple(base + c['u'] * u + c['v'] * v)
            local.setdefault(xyz, 'minecraft:' + block_id(c['block_id']))
            sources.setdefault(xyz, ident)
    if decoded_pack(folder / 'reconstruction.zip') != local:
        raise ValueError(f'Cannot attribute decoded placements to selected charts: {group}/{name}')

    score = receipt['score']['fixed'] or receipt['score']['relaxed_yaw']
    totals = Counter()
    for xyz, material in local.items():
        world = tuple(a + b for a, b in zip(_yaw(xyz, score['yaw_quarters']), score['translation']))
        status = 'correct_id' if truth.get(world) == material else 'wrong_id' if world in truth else 'extra'
        ident = sources[xyz]
        row = seed_records.get(ident)
        if row is None:
            label = 'horizontal_added'
        elif any(s['classification'] == 'aligned' for s in row['seeds']):
            label = 'aligned_seed_chart'
        else:
            label = 'no_aligned_seed_chart'
        totals[f'{label}_{status}'] += 1
        if row is not None:
            row.setdefault('exported', Counter())[status] += 1
    return totals


def summarize(rows):
    charts = [c for row in rows for c in row['charts']]
    seeded = [c for c in charts if c['seeds']]
    seeds = [s for c in seeded for s in c['seeds']]
    classified = Counter(s['classification'] for s in seeds)
    no_aligned = [c for c in seeded if all(s['classification'] != 'aligned' for s in c['seeds'])]
    exported = Counter()
    for row in rows:
        exported.update(row['export'])
    return dict(views=len(rows), charts=len(charts), seeded_charts=len(seeded), seeds=len(seeds),
                label_counts=classified, exact_ids=sum(s.get('material_exact', False) for s in seeds),
                aligned_exact_ids=sum(s.get('material_exact', False) for s in seeds if s['classification'] == 'aligned'),
                supported_materials=sum('material_exact' in s and s['proposal_native_fraction'] > 0 for s in seeds),
                supported_exact_ids=sum(s.get('material_exact', False) and s.get('proposal_native_fraction', 0) > 0 for s in seeds),
                no_aligned_charts=len(no_aligned), grown_cells=sum(c['grown_cells'] for c in charts),
                no_aligned_grown_cells=sum(c['grown_cells'] for c in no_aligned),
                no_aligned_center_faces=sum(c.get('center_faces', 0) for c in no_aligned),
                no_aligned_mask_faces=sum(c.get('mask_faces', 0) for c in no_aligned), exported=exported)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('cohorts', nargs='*', default=COHORTS, choices=(*COHORTS, *NEW_COHORTS))
    cli.add_argument('--workers', type=int, default=4)
    cli.add_argument('--output', default='seed-audit')
    args = cli.parse_args()
    jobs = [(g, c) for g in args.cohorts for c in cohort_inputs(g)[3]]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(audit, jobs))
    groups = {g: summarize([r for r in rows if r['group'] == g]) for g in args.cohorts}
    save(REVIEW / f'{args.output}.json.gz', dict(thresholds=THRESHOLDS, groups=groups, cases=rows))
    for g, data in groups.items():
        print(g, data, flush=True)


if __name__ == '__main__':
    main()
