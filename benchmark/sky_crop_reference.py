"""Score Sky Surface Finder crops on exact frontmost, visible-only surface rasters."""

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from benchmark.relative_oracle_charts import components
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha


FIXTURES = SOURCE / 'floating-build-grid-01/fixtures.json'
HANDOFFS = WORKTREES / 'floating-visible-builds/runs/floating-cloud-gate-01/sky'
OUTPUT = ROOT / 'runs/sky-crop-reference-01'


@dataclass
class Reference:
    """A native source-aligned label image, with one label per connected cube plane."""

    labels: np.ndarray
    targets: list
    areas: np.ndarray
    non_cube: np.ndarray


def reference_raster(frame):
    """Group native face IDs exactly as relative_oracle_charts groups visible cells.

    Distinct world components keep distinct labels even when adjacent in the
    image. Occlusion can leave one world component's native pixels disconnected.
    Non-full-cube model quads are tracked outside this connected-cube target.
    """
    refs = read(frame / 'labels.json')
    with np.load(frame / 'labels.npz', allow_pickle=False) as data:
        owners = data['face_id']
    counts = np.bincount(owners.ravel())
    blocks = {block['id']: block for block in refs['blocks']}
    face_ids = {face['id'] for face in refs['faces']}
    if set(np.flatnonzero(counts)) - {0} != face_ids:
        raise ValueError(f'Inconsistent visible face IDs in {frame}')
    planes = defaultdict(dict)
    non_cube = np.zeros(len(counts), dtype=bool)
    for face in refs['faces']:
        face_id = face['id']
        if counts[face_id] != face['visible_pixels']:
            raise ValueError(f'Inconsistent visible pixel count for face {face_id} in {frame}')
        block = blocks[face['block']]
        if block.get('geometry_kind', 'full_cube') != 'full_cube':
            non_cube[face_id] = True
            continue
        normal = face['normal_world']
        axis = next(index for index, value in enumerate(normal) if value)
        signed = normal[axis] * (axis + 1)
        xyz = face['xyz_world']
        offset = xyz[axis] + int(signed > 0)
        uv = tuple(value for index, value in enumerate(xyz) if index != axis)
        plane = planes[signed, offset]
        if uv in plane:
            raise ValueError(f'Duplicate full-cube face cell in {frame}')
        plane[uv] = face

    groups = []
    for (signed, offset), plane in planes.items():
        for cells in components(plane):
            faces = [plane[cell] for cell in cells]
            face_ids = sorted(face['id'] for face in faces)
            groups.append(dict(id=f'T{face_ids[0]:04}', face_ids=face_ids,
                               direction=faces[0]['direction'], normal_axis=signed,
                               plane_offset=offset,
                               capture_surface_ids=sorted({face['surface_id'] for face in faces})))
    targets = sorted(groups, key=lambda group: group['face_ids'][0])
    face_to_target = np.zeros(len(counts), dtype=np.int32)
    for index, target in enumerate(targets, 1):
        face_to_target[target['face_ids']] = index
    labels = face_to_target[owners]
    areas = np.bincount(labels.ravel(), minlength=len(targets) + 1)
    return Reference(labels, targets, areas, non_cube[owners])


def native_mask(support, size):
    """Apply the same native pixel-center sampling used by the photo crop."""
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    result = _native_alpha(support, size)
    if result is not None:
        alpha, (left, top, right, bottom) = result
        mask[top:bottom, left:right] = np.asarray(alpha) > 0
    return mask


def check_support(support):
    """Reject malformed replacements before a pair can silently score."""
    width, height = support['size']
    if not all(isinstance(value, int) and value > 0 for value in (width, height)):
        raise ValueError('Replacement size must contain positive integer dimensions')
    for y, left, right in support['runs']:
        if not all(isinstance(value, int) for value in (y, left, right)):
            raise ValueError('Replacement runs must contain integer coordinates')
        if not (0 <= y < height and 0 <= left < right <= width):
            raise ValueError(f'Run is outside support size: {(y, left, right)}')


def intersection(reference, mask):
    return np.bincount(reference.labels[mask], minlength=len(reference.areas))


def overlap_iou(areas, hits, prediction_pixels):
    union = areas + prediction_pixels - hits
    return np.divide(hits, union, out=np.zeros_like(union, dtype=float), where=union > 0)


def best_target(areas, hits, prediction_pixels):
    """Max IoU among positive overlaps, ties by stable target ID."""
    if not np.any(hits[1:]):
        return 0
    return int(np.argmax(overlap_iou(areas, hits, prediction_pixels)[1:]) + 1)


def score_mask(reference, mask, target_index, hits):
    pixels = int(hits.sum())
    target_pixels = int(reference.areas[target_index]) if target_index else 0
    intersection_pixels = int(hits[target_index]) if target_index else 0
    missed = target_pixels - intersection_pixels
    extra = pixels - intersection_pixels
    non_cube = int((reference.non_cube & mask).sum())
    other_targets = pixels - int(hits[0]) - intersection_pixels
    return dict(
        mask_pixels=pixels, target_pixels=target_pixels, intersection=intersection_pixels,
        missed=missed, extra=extra, other_cube_pixels=other_targets,
        non_cube_pixels=non_cube, background_pixels=int(hits[0]) - non_cube,
        overlap_targets=int(np.count_nonzero(hits[1:])),
        iou=ratio(intersection_pixels, intersection_pixels + missed + extra),
        precision=ratio(intersection_pixels, pixels),
        recall=ratio(intersection_pixels, target_pixels),
        reference_reassignment=target_id(reference, best_target(reference.areas, hits, pixels)),
    )


def ratio(n, d):
    return float(n / d) if d else 0.0


def target_id(reference, target_index):
    return reference.targets[target_index - 1]['id'] if target_index else None


def verified_crop(image, part, handoff, baseline):
    """Ensure the review compares the actual on-disk photo crop and native mask."""
    record = part['source_crop']
    path = handoff / record['path']
    with Image.open(path) as saved:
        actual = np.asarray(saved.convert('RGBA'))
    box = tuple(record['xyxy'])
    left, top, right, bottom = box
    mask = baseline[top:bottom, left:right]
    if (mask.shape != actual.shape[:2] or int(mask.sum()) != int(baseline.sum())
            or not np.all(actual[:, :, 3] == 255 * mask)
            or not np.array_equal(actual[:, :, :3][mask], np.asarray(image.crop(box))[mask])):
        raise ValueError(f'Saved crop differs from source RGB or native support: {path}')
    return box, digest(path)


def arm_summary(rows, coverage):
    overlap = sum(row['intersection'] for row in rows)
    missed = sum(row['missed'] for row in rows)
    extra = sum(row['extra'] for row in rows)
    return dict(
        active_masks=len(rows), nonempty_masks=sum(row['mask_pixels'] > 0 for row in rows),
        matched_masks=sum(row['intersection'] > 0 for row in rows),
        masks_iou_at_least_50=sum(row['iou'] >= .50 for row in rows),
        masks_iou_at_least_75=sum(row['iou'] >= .75 for row in rows),
        masks_exact=sum(row['iou'] == 1 for row in rows),
        macro_iou=ratio(math.fsum(row['iou'] for row in rows), len(rows)),
        pixel_weighted_iou=ratio(overlap, overlap + missed + extra),
        intersection=overlap, missed=missed, extra=extra,
        other_cube_pixels=sum(row['other_cube_pixels'] for row in rows),
        non_cube_pixels=sum(row['non_cube_pixels'] for row in rows),
        background_pixels=sum(row['background_pixels'] for row in rows),
        **coverage,
    )


def coverage_summary(reference, union, scores, selections):
    hits = intersection(reference, union)
    targets = []
    for index, target in enumerate(reference.targets, 1):
        areas = reference.areas[index]
        best_iou = max((item[index] for item in scores), default=0.0)
        targets.append(dict(**target, pixels=int(areas), covered_pixels=int(hits[index]),
                            best_iou=float(best_iou), primary_predictions=selections.count(index)))
    covered = int(hits[1:].sum())
    total = int(reference.areas[1:].sum())
    non_cube = int((reference.non_cube & union).sum())
    coverage = dict(
        targets=len(targets), targets_without_pixels=sum(t['covered_pixels'] == 0 for t in targets),
        targets_without_primary=sum(t['primary_predictions'] == 0 for t in targets),
        targets_iou_at_least_50=sum(t['best_iou'] >= .50 for t in targets),
        targets_iou_at_least_75=sum(t['best_iou'] >= .75 for t in targets),
        full_cube_covered_pixels=covered, full_cube_target_pixels=total,
        full_cube_coverage=ratio(covered, total),
        union_non_cube_pixels=non_cube, reference_non_cube_pixels=int(reference.non_cube.sum()),
        union_background_pixels=int(hits[0]) - non_cube)
    return coverage, targets


def evaluate_case(case, handoffs, candidates):
    """Return scoring records and the baseline's same-coordinate visual evidence."""
    ident = case['id']
    source = Path(case['variants']['sky']['image'])
    source_hash = digest(source)
    frame = Path(case['frame'])
    reference_hashes = {name: digest(frame / name) for name in case['reference_sha256']}
    if source_hash != case['variants']['sky']['source_sha256'] or reference_hashes != case['reference_sha256']:
        raise ValueError(f'Frozen source or reference changed on {ident}')
    image = Image.open(source).convert('RGB')
    reference = reference_raster(frame)
    if image.size != reference.labels.shape[::-1]:
        raise ValueError(f'Native reference dimensions differ from source on {ident}')

    handoff = handoffs / ident / 'source-handoff'
    manifest = read(handoff / 'build-manifest.json')
    surfaces = read(handoff / 'surfaces.json')
    if (manifest['image_sha256'] != source_hash or surfaces['image_size'] != list(image.size)
            or manifest['stages']['background_filter'] != 'border-connected-whole-sky-v1'):
        raise ValueError(f'Handoff is not the frozen Sky baseline on {ident}')
    parts = sorted(surfaces['planes'], key=lambda part: -part['support_area'])
    ids = {part['id'] for part in parts}
    kept_count = manifest['surface_finder']['boundary_settings']['background_filter']['kept_count']
    if len(ids) != len(parts) or kept_count != len(parts):
        raise ValueError(f'Active proposal inventory mismatch on {ident}')
    for name, replacements in candidates.items():
        unknown = set(replacements.get(ident, {})) - ids
        if unknown:
            raise ValueError(f'Unexpected active plane IDs on {name}/{ident}: {sorted(unknown)}')

    arms = ['baseline', *candidates]
    all_rows = {name: [] for name in arms}
    scores = {name: [] for name in arms}
    selections = {name: [] for name in arms}
    unions = {name: np.zeros(reference.labels.shape, dtype=bool) for name in arms}
    exact_masks = {}
    exact_support_by_target = {}
    visuals = []
    for order, part in enumerate(parts, 1):
        proposal = part['id']
        baseline = native_mask(part['support'], image.size)
        crop_box, crop_hash = verified_crop(image, part, handoff, baseline)
        hits = intersection(reference, baseline)
        paired = best_target(reference.areas, hits, int(hits.sum()))
        if paired:
            if paired not in exact_support_by_target:
                exact_support_by_target[paired] = dict(
                    size=list(image.size), runs=row_runs(reference.labels == paired))
            exact_masks[proposal] = exact_support_by_target[paired]
        label = f'S{order:02}'
        visuals.append((label, part, baseline, paired))
        for name in arms:
            replacement = candidates.get(name, {}).get(ident, {}).get(proposal)
            if replacement is None:
                mask, selected_hits = baseline, hits
            else:
                check_support(replacement)
                mask = native_mask(replacement, image.size)
                selected_hits = intersection(reference, mask)
            metrics = score_mask(reference, mask, paired, selected_hits)
            metrics.update(id=proposal, label=label, target=target_id(reference, paired),
                           changed=not np.array_equal(mask, baseline))
            if name == 'baseline':
                metrics.update(crop_xyxy=list(crop_box), crop_sha256=crop_hash,
                               normal_camera=part['normal_camera'])
            all_rows[name].append(metrics)
            scores[name].append(overlap_iou(reference.areas, selected_hits, metrics['mask_pixels']))
            selections[name].append(best_target(reference.areas, selected_hits, metrics['mask_pixels']))
            unions[name] |= mask

    reports = {}
    for name in arms:
        coverage, targets = coverage_summary(reference, unions[name], scores[name], selections[name])
        reports[name] = dict(summary=arm_summary(all_rows[name], coverage), proposals=all_rows[name], targets=targets)
    report = dict(
        id=ident, family=case['family'], dataset=case['dataset'], split=case['split'],
        source_sha256=source_hash, reference_sha256=reference_hashes,
        surfaces_sha256=digest(handoff / 'surfaces.json'),
        source_handoff_sha256=digest(handoff / 'build-manifest.json'),
        arms=reports,
    )
    return report, exact_masks, (image, handoff, reference, visuals)


def pooled_metrics(cases, arms):
    counts = ('targets', 'targets_without_pixels', 'targets_without_primary', 'targets_iou_at_least_50',
              'targets_iou_at_least_75', 'full_cube_covered_pixels', 'full_cube_target_pixels',
              'union_non_cube_pixels', 'reference_non_cube_pixels', 'union_background_pixels')
    pooled = {}
    for name in arms:
        reports = [case['arms'][name] for case in cases]
        rows = [row for report in reports for row in report['proposals']]
        coverage = {field: sum(report['summary'][field] for report in reports) for field in counts}
        coverage['full_cube_coverage'] = ratio(coverage['full_cube_covered_pixels'],
                                               coverage['full_cube_target_pixels'])
        pooled[name] = arm_summary(rows, coverage)
    return pooled


def grouped_metrics(cases, field, arms):
    groups = defaultdict(list)
    for case in cases:
        groups[case[field]].append(case)
    return {name: dict(frames=[case['id'] for case in group], pooled=pooled_metrics(group, arms))
            for name, group in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--handoffs', type=Path, default=HANDOFFS, help='Root with CASE/source-handoff')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--candidates', metavar='NAME=PATH', nargs='*', default=[])
    parser.add_argument('--scores-only', action='store_true', help='Skip regenerating review images and HTML')
    args = parser.parse_args()
    fixtures = read(args.fixtures)
    cases = fixtures['cases']
    ids = {case['id'] for case in cases}
    if len(ids) != len(cases):
        parser.error('Fixture manifest contains duplicate frame IDs')
    candidates, candidate_hashes = {}, {}
    for spec in args.candidates:
        name, separator, path = spec.partition('=')
        if not separator or not name or name in candidates or name == 'baseline':
            parser.error(f'Expected a unique candidate name and path: {spec}')
        path = Path(path)
        candidates[name] = read(path)['frames']
        candidate_hashes[name] = digest(path)
        if unknown := set(candidates[name]) - ids:
            parser.error(f'Unknown candidate frame IDs for {name}: {sorted(unknown)}')
    reports = []
    exact = {}
    for case in cases:
        report, exact[case['id']], visual_data = evaluate_case(case, args.handoffs, candidates)
        reports.append(report)
        metrics = report['arms']['baseline']['summary']
        print(case['id'], metrics['active_masks'], metrics['targets'],
              f"mean IoU {metrics['macro_iou']:.3f}", flush=True)
        if not args.scores_only:
            from benchmark.sky_crop_reference_review import write_case
            write_case(args.output, report, *visual_data)
    arms = ['baseline', *candidates]
    report = dict(
        method='native-connected-full-cube-surfaces-v1',
        handoff_revision='52de2c9',
        fixture_revision=fixtures['revision'], fixture_sha256=digest(args.fixtures),
        scorer_sha256=digest(Path(__file__)), candidate_sha256=candidate_hashes,
        cases=reports, pooled=pooled_metrics(reports, arms),
        by_split=grouped_metrics(reports, 'split', arms),
        by_dataset=grouped_metrics(reports, 'dataset', arms),
    )
    save(args.output / 'exact-target-masks.json', dict(
        method='reference-only-positive-baseline-best-iou', fixtures_sha256=report['fixture_sha256'],
        frames=exact))
    save(args.output / 'report.json', report)
    if not args.scores_only:
        from benchmark.sky_crop_reference_review import write_index
        write_index(args.output, report)


if __name__ == '__main__':
    main()
