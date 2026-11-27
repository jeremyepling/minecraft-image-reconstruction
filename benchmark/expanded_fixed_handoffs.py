"""Replay the fixed 104 screenshot controls from their checked pre-inset handoff."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from zipfile import ZipFile

os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

from PIL.Image import Image

from benchmark import facade_inset_round as suite, typed_structures as typed
from benchmark.edge_vote_outline_eval import recount
from benchmark.expanded_baseline import MEASURED_REVISION, ROOT, digest, read, save
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.expanded_scores import FIXED_DATASET, FrameReference, camera_fields, target_breakdown
from benchmark.mask_blocks import decoded_pack
from benchmark.pavement_opening_eval import visible_faces
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.thin_face_placement_run import ref_inputs
from engine import counted_ground_corner, eave_pinholes, exterior_foreground, horizon_continuity
from engine import inset_returns, jagged_texture_join, pavement_opening, placed_materials
from engine import smooth_atlas_outline, smooth_extent_placement, surface_model, texture_guided_count
from engine import visible_inset_cells, visible_placement, window_wall_recheck
from engine.camera_atlas_floor import _atlas
from engine.pavement_obstructions import window_obstructions
from engine.surface_perimeter import with_outline


VERSION = '26.3'
OUTPUT = ROOT / 'runs/fixed-round-controls'


@lru_cache(maxsize=1)
def _jobs():
    return {frame['id']: (split, frame, prior, control)
            for split, frame, prior, control in suite.jobs(suite.framed.EVALUATION)}


@lru_cache(maxsize=1)
def _release():
    manifest = read(typed.REVIEW / 'summary.json')
    cases = typed.REVIEW / 'cases.json'
    archive = typed.REVIEW / 'typed-exports.zip'
    if digest(cases) != manifest['cases_sha256'] or digest(archive) != manifest['exports_sha256']:
        raise ValueError('Released fixed controls failed checksum verification')
    rows = {row['id']: row for row in read(cases) if row['layout'] is not None}
    if len(rows) != 104 or rows.keys() != _jobs().keys():
        raise ValueError('Released and archived fixed frame IDs differ')
    return manifest, rows, archive


def fixed_frames():
    """Enumerate the released fixed frame IDs and immutable RGB metadata."""
    manifest, rows, _ = _release()
    for key, row in sorted(rows.items()):
        split, frame, prior, control = _jobs()[key]
        source_hash = frame['files']['rgb.png']
        if not source_hash == row['source_hash'] == prior['rgb_sha256'] == control['rgb_sha256']:
            raise ValueError(f'Fixed RGB checksums disagree: {key}')
        yield dict(id=key, dataset=FIXED_DATASET, split=split, layout=row['layout'],
                   family=frame.get('family', row['layout']), palette=frame.get('palette'),
                   shape_sha256=row['shape_sha256'], rgb_sha256=source_hash,
                   rgb=typed.source_path(split, frame), dataset_manifest_sha256=manifest['cases_sha256'])


@dataclass(frozen=True)
class FixedInputs:
    image: Image
    parts: list
    counts: dict
    camera: dict
    surfaces: list
    public_counts: dict
    pre_inset: list
    measured_surfaces: list
    scene_context: list | None
    roof_surfaces: list
    inset_events: list
    phase_events: list
    confirmed_outlines: list
    exterior_events: list
    common_wall_regions: list


@lru_cache(maxsize=1)
def _export_atlas():
    atlas = _atlas(VERSION, suite.ATLAS)
    if atlas is None:
        raise ValueError('The fixed placement replay requires the released Java 26.3 atlas')
    return atlas


def source_inputs(entry):
    """Load the checked RGB, pre-inset regions, saved counts and estimated camera."""
    job = _jobs()[entry['id']]
    if job[1]['files']['rgb.png'] != entry['rgb_sha256']:
        raise ValueError(f'Wrong fixed source hash: {entry["id"]}')
    return suite.ordinary_inputs(*job)


def inference_inputs(entry, *, boundary=None):
    """Return source-only inputs to ranked placement and every later consumer.

    Optional ``boundary(image, parts)`` replaces the frozen pre-inset regions.
    All ordinary inset, background, count and placement stages then rerun. Each
    call owns its loaded image and dictionaries. The archive is only read.
    """
    image, original, supplied_counts, camera = source_inputs(entry)
    if boundary is not None:
        original = boundary(image, original)
    # The archived Surface Finder output is the ordinary input immediately
    # before inset detection. Recreate its private placement context using the
    # same operators and order as the Java 26.3 through-placement entry point.
    pre_inset = original
    surfaces, insets, phases, confirmed, camera = inset_returns.detect(
        image, original, VERSION, atlas_path=suite.ATLAS, camera=camera)
    surfaces, eaves = eave_pinholes.refine(image, surfaces)
    if not (insets or phases or confirmed):
        surfaces, _, camera = jagged_texture_join.refine(image, surfaces, VERSION, camera=camera, atlas_path=suite.ATLAS)
        pre_inset = surfaces
    surfaces, smooth, common_walls = smooth_atlas_outline.with_common_wall(
        image, surfaces, VERSION, atlas_loader=_export_atlas)
    if camera is None and any(part.get('normal_camera') is not None for part in surfaces):
        camera = visible_placement.seam_camera(image, VERSION, atlas_path=suite.ATLAS)

    scene_context = None
    exterior, exterior_events = exterior_foreground.refine(image, surfaces, camera)
    if exterior_events:
        scene_context = [with_outline(part) for part in surfaces]
        surfaces = exterior
    roof_surfaces = [with_outline(part) for part in surfaces]
    foreground, horizon_events = horizon_continuity.refine(image, roof_surfaces)
    if horizon_events:
        if scene_context is None:
            scene_context = roof_surfaces
        surfaces = [part if 'outline' in part else with_outline(part) for part in foreground]
    else:
        surfaces = roof_surfaces

    def measurement(parts):
        measured = smooth_atlas_outline.measurement_parts(parts) if smooth.get('events') else parts
        return measured, eave_pinholes.measurement_parts(measured) if eaves else measured

    counter_atlas = _export_atlas() if 'atlas_phase' in supplied_counts else None
    measured_surfaces, parts = measurement(surfaces)
    public_counts = recount(image, parts, counter_atlas)
    counts = public_counts
    if scene_context is not None:
        _, parts = measurement(scene_context)
        counts = recount(image, parts, counter_atlas)
    if insets or confirmed:
        parts = [with_outline(part) for part in pre_inset]
        hosts = {event['host'] for event in insets + confirmed}
        counts = surface_model._inset_layout_counts(image, VERSION, parts, counts, hosts,
            texture_guided_count.METHOD, counter_atlas, public_counts=public_counts)
    if insets:
        spans = visible_inset_cells.observed_back_spans(image.size, camera, measured_surfaces, insets, phases)
        for row in public_counts['surfaces']:
            if row['id'] in spans:
                row['observed_visible_grid_span'] = spans[row['id']]
    return FixedInputs(image, parts, counts, camera, surfaces, public_counts, pre_inset,
                       measured_surfaces, scene_context, roof_surfaces, insets, phases,
                       confirmed, exterior_events, common_walls)


@dataclass(frozen=True)
class ReplayResult:
    pack: bytes
    assignment: dict
    status: dict


def replay_placement(inputs, *, place=None, count_transform=None):
    """Run ranked placement, ordinary terminal checks, and the material export.

    ``place`` has the ordinary placer signature. ``count_transform(image, parts,
    counts, camera)`` replaces just the ranked placement counts and feeds the
    same modified rows to counted-ground-corner refinement.
    """
    image, parts, counts, camera = inputs.image, inputs.parts, inputs.counts, inputs.camera
    if count_transform is not None:
        counts = count_transform(image, parts, deepcopy(counts), camera)
    place = visible_placement.place if place is None else place
    cells, status = place(image, VERSION, parts, counts=counts, atlas_path=suite.ATLAS,
                          joint_foreground=True, camera=camera)
    ranked = set(cells)
    cells = ranked.copy()
    if inputs.inset_events:
        added, removed = visible_inset_cells.inset_edits(
            image.size, camera, inputs.measured_surfaces, inputs.inset_events, cells)
        cells = (cells | added) - removed
    joint = status.get('foreground_depth', {}).get('method') == 'joint'
    if cells:
        matched = set(pavement_opening.detect(image, inputs.pre_inset, status))
        if matched:
            regions = [part for part in inputs.pre_inset if part['id'] in matched]
            cells -= window_obstructions(image.size, status, regions, cells)
            cells |= window_wall_recheck.recover(image, parts, status, matched, cells)
        elif joint and not (inputs.inset_events or inputs.phase_events or inputs.confirmed_outlines):
            source_camera = camera if camera is not None else dict(
                P=status['P'], scale_status=status['camera_status'], reason=status['reason'])
            additions, _ = smooth_extent_placement.recover(image, VERSION, parts, cells, source_camera, suite.ATLAS)
            cells |= additions
            if inputs.common_wall_regions:
                cells |= smooth_extent_placement.recover_from_regions(
                    image, VERSION, parts, cells, source_camera, suite.ATLAS, inputs.common_wall_regions)
            added, removed, _ = counted_ground_corner.edits(image, parts, counts, status, cells)
            cells = (cells - set(removed)) | set(added)
    if joint:
        background = exterior_foreground.exterior(image)
        removed = exterior_foreground.misplaced(image, inputs.roof_surfaces, status, cells, background)
        added = set()
        if inputs.exterior_events:
            added, shelf = exterior_foreground.roof_edits(
                image, inputs.scene_context, inputs.roof_surfaces, inputs.exterior_events, status, cells, background)
            removed |= shelf
        cells = (cells - removed) | added
    status.update(placed_blocks=len(cells), final_refinement=dict(
        before=len(ranked), added=len(cells - ranked), removed=len(ranked - cells), after=len(cells)))
    blocks, assignment = placed_materials.assign(image, status, cells, _export_atlas())
    pack, _ = typed.typed_pack(blocks)
    return ReplayResult(pack, assignment, status)


def fixed_reference(entry):
    """Load grader-only native labels, face IDs, and the historical registration."""
    job = _jobs()[entry['id']]
    if job[1]['files']['rgb.png'] != entry['rgb_sha256']:
        raise ValueError(f'Wrong fixed reference hash: {entry["id"]}')
    labels, face_ids = visible_faces(*job[:2])
    expected, _, transform, _ = ref_inputs(job)
    if expected != {tuple(block['xyz_world']) for block in labels['blocks']}:
        raise ValueError(f'Fixed visible reference coordinates disagree: {entry["id"]}')
    return labels, face_ids, transform


def score_result(entry, result, *, reference=None):
    """Return the common score row and every aligned block ID from the decoded pack."""
    labels, face_ids, transform = fixed_reference(entry) if reference is None else reference
    truth, by_kind = {}, {}
    for block in labels['blocks']:
        point = tuple(block['xyz_world'])
        truth[point] = block['block_id']
        by_kind.setdefault(block.get('geometry_kind', 'full_cube'), set()).add(point)
    optional = horizontal_slivers(labels, face_ids)
    files = _jobs()[entry['id']][1]['files']
    hashes = {name: value for name, value in files.items()
              if name in ('camera.json', 'labels.json', 'labels.npz', 'world-readback.json')}
    reference = FrameReference(truth, optional, labels['camera'], by_kind, hashes)
    local = decoded_pack(BytesIO(result.pack))
    if local.keys() != result.assignment.keys():
        raise ValueError(f'Assignment evidence differs from decoded fixed pack: {entry["id"]}')
    registered = typed.aligned_sites(list(local), transform, (0, 0, 0))
    translation, overlap = best_translation_fast(registered, truth)
    points = [tuple(a + b for a, b in zip(point, translation, strict=True)) for point in registered]

    def grade(destinations):
        names = dict(zip(destinations, local.values(), strict=True))
        assigned = {point for point, cell in zip(destinations, local, strict=True)
                    if result.assignment[cell] != 'fallback'}
        return typed.geometry_score(names, truth, optional), typed.score(names, truth, optional, assigned), names

    geometry, scores, predictions = grade(points)
    registered_geometry, registered_scores, _ = grade(registered)
    if overlap != geometry['strict']['correct']:
        raise ValueError(f'Fixed placement translation differs from the exact optimum: {entry["id"]}')
    case = {key: value for key, value in entry.items() if key not in ('rgb', 'rgb_sha256')}
    return dict(case, source_sha256=entry['rgb_sha256'], reference_sha256=hashes,
                pack_sha256=sha256(result.pack).hexdigest(), geometry=geometry, scores=scores,
                targets=target_breakdown(predictions, reference), translation=list(translation), exported=len(local),
                visible_truth=len(truth), required_truth=len(truth) - len(optional),
                registration=dict(transform=transform.tolist() if transform is not None else None,
                                  geometry=registered_geometry, scores=registered_scores),
                provenance='fixed pre-inset replay', **camera_fields(result.status)), predictions


def _verify(task):
    entry, output = task
    case = entry['id']
    _, rows, path = _release()
    archived = rows[case]
    with ZipFile(path) as archive:
        released_pack = archive.read(f'{case}/reconstruction.zip')
    if sha256(released_pack).hexdigest() != archived['pack_sha256']:
        raise ValueError(f'Released pack fingerprint differs: {case}')
    inputs = inference_inputs(entry)
    result = replay_placement(inputs)
    row, _ = score_result(entry, result)
    before, after = map(decoded_pack, (BytesIO(released_pack), BytesIO(result.pack)))
    report = dict(row, released_pack_sha256=archived['pack_sha256'], byte_match=released_pack == result.pack,
                  typed_match=before == after, geometry_match=before.keys() == after.keys(),
                  added=sorted(after.keys() - before.keys()), removed=sorted(before.keys() - after.keys()),
                  changed_ids=sorted(p for p in before.keys() & after.keys() if before[p] != after[p]))
    if report['typed_match'] and (row['geometry'] != archived['geometry'] or row['scores'] != archived['result']):
        raise ValueError(f'Fixed replay scoring differs for an identical pack: {case}')
    save(output / 'cases' / (case + '.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    frames = list(fixed_frames())
    selected = [frame for frame in frames if args.ids is None or frame['id'] in args.ids]
    if args.ids and {case['id'] for case in selected} != set(args.ids):
        raise ValueError('Requested IDs outside the fixed suite')
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_verify, (frame, args.output)) for frame in selected]
        for future in as_completed(futures):
            row = future.result()
            print(row['id'], row['byte_match'], row['typed_match'], row['geometry_match'], flush=True)
    cases = [read(args.output / 'cases' / (frame['id'] + '.json')) for frame in selected]
    release, _, _ = _release()
    save(args.output / 'summary.json', dict(revision=MEASURED_REVISION, version=VERSION, views=len(cases),
        released_cases_sha256=release['cases_sha256'], released_exports_sha256=release['exports_sha256'],
        byte_matches=sum(row['byte_match'] for row in cases), typed_matches=sum(row['typed_match'] for row in cases),
        geometry_matches=sum(row['geometry_match'] for row in cases),
        differing_cases=[row['id'] for row in cases if not row['typed_match']], cases=cases))


if __name__ == '__main__':
    main()
