"""Trace shipped block misses to signed source-image footprints and current mask/count evidence."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.inset_default_verification import local_release
from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_worst_stages import read_view
from benchmark.strip_visibility_audit import block_footprints
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.visible_error_budget import projection, stage_evidence, tally
from engine import inset_returns, surface_model, texture_guided_count
from engine.surface_perimeter import with_outline

REVIEW = suite.ROOT / 'benchmark/reviews'
LATEST = REVIEW / 'facade-inset-round-01/combined.json'
RANKING = REVIEW / 'facade-placement-01/worst/ranking.json'
STAGES = ('no_mask', 'mask_without_correct_axis', 'correct_axis_no_dimensions', 'dimensions_no_origin', 'located_grid')


def current_pack(job):
    split, frame, _, control = job
    rows, archive = local_release()
    row = rows[frame['id']]
    assert row['rgb_sha256'] == frame['files']['rgb.png']
    assert row['control_pack_sha256'] == control['pack_sha256']
    after = row['methods'].get('clear_front')
    if after is None:
        key = control['pack_sha256']
        packed = decoded_pack(checked(suite.CONTROL / split / frame['id'] / 'reconstruction.zip', key))
        return sorted(packed), key, control['translation'], control['relative_geometry']
    member = archive.read(f"clear_front/{split}/{frame['id']}/reconstruction.zip")
    assert sha256(member).hexdigest() == after['pack_sha256']
    return sorted(decoded_pack(BytesIO(member))), after['pack_sha256'], after['translation'], after


def current_stages(job, expected):
    image, base, counts, camera = suite.ordinary_inputs(*job)
    parts, strict, phase, outlines, _ = inset_returns.detect(image, base, '26.3', camera=camera)
    assert [len(x) for x in (strict, phase, outlines)] == [expected[k] for k in ('returns','texture','outlines')]
    if not (strict or phase or outlines):
        return image, parts, counts, camera
    atlas = None
    if 'atlas_phase' in counts:
        from engine.atlas_count import load_atlas
        atlas = load_atlas(suite.ATLAS, '26.3')
    parts = [with_outline(p) for p in parts]
    with TemporaryDirectory() as out:
        counts = surface_model._count_regions(image, parts, '26.3', texture_guided_count.METHOD, atlas, Path(out))
        surface_model._inset_layout_counts(image, '26.3', [with_outline(p) for p in base], counts,
            {e['host'] for e in strict + outlines}, texture_guided_count.METHOD, atlas)
    return image, parts, counts, camera


def rate(correct, missed, extras):
    return 2 * correct / (2 * correct + missed + extras)


def measure(task):
    job, expected, layout, shape = task
    split, frame, prior, control = job
    image, parts, counts, camera = current_stages(job, expected)
    local, pack_sha, shift, metric = current_pack(job)
    _, _, _, _, pose, labels, ids, truth, _, _, _ = read_view(suite.CONTROL, split, frame, prior, control)
    owner, axis, world_axis, world = projection(camera, pose, local, shift, ids.shape)
    placed = set(map(tuple,world))
    correct = len(placed & truth)
    missing = truth - placed
    extras = placed - truth
    assert (correct, len(missing), len(extras)) == (metric['correct'], metric['missed'], metric['extras'])
    owners, axes, footprint, bits, face_square, _ = block_footprints(labels, ids)
    n = len(footprint)
    source = ids.ravel()
    ref_block = owners[source]
    owner, axis = owner.ravel(), axis.ravel()
    hit = (owner >= 0) & (source > 0)
    same = hit & (world_axis[axis] == axes[source])
    known = np.zeros(hit.shape, bool)
    if len(world):
        block_correct = np.array([tuple(p) in truth for p in world], bool)
        known[hit] = block_correct[owner[hit]]
    covered, aligned, known_count = (tally(ref_block, votes, n) for votes in (hit,same,known))
    # Compare the signed displacement along the independently recorded outward normal.
    native_pixels = np.flatnonzero(same)
    xyz = np.array([[0,0,0]] + [b['xyz_world'] for b in labels['blocks']])
    normals = np.array([[0,0,0]] + [f['normal_world'] for f in labels['faces']])
    displacement = ((world[owner[native_pixels]] - xyz[ref_block[native_pixels]])
                    * normals[source[native_pixels]]).sum(axis=1)
    depth_signs = ('closer_plane','deeper_plane','same_plane')
    depth_votes = np.stack([tally(ref_block[native_pixels], s, n)
                            for s in (displacement>0, displacement<0, displacement==0)], axis=1)
    reference_axes = axes[ids]
    support = stage_evidence(parts, counts['surfaces'], image.size, camera, world_axis, reference_axes).ravel()
    stage_votes = np.array([tally(ref_block, support >= k, n) for k in range(1, 5)])
    stage = np.count_nonzero(2 * stage_votes >= footprint[None,:], axis=0)
    optional = ((bits == 1) | (bits == 2)) & (face_square < 4)
    groups = defaultdict(Counter)
    stages = defaultdict(Counter)
    covering = defaultdict(Counter)
    depth = defaultdict(Counter)
    faces = defaultdict(Counter)
    optional_correct = optional_missing = 0
    for block in labels['blocks']:
        b = block['id']
        coord = tuple(block['xyz_world'])
        if coord not in missing:
            optional_correct += int(optional[b])
            continue
        optional_missing += int(optional[b])
        kind = 'optional' if optional[b] else 'required'
        if 2 * covered[b] < footprint[b]:
            error = 'uncovered'
        elif 2 * covered[b] == footprint[b]:
            error = 'coverage_tie'
        elif 2 * aligned[b] > covered[b]:
            error = 'same_axis'
        elif 2 * aligned[b] < covered[b]:
            error = 'different_axis'
        else:
            error = 'axis_tie'
        groups[kind][error] += 1
        face = 'wall' if bits[b] & 4 else 'horizontal_only'
        faces[kind][face] += 1
        if error == 'uncovered':
            stages[kind][STAGES[stage[b]]] += 1
        if error in ('same_axis','different_axis'):
            who = 'extra' if known_count[b]*2 < covered[b] else 'correct_neighbor' if known_count[b]*2 > covered[b] else 'tie'
            covering[kind][error + '_' + who] += 1
        if error == 'same_axis':
            votes = depth_votes[b]
            label = depth_signs[votes.argmax()] if 2*votes.max() > votes.sum() else 'mixed'
            depth[kind][label] += 1
    on_build = np.bincount(owner[hit], minlength=len(world))
    all_hits = np.bincount(owner[owner>=0], minlength=len(world))
    extra_owners = Counter()
    for i, xyz in enumerate(world):
        if tuple(xyz) in extras:
            extra_owners['occluded' if all_hits[i]==0 else 'mostly_on_build' if on_build[i]*2>all_hits[i]
                         else 'mostly_on_background' if on_build[i]*2<all_hits[i] else 'half_on_build'] += 1
    strict = dict(correct=correct, missed=len(missing), extras=len(extras))
    required = dict(correct=correct-optional_correct, missed=len(missing)-optional_missing, extras=len(extras))
    return dict(id=frame['id'], split=split, layout=layout, shape_sha256=shape, family=control['family'], rgb_sha256=expected['rgb_sha256'],
                pack_sha256=pack_sha, translation=shift, hidden_extras=metric['hidden_extras'], strict=strict, required=required,
                optional_correct=optional_correct, optional_missed=optional_missing, missing_projection=groups,
                uncovered_stage=stages, missing_faces=faces, covering_predictions=covering, same_axis_depth=depth,
                extra_projection=extra_owners, masks=len(parts),
                complete_dimensions=sum(None not in r['dimensions']['blocks'] for r in counts['surfaces']),
                both_grid_origins=sum(None not in r['phase'] for r in counts['surfaces']))


def combine(rows):
    out = dict(views=len(rows))
    for key in ('strict','required'):
        totals = Counter()
        for row in rows:
            totals.update(row[key])
        out[key] = dict(totals, f1=rate(**totals))
    for key in ('hidden_extras','optional_correct','optional_missed'):
        out[key] = sum(r[key] for r in rows)
    for key in ('missing_projection','uncovered_stage','missing_faces','covering_predictions','same_axis_depth'):
        totals = defaultdict(Counter)
        for row in rows:
            for k, values in row[key].items():
                totals[k].update(values)
        out[key] = totals
    out['extra_projection'] = sum((Counter(r['extra_projection']) for r in rows), Counter())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REVIEW/'shared-face-round-01/error-budget.json')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    stages = {r['id']:r for r in read(LATEST)['rows']}
    layout = {k:(r['name'],r['shape_sha256']) for r in read(RANKING)['all_layouts'] for k in r['frame_ids']}
    tasks = [(j, stages[j[1]['id']], *layout[j[1]['id']]) for j in suite.jobs(suite.framed.EVALUATION)]
    assert len(tasks)==len(layout)==104
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(measure, tasks))
    totals = combine(rows)
    expected_strict = read(REVIEW/'facade-local-placement-01/summary.json')['methods']['clear_front']['fixed']
    expected_priority = read(REVIEW/'facade-local-placement-01/priority.json')['methods']['clear_front']
    for k in ('correct','missed','extras','f1'):
        assert totals['strict'][k]==expected_strict[k] and totals['required'][k]==expected_priority[k], k
    by_layout = defaultdict(list)
    by_name = defaultdict(list)
    for r in rows:
        by_layout[r['shape_sha256']].append(r)
        by_name[r['layout']].append(r)
    groups = [dict(name=v[0]['layout'],shape_sha256=k,**combine(v)) for k,v in by_layout.items()]
    groups.sort(key=lambda group:group['strict']['f1'])
    assert len(groups) == 40
    args.report.parent.mkdir(parents=True,exist_ok=True)
    save(args.report, dict(revision='abfc0be', split='fixed 104 whole-structure views', latest_masks_sha256=digest(LATEST),
        layout_ids_sha256=digest(RANKING), export_manifest_sha256=digest(REVIEW/'facade-local-placement-01/cases.json'),
        definitions=dict(uncovered='Fewer than half the reference block pixels hit any exported block.',
            same_axis='Most pixels covered and majority of covered pixels hit the reference face axis at another coordinate.',
            different_axis='Most pixels covered and majority of covered pixels hit another face axis.',
            uncovered_stage='Furthest mask/count stage agreeing on at least half the reference block pixels. These are associations, not proven causes.',
            optional='Horizontal-only block with no native 4x4 square on a visible face.'),
        totals=totals, layouts=groups, by_source_name={k:combine(v) for k,v in by_name.items()}, frames=rows))
    print(totals)


if __name__ == '__main__':
    main()

