"""Replay placement order checks against the frozen Java 26.3 release at 9e69d7c.

Run from a checkout with its benchmark fixtures installed. Only the selected
output directory receives new files. Reference coordinates enter after export.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'engine/surface_model.py').is_file())
sys.path.insert(0, str(ROOT))

from benchmark import counted_ground_corner_eval as corners, current_placement_viewer as viewer
from benchmark import exterior_foreground_release as foreground, pavement_opening_eval as scoring
from benchmark import smooth_joint_placement, smooth_local_release
from benchmark.edge_vote_outline_eval import recount
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import COHORTS, ref_inputs
from engine import counted_ground_corner, eave_pinholes, exterior_foreground, inset_returns
from engine import jagged_texture_join, smooth_atlas_outline, smooth_extent_placement, surface_model
from engine import texture_guided_count, visible_placement
from engine.surface_perimeter import with_outline
from engine.visible_ray_filter import fully_occluded_cells

SUITE = viewer.suite
METHODS = ('unblock', 'second_exterior', 'final_ray')


@lru_cache(maxsize=1)
def releases():
    folders = [viewer.RELEASE, viewer.ROOT / 'smooth-local-placement-01',
               viewer.ROOT / 'smooth-joint-placement-01', viewer.ROOT / 'window-wall-01',
               corners.OUT, foreground.OUT]
    summary, _, packs = viewer.read_releases(folders)
    cases = {r['id']: r for r in read(checked(foreground.OUT / 'cases.json', summary['cases_sha256']))}
    openings = {r['id']: r['openings'] for r in read(folders[3] / 'cases.json')}
    return cases, packs, foreground.controls()[-1], openings


def recheck_skipped(image, baseparts, basecounts, camera, cells, inset_stats):
    parts, strict, phase, outlines, _ = inset_returns.detect(image, baseparts, '26.3', camera=camera)
    assert [len(x) for x in (strict, phase, outlines)] == [inset_stats[k] for k in ('returns', 'texture', 'outlines')]
    parts, eaves = eave_pinholes.refine(image, parts)
    if not (strict or phase or outlines):
        parts, _, _ = jagged_texture_join.refine(image, parts, '26.3', camera=camera)
    parts, _, common = smooth_atlas_outline.with_common_wall(image, parts, '26.3', atlas_loader=viewer.calibrated_atlas)
    parts = smooth_atlas_outline.measurement_parts(parts)
    if eaves:
        parts = eave_pinholes.measurement_parts(parts)
    parts = [with_outline(p) for p in parts]
    atlas = viewer.calibrated_atlas() if 'atlas_phase' in basecounts else None
    counts = recount(image, parts, atlas)
    if strict or outlines:
        parts = [with_outline(p) for p in baseparts]
        counts = surface_model._inset_layout_counts(image, '26.3', parts, counts,
                    {e['host'] for e in strict + outlines}, texture_guided_count.METHOD, atlas)
    add, local = smooth_extent_placement.recover(image, '26.3', parts, cells, camera, SUITE.ATLAS)
    trial = cells | add
    if common:
        trial |= smooth_extent_placement.recover_from_regions(image, '26.3', parts, trial, camera, SUITE.ATLAS, common)
    _, status = visible_placement.place(image, '26.3', parts, counts=counts, camera=camera, atlas_path=SUITE.ATLAS)
    add, remove, anchors = counted_ground_corner.edits(image, parts, counts, status, trial)
    return (trial - remove) | add, dict(local=len(local), common=len(common), corner_edits=len(anchors))


def grade(task):
    job, output = task
    split, frame, _, control = job
    key = frame['id']
    cases, packs, stages, openings = releases()
    base = cases[key]
    pack_hash = base.get('pack_sha256', base['previous_pack_sha256'])
    assert base['source_hash'] == frame['files']['rgb.png'] == stages[key]['rgb_sha256']
    if key in packs:
        assert sha256(packs[key]).hexdigest() == pack_hash
        cells = set(decoded_pack(BytesIO(packs[key])))
    else:
        actual_hash, cells = smooth_joint_placement.old_pack(job)
        assert actual_hash == pack_hash
    image, parts, counts, camera = SUITE.ordinary_inputs(*job)
    inset = stages[key]
    gate = 'opening' if openings[key] else ('inset' if any(inset[k] for k in ('returns', 'texture', 'outlines')) else 'open')
    status = control['status']
    joint = status.get('foreground_depth', {}).get('method') == 'joint'
    row = dict(id=key, split=split, rgb_sha256=base['source_hash'], baseline_pack=pack_hash,
               baseline=base['result'], gate=gate, joint=joint)
    trials = dict.fromkeys(METHODS, cells)
    if gate != 'open' and cells and joint:
        trials['unblock'], row['screen'] = recheck_skipped(image, parts, counts, camera, cells, inset)
    if base.get('added') or base.get('removed'):
        # If the first pass did not edit a cell, repeating it sees the identical build.
        _, before, _, _ = viewer.pre_background_stages(job, inset)
        background = exterior_foreground.exterior(image)
        revised, _ = exterior_foreground.refine(image, before, camera, background)
        extra = exterior_foreground.misplaced(image, revised, status, cells, background)
        trials['second_exterior'] = cells - extra
        row['swept_again'] = True
    assert status['P'] is not None or not cells
    if cells:
        trials['final_ray'] = cells - fully_occluded_cells(image.size, status['P'], cells)

    for method, trial in trials.items():
        row[method] = base['result']
        if trial == cells:
            continue
        path = output / 'packs' / method / (key + '.zip')
        path.parent.mkdir(parents=True, exist_ok=True)
        packed = _write_read(path, trial)
        assert set(packed) == trial
        truth, occupied, transform, _ = ref_inputs(job)
        optional = horizontal_slivers(*scoring.visible_faces(split, frame))
        current, prior = registered(packed, transform), registered(cells, transform)
        scores = placement_scores(current, prior, dict.fromkeys(truth), occupied)
        shift = SUITE.framed.spacing.shift
        assert scoring.priority(shift(prior, scores['baseline_translation']), truth, optional) == base['result']['facade']
        relative = scores['relative_geometry']
        row[method] = dict(strict={k: relative[k] for k in scoring.FIELDS},
            registered={k: scores['metrics'][k] for k in scoring.FIELDS},
            facade=scoring.priority(shift(current, scores['translation']), truth, optional), translation=scores['translation'])
        row[method + '_changes'] = dict(added=sorted(trial-cells), removed=sorted(cells-trial),
            gained=relative['gained'], lost=relative['lost'], extra_added=relative['extra_added'],
            extra_removed=relative['extra_removed'])
    return row


def summary(rows):
    assert len(rows) == len({r['id'] for r in rows}) == 204
    assert smooth_local_release.summarize(rows, 'baseline') == read(foreground.OUT / 'summary.json')['cohorts']
    result = dict(revision='9e69d7c', source_images=len(rows),
                  unique_rgb_sha256=len({r['rgb_sha256'] for r in rows}), cohorts={})
    for group, cases in [('fixed', [r for r in rows if r['split'] in COHORTS['fixed']]),
                         ('other', [r for r in rows if r['split'] not in COHORTS['fixed']])]:
        data = dict(views=len(cases), joint=sum(r['joint'] for r in cases),
                    gates=dict(Counter(r['gate'] for r in cases)),
                    skipped_screen={k: sum(r.get('screen', {}).get(k,0) for r in cases)
                                    for k in ('local','common','corner_edits')},
                    exterior_rechecks=sum(r.get('swept_again', False) for r in cases))
        for method in ('baseline', *METHODS):
            metrics = {}
            for kind in ('strict','facade'):
                pooled = Counter()
                for row in cases:
                    pooled.update({k:row[method][kind][k] for k in ('correct','missed','extras')})
                f1 = 2 * pooled['correct'] / (2 * pooled['correct'] + pooled['missed'] + pooled['extras'])
                metrics[kind] = dict(pooled, f1=f1)
            data[method]=metrics
            if method != 'baseline':
                data[method]['changes'] = [{**r[method+'_changes'], 'id':r['id']} for r in cases if method+'_changes' in r]
        result['cohorts'][group]=data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT / 'runs' / 'placement-order-review')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    output=args.out.resolve()
    output.mkdir(parents=True,exist_ok=True)
    jobs=list(SUITE.jobs([*SUITE.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(grade, [(job,output) for job in jobs]))
    save(output/'cases.json',rows)
    evidence=summary(rows)
    save(output/'order-checks.json',evidence)
    print('Scored',len(rows),'views; summary in',output/'order-checks.json')


if __name__=='__main__':
    main()
