"""Check proposed ground-through-opening masks against decoded whole-structure builds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.edge_vote_outline_eval import baseline_pack, recount
from benchmark.inset_default_verification import local_release
from benchmark.mask_blocks import _write_read, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import COHORTS, atlas, ref_inputs
from benchmark.wall_texture_evaluate import metrics
from engine import inset_returns, surface_model, texture_guided_count, visible_placement
from engine.pillar_seams import native_mask
from engine.surface_perimeter import with_outline
from engine.visible_inset_cells import inset_edits
from training.visible_labels import visible_labels


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/pavement-eval'
REPORT = ROOT / 'benchmark/reviews/pavement-opening-round-01'
FIELDS = ('correct', 'missed', 'extras', 'hidden_extras')


def visible_faces(split, frame):
    """Native visible face IDs for facade-priority and per-surface grading."""
    files = frame['files']
    if split in ('train', 'test', 'validation', 'regression'):
        folder = suite.framed.TRAINING_DATA / 'data' / split / frame['directory']
        labels = read(checked(folder / 'labels.json', files['labels.json']))
        with np.load(checked(folder / 'labels.npz', files['labels.npz'])) as arrays:
            return labels, arrays['face_id'].copy()
    if split == 'native':
        rgb = suite.framed.native.rgb_path(frame, suite.framed.native.BUILDS)
    else:
        root = {'new': suite.framed.NEW_INPUTS, 'shelter': suite.SHELTER}.get(split, suite.framed.transfer.BUILDS)
        rgb = root / frame['directory'] / 'captures' / (frame['source_id'] + '.png')
    camera = read(checked(rgb.with_suffix('.json'), files['camera.json']))
    blocks = read(checked(rgb.with_name(rgb.stem + '-blocks.json'), files['world-readback.json']))
    labels, ids, _ = visible_labels(camera, {tuple(b[k] for k in 'xyz'): b for b in blocks}, (0, 0, 0))
    return labels, ids


def priority(placed, truth, optional):
    return dict(correct=len((truth - optional) & placed), missed=len((truth - optional) - placed),
                extras=len(placed - truth))


def evaluate_baseline(job):
    split, frame, _, control = job
    key = frame['id']
    pack_sha, cells = baseline_pack(job)
    truth, occupied, transform, hashes = ref_inputs(job)
    world = registered(cells, transform)
    scores = placement_scores(world, world, dict.fromkeys(truth), occupied)
    labels, ids = visible_faces(split, frame)
    assert truth == {tuple(b['xyz_world']) for b in labels['blocks']}
    optional = horizontal_slivers(labels, ids)
    placed = suite.framed.spacing.shift(world, scores['translation'])
    counts = {k: scores['relative_geometry'][k] for k in FIELDS}
    expected = local_release()[0][key]
    released = expected['methods'].get('clear_front', expected['baseline'])
    assert all(counts[k] == released[k] for k in FIELDS), key
    row = dict(id=key, split=split, rgb_sha256=frame['files']['rgb.png'], source_inputs=hashes,
               capture_sha256=frame['files']['camera.json'], readback_sha256=frame['files']['world-readback.json'],
               pack_sha256=pack_sha, coordinates_sha256=coordinate_hash(cells), strict=counts,
               translation=scores['translation'], facade=priority(placed, truth, optional),
               registered={k: scores['metrics'][k] for k in FIELDS})
    if key in suite.framed.roof_specs():
        row['canopy'] = suite.framed.roof(frame, world, truth)
    save(RUN / 'baseline' / (key + '.json'), row)
    return key


def coordinate_hash(points):
    return sha256(json.dumps(sorted(points), separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_rule(path):
    spec = spec_from_file_location('engine.pavement_opening', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detect


def build(image, parts, saved_counts, camera):
    revised, events, _, outlines, _ = inset_returns.detect(image, parts, '26.3', camera=camera)
    revised = [with_outline(p) for p in revised]
    material = atlas() if 'atlas_phase' in saved_counts else None
    counts = recount(image, revised, material)
    place_parts, place_counts = revised, counts
    if events or outlines:
        place_parts = [with_outline(p) for p in parts]
        place_counts = surface_model._inset_layout_counts(image, '26.3', place_parts, counts,
            {e['host'] for e in events + outlines}, texture_guided_count.METHOD, material)
    cells, _ = visible_placement.place(image, '26.3', place_parts, counts=place_counts, camera=camera,
                                        atlas_path=suite.ATLAS)
    if events:
        added, removed = inset_edits(image.size, camera, revised, events, cells)
        cells = (set(cells) | added) - removed
    dimensions = {r['id']: r['dimensions']['blocks'] for r in counts['surfaces']}
    return set(cells), revised, dimensions


def grade_changed(job, base, candidate, orig, revised, inputs, removed):
    split, frame, _, _ = job
    truth, occupied, transform, _ = ref_inputs(job)
    labels, ids = visible_faces(split, frame)
    optional = horizontal_slivers(labels, ids)
    raw = registered(candidate, transform)
    old = registered(base, transform)
    scores = placement_scores(raw, old, dict.fromkeys(truth), occupied)
    placed = suite.framed.spacing.shift(raw, scores['translation'])
    previous = suite.framed.spacing.shift(old, scores['baseline_translation'])
    target = truth - optional
    delta = dict(required_gained=len((placed - previous) & target), required_lost=len((previous - placed) & target))
    heights = [b[1] for b in truth]
    top = {p for p in truth if p[1] == max(heights)}
    delta.update(top_course_gained=len((placed - previous) & top), top_course_lost=len((previous - placed) & top))
    lut = np.array([0] + [f['surface_id'] for f in labels['faces']])
    truth_masks = lut[ids]
    size = ids.shape[::-1]
    pre = metrics(orig, truth_masks, size)
    post = metrics(revised, truth_masks, size)
    drops = []
    for part in inputs:
        if part['id'] in removed:
            mask = native_mask(part, size)
            foreground = int(np.sum(mask & (truth_masks != 0)))
            drops.append(dict(id=part['id'], pixels=int(mask.sum()), foreground=foreground, background=int(mask.sum())-foreground))
    surface = dict(removed=drops, matched_gained=sorted(set(post['matched']) - set(pre['matched'])),
                   matched_lost=sorted(set(pre['matched']) - set(post['matched'])),
                   unpaired_change=post['unmatched_predictions'] - pre['unmatched_predictions'],
                   paired_pixels_change=post['paired_pixels'] - pre['paired_pixels'])
    out = dict(strict={k:scores['relative_geometry'][k] for k in FIELDS}, facade=priority(placed, truth, optional),
               registered={k:scores['metrics'][k] for k in FIELDS}, translation=scores['translation'],
               delta=dict(**delta, **{k:scores['relative_geometry'][k] for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')}),
               surfaces=surface)
    if frame['id'] in suite.framed.roof_specs():
        out['canopy'] = suite.framed.roof(frame, raw, truth)
    return out


def evaluate_candidate(task):
    job, path = task
    split, frame, _, _ = job
    key = frame['id']
    image, parts, counts, camera = suite.ordinary_inputs(*job)
    start = perf_counter()
    removed = set(load_rule(path)(image, parts, camera))
    duration = perf_counter() - start
    assert removed <= {p['id'] for p in parts}
    baseline = read(RUN / 'baseline' / (key + '.json'))
    row = dict(id=key, split=split, baseline=baseline, removed=sorted(removed), selection_seconds=round(duration, 4))
    if removed:
        frozen, cells = baseline_pack(job)
        original, pre, dims = build(image, parts, counts, camera)
        assert frozen == baseline['pack_sha256'] and original == cells, (key, 'control did not reproduce', sorted(cells ^ original))
        selected = [p for p in parts if p['id'] not in removed]
        trial, post, new_dims = build(image, selected, counts, camera)
        saved = RUN/'packs'/key/'reconstruction.zip'
        saved.parent.mkdir(parents=True, exist_ok=True)
        trial = _write_read(saved, trial)
        row['trial'] = grade_changed(job, original, trial, pre, post, parts, removed)
        row['trial'].update(verified_control=True, pack_sha256=digest(saved), coordinates_sha256=coordinate_hash(trial),
                            changed_dimensions={k:[dims[k],new_dims[k]] for k in sorted(dims.keys() & new_dims.keys()) if dims[k]!=new_dims[k]})
    save(RUN / 'candidate' / (key + '.json'), row)
    return key, sorted(removed)


def pooled(rows, method, protocol):
    totals = Counter()
    for row in rows:
        record = row.get(method, row['baseline'])
        totals.update(record[protocol])
    totals = dict(totals)
    totals['f1'] = 2*totals['correct'] / (2*totals['correct'] + totals['missed'] + totals['extras'])
    return totals


def archive_changed(bundle, row, job):
    """Keep both actual packs and the precise source-only stage inputs for a positive case."""
    key = row['id']
    split, _, _, _ = job
    release, exports = local_release()
    if release[key]['methods'].get('clear_front'):
        blob = exports.read(f'clear_front/{split}/{key}/reconstruction.zip')
    elif split == 'shelter':
        with ZipFile(suite.ARCHIVE) as src:
            blob = src.read(f'production/transfer/{key}/reconstruction.zip')
    else:
        blob = (suite.CONTROL/split/key/'reconstruction.zip').read_bytes()
    assert sha256(blob).hexdigest() == row['baseline']['pack_sha256']
    bundle.writestr(f'baseline/{key}/reconstruction.zip', blob)
    bundle.write(checked(RUN/'packs'/key/'reconstruction.zip', row['trial']['pack_sha256']),
                 f'candidate/{key}/reconstruction.zip')
    _, parts, _, camera = suite.ordinary_inputs(*job)
    bundle.writestr(f'inputs/{key}.json', json.dumps(dict(planes=parts, camera=camera), separators=(',', ':')))


def summarize(rows, path, jobs):
    REPORT.mkdir(parents=True, exist_ok=True)
    save(REPORT / 'cases.json', rows)
    archive = REPORT / 'changed-packs.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for row, job in zip(rows, jobs, strict=True):
            if row['removed']:
                archive_changed(bundle, row, job)
    cohorts = {}
    for name, splits in COHORTS.items():
        selection = [r for r in rows if r['split'] in splits]
        changed = [r for r in selection if r['removed']]
        delta = Counter()
        for row in changed:
            delta.update(row['trial']['delta'])
        cohorts[name] = dict(frames=len(selection), triggered=len(changed), changes={k:dict((m,pooled(selection,m,k)) for m in ('baseline','trial'))
            for k in ('strict','facade','registered')}, delta=dict(delta))
    historical_target = read(suite.ROOT/'benchmark/reviews/shared-face-round-01/error-budget.json')['totals']['strict']
    assert cohorts['fixed']['changes']['strict']['baseline']['correct'] == historical_target['correct']
    summary = dict(baseline='a6e0195', rule_sha256=digest(path), frames=len(rows), cases_sha256=digest(REPORT/'cases.json'),
                   exports_sha256=digest(archive), cohorts=cohorts)
    save(REPORT/'summary.json', summary)
    for k, group in cohorts.items():
        print(k,group['frames'],group['triggered'],group['changes']['strict'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['baseline','candidate','report'])
    p.add_argument('--rule', type=Path, default=ROOT/'engine/pavement_opening.py')
    p.add_argument('--workers', type=int, default=4)
    args=p.parse_args()
    jobs=list(suite.jobs([*suite.framed.SIZES, 'shelter']))
    assert len(jobs)==204 and len({j[1]['id'] for j in jobs})==204
    if args.stage == 'report':
        rows=[read(RUN/'candidate'/(j[1]['id']+'.json')) for j in jobs]
        summarize(rows, args.rule, jobs)
        return
    (RUN/args.stage).mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage == 'baseline':
            results=pool.map(evaluate_baseline,jobs)
        else:
            results=pool.map(evaluate_candidate,[(j,args.rule.resolve()) for j in jobs])
        for i,result in enumerate(results,1):
            if i%25==0 or (args.stage=='candidate' and result[1]):
                print(i, result, flush=True)


if __name__ == '__main__':
    main()
