"""Test whether a protected-edge proposal can safely split one zoom-detected surface."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark import contained_splits
from benchmark.afterpass_round import COHORTS, picture_table
from benchmark.boundary_guard_round import DEPTH
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.crop_split_round import PREVIOUS, audit_partition, union_at_scoring_resolution
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL, footprint, measure
from benchmark.normal_round import groups
from benchmark.plane_acceptance_round import PLANES
from benchmark.split_round import save_and_score
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from runtime.mode import image_only, mark_image_only


BASE = PREVIOUS.parent / 'crop-split-01'
GUARD = PREVIOUS / 'candidate-runs/guard'
NAMES = {'current': 'Split-only zoom', 'contained': 'One-wall split with edge confirmation'}
RULE = ('Keep every split-only zoom match, gain at least one on the six regression buildings, '
        'and preserve or improve extra guesses, opening coverage and correct block dimensions.')


def overlap(mask, other):
    union = np.count_nonzero(mask | other)
    return float(np.count_nonzero(mask & other) / union) if union else 0.


def diagnose(reference, protected, parents, parts, decisions, previous, result):
    """Explain the earlier methods' disagreements only after counting and scoring predictions."""
    split_ids = set(result['scopes']['current']['matched'])
    guard_ids = set(previous['scopes']['guard']['matched'])
    if split_ids == guard_ids:
        return []
    ids = reference['id_map']
    def masks(regions):
        return {p['id']: footprint(p, ids.shape, reference['image_size'], True) for p in regions}
    guard_masks, original = masks(protected['guard']), masks(protected['current'])
    parent_masks, new_masks = masks(parents), masks(parts)
    _, pairs = measure(protected['guard'], reference, use_mask=True)
    matched = {rid: protected['guard'][index]['id'] for index, rid in pairs}
    if set(matched) != guard_ids:
        raise ValueError('Earlier protected-edge reference matches failed to reproduce')
    by_id = {d['proposal']: d for d in decisions}
    diagnostics = []
    for rid in sorted(split_ids ^ guard_ids):
        target = ids == rid
        def best(pool):
            return max((dict(proposal=name, overlap=overlap(mask, target)) for name, mask in pool.items()),
                       key=lambda rec: rec['overlap'], default=None)
        name = matched.get(rid)
        related = []
        if name in by_id:
            related.append(dict(decision=by_id[name], relationship='direct protected-edge match'))
        elif name in original:
            # The earlier agreement sometimes recovered a wall indirectly. Identify which suggestions
            # claimed the part removed from its old mask, including close proposals failing this test.
            removed = original[name] & ~guard_masks[name]
            for pid, decision in by_id.items():
                cells = int(np.count_nonzero(removed & guard_masks[pid]))
                if cells:
                    related.append(dict(decision=decision, relationship='covered pixels trimmed from earlier parent',
                                        samples=cells))
        diagnostics.append(dict(reference=rid, matched_zoom=rid in split_ids, matched_guard=rid in guard_ids,
                                matched_contained=rid in result['scopes']['contained']['matched'],
                                guard_match=name, closest_guard=best(guard_masks), closest_zoom=best(parent_masks),
                                closest_contained=best(new_masks), supporting_decisions=related))
    return diagnostics


def evaluate(job):
    capture, row, prior, hashes, out = job
    with Image.open(checked(capture.path('image'), row['sha256'])) as source:
        image = source.convert('RGB')
    parents = read_gzip(checked(BASE / row['predictions'], hashes['baseline']))['split']
    protected = read_gzip(checked(GUARD / prior['predictions'], hashes['guard']))
    evidence_path = 'fields/' + row['sha256'] + '.npz'
    with np.load(checked(DEPTH / evidence_path, hashes['depth']), allow_pickle=False) as data:
        depth = data['inverse_depth']
    with np.load(checked(PLANES / evidence_path, hashes['planes']), allow_pickle=False) as data:
        planes = {name: data[name] for name in ('relaxed', 'relaxed_normals')}
    start = perf_counter()
    parts, decisions = contained_splits.predict(image, parents, protected['guard'], depth, planes)
    elapsed = perf_counter() - start
    predictions = dict(current=parents, contained=parts)
    result = save_and_score(capture, row, predictions, dict(current=0., contained=elapsed), out, 'split')
    score_dimensions(capture, row, predictions, result, baseline_method='split')
    result['decisions'] = decisions
    splits = [d for d in decisions if d['reason'] == 'accepted']
    result['parent_partitions'] = audit_partition(parents, parts, splits)
    _, reference = recorded(capture)
    before = union_at_scoring_resolution(parents, reference)
    after = union_at_scoring_resolution(parts, reference)
    if not np.array_equal(before, after):
        raise ValueError(f"Covered area changed on {row['key']}")
    result['footprint_changed_samples'] = 0
    result['reference_diagnostics'] = diagnose(reference, protected, parents, parts, decisions, prior, result)
    return result


@image_only()
def run(out, workers):
    base = read(BASE / 'manifest.json')
    old = read(checked(PREVIOUS / 'manifest.json', base['source_manifests'][
        (PREVIOUS / 'manifest.json').relative_to(ROOT).as_posix()]))
    guard = read(checked(GUARD / 'manifest.json', old['artifacts']['candidate-runs/guard/manifest.json']))
    depth = read(checked(DEPTH / 'manifest.json', guard['upstream_manifests'][DEPTH.name]))
    planes = read(checked(PLANES / 'manifest.json', guard['inherited_model_manifests']['planarity']))
    rows = read(checked(BASE / 'cases.json', base['artifacts']['cases.json']))
    prior_rows = read(checked(GUARD / 'cases.json', guard['artifacts']['cases.json']))
    prior = {r['key']: r for r in prior_rows}
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}': c for group in cohorts.values() for c in group}
    roster = {r['key']: r['sha256'] for r in rows}
    checked(ARCHIVE, base['reference_sha256'])
    if (len(rows) != 56 or len(prior_rows) != 56 or roster.keys() != captures.keys()
            or len(set(roster.values())) != 52 or {k: r['sha256'] for k, r in prior.items()} != roster
            or depth['frames'] != roster or fixtures != base['fixtures'] or fixtures != guard['fixtures']
            or fixtures != depth['fixtures'] or fixtures != planes['fixtures']
            or guard['reference_sha256'] != base['reference_sha256'] or guard['protocol'] != PROTOCOL
            or base['protocol'] != PROTOCOL or guard['frames'] != base['frames']):
        raise ValueError('Frozen frames, fixtures or evaluator differ')
    dev = [r for r in rows if r['cohort'] == 'development']
    if len(dev) != 20 or {r['layout'] for r in dev} != {'wing', 'shopfront'}:
        raise ValueError('Development split has changed')
    out.mkdir(parents=True, exist_ok=False)
    code = revision()
    freeze = dict(selection='One fixed gate before suite scoring', rule=RULE, settings=contained_splits.SETTINGS,
                  development_frames=[r['key'] for r in dev])
    save(out / 'development-freeze.json', freeze)
    jobs = []
    for row in rows:
        earlier = prior[row['key']]
        field = 'fields/' + row['sha256'] + '.npz'
        hashes = dict(baseline=base['artifacts'][row['predictions']], guard=guard['artifacts'][earlier['predictions']],
                      planes=planes['artifacts'][field], depth=depth['artifacts'][field])
        jobs.append((captures[row['key']], row, earlier, hashes, out))
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        results = list(pool.map(evaluate, jobs))
    scores = {'all': summary(results, NAMES)}
    for group in (*COHORTS, *sorted({r['layout'] for r in results})):
        scores[group] = summary([r for r in results if group in (r['cohort'], r['layout'])], NAMES)
    save(out / 'cases.json', results)
    save(out / 'scores.json', scores)
    (out / 'pictures.md').write_text(picture_table(results, NAMES, NAMES).replace(
        'starts from the same default', 'starts from split-only zoom'), encoding='utf8')
    print({m: (s['matched'], s['lost'], s['unpaired'], s['opening_filled'], s['sizes']['correct'])
           for m, s in scores['all'].items()}, flush=True)
    from benchmark.guard_zoom_sheet import render_contained
    render_contained(out, results, captures, scores)
    decisions = Counter(d['reason'] for row in results for d in row['decisions'])
    control, candidate = scores['all']['current'], scores['all']['contained']
    acceptance = dict(kept_every_match=candidate['lost'] == 0,
                      gained_regression_match=scores['previously_reviewed_regression']['contained']['gained'] > 0,
                      extra_guesses=candidate['unpaired'] <= control['unpaired'],
                      opening_coverage=candidate['opening_filled'] <= control['opening_filled'],
                      block_sizes=candidate['sizes']['correct'] >= control['sizes']['correct'])
    source_checkpoint = ROOT / 'benchmark/reviews/crop-split-entry-01/workshop-proof.json'
    save(out / 'manifest.json', dict(
        run_id=out.name, code=code, version=suite['version'], fixtures=fixtures, frames=base['frames'],
        reference_sha256=base['reference_sha256'], protocol=PROTOCOL, capture=base['capture'], splits=base['splits'],
        methods=NAMES, baseline='plane-recovery-crop-splits', settings=freeze, selection='development-freeze.json',
        acceptance_checks=acceptance, meets_promotion_rule=all(acceptance.values()),
        stages=dict(control='frozen split-only zoom masks', proposals='frozen two-model protected-edge suggestions',
                    refinement='one existing wall with cross-edge depth or normal evidence', counter='masked-repeat'),
        source_manifests={p.relative_to(ROOT).as_posix(): digest(p) for p in (
            BASE/'manifest.json', PREVIOUS/'manifest.json', GUARD/'manifest.json', DEPTH/'manifest.json',
            PLANES/'manifest.json', source_checkpoint)},
        checkpoints=read(source_checkpoint)['model_sha256'], plane_model={k: planes[k] for k in (
            'model', 'model_revision', 'checkpoint_sha256', 'upstream', 'moge', 'utils3d')},
        new_inference_s=0, source_model_timing=base['source_model_timing'],
        timing='All shared neural inference reused. Per-image cached_processing_s times only the new gate.',
        cached_gate_seconds=scores['all']['contained']['cached_extra_seconds'], decision_counts=dict(decisions),
        scores='scores.json', per_picture='pictures.md', sheet='comparison.jpg',
        reference_diagnostics='cases.json: each earlier disagreement traced after prediction',
        reproduce=r'.\Invoke-SurfaceRound.ps1 -Round contained',
        artifacts={p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs' / f'contained-{uuid4().hex[:12]}')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('At least one worker is required.')
    run(args.out.resolve(), args.workers)

