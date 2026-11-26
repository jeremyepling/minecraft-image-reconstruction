"""Fit, apply and score a small CPU chooser for the Block Counter's proposed grids."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.camera_round import digest, read, save


DEV = {'shopfront', 'wing'}
FEATURES = (
    'ordinary_choice', 'one_axis_transferred', 'repeat_weak', 'repeat_strong',
    'repeat_gap_to_axis_best', 'texel_weak', 'texel_strong', 'log_observed_rows',
    'aspect_known', 'aspect_log_error', 'mask_fill', 'log_smaller_pixels_per_block',
    'log_larger_pixels_per_block',
)
RIDGE = 0.05
# Choose a guard from these fixed options with out-of-building predictions.
THRESHOLDS = (0.35, 0.5, 0.65, 0.8)
MARGINS = (0.025, 0.1)


def features(surface):
    """Only these fields from image-derived candidates enter the model."""
    candidates = surface['candidates']
    if not candidates:
        return np.empty((0, len(FEATURES)))
    evidence = [c['features'] for c in candidates]
    axes = ('u', 'v')
    best = {a: max(f['agreement_'+a] for f in evidence) for a in axes}
    rows = []
    for i, (candidate, f) in enumerate(zip(candidates, evidence)):
        measured = [a for a in axes if not f['transfer_'+a]]
        if not measured:
            raise ValueError('A complete grid must have at least one measured axis')
        agreements = [f['agreement_'+a] for a in measured]
        native = [f['native_'+a] for a in measured]
        pixel_pitch = np.array([f['chart_width'], f['chart_height']]) / candidate['counts']
        rows.append([
            i == surface['baseline_index'], len(measured) == 1, min(agreements), max(agreements),
            max(best[a] - f['agreement_'+a] for a in measured), min(native), max(native),
            np.log1p(min(f['rows_'+a] for a in measured)), f['aspect_known'],
            min(1.5, abs(f['aspect_error'])) if f['aspect_known'] else 0., f['fill'],
            np.log1p(min(pixel_pitch)), np.log1p(max(pixel_pitch)),
        ])
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite image evidence')
    return matrix


def aligned(predictions, targets):
    """Check identity before pairing candidate lists with scorer-only labels."""
    for frame, truth in zip(predictions, targets, strict=True):
        for field in ('key', 'sha256', 'layout', 'cohort'):
            if frame[field] != truth[field]:
                raise ValueError(f'Frame {field} mismatch in {frame["key"]}')
        for region, answer in zip(frame['surfaces'], truth['surfaces'], strict=True):
            if region['id'] != answer['id']:
                raise ValueError(f'Proposal mismatch in {frame["key"]}')
            yield region, answer


def fit_weights(predictions, targets):
    xs, ys, ws = [], [], []
    for surface, target in aligned(predictions, targets):
        x = features(surface)
        if not len(x):
            continue
        pairs = [tuple(c['blocks']) for c in surface['candidates']]
        occurrences = Counter(pairs)
        # A surface has weight one and each distinct dimension pair has an equal
        # share. A wall with many duplicate harmonic peaks cannot dominate.
        ws.extend(1. / (len(occurrences) * occurrences[p]) for p in pairs)
        ys.extend(list(p) == target['expected'] for p in pairs)
        xs.append(x)
    if not xs or len(set(ys)) != 2:
        raise ValueError('Training needs both exact and incorrect grids')
    x, y, weight = np.vstack(xs), np.asarray(ys, dtype=float), np.array(ws)
    weight /= weight.sum()
    mean = weight @ x
    scale = np.maximum(np.sqrt(weight @ (x - mean)**2), 0.05)
    x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.diag([0.] + [RIDGE] * len(FEATURES))
    coef = np.zeros(x.shape[1])
    for _ in range(40):
        p = sigmoid(x @ coef)
        gradient = x.T @ (weight * (p-y)) + penalty @ coef
        curvature = x.T @ ((weight * p * (1.-p))[:, None] * x) + penalty
        step = np.linalg.solve(curvature, gradient)
        coef -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return dict(features=list(FEATURES), ridge=RIDGE, mean=mean.tolist(), scale=scale.tolist(),
                intercept=float(coef[0]), coefficients=coef[1:].tolist(), training_rows=len(x))


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -35., 35.)))


def proposal(surface, model):
    """Find the top candidate and its nearest *different dimension pair*."""
    x = features(surface)
    if not len(x):
        return None
    scores = sigmoid(model['intercept'] + ((x - model['mean']) / model['scale']) @ model['coefficients'])
    idx = int(scores.argmax())
    candidates = surface['candidates']
    winning_pair = candidates[idx]['blocks']
    competitor = max((s for c, s in zip(candidates, scores) if c['blocks'] != winning_pair), default=0.)
    return (idx, float(scores[idx]), float(scores[idx] - competitor),
            winning_pair != surface['baseline']['blocks'])


def apply_guard(result, guard):
    if result and guard and result[3] and result[1] >= guard['threshold'] and result[2] >= guard['margin']:
        return result[0]
    return None


def score_frames(frames, targets, selections):
    totals = Counter(eligible=sum(t['eligible'] for t in targets), correct=0, incorrect=0,
                     missing=0, unmatched_returned=0, gained=0, lost=0, changed=0)
    for frame, target, picks in zip(frames, targets, selections, strict=True):
        if (frame['key'], frame['sha256']) != (picks['key'], picks['sha256']):
            raise ValueError('Selection frame mismatch')
        for (surface, answer), pick in zip(aligned([frame], [target]), picks['surfaces'], strict=True):
            if pick['id'] != surface['id']:
                raise ValueError('Selection proposal mismatch')
            idx = pick['index']
            baseline = surface['baseline']['blocks']
            guess = baseline if idx is None else surface['candidates'][idx]['blocks']
            expected = answer['expected']
            totals['changed'] += int(guess != baseline)
            if expected is None:
                totals['unmatched_returned'] += int(None not in guess)
            elif None not in guess:
                correct = guess == expected
                totals['correct' if correct else 'incorrect'] += 1
                totals['gained'] += int(correct and baseline != expected)
                totals['lost'] += int(not correct and baseline == expected)
    totals['missing'] = totals['eligible'] - totals['correct'] - totals['incorrect']
    return dict(totals)


def proposals(frames, model):
    return [[proposal(s, model) for s in f['surfaces']] for f in frames]


def selections(frames, ranked, guard, force=False):
    rows = []
    for f, results in zip(frames, ranked, strict=True):
        ids = [dict(id=s['id'], index=result[0] if force and result else apply_guard(result, guard)) for s, result in
               zip(f['surfaces'], results, strict=True)]
        rows.append(dict(key=f['key'], sha256=f['sha256'], surfaces=ids))
    return rows


def group(items, layout):
    return [f for f in items if f['layout'] == layout]


def fit(dev, targets):
    for frames in (dev, targets):
        if {f['layout'] for f in frames} != DEV or any(f['cohort'] != 'development' for f in frames):
            raise ValueError('Fitting is restricted to development Shopfront and Wing')
    folds = {}
    for layout in sorted(DEV):
        other = next(iter(DEV - {layout}))
        held = group(dev, layout)
        model = fit_weights(group(dev, other), group(targets, other))
        folds[layout] = held, group(targets, layout), proposals(held, model)
    trials = []
    guards = [None] + [dict(threshold=t, margin=m) for t in THRESHOLDS for m in MARGINS]
    for guard in guards:
        scores = {k: score_frames(f, t, selections(f, ranked, guard)) for k, (f,t,ranked) in folds.items()}
        utility = np.mean([(s['correct'] - s['incorrect'] - s['unmatched_returned'])/s['eligible']
                           for s in scores.values()])
        trials.append(dict(guard=guard, mean_layout_utility=float(utility), scores=scores))
    # Retain the ordinary choice when the override cannot improve cross-fit utility.
    best = max(trials, key=lambda trial: trial['mean_layout_utility'])
    forced = {k:score_frames(f, t, selections(f, ranked, None, force=True)) for k,(f,t,ranked) in folds.items()}
    return dict(**fit_weights(dev, targets), guard=best['guard'], crossfit=trials, forced_crossfit=forced,
                training_frames=[{k:f[k] for k in ('key','sha256','layout')} for f in dev])


def run(inputs, out):
    source = read(inputs / 'manifest.json')
    file_hashes = {p:digest(inputs/p) for p in ('candidates.json','targets.json')}
    if any(value != source['files'][p] for p,value in file_hashes.items()):
        raise ValueError('Candidate or target file differs from the frozen round')
    frames = read(inputs / 'candidates.json')
    targets = read(inputs / 'targets.json')
    roster = {row['key']:(row['sha256'],row['layout'],row['cohort']=='development') for row in source['frames']}
    received = {f['key']:(f['sha256'],f['layout'],f['cohort']=='development') for f in frames}
    if roster != received or len(frames) != len(roster) or len(targets) != len(frames):
        raise ValueError('Frame roster changed')
    dev = [f for f in frames if f['cohort'] == 'development']
    dev_targets = [t for t in targets if t['cohort'] == 'development']
    out.mkdir(parents=True, exist_ok=False)
    model = fit(dev, dev_targets)
    save(out / 'model.json', model)
    # Applying the frozen model needs only the candidate file.
    ranked = proposals(frames, model)
    picks = selections(frames, ranked, model['guard'])
    forced = selections(frames, ranked, None, force=True)
    save(out / 'selections.json', picks)
    save(out / 'forced-selections.json', forced)
    groups = ['all', 'development', 'regression', *sorted({f['layout'] for f in frames})]
    scores, forced_scores = {}, {}
    for name in groups:
        idx = [i for i,f in enumerate(frames) if name in ('all', f['layout'], f['cohort'])]
        batch = [frames[i] for i in idx]
        truth = [targets[i] for i in idx]
        scores[name] = score_frames(batch, truth, [picks[i] for i in idx])
        forced_scores[name] = score_frames(batch, truth, [forced[i] for i in idx])
    save(out / 'scores.json', scores)
    save(out / 'forced-scores.json', forced_scores)
    save(out / 'frames.json', [dict(key=f['key'], sha256=f['sha256'], guarded=score_frames([f],[t],[p]),
         forced=score_frames([f],[t],[q])) for f,t,p,q in zip(frames,targets,picks,forced,strict=True)])
    save(out / 'manifest.json', dict(version='26.3', method='ridge-logistic grid reranker',
         train=sorted(DEV), selection='Held-one-development-building-out mean (exact - wrong - unmatched guesses)/eligible; ordinary retained on ties.',
         reference_use='Development candidate pair labels during fitting. Other reference data used only for final grades.',
         forced_diagnostic='Highest model score on every nonempty pool. No confidence threshold, margin or baseline fallback when a candidate exists. Same model, excluded from policy selection.',
         input_sha256=file_hashes, source_manifest_sha256=digest(inputs / 'manifest.json'),
         reference_sha256=source['reference_sha256'], region_signatures_sha256=source['signatures_sha256'],
         artifacts={p.name:digest(p) for p in sorted(out.iterdir())},
         runner_sha256=digest(Path(__file__))))
    print('Guarded:', scores['regression'], 'Forced:', forced_scores['regression'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.out)
