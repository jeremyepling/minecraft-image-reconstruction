"""Audit successive Sky mask snapshots against fixed native visible targets."""

import argparse
from collections import Counter
from math import isclose
from pathlib import Path

import numpy as np

from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.sky_crop_reference import (
    FIXTURES, HANDOFFS, arm_summary, evaluate_case, native_mask, pooled_metrics,
)
from benchmark.sky_crop_trajectory_review import write_case, write_index


OUTPUT = ROOT / 'runs/sky-crop-corrections-audit-01'
THRESHOLD = .95
EDIT_FIELDS = ('added_target', 'added_off_target', 'removed_target', 'removed_off_target')
CATEGORIES = {
    'missed': 'Missing pixels on the target',
    'other_cube_pixels': 'Pixels from another cube surface',
    'non_cube_pixels': 'Pixels from decorative or partial faces',
    'background_pixels': 'Background pixels',
}


def read_trajectory(spec):
    name, separator, folder = spec.partition('=')
    if not separator or not name or not folder:
        raise ValueError('Expected --trajectory NAME=FOLDER')
    folder = Path(folder)
    numeric = sorted((int(path.stem.removeprefix('round-')), path)
                     for path in folder.glob('round-*.json'))
    if not numeric or [number for number, _ in numeric] != list(range(len(numeric))):
        raise ValueError(f'Expected consecutive snapshots beginning with round 0 in {folder}')
    rounds = [(str(number), path) for number, path in numeric]
    for endpoint in ('selected', 'final'):
        path = folder / f'{endpoint}.json'
        if path.is_file():
            rounds.append((endpoint, path))
    packets = {}
    for token, path in rounds:
        packet = read(path)
        packets[token] = dict(frames=packet['frames'], filename=path.name, sha256=digest(path),
                              metadata={key: value for key, value in packet.items() if key != 'frames'})
    trace_path = folder / 'trace.json'
    trace = read(trace_path) if trace_path.is_file() else {}
    return dict(name=name, packets=packets, trace=trace,
                trace_sha256=digest(trace_path) if trace_path.is_file() else None)


def arm_name(index, token):
    return f'{index}#{token}'


def edit_delta(before, after, target):
    added = after & ~before
    removed = before & ~after
    added_target = int(np.count_nonzero(added & target))
    removed_target = int(np.count_nonzero(removed & target))
    return dict(added_target=added_target, removed_target=removed_target,
                added_off_target=int(np.count_nonzero(added)) - added_target,
                removed_off_target=int(np.count_nonzero(removed)) - removed_target)


def failure_category(score):
    if score['iou'] >= THRESHOLD:
        return None
    if score['target'] is None:
        return 'No overlapping full-cube target at baseline'
    dominant = max(CATEGORIES, key=score.__getitem__)
    return CATEGORIES[dominant]


def receipt(trace, frame, proposal, token):
    item = trace.get('frames', {}).get(frame, {}).get(proposal)
    if isinstance(item, list):
        steps, number_field = item, 'step'
    elif isinstance(item, dict):
        steps, number_field = item.get('steps', []), 'round'
    else:
        return []
    if not token.isdigit():
        return []
    return [step for step in steps if step.get(number_field) == int(token)]


def check_edit_records(records, before, target, before_score, score, delta, location):
    for record in records:
        if 'before_iou' not in record:
            if 'area_before' in record and (
                    record['area_before'] != before_score['mask_pixels']
                    or record['area_after'] != score['mask_pixels']
                    or record['added'] != delta['added_target'] + delta['added_off_target']
                    or record['removed'] != delta['removed_target'] + delta['removed_off_target']):
                raise ValueError(f'Policy receipt disagrees with native edit size on {location}')
            continue
        x, y = record['x'], record['y']
        expected = bool(target[y, x]) and not before[y, x] if record['label'] == 1 else (
            bool(before[y, x]) and not target[y, x])
        if (not expected or not isclose(record['before_iou'], before_score['iou'], abs_tol=1e-10)
                or not isclose(record['after_iou'], score['iou'], abs_tol=1e-10)
                or record['accepted'] != (score['iou'] > before_score['iou'])):
            raise ValueError(f'Prompt receipt disagrees with independent native score on {location}')


def frame_rows(case, visuals, trajectories):
    image, _, reference, input_rows = visuals
    ident = case['id']
    for index, trajectory in enumerate(trajectories):
        source_trace = trajectory['trace'].get('inputs', {}).get(ident, {})
        for field, expected in source_trace.items():
            if field in case and case[field] != expected:
                raise ValueError(f'Trajectory input differs from the frozen fixture on {ident}/{field}')
        for order, (_, part, baseline, paired) in enumerate(input_rows):
            proposal = part['id']
            target = reference.labels == paired if paired else np.zeros_like(baseline)
            previous = baseline
            previous_score = case['arms']['baseline']['proposals'][order]
            for token, packet in trajectory['packets'].items():
                support = packet['frames'].get(ident, {}).get(proposal)
                mask = baseline if support is None else native_mask(support, image.size)
                arm = arm_name(index, token)
                score = case['arms'][arm]['proposals'][order]
                records = receipt(trajectory['trace'], ident, proposal, token)
                if token.isdigit():
                    delta = edit_delta(previous, mask, target)
                    if (delta['added_target'] - delta['removed_target'] != score['intersection'] - previous_score['intersection']
                            or delta['added_off_target'] - delta['removed_off_target'] != score['extra'] - previous_score['extra']):
                        raise ValueError(f'Edit count differs from scored mask changes on {ident}/{proposal}/{token}')
                    score.update(delta, changed_this_round=any(delta.values()),
                                 previous_iou=previous_score['iou'], delta_iou=score['iou'] - previous_score['iou'])
                    check_edit_records(records, previous, target, previous_score, score, delta,
                                       f'{ident}/{proposal}/{token}')
                    previous = mask
                    previous_score = score
                score['failure_category'] = failure_category(score)
                score['edit_receipts'] = records
            trace = trajectory['trace'].get('frames', {}).get(ident, {}).get(proposal)
            if isinstance(trace, dict) and trace.get('stop'):
                tokens = [token for token in trajectory['packets'] if token.isdigit()]
                case['arms'][arm_name(index, tokens[-1])]['proposals'][order]['producer_stop'] = trace['stop']
        for token in trajectory['packets']:
            arm = case['arms'][arm_name(index, token)]
            arm['summary'] = round_summary(arm['proposals'], arm['summary'])


def round_summary(rows, coverage):
    paired = [row for row in rows if row['target'] is not None]
    result = dict(
        **coverage,
        with_target=len(paired),
        masks_iou_at_least_95=sum(row['iou'] >= THRESHOLD for row in rows),
        matched_macro_iou=arm_summary(paired, {})['macro_iou'],
        failures_by_category=dict(Counter(row['failure_category'] for row in rows if row['failure_category'])),
        producer_stop=dict(Counter(row['producer_stop'] for row in rows if row.get('producer_stop'))),
    )
    if rows and 'changed_this_round' in rows[0]:
        prompts = [record for row in rows for record in row['edit_receipts'] if 'accepted' in record]
        result.update(
            changed_this_round=sum(row['changed_this_round'] for row in rows),
            improved_this_round=sum(row['delta_iou'] > 0 for row in rows),
            regressed_this_round=sum(row['delta_iou'] < 0 for row in rows),
            newly_at_least_95=sum(row['iou'] >= THRESHOLD > row['previous_iou'] for row in rows),
            fell_below_95=sum(row['iou'] < THRESHOLD <= row['previous_iou'] for row in rows),
            attempted_prompts=len(prompts),
            accepted_prompts=sum(record['accepted'] for record in prompts),
            prompt_diagnoses=dict(Counter(record['diagnosis'] for record in prompts)),
            **{field: sum(row[field] for row in rows) for field in EDIT_FIELDS},
        )
    return result


def trajectory_summary(cases, arms):
    if not cases:
        return {}
    coverage = pooled_metrics(cases, arms)
    return {arm: round_summary([row for case in cases for row in case['arms'][arm]['proposals']], coverage[arm])
            for arm in arms}


def summarize(cases, trajectories):
    parts = {
        'all': cases,
        'training': [case for case in cases if case['split'] == 'train'],
        'held-out': [case for case in cases if case['split'] != 'train'],
        'validation': [case for case in cases if case['split'] == 'validation'],
        'test': [case for case in cases if case['split'] == 'test'],
    }
    result = {}
    for index, trajectory in enumerate(trajectories):
        tokens = list(trajectory['packets'])
        arms = [arm_name(index, token) for token in tokens]
        grouped = {}
        for name, group in parts.items():
            by_arm = trajectory_summary(group, arms)
            if by_arm:
                grouped[name] = dict(zip(tokens, by_arm.values(), strict=True))
        result[trajectory['name']] = grouped
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--handoffs', type=Path, default=HANDOFFS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--trajectory', action='append', required=True, metavar='NAME=FOLDER')
    parser.add_argument('--scores-only', action='store_true')
    args = parser.parse_args()
    fixtures = read(args.fixtures)
    trajectories = [read_trajectory(spec) for spec in args.trajectory]
    names = [item['name'] for item in trajectories]
    if len(set(names)) != len(names):
        parser.error('Trajectory names must be unique')
    candidates = {arm_name(index, token): packet['frames']
                  for index, trajectory in enumerate(trajectories)
                  for token, packet in trajectory['packets'].items()}
    ids = {case['id'] for case in fixtures['cases']}
    for arm, replacements in candidates.items():
        if unknown := set(replacements) - ids:
            parser.error(f'Unknown fixture IDs for {arm}: {sorted(unknown)}')

    cases = []
    for fixture in fixtures['cases']:
        case, _, visual_data = evaluate_case(fixture, args.handoffs, candidates)
        frame_rows(case, visual_data, trajectories)
        if not args.scores_only:
            write_case(args.output, case, visual_data, trajectories)
        cases.append(case)
        print(case['id'], flush=True)

    report = dict(fixture_sha256=digest(args.fixtures), scorer_sha256=digest(Path(__file__)),
                  target_policy='Fixed highest-IoU positive full-cube target of saved crop',
                  trajectory_provenance=[
                      dict(name=t['name'], trace_sha256=t['trace_sha256'],
                           trace_settings={k: v for k, v in t['trace'].items() if k not in ('frames', 'inputs')},
                           rounds={token: {k: v for k, v in packet.items() if k != 'frames'}
                                   for token, packet in t['packets'].items()})
                      for t in trajectories],
                  cases=cases, trajectories=summarize(cases, trajectories))
    save(args.output / 'report.json', report)
    if not args.scores_only:
        write_index(args.output, report)


if __name__ == '__main__':
    main()
