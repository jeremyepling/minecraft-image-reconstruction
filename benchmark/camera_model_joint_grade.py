"""Independently grade joint camera ablations and the reference-only chooser."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.camera_horizon_replay import camera_validation
from benchmark.camera_model_comparison import MOGE_RUN, check_grade
from benchmark.camera_model_controls import REVIEW, check, frozen_frames, indexes, summary
from benchmark.camera_model_grade import verify_grade
from benchmark.expanded_baseline import WORKTREES, digest, read, save


JOINT = WORKTREES / 'camera-grid-joint'
JOINT_RUN = JOINT / 'runs/camera-grid-joint-01'
JOINT_REVIEW = JOINT / 'benchmark/reviews/camera-grid-joint-01'


def gate(proposals):
    passes = []
    for p in proposals:
        accepted = p['held_n'] >= 2 and max(p['held_p90_px'], p['fit_p90_px']) <= 3 and p['ground_error_blocks'] <= .16
        check(p['accepted'], accepted, 'individual source acceptance')
        passes.append(accepted)
    tie = len(passes) == 2 and passes[1] and abs(proposals[0]['rank'] - proposals[1]['rank']) < .5
    return bool(passes and passes[0] and not tie)


def verify_ablation(entry, inventory, record, arm):
    key = entry['id']
    folder = JOINT_RUN / key
    raw = read(MOGE_RUN / entry['dataset'] / key / 'proposal.json')['camera']
    proposal = record['prediction']['best'][int(arm == 'joint2')]
    camera = raw if arm == 'grid' else proposal['camera']
    overrides = {} if arm == 'camera' else proposal['overrides']
    shift = [0, 0, 0]
    if arm == 'grid':
        raw_matrix, fitted = map(np.asarray, (raw['P'], proposal['camera']['P']))
        check(np.allclose(raw_matrix[:, :3], fitted[:, :3]), True, f'{key} common grid-only axes and optics')
        displacement = np.linalg.solve(fitted[:, :3], fitted[:, 3] - raw_matrix[:, 3])
        shift = [round(displacement[0]), 0, round(displacement[2])]
        overrides = {name: (axis, level + shift[axis]) for name, (axis, level) in overrides.items()}
    report = record['ablations'][arm]
    receipt = read(folder / (arm + '.json'))
    check(receipt['calls'], report['calls'], f'{key} {arm} placement receipt')
    if arm == 'grid':
        check(receipt['calls'][-1]['horizontal_gauge_shift'], shift, f'{key} {arm} grid origin')
    build = folder / arm
    path = build / 'build-manifest.json'
    check(digest(path), report['manifest_sha256'], f'{key} {arm} build manifest')
    placement = read(path)['placement']
    check(placement['P'], camera['P'], f'{key} {arm} projection at placement')
    offsets = {r['id']: (r['axis'], r['level']) for r in placement['plane_offsets']}
    check({name: offsets[name] for name in overrides}, {k: tuple(v) for k, v in overrides.items()},
          f'{key} {arm} offsets at final placement')
    check(receipt['calls'][-1]['applied'], {k: v[1] for k, v in overrides.items()}, f'{key} {arm} consumed override levels')
    scored = verify_grade((entry, build, inventory))
    check_grade(arm, scored, receipt['case'])
    for field in ('geometry', 'scores', 'pack_sha256'):
        check(scored[field], report[field], f'{key} {arm} published {field}')
    check(scored['registration']['geometry']['strict'], report['registered'], f'{key} {arm} registered score')
    check(camera_validation(entry, placement), report['native'], f'{key} {arm} native camera errors')
    return scored


def verify(output):
    release = read(JOINT_REVIEW / 'manifest.json')
    for filename, expected in release['hashes'].items():
        check(digest(JOINT_REVIEW / filename), expected, f'joint release {filename}')
    check(digest(REVIEW / 'frames.json'), release['frozen_frames_sha256'], 'joint frozen frames')
    check(digest(REVIEW / 'baseline-cases.json.gz'), release['canonical_cases_sha256'], 'joint baseline cases')
    entries = {e['id']: e for e in frozen_frames()}
    reports = indexes(JOINT_REVIEW / 'cases.json.gz', release['hashes']['cases.json.gz'])
    inventory = {r['id']: r for r in read(REVIEW / 'inventory.json.gz')}
    inc = indexes(REVIEW / 'independent-matched-cases.json.gz', read(REVIEW / 'independent-matched-summary.json')['cases_sha256'])
    experiments = read(REVIEW / 'experimental-cases.json.gz')
    geocalib = {r['id']: r for r in experiments['geocalib'] if r['model_coverage'] == 'candidate_export'}
    moge = {r['id']: r for r in experiments['moge-standalone'] if r['model_coverage'] == 'candidate_export'}
    check(reports.keys(), entries.keys(), 'joint frame IDs')
    residual = {key for key, row in inc.items() if not row['camera_usable']}
    check({key for key, row in reports.items() if 'prediction' in row}, residual, 'joint source evaluation coverage')
    choices_by_arm = {key: [] for key in ('incumbent', 'source', 'raw-oracle', 'joint-oracle')}
    cases, picked_by_oracle = [], Counter()
    for key, row in reports.items():
        entry, incumbent = entries[key], inc[key]
        check(row['dataset'], entry['dataset'], f'{key} joint cohort')
        check(row['split'], entry['split'], f'{key} joint split')
        check(row['incumbent_pack_sha256'], incumbent['pack_sha256'], f'{key} incumbent pack')
        options = {'keep': incumbent}
        if key in residual:
            proposal = row['prediction']
            check(proposal['source_sha256'], entry['rgb_sha256'], f'{key} joint RGB')
            check(proposal['field_sha256'], digest(MOGE_RUN / row['dataset'] / key / 'field.npz'), f'{key} joint MoGe field')
            check(proposal['accepted'], gate(proposal['best']), f'{key} joint source decision')
            check(proposal['accepted'], False, f'{key} source accepted')
            for arm, available in (('raw-geocalib', geocalib), ('raw-moge', moge)):
                if key in available:
                    options[arm] = available[key]

        def best():
            return max(options, key=lambda arm: options[arm]['geometry']['strict']['f1'])

        raw_choice = best()
        checked = {}
        if 'ablations' in row:
            for arm in row['ablations']:
                checked[arm] = verify_ablation(entry, inventory[key], row, arm)
                if arm.startswith('joint'):
                    options[arm] = checked[arm]
            cases.append(dict(id=key, scores=checked))
        joint_choice = best()
        check(row['source_pack_sha256'], incumbent['pack_sha256'], f'{key} joint source policy retains pack')
        check(row['options'].keys(), options.keys(), f'{key} oracle option scope')
        check(row['raw_oracle'], raw_choice, f'{key} raw oracle choice')
        check(row['oracle'], joint_choice, f'{key} joint oracle choice')
        for arm, available in options.items():
            for metric in ('geometry', 'scores', 'pack_sha256'):
                check(row['options'][arm][metric], available[metric], f'{key} {arm} {metric}')
        picked_by_oracle[joint_choice] += 1
        for method, candidate in (('incumbent', incumbent), ('source', incumbent),
                                  ('raw-oracle', options[raw_choice]), ('joint-oracle', options[joint_choice])):
            choices_by_arm[method].append(candidate)
    counts = Counter(arm for case in cases for arm in case['scores'])
    check(counts, Counter(joint1=2, joint2=2, grid=2, camera=2), 'joint diagnostic export scope')
    pooled = {arm: summary(rows, list(entries.values())) for arm, rows in choices_by_arm.items()}
    original = read(JOINT_REVIEW / 'summary.json')
    for arm, cohorts in pooled.items():
        for dataset, row in cohorts.items():
            for metric in ('geometry', 'scores', 'tested', 'available'):
                check(row[metric], original[arm][dataset][metric], f'{arm} {dataset} pooled {metric}')
    save(output / 'independent-joint-check.json', dict(checked_frames=len(entries), residual_frames=len(residual),
         diagnostic_exports=dict(counts), source_changed_packs=0, oracle_options_selected=dict(picked_by_oracle),
         joint_manifest_sha256=digest(JOINT_REVIEW / 'manifest.json'), cohorts=pooled, cases=cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REVIEW)
    args = parser.parse_args()
    verify(args.output)


if __name__ == '__main__':
    main()
