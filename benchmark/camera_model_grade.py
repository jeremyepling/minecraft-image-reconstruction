"""Independently score camera-model datapacks with the frozen controls."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import expanded_scores as scores
from benchmark.camera_model_controls import check, fixed_reference, frozen_frames, inventory_hash, REVIEW, summary
from benchmark.expanded_baseline import digest, read, save
from benchmark.mask_blocks import decoded_pack

def grade_fixed(entry, build):
    manifest = read(build / 'build-manifest.json')
    check(digest(entry['rgb']), entry['rgb_sha256'], f'{entry["id"]} frozen RGB')
    check(manifest['image_sha256'], entry['rgb_sha256'], f'{entry["id"]} candidate RGB')
    labels, _ = fixed_reference(entry)
    rgb = entry['rgb']
    if entry['split'] in ('train', 'test', 'validation', 'regression'):
        pose_path = rgb.with_name('camera.json')
    else:
        pose_path = rgb.with_suffix('.json')
    check(digest(pose_path), labels.hashes['camera.json'], f'{entry["id"]} frozen pose')
    labels.pose = read(pose_path)
    placements = read(build / 'placements.json')['blocks']
    assignment = {(b['x'], b['y'], b['z']): b['material_source'] for b in placements}
    registry_ids = {(b['x'], b['y'], b['z']): 'minecraft:' + b['block_id'].removeprefix('minecraft:') for b in placements}
    check(len(registry_ids), len(placements), f'{entry["id"]} unique placement cells')
    pack = build / 'reconstruction.zip'
    check(registry_ids, decoded_pack(pack), f'{entry["id"]} placements and datapack')
    row, _ = scores.evaluate_pack(entry, pack, manifest['placement'], assignment=assignment, reference=labels)
    check(row['exported'], manifest['placement']['placed_blocks'], f'{entry["id"]} exported count')
    return row


def verify_grade(task):
    entry, build, expected = task
    dataset, key = entry['dataset'], entry['id']
    row = grade_fixed(entry, build) if dataset == scores.FIXED_DATASET else scores.score_build(entry, build)
    check(row['source_sha256'], expected['rgb_sha256'], f'{key} source')
    check(row['reference_sha256'], expected['source_reference_sha256'], f'{key} reference')
    row['decoded_sha256'] = inventory_hash(decoded_pack(build / 'reconstruction.zip'))
    row['provenance'] = 'independently graded candidate export'
    return row


def deltas(after, before):
    result = {}
    for name in ('geometry', 'scores'):
        result[name] = {}
        for scope in ('strict', 'facade'):
            new, old = after[name][scope], before[name][scope]
            result[name][scope] = {k: new[k] - old[k] for k in ('correct', 'missed', 'extras', 'f1')}
    result['camera'] = {k: after['camera'][k] - before['camera'][k] for k in ('usable', 'unusable', 'supported_empty')}
    return result


def grade(entries, cli):
    control_manifest = read(cli.controls / 'manifest.json')
    for name in ('frames.json', 'inventory.json.gz', 'baseline-cases.json.gz', 'baseline-summary.json'):
        check(digest(cli.controls / name), control_manifest['hashes'][name], f'frozen {name}')
    inventory = {r['id']: r for r in read(cli.controls / 'inventory.json.gz')}
    published = read(cli.controls / 'baseline-cases.json.gz')
    tasks, expected_ids = [], set()
    if cli.attempts:
        attempts = set(read(cli.attempts))
        check(attempts - entries.keys(), set(), 'unknown attempted frame IDs')
    else:
        attempts = set()
    for key, entry in entries.items():
        if cli.inherit == 'repeat' and entry['repeat_supported']:
            continue
        if cli.inherit == 'ordinary' and entry['ordinary_supported'] and not cli.include_supported:
            continue
        expected_ids.add(key)
        frame_dir = cli.builds / entry['dataset'] / key
        build = frame_dir / cli.build_subdir
        if cli.complete_receipt and not (frame_dir / cli.complete_receipt).is_file():
            continue
        if (build / 'build-manifest.json').is_file():
            tasks.append((entry, build, inventory[key]))
    check(attempts - expected_ids, set(), 'attempt IDs outside this arm')
    with ProcessPoolExecutor(cli.workers) as pool:
        measured = {row['id']: row for row in pool.map(verify_grade, tasks)}
    attempted_ids = attempts | measured.keys()
    candidate = []
    per_frame = {}
    for inherited in published[cli.inherit]:
        key = inherited['id']
        measured_case = measured.get(key)
        candidate.append(measured_case or inherited)
        per_frame[key] = ('candidate_export' if measured_case else 'attempted_carried_forward' if key in attempted_ids
                          else 'pending_carried_forward' if key in expected_ids else 'policy_carried_forward')
    pooled = summary(candidate, list(entries.values()))
    baselines = read(cli.controls / 'baseline-summary.json')
    for dataset, group in pooled.items():
        splits = group['splits']
        for split, node in ((None, group), *splits.items()):
            cohort = [key for key, entry in entries.items()
                      if entry['dataset'] == dataset and (split is None or entry['split'] == split)]
            node['coverage'] = dict(Counter(per_frame[key] for key in cohort))
            node['delta'] = {baseline: deltas(node, data[dataset] if split is None else data[dataset]['splits'][split])
                             for baseline, data in baselines.items()}
    casefile = cli.output / 'cases.json.gz'
    save(casefile, [dict(row, model_coverage=per_frame[row['id']]) for row in candidate])
    save(cli.output / 'summary.json', dict(inherited=cli.inherit, cases_sha256=digest(casefile), datasets=pooled,
         control_manifest_sha256=digest(cli.controls / 'manifest.json'), expected_attempts=len(expected_ids),
         candidate_exports=len(measured), attempted_no_export=len(attempted_ids - measured.keys()),
         pending_attempts=len(expected_ids - attempted_ids)))
    print(f'{len(measured)} candidate exports, {len(attempted_ids - measured.keys())} attempted without new export, '
          f'{len(expected_ids - attempted_ids)} pending, {len(entries) - len(expected_ids)} policy carry-forward.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, default=REVIEW)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inherit', choices=('ordinary', 'repeat'), required=True)
    parser.add_argument('--include-supported', action='store_true', help='Allow candidates to replace ordinary supported builds')
    parser.add_argument('--build-subdir', default='.')
    parser.add_argument('--complete-receipt', help='Require a completion file beside the build folder')
    parser.add_argument('--attempts', type=Path, help='JSON array of frame IDs whose policy ran, including abstentions')
    parser.add_argument('--workers', type=int, default=4)
    cli = parser.parse_args()
    grade({entry['id']: entry for entry in frozen_frames()}, cli)


if __name__ == '__main__':
    main()
