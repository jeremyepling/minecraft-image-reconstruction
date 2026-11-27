"""Independently decode and score the seed-only and connected wall-ruler exports."""

import argparse
from pathlib import Path

import numpy as np

from benchmark.wall_ruler_transfer_controls import (
    ARM, CONTROL_FILE, REVIEW, RULER_RUNS, WORK, WORKTREES, candidate_inventory, check,
    file_digest, fixed_reference, frozen_frames, paired, read, registration, rescore,
    save, scores, typed, verified_payload,
)


TRIAL = WORKTREES / 'wall-ruler-local-transfer'
PILOT_RUN = TRIAL / 'runs/wall-ruler-local-transfer-01'
EXTENSION_RUN = TRIAL / 'runs/wall-ruler-local-transfer-extension-01'
PILOT_SUMMARY = TRIAL / 'benchmark/reviews/wall-ruler-local-transfer-01/exports.json'
EXTENSION_SUMMARY = TRIAL / 'benchmark/reviews/wall-ruler-local-transfer-extension-01/exports.json'
ARMS = ('direct', 'connected')
CHANGES = ('recovered', 'lost', 'extra_added', 'extra_removed', 'changed_positions', 'changed_ids_or_positions')


def verify_region(region):
    seeds = {tuple(c) for c in region['direct']}
    connected = {tuple(c) for c in region['connected']}
    check(len(seeds), len(region['direct']), 'unique seeds')
    check(len(connected), len(region['connected']), 'unique connected cells')
    if not seeds <= connected:
        raise ValueError(f'Seeds missing from connected result for {region["surface"]}')
    visited = set(seeds)
    fringe = list(seeds)
    while fringe:
        x, y = fringe.pop()
        neighbors = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)} & connected - visited
        visited.update(neighbors)
        fringe.extend(neighbors)
    check(visited, connected, f'{region["surface"]} seed-connected cells')


def audit_frame(entry, frame, run, previous_rulers, optional_arms):
    key = frame['id']
    base = frame['baseline']
    if frame['dataset'] == scores.FIXED_DATASET:
        reference, transform = fixed_reference(entry)
    else:
        reference = scores.load_reference(entry)
        transform = (registration(np.asarray(frame['destination']['P']), reference.pose)
                     if base['camera_usable'] else None)
    before, source_cells = rescore(verified_payload(frame['pack']), base, reference, transform)
    baseline = {k: base[k] for k in ('geometry', 'scores', 'exported', 'translation')}
    row = dict(id=key, dataset=frame['dataset'], split=frame['split'], usable=base['camera_usable'], baseline=baseline)
    out = run / key
    source = out / 'regions.json'
    regions = read(source)
    check(regions['source_sha256'], frame['rgb']['sha256'], f'{key} region source')
    records = regions['records']
    for record in records:
        verify_region(record)
    if previous_rulers:
        previous = read(previous_rulers / f'{key}.json')['records']
        signature = [(r['surface'], r['axis'], r['arms'][ARM]['level']) for r in previous
                     if r['arms'][ARM]['accepted'] and r['arms'][ARM]['legal_plane']]
        check([(r['surface'], r['axis'], r['level']) for r in records], signature, f'{key} reused ruler planes')
    row['regions_sha256'] = file_digest(source)

    def decoded(name):
        path = out / name / 'case.json'
        receipt = read(path)
        pack_source = receipt.get('reused', name)
        pack_receipt = out / pack_source / 'case.json'
        if pack_source != name:
            check((name, pack_source), ('overlap', 'connected'), f'{key} replay reuse')
            reused = read(pack_receipt)
            check(receipt['case'], reused['case'], f'{key} reused case')
            for local_event, reused_event in zip(receipt['events'], reused['events'], strict=True):
                check(local_event, reused_event, f'{key} unchanged local handoff')
                check(local_event['old_removed'], 0, f'{key} unchanged overlap removals')
        placed, local = candidate_inventory(pack_receipt, receipt, frame, entry, reference, transform)
        case = receipt['case']
        return placed, local, dict({k: case[k] for k in baseline}, export=name, pack_source=pack_source,
                                  pack_sha256=case['pack_sha256'],
                                  paired=paired(before, placed, reference.truth),
                                  receipt_sha256=file_digest(path), events=receipt['events'])

    noop, local, carry = decoded('noop')
    check(noop, before, f'{key} local trial no-op aligned IDs')
    check(local, source_cells, f'{key} local trial no-op decoded IDs')
    if not base['camera_usable']:
        check(records, [], f'{key} camera-abstained candidate list')
        return dict(row, **dict.fromkeys((*ARMS, *optional_arms), carry))
    for arm in ARMS:
        export = arm
        if export == 'connected' and all(r[arm] == r['direct'] for r in records):
            export = 'direct'
        if not any(r[export] for r in records):
            export = 'noop'
        _, _, row[arm] = decoded(export)
        row[arm]['selected_cells'] = sum(len(r[arm]) for r in records)
    for extra in optional_arms:
        export = extra if (out / extra / 'case.json').is_file() else 'noop'
        if 'whole' in optional_arms:
            required = bool(records) if extra == 'whole' else any(r['connected'] for r in records)
            check(export == extra, required, f'{key} {extra} saved export coverage')
        _, _, row[extra] = decoded(export)
    return row


def audit(frames, run, previous_rulers, optional_arms, published=None):
    entries = {entry['id']: entry for entry in frozen_frames()}
    rows = [audit_frame(entries[frame['id']], frame, run, previous_rulers, optional_arms) for frame in frames]
    if published is not None:
        for row, claim in zip(rows, published['cases'], strict=True):
            check(row['id'], claim['id'], 'local trial frame order')
            for arm in (*ARMS, *(arm for arm in optional_arms if arm in claim)):
                actual = row[arm]
                for key in ('geometry', 'scores', 'pack_sha256', 'translation'):
                    check(actual[key], claim[arm][key], f'{row["id"]} {arm} {key}')
                for key in CHANGES[:4]:
                    check(actual['paired'][key], claim[arm][key], f'{row["id"]} {arm} {key}')
    pools = {}
    for group in ('all', 'fixed-104', 'visible-builds-v3', 'decorative-builds-v1', 'development', 'transfer'):
        subset = [r for r in rows if group in ('all', r['dataset'], 'development' if r['split'] == 'train' else 'transfer')]
        if not subset:
            continue
        pools[group] = dict(tested=len(subset), usable=sum(r['usable'] for r in subset))
        pooled_arms = (*ARMS, *optional_arms) if 'whole' in optional_arms else ARMS
        for arm in ('baseline', *pooled_arms):
            arms = [r[arm] for r in subset]
            pooled = {metric: typed.pooled_scores([r[metric] for r in arms]) for metric in ('geometry', 'scores')}
            if arm != 'baseline':
                pooled['paired'] = {key: sum(r['paired'][key] for r in arms) for key in CHANGES}
            pools[group][arm] = pooled
    return dict(rows=rows, pools=pools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--extension', action='store_true')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if options.extension:
        control_path = REVIEW / 'extension.json'
        frames = read(control_path)['frames']
        run = options.run or EXTENSION_RUN
        previous_rulers = run / 'candidates'
        optional_arms = ('whole', 'overlap')
        published_path = EXTENSION_SUMMARY
        output = options.output or REVIEW / 'local-extension.json'
    else:
        control_path = CONTROL_FILE
        frames = read(control_path)['pilot']['frames']
        run = options.run or PILOT_RUN
        previous_rulers = RULER_RUNS[0]
        optional_arms = ('overlap',)
        published_path = PILOT_SUMMARY
        output = options.output or REVIEW / 'local-pilot.json'
    published = read(published_path) if published_path.is_file() else None
    if published is not None:
        check(published['controls_sha256'], file_digest(control_path), 'trial frozen controls')
    result = audit(frames, run, previous_rulers, optional_arms, published)
    result['controls_sha256'] = file_digest(control_path)
    result['trial_summary_sha256'] = file_digest(published_path) if published is not None else None
    save(output, result)
    print(f'Decoded and independently scored {len(result["rows"])} local trial comparisons')


if __name__ == '__main__':
    main()
