"""Estimate the headroom for switching an entire build back to the previous placer."""
import argparse
from pathlib import Path

from benchmark.texture_camera_eval import read, save


def compare(previous, current):
    old = {r['id']: r for r in read(previous)}
    new = {r['id']: r for r in read(current)}
    if old.keys() != new.keys():
        raise ValueError('Cohorts differ')
    decisions = []
    totals = dict(correct=0, extras=0, target=0, hidden_extras=0)
    def f1(m):
        return 2 * m['correct'] / max(1, m['target'] + m['placed'])
    for key, row in new.items():
        before, after = old[key]['metrics'], row['metrics']
        if row['rgb_sha256'] != old[key]['rgb_sha256'] or before['target'] != after['target']:
            raise ValueError(f'Frame mismatch: {key}')
        previous_wins = f1(before) > f1(after)
        chosen = before if previous_wins else after
        for k in totals:
            totals[k] += chosen[k]
        decisions.append(dict(id=key, previous_selected=previous_wins, previous_f1=f1(before), current_f1=f1(after)))
    totals['f1'] = 2 * totals['correct'] / max(1, totals['target'] + totals['correct'] + totals['extras'])
    return dict(previous_selected=sum(r['previous_selected'] for r in decisions), selected_totals=totals, frames=decisions)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True, help='Previous per-frame cases.json')
    p.add_argument('--current', type=Path, required=True, help='New per-frame cases.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    save(args.out, compare(args.previous, args.current))
