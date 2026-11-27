"""Summarize a frozen placement release from its scored per-view records."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
from urllib.parse import quote


REVIEWS = Path(__file__).resolve().parent / 'reviews'
FIXTURE = REVIEWS / 'workshop-foreground-01'
LAYOUTS = REVIEWS / 'facade-placement-01/worst/ranking.json'
METRICS = ('strict', 'facade', 'registered')
FIELDS = ('correct', 'missed', 'extras')
PAIRED = ('gained', 'lost', 'extra_added', 'extra_removed')


def load_release(directory):
    summary = json.loads((directory / 'summary.json').read_bytes())
    data = (directory / 'cases.json').read_bytes()
    if sha256(data).hexdigest() != summary['cases_sha256']:
        raise ValueError(f'{directory}: cases.json differs from its release digest')
    cases = json.loads(data)
    by_id = {row['id']: row for row in cases}
    if len(by_id) != len(cases) or len(cases) != summary['frames']:
        raise ValueError(f'{directory}: duplicate or missing source-image records')
    return summary, by_id


def validate(rows, fixture, member):
    if rows.keys() != fixture.keys():
        raise ValueError('The release must contain exactly the frozen 204 frame IDs')
    for key, row in rows.items():
        reference = fixture[key]
        if (row['source_hash'], row['split']) != (reference['source_hash'], reference['split']):
            raise ValueError(f'{key}: source image or evaluation split changed')
        for metric in METRICS:
            score = row[member][metric]
            truth = reference['result'][metric]
            if any(type(score[k]) is not int or score[k] < 0 for k in FIELDS):
                raise ValueError(f'{key}: invalid {metric} counts')
            if score['correct'] + score['missed'] != truth['correct'] + truth['missed']:
                raise ValueError(f'{key}: {metric} target count changed')
        strict = row[member]['strict']
        facade = row[member]['facade']
        if facade['extras'] != strict['extras'] or facade['correct'] > strict['correct'] or facade['missed'] > strict['missed']:
            raise ValueError(f'{key}: inconsistent strict and facade counts')


def f1(score):
    return Fraction(2 * score['correct'], 2 * score['correct'] + score['missed'] + score['extras'])


def pooled(scores):
    totals = Counter()
    for score in scores:
        totals.update({k: score[k] for k in FIELDS})
    return totals


def counts(score):
    return ' / '.join(f'{score[k]:,}' for k in FIELDS)


def pack(row, member):
    if member == 'result' and 'pack_sha256' in row:
        return row['pack_sha256']
    return row['previous_pack_sha256']


def paired_table(ids, before, member, after):
    paired = Counter()
    unknown = 0
    for key in ids:
        row = after[key]
        old = before[key]
        if pack(old, member) == pack(row, 'result'):
            if any(old[member][m] != row['result'][m] for m in ('strict', 'facade')):
                raise ValueError(f'{key}: identical exports have different placement scores')
            continue
        deltas = row.get('block_deltas')
        if pack(old, member) != row['previous_pack_sha256'] or deltas is None:
            unknown += 1
            continue
        if any(type(deltas[k]) is not int or deltas[k] < 0 for k in PAIRED):
            raise ValueError(f'{key}: invalid paired block deltas')
        a, b = old[member]['strict'], row['result']['strict']
        if (deltas['gained'] - deltas['lost'], deltas['extra_added'] - deltas['extra_removed']) != (
                b['correct'] - a['correct'], b['extras'] - a['extras']):
            raise ValueError(f'{key}: paired block deltas disagree with net change')
        paired.update({k: deltas[k] for k in PAIRED})
    text = (f"Correct gained/lost {paired['gained']:,}/{paired['lost']:,}. "
            f"Extras removed/added {paired['extra_removed']:,}/{paired['extra_added']:,}.")
    if unknown:
        return f'Paired evidence covers {len(ids)-unknown}/{len(ids)} views. Within that subset: {text} Other changes have net counts only.'
    return text


def perfect_repairs(groups, total):
    # At 95% F1, 2 * correct >= 19 * (missed + extras). A recovered miss also becomes correct.
    deficit = 19 * (total['missed'] + total['extras']) - 2 * total['correct']
    gains = sorted((21 * group['missed'] + 19 * group['extras'] for group in groups), reverse=True)
    used = 0
    for gain in gains:
        if deficit <= 0:
            break
        deficit -= gain
        used += 1
    return used


def report(directory, baseline=None, *, link_from=None):
    directory = directory.resolve()
    candidate, after = load_release(directory)
    _, fixture = load_release(FIXTURE)
    if len(fixture) != 204:
        raise ValueError('Expected the frozen 204-view placement suite')
    if baseline is None:
        before, member = after, 'baseline'
        old_name = candidate['baseline']
    else:
        _, before = load_release(baseline)
        member = 'result'
        old_name = baseline.name
    validate(before, fixture, member)
    validate(after, fixture, 'result')
    layouts = json.loads(LAYOUTS.read_bytes())['all_layouts']
    fixed = [key for item in layouts for key in item['frame_ids']]
    if len(fixed) != 104 or len(set(fixed)) != 104 or set(fixed) - fixture.keys():
        raise ValueError('The layout manifest must cover exactly the fixed 104 views')
    lines = [f'# Placement comparison for {directory.name}', '',
             f"Baseline `{old_name}`. Saved per-view scores from `cases.json`, SHA256 `{candidate['cases_sha256']}`. "
             'This report reads the existing scores and runs zero inference or export jobs.', '',
             '| Suite | Metric | Previous correct / missed / extra | Current correct / missed / extra | Previous F1 | Current F1 | Change (pp) | Views up / down / tied |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    details = []
    for group, ids in [('Fixed', fixed), ('Other', sorted(after.keys() - set(fixed)))]:
        name = f'{group} {len(ids)}'
        for metric in METRICS:
            a = pooled(before[key][member][metric] for key in ids)
            b = pooled(after[key]['result'][metric] for key in ids)
            previous, current = float(f1(a))*100, float(f1(b))*100
            changes = [f1(after[key]['result'][metric]) - f1(before[key][member][metric]) for key in ids]
            wins, losses = sum(n > 0 for n in changes), sum(n < 0 for n in changes)
            lines.append(f'| {name} | {metric} | {counts(a)} | {counts(b)} | {previous:.4f}% | {current:.4f}% '
                         f'| {current-previous:+.4f} | {wins} / {losses} / {len(ids)-wins-losses} |')
        details.append(f'- **{name}.** {paired_table(ids, before, member, after)}')
    lines += ['', 'Strict and facade F1 allow one whole-build translation. Registered F1 uses the camera frame. '
              'F1 is pooled over block counts. Each source image casts one vote in the last column.',
              '', '## Paired relative positions', '', *details]
    if 'historical_112_strict_f1' in candidate:
        lines.extend(['', f"Historical 112-view strict F1 from the release: {100*candidate['historical_112_strict_f1']:.4f}%."])
    facade = {key: after[key]['result']['facade'] for key in fixed}
    total = pooled(facade.values())
    grouped = [pooled(facade[key] for key in item['frame_ids']) for item in layouts]
    remaining = max(0.0, 95 - 100 * float(f1(total)))
    lines += ['', f'**95% facade-priority target.** {remaining:.4f} percentage points remain. '
              'A perfect repair of the best-selected '
              f'{perfect_repairs(facade.values(), total)} whole views or {perfect_repairs(grouped, total)} '
              'whole layouts would reach the target. This reference-only lower bound assumes all required misses '
              'and extras in each selected group are fixed with every previously correct position preserved.']
    errors = sorted(((group['missed'] + group['extras'], layout['name']) for layout, group in zip(layouts, grouped)),
                    key=lambda pair: (-pair[0], pair[1]))
    largest = [f'{name} ({count:,})' for count, name in errors[:3] if count]
    if largest:
        lines += ['', 'Largest counts of outstanding required misses and extras by layout: ' + ', '.join(largest) + '.']
    lines += ['', '## Largest facade-priority changes', '']
    changes = defaultdict(list)
    for key in after:
        a, b = before[key][member]['facade'], after[key]['result']['facade']
        delta = f1(b) - f1(a)
        if delta:
            changes['improved' if delta > 0 else 'regressed'].append((abs(delta), key, delta, a, b))
    if not changes:
        lines.append('All views tied on facade-priority F1.')
    for direction in ('improved', 'regressed'):
        items = sorted(changes[direction], key=lambda item: (-item[0], item[1]))
        if items:
            lines.extend([f'{len(items)} views {direction}. Showing {min(5,len(items))}.', ''])
            for _, key, delta, a, b in items[:5]:
                lines.append(f'- `{key}` {float(delta)*100:+.2f} pp. {counts(a)} to {counts(b)}.')
            lines.append('')
    lines.extend(['', '## Lowest-scoring structures on the fixed suite', '',
                  'Each structure contributes its weakest strict-F1 view, as in the interactive viewer.', '',
                  '| Structure | Source image | Strict F1 | Facade F1 | Strict correct / missed / extra |',
                  '| --- | --- | ---: | ---: | ---: |'])
    def rank(key):
        return f1(after[key]['result']['strict']), key
    weakest = [(item['name'], min(item['frame_ids'], key=rank)) for item in layouts]
    for name, key in sorted(weakest, key=lambda pair: rank(pair[1]))[:10]:
        row = after[key]['result']
        lines.append(f"| {name} | `{key}` | {float(f1(row['strict'])):.2%} | {float(f1(row['facade'])):.2%} | {counts(row['strict'])} |")
    evidence = directory / 'README.md'
    if evidence.is_file():
        try:
            path = Path(os.path.relpath(evidence, start=link_from))
        except ValueError:  # A Windows report on a different drive needs an absolute target.
            path = evidence
        lines += ['', f'[Release findings and evidence]({quote(path.as_posix(), safe="/:")})']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path, help='Directory containing cases.json and summary.json')
    parser.add_argument('--baseline', type=Path, help='Compare with this release result instead of the embedded baseline')
    parser.add_argument('--out', type=Path, help='Write Markdown to a file. Default: standard output.')
    args = parser.parse_args()
    try:
        result = report(args.release, args.baseline, link_from=args.out.parent if args.out else None)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result, encoding='utf-8')
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
