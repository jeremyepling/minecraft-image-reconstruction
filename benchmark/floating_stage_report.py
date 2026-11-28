"""Summarize completed Floating runs without running the predictor or scorer."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


DATASETS = ('Visible Builds', 'Decorative Builds V1')
COUNTS = ('correct', 'missed', 'extras')
REGIMES = ('strict', 'facade')
MEASURES = {
    'surface': ('Surface Finder', 'Masks kept / masks entering the whole-mask sky check'),
    'counter': ('Block Counter', 'Surfaces with both dimensions / surfaces sent to counting'),
    'depth': ('Depth-first refit', 'Split parts with a fresh atlas fit offered / split parts tried'),
    'axes': ('Local orientation', 'Views with oriented charts / views with assembly receipts'),
    'charts': ('Relative joins', 'Charts in the selected group / all oriented charts'),
    'groups': ('Component selection', 'Exported blocks / cells across all assembled components'),
    'material': ('Material matching', 'Exact IDs / correct positions in the decoded export'),
    'placement': ('Decoded placement', 'Correct / visible reference blocks'),
}


def read(path):
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw)


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def handoff_metrics(folder, row, receipt):
    manifest = folder / 'build-manifest.json'
    if not manifest.is_file() or receipt is None:
        return {}
    source = read(manifest)
    if source['image_sha256'] != row['source_sha256']:
        raise ValueError(f'Source handoff belongs to another image: {folder}')
    masks = source['surface_finder']['boundary_settings'].get('background_filter')
    result = {'surface': (masks['kept_count'], masks['input_count'])} if masks else {}
    counts = folder / 'counts.json'
    if counts.is_file():
        if receipt.get('counts_sha256') != hash_file(counts):
            raise ValueError(f'Count handoff differs from the scored packet: {counts}')
        surfaces = read(counts)['surfaces']
        dimensions = [item.get('dimensions', {}).get('blocks', ()) for item in surfaces]
        complete = sum(len(blocks) == 2 and all(value is not None for value in blocks) for blocks in dimensions)
        result['counter'] = (complete, len(surfaces))
    return result


def depth_metrics(audit):
    parts = [part for mask in audit.values() if 'parts' in mask for part in mask['parts'].values()]
    return sum(bool(part.get('grids')) for part in parts), len(parts)


def archived_arm(row, mode):
    if 'arms' in row:
        arm = row['arms'][mode]
        return arm, Path(arm['pack']).with_name('score.json')
    root = Path(row['roots'][mode])
    receipt = root / 'connected-atlas' / 'sky' / row['id'] / 'score.json'
    return {'score': row['score'][mode], 'pack_sha256': row['packs'][mode]}, receipt


def score_arm(row, mode, source, audit):
    arm, path = archived_arm(row, mode)
    receipt = read(path) if path.is_file() else None
    if receipt is not None:
        expected = {'id': row['id'], 'source_sha256': row['source_sha256'],
                    'reference_sha256': row['reference_sha256'], 'pack_sha256': arm['pack_sha256']}
        if any(receipt[k] != value for k, value in expected.items()):
            raise ValueError(f'Score receipt differs from the archived comparison: {path}')
        exported = receipt['score']['exported']
        unresolved = receipt['score']['fixed'] is None and exported > 0
        selected = sorted(receipt['selected_origins'])
    elif 'nonempty_unresolved' in arm:
        exported, unresolved, selected = arm['exported'], arm['nonempty_unresolved'], sorted(arm['selected'])
    else:
        raise FileNotFoundError(f'Restore the saved orientation and export receipt: {path}')

    scores = {}
    for measure in ('geometry', 'scores'):
        for regime in REGIMES:
            frozen = arm['score'][measure][regime]
            scores[f'{measure}_{regime}'] = ({'correct': 0, 'missed': frozen['correct'] + frozen['missed'],
                                            'extras': exported} if unresolved else {k: frozen[k] for k in COUNTS})
    strict = scores['geometry_strict']
    stages = {'material': (scores['scores_strict']['correct'], strict['correct']),
              'placement': (strict['correct'], strict['correct'] + strict['missed'])}
    if source is not None:
        stages.update(handoff_metrics(source, row, receipt))
    if audit is not None:
        stages['depth'] = depth_metrics(audit)
    if receipt is not None:
        # Horizontal charts can be generated inside assembly after charts_seen.
        stages.update(axes=(int(receipt['oriented_charts'] > 0), 1),
                      charts=(len(selected), receipt['oriented_charts']),
                      groups=(exported, sum(receipt['component_sizes'])))
    return dict(scores=scores, stages=stages, exported=exported, unresolved=unresolved, selected=selected,
                pack_sha256=arm['pack_sha256'])


def capture(row, baseline, candidates, run_root=None):
    source = None
    audit = row.get('audit')
    if run_root is not None:
        source = run_root / 'source' / 'sky' / row['id'] / 'source-handoff'
        raw = run_root / 'rows' / f"{row['id']}.json.gz"
        if raw.is_file():
            replay = read(raw)
            if replay['source_sha256'] != row['source_sha256'] or replay['reference_sha256'] != row['reference_sha256']:
                raise ValueError(f'Source diagnostics belong to another frame: {raw}')
            audit = replay['audit']
    cases = {key: score_arm(row, key, source, audit if i else None)
             for i, key in enumerate((baseline, *candidates))}
    changes = {}
    for key in candidates:
        if 'arms' in row:
            pairs = row.get('paired') if (baseline, key) == ('split-border', 'current') else None
        else:
            pairs = row['paired'].get(key) if baseline in ('adopted', 'noop') else None
            if pairs is not None:
                pairs = {short: pairs[full] for short, full in (
                    ('gain', 'gained'), ('loss', 'lost'), ('add', 'added'), ('remove', 'removed'))}
                exact = row.get('exact_paired', {}).get(key)
                if exact is not None:
                    pairs.update(exact_gain=exact['gained'], exact_loss=exact['lost'])
        a, b = cases[baseline], cases[key]
        x, y = [{re.split(r'~p\d+$', k)[0] for k in v['selected']} for v in (a, b)]
        if a['unresolved'] or b['unresolved']:
            pairs = None
        changes[key] = dict(changed_pack=a['pack_sha256'] != b['pack_sha256'],
                            dropped=sorted(x - y), added=sorted(y - x),
                            disjoint=bool(x and y and x.isdisjoint(y)), paired=pairs)
    return dict(id=row['id'], panel=row.get('panel'), dataset=row['dataset'],
                source_sha256=row['source_sha256'], reference_sha256=row['reference_sha256'],
                arms=cases, changes=changes)


def percentage(counts):
    return 100 * 2 * counts['correct'] / max(1, sum(counts.values()) + counts['correct'])


def group(rows, modes, available):
    final, stages, failures = {}, {}, {}
    for mode in modes:
        arms = [row['arms'][mode] for row in rows]
        pools = {}
        for name in (f'{m}_{r}' for m in ('geometry', 'scores') for r in REGIMES):
            totals = Counter()
            for arm in arms:
                totals.update(arm['scores'][name])
            pools[name] = dict(totals, f1_percent=percentage(totals) if arms else None)
        final[mode] = pools
        stages[mode] = {}
        for name in MEASURES:
            values = [arm['stages'][name] for arm in arms if name in arm['stages']]
            if values:
                stages[mode][name] = dict(numerator=sum(x for x, _ in values),
                                          denominator=sum(y for _, y in values), views=len(values))
        failures[mode] = dict(unresolved=sum(arm['unresolved'] for arm in arms),
                              empty=sum(arm['exported'] == 0 for arm in arms))
    swaps = {}
    for mode in modes[1:]:
        cases = [dict(id=row['id'], panel=row['panel'], **row['changes'][mode]) for row in rows]
        paired = [case['paired'] for case in cases if case['paired'] is not None]
        changes = Counter()
        for case in paired:
            changes.update(case)
        regressions = []
        for row in rows:
            before, after = [row['arms'][key]['scores']['geometry_strict']['correct'] for key in (modes[0], mode)]
            if after < before:
                regressions.append(dict(id=row['id'], panel=row['panel'], before=before, after=after))
        regressions.sort(key=lambda case: (case['after'] - case['before'], case['id']))
        swaps[mode] = dict(changed_packs=sum(case['changed_pack'] for case in cases),
                           disjoint_groups=sum(case['disjoint'] for case in cases),
                           paired=dict(changes), paired_views=len(paired), regressions=regressions,
                           cases=[case for case in cases if case['dropped'] or case['disjoint']])
    return dict(views=len(rows), available=available, scores=final, stages=stages, failures=failures,
                component_changes=swaps)


def markdown(groups, names, cohort=None, perfect=None, carry_forward=None):
    report = ['# Floating evaluation from saved receipts', '',
              'Stage rows measure different opportunities. Assembly cells can overlap across components.',
              'A fresh atlas fit can pass its local checks and still be left out of the export.',
              'Unresolved nonempty exports receive no correct placements. Paired counts omit unresolved views.', '']
    if cohort:
        report[2:2] = [f'**Cohort** {cohort}', '']
    report[2:2] = ['## Decision for the next round', '',
                   carry_forward or 'Decision for this comparison is pending.', '']
    for name, totals in groups.items():
        report.extend([f'## {name}', ''])
        n = totals['views']
        if not n:
            report.extend(['Unmeasured. A separate Decorative run is available on request.', ''])
            continue
        available = totals['available']
        views = f'{n}/{available}' if available is not None else f'{n} tested'
        report.extend(['| Method | Tested / available | Strict / facade F1 | Strict / facade position + ID F1 | Correct / missed / extra | Unresolved / empty |',
                       '| --- | ---: | ---: | ---: | ---: | ---: |'])
        for key, values in totals['scores'].items():
            def f1(measure):
                return ' / '.join(f"{values[f'{measure}_{regime}']['f1_percent']:.2f}%" for regime in REGIMES)
            counts = ' / '.join(f"{values['geometry_strict'][m]:,}" for m in COUNTS)
            failures = ' / '.join(str(totals['failures'][key][m]) for m in ('unresolved', 'empty'))
            report.append(f'| {names[key]} | {views} | {f1("geometry")} | {f1("scores")} | {counts} | {failures} |')
        report.extend(['', '### Stage measures', '',
                       '| Stage | Measure | ' + ' | '.join(names.values()) + ' |',
                       '| --- | --- | ' + ' | '.join('---:' for _ in names) + ' |'])
        for key, (stage, meaning) in MEASURES.items():
            cells = []
            for metrics in totals['stages'].values():
                datum = metrics.get(key)
                cell = 'Unmeasured' if datum is None else f"{datum['numerator']:,} / {datum['denominator']:,}"
                if datum and datum['views'] != n:
                    cell += f" ({datum['views']}/{n} views)"
                cells.append(cell)
            report.append(f'| {stage} | {meaning} | ' + ' | '.join(cells) + ' |')
        if perfect and name == DATASETS[0]:
            from benchmark.floating_perfect_report import markdown as perfect_markdown
            report.extend(['', *perfect_markdown(perfect)])
        report.extend(['', '### Component changes', ''])
        for arm, changes in totals['component_changes'].items():
            report.append(f"**{names[arm]}** changed {changes['changed_packs']}/{n} packs. "
                          f"Across {changes['disjoint_groups']} views, the selected and prior groups shared no parent chart.")
            if changes['paired_views']:
                p = changes['paired']
                totals_line = (f"On {changes['paired_views']} paired views, correct gains / losses "
                               f"{p['gain']:,} / {p['loss']:,} and extras added / removed {p['add']:,} / {p['remove']:,}.")
                if 'exact_gain' in p:
                    totals_line += f" Exact IDs gained / lost {p['exact_gain']:,} / {p['exact_loss']:,}."
                report.extend(['', totals_line])
            report.append('')
            cases = sorted(changes['cases'], key=lambda case: (
                not case['disjoint'], -(case['paired'] or {}).get('loss', 0), case['id']))
            for case in cases[:5]:
                old = ', '.join(case['dropped']) or 'none'
                new = ', '.join(case['added']) or 'none'
                line = f"- `{case['id']}` dropped {old} and added {new}."
                if case['paired'] is not None:
                    p = case['paired']
                    line += f" Correct gains / losses {p['gain']} / {p['loss']}. Extras added / removed {p['add']} / {p['remove']}."
                report.append(line)
            report.append('')
            regressions = changes['regressions'][:5]
            if regressions:
                report.extend(['Largest declines in correct placement', ''])
                report.extend(f"- `{case['id']}` went from {case['before']} to {case['after']} correct."
                              for case in regressions)
                report.append('')
    return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('conditions', type=Path, help='Saved Floating comparison conditions.json.gz')
    parser.add_argument('--baseline', default='split-border')
    parser.add_argument('--baseline-name', help='Human-readable description of the baseline')
    parser.add_argument('--candidate', action='append', help='Candidate arm key. Repeat to compare several.')
    parser.add_argument('--candidate-name', action='append', help='Description of each candidate, in arm order')
    parser.add_argument('--run-root', type=Path, help='Optional V1 run root with source handoffs and depth audit rows')
    parser.add_argument('--cohort-label', help='Name of the actual saved comparison cohort')
    parser.add_argument('--carry-forward', help='Next baseline, each candidate disposition, and what landed')
    parser.add_argument('--visible-available', type=int, help='Size of the named Visible Builds cohort')
    parser.add_argument('--decorative-available', type=int, default=40)
    parser.add_argument('--perfect', type=Path, help='Optional saved Perfect Stage controls for this exact comparison')
    parser.add_argument('--perfect-inputs', type=Path, help='Folder of early handoff controls matching --perfect')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidates = list(args.candidate or ('current',))
    modes = (args.baseline, *candidates)
    if len(set(modes)) != len(modes):
        parser.error('Choose distinct Baseline and Candidate arm keys.')
    labels = args.candidate_name or candidates
    if len(labels) != len(candidates):
        parser.error('Supply one --candidate-name per candidate, or omit the labels.')
    names = dict(zip(modes, [f'Baseline — {args.baseline_name or args.baseline}', *[f'Candidate {i} — {label}'
                 for i, label in enumerate(labels, 1)]], strict=True))
    frames = [capture(row, args.baseline, candidates, args.run_root) for row in read(args.conditions)]
    keys = {(r['dataset'], r['panel'], r['id']) for r in frames}
    if not frames or len(keys) != len(frames):
        raise ValueError('Expected a nonempty, unique list of Floating conditions.')
    split = {DATASETS[0]: [], DATASETS[1]: []}
    for row in frames:
        split[DATASETS[row['dataset'] == 'decorative-builds-v1']].append(row)
    available = dict(zip(DATASETS, (args.visible_available, args.decorative_available), strict=True))
    data = {key: group(rows, modes, available[key]) for key, rows in split.items()}
    comparison_sha = hash_file(args.conditions)
    perfect = None
    if args.perfect_inputs and not args.perfect:
        parser.error('--perfect-inputs requires --perfect.')
    if args.perfect:
        from benchmark.floating_perfect_report import EARLY, from_archive, viewer
        early = {stage: read(args.perfect_inputs / f'{stage}-conditions.json.gz') for stage in EARLY
                 if (args.perfect_inputs / f'{stage}-conditions.json.gz').is_file()} if args.perfect_inputs else None
        if args.perfect_inputs and not early:
            parser.error('--perfect-inputs has no recognized early handoff archives.')
        perfect = from_archive(read(args.perfect), comparison_sha, frames, names, early)
    args.out.mkdir(parents=True, exist_ok=True)
    if perfect:
        (args.out / 'perfect-stage.html').write_text(viewer(perfect, args.cohort_label), encoding='utf8', newline='\n')
    (args.out / 'report.md').write_text(markdown(data, names, args.cohort_label, perfect, args.carry_forward),
                                        encoding='utf8', newline='\n')
    (args.out / 'report.json').write_text(json.dumps(dict(schema=1, comparison_sha256=comparison_sha,
        cohort_label=args.cohort_label, names=names, cohorts=data, frames=frames, perfect_stage=perfect,
        carry_forward=args.carry_forward), separators=(',', ':')) + '\n',
        encoding='utf8', newline='\n')


if __name__ == '__main__':
    main()
