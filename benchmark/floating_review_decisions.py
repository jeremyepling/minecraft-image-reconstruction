"""Turn paired Floating receipts into testable decisions for the next review."""
import hashlib
import json
import math
import re
from collections import defaultdict

from benchmark.floating_perfect_report import COUNTS, METRICS, aggregate


# These handoffs have different consumers. Keep the writer control out of opportunity rankings.
HANDOFFS = {
    'Surface Finder': ('surface', 'Grouping visible pixels by the source-image contour may preserve complete faces.',
                       'Compare native visible ownership and openings, then rerun ordinary counting and assembly.'),
    'Surface and depth separation': ('depth', 'Separating parallel visible planes before counting may prevent a grid spanning two depths.',
                                     'Compare each source-only split with native plane ownership and rerun counts, joins and export.'),
    'Block Counter': ('grid', 'A full visible face may constrain grid pitch and origin better than an internal texture repeat.',
                      'Compare a bounded chooser over the saved grids, including the current grid, with exact visible wall grids on the same views. If proposals contain the right grid, test a source-only chooser. Otherwise test full-face grid proposals on other buildings.'),
    'Relative joins': ('joins', 'Adjacent visible edges may constrain the relative integer offset.',
                       'Correct only the join offset, rerun component selection and compare recovered positions and new extras.'),
    'Component selection': ('component', 'Keeping a compatible neighboring wall may recover visible positions lost in a component swap.',
                            'Choose among existing components including the current selection, then decode under one whole-build translation. Test a source-only attachment if the chooser recovers visible positions.'),
    'Local heading': ('heading', 'Visible axis and texture direction may distinguish the available headings.',
                      'Compare a source-image heading choice against the current choice on unresolved views and resolved controls.'),
    'Material assignment': ('material', 'Sampling an interior patch of each projected face may disambiguate block IDs on correctly placed cells.',
                            'Hold placement fixed and compare source-only material assignments on correctly placed visible blocks across buildings.'),
    'Pack writing': ('writer', '', ''),
}

# About a thousand extras merits a complete cohort sweep, including the archived 917-extra case.
EXTRA_AUDIT_THRESHOLD = 900


def _frame_map(report):
    frames = defaultdict(list)
    for frame in report['frames']:
        if frame['dataset'] != 'decorative-builds-v1':
            frames[frame['id']].append(frame)
    return frames


def _capture_group(marker):
    """V1 archive naming encodes the capture split and building family."""
    match = re.match(r'^(train|validation|test)-([a-z]+)[0-9]+-', marker)
    if match:
        return match[1], match[2]
    legacy = re.match(r'^legacy-[a-z_]+-(?:dev|holdout)-([a-z]+)-', marker)
    return ('regression', legacy[1]) if legacy else ('unclassified', 'unclassified')


def _matched_frame(frames, case, pack, arm):
    matches = [frame for frame in frames.get(case['id'], [])
               if case['source_sha256'] == frame['source_sha256']
               and case['reference_sha256'] == frame['reference_sha256']
               and pack == frame['arms'][arm]['pack_sha256']]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous or mismatched frame receipt for {case['id']} on {arm}")
    return matches[0]


def _category(row):
    kind = row['kind'].lower()
    if row['stage'] == 'Pack writing' or 'writer' in kind:
        return 'writer-check'
    if 'chooser' in kind:
        return 'bounded-chooser'
    if kind.startswith('exact'):
        return 'exact-input'
    return 'other'


def _rank(row):
    delta = row['deltaCounts']['geometry_strict']
    return (-delta['correct'], delta['extras'], -row['delta']['material_strict'], row['stage'], row['kind'])


def _diagnostics(report, frames):
    perfect = report.get('perfect_stage') or {}
    groups = {}
    anchors = [key for key, name in report['names'].items() if name == perfect.get('anchor')]
    if perfect.get('rows') and len(anchors) != 1:
        raise ValueError('Perfect Stage rows require one matching source-only arm.')
    for row in perfect.get('rows', []):
        anchor = anchors[0]
        cases = row['cases']
        identities = []
        for case in cases:
            frame = _matched_frame(frames, case, case['base_pack_sha256'], anchor)
            identities.append((frame['dataset'], frame.get('panel') or '', frame['id'],
                               frame['source_sha256'], frame['arms'][anchor]['pack_sha256']))
            for metric in METRICS:
                scored = frame['arms'][anchor]['scores'][metric.replace('material_', 'scores_')]
                if any(scored[k] != case['before'][metric][k] for k in COUNTS):
                    raise ValueError(f"Diagnostic baseline differs from the matched export for {case['id']}")
        if not cases or len(set(identities)) != len(cases) or row['tested'] != len(cases):
            raise ValueError('Perfect Stage coverage differs from its distinct frame receipts.')
        before, after = [aggregate([case[key] for case in cases]) for key in ('before', 'after')]
        if before != row['before'] or after != row['after']:
            raise ValueError(f"Perfect Stage totals differ from their receipts for {row['stage']}")
        digest = hashlib.sha256(json.dumps(sorted(identities), separators=(',', ':')).encode()).hexdigest()
        group = groups.setdefault(digest, dict(cohort=row['cohort'], tested=len(cases), anchor=perfect['anchor'],
                                               frameSetSha256=digest, rows=[]))
        group['rows'].append(dict(stage=row['stage'], kind=row['kind'], change=row['change'], category=_category(row),
                                  before=before, after=after,
                                  delta={metric: after[metric]['f1'] - before[metric]['f1'] for metric in METRICS},
                                  deltaCounts={metric: {k: after[metric][k] - before[metric][k] for k in COUNTS}
                                               for metric in METRICS}))
    for group in groups.values():
        group['rows'].sort(key=lambda r: (r['category'] in ('writer-check', 'other'), _rank(r)))
        group['missing'] = [name for name in HANDOFFS if name not in {r['stage'] for r in group['rows']}]
    return sorted(groups.values(), key=lambda g: (-g['tested'], g['frameSetSha256']))


def _card(key, stage, theme, title, evidence, hypothesis, test, *, cases=(), basis='source-only'):
    return dict(id=key, stage=stage, theme=theme, title=title, disposition='machine', evidence=evidence,
                hypothesis=hypothesis, test=test, caseIds=sorted(set(cases)), basis=basis, status='open')


def _opportunities(groups, candidate_name):
    cards = []
    seen = set()
    for group in groups:
        # Opportunities for an earlier arm are informative, but need a new receipt before guiding this candidate.
        if group['anchor'] != candidate_name:
            continue
        for row in group['rows']:
            stage = row['stage']
            category = row['category']
            if stage not in HANDOFFS or category not in ('exact-input', 'bounded-chooser') or stage in seen:
                continue
            delta = row['deltaCounts']['geometry_strict']
            if delta['correct'] <= 0 and delta['extras'] >= 0 and row['delta']['material_strict'] <= 0:
                continue
            seen.add(stage)
            theme, hypothesis, test = HANDOFFS[stage]
            remaining = row['after']['geometry_strict']['missed']
            scope = f"{group['tested']} matched views ({group['cohort']})"
            evidence = (f"{row['kind']} on {scope} changed strict correct positions by {delta['correct']:+,}, "
                        f"extras by {delta['extras']:+,}, strict placement F1 by {row['delta']['geometry_strict']:+.2f} pp "
                        f"and strict position + material F1 by {row['delta']['material_strict']:+.2f} pp. "
                        f"{remaining:,} visible positions remained missing after this intervention.")
            card = _card('replay-' + theme, stage, theme, f'Test source evidence for {stage.lower()}', evidence,
                         hypothesis, test, basis='reference-assisted')
            card.update(diagnosticCategory=category, frameSetSha256=group['frameSetSha256'])
            cards.append(card)
    return cards


def _queue_cases(report, receipts, frames, baseline, candidate):
    if receipts['comparison_sha256'] != report['comparison_sha256']:
        raise ValueError('The review queue and report belong to different comparisons.')
    cases = []
    for case in receipts['cases']:
        if case.get('cohort') == 'Decorative Builds V1':
            continue
        frame = _matched_frame(frames, case, case['candidate']['pack_sha256'], candidate)
        if frame['arms'][baseline]['pack_sha256'] != case['baseline']['pack_sha256']:
            raise ValueError(f"Review baseline differs for {case['id']}")
        paired = frame['changes'][candidate]['paired']
        if paired is None or any(paired[k] != case['paired'].get(k) for k in paired):
            raise ValueError(f"Paired review counts differ for {case['id']}")
        if case.get('component') and (set(case['component']['dropped']) != set(frame['changes'][candidate]['dropped'])
                                      or set(case['component']['added']) != set(frame['changes'][candidate]['added'])):
            raise ValueError(f"Component receipt differs for {case['id']}")
        cases.append(case)
    return cases


def _machine_checks(frames, cases, candidate, chooser_covered):
    rows = [f for family in frames.values() for f in family]
    checks = []
    outliers = [(f['id'], f['arms'][candidate]['scores']['geometry_strict']['extras']) for f in rows
                if f['arms'][candidate]['scores']['geometry_strict']['extras'] >= EXTRA_AUDIT_THRESHOLD]
    if outliers:
        worst, count = max(outliers, key=lambda pair: (pair[1], pair[0]))
        checks.append(_card('audit-full-face-grid', 'Block Counter', 'grid',
                            'Trace every dense export to its grid seed',
                            f'{len(outliers)} of {len(rows)} views had at least {EXTRA_AUDIT_THRESHOLD:,} extras. '
                            f'{worst} had {count:,}.',
                            'A partial face, an internal texture repeat or a grid crossing a depth boundary could expand into many cells.',
                            'For every flagged view, compare grid width and height with a complete visible face. Trace the seed, mask boundary and first unsupported exported coordinate. Test a source-only grid gate on these views and low-extra controls.',
                            cases=[marker for marker, _ in outliers]))
    swapped = [f for f in rows if f.get('changes', {}).get(candidate, {}).get('disjoint')]
    if swapped:
        paired = [f['changes'][candidate]['paired'] for f in swapped if f['changes'][candidate]['paired'] is not None]
        evidence = (f'{len(swapped)} of {len(rows)} views switched to a component with no shared parent chart. '
                    f"On {len(paired)} registered swaps, {sum(p['gain'] for p in paired):,} positions were gained "
                    f"and {sum(p['loss'] for p in paired):,} previously correct positions were lost.")
        checks.append(_card('audit-component-swap', 'Component selection', 'component',
                            'Replay the dropped components', evidence, HANDOFFS['Component selection'][1],
                            'Trace one lost coordinate per dropped chart back through the grid and join. Replay keeping the prior chart when its visible edge fits the selected group. Count recovered and lost correct positions and added extras.',
                            cases=[f['id'] for f in swapped]))
    for field, title, stage, hypothesis, test in [
            ('unresolved', 'Resolve the heading before comparing placements', 'Local heading',
             'A viable heading may exist among the offered orientations.',
             'Score the existing headings with the current output as a control. If one recovers visible coordinates, test source-image cues on other buildings.'),
            ('empty', 'Trace empty exports to the first missing handoff', 'Decoded placement',
             'A counted surface may lose its last eligible chart during orientation or assembly.',
             'Trace mask, grid, orientation and selected component through decoding. Rerun the first failed handoff and a nonempty control.')]:
        affected = [f['id'] for f in rows if (f['arms'][candidate]['exported'] == 0 if field == 'empty'
                                             else f['arms'][candidate]['unresolved'])]
        if affected:
            if field == 'unresolved' and chooser_covered:
                test = ('Use the checksum-matched heading chooser to identify recoverable views. Test source-only axis and texture cues on other buildings and compare against the current choice on resolved controls.')
            checks.append(_card('audit-' + field, stage, 'orientation' if field == 'unresolved' else 'export', title,
                                f"{len(affected)} of {len(rows)} views had " +
                                ('an empty export.' if field == 'empty' else 'a nonempty export with unresolved orientation.'),
                                hypothesis, test, cases=affected))
    losses = [c for c in cases if c['paired']['loss'] and not c.get('component')]
    if losses:
        checks.append(_card('trace-paired-loss', 'Decoded placement', 'placement', 'Trace changed coordinates to their first lost handoff',
                            f"{len(losses)} registered review overlays with no dropped-component overlay contain "
                            f"{sum(c['paired']['loss'] for c in losses):,} previously correct positions lost.",
                            'Grid scale, parallel depth or ownership at a shared edge could explain different lost coordinates.',
                            'Trace the visible owner of a lost coordinate through its grid, plane, selected chart and decoded placement. Replay the first differing handoff on a gain and a loss.',
                            cases=[c['id'] for c in losses]))
    return checks


def _surface_checks(evidence, cases):
    by_id = {case['id']: case for case in cases}
    events, split_records, conflicts = {}, {}, defaultdict(set)
    for marker, package in (evidence or {}).items():
        case = by_id.get(marker)
        if case is None or package['id'] != marker or package['sourceSha256'] != case['source_sha256']:
            raise ValueError(f'Surface evidence belongs to a different source: {marker}')
        if package.get('schema') == 2:
            for label, arm in [('Baseline', 'baseline'), ('Candidate', 'candidate')]:
                if package['provenance']['arms'][label]['packSha256'] != case[arm]['pack_sha256']:
                    raise ValueError(f'Surface evidence uses a different {label} export for {marker}')
        if package.get('schema') not in (1, 2):
            raise ValueError(f'Unknown surface evidence schema for {marker}')
        for surface in package['surfaces']:
            parts = surface.get('parts') or {}
            if len(parts) > 1:
                split_records.setdefault(marker, []).append((surface['id'], len(parts)))
            if package['schema'] < 2:
                continue
            for event in (surface.get('exports', {}).get('Candidate') or {}).get('overlaps', []):
                events.setdefault(marker, []).append(event)
            for decision in surface.get('decisions', []):
                if decision['arm'] == 'Candidate' and decision['kind'] == 'conflict':
                    conflicts[marker].add(json.dumps(decision['detail'], sort_keys=True))
    checks = []
    if events:
        marker = sorted(events)[0]
        example = events[marker][0]
        checks.append(_card('audit-chart-ownership', 'Shared cells', 'ownership', 'Trace first-writer ownership at shared cells',
                            f"{sum(map(len, events.values())):,} losing cell writes in {len(events)} reviewed views. "
                            f"In {marker}, {example['chart']} lost {example['xyz']} to {example['winnerChart']}.",
                            'At a visible corner, the winning chart may assign a different face material or retain a displaced edge.',
                            'At each shared coordinate compare the winner and losing face with the recorded visible owner. Trace selection, decoded material and the placement under the same translation before testing a corner rule.',
                            cases=events))
    if split_records:
        count = sum(map(len, split_records.values()))
        checks.append(_card('audit-depth-boundary', 'Surface and depth separation', 'depth', 'Check where split surfaces meet',
                            f'{count} parent masks in {len(split_records)} reviewed views have multiple saved depth partitions.',
                            'A grid may extend across a parallel plane at a different depth after splitting or attachment.',
                            'Compare the split edge to frontmost pixel ownership; trace each proposed grid into the decoded export. Replay an offset change only where visible faces support it.',
                            cases=split_records))
    if conflicts:
        checks.append(_card('audit-join-conflict', 'Relative joins', 'joins', 'Trace competing join offsets',
                            f'{sum(map(len, conflicts.values()))} distinct candidate conflict receipts across {len(conflicts)} reviewed views.',
                            'Conflicting relative offsets may discard an otherwise visible neighbor.',
                            'Trace each rejected offset and the accepted join through the selected component to its decoded blocked or recovered positions.',
                            cases=conflicts))
    return checks


def _missing_card(candidate_stats, missing):
    stage = next((name for name in ('Block Counter', 'Relative joins', 'Component selection',
                                    'Surface and depth separation', 'Local heading', 'Material assignment',
                                    'Surface Finder', 'Pack writing') if name in missing), None)
    if stage is None:
        return None
    fields = {'Block Counter': ('counter', 'surfaces have both dimensions'),
              'Relative joins': ('charts', 'oriented charts reached the selected group'),
              'Component selection': ('groups', 'assembled cells contributed to the exported block count')}
    measurement = ''
    if stage in fields:
        key, description = fields[stage]
        values = candidate_stats.get(key)
        if values:
            measurement = (f"{values['numerator']:,} / {values['denominator']:,} {description} "
                           f"across {values['views']} measured views. ")
    hypothesis = HANDOFFS[stage][1] or 'Exact visible coordinates and material IDs should survive serialization.'
    return _card('measure-' + HANDOFFS[stage][0], stage, HANDOFFS[stage][0],
                 'Measure the independent ' + stage.lower() + ' intervention',
                 measurement + 'No matched independent reference replay for this handoff on the candidate was supplied.',
                 hypothesis, 'Replace only this handoff with its recorded visible answer, include a no-op, '
                 'rerun the same downstream consumers and score decoded correct positions, extras and material F1.',
                 basis='unmeasured')


def _measured_outcome(entry):
    values = entry.get('measuredDelta') or {}
    fields = ('correct', 'missed', 'extras', 'geometry_strict', 'material_strict')
    if (entry.get('measuredEffect') is not True or not isinstance(entry.get('measuredViews'), int)
            or entry['measuredViews'] < 1 or not re.fullmatch(r'[0-9a-f]{64}', str(entry.get('measuredComparison', '')))
            or not all(type(values.get(k)) in (int, float) and math.isfinite(values[k]) for k in fields)):
        return None
    return dict(comparisonSha256=entry['measuredComparison'], views=entry['measuredViews'],
                delta={field: values[field] for field in fields})


def _questions(ledger, frames, cases):
    pictured = {c['id'] for c in cases}
    entries = ledger.get('entries', [])
    cards = []
    metrics = dict(proposed=0, presented=0, pending=0, answered=0, tested=0, changedExperiment=0,
                   measuredEffects=0, byType={})
    outcomes = []
    # One note may acquire a test outcome in a later round. Count unique answered questions.
    history = defaultdict(list)
    for entry in entries:
        if entry.get('actionId') and entry.get('questionType') in ('visible-ambiguity', 'preference'):
            history[(entry['actionId'], entry.get('caseId'), entry.get('sourceSha256'))].append(entry)
    for (action_id, _, _), records in history.items():
        answered = [e for e in records if e.get('status') in ('answered', 'tested') and e.get('answer')]
        if not answered:
            continue
        tested = [e for e in records if e.get('status') == 'tested']
        kind = answered[-1]['questionType']
        counts = metrics['byType'].setdefault(kind, dict(answered=0, tested=0, changedExperiment=0,
                                                        measuredEffects=0, gainedCorrect=0, reducedExtras=0, gainedMaterial=0))
        changed = any(e.get('changedExperiment') is True for e in records)
        for name, increment in [('answered', 1), ('tested', bool(tested)), ('changedExperiment', changed)]:
            counts[name] += increment
            metrics[name] += increment
        # Private commentary, answers and free-text outcomes stay off the public manifest.
        if re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}', action_id):
            outcome = dict(actionId=action_id, status='tested' if tested else 'answered', changedExperiment=changed)
            measurements = [m for entry in tested if (m := _measured_outcome(entry)) is not None]
            if measurements:
                outcome['measured'] = measurements[-1]
                metrics['measuredEffects'] += 1
                counts['measuredEffects'] += 1
                delta = measurements[-1]['delta']
                counts['gainedCorrect'] += delta['correct'] > 0
                counts['reducedExtras'] += delta['extras'] < 0
                counts['gainedMaterial'] += delta['material_strict'] > 0
            outcomes.append(outcome)
    checked_themes = set()
    gates = sorted(ledger.get('questions', []), key=lambda g: (g.get('priority', 99), g['id']))
    for gate in gates:
        marker = gate['caseId']
        key = gate['id']
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}', key):
            raise ValueError('Question IDs must be stable lower-case identifiers.')
        kind = gate['type']
        if kind not in ('visible-ambiguity', 'preference'):
            raise ValueError(f'Unsupported human question type: {kind}')
        if marker not in pictured or not any(f['source_sha256'] == gate['sourceSha256'] for f in frames[marker]):
            continue
        if any(not isinstance(gate.get(field), str) or not gate[field].strip()
               for field in ('theme', 'stage', 'prompt', 'cue', 'firstPass', 'checked')):
            raise ValueError(f'Question {key} lacks a cue, prior check or actionable first pass.')
        choices = gate['choices']
        if (len(choices) < 2 or any(not c.get('answer') or not c.get('experiment') for c in choices)
                or len({c['experiment'].strip() for c in choices}) < 2
                or len({c['answer'].strip() for c in choices}) != len(choices)):
            raise ValueError(f'Question {key} needs distinct answers that change the next experiment.')
        records = history.get((key, marker, gate['sourceSha256']), [])
        if any(e.get('status') in ('answered', 'tested') and e.get('answer') for e in records):
            continue
        metrics['proposed'] += 1
        if gate['theme'] in checked_themes or len(cards) >= 3:
            continue
        checked_themes.add(gate['theme'])
        card = _card(key, gate['stage'], gate['theme'], gate['prompt'], gate['checked'],
                     gate['firstPass'], ' · '.join(f"{c['answer']} → {c['experiment']}" for c in choices),
                     cases=[marker])
        card.update(disposition='question', status='pending',
                    question=dict(id=key, type=kind, prompt=gate['prompt'], cue=gate['cue'],
                                  choices=[dict(answer=c['answer'], experiment=c['experiment']) for c in choices]))
        cards.append(card)
    metrics.update(presented=len(cards), pending=metrics['proposed'])
    return cards, sorted(outcomes, key=lambda o: o['actionId']), metrics


def _chooser_pilot(report, frames, baseline, candidate, groups, evidence):
    # Both arms are existing, scored source-only exports. The reference selects among them only in this diagnostic.
    def objective(scores):
        return (scores['geometry_strict']['correct'], -scores['geometry_strict']['extras'],
                scores['scores_strict']['correct'])

    def metrics(scores):
        return {k: scores[k.replace('material_', 'scores_')] for k in METRICS}

    current, best, changes = [], [], 0
    for family in frames.values():
        for frame in family:
            if baseline == candidate:
                continue
            before, after = [frame['arms'][arm]['scores'] for arm in (baseline, candidate)]
            use_before = objective(before) > objective(after)
            changes += use_before
            current.append(metrics(after))
            best.append(metrics(before if use_before else after))
    base_scores, best_scores = (aggregate(current), aggregate(best)) if current else (None, None)
    pack = dict(tested=len(current), switches=changes if current else None, before=base_scores, after=best_scores,
                status='measured' if current else 'unmeasured',
                options=[report['names'][arm] for arm in dict.fromkeys((candidate, baseline))],
                criterion='More correct visible placements, then fewer extras, then more exact material IDs. Keep the candidate on ties.')

    reviewed = dict(views=len(evidence), parents=0, multipleGridLayers=0, splits={}, families=[], sourceFitEvidence=0)
    families = set()
    for marker, package in evidence.items():
        # The V1 capture IDs encode the family and split. Leave unfamiliar naming schemes unclassified.
        split, family = _capture_group(marker)
        reviewed['splits'][split] = reviewed['splits'].get(split, 0) + 1
        if family != 'unclassified':
            families.add(family)
        for surface in package['surfaces']:
            reviewed['parents'] += 1
            reviewed['multipleGridLayers'] += len({(layer.get('chart'), layer['path']) for layer in surface['layers']
                                                  if layer.get('path')}) > 1
            reviewed['sourceFitEvidence'] += sum(bool(part.get('fitEvidence')) for part in surface.get('parts', {}).values())
    reviewed['families'] = sorted(families)
    heading = {}
    for row in (report.get('perfect_stage') or {}).get('rows', []):
        if (row['stage'] != 'Local heading' or _category(row) != 'bounded-chooser'
                or (report.get('perfect_stage') or {}).get('anchor') != report['names'][candidate]):
            continue
        for case in row['cases']:
            frame = _matched_frame(frames, case, case['base_pack_sha256'], candidate)
            heading.setdefault((case['split'], case['id'], case['base_pack_sha256']),
                               (int(frame['arms'][candidate]['unresolved']),
                                case['after']['geometry_strict']['correct'] - case['before']['geometry_strict']['correct']))
    heading_splits = {}
    for (split, _, _), (unresolved, correct) in heading.items():
        measured = heading_splits.setdefault(split, dict(views=0, unresolved=0, correctGain=0))
        measured['views'] += 1
        measured['unresolved'] += unresolved
        measured['correctGain'] += correct

    grid_choosers = [r for g in groups if g['anchor'] == report['names'][candidate] for r in g['rows']
                     if r['stage'] == 'Block Counter' and r['category'] == 'bounded-chooser']
    gaps = []
    if not current:
        gaps.append('A second ordinary scored export on the same views is needed for the final-pack chooser.')
    if not grid_choosers:
        gaps.append('Measure a bounded grid chooser over the saved alternatives, including the incumbent, with an ordinary decoded export for each decision.')
    if not reviewed['splits'].get('test'):
        gaps.append('Export source-only grid features and option-level outcomes for untouched building families.')
    gaps.append('Link each offered option to its source-only features, building split and paired decoded score before fitting a chooser.')
    return dict(status='defer-training', exportChooser=pack, reviewedGrids=reviewed, headingBySplit=heading_splits,
                gridChooserMeasured=bool(grid_choosers), missing=gaps)


def analyze(report, queue_receipts=None, candidate='current', *, ledger=None, surface_evidence=None):
    """Analyze the Visible Builds arm from saved receipts. No inference, reference lookups or publication."""
    baseline = next(iter(report['names']))
    single = len(report['names']) == 1
    if single and candidate == 'current':
        candidate = baseline
    if (candidate == baseline and not single) or candidate not in report['names']:
        raise ValueError('Choose a reported source-only candidate distinct from the baseline for a comparison.')
    stats = report['cohorts']['Visible Builds']
    frames = _frame_map(report)
    if sum(map(len, frames.values())) != stats['views']:
        raise ValueError('Visible cohort size differs from the stage report frames.')
    if single and queue_receipts and queue_receipts.get('cases'):
        raise ValueError('Paired review cases require two distinct scored arms.')
    cases = _queue_cases(report, queue_receipts, frames, baseline, candidate) if queue_receipts is not None else []
    groups = _diagnostics(report, frames)
    stages = {r['stage'] for g in groups if g['anchor'] == report['names'][candidate] for r in g['rows']}
    missing = [stage for stage in HANDOFFS if stage not in stages]
    cards = _opportunities(groups, report['names'][candidate])
    chooser_covered = any(g['tested'] == stats['views'] and g['anchor'] == report['names'][candidate]
                         and any(row['stage'] == 'Local heading' and row['category'] == 'bounded-chooser'
                                 for row in g['rows']) for g in groups)
    checks = _machine_checks(frames, cases, candidate, chooser_covered)
    checks.extend(_surface_checks(surface_evidence, cases))
    measurement = _missing_card(stats['stages'][candidate], missing)
    if measurement and frames:
        checks.append(measurement)
    # The first investment is supported by the widest matched replay. Follow with concrete positions
    # to trace before lower-priority models, including material-only opportunities.
    cards = cards[:1] + checks + cards[1:]
    questions, outcomes, metrics = _questions(ledger or {}, frames, cases)
    cards.extend(questions)
    ids = [card['id'] for card in cards]
    if len(set(ids)) != len(ids):
        raise ValueError('Question IDs must be unique and distinct from machine action IDs.')
    for priority, card in enumerate(cards, 1):
        card['priority'] = priority
        by_family = defaultdict(list)
        for marker in card['caseIds']:
            by_family[_capture_group(marker)[1]].append(marker)
        card['caseFamilies'] = dict(sorted(by_family.items()))
    pilot = _chooser_pilot(report, frames, baseline, candidate, groups, surface_evidence or {})
    return dict(schema=1, actions=cards, diagnosticGroups=groups, diagnosticsMissing=missing, chooserPilot=pilot,
                outcomes=outcomes, questionMetrics=metrics)
