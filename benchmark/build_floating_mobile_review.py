"""Build a compact static run manifest and GitHub-hosted review imagery."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from benchmark.floating_stage_report import MEASURES, read
from benchmark.mobile_surface_maps import maps, visible_edges


DEFAULT_LABELS = {
    'train-arcade1-cool-high-right': (
        'Arcade 1', 'Component selection',
        'Can the right wall stay attached when the roof wins?',
        'The candidate gained the roof and dropped the right wall. Check the visible corner for a shared block row.'),
    'train-terrace1-cool-high-right': (
        'Terrace 1', 'Relative joins',
        'Which visible edge would anchor the dropped wall?',
        'Try the gold grid to see whether the opening rim or the outer corner gives a clearer join.'),
    'train-sawtooth3-earth-high-right': (
        'Sawtooth 3', 'Placement',
        'What separates the pink section from the recovered blocks?',
        'This view lost the most previously correct placements. Look for a change in depth or grid spacing where pink meets cyan.'),
    'train-arcade3-cool-high-right': (
        'Arcade 3', 'Wall grid and placement',
        'Which seam would you use to carry this grid?',
        'This view gained the most correct positions. Check the visible texture seams beside the remaining losses.'),
}


def checked(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Content changed from the visual review receipt: {path}')


def reference_grid(path, owner_path):
    rgb = np.array(Image.open(path).convert('RGB'))
    with np.load(owner_path, allow_pickle=False) as channels:
        edges = visible_edges(channels['face_id'])
    if rgb.shape[:2] != edges.shape:
        raise ValueError('Reference labels and source image have different native sizes.')
    rgb[edges] = (249, 212, 102)
    return Image.fromarray(rgb)


def webp(image, target):
    image.save(target, 'WEBP', quality=86, method=6)


def make_case(case, queue, frames, assets, asset_base, labels, run_root=None, evidence_root=None, baseline_evidence_root=None):
    marker = case['id']
    split = case.get('split') or marker.split('-', 1)[0]
    frame = frames / split / 'frames' / marker
    source = frame / 'rgb.png'
    owner = frame / 'labels.npz'
    out = assets / marker
    out.mkdir(parents=True, exist_ok=True)
    checked(source, case['source_sha256'])
    checked(owner, case['reference_sha256']['labels.npz'])
    original = Image.open(source).convert('RGB')
    placement = queue / case['folder'] / 'placement.png'
    layers = [('source', 'Source', 'Unaltered visible source, shown as a mobile WebP.', original),
              ('changes', 'Changes', 'Cyan gained · pink lost · compared with the baseline.', Image.open(placement).convert('RGB')),
              ('gold', 'Gold grid', 'Recorded frontmost visible faces. Gold follows their pixel boundaries.',
               reference_grid(source, owner))]
    if case.get('component'):
        layers.append(('outline', 'Dropped', 'Amber outlines the dropped surface support.',
                       Image.open(queue / case['folder'] / 'dropped.png').convert('RGB')))
    modes = []
    readable = DEFAULT_LABELS[marker][0] if marker in DEFAULT_LABELS else marker
    title, stage, prompt, first_pass = labels.get(marker, (readable, 'Placement',
        'Where would you adjust placement?', 'Compare changes with the source and recorded visible grid.'))
    result = dict(id=marker, title=title, stage=stage, prompt=prompt, firstPass=first_pass,
                  width=original.width, height=original.height, modes=modes, counts=case['paired'],
                  sourceSha256=case['source_sha256'],
                  baselinePackSha256=case['baseline']['pack_sha256'], candidatePackSha256=case['candidate']['pack_sha256'])
    data = None
    handoffs = []
    if run_root:
        handoffs.append(run_root / 'source' / 'sky' / marker / 'source-handoff' / 'surfaces.json')
    if baseline_evidence_root:
        handoffs.extend(directory / marker / 'source-handoff' / 'surfaces.json'
                        for directory in (evidence_root, baseline_evidence_root))
    if run_root or evidence_root:
        from benchmark.mobile_surface_evidence import build, pair_evidence, reuse_evidence
        package = (evidence_root or run_root / 'evidence') / marker / 'surfaces.json'
        if baseline_evidence_root:
            pair_evidence(baseline_evidence_root / marker / 'surfaces.json', package, out / 'surfaces.json', case, original.size)
        elif evidence_root or package.is_file():
            reuse_evidence(package, out / 'surfaces.json', case, original.size)
        else:
            build(run_root, case, original, out / 'surfaces.json')
        from benchmark.floating_surface_loss_trace import trace_case
        public_package = out / 'surfaces.json'
        data = read(public_package)
        folders = None
        if baseline_evidence_root:
            folders = (baseline_evidence_root / marker, evidence_root / marker)
        elif run_root and all('score' in arm for arm in data['provenance']['arms'].values()):
            folders = tuple((run_root / data['provenance']['arms'][name]['score']).parent
                            for name in ('Baseline','Candidate'))
        if folders:
            trace = trace_case(case, data, frame, *folders)
            result['placementTrace'] = {k: trace[k] for k in ('status','reason','droppedLosses') if k in trace}
            if trace['status'] == 'verified':
                for surface in data['surfaces']:
                    surface['placementChange'] = trace['surfaces'].get(surface['id'])
        result['surfaceEvidence'] = f'{asset_base}/{marker}/surfaces.json'
    surface_layers, coverage = maps(original, frame, case['reference_sha256'], data, handoffs)
    if data is not None:
        public_package.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf-8')
    if coverage:
        result['maskCoverage'] = coverage
    for key, label, caption, image in [*layers, *surface_layers]:
        if image.size != original.size:
            raise ValueError(f'Unregistered review layer: {marker}/{key}')
        filename = f'{key}.webp'
        webp(image, out / filename)
        modes.append(dict(key=key, label=label, caption=caption, url=f'{asset_base}/{marker}/{filename}'))
    return result


def manifest(report, queue, frames, assets, asset_base, run, title, candidate, labels, run_root=None,
             evidence_root=None, baseline_evidence_root=None, *, questions=None, database=None, downloads=None,
             carry_forward=None):
    from benchmark.floating_review_decisions import analyze
    from benchmark.floating_review_intake import private_history

    saved = read(queue / 'receipts.json')
    if report['comparison_sha256'] != saved['comparison_sha256']:
        raise ValueError('The visuals and scorecard describe different comparisons.')
    visible = report['cohorts']['Visible Builds']
    baseline = next(iter(report['names']))
    if baseline == candidate:
        raise ValueError('Choose a distinct candidate.')
    recorded_frames = {(frame['id'], frame.get('panel')): frame for frame in report['frames']
                       if frame['dataset'] != 'decorative-builds-v1'}
    cases = [case for case in saved['cases'] if case.get('cohort') != 'Decorative Builds V1']
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Use a distinct mobile comparison per panel when case IDs repeat.')
    for case in cases:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', case['id']):
            raise ValueError('Invalid case ID in mobile review.')
        frame = recorded_frames[(case['id'], case.get('panel'))]
        if (case['source_sha256'] != frame['source_sha256'] or case['reference_sha256'] != frame['reference_sha256'] or
                any(case[key]['pack_sha256'] != frame['arms'][arm]['pack_sha256']
                    for key, arm in [('baseline', baseline), ('candidate', candidate)])):
            raise ValueError(f'Visual evidence differs from the scorecard: {case["id"]}')
    perfect = []
    for row in (report.get('perfect_stage') or {}).get('rows', []):
        entry = {key: row[key] for key in ('stage', 'kind', 'change', 'tested', 'cohort', 'before', 'after')}
        if row['stage'] in ('Surface Finder', 'Surface and depth separation'):
            entry['surfaceExamples'] = [case['id'] for case in surface_examples(row)]
        perfect.append(entry)
    data = dict(schema=1, id=run, title=title, cohort=report['cohort_label'],
                views=visible['views'], available=visible['available'], comparisonSha256=report['comparison_sha256'],
                carryForward=carry_forward or report.get('carry_forward'),
                datasetNote=report.get('dataset_note') or ('Decorative Builds V1 unmeasured.'
                if not report['cohorts']['Decorative Builds V1']['views'] else 'Decorative Builds V1 has a separate benchmark result.'),
                methods=[dict(name=report['names'][arm], scores=visible['scores'][arm], failures=visible['failures'][arm])
                         for arm in (baseline, candidate)],
                stages=[dict(name=name, measure=description, values=[visible['stages'][arm].get(key)
                             for arm in (baseline, candidate)]) for key, (name, description) in MEASURES.items()],
                perfect=perfect, cases=[
                    make_case(case, queue, frames, assets, asset_base, labels, run_root,
                              evidence_root, baseline_evidence_root) for case in cases
                ])
    packages = {case['id']: read(assets / case['id'] / 'surfaces.json')
                for case in data['cases'] if case.get('surfaceEvidence')}
    for case in data['cases']:
        replays = []
        for row in (report.get('perfect_stage') or {}).get('rows', []):
            if row['stage'] not in ('Surface Finder', 'Surface and depth separation'):
                continue
            measured = next((frame for frame in row['cases'] if frame['id'] == case['id']), None)
            if measured:
                replays.append(dict(stage=row['stage'], before=measured['before']['geometry_strict'],
                                    after=measured['after']['geometry_strict']))
        if replays:
            case['surfaceReplays'] = replays
    ledger = private_history(data, database=database, downloads=downloads, questions=questions)
    data['analysis'] = analyze(report, saved, candidate, ledger=ledger, surface_evidence=packages)
    return data


def surface_examples(row):
    """Show both directions of the downstream effect where measured."""
    def delta(case):
        return case['after']['geometry_strict']['f1'] - case['before']['geometry_strict']['f1']
    ordered = sorted(row.get('cases', []), key=delta)
    if not ordered:
        return []
    return ([ordered[-1]] if delta(ordered[-1]) > 0 else []) + ([ordered[0]] if delta(ordered[0]) < 0 else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('queue', type=Path)
    parser.add_argument('frames', type=Path, help='Frozen data/<split>/frames directory tree')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--asset-base', required=True, help='URL prefix for the published asset directory')
    parser.add_argument('--run', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--candidate', default='current')
    parser.add_argument('--run-root', type=Path, help='Saved Floating V1 run root for mask and grid evidence')
    parser.add_argument('--evidence-root', type=Path, help='Portable runner packages in CASE/surfaces.json')
    parser.add_argument('--baseline-evidence-root', type=Path, help='Baseline local-grid packages to compare with --evidence-root')
    parser.add_argument('--labels', type=Path, help='Optional case ID to [title, stage, prompt, first pass] JSON')
    parser.add_argument('--questions', type=Path, help='Public-safe unresolved decision proposals (JSON array)')
    parser.add_argument('--carry-forward', help='Next baseline, each candidate disposition, and what landed')
    parser.add_argument('--site', type=Path, required=True, help='Directory for the staged review app and run manifest')
    args = parser.parse_args()
    if args.baseline_evidence_root and not args.evidence_root:
        parser.error('--baseline-evidence-root requires --evidence-root')
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.run):
        parser.error('Use a lowercase, hyphenated run ID.')
    data = manifest(read(args.report), args.queue, args.frames, args.assets, args.asset_base.rstrip('/'),
                    args.run, args.title, args.candidate, read(args.labels) if args.labels else DEFAULT_LABELS,
                    args.run_root, args.evidence_root, args.baseline_evidence_root,
                    questions=read(args.questions) if args.questions else None, carry_forward=args.carry_forward)
    target = args.site / 'runs' / args.run / 'manifest.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'{len(data["cases"])} cases, {target.stat().st_size:,} bytes of run data')


if __name__ == '__main__':
    main()
