
"""Archive the portable paired scores and generate the experiment review."""
from pathlib import Path

from benchmark.expanded_baseline import STABLE, digest, read, save

HERE = Path(__file__).resolve().parent
RUN = STABLE / 'runs/floating-build-grid-01'


def pair(metrics):
    return ' / '.join(f"{metrics[mode]['f1']:.2%}" for mode in ('strict','facade'))


def main():
    fixtures = read(RUN/'fixtures.json')
    report = read(RUN/'paired-scores.json')
    checks = read(RUN/'independent-checks.json')
    summary = report['summary']['all']
    for method, groups in summary.items():
        expected = 6 if method == 'exact-distance-sky' else 18
        actual = sum(s['n'] for s in groups.values())
        if actual != expected:
            raise ValueError(f'{method}: expected {expected} scored packs, got {actual}')
    if len(checks['origins'])!=36 or len(checks['composition'])!=18:
        raise ValueError('Run the independent fixture and origin checks first')
    save(HERE/'results.json', dict(
        revision=fixtures['revision'], fixture_manifest_sha256=digest(RUN/'fixtures.json'),
        checks=checks, summary=report['summary'],
        fixtures=[dict(id=e['id'], split=e['split'], dataset=e['dataset'],
            source_sha256={v:x['source_sha256'] for v,x in e['variants'].items()},
            reference_sha256=e['reference_sha256'], clearance_ltrb_px=e['clearance_ltrb_px'])
            for e in fixtures['cases']],
        scores=[{name:r[name] for name in ('id','method','variant','pack_sha256','source_sha256','reference_sha256',
                                          'camera_reason','camera_usable','geometry','scores','exported','translation')}
                for r in report['rows']]))
    lines = ['# Floating builds and local grid', '',
        '**Keep both branches experimental.** The ordinary camera failed on every paired background replacement. The image-only grid recovered small visible sections.', '',
        '## Matched results', '',
        '| Method | Background | Views | No full projection | Local-only exports | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method,name in report['methods'].items():
        for variant,score in summary[method].items():
            strict = score['geometry']['strict']
            counts = ' / '.join(f'{strict[key]:,}' for key in ('correct','missed','extras'))
            lines.append(f"| {name} | {report['variants'][variant]} | {score['n']} | {score['camera_failures']} | "
                         f"{score['local_only']} | {pair(score['geometry'])} | {pair(score['typed'])} | {counts} |")
    for policy in ('floor','floating'):
        rows=[r for r in checks['origins'] if r['floor_policy']==policy]
        lines.extend(['', f"Moving the local Y origin from base to roof with floor policy **{policy}** preserved the same decoded geometry on **{sum(r['identical_positions'] for r in rows)}/{len(rows)}** paired inputs and the same geometry plus IDs on **{sum(r['identical_ids'] for r in rows)}/{len(rows)}**."])
    lines.extend(['',
        'Every arm scores the same 4,041 visible positions per background. The facade score exempts 86 visible sliver-only targets. Scores permit one whole-build integer XYZ translation. For local sheets the evaluator registers tangent axes to the recorded view, then applies that same translation rule. On Castle, ties between geometry-optimal translations can change the reported material score for identical rebased packs. The origin check compares decoded materials directly.', '',
        'The subset contains four Visible Builds v3 views and two Decorative Builds v1 views. The full historical cohorts carry forward from [the measured default at e3bd220](../camera-native-round-01/README.md). Fixed 104 placement is 91.27% strict / 93.42% facade with 20,773 correct, 2,929 missed and 1,046 extra. Full v3 is 46.63% / 48.04% and full decorative is 63.46% / 65.23%.', '',
        '## What each comparison isolates', '',
        '- All labeled structure pixels, relative viewpoint and visible targets are identical within each triplet. Plain and sky replace all remaining pixels. The horizonless sky is synthetic and preserves the original camera visibility and occlusions. Three views look upward.', '',
        '- The image-only candidate rebuilds Regions First masks and Block Counter using the source and its predicted normal and depth fields. Native texture fits establish a local scale and phase. The exploratory fallback also grows connected wall sheets with separate texture verification per adjoining cell after a full grid fails.', '',
        '- Exact-projection runs share each background’s frozen Surface Finder and Counter handoff. Both policies rerun ranked placement and materials and omit the same build-level post-placement passes. Floor-off disables assumed contact and keeps signed Y. The default distance seed is about 23 blocks, derived from lens and frame height. The distance diagnostic receives the actual median depth of the visible labeled block centers as additional reference information.', '',
        '![Six matched structures](fixture-preview.jpg)', '',
        '## Reproduce', '',
        'Use the pinned Java 26.3 model assets and atlas. From the stable checkout set PYTHONPATH to that checkout, then run prepare_fixtures.py and run_baseline.py. From their candidate checkouts run python -m benchmark.floating_build_local_run with the fixture manifest as its positional argument, both with and without --native-fallback, and python -m benchmark.floating_exact --wait. Back in the stable environment run report_round.py, validate_round.py and archive_results.py. The local report and original run artifacts live under runs/floating-build-grid-01 in the stable checkout. [Fixture, pose and score receipts](results.json)', '',
        'The local grid and paired review are on experiment/floating-build-local-grid. Placement bypasses are on experiment/floating-build-floor-bypass. Main stays on its verified default.', ''])
    (HERE/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
