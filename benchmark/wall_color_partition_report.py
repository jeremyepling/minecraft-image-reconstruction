"""Summarize actual decoded trials, retaining their limited cohort coverage."""
from pathlib import Path
from benchmark.expanded_baseline import read, save, digest
from benchmark.wall_color_partition import ROOT, REVIEW, CASES


def main():
    cases = [read(p) for p in sorted(REVIEW.glob('*.json')) if p.name != 'summary.json']
    cases = [c for c in cases if 'handoff' in c]
    summary = dict(baseline_revision='62611ee', experiment_revision='3265796',
                   script_sha256=digest(ROOT/'benchmark/wall_color_partition.py'),
                   reference_baseline_sha256=digest(CASES),
                   policy=dict(stride=4, color_difference=20, minimum_samples=16,
                               radius=6, guarded_rank_margin=.5,
                               carrier='largest source-mask area with selected vertical axis'),
                   decision='keep experimental', views_by_mode={}, cases=[])
    lines = ['# Source-color wall partitions and local depth selection', '',
        'Keep `experiment/wall-color-partition` experimental. The source-only repeat ranker improves several training buildings, but it regresses Shrine and swaps many correct positions on Harbor. A training-selected confidence margin preserves Shrine while reducing the Harbor gain. Four fixed-suite regression views remain unchanged. This is insufficient evidence for a global default.', '',
        'The ordinary default remains `main` at `62611ee`. No production engine code changed. The diagnostic and source-only trial are separate: [verified reference partition opportunity](../harbor-placement-evidence-01/README.md).', '',
        '## What ran', '',
        'Choose the largest vertical carrier by its source-mask area. Connect neighboring four-pixel samples when their 4×4 median RGB differs by at most 20 per channel. Keep components of at least 16 samples; smaller components retain the parent. For each component, offer the ordinary plane and integer depths within six blocks. The ranked arm uses the existing frozen texture-repeat depth model on each component. The boundary arm accepts a boundary score of at least 0.60 and a gain of at least 0.15. The guarded arm additionally requires a 0.5 ranking-score margin over the parent; this is a score margin, not a calibrated probability.', '',
        'The first training comparison used Amphitheater ashlar high-right and Viaduct riverstone high-right. Additional training families exposed Shrine’s low-margin failure. The 0.5 guard was selected from training results, not Harbor labels. Harbor has already been inspected and is a regression family, not a fresh holdout. Both source variants preserve the parent when no eligible change is selected. Components assigned the same depth are joined before ordinary visible-cell generation. All later visibility, foreground, material, and export steps rerun. Reference labels enter only after export for scoring.', '',
        '## Decoded exports', '',
        'Each row compares identical source images against the verified default. F1 is in percent. Position-plus-ID uses the geometry-selected whole-build integer translation. Gains/losses count actual required or optional visible positions after the permitted global shift, so a tied score can still hide changed cells.', '',
        '| Source image | Arm | Strict placement, before → after | Facade placement, before → after | Strict / facade position + ID, after | Correct gained / lost | Extras added / removed |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in cases:
        summary['views_by_mode'].setdefault(c['mode'], {}).setdefault(c['dataset'], 0)
        summary['views_by_mode'][c['mode']][c['dataset']] += 1
        summary['cases'].append(dict(id=c['id'], dataset=c['dataset'], mode=c['mode'],
             source_sha256=c['source_sha256'], pack_sha256=c['pack_sha256'],
             geometry=c['geometry'], scores=c['scores'], baseline_geometry=c['baseline']['geometry'],
             baseline_scores=c['baseline']['scores'], paired=c['paired']))
        if c['mode'] == 'control':
            continue
        before, after = c['baseline']['geometry'], c['geometry']
        pair = c['paired']
        def f(n): return f'{100*n:.2f}'
        lines.append(f"| {c['id']} | {c['mode']} | {f(before['strict']['f1'])} → {f(after['strict']['f1'])} | "
            f"{f(before['facade']['f1'])} → {f(after['facade']['f1'])} | "
            f"{f(c['scores']['strict']['f1'])} / {f(c['scores']['facade']['f1'])} | "
            f"{pair['gained']} / {pair['lost']} | {pair['extras_added']} / {pair['extras_removed']} |")
    lines += ['', '## What to test next', '',
        'Use the newly calibrated cameras to distinguish two situations: one mask combines genuinely different wall depths, or one flat wall merely changes material. Trace every lost correct cell to its first changed component and depth. Then test independent visible corners, junctions, or depth evidence to authorize individual changes. A global confidence threshold does not reliably distinguish those situations. Do not remove the grounded-depth filter: its isolated bypass introduced too many extras in the diagnostic.', '',
        '## Placement status', '',
        'These are carried-forward ordinary scores at `62611ee`, not a full-cohort score for the challenger. The trial has four fixed views, nine v3 views, and two decorative views; arms have different coverage recorded in `summary.json`. No new whole-cohort default was measured.', '',
        '| Cohort | Verified baseline views | Strict / facade placement | Strict / facade position + ID | Correct / missed / extras |',
        '|---|---:|---:|---:|---:|',
        '| Fixed trend | 104/104 | 91.27 / 93.42 | 84.68 / 88.51 | 20,773 / 2,929 / 1,046 |',
        '| Visible Builds v3 | 144/144 | 21.84 / 22.75 | 14.13 / 14.79 | 28,086 / 186,064 / 14,949 |',
        '| Decorative Builds v1 | 40/40 | 63.46 / 65.23 | 50.51 / 52.56 | 9,240 / 7,686 / 2,957 |', '',
        'Camera failures are unchanged: 0 fixed, 70 v3, and 0 decorative. Six control exports match the verified baseline coordinates and IDs exactly. Production inputs are frozen source-only model fields and source-derived cameras; the camera stage itself is unchanged in this experiment.', '',
        '```powershell',
        'python -m benchmark.wall_color_partition train-amphitheater-02-ashlar-high-right train-viaduct-02-riverstone-high-right --modes control ranked boundary --workers 3',
        'python -m benchmark.wall_color_partition train-shrine-01-ochre-low-left validation-harbor-02-kiln-high-right validation-harbor-02-seafoam-high-right --modes guarded --workers 3',
        'python -m benchmark.wall_color_partition_report',
        '```', '']
    save(REVIEW/'summary.json', summary)
    (REVIEW/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary['views_by_mode'])


if __name__ == '__main__':
    main()
