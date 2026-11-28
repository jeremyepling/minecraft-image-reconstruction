"""Report paired final-pack scores for inherited wall depth partitions."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

ARMS = {
    'baseline': ('reference', 'identity', 'Expanded Split-Border'),
    'separate': ('reference', 'reference', 'Native regions, siblings disconnected'),
    'same-plane': ('reference-zero', 'reference-zero', 'Native regions, signed offset zero'),
    'true-depth': ('reference', 'reference-offset', 'Native regions, measured signed offset'),
    'source-depth': ('source', 'source', 'Image depth regions, guarded split'),
    'source-appearance': ('source-appearance', 'source-appearance', 'Image appearance regions, guarded split'),
    'native-regions-image-depth': ('reference-image-offset', 'reference-image-offset', 'Native regions, image depth offsets'),
    'source-image-depth': ('source-image-offset', 'source-image-offset', 'Appearance regions, image depth offsets'),
    'native-regions-image-pitch': ('reference-pitch-offset', 'reference-pitch-offset', 'Native regions, image pitch offsets'),
    'source-image-pitch': ('source-pitch-offset', 'source-pitch-offset', 'Appearance regions, image pitch offsets'),
}


def read(path):
    return json.loads(path.read_bytes())


def aggregate(rows):
    result = {}
    for field in ('geometry', 'scores'):
        result[field] = {}
        for grading in ('strict', 'facade'):
            counts = {key: sum(row[field][grading][key] for row in rows) for key in ('correct','missed','extras')}
            denom = 2*counts['correct'] + counts['missed'] + counts['extras']
            result[field][grading] = dict(counts, f1=2*counts['correct']/denom)
    return result


def f1s(s, key='geometry'):
    return ' / '.join(f"{100*s[key][grade]['f1']:.2f}%" for grade in ('strict','facade'))


def report(root):
    summary = {}
    for key, (mode, arm, _) in ARMS.items():
        source = root / 'pitch-native-support' if 'pitch' in mode and (root / 'pitch-native-support').exists() else root
        rows = [read(p) for p in sorted(source.glob('front-*/'+mode+'.json'))]
        assert len(rows) == 100
        record = {}
        for split in ('all','train','validation','test','focus','control'):
            ids = ('006','060') if split == 'focus' else ('034','100') if split == 'control' else ()
            selected = [r for r in rows if split=='all' or r['split']==split or r['id'].split('-')[1] in ids]
            scores, active, changed, paired = [], [], [], Counter()
            for row in selected:
                measured = row['arms'].get(arm, dict(score=row['baseline_score'],paired={}))
                score = measured['score']
                scores.append(score)
                paired.update(measured['paired'])
                if row['partition_cells'] and key!='baseline':
                    active.append(row['id'])
                if score['geometry'] != row['baseline_score']['geometry']:
                    changed.append(row['id'])
            record[split] = dict(n=len(selected),**aggregate(scores),paired=dict(paired),active=active,changed=changed)
        summary[key] = record
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    markdown = ['# Front 100 wall depth partition and placement', '', '## Decision for the next round', '',
        'Keep Expanded Split-Border at measured revision `b0b6e3b` as the baseline. Retire the three active image-only alternatives and the image-pitch rule that never activated. Keep the signed-offset diagnostic for future work on placing parallel visible walls. The candidate code is isolated on `experiment/front100-wall-depth-partition` at `16a672b` with the source-support boundary check at `38a8178`. No prediction behavior landed on main. This round started from `c33f3b0`.', '',
        '## Clean full-block noon, 100 of 100 views', '',
        'Fixtures use release `f3e416acdbd6e0fa171260997fcdf62a86bc0c42c001d3a16d13e3f0e3c01187`. The 72 training, 14 validation and 14 test views retain the release split. No tuning used validation or test. All 100 identity reassemblies reproduced decoded local coordinates, block IDs, aligned coordinates and strict/facade scores. Source and chart hashes were checked against their existing receipts. Unchanged proposals reuse those verified packs. Decorative and other lighting conditions were unmeasured.', '',
        '| Method | Active / changed geometry score | Placement F1, strict / facade | Position + ID F1, strict / facade | Strict correct / missed / extra | Gained / lost correct | Added / removed extra |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key, (_,_,name) in ARMS.items():
        total=summary[key]['all']
        geo=total['geometry']['strict']
        p=total['paired']
        markdown.append(f"| {name} | {len(total['active'])} / {len(total['changed'])} | {f1s(total)} | {f1s(total,'scores')} | {geo['correct']:,} / {geo['missed']:,} / {geo['extras']:,} | {p['correct_gained']:,} / {p['correct_lost']:,} | {p['extras_added']:,} / {p['extras_removed']:,} |")
    markdown.extend(['', '## Split check', '',
        'Each cell gives strict / facade placement F1. The machine-readable summary includes exact position-and-ID scores, all counts and paired site changes for each split and the two diagnostic/control pairs. [Split scores](summary.json)', '',
        '| Method | Train, 72 views | Validation, 14 views | Test, 14 views |', '|---|---:|---:|---:|'])
    for key, (_,_,name) in ARMS.items():
        markdown.append('| '+name+' | '+' | '.join(f1s(summary[key][split]) for split in ('train','validation','test'))+' |')
    markdown.extend(['', '## What changed at assembly', '',
        'Of 100 views, 51 offered a previously unsplit source vertical chart with at least 256 cells. We tested the largest eligible chart in each. Each child inherits every parent homography coordinate, original cell ID, block ID and export eligibility. Only the chart membership and sibling-origin constraints change. Native diagnostics group a source cell by the frontmost reference face at its projected center. The dominant wall-facing direction must cover at least 60% of centers, and two or more same-facing planes must each own at least 24 centers and 5% of the chart. All other centers remain in a separate residual chart. The native tests activated on 35 views.', '',
        'For disconnected siblings, the ordinary largest-component selection can discard substantial walls. The zero-offset and native-offset arms connect the same substantial siblings, excluding the residual, with the same order and tangent coordinates. On all 100 views the signed offsets recover 3,837 more correct positions and remove 3,837 extras relative to those identical zero-offset links. The offset comes from the signed difference of same-facing block planes along the inferred chart normal. Among the 35 activated images, 33 selected split siblings; every selected sibling’s scored yaw mapped that normal to the actual observed world normal. The other two selected another component. There is no correction to tangent origins, pitch, material or residual-chart depth. This diagnostic is not a perfect geometry ceiling. The validation split lost 41 correct positions against zero-offset links; the test split gained 119. [Native boundary and normal receipts](boundary-audit.json)', '',
        'On Bell Tower 006 and Terrace 060, the incumbent exported 1,924 correct / 2,701 extra positions. A partition alone exported 1,581 / 447; zero-offset links gave 1,758 / 2,512; native signed offsets gave 3,290 / 980. Both near-planar controls, Bell Tower 034 and Sawtooth 100, retained their original packs under the guarded source and native partition tests.', '',
        'The depth-first source proposal produced a single substantial region on both original problem walls. Appearance regions exposed candidates, but neither view passed sustained signed relief between parallel flanks. Comparing measured image depth on both sides of native plane boundaries confirms the ambiguity. After subtracting local depth slopes, the nearer plane had the larger predicted inverse depth in only 20/46 Bell Tower probes and 32/67 Terrace probes. Across all 35 tested charts the sign agreed in 1,656/2,080 probes. These are neighboring cell comparisons, not independent buildings. [Bell Tower RGB, native regions and image depth](front-006-bell-tower-full-noon/native-and-source.png) · [Terrace RGB, native regions and image depth](front-060-terrace-full-noon/native-and-source.png)', '',
        'The two guarded source splits proposed regions from the existing model-depth watershed or from Lab appearance plus relative inverse depth, then required parallel flanks, an extended boundary and a sustained signed depth jump. The appearance version changed geometry on training views Bell Tower 020 and Observatory 092, and tied both held-out splits. The image-offset alternatives estimated a metric scale from the original grid homography, then tried either planar inverse-depth calibration or the relative periods of aligned RGB gradients in both UV axes. The former lost 16 correct blocks even with native partitions against zero-offset links. The latter produced no accepted shift, including with native masks. Both the initial whole-plane probe and the corrected source-mask-only rerun tied their respective controls on all 100 views. Future depth work should evaluate a distinct distance cue, such as a projected visible return with measurable width, against the zero-offset constraint and actual signed distances.', '',
        '## Reproduce', '',
        'Restore the stock Front 100 runtime/atlas as described in `benchmark/FLOATING_SHARED_DATASETS.md`. Set `$base` to the Expanded Split-Border replay root, `$fields` to the verified field archive directory containing per-case field folders and `$out` to a fresh private output directory. Use the prepared benchmark Python. The mobile exporter materializes the appearance-gated candidate with paired pack, charts, source-space surfaces, assembly, scores and aligned coordinates.', '',
        '```powershell', "$modes = 'reference','reference-zero','reference-offset-trace','source','source-appearance','reference-image-offset','source-image-offset','reference-pitch-offset','source-pitch-offset'",
        'foreach ($mode in $modes) {', '    & $benchmarkPython -m benchmark.floating_wall_depth_partition --base $base --fields $fields --output $out --mode $mode --workers 3', '}',
        '& $benchmarkPython -m benchmark.floating_wall_depth_audit --base $base --fields $fields --output $out',
        '& $benchmarkPython -m benchmark.floating_wall_depth_mobile --base $base --fields $fields --output $out --mode source-appearance',
        '& $benchmarkPython -m benchmark.floating_wall_depth_report --output $out', '```', '', '## Floating branch scorecard', '',
        'Expanded Split-Border `b0b6e3b`, carried on `c33f3b0`, scored **23.57% / 24.66%** strict / facade placement F1 and **4.67% / 4.89%** position-plus-ID F1 on all 100 clean noon Front views, with **25,960 correct, 115,612 missed, 52,734 extra**. The appearance-gated challenger scored **23.59% / 24.68%** and **4.67% / 4.90%**, with **25,954 correct, 115,618 missed, 52,538 extra**. It gained 25 correct, lost 31, added 171 extras and removed 367. No candidate landed on the integration branch.'])
    (root/'README.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    report(parser.parse_args().output)
