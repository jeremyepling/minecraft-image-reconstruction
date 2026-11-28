# Front 100 wall depth partition and placement

## Decision for the next round

Keep Expanded Split-Border at measured revision `b0b6e3b` as the baseline. Retire the three active image-only alternatives and the image-pitch rule that never activated. Keep the signed-offset diagnostic for future work on placing parallel visible walls. The candidate code is isolated on `experiment/front100-wall-depth-partition` at `16a672b` with the source-support boundary check at `38a8178`. No prediction behavior landed on main. This round started from `c33f3b0`.

## Clean full-block noon, 100 of 100 views

Fixtures use release `f3e416acdbd6e0fa171260997fcdf62a86bc0c42c001d3a16d13e3f0e3c01187`. The 72 training, 14 validation and 14 test views retain the release split. No tuning used validation or test. All 100 identity reassemblies reproduced decoded local coordinates, block IDs, aligned coordinates and strict/facade scores. Source and chart hashes were checked against their existing receipts. Unchanged proposals reuse those verified packs. Decorative and other lighting conditions were unmeasured.

| Method | Active / changed geometry score | Placement F1, strict / facade | Position + ID F1, strict / facade | Strict correct / missed / extra | Gained / lost correct | Added / removed extra |
|---|---:|---:|---:|---:|---:|---:|
| Expanded Split-Border | 0 / 0 | 23.57% / 24.66% | 4.67% / 4.89% | 25,960 / 115,612 / 52,734 | 0 / 0 | 0 / 0 |
| Native regions, siblings disconnected | 35 / 34 | 22.82% / 23.96% | 4.79% / 5.03% | 23,510 / 118,062 / 40,971 | 2,512 / 4,962 | 6,914 / 18,677 |
| Native regions, signed offset zero | 35 / 33 | 23.09% / 24.18% | 4.65% / 4.88% | 25,078 / 116,494 / 50,561 | 863 / 1,745 | 3,288 / 5,461 |
| Native regions, measured signed offset | 35 / 33 | 26.62% / 27.88% | 6.00% / 6.29% | 28,915 / 112,657 / 46,724 | 8,190 / 5,235 | 13,325 / 19,335 |
| Image depth regions, guarded split | 2 / 2 | 23.40% / 24.48% | 4.64% / 4.86% | 25,701 / 115,871 / 52,440 | 0 / 259 | 0 / 294 |
| Image appearance regions, guarded split | 4 / 2 | 23.59% / 24.68% | 4.67% / 4.90% | 25,954 / 115,618 / 52,538 | 25 / 31 | 171 / 367 |
| Native regions, image depth offsets | 35 / 33 | 23.08% / 24.16% | 4.65% / 4.88% | 25,062 / 116,510 / 50,577 | 883 / 1,781 | 4,883 / 7,040 |
| Appearance regions, image depth offsets | 5 / 3 | 23.37% / 24.46% | 4.64% / 4.86% | 25,743 / 115,829 / 52,951 | 103 / 320 | 1,864 / 1,647 |
| Native regions, image pitch offsets | 35 / 33 | 23.09% / 24.18% | 4.65% / 4.88% | 25,078 / 116,494 / 50,561 | 863 / 1,745 | 3,288 / 5,461 |
| Appearance regions, image pitch offsets | 0 / 0 | 23.57% / 24.66% | 4.67% / 4.89% | 25,960 / 115,612 / 52,734 | 0 / 0 | 0 / 0 |

## Split check

Each cell gives strict / facade placement F1. The machine-readable summary includes exact position-and-ID scores, all counts and paired site changes for each split and the two diagnostic/control pairs. [Split scores](summary.json)

| Method | Train, 72 views | Validation, 14 views | Test, 14 views |
|---|---:|---:|---:|
| Expanded Split-Border | 21.74% / 22.63% | 44.10% / 46.27% | 17.80% / 19.54% |
| Native regions, siblings disconnected | 21.14% / 22.07% | 42.25% / 44.49% | 16.77% / 18.59% |
| Native regions, signed offset zero | 21.37% / 22.25% | 43.25% / 45.46% | 17.16% / 18.88% |
| Native regions, measured signed offset | 25.57% / 26.63% | 42.85% / 45.04% | 18.50% / 20.37% |
| Image depth regions, guarded split | 21.74% / 22.63% | 42.39% / 44.51% | 17.99% / 19.77% |
| Image appearance regions, guarded split | 21.76% / 22.65% | 44.10% / 46.27% | 17.80% / 19.54% |
| Native regions, image depth offsets | 21.35% / 22.24% | 43.25% / 45.46% | 17.16% / 18.88% |
| Appearance regions, image depth offsets | 21.73% / 22.62% | 42.18% / 44.25% | 17.80% / 19.54% |
| Native regions, image pitch offsets | 21.37% / 22.25% | 43.25% / 45.46% | 17.16% / 18.88% |
| Appearance regions, image pitch offsets | 21.74% / 22.63% | 44.10% / 46.27% | 17.80% / 19.54% |

## What changed at assembly

Of 100 views, 51 offered a previously unsplit source vertical chart with at least 256 cells. We tested the largest eligible chart in each. Each child inherits every parent homography coordinate, original cell ID, block ID and export eligibility. Only the chart membership and sibling-origin constraints change. Native diagnostics group a source cell by the frontmost reference face at its projected center. The dominant wall-facing direction must cover at least 60% of centers, and two or more same-facing planes must each own at least 24 centers and 5% of the chart. All other centers remain in a separate residual chart. The native tests activated on 35 views.

For disconnected siblings, the ordinary largest-component selection can discard substantial walls. The zero-offset and native-offset arms connect the same substantial siblings, excluding the residual, with the same order and tangent coordinates. On all 100 views the signed offsets recover 3,837 more correct positions and remove 3,837 extras relative to those identical zero-offset links. The offset comes from the signed difference of same-facing block planes along the inferred chart normal. Among the 35 activated images, 33 selected split siblings; every selected sibling’s scored yaw mapped that normal to the actual observed world normal. The other two selected another component. There is no correction to tangent origins, pitch, material or residual-chart depth. This diagnostic is not a perfect geometry ceiling. The validation split lost 41 correct positions against zero-offset links; the test split gained 119. [Native boundary and normal receipts](boundary-audit.json)

On Bell Tower 006 and Terrace 060, the incumbent exported 1,924 correct / 2,701 extra positions. A partition alone exported 1,581 / 447; zero-offset links gave 1,758 / 2,512; native signed offsets gave 3,290 / 980. Both near-planar controls, Bell Tower 034 and Sawtooth 100, retained their original packs under the guarded source and native partition tests.

The depth-first source proposal produced a single substantial region on both original problem walls. Appearance regions exposed candidates, but neither view passed sustained signed relief between parallel flanks. Comparing measured image depth on both sides of native plane boundaries confirms the ambiguity. After subtracting local depth slopes, the nearer plane had the larger predicted inverse depth in only 20/46 Bell Tower probes and 32/67 Terrace probes. Across all 35 tested charts the sign agreed in 1,656/2,080 probes. These are neighboring cell comparisons, not independent buildings. [Bell Tower RGB, native regions and image depth](front-006-bell-tower-full-noon/native-and-source.png) · [Terrace RGB, native regions and image depth](front-060-terrace-full-noon/native-and-source.png)

The two guarded source splits proposed regions from the existing model-depth watershed or from Lab appearance plus relative inverse depth, then required parallel flanks, an extended boundary and a sustained signed depth jump. The appearance version changed geometry on training views Bell Tower 020 and Observatory 092, and tied both held-out splits. The image-offset alternatives estimated a metric scale from the original grid homography, then tried either planar inverse-depth calibration or the relative periods of aligned RGB gradients in both UV axes. The former lost 16 correct blocks even with native partitions against zero-offset links. The latter produced no accepted shift, including with native masks. Both the initial whole-plane probe and the corrected source-mask-only rerun tied their respective controls on all 100 views. Future depth work should evaluate a distinct distance cue, such as a projected visible return with measurable width, against the zero-offset constraint and actual signed distances.

## Reproduce

Restore the stock Front 100 runtime/atlas as described in `benchmark/FLOATING_SHARED_DATASETS.md`. Set `$base` to the Expanded Split-Border replay root, `$fields` to the verified field archive directory containing per-case field folders and `$out` to a fresh private output directory. Use the prepared benchmark Python. The mobile exporter materializes the appearance-gated candidate with paired pack, charts, source-space surfaces, assembly, scores and aligned coordinates.

```powershell
$modes = 'reference','reference-zero','reference-offset-trace','source','source-appearance','reference-image-offset','source-image-offset','reference-pitch-offset','source-pitch-offset'
foreach ($mode in $modes) {
    & $benchmarkPython -m benchmark.floating_wall_depth_partition --base $base --fields $fields --output $out --mode $mode --workers 3
}
& $benchmarkPython -m benchmark.floating_wall_depth_audit --base $base --fields $fields --output $out
& $benchmarkPython -m benchmark.floating_wall_depth_mobile --base $base --fields $fields --output $out --mode source-appearance
& $benchmarkPython -m benchmark.floating_wall_depth_report --output $out
```

## Floating branch scorecard

Expanded Split-Border `b0b6e3b`, carried on `c33f3b0`, scored **23.57% / 24.66%** strict / facade placement F1 and **4.67% / 4.89%** position-plus-ID F1 on all 100 clean noon Front views, with **25,960 correct, 115,612 missed, 52,734 extra**. The appearance-gated challenger scored **23.59% / 24.68%** and **4.67% / 4.90%**, with **25,954 correct, 115,618 missed, 52,538 extra**. It gained 25 correct, lost 31, added 171 extras and removed 367. No candidate landed on the integration branch.
