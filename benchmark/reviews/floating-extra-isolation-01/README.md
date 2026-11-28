# Isolating the wall charts responsible for extra placements

Keep the large-contraction rule experimental. On the 86 reviewed conditions it removes 4,783 net extras from the expanded horizontal baseline, at a cost of 88 net correct positions. Unrestricted replacement removes 5,212 extras and costs 486 correct positions. The gate retains the largest reductions on Chapel and Riverside Mill without damaging the original Castle, Watchpost and Shrine exports. It changes five exports. On the six new views it never triggers, including a wall with 1,479 extras.

## Decoded scorecard

| Panel | Method | Correct | Extra | Missed | Strict precision | Strict recall | Strict F1 | Facade F1 | Strict + ID F1 | Facade + ID F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous 86 | Floating integration default | 2,374 | 127 | 89,393 | 94.92% | 2.59% | 5.04% | 5.19% | 2.14% | 2.22% |
| Previous 86 | Expanded horizontal baseline | 20,993 | 20,667 | 70,774 | 50.39% | 22.88% | 31.47% | 31.96% | 4.67% | 4.76% |
| Previous 86 | Replace every accepted chart | 20,507 | 15,455 | 71,260 | 57.02% | 22.35% | 32.11% | 32.64% | 4.82% | 4.91% |
| Previous 86 | Only large contractions | 20,905 | 15,884 | 70,862 | 56.82% | 22.78% | 32.52% | 33.06% | 4.92% | 5.02% |
| New six | Floating integration default | 181 | 0 | 1,648 | 100.00% | 9.90% | 18.01% | 19.17% | 17.81% | 18.96% |
| New six | Expanded horizontal baseline | 335 | 1,490 | 1,494 | 18.36% | 18.32% | 18.34% | 18.92% | 11.22% | 11.55% |
| New six | Replace every accepted chart | 344 | 1,487 | 1,485 | 18.79% | 18.81% | 18.80% | 19.39% | 11.15% | 11.54% |
| New six | Only large contractions | 335 | 1,490 | 1,494 | 18.36% | 18.32% | 18.34% | 18.92% | 11.22% | 11.55% |

Relative to the actual floating integration default, the gated expansion buys 18,531 net visible positions while adding 15,757 extra positions on the reviewed 86. Strict position-plus-ID matches rise from 1,011 to 3,165. On the fresh six it gains 154 correct positions while adding 1,490 extras and gains 26 exact IDs. That tradeoff and the lack of a fresh trigger argue against promotion to the floating integration branch or production main. A conservative expansion option on this experiment branch is justified for comparing candidate exports. [All panel metrics, counts and denominators](scores.md)

The rule examines only charts with an accepted source-image replacement. It uses that replacement and its atlas matches if the old expanded wall has at least 256 cells and the newly sampled wall has at most one quarter as many. Every other wall retains its original grid. These thresholds were selected after inspecting archived development and regression results and before reading the new panel. They express a large reduction on an unusually large existing chart. No reference dimensions, received camera, mask ownership or answer material enters that choice.

## Which wall caused each change

The intervention replaces exactly one wall chart and holds every other chart at the expanded baseline, or restores one original chart while applying all the other accepted replacements. The regular horizontal-face generator, connected component selection, material precedence and pack exporter rerun on every distinct set of substitutions. Each export is decoded and checked against its placements file. Translation is re-estimated once for the whole build, keeping geometry and materials together. [Each individual chart and export](chart-traces.md)

| View | Intervention and first affected chart | Export effect |
|---|---|---|
| Riverside Mill 1 high-right | Wall mask-0 contracts from 2,875 to 150 cells. Of four accepted charts, this one alone reproduces the entire change. Restoring it while replacing the other three restores the old export. | The same wall wins both times. Correct positions fall from 152 to 97. Extra positions fall from 2,723 to 53. Exact IDs rise from 5 to 6. Registration translation changes by (−28, −22, 0). |
| Riverside Mill 1 low-left | The only replacement, mask-0, changes 197 cells to 178. The gate retains the old chart. | Both exports select mask-0 and mask-1. Translation changes by (+12, −2, 0) with replacement. Correct positions fall from 121 to 110, extras from 82 to 74 and exact IDs from 52 to 8. Among 99 jointly matched positions, 40 old exact IDs become wrong and one previously wrong ID becomes correct. |
| Chapel sky | Mask-0 contracts from 1,549 to 165 cells. The original wall used texture inside calcite as its unit. | The same wall wins, with 107 of 118 originally correct positions retained and one newly added extra. Extras fall from 1,431 to 58. Among the 107 shared correct positions, 32 gain the correct ID and ten lose it. Overall exact IDs rise from 19 to 39. |
| Castle limestone low-left | Mask-0-wall-1 contracts from 381 to 169. It then loses the largest component contest to untouched mask-1. The gate retains the original. | Unrestricted replacement changes 272 correct / 109 extra / 25 exact IDs to 171 / 12 / 36. |
| Watchpost 2 low-left | Changing either member of the original two-wall component breaks the join. Mask-0 has 80 cells before and after replacement. The gate retains both. | Even replacing only mask-0 loses 40 correct positions and removes 25 extras. |
| Shrine 02 low-left | Mask-0-wall-0 contracts from 124 to six cells. The gate retains it. | Unrestricted replacement loses 54 correct positions and ten correct IDs while removing 64 extras. |
| Market Court 1 high-right | Mask-1 contracts from 593 to 18 cells. An existing three-wall group with a measured horizontal surface then wins. | 77 correct / 516 extra becomes 51 / 59. Seven newly gained and 33 lost correct positions yield the net loss of 26. |
| Harbor 02 low-left | Mask-1 contracts from 264 to 44 cells, uncovering untouched mask-0. | 66 correct / 198 extra becomes 69 / 138. |
| Sawtooth 2 high-right | Replacing only mask-0-wall-1 (291 to 36) switches selection to mask-1. The scorer's camera-based compass registration changes by a quarter-turn. | 68 correct / 223 extra becomes 69 / 0. All replacements yield 70 / 0. |
| Twinspire 2 high-right | Replacing only mask-2 (175 to 33 cells) switches selection to the joined original mask-0 and mask-1. The gate retains the baseline. | Baseline 97 correct / 190 extra / 2 exact IDs. Only mask-2 yields 110 / 63 / 42. All replacements yield 80 / 10 / 26. Other meaningful chart choices exist below this gate's size threshold. |

Two Pillars sky charts pass the gate without changing the selected export. Together with Harbor, Chapel, Market Court, Sawtooth and Riverside Mill high, these account for all seven qualifying charts across six historical views. The [source-image comparisons](chart-substitutions.jpg) show the grids before and after accepting all replacements and the strict truth positions that are jointly retained (green), lost (coral) or gained (blue). Extras have no matching truth face and are counted in the captions.

A reference chooser over the actually replayed whole-view exports (baseline, all replacements, the gate, plus only-one/drop-one on the ten diagnostic images) reaches 35.92%, 30.71%, 40.20%, 27.50%, 18.83%, 23.51%, 26.04% and 33.08% strict F1 respectively on the eight historical panels in scorecard order. Its new-six score is 18.80%. This measures only the available arms and excludes the post-outcome Amphitheater diagnosis.

## Fresh-panel miss and direct bypass

The six further views comprise two angles each of Arcade 1 and Arcbridge 1 from Visible Builds v2 and Amphitheater 01 from Visible Builds v3. No chart passed the frozen gate. On Amphitheater low-left, mask-0 had 42 face offers, none accepted. Its ordinary two seeds had been extended to 1,548 wall cells. The two accepted face replacements were on other components. Unrestricted face replacement still exports the same 69 correct positions, 1,479 extras and three exact IDs.

After inspecting that failure we replayed two interventions only on this view. Returning mask-0 to its original sparse two-cell atlas chart or deleting only mask-0 lets the same two untouched walls, mask-3 and mask-5, win component selection. Both export the same 89 positions with 85 correct, four extras, 462 missed and 48 exact IDs. This gains 16 net correct positions, removes 1,475 net extras and raises strict F1 from 6.59% to 26.73%. The actual floating default on this image places 36 blocks, all at correct positions with correct IDs. The selective bypass was chosen with knowledge of this failure and is excluded from the fresh source-only score. [New panel and diagnostic receipts](holdout5.json.gz)

## Reproduction and coverage

Run from `experiment/floating-extra-isolation` using the benchmark Python and the existing source packets from the face-proposal and audit worktrees.

~~~powershell
python -m benchmark.floating_extra_isolation
python -m benchmark.floating_extra_isolation holdout5
python -m benchmark.floating_extra_isolation_sheet
~~~

The gzip JSON files contain scores, per-chart old/new cell extents and material histograms, substitution sets, charts selected for export, component sizes, registration, paired position changes and source, reference, proposal and pack hashes. The untouched horizontal and floating-default packs were decoded from the published experiments. The gated packs and named interventions ran from the same frozen RGB and chart roots. Source images and complete native scoring maps remain outside the export directory. Evaluator-only visible labels, including camera for compass registration, support the score and its explanation.

Including the new panel there are 30 Visible Builds v2 conditions, 38 Visible Builds v3 conditions (37 unique of 144 released views) and 24 Decorative Builds v1 conditions (23 unique of 40). The six sky composites are reported separately. This experiment is based on `d8f4321`. Production's fixed 104 is carried forward at 91.27% strict (20,773 correct, 1,046 extra, 2,929 missed) and 93.42% facade (18,952 correct, 1,046 extra, 1,624 missed), with zero default delta. The results landed on the experiment branch only.
