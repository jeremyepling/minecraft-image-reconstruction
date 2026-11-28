# Floating wall-grid feedback audit

All five decisive wall-grid choices select the same final placements and block IDs as the existing large-contraction rule. The other response, “Neither,” rejects both Amphitheater grids. Across the 81 archived conditions from layouts outside this review, the rule gives up 23 net correct placements and removes 517 net extras, improving strict placement F1 by 0.10 percentage points. Keep this audit and the rule experimental. A new appearance of an oversized grid should be diagnosed at the wall chart and the final selected component.

## The six decisions

A is the expanded horizontal baseline in every row. The B variants reproduce the chart and whole-build scores from the [interactive review](../floating-extra-review-01/review.html). Positions below are correct / missed / extra, and the scores use one whole-build registration for geometry and material.

| Image | Response | Wall cells shown, A → B | Strict positions, A → B | Strict F1, A → B | Exact IDs, A → B |
|---|---|---:|---|---:|---:|
| Riverside Mill, high right | B | 2,875 → 150 | 152 / 324 / 2,723 → 97 / 379 / 53 | 9.07% → 30.99% | 5 → 6 |
| Riverside Mill, low left | A | 197 → 178 | 121 / 312 / 82 → 110 / 323 / 74 | 38.05% → 35.66% | 52 → 8 |
| Sawtooth, high right | B | 291 → 36 | 68 / 176 / 223 → 69 / 175 / 0 | 25.42% → 44.09% | 0 → 23 |
| Twinspire, low left | A | 40 → 40 | 112 / 166 / 0 → 88 / 190 / 19 | 57.44% → 45.71% | 48 → 43 |
| Hillside Chapel, low-left sky | B | 1,549 → 165 | 118 / 234 / 1,431 → 107 / 245 / 58 | 12.41% → 41.39% | 19 → 39 |
| Amphitheater, low left | Neither | 1,548 → 2 | 69 / 478 / 1,479 → 85 / 462 / 4 | 6.59% → 26.73% | 3 → 48 |

The five decisive choices agree with the higher strict geometry and strict material F1 on all five images. Always taking A scores 16.76% strict and 16.88% facade. Always taking B scores 38.32% and 39.05%. Following the five choices scores 40.77% and 41.54%, with 506 correct, 1,277 missed and 193 extra. Compared with A this costs 65 net correct positions, saves 4,266 extras and adds 44 exact position-and-ID matches. Strict/facade position-plus-ID F1 rises from 3.64%/3.66% to 13.54%/13.79%. The same five exports come from the large-contraction rule.

The Amphitheater review shows mask-0 returning to two sparse atlas cells. Its better whole export comes from separate, untouched walls mask-3 and mask-5. Sawtooth and Twinspire B also switch the selected group away from the displayed wall. A whole-export win on these images therefore cannot establish the accuracy of the displayed wall's unit grid. With “Neither” keeping the existing rule's output, the six recorded choices score 25.13% strict. An exhaustive reference selector over the 64 offered A/B combinations scores 37.91%, 39.05% facade, 13.86% strict position plus ID and 14.27% facade position plus ID. The difference comes entirely from allowing the already inspected Amphitheater bypass, which recovers 16 net correct and avoids 1,475 net extras. It is a reference estimate for the offers, with no new source-only prediction.

## Archived comparisons

The [decoded manifest](decoded.json.gz) identifies each condition by panel and frame, including the source image and reference digests, exact layout and dataset split, chart and pack hashes, strict/facade counts, and individual gained and lost placements. Every condition of the five reviewed layouts goes in the regression group. The 81 other conditions represent 79 unique native frames and 31 layouts. They were outside this feedback review and include cases already inspected in earlier experiments.

| Cohort | Method | Strict F1 | Facade F1 | Strict + ID F1 | Facade + ID F1 | Correct / missed / extra |
|---|---|---:|---:|---:|---:|---:|
| All 92 conditions, 36 layouts | Expanded horizontal | 31.12% | 31.61% | 4.85% | 4.94% | 21,328 / 72,268 / 22,157 |
| All 92 | Unrestricted full-face replacement | 31.74% | 32.27% | 5.00% | 5.10% | 20,851 / 72,745 / 16,942 |
| All 92 | Large-contraction rule | 32.13% | 32.67% | 5.10% | 5.20% | 21,240 / 72,356 / 17,374 |
| Reviewed layouts, 11 conditions | Expanded horizontal | 20.81% | 21.14% | 6.90% | 7.01% | 1,174 / 2,793 / 6,141 |
| Reviewed layouts, 11 | Large-contraction rule | 31.91% | 32.74% | 12.46% | 12.78% | 1,109 / 2,858 / 1,875 |
| Other archived layouts, 81 conditions | Expanded horizontal | 32.04% | 32.56% | 4.66% | 4.75% | 20,154 / 69,475 / 16,016 |
| Other archived layouts, 81 | Unrestricted full-face replacement | 31.82% | 32.33% | 4.65% | 4.74% | 19,837 / 69,792 / 15,221 |
| Other archived layouts, 81 | Large-contraction rule | 32.14% | 32.67% | 4.69% | 4.78% | 20,131 / 69,498 / 15,499 |

The large-contraction gate removes 4,783 net extras overall, including 4,266 on the reviewed layouts. The last six conditions remain at 18.34% strict under the gate because it never triggers in their fresh panel. Their inspected Amphitheater bypass is outside this source-only baseline. The combined archive covers 30 of 144 Visible Builds v2 views, 37 of 144 Visible Builds v3 views and 23 of 40 Decorative Builds v1 views. v3 and Decorative counts include one additional synthetic-sky condition each on a duplicate native frame. The six sky conditions have their own scorecard below.

## Prior reference controls

These are archived, paired end-to-end measurements using different frozen paths. Changes are strict / facade placement F1 in percentage points, and same-row comparisons share a cohort. Earlier baseline rows and exact-chart assembly used different upstream stages, so their gains are neither additive nor stage rankings.

| Supplied information | Cohort and matched comparison | Strict / facade before → after | Change | Scope |
|---|---|---:|---:|---|
| All frontmost native surface masks | Six sky, integration connected-atlas no-op → exact masks | 12.64% / 12.90% → 14.46% / 14.75% | +1.82 / +1.86 | Refits charts from the supplied crop list. Joint ID F1 falls from 4.68% / 4.77% to 3.86% / 3.93%. The full-cube-only variant reaches 12.09% / 12.33%, and a reference choice over both mask scopes and the no-op reaches 15.60% / 15.91%. [Mask receipts](../expanded-crop-ceiling-01/six-sky-chooser.json.gz) |
| All frontmost native surface masks | Original 38, source crops → exact masks before outline and count | 3.72% / 3.76% → 8.47% / 8.56% | +4.76 / +4.80 | Supplies grouping and coverage together, including decorative quads. Directions and depth are refit from predicted fields. Recovery is +1,505 correct and −33 extras. [Original mask control](../expanded-crop-ceiling-01/README.md) |
| Recorded projection and separate depth handoff | Six sky, older floor-off placement path with recorded projection and heuristic distance → same with true median camera-space depth of visible block centers | 36.76% / 36.43% → 62.08% / 62.39% | +25.32 / +25.96 | Uses the same frozen source mask/count handoff and omits the same post-placement passes. The old ordinary camera abstained on all six, and the old image-only native wall fallback was 2.30% / 2.25% via a different pipeline. A pose-only causal gain for current relative assembly is unavailable. [Projection and depth controls](../floating-build-grid-01/README.md) |
| Exact numeric width and height spans | Original 38, exact masks with ordinary counter → exact masks with reference numeric counts | 8.47% / 8.56% → 9.19% / 9.30% | +0.72 / +0.74 | Replaces numeric counts at the predicted quadrilateral center. Quads, orientation, phase, confidence and texture evidence remain as predicted. [Numeric count control](../expanded-wall-rulers-01/README.md) |
| Local pitch, then one local phase anchor | Original 38, native-mask resampling with predicted grid → reference pitch → plus one correct corner | 41.18% / 41.57% → 51.88% / 52.33% → 53.32% / 53.78% | +10.70 / +10.76, then +1.44 / +1.45 | Uses native pixels per block from two Jacobian column lengths near one anchor. Predicted vanishing points and directions can still drift. [Pitch and origin ladder](../expanded-wall-rulers-01/README.md) |
| One exact complete unit face per eligible wall | Six sky, expanded horizontal → exact corners in the same source mask | 22.93% / 23.07% → 38.12% / 38.70% | +15.19 / +15.63 | Gives scale, axes and phase together to eligible seeded charts. Rounding corners drops to 18.70% / 19.05%. Retaining source perspective with rounded corners reaches 29.93% / 30.36%. Two rounded faces with known integer offset reach 32.50% / 32.97%. Relative offsets across charts and source material matching still run normally. [One-face controls](../floating-one-face-01/reference-scores.md) |
| One exact complete unit face per eligible wall | Original 38, expanded horizontal → exact corners in the same source mask | 35.64% / 35.82% → 48.01% / 48.24% | +12.37 / +12.42 | The same handoff gains 7,502 net correct and adds 3,424 extras. Rounding corners drops to 33.41% / 33.64%. [One-face controls](../floating-one-face-01/reference-scores.md) |
| True surface ownership, exact face lattice and visible regrowth | Six sky, whole-wall expanded → owner splits only → refit sampled faces → add supported visible faces | 22.93% / 23.07% → 23.00% / 23.15% → 47.67% / 48.45% → 54.99% / 55.84% | +0.07 / +0.08, then +24.67 / +25.30, then +7.32 / +7.39 | Split reads each propagated cell's native center owner. Refit supplies existing visible unit face corners. Regrow additionally offers native full-cube wall faces with at least one visible pixel inside existing measured masks. All use source material and ordinary assembly without the optional horizontal stage. On the first new six views, refit alone drops strict F1 from 18.85% to 15.92%, then regrowth reaches 26.61%. [Ownership and coverage controls](../floating-patch-controls-01/reference-controls.md) |
| Exact independently rebased charts and reliable inter-chart joins | Six sky, exact charts with shared edges → add same-plane grid alignment | 77.58% / 77.36% → 93.71% / 93.70% | +16.13 / +16.34 | Both arms receive exact axes, full-cube visible wall and horizontal cells, native image projection/support and known IDs. Each chart has an independent origin and the assembler gets no common depth or absolute offset. Result is 3,563 / 478 / 0, with 76 non-full-cubes omitted from inputs but included in the denominator. Exact 0.01-pixel shared-edge agreement illustrates opportunity under exact charts. [Assembly receipts](../relative-face-assembly-01/README.md) |
| Perfect IDs on the existing positions | All 92, large-contraction positions held fixed | Geometry 32.13% / 32.67% in both arms | +0.00 / +0.00 | Position-plus-ID F1 would rise from 5.10% / 5.20% to 32.13% / 32.67%, rising from 3,370 to 21,240 strict exact matches. This arithmetic ceiling holds occupancy and registration fixed. A floating matched replay supplying perfect material upstream of geometry was unavailable. [Decoded manifest](decoded.json.gz) |

The 38-view source-only material test held its coordinates, chart confidence and ordering fixed while retaining old IDs and reranking newly sampled cells. Strict/facade position-plus-ID F1 rose from 0.99% / 0.99% to 4.51% / 4.53% and correct IDs rose from 389 to 1,773. The joint exact-chart assembly score already receives true IDs, so it measures placement under those supplied prerequisites. [Matched material comparison](../expanded-wall-rulers-01/README.md)

## Reproduction

From `experiment/floating-feedback-audit`, with the frozen sibling worktrees and exports used by the previous isolation, audit and distant seam experiments:

```powershell
& ../mask-scalelsd/runs/env/Scripts/python.exe -B -m benchmark.floating_feedback_audit (Join-Path $env:USERPROFILE Downloads/floating-wall-feedback.json) --workers 3
```

The run verifies the archived per-frame scores by decoding 374 pack inputs and scoring against the original frontmost visible geometry. It checks the dumped coordinates and IDs, expected score receipts, SHA-256 digests and global integer alignment. The preference JSON stays local. Only the preference choices and measured receipts are stored.

Production fixed 104 scores carry forward from `e3bd220` at 91.27% strict / 93.42% facade, with strict counts 20,773 / 2,929 / 1,046 and facade counts 18,952 / 1,624 / 1,046. The full production Visible Builds v3 (144) and Decorative Builds v1 (40) were not rerun for this replay. Archived source-only and reference controls above retain their original cohort definitions.

## Floating branch scorecard

Six identical fixed sky composites, connected-atlas integration runtime `52de2c9` at recorded revision `c0accfd`. This audit decoded and rescored the saved packs. It generated no new reconstruction outputs.

| Method | Strict F1 | Facade F1 | Strict + ID F1 | Facade + ID F1 | Correct / missed / extra |
|---|---:|---:|---:|---:|---:|
| Integration default, connected-atlas | 12.64% | 12.90% | 4.68% | 4.77% | 273 / 3,768 / 5 |
| Expanded horizontal | 22.93% | 23.07% | 4.40% | 4.40% | 901 / 3,140 / 2,915 |
| Unrestricted full-face replacement | 27.45% | 27.68% | 5.97% | 5.99% | 888 / 3,153 / 1,542 |
| Large-contraction rule | 27.50% | 27.73% | 5.96% | 5.99% | 890 / 3,151 / 1,542 |

The rule gains 14.86 strict points, 617 correct positions and 1,537 extras over connected-atlas. Relative to expanded horizontal on these sky views, strict F1 gains 4.56 points with 11 fewer correct and 1,373 fewer extras, all net changes from Chapel. The manifest, reference inventory and audit landed on `experiment/floating-feedback-audit`. The floating integration and production branches received no runtime changes from this audit.
