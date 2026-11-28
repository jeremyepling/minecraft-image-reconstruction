# Floating wall feedback and stage headroom

[Open the eight-case experiment viewer](experiments.html). The first five entries compare source-only rules. The last three use reference data to locate the failure. Each entry has its own vote and comment. Download the JSON to carry the annotations into the next round. [Previous six-case review](../floating-extra-review-01/review.html) · [Submitted feedback](feedback.json)

## What the annotations tell us

Jeremy's five definitive A/B choices all agree with the better whole-structure strict placement and position-plus-material score of the two options on those examples. The existing large-contraction rule already produced those five chosen exports. His sixth answer, neither, identifies a new missing alternative. Two sparse cells on Amphitheater imply the same wrong spacing as the 1,548-cell wall. Withdrawing the expansion helped because the assembler switched to another part of the structure.

| Reviewed wall | Median grid cells covering one reference full face, A / B | Source of error or evidence |
|---|---:|---|
| Riverside high right | 15.04 / 0.96 | The original atlas matches repeated inside one actual block. |
| Riverside low left | 0.97 / 0.82 | A is closer in both size and corner alignment. The mask contains 86,352 red sandstone pixels and 13,736 background pixels, 5.26% of its total. |
| Sawtooth | 9.74 / 1.01 | B matches full faces over a clean mask. One native surface accounts for 98.4% of the mask. |
| Hillside Chapel | 7.80 / 1.15 | B fixes most of the scale error. Its left-quarter corners still have 0.40 cells median error to the nearest grid line per coordinate. |
| Twinspire low left | 1.02 / 1.07 | Both scales are plausible. Changes to which walls join can dominate end placement. |
| Amphitheater | 13.06 / 13.06 | Both alternatives use the same ruler. Only 21.6% of the mask belongs to the largest single recorded surface. |

These areas use projected corners of complete unit-square faces on the most represented reference surface, with four visible corners and at least half their visible pixels inside the mask. An area of one is ideal and can still hide a shape or origin error. Background means any pixel outside the target structure. [Mask composition and ruler sizes](mask-and-scale.jpg) · [Measurements](measurements.json)

## Experiments and decisions

The frozen regression group consists of 11 conditions from the five annotated layouts. Transfer comparison uses 81 archived conditions from other layouts. Their images were used in earlier research rounds, so these are not a prospective blind test. No build labels entered the source-only rules. Every changed placement comes from the ordinary connected-group assembler and decoded pack.

- **Larger textured faces.** An accepted face had to cover at least two cells of the old grid, agree with a neighbor and beat the old grid's atlas correlation. Adding this rule to large-contraction scored 32.13% to 32.13% across 92 (32.1307% to 32.1251% before rounding). The only changed transfer export, Shrine, lost 54 correct blocks and removed 64 extras. The viewer pairs Shrine with a Twinspire view that gained 13 correct and removed 127 extras. Keep experimental. [Atlas tests](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-feedback-scale/benchmark/reviews/floating-feedback-scale-01)
- **Weak evidence on smooth walls.** Reverting large expansions with at most two seed cells and no verified full face improved the reviewed Amphitheater, but transfer strict F1 fell from 32.14% to 28.61%. Requiring a majority of locally smooth pixels reduced the drop to 31.59%. The smoother Amphitheater 02 view lost 337 correct blocks. Keep experimental. [Support tests](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-feedback-support/benchmark/reviews/floating-feedback-support-01)
- **Corner versus extent, reference-assisted.** We refit only the fractional grid origin, keeping perspective and scale fixed. We separately clipped the reviewed wall to its intersection with the most represented visible reference surface. With the same resampler on both sides, the origin change raised Riverside high-right from 97 correct / 53 extra to 129 / 67, with exact material matches rising from six to 33. On Chapel it moved 107 / 58 to 109 / 67 and slightly reduced F1. On Riverside low-left the boundary clip changed 121 / 82 to 91 / 0. Those removed regions include other genuine visible surfaces. A split should give each region a new opportunity to place its blocks. Clipping alone tests only the removal half. The six same-grid resampling controls reproduced baseline correct and extra counts. [Paired controls](boundary-and-origin.json.gz)

## How much can perfect inputs change placement?

Each row compares the same downstream path on the same cohort. Rows from different pipelines are separate experiments. Values are strict visible placement F1. Gains cannot be added. Some rows replace multiple fields and are labeled accordingly.

| Handoff changed | Cohort and control | Before | With reference | Gain in points |
|---|---|---:|---:|---:|
| Correct native visible surface masks | Six sky, sparse atlas path | 12.64% | 14.46% | +1.82 |
| Correct count dimensions after supplying exact masks | Original 38, sparse path | 8.47% | 9.19% | +0.72 |
| Correct local block spacing on visible masks | Original 38, matched resampler | 41.18% | 51.88% | +10.70 |
| Correct local grid corner after correct spacing | Same 38 and resampler | 51.88% | 53.32% | +1.44 |
| One exact projected full face per eligible wall | Six sky, expanded walls and horizontal faces | 22.93% | 38.12% | +15.19 |
| Split cells by true surface owner only | Six sky, expanded wall path | 22.93% | 23.00% | +0.07 |
| Per-fragment grid fitted using true face geometry on sampled faces | Same six, split wall path | 23.00% | 47.67% | +24.67 |
| Regrow visible full-cube faces within those masks | Same six, per-fragment reference grid | 47.67% | 54.99% | +7.32 |
| True anchor inverse depth after true projection | Six sky, earlier floor-off projection path | 36.76% | 62.08% | +25.32 |
| Enable same-plane alignment with exact local visible charts and true IDs already supplied | Six sky, exact-chart assembly | 77.58% | 93.71% | +16.13 |
| Perfect decision among seven existing whole exports | All 92, large-contraction vs best pooled strict F1 | 32.13% | 33.81% | +1.68 |
| Correct all material IDs at fixed positions | All 92, **position-plus-ID** F1 arithmetic ceiling | 5.10% | 32.13% | +27.03 |

The 93.71% assembly result requires clean independent charts, excludes 76 decorative non-full-cube blocks from its inputs and counts them as targets. It gives no estimate of what changing assembly alone would do on today's noisy surfaces. None of these controls requires reconstructing hidden blocks. The inverse-depth result concerns relative chart depth in an earlier path. The whole-export oracle still misses 72,560 positions and adds 9,800 extras. [Stage definitions, facade and typed results, and original studies](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-feedback-audit/benchmark/reviews/floating-feedback-audit-01) · [Per-view export chooser](export-chooser.json.gz)

## Best next experiment

Find two separated, independently verified visible full faces on one **continuous wall patch**. Predict the second face's four corners and integer row/column offset from the first. Keep alternatives for half/double scale, grid origin and same-plane versus separate depth. Reserve a third cue such as a jamb, corner, or change of material to judge alignment. Earlier two-face work already tried a corner-agreement gate. The change here is to carry competing local patches and offsets into assembly, with the roof, wall, and porch allowed to stay separate, and to include a sparse or unresolved option for smooth patches. An unrelated facade material can share spacing when independent edges establish the same plane. Every proposed cell still needs visible support.

Measure candidate coverage first. Use a reference-only chooser among the same local grid options and run the chosen charts through assembly before training. If it offers substantial held-layout headroom, test a small learned ranker on source features such as full-face texture fit, held-back corner residual, distance between anchors, and continuity to adjacent faces. Keep an unchanged or abstain option. If no plausible candidate exists, test a new surface-normal or depth model for splitting parallel or offset planes, using separate mask, count and end-placement checks. Train and validate by building; reviewer comments are regression cases. Keep extra placements and recall visible alongside F1.

## Floating branch scorecard

| Six fixed sky composites | Strict placement | Facade placement | Strict position + ID | Facade position + ID | Correct / missed / extra |
|---|---:|---:|---:|---:|---|
| Running sparse default | 12.64% | 12.90% | 4.68% | 4.77% | 273 / 3,768 / 5 |
| Leading experimental large-contraction rule | 27.50% | 27.73% | 5.96% | 5.99% | 890 / 3,151 / 1,542 |
| Additive atlas scale test | 27.50% | 27.73% | 5.96% | 5.99% | 890 / 3,151 / 1,542 |
| Smooth-wall fallback test | 27.50% | 27.73% | 5.96% | 5.99% | 890 / 3,151 / 1,542 |

This round's candidate gain on the six fixed sky images was zero. The experimental lead over the default is 14.86 placement points with 617 additional correct positions and 1,537 additional extras. The integration branch received the reporting workflow only. Keep the reconstruction changes experimental. The fixed 104 production trend is carried at 91.27% strict / 93.42% facade. The full production Visible Builds v3 and Decorative v1 suites were not rerun.
