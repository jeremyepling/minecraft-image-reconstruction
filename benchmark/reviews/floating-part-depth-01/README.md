# Can a wall keep its grid and put the projecting parts at a different depth?

On the 84 active conditions, the source rule gained 132 net correct positions and removed 127 net extras. The clearest example is Riverside Mill low-left. The roof fascia and the cross-shaped window frame move forward together. Correct placements rise from 121 to 160 and extras fall from 82 to 48. Keep the change experimental. The six fixed sky exports tied, two textured views regressed, and the other-layout trials use earlier benchmark images.

[Mill source grid](examples/train-riverside-mill-1-warm-low-left/source.jpg) · [Native visible-face comparison](examples/train-riverside-mill-1-warm-low-left/visible-reference.jpg) · [Decoded placement changes](examples/train-riverside-mill-1-warm-low-left/placement-changes.jpg)

## What moved

The large-contraction export from `e97dd44` places every cell in one wall chart on the same plane. This test adds a signed integer offset along the wall normal to individual cells when the joined chart is converted into voxels. The source pixel grid, UV coordinates, sampled material IDs, chart poses and chart joins stay fixed. Offsets move existing visible proposals. There is no infill behind them. The native visible references enter the two diagnostic arms and the scoring pass. A zero-offset replay on Mill and Shrine reproduced every local and aligned position and ID.

For each proposed wall cell, the source arm samples nine inset points from the projected grid. At least five must belong to the existing image mask. It takes median source normal and inverse depth and requires the sampled normal to agree with the chart normal (cosine ≥ 0.90). Neighboring cell differences estimate the wall's slope in the depth map. Two clusters of residual depth can establish parallel parts if their separation exceeds four within-band median deviations, five neighbor noise estimates and 2.5% of the median depth reading. Each minority part needs at least four adjacent cells, two contacts with the main wall and a median discontinuity at those contacts of at least 35% of the cluster separation. The sampler moves a qualifying part one block forward or back. The two source fields come from the archived single-image Depth Anything and MoGe passes.

Two controls locate the remaining error. **Reference step on source parts** retains the source-selected cells and measures one relative step per part from adjacent visible parallel full-cube faces. It caps the difference at four blocks and uses zero when the strongest vote is ambiguous or has fewer than two samples. **Reference parallel parts** uses the native frontmost face ID at each of the nine samples. At least five must report the same outward direction and physical plane. It moves cells one block toward their measured depth relative to the most frequently observed plane on that chart. This control changes membership and depth. It does not refit the UV grid or infer the true extent of a partly hidden wall. Both controls can worsen final placement.

## Paired decoded results

Each row reports pooled strict placement, facade placement, strict position plus material, facade position plus material, and strict correct / missed / extra counts. All four scores share one whole-build integer translation per export. Model settings were fixed before the archived comparison. The all-92 replay preserves the historical series. The predeclared active mask omits eight conditions from smooth-dominated layouts.

| Conditions | Chart treatment | Strict | Facade | Strict + ID | Facade + ID | Correct / missed / extra |
|---|---|---:|---:|---:|---:|---:|
| Active 84 | Large-contraction flat chart | 33.54% | 34.02% | 5.51% | 5.61% | 18,323 / 61,187 / 11,420 |
| Active 84 | Source depth parts | **33.78%** | **34.28%** | **5.74%** | **5.84%** | 18,455 / 61,055 / 11,293 |
| Active 84 | Reference step on source parts | 33.86% | 34.36% | 5.80% | 5.90% | 18,497 / 61,013 / 11,251 |
| Active 84 | Reference parallel parts | 38.24% | 38.92% | 7.67% | 7.85% | 20,898 / 58,612 / 8,878 |
| Historical 92 | Large-contraction flat chart | 32.13% | 32.67% | 5.10% | 5.20% | 21,240 / 72,356 / 17,374 |
| Historical 92 | Source depth parts | 32.33% | 32.89% | 5.29% | 5.39% | 21,372 / 72,224 / 17,247 |
| Historical 92 | Reference parallel parts | 35.65% | 36.34% | 6.92% | 7.10% | 23,573 / 70,023 / 15,074 |

On the four conditions from the reviewed Mill-1 and Shrine-02 layouts, the source rule raised strict placement from 20.42% to 22.49%. On the other 80 active conditions it raised strict placement from 34.00% to 34.18% with 93 more correct blocks and 93 fewer extras. Across all 84, the registered placements gained 374 correct positions and lost 242. They added 983 extras and removed 1,110. Ten images gained correct blocks and two lost them. The predeclared active subset covers 30/92 active v2 captures, 29/108 active v3 and 23/40 Decorative v1. Two of those captures also appear under the six-sky condition. Full production scores were not recomputed.

## Where an offset helps or hurts

- Mill low-left moved 88 of 197 cells on mask-0. Of the 88, 77 agreed with the signed native per-cell control. The export gained 60 correct blocks and lost 21. On Mill high-right the source rule retained the flat chart at 97 correct / 53 extra. The reference partition placed 129 correct / 21 extra. [Low-left mask](examples/train-riverside-mill-1-warm-low-left/mask.jpg) · [High-right source prediction](examples/train-riverside-mill-1-warm-high-right/source.jpg) · [High-right reference](examples/train-riverside-mill-1-warm-high-right/visible-reference.jpg)
- Shrine low-left produced two depth bands but had insufficient local step evidence to move a patch. Its right wall and stairs still share one chart, at 60 correct / 64 extra. The one-block native partition moved 80 cells and worsened the export to 48 correct / 76 extra. Distinct horizontal treads, risers, part orientation and local grid origin are outside this depth-only control. [Shrine source grid](examples/train-shrine-02-ochre-low-left/source.jpg) · [Native comparison](examples/train-shrine-02-ochre-low-left/visible-reference.jpg)
- Viaduct rose from 111 correct / 100 extra to 170 / 41. Its measured zero-depth band differs between source and reference. Across 25 comparable boundary pairs, 17 have the correct signed one-block step. This relative comparison survives changes in which band supplies the zero plane. [Source prediction](examples/train-viaduct-01-riverstone-low-left/source.jpg) · [Placement changes](examples/train-viaduct-01-riverstone-low-left/placement-changes.jpg)
- Castle 01 ashlar high-right, original RGB, dropped from 562 correct / 403 extra to 548 / 417. Only 3 of 16 comparable boundary pairs had the correct step direction. On Stepped Hall, all four comparable pairs had the correct direction but none had a one-block step. It dropped from 45 / 6 to 40 / 11. [Castle source prediction](examples/test-castle-01-ashlar-high-right/source.jpg) · [Castle reference](examples/test-castle-01-ashlar-high-right/visible-reference.jpg) · [Stepped Hall source prediction](examples/train-steppedhall2-cool-high-right/source.jpg)

Across the 92 conditions the source rule moved 890 proposed chart cells, including cells outside the selected export. Of these, 647 matched the native one-step offset, 161 had a different native offset and 82 lacked a comparable visible parallel face. Along source-proposed part boundaries, 211 of 268 comparable adjacent pairs had the correct forward/back ordering and 79 had the exact one-block difference. Another 62 pairs lacked a comparable native face. The bounded native-step replay on the same source masks added only 42 net correct blocks beyond the source estimate. The larger opportunity was expanding the set of correctly identified parallel parts.

## Replay and review

Run `python -m benchmark.floating_part_depth --workers 5` and `python -m benchmark.floating_part_report` from the experiment checkout with the existing benchmark Python environment. The runner verifies input and archived source hashes, validates native pixel counts, and grades the decoded local packs. Empty interventions reuse the identical archived export. The active filter comes from the versioned `floating-textured-cohort` audit. The per-condition archive records source and model-field digests, reference hashes, final pack hashes, chart membership, cell offsets, boundary evidence and placement changes.

[All condition results](conditions.json.gz) · [Pooled scores](summary.json) · [Review image manifest](examples.json)

## Floating branch scorecard

Six fixed sky composites, with the connected-atlas runtime `52de2c9` measured in the saved integration replay at `c0accfd`. The integration scores are carried forward. This benchmark branch measures the additional part-depth behavior on the expanded large-contraction path.

| Method | Strict | Facade | Strict + ID | Facade + ID | Correct / missed / extra |
|---|---:|---:|---:|---:|---:|
| Floating integration | 12.64% | 12.90% | 4.68% | 4.77% | 273 / 3,768 / 5 |
| Large-contraction and source depth parts | 27.50% | 27.73% | 5.96% | 5.99% | 890 / 3,151 / 1,542 |
| Visible-reference parallel parts | 35.62% | 36.03% | 9.39% | 9.49% | 1,153 / 2,888 / 1,279 |

The source change gained zero correct blocks, extras and F1 points over large-contraction on these six images. Relative to integration, the expanded path adds 617 correct blocks and 1,537 extras, gaining 14.86 strict points. The measurement and review assets landed on `experiment/floating-part-grids` only. Keep this experiment off production main pending an independent test on new textured layouts. The separate fixed-104 production trend is carried forward at 91.27% strict and 93.42% facade.
