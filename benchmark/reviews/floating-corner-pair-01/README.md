# Restoring compatible wall grids at a visible corner

On the low-left earth view of Twinspire 1 (O2), the Block Counter measured 7.11 columns on the sandstone wall from 47 image strips, with 0.757 agreement. The corner-derived grid spans 20.73 cells across the top of the same source outline and 18.03 across the bottom. The earlier atlas grid spans 7.11 at both. Four additional sandstone tiles outside the original fitting cells matched the atlas across multiple rows and columns.

Restoring that wall alone puts its closest corner edge 4.87 native pixels from the neighboring grid. The assembler allows 2.5. The 162-block component splits and the exporter selects the other tower, swapping 85 correct blocks for 80 different ones. Restoring both adjacent earlier grids gives a 2.04-pixel corner match and keeps every previously selected chart and join. The pair's ordinary pack has 84 correct blocks, zero extras and 47 exact IDs. The prior build had 85 correct blocks, 77 extras and 28 exact IDs. The lost strict target was a red-sandstone underside with 30 visible pixels. The prior build placed chiseled sandstone there. All 84 previously correct facade-required positions survive.

| O2 expanded input | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---|---|---|
| Keep corner grids | 46.96% / 46.67% | 15.47% / 15.56% | 85 / 115 / 77 |
| Restore receiver alone, prior diagnostic | 56.74% / 56.94% | 23.40% / 23.49% | 80 / 120 / 2 |
| Source-selected pair | 59.15% / 59.36% | 33.10% / 33.22% | 84 / 116 / 0 |

## Decision and controls

The rule considers an accepted corner only when both earlier wall grids are available. The receiver needs a confident repeated-texture width from the existing Block Counter. Both ends of its current grid must miss the integer width by at least one column, and both ends of the prior grid must fit within half a column. The prior grid must have at least two atlas matches to one material and additional matches to the same material outside its fitting cells, spanning at least two rows and columns. Atlas confidence must be at least 0.5. A trial assembly must retain every existing selected chart and corner connection and join the two proposed walls. Both proposed charts must belong to the original export. At most one pair changes per image. The scorer receives native reference files after source selection.

Across 84 active frozen images, ten images offered 13 pairs. Only O2 changed. The two Watchpost views had atlas matches but lacked a confident receiver width and lost connected surfaces under the pair replacement. Cloister lacked an earlier receiver grid. None of the six fixed sky inputs offered a pair. O2 was previously reviewed and informed the decision rule. The other 83 views check for regressions, with no independently successful second-building change in this cohort.

A reference-only chooser maximizing pooled strict placement F1 across KEEP and the available single-pair exports could raise strict F1 from 33.54% to 33.62%, with 13 correct gained, one lost, 240 extras removed and 36 added. Choosing for exact material and position instead loses 79 net correct positions. [Per-pair source measurements and decoded outcomes](rows.json.gz) · [Pooled, dataset and oracle scores](summary.json)

## Sparse floating replay

The connected-atlas default at `9435934` uses sparse charts and the wall-only assembler. Its six saved inputs reproduced their published geometry. On an additional sparse O2 input, the original export selects the opposite tower with 52 correct, 148 missed and zero extras. The guard declines the pair because it would replace that tower. Forcing the corrected corner produces 70 correct, 130 missed and zero extras on the other tower. This forced result uses the same source-only grids and a diagnostic intervention on the final decision. [Sparse source and pack receipts](sparse.json.gz)

Prefer this pair to the expanded baseline when the rule accepts it. Keep `experiment/floating-corner-pair-repair` experimental for the sparse floating default and production. A clean new source-selected building or a gain through the actual sparse exporter would strengthen the case for integration. Only the benchmark implementation and evidence landed on the challenger.

Reproduce with `python -m benchmark.floating_corner_pair_repair` and `python -m benchmark.floating_corner_pair_sparse` using the saved chart and source packets. The frozen baseline is the large-contraction chart cohort audited at `003c5fb`. The study covers 30 Visible Builds v2, 30 Visible Builds v3 and 24 Decorative Builds v1 views. The separate fixed-104 production trend carries forward at 91.27% strict and 93.42% facade.

## Floating branch scorecard

| Frozen inputs | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---|---|---|
| Integration at `9435934`, six sparse fixed sky exports replayed | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Pair rule on the same six sparse inputs | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Expanded six fixed sky inputs, both arms | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |
| Expanded 84-view baseline | 33.54% / 34.01% | 5.51% / 5.61% | 18,323 / 61,187 / 11,420 |
| Pair rule on the same 84 images | 33.56% / 34.04% | 5.55% / 5.65% | 18,322 / 61,188 / 11,343 |

The fixed sky controls changed by zero correct and zero extras. Across the expanded 84, strict placement F1 rose 0.02 points and facade placement F1 rose 0.02 points. There was one fewer strict correct and 77 fewer extras. The integration branch received no behavior changes.
