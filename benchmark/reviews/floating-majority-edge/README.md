# Majority support and a visible material seam

Keep this challenger experimental. On Twinspire the fitted top edge follows the observed brick-to-sandstone transition to within a median 1.0 native pixel, versus 12.0 pixels for the approved row proposal. Across the 84 active conditions the fit lost five net correct placements and removed seven extras. No reconstruction behavior landed on integration or main.

The reviewed Twinspire wall uses the scale chart from the prior review. Its eight proposed top cells each have 54–61% source support in the wall mask and on the inward side of the color transition. The visible-only scoring reference independently assigns 60–62% of each original quadrilateral to the east brick plane. All eight therefore pass the strict majority test. The fitted grid finds the transition 0.375 grid units below the old upper edge across all columns, using the opposite wall edge as the hinge. After correction, 98–100% of each footprint lies on the correct brick plane. The existing atlas IDs are preserved to isolate geometry, leaving one smithing-table error. The target wall is absent from the selected export component. All four decoded Twinspire builds have 110 correct positions, 184 misses, 63 extras and 42 correct IDs.

[Source, cyan grid, and gold reference boundary](twinspire-row.png) · [Cell-by-cell measurements](test-twinspire2-earth-high-right.json.gz) · [Corner visibility and seam distances](twinspire-upper-corners.json)

The seam error uses 21 upper-edge samples per cell against native adjacent brick and sandstone pixel owners. The 16 projected upper brick corners sampled across the eight cells were all marked occluded. Their median nearest projected-corner distance changes from 8.59 to 3.30 pixels. Fully visible corners elsewhere on the target plane show a small increase in median grid phase error from 0.020 to 0.026 cells. Only the first measurement directly checks the visible seam.

## Paired export experiment

The frozen rule takes the previous contiguous, source-atlas-ranked top-row proposals. For each column it compares inward and outward source colors. A coherent transition and more than half the native cell area inside both the predicted wall mask and inward color region are required. At least three coherent column transitions at the same row can adjust the chart height, phase and slope. Every existing cell index and atlas ID is retained. A material transition provides local image alignment evidence and never assigns depth. The predicted wall mask remains a proxy for plane ownership.

The generalization replay starts all four arms at the large-contraction packets on all 84 active conditions. It applies the identical rule to every nonempty vertical chart, including charts absent from the final component. Each distinct packet goes through the normal oriented wall and horizontal-face assembly, component selection, ZIP decode, and whole-build registration. Identical packet inputs reuse a verified decoded export. Five focused replays separately hold the exact previously reviewed packet and wall fixed. Chapel kept four of eight proposed cells, Shrine rejected all five, and Riverside and Sawtooth had no proposals.

| 84 active conditions | Strict F1 | Facade F1 | Position + ID F1 | Facade + ID F1 | Correct / missed / extra |
|---|---:|---:|---:|---:|---:|
| Large-contraction | 33.542% | 34.015% | 5.512% | 5.612% | 18,323 / 61,187 / 11,420 |
| Original top row | 33.702% | 34.175% | 5.473% | 5.572% | 18,493 / 61,017 / 11,740 |
| Strict majority only | 33.555% | 34.027% | 5.540% | 5.641% | 18,348 / 61,162 / 11,502 |
| Majority and seam fit | 33.537% | 34.036% | 5.511% | 5.610% | 18,318 / 61,192 / 11,413 |

The majority gate alone gains 72 correct positions and loses 47, adding 82 net extras. The seam arm gains 158 and loses 163. Its validation strict F1 increases from 29.66% to 30.86%, while train goes from 28.21% to 28.16% and test from 35.24% to 35.19%. These are reviewed regression examples. The motivating Twinspire example belongs to the test split. The 84 conditions cover 82 release captures (30/92 active v2, 29/108 active v3 and 23/40 Decorative). This round did not rerun full production suites. [All per-case source and packet hashes, paired gains and decoded scores](cohort.json.gz)

Run the following modules from the experiment checkout with the benchmark Python environment and saved floating packets present.

    python -m benchmark.floating_majority_edge
    python -m benchmark.floating_majority_cohort
    python -m benchmark.floating_majority_overlay

## Floating branch scorecard

The six fixed sky composites use the running connected-atlas baseline measured at 7033891, carried forward at integration baaa572. Candidate rows use the same images. The separate fixed-104 production trend carries forward at 91.27% strict and 93.42% facade (20,773 correct, 2,929 missed, 1,046 extra).

| Six fixed sky composites | Strict F1 | Facade F1 | Position + ID F1 | Facade + ID F1 | Correct / missed / extra |
|---|---:|---:|---:|---:|---:|
| Integration connected atlas, carried forward | 12.64% | 12.90% | 4.68% | 4.77% | 273 / 3,768 / 5 |
| Large-contraction control | 27.50% | 27.73% | 5.96% | 5.99% | 890 / 3,151 / 1,542 |
| Original top row | 27.44% | 27.67% | 5.92% | 5.94% | 895 / 3,146 / 1,587 |
| Strict majority only | 27.55% | 27.79% | 5.95% | 5.98% | 893 / 3,148 / 1,548 |
| Majority and seam fit | 27.59% | 27.83% | 5.95% | 5.97% | 895 / 3,146 / 1,552 |

The seam arm gains five net correct placements and adds ten extras against its matched control on the six sky images. Relative to integration it has 622 more correct placements and 1,547 more extras. Only experimental code and measurements landed on experiment/floating-majority-edge.
