# Can a screenshot distinguish useful floating wall joins?

Keep `experiment/floating-join-evidence` experimental. On twelve new-layout images, all three proposed rules reproduced the existing export, including every block ID. All possible subsets of the available perpendicular joins also topped out at the existing 27.25% strict placement F1. Eight of those images offered zero perpendicular joins. The coverage and alignment of the source wall grids limit what this decision can change.

On the original 38 images the existing rule found 13,895 correct positions, missed 45,543 and placed 5,302 extras. The mask-and-depth rule found 13,899, missed 45,539 and placed 5,278 extras. It recovered 99 and lost 95 correctly placed positions across the registered frames. Its strict position-plus-ID F1 fell from 4.51% to 4.43%. It kept 28 of 38 perpendicular joins. The fewer proposals and occasional alignment changes reduced correct IDs from 1,773 to 1,741. The new-layout rule kept six of eight joins, leaving 1,409 correct positions, 6,105 misses and 1,418 extras.

| Original 38, image-derived wall grids and masks | Strict placement | Facade placement | Strict position + ID | Facade position + ID |
|---|---:|---:|---:|---:|
| Existing, one agreeing edge | 35.340% | 35.551% | 4.509% | 4.526% |
| Two distinct edges | 35.385% | 35.598% | 4.459% | 4.478% |
| Two separated edges | 35.360% | 35.572% | 4.429% | 4.448% |
| Mask and depth evidence | 35.360% | 35.572% | 4.429% | 4.448% |
| Disable perpendicular joins | 34.469% | 34.668% | 3.943% | 3.957% |
| Reference chooser maximizing strict placement | 35.429% | 35.643% | 4.485% | 4.504% |
| Reference chooser maximizing strict position + ID | 35.324% | 35.534% | 4.624% | 4.642% |

| New-layout 12, image-derived wall grids and masks | Strict placement | Facade placement | Strict position + ID | Facade position + ID |
|---|---:|---:|---:|---:|
| Existing, two distinct, two separated, mask and depth, or geometry chooser | 27.251% | 28.386% | 5.628% | 6.041% |
| Disable perpendicular joins | 27.046% | 28.168% | 5.541% | 5.948% |
| Reference chooser maximizing strict position + ID | 27.113% | 28.239% | 5.636% | 6.050% |

## How it transfers

| Dataset and frames | Existing strict / facade placement | Mask and depth strict / facade placement | Existing strict / facade position + ID | Mask and depth strict / facade position + ID |
|---|---:|---:|---:|---:|
| Original Visible Builds v2, 12 spires | 41.90 / 44.06% | 42.23 / 44.46% | 19.42 / 20.44% | 18.19 / 19.19% |
| Original Visible Builds v3, 18 castles | 35.89 / 35.82% | 35.89 / 35.82% | 3.62 / 3.60% | 3.62 / 3.60% |
| Original Decorative Builds v1, 8 gardens | 21.14 / 22.76% | 21.14 / 22.76% | 2.07 / 2.24% | 2.07 / 2.24% |
| New Visible Builds v2, 4 cloister / arcade | 36.19 / 38.62% | 36.19 / 38.62% | 17.15 / 18.30% | 17.15 / 18.30% |
| New Visible Builds v3, 4 harbor / factory | 23.66 / 24.46% | 23.66 / 24.46% | 2.64 / 2.77% | 2.64 / 2.77% |
| New Decorative Builds v1, 4 chapel / market | 33.10 / 34.66% | 33.10 / 34.66% | 8.18 / 9.06% | 8.18 / 9.06% |

Two agreeing edges improved one original spire by 9.82 points and regressed another by 2.41. Requiring spatial separation cost an additional 3.84 points on a third spire. The evidence rule accepted two additional joins compared with separation alone and produced the same selected component and IDs on every original image. Source gardens offered eight joins over three of eight views, every one with multiple matching segments. Source castle charts offered just three across eighteen views. On 23 of 38 original images there was no perpendicular proposal to accept or reject. Disabling the existing step lost 468 correct positions overall.

The original dataset test split contains twelve spire, eighteen castle and eight garden images. All had been reviewed before this experiment. The twelve new images cover six different layout families, with two views per family, six validation and six training images as labeled by their original dataset. Their frame list was selected by the baseline owner. The rules were committed at `25dbe2b` before fresh scoring. These two evaluations cover 16/144 v2, 22/144 v3 and 12/40 decorative images. Historical fixed and paired lighting cohorts were unmeasured (0/104 historical). This experiment changed the offline runner only.

## What the oracle could select

The decision accepts or rejects a proposed integer translation in the shared local XYZ block frame between perpendicular wall charts. The proposer selects one translation per chart pair. All arms preserve the existing chart inputs, orientations, coplanar constraints, deterministic solver order, largest-component selection, materials and exporter. The bypass removes every perpendicular constraint. Its downstream solver still processes coplanar constraints and selects a component. It reduced fresh strict placement by 0.205 points.

The source runs score every subset of the available perpendicular translations in each frame, including keeping all and keeping none. There were 145 unique subsets across the 38 original views, resulting in 58 distinct exports by frame, and 32 subsets across twelve new images, resulting in 14. To compute each oracle, the evaluator chooses one complete export per frame to maximize the indicated pooled strict F1. The geometry and typed winners can differ. The original geometry winner still misses 45,512 positions, and 45,199 visible targets are absent from **every** registered option. All 6,105 fresh misses are absent from every option. The ceiling applies to the available chart-pair translations with fixed coplanar joins and component selection. It does not measure additional proposed connections or solutions that place disconnected components.

Each unique export is written and decoded. Coordinates, IDs and all score fields were verified against the frozen baseline pack on every source and diagnostic view. The scorer uses one whole-build integer translation selected by geometry. It applies that same registration to the strict/facade geometry and typed metrics. Material-assigned/fallback counts on original images are 18,888/309 for the current rule and 18,868/309 for the evidence rule. The new images had 2,767/60 under both.

## What qualifies a join

The existing solver requires corresponding cell-edge endpoints to lie within both 2.5 native pixels and 19% of the shorter edge. The two-distinct control requires two different segments with the same integer shift. The separated control requires segment midpoints to be at least 1.5 times the longer segment's pixel length apart.

The mask-and-depth arm samples nine positions along each proposed segment, excluding its ends. At a band 1.5–4 native pixels from the segment, the two masks must exclusively occupy opposite sides. Readings from that band and a second band twice as far away extrapolate the source-image depth to the segment. The permissible depth difference is the larger of 4.5% of the local depth magnitude and twice the sum of both local gradients. Seven of nine samples on one segment must pass both checks. A join with separated agreements can then pass. For an isolated or adjacent agreement the mean endpoint error must also be within one pixel and 7% of that segment's length. On ordinary inputs the masks, depth, charts and material IDs all derive from the same RGB screenshot. No camera, reference mask, world depth or neighboring view enters that rule.

## Why the ideal-grid result is different

| Exact visible wall grids and masks, image-predicted directions | Perpendicular proposals | Strict placement, existing / two distinct / supported | Strict position + ID, existing / two distinct / supported |
|---|---:|---:|---:|
| 12 spires | 92 | 83.08 / 83.08 / 83.08% | 75.00 / 75.00 / 75.00% |
| 18 castles | 201 | 90.42 / 92.95 / 92.91% | 50.53 / 51.19 / 51.21% |
| 8 gardens | 104 | 34.08 / 32.47 / 33.70% | 21.12 / 20.35 / 21.17% |
| All 38 | 397 | 87.76 / 90.00 / 89.99% | 50.66 / 51.24 / 51.29% |

Reference grids supply full visible wall membership, projection, scale and phase. This arm also supplies exact frontmost mask ownership to the mask check. Depth and wall directions remain image-predicted. The evidence rule restores both Garden 1 high-right views from 84 to 105 correct blocks relative to two distinct edges. It drops Garden 2 warm low-left from 31.99% to 28.98%, so even exact masks and grids leave a trade-off. It removes 1,430 extras while adding 17 over the existing exact-grid exports. This diagnoses what a reliable single edge can achieve once the upstream measurement provides it. It does not establish a source-only gain.

[Example join overlays](junctions.jpg) · [Original scores, hashes and decoded export receipts](prior.json.gz) · [Fresh scores, hashes and receipts](fresh.json.gz) · [Exact-grid diagnostic](ideal.json.gz)

## Reproduction

Use the linked worktrees' archived `source_material`, native exact-grid and frozen fresh artifacts. Run the four commands in the benchmark Python environment from this checkout. Intermediate packs and per-case receipts are retained under `runs/floating-join-evidence`.

```powershell
python -m benchmark.floating_join_evidence prior --workers 4
python -m benchmark.floating_join_evidence fresh --workers 4
python -m benchmark.floating_join_evidence ideal --workers 4
python -m benchmark.floating_join_report
```

The existing source baseline derives from `experiment/expanded-wall-rulers` at `f389a08`. The exact-grid packets come from `experiment/expanded-wall-grids`, and the fresh source manifest has SHA-256 `f2a51c5fe126eb748379e0f0d736bbb8849a99d248abcf4c9e657060769319aa`. The three machine-readable reports retain each image hash, native reference hashes, mask and chart packet hashes, depth hash, edge evidence, selected chart origins, paired coordinate counts and decoded pack hashes.
