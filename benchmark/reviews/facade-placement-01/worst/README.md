# Ten lowest-scoring structures

Open the [overlay viewer](viewer.html) from a local checkout to switch between the source photograph, surfaces, measured grid, reconstruction, visible reference and placement errors at full resolution. The opacity slider reveals the source pixels under each overlay.

Each of the 104 fixed, whole-structure views contributes its visible-block counts. For each distinct layout, we pooled true positives, extras and misses over all of its views and ranked by relative geometry F1. The 40 layout identities come from full occupancy after accounting for translations, rotations, horizontal reflections and palette changes. All parts of a predicted build receive one common translation during scoring. Each column shows its layout’s lowest-scoring view. [Zoomable stage map](comparison.jpg) · [All ten layouts and individual frame scores](ranking.json) · [Measured stages and source hashes](stage-traces.json)

| Rank | Layout and stage sheet | Views | Layout F1 | Shown view F1 | Correct / visible | Extras |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [Canopy 00 (open)](structure-01.jpg) | 1 | 70.13% | 70.13% | 27 / 50 | 0 |
| 2 | [Canopy 04](structure-02.jpg) | 1 | 72.73% | 72.73% | 36 / 60 | 3 |
| 3 | [Colonnade 01](structure-03.jpg) | 1 | 75.22% | 75.22% | 126 / 191 | 18 |
| 4 | [Galleries](structure-04.jpg) | 4 | 78.35% | 75.05% | 188 / 274 | 39 |
| 5 | [Canopy 07](structure-05.jpg) | 1 | 82.52% | 82.52% | 59 / 83 | 1 |
| 6 | [Physical Shade](structure-06.jpg) | 2 | 83.62% | 81.99% | 255 / 365 | 2 |
| 7 | [Dev Relief 00](structure-07.jpg) | 1 | 83.70% | 83.70% | 380 / 480 | 48 |
| 8 | [Shopfront](structure-08.jpg) | 8 | 86.06% | 70.31% | 103 / 169 | 21 |
| 9 | [Facade](structure-09.jpg) | 2 | 86.20% | 82.69% | 277 / 361 | 32 |
| 10 | [Relief 00](structure-10.jpg) | 1 | 86.21% | 86.21% | 394 / 482 | 38 |

## Reading a column

The first row shows the entire source photograph. The detected-surface overlay traces every returned region in orange, with one surface highlighted in cyan. The crop, straightened texture, measured grid and atlas tile follow that surface. The selection favors a complete tile, then a usable chart with a located axis or measured dimensions. Among comparable charts it favors overlap with missed visible blocks, prioritizing the roof on canopy probes. Evaluation labels guide this illustration after scoring.

The Surface Finder count is a display diagnostic. It compares masks with same-plane, edge-connected visible reference surfaces at native resolution, greedily pairing the highest IoU first down to 0.50. A partial outline can contribute pixels to the build even if it falls below the threshold. The Block Counter uses its frozen per-surface chart, dimensions and grid origins. The source-grid overlay projects each measured axis through its chart and clips every line to the detected mask; two located origins are needed to place a complete tile. Question marks indicate a missing estimate.

The material row replays the calibrated vanilla atlas matcher on a complete tile. If the frozen chart lacked a grid origin, a separate atlas diagnostic tries to register it at the already measured spacing. Each panel identifies which source located the tile. The original measurement still supplies both grid rows and the evaluated coordinates. The comparison material comes from the labeled face at the center of the tile. Unresolved panels indicate insufficient evidence. The stage trace records the materials encoded in the evaluation pack.

The final rows place the decoded export on the source photo in teal, draw only the visible reference blocks in purple, then compare both after the shared scoring shift. Green is correct, red is extra and blue is missing. For overlap, the image shows the frontmost colored face. The counts include all scored coordinates. The overhead panel shows the highest scored block in each X/Z column. Each source overlay has a native-resolution copy in the overlays folder.

## Stage counts for the ten shown views

| Layout | Surface matches / visible | Detected | Straightened | Both dimensions measured | Both grid origins located | Sampled tile guess / recorded center |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Canopy 00 (open) | 8 / 10 | 12 | 9 | 7 | 0 | unresolved / unresolved |
| Canopy 04 | 7 / 11 | 13 | 8 | 4 | 0 | unresolved / unresolved |
| Colonnade 01 | 12 / 19 | 18 | 12 | 5 | 0 | unresolved / unresolved |
| Galleries | 20 / 30 | 31 | 20 | 7 | 0 | unresolved / unresolved |
| Canopy 07 | 9 / 11 | 9 | 8 | 6 | 0 | unresolved / unresolved |
| Physical Shade | 4 / 9 | 8 | 5 | 2 | 0 | unresolved / unresolved |
| Dev Relief 00 | 9 / 20 | 12 | 8 | 5 | 0 | unresolved / unresolved |
| Shopfront | 12 / 19 | 15 | 13 | 2 | 0 | unresolved / unresolved |
| Facade | 12 / 20 | 15 | 11 | 2 | 0 | unresolved / unresolved |
| Relief 00 | 10 / 20 | 15 | 10 | 5 | 1 | smooth_stone / smooth_stone |

## Ten lowest individual views

A layout can appear more than once in this separate list.

| View | F1 | Correct / visible | Extra |
| --- | ---: | ---: | ---: |
| shape-benchmark-open-26.3--train-canopy-00-right | 70.13% | 27 / 50 | 0 |
| legacy-light_gray_concrete-dev-shopfront-plaster-low-left | 70.31% | 103 / 169 | 21 |
| shape-benchmark-26.3--train-canopy-04-left | 72.73% | 36 / 60 | 3 |
| legacy-light_gray_concrete-holdout-galleries-plaster-low-left | 75.05% | 188 / 274 | 39 |
| inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right | 75.22% | 126 / 191 | 18 |
| legacy-light_gray_concrete-holdout-galleries-masonry-high-right | 75.36% | 185 / 297 | 9 |
| legacy-light_gray_concrete-holdout-galleries-masonry-low-left | 75.82% | 185 / 274 | 29 |
| legacy-light_gray_concrete-holdout-portico-plaster-low-left | 76.49% | 148 / 209 | 30 |
| legacy-smooth_stone-holdout-portico-plaster-low-left | 77.89% | 148 / 209 | 23 |
| legacy-light_gray_concrete-dev-shopfront-masonry-high-right | 81.36% | 192 / 216 | 64 |

## Reproduce

Run from the repository with the scored fixture data and calibrated atlas available. Supply the winning score directory, including the exported packs and decoded coordinates.

```powershell
python -m benchmark.roof_span_worst --run "runs/roof_then_walls" --out "benchmark/reviews/facade-placement-01/worst"
```
