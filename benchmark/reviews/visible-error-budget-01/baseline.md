# Where the current build loses visible blocks

The ordinary placer at `bec7c94` recovered 20,510 of 23,702 visible blocks, missed 3,192 and added 1,157 extras. Fixed-suite F1 is **90.414%** across 104 whole-structure screenshots from 40 layouts. All 104 have a metric camera. The 18 failed-camera side and elevated challenge views belong to separate cohorts.

## Each missed block counted once

The rows describe the faces of a missed block that appeared in the original screenshot. The columns describe a native-resolution render of the decoded export using the estimated camera. A block is mostly uncovered when more than half its visible pixels hit empty space in this render. For a mostly covered block, the face-axis decision uses a majority of its covered pixels. Ties at either decision are shown separately. A hit with the right axis can land on a different depth or a different tile on the same plane.

| Visible faces of missed block | Mostly uncovered | Covered, same axis | Covered, different axis | Ties | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wall only | 767 | 592 | 231 | 2 | 1,592 |
| Wall and top | 15 | 32 | 29 | 0 | 76 |
| Wall and underside | 5 | 17 | 9 | 0 | 31 |
| Top only | 669 | 26 | 279 | 11 | 985 |
| Underside only | 393 | 7 | 105 | 3 | 508 |
| **Total** | **1,849** | **674** | **653** | **16** | **3,192** |

Most misses have appreciable source support. Their total observed footprint is at least 64 native pixels on 1,708 blocks, 16–63 pixels on 1,057 and 1–15 pixels on 427. These areas combine all visible faces of a block. A long thin strip can have substantial area.

**Turn detected horizontal pixels into a measured surface.** In the 1,849 mostly uncovered misses, 452 have less than half their footprint covered by any detected mask. Another 792 have majority mask coverage but lack majority coverage from a mask with the correct face axis. These include 397 top-only and 235 underside-only misses. A further 486 have the correct mask axis but lack both recorded block dimensions at that coverage. The last 119 have both dimensions and lack both grid origins at that coverage. These are observed stage outputs. A small feature can contribute to placement even when it covers less than half a block. Only 467 of the 20,510 correctly placed blocks have a fully measured grid covering at least half their footprint, so missing origins alone make a poor rejection criterion.

**Test where moving an existing panel is appropriate.** On 603 of the 674 same-axis misses, most covering pixels belong to an extra block. On the remaining 71, most belong to a block correctly placed elsewhere. Among those 674, the covering same-axis rays predominantly land at a coordinate farther toward the camera along the target face normal on 496, behind it on 134, and at the same level on 44. The classification uses the normal-axis coordinate after the shared scoring translation. A nearer panel can hide a recess and a farther panel can stand in for a protrusion. Conversely, on 424 of the 653 different-axis misses, most covering pixels belong to correct blocks elsewhere. Another 228 mostly hit extras and one ties. These groups identify candidates for a move or a separately witnessed addition. One moved block can affect multiple target footprints.

**Extras need their own check.** Of 1,157 extras, 771 put most of their frontmost projected pixels inside the building silhouette, 385 put most onto background, and one has no first-hit pixels. The source world contains an unobserved occupied cell at 106 extra coordinates. This hidden-occupancy count overlaps the projection groups.

## Jeremy's named examples

Each row uses the representative screenshot from the published lowest-scoring layout viewer. The earlier surface-seam images remain useful even where changing a mask left the export unchanged. [Stage viewer](../facade-placement-01/worst/viewer.html) · [Prior seam and projection experiments](../wall-section-01/README.md)

| Shown view | Misses | Mostly uncovered | Same axis | Different axis | Ties |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canopy 04 | 24 | 17 | 0 | 6 | 1 |
| Colonnade 01 | 65 | 18 | 6 | 39 | 2 |
| Dev Relief 00 | 100 | 30 | 65 | 5 | 0 |
| Relief 00 | 88 | 41 | 38 | 9 | 0 |
| Shopfront | 66 | 46 | 8 | 12 | 0 |
| Facade | 84 | 40 | 34 | 10 | 0 |

All 34 same-axis Facade misses predominantly hit a nearer normal-axis coordinate. On Dev Relief 00, 27 are nearer, 26 farther and 12 on the same level. Colonnade 01 has 61 top-only misses. Of its 18 mostly uncovered misses, 17 have majority coverage from a correct-axis mask but lack measured dimensions there. Shopfront has 18 mostly uncovered wall-only misses in this same correct-axis but missing-dimensions group.

Count views when choosing examples. The saved family tags contribute 355 misses over four Galleries views, 319 over twelve Twinspire views, 271 over eight Portico views and 265 over eight Shopfront views. The JSON retains the family identifier and layout name on each row.

At the existing extra count, recovering every mostly uncovered miss would raise F1 to 94.71%. Recovering every miss with any wall face would reach 94.37%, and recovering all top-only and underside-only misses would reach 93.91%. Each calculation starts from the current default and assumes perfect clean additions. Reaching 95% with extras fixed would require at least 1,982 net correct additions. These ceilings overlap.

## Reproduce the baseline audit

Use the saved complete-structure fixture, its visible labels and the `wall-section-round` baseline exports. The audit verifies the screenshot, reference and datapack fingerprints, reconstructs the registered cell coordinates and renders every native pixel with the estimated camera. Geometry uses a single whole-build translation per screenshot. The recorded normals and grid coordinates are scoring evidence. [Per-frame and per-family counts](baseline.json)

```powershell
python -m benchmark.visible_error_budget --run ../wall-section-round/runs/control --workers 4 --out benchmark/reviews/visible-error-budget-01/baseline.json
```
