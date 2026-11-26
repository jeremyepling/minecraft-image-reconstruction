# Where to place the vertical face check

The existing learned vertical-face classifier recovered **24 additional visible blocks across 104 framed photos** when it ran again after roof correction and required a score of at least 0.95. It added zero false placements and lost zero correct positions. Pooled relative geometry F1 increased from 90.2981% to **90.3562%**. Eleven views improved and 93 tied. [Highlighted source pixels for three examples](facade-additions.jpg)

| Verification schedule | Development 72 | Fixed 104 | Alternate 20 | Loading dock and switchback 4 | Shelter and gate 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior default | 91.16% | 90.30% | 88.48% | 83.58% | 83.05% |
| Move the existing 0.80 check before horizontal placement | 91.05% | 90.20% | 88.27% | 83.76% | 83.12% |
| Additional final check at 0.80 | 91.21% | 90.36% | 88.50% | 83.58% | 83.47% |
| Up to three additional checks at 0.80 | 91.26% | 90.38% | 88.50% | 83.58% | 83.47% |
| Additional final check at 0.95 | 91.18% | 90.34% | 88.53% | 83.58% | 83.26% |
| **Up to three additional checks at 0.95** | **91.19%** | **90.36%** | **88.53%** | **83.58%** | **83.33%** |

### Why order matters

Moving vertical verification ahead of dense horizontal recovery gained 45 correct fixed positions but lost 91. Alternate roof positions at the correct offsets from both posts fell from 311 to 234 of 398. All 18 measured opening widths remained correct, and all 243 clear sightlines stayed clear. An opening width alone can miss a large change in the relative positions of the posts and roof.

Keeping the existing order lets initial roof, depth and visibility corrections settle before new wall blocks compete for pixels. A coplanar neighbor nominates each cell. Photographed texels must fit a vertical atlas tile at that depth. The classifier uses disjoint texture samples, neighboring tile evidence, silhouette and depth alternatives. Proposed cells compete together in a second projection. An addition must preserve observed pixels on the rest of the build. A newly measured block can nominate another neighbor. Stop when no cell changes, or after three passes.

The first frozen final-pass setting, 0.80, added two extras at hidden reference coordinates on alternate views. We tightened the classifier threshold to 0.95 after observing that result and also tested the three-pass limit. These historical views are regression fixtures. The higher threshold retained 24 of the 34 additional positions the permissive three-pass rule had recovered on the fixed suite, and eliminated the two new alternate errors. [Individual scores, source hashes and changed packs](comparisons.zip) · [Aggregate counts and openings](results.json)

### Transfer and timing

The selected schedule gained 11 development, 24 fixed, three alternate, and four shelter/gate positions. It changed no blocks on the four loading-dock/switchback photos. The 22 recently released native canopy views also retained identical geometry. The 18 side and elevated camera failures stay in the denominator. The four low and tall roof views score 86.46%. No new placements were at hidden coordinates in any cohort. Anchored fixed roof positions rose from 342 to 344 of 459. All 18 opening widths and all 243 clear sightlines on the fixed and alternate canopies were preserved.

The additional stage took 0.30 to 1.47 process CPU seconds per photograph in three measured examples. Canopy 06 gained two blocks in two successive passes. Bays 02 gained five in one pass. The final historical score including the eight retired wall controls is 89.60% on 112 images. The active 104-view fixed score uses only complete structures. [Timings](timing.json) · [Ordinary build and CLI verification](verification.json)

### Reproduce

The experiments use Java 26.3, fixed camera and surface evidence, and the source photographs and v2 admission list shared with the preceding roof-height benchmark. Scores allow one translation of the complete predicted build. The baseline is `1c2f77d`. The schedule adapter was frozen at `18399ed` on `experiment/placement-order-facade`. The ordinary implementation is `3ff90a2`. Keep the earlier-placement and lower-threshold schedules experimental. The recommendation is to merge the strict final verifier. The change uses the existing trained model and weights.

From the checkout containing `benchmark.facade_placement_eval`, supply the candidate worktree with `--candidate-root` and give each run a private output directory. The regular suite has 72 development and 128 other views, including the 104 fixed, 20 alternate and four loading-dock/switchback photos. Supply `--transfer-inputs ../framed-depth-transfer/runs/transfer-input` to score the separate four shelter/gate photos.

```powershell
python -m benchmark.facade_placement_eval --candidate benchmark.facade_placement_order:strict_repeat --out runs/order --workers 6
python -m benchmark.facade_placement_eval --candidate benchmark.framed_geometry_order:production --out runs/production --workers 6
```
