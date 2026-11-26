# Nearby depths, face directions and thin atlas faces

The Java 26.3 screenshot placer at `36d406a` recovers **20,203 of 24,086 visible positions** in the fixed 112 views, with **1,894 extras**. That raises relative geometry F1 from **85.71% to 87.49%**. Precision is 91.43% and recall is 83.88%. Each image receives one integer translation of the entire reconstruction. Errors in the height, orientation, depth or spacing of its parts still count. Material identification is scored separately.

| Placement method | 112 fixed views | 20 alternate views of known structures |
| --- | ---: | ---: |
| Previous default, `cebbffa` | 85.71% | 81.94% |
| One additional horizontal-face pass alone | 85.92% | 82.27% |
| Nearby depth/direction model alone | 85.98% | 82.52% |
| Two thin-face atlas passes alone | 87.25% | 86.47% |
| Combined, conservative single atlas pass | 87.13% | 85.85% |
| **Shipped combination, two atlas passes** | **87.49%** | **86.69%** |

The nearby model compares small depth changes and competing face directions on already observed pixels. The atlas model tests photographed horizontal strips against specific texture tiles, including shifted tiles and competing planes, and verifies the result after rendering the candidates together. A first successful batch can supply neighboring cells for another check against the same original image. The models were fitted on the 72 development views, with folds grouped by building family. The .75 threshold and two-pass limit were selected from those folds. The two new structures were held sealed until the production combination was frozen. [Depth decisions](../nearby-face-01/README.md) · [Atlas model, thresholds and family comparisons](../thin-face-atlas-01/README.md) · [Horizontal pass](../visible-face-second-pass-01/README.md)

## What changed in actual builds

- **112 fixed views.** 801 visible positions gained and six lost, with 151 extras added and 51 removed. Extras coinciding with hidden source blocks rose from 96 to 110. Seventy-eight images improved, 29 tied and five declined. The eight camera abstentions remain in the denominator.
- **20 alternate views.** 353 positions gained and two lost, with 34 net additional extras. Hidden extras rose from 56 to 57. Seventeen views improved, two tied and one declined.
- **Two new structures, four views.** F1 rose from 82.20% to 82.79%. The calibrated-atlas run added 20 correct and 18 extra positions, retained every previously correct position and placed zero extras on hidden structure. The high-right switchback declined from 75.56% to 74.64%. The dock improved from both angles. Camera corners were accurate to within two pixels on all four images. The pre-existing fallback with the atlas missing scored 48.67% under both revisions. [Independent captures, per-view scores and fallback check](../blind-shapes-01/README.md)
- **Internal spacing.** Roof cells placed correctly relative to both posts rose from 194 to 294 of 459 on ten fixed canopies, and from 131 to 232 of 398 on eight alternate views. All 18 measured opening widths now match, and all 243 sampled clear sightlines remain open. Recovered visible undersides across the 80 fixed validation, test and regression views rose from 899 to 998 of 1,472. [Source-angle comparison](comparison.jpg) · [Underside measurements](undersides.json)

## Worst-performing screenshots

The structure-only comparison scores the same frozen predictions on 104 fixed building views. The previous default scores 86.44% and the shipped default scores 88.22%. Canopy 06 left is the weakest structure. The eight texture-only controls were defined by inspecting the full source images. [Recomputed structure scope](structure-scope.json)

Eight fixed close-ups tie at 0% because the camera could not recover scale. They contain four distinct RGB images. The lowest nonempty fixed build, Canopy 06 left, recovered 29 of 84 visible cells and placed 92 extras (28.29%). Its 62-block depth span is visible in the [top-down comparison](worst/worst-footprint.png). The reference spans ten blocks. The lowest alternate view, Colonnade 01 oblique-left, recovered 105 of 187 with 61 extras (59.49%). The weakest new view, the high-right switchback, missed part of the brick pier and placed extra blocks at the lookout height. [All eight abstentions](worst/camera-abstentions.jpg) · [Three lowest nonempty fixed builds](worst/fixed.jpg) · [Two lowest alternate views](worst/alternate.jpg) · [Switchback detail](../blind-shapes-01/switchback-high-errors.png) · [Full fixed and alternate rankings](worst/ranking.json)

## Next test toward 95%

Of the remaining 3,883 misses, 1,433 have strong support in the existing face-direction masks. Wrong blocks already in the prediction cover most reference pixels for 809 misses, including 665 of those with strong direction support. A useful next comparison is to identify an overextended visible plane and **shorten or move its offending section**. Compare that joint change with keeping the current geometry, using the silhouette, atlas alignment and other observed faces. Include top-down depth span and roof-to-post offsets in the decision, evaluated against reference geometry only after export. The 947 misses between four-pixel sample points and 384 abstained close-up positions form separate opportunities. [Final error counts and per-case budgets](remaining-errors.json)

Using the *pre-round* build, a perfect selector over the union of nearby-depth and thin-face proposals could reach 93.08% if it only added cells. It could reach 96.67% if it also deleted all previous extras. These are diagnostic ceilings for that specific pool. Simply accepting more candidates is costly. The extra two-model agreement gate found only three additional correct positions on the fixed suite, and one additional wrong position on the alternate views. Keep it and the neighboring-chart variant experimental. Retire the RGB mask-edge extension, which gained one correct position on the fixed suite and lost three on the alternate views relative to the conservative atlas rule. [Candidate overlap and ceilings](../nearby-face-01/joint-candidate-headroom.json) · [Mask-edge comparison](../thin-face-boundaries-01/README.md)

## Replay and artifacts

The audit reopens the datapacks, validates screenshot and upstream input hashes, registers local cells and scores only blocks visible in that view. The archive contains all 204 candidate packs, decoded cell coordinates and cohort manifests (72 development, 112 fixed and 20 alternate). All production runs identify the same frozen engine revision. Some manifests flag a dirty checkout while diagnostic and test files were being written. The ordinary Shopfront command reproduced every one of its 214 exported coordinates and materials. The independent dock command reproduced all 260. Use the linked fixture reports for restoring the input data. [Frozen control](baseline/README.md) · [Candidate packs](scores.zip) · [Cohort scores](final/summary.json) · [Per-frame scores](final/cases.json) · [Single-pass combined control](conservative/summary.json) · [Shopfront CLI check](cli-check.json)

From the repository root, extract this review’s `scores.zip` into `runs/nearby-replay`. With the input fixtures restored, run:

```powershell
python -m benchmark.nearby_thin_comparison --predictions runs/nearby-replay --extra-predictions runs/nearby-replay/extra --out runs/recomputed-nearby-summary
```
