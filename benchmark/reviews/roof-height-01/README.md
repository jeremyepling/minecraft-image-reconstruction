# Choosing the roof height before placing blocks

A one-block height error on a large visible surface can send dozens of blocks to the wrong positions. On Canopy 06 left, the build already contained blocks at the correct highest course. The main roof patch, however, had been projected onto a plane one block too low. Moving that patch up **before final placement** increased the score from **41.33% to 93.08%**. The build went from 31 to 74 of 84 visible positions, with extras falling from 35 to one. Roof positions at the correct offsets from both posts rose from eight to 50 of 60. [Three photographs and their before/after overlays](height-comparison.jpg)

## Ordinary build

The Java 26.3 placer makes a provisional build. It compares photographed texels across a large horizontal patch at different integer heights and checks the visible rim and supports independently. The patch must account for at least 65% of the sampled horizontal area and the winning texture score must beat the runner-up by at least 0.18. If both methods agree and their height differs by exactly one block from the existing resolved plane, the placer rebuilds with that plane fixed. A strict atlas check then verifies individual horizontal faces that have native pixel evidence. Missing camera or atlas evidence leaves the existing fallback in control. Only photographed faces can justify blocks. The method adds no unseen roof interior, back wall or support.

| Placement method | Fixed 104 | Alternate 20 | Loading dock / switchback 4 | Shelter / gate 4 | Fixed roof positions at correct offsets from both posts, of 459 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous default | 90.10% | 88.29% | 83.64% | 83.05% | 296 |
| Late atlas with support fallback | 90.23% | 88.65% | 83.47% | 83.33% | 304 |
| Unrestricted early height from texture | 90.22% | 88.29% | 85.58% | 76.29% | 338 |
| Require rim and atlas agreement for an early move | 90.28% | 88.29% | 83.64% | 83.05% | 338 |
| **Early agreement plus strict face verification** | **90.30%** | **88.48%** | **83.58%** | **83.05%** | **342** |

On the 104 framed views, the selected method recovers 20,462 of 23,702 visible positions, with 1,157 extras. Seven images improved and 97 tied. It gained 51 correct positions, lost zero, removed 37 extras and added two. The preexisting 106 extras at hidden reference coordinates were unchanged. The 20 alternate views gained 14 correct positions without an extra. The loading dock gained one extra visible-world error. All 18 measured canopy widths stayed exact and all 243 measured clear sightlines stayed open. The 104-view camera-registered score is also 90.30%. Recombining the eight retired empty-output texture controls for historical comparison yields 89.54% over 112 images, previously 89.35%. [Breakdown by split](results.json) · [Relative roof positions and explicit height decisions](heights.json) · [Updated stage maps and overlay viewer](worst/README.md)

The main improvement is concentrated in one familiar photo. Of the 18 fixed and alternate canopies, 15 have a roof at course five and three at course six. The previous build's highest occupied course already matched on all 18. The early agreement rule explicitly chose the correct roof in two views and abstained in 16. The unrestricted early rule also chose a plausible plane for a multi-level shelter and lost 95 correct blocks across the four shelter/gate views. A vertical-rim-only decision read 12 of the 18 canopy heights correctly but did little for placement. Keep unrestricted early revision, loose atlas placement and stand-alone rim placement experimental. [Atlas branch and blocker trace](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/5435770/benchmark/reviews/roof-height-atlas-01) · [Rim and vertical band experiments](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/660dffd/benchmark/reviews/roof-height-rim-01)

## New structures and angles

These panels are separate from the fixed benchmark. Every selected screenshot shows the complete building with at least 2% clearance at every edge. All camera failures count as misses. The predictions use a single screenshot. Reference geometry and additional angles are for scoring.

- **Four native photos of two newly built structures at roof courses three and eight.** The old and selected builds made identical placements. Together they recovered 313 of 404 visible blocks, with seven extras (86.46% F1). The two tall timber views scored 95.04% and 99.02%. The low masonry views scored 72.97% and 0%. The latter lacked a supported image-only camera. [Photos, per-view scores and frozen inputs](../roof-height-new-views/README.md)
- **Twelve side views of six additional canopies at courses five through seven.** The image-only camera failed on every photo, so both methods recovered zero of 954 visible blocks. With the recorded camera supplied solely for diagnosis, the previous placer scored 97.17% and the selected method scored 98.52%. The separate vertical-rim method identified 11 of 12 roof heights with one abstention, including eight views with zero horizontal roof pixels. [Side-view results and visibility](panel/README.md)
- **Six elevated views of those same six canopies.** All six also failed the ordinary camera stage. With the camera supplied, the whole-patch atlas read the correct roof height on all six. Every candidate retained the previous geometry, with 540 of 552 visible positions and 9,830 extras (9.89% F1). Three projections spread over far too much area, despite having an explicit height available. [Elevated panel, per-view results and exports](panel/README.md)

The next experiment is a camera fallback for a complete structure on weakly patterned ground. Use the vertical edges and observed texture period or atlas transitions on the building itself to propose metric cameras. Check each against independent photographed edges and block spacing. Start with the 18 camera misses, keep the four new native views and the established fixed suite as regression checks, and score over-wide placements explicitly. The supplied-pose runs show why both camera support and visible area must be measured.

## Reproduce

The ordinary placer reproduced the evaluated coordinates and materials across all 204 established exports. Its 22 additional exports also matched on all three new panels. The full screenshot-to-datapack command reproduced the 75-block Canopy 06 export. Over three warm timing runs, the added work took 0.05–0.22 process CPU seconds on two photographs where the placer kept its planes and 7.70 seconds on the canopy that required a second placement pass. [Export verification](verification.json) · [New panel verification](new-panel-verification.json) · [Per-image CPU timings](timing.json)

Use the saved inputs from the [prior release and comparison](../roof-span-01/README.md). The 72 development photos are distinct from the 104 fixed, 20 alternate and four loading-dock/switchback views. Four shelter/gate views use a separate replay. Each case preserves screenshot, surface, count, camera, label and exported-pack provenance. [Compressed comparisons and changed exports](comparisons.zip) · [Ordinary build verification](verification.json)

```powershell
python -m benchmark.roof_height_eval --candidate benchmark.roof_height_consensus_trial:hybrid --out runs/hybrid --workers 6
python -m benchmark.roof_height_transfer --inputs ../framed-depth-transfer/runs/transfer-input --candidate benchmark.roof_height_consensus_trial:hybrid --out runs/transfer-hybrid
python -m benchmark.roof_height_eval --candidate benchmark.framed_geometry_order:production --out runs/production --workers 6
python -m benchmark.roof_height_transfer --inputs ../framed-depth-transfer/runs/transfer-input --candidate benchmark.framed_geometry_order:production --out runs/transfer-production
.\Invoke-WallRecovery.ps1 C:/code/minecraft/workspace/builds/shape-benchmark-26.3/captures/train-canopy-06-left.png -Out runs/cli-canopy
python -m benchmark.roof_span_verify --runs runs --winner hybrid --out benchmark/reviews/roof-height-01/verification.json
```
