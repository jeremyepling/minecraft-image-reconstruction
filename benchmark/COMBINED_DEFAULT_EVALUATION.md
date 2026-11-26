# How much of a visible build reaches the datapack?

The subsequent [camera evaluation](TEXTURE_CAMERA.md) measures the default after adding floor atlas registration. The results below preserve the earlier baseline.

On the 56 fixed exterior screenshots, the Java 26.3 default at `7c2137c` placed **2,070 of 10,626 visible blocks** at the right coordinates, added **574 extras**, and exported **zero with the correct material ID**. The optional Texture matcher correctly located and named 154 face appearances in the counts report. The exporter uses smooth stone to display geometry. None of these 56 builds uses smooth stone for a visible target block. [Per-screenshot scores](reviews/combined-default-01/frames.json) · [Eight-building visual walkthrough](reviews/combined-default-01/comparison.jpg)

| Screenshots | Metric cameras | Correct coordinates / visible target blocks | Extra coordinates | Correctly located and named faces in the report / eligible faces | Correct coordinate + ID in the default datapack | Correct coordinate + ID in the projected-face trial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 56, eight buildings | 16/56 | 2,070 / 10,626 | 574 | 154 / 10,438 | 0 | 302 |
| Cloister and Twin Spires, all 24 validation and test views | 0/24 | 0 / 5,890 | 0 | 154 / 6,116 | 0 | 0 |
| Additional 12 views selected for camera support | 12/12 | 2,760 / 3,780 | 489 | 0 / 4,142 | 73 | 1,731 |
| **Total in this evaluation mix** | **28/92** | **4,830 / 20,296** | **1,063** | **308 / 20,696** | **73** | **2,033** |

The unit is one visible block or face in one screenshot. A second view of the same block counts as a second observation. The 12 camera-supported views came from a screen of 70 native test screenshots. Their coordinate recall is 73.0% within the selected 12 and 27.1% across all 70. A separate camera screen of the 96 Visible Builds pilot images returned zero metric cameras, including the 24 Cloister and Twin Spires views above. Those screens overlap the table. [Scorecard and fixture fingerprints](reviews/combined-default-01/scorecard.json)

## Where the pipeline loses blocks

**Camera scale determines whether the exporter produces anything.** On the fixed suite, 7,536 of the 8,556 missed visible coordinates were in the 40 views with no metric camera. On the 16 supported views, 2,070 of 3,090 visible coordinates were correct (67.0% recall). All 4,240 evaluated visible vertices on those views reprojected within two native pixels after coordinate registration. Geometry can still be wrong at that precision. Of 574 extra placements, 331 were one block from a visible target by Manhattan distance. A perfect per-frame choice among the three existing camera methods returned the same 16 cameras and the same 4,240 accurate corners. [Camera chooser by frame](reviews/combined-default-01/camera.json)

**Finding surfaces is easier than producing usable local material crops.** Across the fixed 56 and additional 24, the Surface Finder matched 769 of 896 eligible surfaces. Of 1,067 proposed masks, 326 had both block spacings, 53 had both grid offsets, and only 27 yielded a complete cell. On those 80 views, the material report made 325 named guesses. Of these, 321 met the corner-alignment tolerance and 308 had the exact ID. All 13 wrong IDs on aligned faces were on polished-andesite faces on Twin Spires. On the 12 camera-selected views all 36 proposed material crops went unpaired. These results explain why the earlier eight-view family check and the full 24-view family check give different coverage numbers. [Material scores and face decisions](reviews/combined-default-01/evidence.zip)

**Merely transferring the existing names would have little effect.** An answer-assisted join found that only 16 of the 308 correctly named blocks were also at a correct exported coordinate in the same view. Attaching those verified names could raise the total from 73 to 89. The 73 existing matches all came from real smooth-stone blocks on the camera-selected views.

## The most promising change

Start with an exported position, project its exposed faces back into the screenshot using the estimated camera, and let Texture identify the best visible face. We tested this direction without training. Each sampled face had to be on-screen, at least five pixels wide, frontmost in the candidate build at every rasterized pixel, and fully inside one predicted surface mask. The widest eligible face supplied the crop. Texture’s existing ambiguity flag kept smooth stone when the read was uncertain. The trial decoded and checked a second datapack for each of the 28 nonempty builds. The other 64 builds had no positions to label.

**The trial gained 1,960 exact coordinate-and-ID matches, lost zero and preserved every coordinate.** It accepted 2,099 readings. Of those, 1,960 were exact, 27 gave a wrong ID at a correct position, and 112 named a block at a pre-existing extra coordinate. On Courtyard masonry high-right it gained 61 exact blocks. On Portico plaster low-left it gained zero, gave two wrong IDs at correct positions and attached 11 names to extra positions. Most gains came from the camera-selected cohort. The combined 10.0% exact recovery is the score on these 92 views. Performance on an unseen family, resource pack, or lighting condition remains open. [Trial packs, per-face predictions and scoring](reviews/combined-default-01/evidence.zip)

**Test a small accept-or-abstain model for projected materials first.** Compare it with the frozen projected-face rule. Useful screenshot inputs are face width, agreement between different visible faces of the same block, distance to a mask edge, evidence for the supporting plane, texture score margin and whether several block IDs share the texture. Of 809 withheld readings on the fixed suite, 509 already had the correct top choice. The other 300 broke down into 123 wrong materials at correct positions and 177 extra positions. Label these separately in training. Grade exact coordinate + ID, losses of prior exact blocks, and named extras per building. Include a plain threshold and a shallow decision tree as baselines. Reserve whole camera-supported building families for final testing. The existing 72 pilot training images can supply supervised crop examples but the current exporter generates no placed-face training rows on them. [Confidence audit](reviews/combined-default-01/evidence.zip) · [Grouped evaluation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)

**In parallel, generate a camera candidate from the building itself.** Use observed block spacing and offset on a wall as a scale cue. Where there are two visible nonparallel planes, check that both agree when reprojected. Keep the existing pavement estimator as the baseline. First measure whether a perfect chooser can recover accurate scale on previously empty images. Only then train a selector with an abstain option. Grade the resulting visible placements and extras in addition to corner error. A gate over today's three camera candidates has zero observed headroom on the fixed 56.

## Which experiments to continue

| Work | Decision and measured reason |
| --- | --- |
| Projected material reads from actual placements | **Continue first.** The image-only export trial reached 2,033 exact blocks. Freeze this rule as the next comparison and check on new camera-supported buildings before integrating it. |
| Trained RGB seam detector after the current atlas | **Continue as a challenger.** Fill only still-missing offsets at measured spacings. On 92 views it raised exact local face IDs from 308 to 349, kept every prior correct face, added three wrong IDs on aligned faces and two unverified names. Gains came from five views across four buildings. Measure incremental exported-block value alongside projected-face matching. |
| Outline-edge seam fallback | **Keep as a cheap control.** It gained 16 exact face IDs on one Crossing view and added two wrong IDs. Thirteen gains overlap the trained detector’s. Even perfect per-frame selection between them adds only three beyond the trained fallback. |
| Crop refit and count acceptance | **Secondary experiment.** Current sizes were exactly right on 194/896 surfaces. A perfect chooser across compatible count candidates could reach 210. The outline refit alone supplied 11 complementary exact sizes but would sacrifice 17 in a wholesale swap. Among 63 new counts accepted by the shipped fallback, 40 were correct, 18 wrong, and five on unmatched proposals. Use the refit as an alternative in a count-or-abstain test. |
| Atlas phase prototype, broad atlas spacing search, forced grid ranker | **Close as standalone lines and keep the fixtures.** Restricted atlas registration at measured spacing has shipped. Atlas confirmation supplied zero additional exact sizes in the matched candidate pool. The earlier forced grid ranker added two exact sizes and nine wrong sizes on its regression set, with zero exported placements changed. |
| Atlas, exposure, Texture and shadow matcher selector | **Defer on these crops.** A perfect chooser among the four top choices raised 308 to only 311 across the 80 views with comparable crops. Perfect reranking of Texture’s top five gave 316. Keep the other matchers as lighting and material controls. |
| Uncertain-camera local placement, unguarded neighbor scale or seam ownership, fixed-threshold partial cells | **Retire the broad variants.** Earlier uncertain-camera placement added 1,384 correct positions and 14,458 extras. Neighbor-scale transfer added eight correct positions and 46 extras in its original comparison. The guarded ownership rule and camera-supported placer already shipped. Keep partial-face work conditional on an independently supported block and accurate corners. |
| Within-wall patch, unrestricted plane additions, extra Surface Finder passes | **Park for this objective.** The patch added two correct sizes and one wrong in the fixed set and gained zero on the 24 additional views. Surface experiments also produced regressions in visible walls or openings. Reopen them on specific downstream failures. [Within-wall test](reviews/within-wall-grid-01/README.md) · [Surface tests](GUARD_ZOOM.md) |

These are research priorities. The working branches and fixtures remain available. Only the evaluation and its replay tools landed from this review.

For the next seam trial, start with `experiment/block-seam-learned` and apply its frozen detector only after the default atlas leaves an offset open. Use `experiment/block-seam-boundaries` as its comparison. Preserve the scorer and previous crop records on `experiment/block-counter-perfect-chooser` and `experiment/atlas-phase-materials`. The retire recommendations cover the standalone strategies on `experiment/block-counter-grid-ranker`, `experiment/block-seam-template`, `experiment/neighbor-surface-scale`, `experiment/partial-visible-cells`, and the unguarded variants on `experiment/local-visible-placement` and `experiment/seam-run-ownership`. The measured-spacing atlas and guarded placement implementations on `main` remain the comparison baselines.

## How the evaluation was run

All inputs were saved Java 26.3 screenshots. The bulk replay verified hashes and reused the frozen image-derived surface predictions, reran the current counting, matching and placement code, then wrote and decoded the Minecraft structures. For the additional 24, the current photo-edge pass ran on the saved preceding surfaces. Recorded camera, blocks and face corners were used for scoring, including aligning the arbitrary local compass and integer origin. Missed visible cells stay in the denominator. Every extra coordinate counts against precision. The joint metric requires an exact Minecraft block ID and coordinate; orientation and other block-state properties would require a separate metric. The ordinary wrapper also ran fresh image inference on Courtyard masonry high-right. All six local face quads and IDs and all 210 exported coordinates and states matched the replay. Of those coordinates, 194 were correct for that screenshot.

The [per-frame table](reviews/combined-default-01/frames.json) includes all 92 RGB and surface hashes. The [evidence archive](reviews/combined-default-01/evidence.zip) includes decoded-coordinate records, pack files, model and fixture hashes, face decisions and perfect-chooser diagnostics. The [replay scripts](reviews/combined-default-01/replay.zip) use the existing calibrated 26.3 atlas, the [Visible Builds v1](../training/DATASET.md) release extracted into `runs/visible-builds-v1/data/{split}`, and this PC’s saved `workspace/builds` native fixtures. The material grader and earlier seam models are from the sibling `atlas-phase-materials` worktree at `1014df5`. Use the repository root and a fresh directory under `runs`. The geometry runner consumes the fixed and additional-family archives plus the native fixtures.

```powershell
Expand-Archive benchmark/reviews/combined-default-01/replay.zip -DestinationPath runs/replay-combined
$env:OPENBLAS_NUM_THREADS = '1'
python runs/replay-combined/geometry/evaluate.py
python runs/replay-combined/geometry/camera_screen.py
python runs/replay-combined/material/replay_material.py
python runs/replay-combined/material/native12.py
python runs/replay-combined/material/join.py
python runs/replay-combined/geometry/projected-material/test_projected.py
python runs/replay-combined/geometry/projected-material/confidence_audit.py
python runs/replay-combined/material/placed_reference_crops.py
python runs/replay-combined/material/placed_reference_crops.py --smooth-only
python runs/replay-combined/decisions/count_chooser.py
python runs/replay-combined/decisions/learned_replay.py
python runs/replay-combined/decisions/boundary_replay.py
python runs/replay-combined/decisions/native_seams.py
python runs/replay-combined/decisions/material_chooser.py
python -m benchmark.camera_chooser_ceiling --out runs/replay-combined/camera.json
python -m benchmark.combined_default_sheet --geometry runs/replay-combined/geometry --out runs/replay-combined/sheet
```
