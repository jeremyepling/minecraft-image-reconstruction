# Recovering camera scale from Minecraft textures

The Java 26.3 default now measures the existing ground seams first, then tries to match the visible floor to a calibrated vanilla sprite. On the fixed Visible Builds v1 release this raised metric camera coverage from **16 to 144 of 152** screenshots. Every projected visible corner on those 144 frames landed within two source pixels. The 128 newly recovered views all have light gray concrete ground. [Floor experiments and saved camera matrices](reviews/floor-atlas-camera-01/README.md)

## What changed in the actual builds

| Locked cohort | Cameras before → after | Correct visible coordinates before → after | Extra coordinates before → after | Visible target |
| --- | ---: | ---: | ---: | ---: |
| Eight fixed regression buildings, 56 views | 16 → 48 | 2,070 → 5,967 | 574 → 3,508 | 10,626 |
| Cloister validation, 12 views | 0 → 12 | 0 → 1,799 | 0 → 713 | 2,476 |
| Twin Spires test, 12 views | 0 → 12 | 0 → 2,218 | 0 → 1,064 | 3,414 |

The 72 training views supplied camera development scores. The table counts blocks in decoded datapacks for the other 80 views. Each camera received one RGB screenshot. The comparison held visible-surface masks and counting rules fixed. An independent replay used the recorded camera only for scoring and for a separate ceiling measurement. Compass direction and integer horizontal origin are arbitrary for a single image. The scorer registers those against one fixed ground ray. All visibility targets refer to the supplied view. [Predictions, input hashes, decoded builds and replay instructions](reviews/texture-camera-01/scoring.zip)

On the fixed suite, coordinate recall rose from 19.5% to 56.2% and precision fell from 78.3% to 63.0%. Each of the original 16 supported builds retained exactly the same coordinates. The new views exposed costly placement errors, especially behind the visible front of a beam or ledge. The exporter displays geometry in smooth stone, which gave zero exact material-and-coordinate matches on these 80 screenshots. Material names used to estimate camera position do not assign exported block IDs.

Supplying Cloister's recorded cameras to the same placement stage gave 1,801 correct blocks and 714 extras. The estimated cameras gave 1,799 and 713. A fresh ordinary single-image run on Cloister 1 cool high-right reproduced all 169 decoded cells from the saved-mask replay, including its 150 correct and 19 extra cells. [Three buildings with camera and placement overlays](reviews/texture-camera-01/comparison.jpg)

## What the three rounds established

| Round | Floor camera on 72 training views | Wall camera on the same inputs |
| --- | --- | --- |
| 1 | Stretching contrast and refitting dark seams recovered 0 cameras. | Visible wall lines and possible ground contacts issued 20 cameras. None had a full-frame 90th-percentile corner error within 2 px. |
| 2 | Measuring spacing between texture pixels issued 48 cameras. Only 2,548 of 15,548 projected corners were within 2 px. | Possible pairs of different atlas tiles issued 8 cameras. A scoring audit found zero true material changes in the strongest pair on those eight images. Shading and partial tiles could mimic a new material. |
| 3 | Matching the repeating 16 × 16 pattern to the real sprite established the start of each block. All 24,644 scored corners on all 72 training images aligned. | Fitting multiple cells and refining the camera from building edges gave 20 accurate full-frame cameras among 36 issued. Requiring two ground contacts left 9 accurate among 10 issued on training. |

The floor fallback measures the perspective spacing of individual texels in two directions. It folds observations into a 16 × 16 tile, then checks the sprite, orientation and offset independently in the left and right parts of the image. If the atlas is missing, damaged, incompatible, or the measurements disagree, the camera abstains. The ordinary camera and placement commands share this selection. The saved seam-only method remains available in the benchmark. [Frozen floor results](reviews/floor-atlas-camera-01/scores.zip) · [Wall rounds, settings and per-frame audit](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/d7d9565/benchmark/reviews/wall-camera-texture-01/README.md)

## When two different block textures meet

**Native screenshots, known face boundaries.** We sampled up to three adjacent pairs per frame, plane orientation and material relation using deterministic hashes. Each face needed four visible corners, at least 98% visible area and 32 pixels. On 36 different-material Cloister wall pairs, the ordinary shape matcher classified all 36 as changes, named both block IDs on 25 and preferred two full tiles to two half-size tiles sharing the same edge on 31. For repeating Cloister wall pairs it correctly classified 34 of 36. On fixed-suite horizontal material changes it classified 13 of 20 and named both IDs on zero. These horizontal faces include roofs. The Twin Spires test frames had zero qualifying different-material pairs under this filter. This experiment supplies the true face crops and measures the recognition stage. [Pair-level scores](reviews/texture-camera-01/material-evidence.zip)

**Native screenshots, RGB-only wall estimates.** The third-round wall camera correctly identified some genuine changes, including cobblestone beside polished diorite and terracotta beside mud bricks. It supplied zero new views beyond the floor camera on the 152-frame release. Across a separate 70 native views, the floor method preserved the 12 original cameras and abstained on 58. The broader wall detector supplied 15 additional canopy cameras. Four passed the full-frame 90th-percentile threshold of 2 px. The other eleven still had 90th-percentile error below 3 px. Its stricter guard selected three and missed all four of the most accurate cameras. [Independent native screen](reviews/texture-camera-01/scoring.zip)

**Mixed-material ground, RGB-only camera.** We repainted the visible lower ground in saved training screenshots using the recorded camera for fixture generation. Every pair of edge-adjacent tiles has different vanilla sprites. The original building pixels and scored building vertices are preserved. The camera receives the resulting RGB alone. Nine views across six training families were each paired with a high-contrast and a similar-gray palette for 18 controls, including a fresh low-left angle group. These are generated controls outside the frozen dataset. [Predicted obsidian-to-bricks boundary](reviews/texture-camera-01/material-joint.jpg)

One edge-map setting recovered a camera on one of the first six controls. Trying multiple edge widths also admitted a camera whose 90th-percentile error exceeded 100 px. Requiring two estimates to agree about projected points above the ground recovered four additional high-contrast cameras across the final 18 controls. All 994 visible building corners in these four landed within 2 px. Of 224 measured neighboring tile pairs used in the four estimates, 218 had both sprite names correct. The stricter trial supplied zero cameras on the nine similar-gray floors and zero additional cameras on the 152 release or 70 native frames. A separate high-contrast control triggered an erroneous estimate in the original seam camera, with 488 px 90th-percentile error. It remains a regression target for seam verification. [Controls, adjacent material identities and per-image errors](reviews/texture-camera-01/material-evidence.zip)

## Decisions and next tests

- **Promote the atlas floor camera** from `experiment/camera-floor-texture`. The two held-out families gained accurate visible placements, and the independent native set retained its earlier predictions. Camera deployment requires a prepared Java 26.3 atlas for the fallback.
- **Keep `experiment/camera-wall-atlas` experimental.** Continue on open canopies and near-front walls. Use independent joints, visible floor contact and the effect on decoded placements to decide when a wall can anchor a whole camera. A wall can still provide a local block count when absolute depth is ambiguous.
- **Keep `experiment/camera-ground-material-joints` experimental.** It demonstrates a camera can use changes of material without requiring any neighboring identical blocks. Validate on native mixed-material floors, including gray-on-gray changes, before considering promotion.
- **Focus downstream on visible-plane assignment.** Compare candidate plane depths and block positions by projecting their visible faces back into the image. Score precision and recall per building; use the same masks and camera throughout. Galleries plaster high-right now has an accurate camera and still exports 259 extra coordinates. A surface-confidence rule can be measured against exact visible coordinate labels there.

## Replay

Download and extract the fixed [Visible Builds v1 shards](../training/DATASET.md). Prepare the calibrated vanilla Java 26.3 atlas in the normal search path. For a single image, optionally supply `--atlas DIR` to the camera stage or `-Atlas DIR` to the placement wrapper. The ordinary camera records its source in `camera.json`; placement records it in `build-manifest.json`.

```powershell
python reconstruct.py build screenshot.png --through camera --version 26.3 --out runs/camera
.\Invoke-WallRecovery.ps1 screenshot.png -Through placement -Out runs/placement
python -m benchmark.texture_camera_eval --split all --out runs/previous-seams
python -m benchmark.texture_camera_eval --split all --baseline runs/previous-seams --candidate-root . --candidate engine.camera_atlas_floor --out runs/registered-floor
python -m benchmark.texture_camera_placement --camera runs/registered-floor --out runs/registered-packs
python -m benchmark.atlas_transition_probe --data runs/visible-builds-v1/data --split validation --out runs/pairs.json
```

To replay the mixed-floor experiment, use the main checkout for the benchmark runner and the `experiment/camera-ground-material-joints` checkout for its predictor. Run both cohorts with `--angle high-right` and the check cohort with `--angle low-left`.

```powershell
python -m benchmark.mixed_floor_controls --data runs/visible-builds-v1/data --candidate-root ../minecraft-image-reconstruction-worktrees/camera-ground-material-joints --candidate engine.camera_material_joints --cohort check --angle low-left --out runs/joint-check
```
