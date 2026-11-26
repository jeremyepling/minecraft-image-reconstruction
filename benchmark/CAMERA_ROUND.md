# Single-image exterior reconstruction · Camera round 01

**The recommended Stage 1 default is the subpixel seam fit (`seam-fit`).** On the held-out courtyard and portico it recovered a metric camera on all eight oblique images. The two close-front images provided insufficient evidence for a full metric camera. [Full-resolution comparison sheet](reviews/camera-01/comparison.png) · [Run manifest](reviews/camera-01/manifest.json) · [Per-image scores](reviews/camera-01/holdout-scores.json)

| Camera | Metric images | Corners within 2 px, layout average | Mean corner error | Mean span error |
| --- | ---: | ---: | ---: | ---: |
| Published pavement fit | 6/10 | 66.7% | 0.66 px | 0.028 blocks |
| Subpixel seam fit | 8/10 | 83.3% | 0.35 px | 0.006 blocks |
| Connected square tiles | 8/10 | 83.3% | 0.68 px | 0.023 blocks |

The seam fit projected 2,292 of 2,362 labeled corners within two pixels and measured 636 of 660 visible spans. Pixel and span errors in the table are conditional on an estimate. On the six images where all three returned a camera, mean corner errors were 0.66, 0.33 and 0.67 px respectively. The development tiebreak favored the seam fit on span error before holdout. [Frozen selection](reviews/camera-01/selection.json)

**Failure to inspect.** In the sheet's low-left courtyard view, the seam-fit camera averaged 0.34 px of error with known surfaces. Its image-only proposal spanned 12 blocks across a 16-block rear wall and selected a parallel plane one block away. Across holdout it returned 14 of 18 eligible vertical planes and selected the reference plane's exact offset twice. The other two methods each selected the exact offset four times. The camera result supports measuring a supplied surface. Choosing which visible surface to measure remains a separate source of error.

Run the comparison from the repository root with Python 3.11+ and the packages in `requirements.txt`.

```console
python -m benchmark.camera_round
```

The command loads the original framebuffers, makes a new timestamped directory in `runs/camera`, saves the development selection, then scores the holdout. Its manifest links the sheet, per-image metrics, camera predictions and visual feedback. To recapture the structures in a fresh Java 26.3 world, run `python -m benchmark.camera_fixture --out runs/fresh-fixture` with Java 25 and the installed game assets, then supply that directory using `--fixture`. The committed framebuffers give byte-identical comparison inputs. [Fixture index](data/camera-round-01/index.json)

## Fixture and input boundary

Four physical layouts use two palettes each. The development wing and shopfront supply flat walls, exterior corners, an awning and one-, two- and three-block recesses. The holdout courtyard and portico supply projecting wings, exposed returns, a roof and open bays between pillars. Each version has a low-left and high-right angle. One layout per split also has two close facade-only controls, for ten frames per split. These are four independent layouts. Other angles and alternate palettes of a layout are repeated observations.

Each camera method receives a single RGB image and the version. The shared camera result records a local projection matrix, homogeneous axis directions, scale evidence and candidate grid extents on vertical planes. A unit ground tile is the scale assumption. Without a metric fit the shared direction finder can still return supported vanishing points. Ground-truth files contain the native projection, camera pose, block-state readbacks (including air), face direction, four projected corners and visible fraction. Face visibility accounts for occlusion and clipping. Only the evaluator reads these files to register a result and measure it.

## What the scores measure

- **Grid registration.** The evaluator resolves the arbitrary compass axes and rounds the difference of ground coordinates at one fixed image ray to an integer translation. The ray passes through 50% of image width and 72% of height. The scorer never fits a translation to the building. Fractional-block phase error survives registration.
- **Visible intersections.** Project each distinct, visible labeled lattice corner at its known surface coordinate. The primary score is the fraction within two native pixels, averaged by view within layout and then equally across layouts. An abstention misses every corner in that frame. The report also gives pooled counts and mean Euclidean pixel error conditional on a projection.
- **Visible spans.** Take uninterrupted runs of at least three unit edges on faces at least 98% visible. Invert the predicted camera at the two recorded endpoints on the known plane. Score the absolute error in the continuous span in blocks. This measures spacing without requiring an edge or face detector. The denominator includes unmeasured spans.
- **Image-only planes.** Separately, ground contacts of the image silhouette vote for a vertical grid plane on each horizontal axis. Its proposed width and height are compared with the visible extent of the most exposed labeled plane of that orientation. The report counts returned proposals, exact plane offsets and extent error. A rectangular chart can cross open air. It does not assert that a block exists there.
- **Selection.** Run the published voxel camera (`pavement`), the two-family subpixel seam fit (`seam-fit`) and the connected square-tile fit (`tile-lines`). Use the development corner score, then metric frame count, then conditional visible-span error. Freeze the settings and selection before processing holdout predictions. Every candidate uses the same inputs, compass registration and scoring code.

## Build and review

```console
python reconstruct.py build IMAGE --through camera --version 26.3 --out runs/camera-build
```

`camera.json` stores the actual method, projective lattice, support decision and proposed planes. `grid.png` draws the automatic plane proposals. The pre-existing full datapack build uses its own facade grid and records the geometry, materials and export methods in `build-manifest.json`. Camera selection is independent of block placement during this round.

The comparison sheet uses source-image coordinates. Its enlarged overlay transfers the predicted camera to known visible surface coordinates. The automatic-plane panels use only image-derived charts. To record a review, use an image ID from the fixture and a native-pixel box from `regions.json`, or specify another rectangle in the same image.

```console
python -m benchmark.camera_feedback RUN --image IMAGE_ID --region X0 Y0 X1 Y1 --preferred METHOD --note "Reviewer's observation"
```

Feedback is stored in the run's `feedback.jsonl`. The camera predictors have no feedback input. The run retains both the measured winner and any reviewer preference.
