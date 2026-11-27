# Joint camera and wall-grid refinement

Keep `experiment/camera-grid-joint` experimental. On 288 frozen Java 26.3 screenshots the independent source checks accepted zero additional cameras. Two diagnostic exports improved strict F1 under whole-building translation, but both had zero camera-registered correct blocks. Production code was unchanged.

## Visible placement

The incumbent uses ordinary placement first, the repeat fallback next and the image-only expanded search last. This experiment starts at its 70 remaining camera abstentions. All 288 source-policy exports preserve the incumbent pack hash. The fixed 104 and Decorative 40 results are carried forward from verified packs. The historical 112-view score was not replayed.

| Cohort | Strict F1 | Facade F1 | Strict +ID | Facade +ID | Strict correct / missed / extra | Camera failures |
|---|---:|---:|---:|---:|---|---:|
| Fixed 104/104 | 91.27% | 93.42% | 84.68% | 88.51% | 20,773 / 2,929 / 1,046 | 0 |
| Visible Builds v3 144/144 | 21.84% | 22.75% | 14.13% | 14.79% | 28,086 / 186,064 / 14,949 | 70 |
| Decorative Builds v1 40/40 | 63.46% | 65.23% | 50.51% | 52.56% | 9,240 / 7,686 / 2,957 | 0 |

On v3 the source policy scored 26.58% strict on train, 22.77% on validation and 5.58% on test. There were 49, 5 and 16 camera failures respectively, with 148,107 visible misses across the failures. [Full split geometry, ID scores, counts and camera-supported scores](summary.json)

## Source evidence

The predictor verifies each residual screenshot hash, source field provenance, checkpoint revisions, dimensions, intrinsics and resize metadata. It adds the cached Z shift to MoGe points in camera coordinates and samples source-visible part masks in the 448 × 280 field. It tries two rotations: the MoGe rotation and a rotation retaining its horizontal direction after projection onto the plane perpendicular to GeoCalib gravity, using GeoCalib intrinsics.

The most populated wall with verified native texture quads supplies the fit. Touching or overlapping patch bounding boxes are deduplicated before the wall split. Half, single and double source scale seeds initialize three alternating steps. Discrete levels within one block of the seed compete for the fit wall, and a bounded least-squares solve updates eye translation and model-to-block scale. Other walls get integer levels from their observed point-field depth before pixel reprojection. The estimator requires 90th-percentile corner error at most 3 native pixels on both fit and held-out walls, at least two held-out quads, and a ground contact within 0.16 block vertically. Ground contacts require an observed wall bottom beside model-ground pixels with less than 5% depth discontinuity. The ground constraint supplies camera height relative to the visible structure. The gate also abstains on near-tied accepted proposals. The verifier can rank up to six hypotheses. The withheld sprites are disjoint from the fit wall but share the source sprite detector and its rectification assumptions.

The 70 remaining v3 images produced 171 numerical fits on 29 views: 17 train, four validation and eight test. The other 41 produced no fit. The procedure only uses photographed sprite, wall and ground evidence. It has no independent opening detector. No thresholds were adjusted after the development run. Compact predictions, source and field hashes, actual consumer receipts, exports and scores are retained [per frame](cases.json.gz).

## Where the numerical improvement came from

The bounded export test picked the best source-ranked raw-camera-available frame independently on development and held-out splits. It evaluated two different joint proposals, the first proposal's P with ordinary wall levels, and the original raw MoGe P with its first proposal's revised wall levels. The grid-only arm keeps the original raw P and shifts the proposed X/Z levels into its integer origin. The shifts were (-50, 0, -77) for Viaduct and (-40, 0, -47) for Castle. The ground datum and fractional camera phase are preserved. The replay injected overrides into every applicable `_place` pass, preserved the existing roof rerun logic, checked the actual consumer's surface axes and mask overlap (IoU ≥ 0.90), and verified the requested plane offsets in the returned status. All four Viaduct and all six Castle overrides reached placement.

| Frame | Raw MoGe | Camera only | Grid only | First joint | Second joint | Second joint +ID | Second joint median native error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Viaduct 02, riverstone rear oblique (train) | 31.42% | 30.18% | 31.30% | 30.80% | 36.83% | 0.00% | 209 px |
| Castle 03, limestone rear oblique (test) | 17.63% | 19.30% | 15.38% | 16.26% | 20.85% | 15.41% | 405 px |

Both frames lacked a supported wall-base contact. First-choice native held-wall p90 error was 5.87 px on Viaduct and 17.97 px on Castle. The second-ranked proposals earned the best translated strict F1 of the raw and joint packs on both frames. Each yielded zero camera-registered TP. The Viaduct's second proposal assigned no correct position-and-ID pairs even after translation. [Viaduct source overlay](train-viaduct-02-riverstone-rear-oblique.jpg) · [Castle source overlay](test-castle-03-limestone-rear-oblique.jpg)

The reference-only chooser picks maximum per-frame strict position F1, preserving the incumbent on ties, and reports IDs for that same selection. On 70 residuals its raw choices include an available verified GeoCalib-camera or MoGe export. Its refined options add the four diagnostic joint exports. Ablations are scored independently.

| v3 chooser options | Strict F1 | Facade F1 | Strict +ID | Facade +ID | Strict correct / missed / extra |
|---|---:|---:|---:|---:|---|
| Incumbent | 21.84% | 22.75% | 14.13% | 14.79% | 28,086 / 186,064 / 14,949 |
| Incumbent + available raw packs | 23.18% | 24.10% | 14.64% | 15.29% | 30,326 / 183,824 / 17,184 |
| Above + four coupled exports | 23.32% | 24.25% | 14.67% | 15.32% | 30,604 / 183,546 / 17,672 |

The four joint exports increased the empirical oracle's strict score by 0.14 percentage points over the raw chooser, adding 278 correct and 488 extra positions. This bounds only the options exported. The source verifier realized none of the oracle gain.

## Reproduction

Base `d9eb377`, frozen [control fingerprints and output manifest](manifest.json), frame-list SHA-256 `ff36ec40da6890af45e66f0581965ce2b5847ef0840fb4182635d447669c3033`. Use registered sibling worktrees `camera-model-controls`, `camera-geocalib`, `camera-moge-surface-first` and `full-expanded-controls` with their published run inputs. The shared ordinary entry point is exercised through the frozen source-field replay. All predictions and eight new builds are private under `runs/camera-grid-joint-01`.

```powershell
python -m benchmark.camera_grid_joint train
python -m benchmark.camera_grid_joint held
python -m benchmark.camera_grid_joint_replay
python -m benchmark.camera_grid_joint_report
python -m unittest discover -s tests -p test_camera_grid_joint.py
```
