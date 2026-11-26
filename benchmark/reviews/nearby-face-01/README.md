# Nearby depth and face-direction decisions

Replay the standalone experiment using its [frozen experiment checkout](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/8feaed2). See the [combined production results](../nearby-thin-01/README.md) for the promoted configuration.

The individual-face model at `5452523` improved relative visible-block F1 from **85.71% to 85.98%** on 112 fixed screenshots and from **81.94% to 82.52%** on 20 alternate views of known structures. It predicts from one screenshot, its estimated camera, detected surface masks, the calibrated Java 26.3 atlas and the already placed local cells. All evaluation counts come from decoded Minecraft packs with one integer translation allowed per whole build.

| Method | Fixed correct / extras | Fixed F1 | Alternate-view correct / extras | Alternate-view F1 |
|---|---:|---:|---:|---:|
| Previous production, `cebbffa` | 19,408 / 1,794 | 85.71% | 3,173 / 474 | 81.94% |
| Individual face decisions, `5452523` | 19,538 / 1,824 | 85.98% | 3,210 / 472 | 82.52% |
| Add neighboring chart agreement, `6fbe906` | 19,555 / 1,811 | 86.05% | 3,205 / 469 | 82.48% |

The individual scorer gained 137 correct fixed-suite positions and lost seven, with 81 extras added and 51 removed. Six new erroneous placements coincide with hidden structure. Across the 20 alternate views it gained 39 correct and lost two, with 11 extras added and 13 removed. Forty-two fixed-suite rows improved, 59 tied and 11 declined. The six-layout native subset declined by about 0.01 F1 points. The other five subsets improved. The 112 and 20 are separate evaluation suites. The alternate views introduce new photographs of previously seen structures. [Per-cohort scores and candidate counts](individual-scores.json) · [All 204 packs, coordinates and paired frame metrics](individual-scores.zip)

## Decisions and development

Each masked image sample currently owned by a cube supplies a ray. Intersect the ray with nearby planes of each mask-supported axis at offsets from minus two through plus two blocks, then compute the full XYZ intersection. A shift can change the coordinate along the plane as well as its depth. Each alternative needs at least three sample witnesses and at least 6% of the source cube's visible sample count. Reject coordinates outside the local neighborhood and keep the current cell as the fallback.

Two shallow boosted classifiers estimate confidence in the photographed placements of the source and destination cells. Their inputs measure support from face normals, projected native appearance at the unshifted and quarter-tile phases, atlas correlations, the current frontmost faces, neighboring placements and reprojected area. The 36-tree classifiers were fitted only on 72 training screenshots using local coordinates labeled by visible-block positions. Complete building families stayed together in three development folds. We compared tree depths two and three and candidate admission/drop thresholds. A nominated replacement must retain mask-supported pixels after competing placements are rasterized together. The native visibility guard protects the last photographed pixel of retained baseline cells. Missing or incompatible weights, an unavailable atlas, or absence of suitable camera or mask evidence returns the supplied placement.

A first photo-only ranker proposed 108 new correct cells for 39 new extras on out-of-family training predictions before joint visibility checks. Adding calibrated atlas features made confident replacement of bad source cells possible. On actual held-family placement decisions with joint visibility checks, the selected individual rule gained 106 correct, lost three, added 33 extras and removed 37. This is 76.3% precision on individual additions. The six family results are retained in the threshold table. After refitting on all 72 training photographs, we froze `5452523` before scoring the evaluation cohorts. Five focused tests cover weights, atlas, camera, projected alternatives, occlusion, and abstention; three raw predictor replays matched the benchmark exactly. [Independent-family comparisons](held-visible.json) · [Training alternatives](training.json)

For the second model we added the mean and strongest measurements from nearby, independently observed source cells with the same face direction and depth shift. It gained 94 correct and lost two in family-held-out development, with 33 extras added and 31 removed. The directions of the fixed-suite and alternate-view differences relative to the individual model disagree. Recommend merging the first scorer through the combined pipeline if its canopy and independent checks pass. Keep the neighborhood context model experimental. [Context development](context-held-visible.json) · [Context per-cohort results](context-scores.json) · [Context decoded packs](context-scores.zip)

## What a perfect chooser could recover

On the 112 fixed rows the individual scorer generated 78,409 distinct novel per-frame candidates, including 2,530 missed visible targets. Of those true targets, 661 have generated witnesses that primarily change the source face axis and 2,308 have a different depth level. Those counts overlap. Of the final 137 newly correct coordinates, 104 had a proposal that changed the dominant source face axis. Retaining every true proposal and no false proposals would reach 91.76% F1 with the existing extras held fixed. This assumes ideal selection before candidate visibility competition.

Against the same unmodified baseline, the thin-face atlas and nearby decision pools together contain 3,124 distinct missed targets, including 1,055 shared by both. The estimated upper bound is 93.08% F1 with perfect additions and 96.67% if every old extra also disappears. Among the 27,954 shared proposals that neither standalone output selected, 715 were true. New iterations of a placer can generate different candidates. [Per-frame overlap and optimistic bounds](joint-candidate-headroom.json)

## Shared-proposal consensus

We tested whether agreement between lower-confidence depth and atlas predictions could accept more of the 715 shared true misses. The depth and atlas confidence cutoffs were selected using predictions from models fitted without each photographed training family. Each candidate required horizontal-face witnesses from both methods and fresh native-pixel support after all proposed additions competed for visibility. The comparison build combined the independent individual-depth placements and single-pass rank85 atlas additions generated from the same source build.

The best of 16 threshold pairs added two correct positions on the 72 family-held-out training images, both on Watchpost. On the fixed 112 it added three correct and zero extras. On the separate 20 it added zero correct and one extra. Keep this gate experimental: the marginal gains were small, concentrated in one training family, and did not repeat on the alternate views. This diagnostic used the single-pass atlas output. It did not measure a new production composite. [Family selection](consensus-train.json) · [Per-frame paired check](consensus-evaluation.json)

## Replay

Start from `cebbffa` frozen upstream inputs, the previous combined release packs in `combined-visible-recovery/runs/triple` and the separate 20-view fixture. The benchmark accepts the standard training, validation, test, regression, native, old, fresh and extra split names. The commands below replay the first model from its frozen weights.

```powershell
python -m benchmark.nearby_face_trial features --sets train validation test regression native old fresh extra
python -m benchmark.nearby_face_trial art --sets train validation test regression native old fresh extra
python -m benchmark.nearby_face_trial evaluate --sets train validation test regression native old fresh extra
python -m benchmark.nearby_face_report
python -m unittest tests.test_nearby_face_decision
```

In a separate development checkout, refit the individual model with `python -m benchmark.nearby_face_fit --atlas`, followed by `python -m benchmark.nearby_face_crosscheck`. The optional context round adds `benchmark.nearby_face_trial context`, `benchmark.nearby_face_fit --context`, `benchmark.nearby_face_crosscheck --context` and `benchmark.nearby_face_trial evaluate-context`. The raw-build API is `engine.nearby_face_decision.recover(image, parts, camera, prior, atlas_path=...)`. This branch does not modify the ordinary entry point.
