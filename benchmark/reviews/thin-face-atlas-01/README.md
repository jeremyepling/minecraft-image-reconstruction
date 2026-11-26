# Matching thin visible horizontal faces to the texture atlas

Two passes with the atlas decision model raised relative visible-geometry F1 from **85.71% to 87.25%** on the fixed 112 screenshots. The exported packs gained 646 correct block positions and 36 extras while retaining all existing positions. On the 20 alternate views of known shapes the score rose from **81.94% to 86.47%**. Recommend merging the two-pass method from `experiment/thin-face-atlas` if the combined ordinary build verifies the gain. The experiment branch itself exposes the callable and leaves the current placement entry point intact.

| Cohort | Visible positions gained | Extras added | Of extras, hidden structure |
|---|---:|---:|---:|
| Fixed exteriors, 56 | 222 | 15 | 5 |
| Cloister validation, 12 | 23 | 5 | 1 |
| Twin Spires test, 12 | 27 | 0 | 0 |
| Earlier supported native, 12 | 132 | 4 | 0 |
| Previously reviewed native, 12 | 167 | 11 | 2 |
| Additional native confirmation, 8 | 75 | 1 | 0 |
| Separate 20 views | 321 | 15 | 0 |

Among the 646 correct gains, 642 actually showed a top or underside, 310 were missed entirely by the original four-pixel sampling grid, and 344 had fewer than two immediate horizontal neighbors in the previous reconstruction. Newly proposed blocks had 94.72% precision. Sixty-nine fixed views improved, 38 tied and five declined. Eighteen of the alternate views improved and two tied. The eight new hidden errors occurred on six captures in the fixed evaluation, including Shopfront and a recessed facade. Each misplaced cube counted as an error in full. [All 204 scored photographs](rank75x2/scores.json) · [Per-view decoded Minecraft packs](rank75x2/evidence.zip)

On the ten fixed canopy probes, blocks with the correct roof-to-post offsets rose from **194 to 284 of 459**. The eight alternate canopy views rose from **131 to 194 of 398**. All ten original opening widths remained correct. Six alternate openings stayed correct and two remained unresolved. The method blocked zero additional sampled clear sightlines. [Roof, post, ray and opening measurements](rank75x2/canopies.json)

## What the chooser observes

The geometry proposes horizontal locations up to three grid steps from a visible block on the same course. Each candidate must have at least twelve original image pixels inside a detected structure mask on the projected horizontal face. It also needs observed intensity variation, usable texels, texture correlation on two disjoint texel subsets and a stronger fit than shifted phases. Every accepted batch must keep an original visible pixel for previously visible blocks. Candidate pixels are checked again after jointly rendering the proposed blocks, so competing proposals cannot all claim the same pixels. A second pass runs only after the first pass changes the reconstruction and undergoes the same checks against the original photograph.

The matcher estimates a palette from independently photographed faces in the current build, evaluates orientations and texture footprints at one, two and four texels, and compares competing nearby heights and wall directions on the same pixel rays. A regularized logistic model uses these features, the native mask coverage and agreement across visible cube faces. It compares color, texture phase, material changes and neighboring evidence. Predictions receive the photo, estimated camera, predicted masks, existing placement and calibrated vanilla atlas. The source camera, world readbacks and visible-only labels enter training or scoring.

## How the rule was selected

A test of three fixed thresholds on the 72 development screenshots found 133 correct positions for 142 extras with a strong local texture rule, 276 for 260 with a moderate rule, and 142 for 267 when any mask orientation could support a strong match. Their candidate pool contained 1,289 missed visible positions among 29,615 distinct image-measured proposals. Counts in the following table come from decoded out-of-family training predictions. All views, palettes and designs of the withheld family were excluded from the corresponding weight fit.

| Rule on the same 72 training images | Correct gained | Extras added | Hidden added |
|---|---:|---:|---:|
| Geometry-only refit, one pass, 0.75 | 180 | 124 | 19 |
| Geometry and atlas, one pass, 0.55 | 394 | 151 | 10 |
| Geometry and atlas, one pass, 0.65 | 353 | 106 | 10 |
| Geometry and atlas, one pass, 0.75 | 315 | 62 | 5 |
| Geometry and atlas, one pass, 0.85 | 244 | 37 | 5 |
| Two passes, 0.55 | 495 | 180 | 13 |
| Two passes, 0.65 | 421 | 136 | 13 |
| Two passes, 0.75 | 376 | 78 | 8 |

The geometry-only ablation used the same candidates with measurable texture in the photo, a newly fitted model with all texture features zeroed, and no final texture-correlation cutoff. The single-pass atlas model both recovered more visible positions and reduced false placements at the same threshold. The two-pass 0.75 and 0.65 rules both scored 87.28% F1 in family-held-out training. The 0.55 rule gained 0.13 points and reduced F1 on two families. We froze 0.75 twice before its fixed-suite and alternate-view runs. The final model weights use the 72 training photographs only. The earlier conservative 0.85 rule had already been evaluated on the two historical suites at the time the additional training-only threshold comparisons were requested. [Development outcomes and per-family counts](development.json) · [Development packs, fold weights and model comparisons](development.zip)

The frozen single-pass 0.85 rule added 475 correct positions and 11 extras, four hidden, on the fixed 112. On the additional 20 it added 246 correct positions and nine extras, none hidden. Its roof positions reached 259 and 178 on the two canopy cohorts. The first-pass proposal pool includes 1,649 actual misses among 45,070 proposals on the fixed 112, and 573 among 8,667 on the alternate 20. These are scoring-side counts before the candidates compete for native pixels. Two passes can reach additional cells outside the initial pool. [Single-pass measurements](rank85/scores.json) · [Single-pass packs](rank85/evidence.zip)

## Replay

Restore the frozen [visible recovery inputs and baseline packs](../visible-recovery-01/README.md) and the [20-view fixture](../visible-recovery-extra-views-01/README.md). The baseline checkout defaults to the registered `combined-visible-recovery/runs/triple` run. Pass `--baseline` and `--extra` when the two restored baseline cohorts are elsewhere. The ordinary callable is `engine.thin_atlas_faces.recover_two_passes(image, parts, camera, prior, atlas)`. Use `--modes live` to run it directly or the following command to compare both frozen settings.

```powershell
python -m benchmark.thin_face_atlas_round --sets train validation test regression native old fresh extra --modes rank85 rank75x2 --workers 4 --out runs/strip-replay
python -m benchmark.thin_face_fit --source runs/strip-replay --weights runs/refitted.json --report runs/family-fit.json --folds runs/refitted-folds
python -m benchmark.thin_face_atlas_round --sets train --folds runs/refitted-folds --modes rank55 rank65 rank75 rank85 rank75x2 --out runs/strip-family-check
python -m unittest tests.test_thin_atlas_faces
```
