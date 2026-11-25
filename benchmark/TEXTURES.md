# Texture and cell-membership benchmark

The benchmark compares known front-face corners, a seeded seam fit, and the production image-only wall finder on the same Minecraft screenshots. Training exemplars come from the version's independently placed calibration world. All south-facing placeable labels in that library compete on every crop. The nine materials in the test wall supply the expected answers.

## Reproduce

With a completed shape fixture and its two condition runs, execute from the project root:

```console
python -m benchmark.textures select --root RUN --library CALIBRATION/face_textures/softened-library.npz --atlas ATLAS_DIRECTORY --out REPORT_DIRECTORY
python -m benchmark.textures test --root RUN --library CALIBRATION/face_textures/softened-library.npz --atlas ATLAS_DIRECTORY --out REPORT_DIRECTORY
```

Optional `--conditions normal small warm-shader` controls the datasets. The default uses all three. Both commands require the same condition set, library digest, atlas digest and fixture layout digest. The test command reads the selected method from the development report.

## What the comparisons measure

- **Known corners** use the actual Minecraft camera, block position, and 25 occlusion rays across each face. Cropped or partially hidden blocks are excluded from the denominator. This isolates texture errors.
- **Seeded grid** starts with the wall outline, 6% pitch errors and a perturbed origin. The seam fitter determines the individual cell boundaries from the image.
- **Image only** runs the production automatic wall and grid finder on the complete screenshot. Its proposed quads enter evaluation before they receive a material label. Both fitted modes pair a proposed quad with at most one expected face, using highest intersection-over-union first and requiring at least 0.5. A missed face counts as an end-to-end identification failure.
- **Membership** includes mean IoU and crop purity (intersection area divided by predicted area). A separate count tracks proposed cells occupying known apertures. Grouped counts compare acquisition conditions, measured Minecraft sky-light levels, screen-space width, shade roof, view angle, and immediate neighbors. Neighbor groups overlap because a cell can touch an identical material, a different material and an opening. Per-pair counts require both labels at their correct positions; an unlike pair assigned the same class is counted separately.
- **Algorithm choice** starts with the production softened-template matcher. One challenger allows candidate brightness to vary by a bounded factor with a log-gain penalty. A second also tries three quarter-turn rotations with a rotation penalty. Neither receives the expected class, an angle, a crop source, or a lighting-condition label. Selection uses average accuracy over materials and conditions on known-corner development crops. The held-out report compares the selected challenger with the production baseline.
- **Evidence** comprises distinct physical development and held-out walls, each seen from several angles in three capture conditions. Views of a single cell are repeated measurements. Surface texture is the target; distinct IDs with identical visible textures cannot be distinguished by these measurements.

The JSON reports contain a sparse confusion matrix mapping actual textures to every predicted texture (including labels outside the target wall palette). `<missed>` denotes an unmatched crop. `counts.all.correct / counts.all.n` is end-to-end top-1 accuracy; `counts.all.correct / counts.all.matched` gives accuracy among recovered crops. Pair counts, capture exclusions and per-view aperture errors stay available for inspection.

## Held-out 26.3 results

The exposure option was selected on the development wall before scoring the held-out wall. The nine target materials occupy 243 physical positions. Four camera poses and three rendering conditions yielded 2,658 fully visible observations after occlusion checks.

| Crop | Matched / expected | Production correct | Exposure correct |
|---|---:|---:|---:|
| known | 2658 / 2658 | 1628 | 1999 |
| seeded_grid | 1745 / 2658 | 448 | 534 |
| image_only | 216 / 2658 | 30 | 31 |

With known corners, ordinary noon improved from 717 to 825 correct, the warmed terrain shader from 590 to 823, and small faces from 321 to 351. Each condition had 886 observations. At sky-light level 11 the result improved from 160 to 471 out of 720, while open sky (level 15) changed from 1,206 to 1,207 out of 1,488. Out of 3,024 unlike neighboring pairs, both labels were correct for 1,516 production pairs and 2,001 exposure-adjusted pairs. The number incorrectly merged into one predicted material fell from 163 to 84.

### Confusion with known corners and exposure matching

Rows are actual textures. Columns are predicted textures. The five coral columns represent dead coral blocks. Other combines predictions outside the nine target materials. The linked JSON retains every individual class and miss.

| Actual / predicted | Tuff | Andesite | Cobble | Deepslate | Brain | Bubble | Fire | Horn | Tube | Other | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tuff | 241 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 63 | 306 |
| Andesite | 0 | 229 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 65 | 294 |
| Cobble | 0 | 14 | 223 | 0 | 0 | 0 | 0 | 0 | 0 | 48 | 285 |
| Deepslate | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 189 | 309 |
| Brain | 0 | 0 | 0 | 0 | 235 | 0 | 0 | 0 | 0 | 86 | 321 |
| Bubble | 0 | 3 | 0 | 0 | 0 | 162 | 0 | 1 | 0 | 41 | 207 |
| Fire | 0 | 0 | 0 | 0 | 0 | 0 | 252 | 1 | 0 | 56 | 309 |
| Horn | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 245 | 0 | 58 | 306 |
| Tube | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 292 | 29 | 321 |

Deepslate had 95 predictions of cobbled deepslate among 309 appearances. Both algorithms struggled with small faces. Whole-image recovery found only 216 usable crops; improving material ranking added one end-to-end correct result. For known corners, bounded exposure improved all nine individual class totals.

To use the challenger, pass `--match-exposure` to `reconstruct.py build`. It fits one achromatic brightness multiplier per candidate (0.28–2.2) and penalizes departure from the original exposure. The original exemplar and catalog constraints still apply. On the independent repair-wall example the default recovered 31 of 36 block IDs at their true coordinates, and the option recovered 34 of 36; both generated 36 placements.

[Development selection](../tests/results/texture-development-26.3.json) · [Full held-out confusion and geometry counts](../tests/results/texture-heldout-26.3.json)

### Combined holdout

After selection, repeat the same invocation with stage `combined` to score the locked `engine.texel_grid.candidate` output against the original test report. The whole screenshot is its sole geometry input. The 12 held-out panel screenshots and membership cutoff are identical to the original material test.

| Image-only geometry | Matched / expected | Production texture correct | Exposure texture correct |
|---|---:|---:|---:|
| Original | 216 / 2,658 | 30 | 31 |
| Texel grid | 156 / 2,658 | 7 | 6 |

Texel-grid crops recovered 87 of 886 ordinary faces, 15 of 886 small faces and 54 of 886 warm-shader faces. With exposure matching, those sets yielded five, zero and one correct end-to-end identifications respectively. The measured geometry gains on the separate architecture scenarios did not transfer to material recovery on this wall. [Combined holdout counts](../tests/results/texture-combined-26.3.json)
