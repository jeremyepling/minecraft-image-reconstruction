# Perspective-aware cube turns

The single-image commands below document the cube-turn prototype at its archived revision. The `benchmark.cube_trials` commands at the end replay the experiment. [CLI snapshot at ca0a60c](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/ca0a60c)

```console
python reconstruct.py inspect-form IMAGE --cube-turns --out REVIEW
python reconstruct.py analyze IMAGE --geometry-method cube --out GRID
python reconstruct.py build IMAGE --geometry GRID/geometry.json --atlas ATLAS --out BUILD
```

The first command writes the silhouette, candidate faces and visible relief labels. In `cube-turns.png`, each color denotes a different projected block direction and each white dot joins two supported edge runs. `form.json` records the three homogeneous vanishing points, corrected runs and corner coordinates. The second command sends the corrected silhouette into elevation-grid estimation. The resulting `geometry.json` can be reused with any supported version's atlas. `build --geometry-method cube` runs this geometry selection and material matching together.

## How an image edge becomes a block edge

1. **Fit one camera to the scene.** Start with the whole-frame RGB line segments. Vote for vanishing points, then search camera focal lengths and sets of three mutually perpendicular directions. The camera model assumes square pixels and a principal point at the image center. A point at infinity represents a family of parallel lines in the picture. Require independent support for all three families. If the third family is too weak, keep the line-first measurements.
2. **Correct supported portions of the silhouette.** Project each candidate segment onto its closest supported axis. Pixels on opposite sides must support a foreground/background boundary. Where two different directions end near their intersection, join them there. Along a textured edge adjoining smooth background, remove a limited amount of spill beyond the edge and its convex corner. Limit the change to local strips so one roofline cannot straighten the entire skyline. Explicit image transparency takes precedence.
3. **Check rectangular faces and relief.** Opposite sides of a face must share a vanishing direction. Adjacent sides switch to another direction. Intersect fitted side lines to recover corners, and bound their displacement from measured endpoints. On a supported face, the remaining direction predicts where to look for a depth edge. Compare actual contrast steps and their termination on the original pixels. Missing or incompatible evidence can leave the face unknown. Flat surfaces require independent texture repetition.

Both the inspection and reconstruction commands use this contour. The reconstruction stage still estimates block spacing and dimensions separately. A single elevation can represent several visible planes poorly, and a cropped source can leave total building height unknown. The local recess/protrusion labels remain inspection results. Slabs and stairs may contribute several short axis-aligned runs; diagonal decorative models, textures and soft shading can leave stretches of outline unconstrained.

## Measured effect on Minecraft Java 26.3

The same frozen source revision scored the development and evaluation captures. The first group contains the 12 oblique views from the line-first study. The second contains front, left, right and elevated views of three open canopies. The scorers project known blocks and camera poses only after image-based inference.

| Evaluation group | Silhouette overlap, line-first → cube | Boundary F1, line-first → cube | Mean projected axis error |
| --- | ---: | ---: | ---: |
| 12 oblique building views | 94.61% → 95.16% | 85.40% → 87.62% | 0.23° |
| 12 open-canopy views | 86.15% → 86.75% | 88.94% → 89.93% | 0.21° |

The camera fit recovered three directions in all 24 evaluation views. Of 266 known clear sightlines through canopies, the outline incorrectly filled 88, down from 91. Silhouette overlap improved on 18 of the 24 views. Those figures refer to projected shape. Whole-building dimensions and hidden volume were outside this comparison.

For the separate test of local shape labels, both versions found 17 of 18 panel observations. The number correctly labeled recess, flat or protruding fell from 10 to 8. Two fewer recesses were correctly labeled. Unmatched positive relief calls fell from 15 to 12. On flush dark-inlay controls, false relief calls fell from 14 to 13. Both versions produced zero eligible panel proposals on the physically shaded flat control. Use the cube constraint for outline inspection and treat signed relief labels as hypotheses.

On the uploaded workshop image, two observed roof runs intersected near pixel (374, 90). The number of pixels labeled foreground in the top 100 rows fell from 8,226 to 1,773. This area includes true roof pixels and residual cloud. Without the original world, there is no ground-truth accuracy score for that photograph.

To reproduce from the existing fixture directories:

```console
python -m benchmark.cube_trials MAIN SIGNED_DEV --split dev --open-root OPEN_CANOPIES --flat-controls DEV_CONTROLS --out dev.json
python -m benchmark.cube_trials MAIN SIGNED_TEST --split test --open-root OPEN_CANOPIES --flat-controls TEST_CONTROLS --out test.json
```

[Development scores](../tests/results/cube-turns-dev-26.3.json) · [Evaluation, flat controls and per-view evidence](../tests/results/cube-turns-test-26.3.json) · [Fixture and outline scoring details](LINE_FORM.md)
