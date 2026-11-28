# One complete block face before a ruler

The source-image finder locates a full stone-brick face on the floating Chapel and substantially improves the export. Applied to every accepted replacement, it improves both new six-view panels but regresses on several previous layouts. The geometry check that substitutes a new face only when the old seed fails a square-shape test changes Chapel alone. Keep these methods experimental.

[Interactive source images and enlarged face comparisons](source-review.html) · [All source-only placement and material scores](source-scores.md) · [Reference controls](reference-scores.md)

## What a correct ruler can recover

| Panel | Views | Floating candidate | Source-discovered face | One exact face per eligible wall chart | One rounded face per chart | Two rounded faces with known offset |
|---|---:|---:|---:|---:|---:|---:|
| Original test views | 38 | 35.64% | 35.28% | 48.01% | 33.41% | 45.51% |
| First additional layouts | 12 | 30.43% | 30.18% | 34.06% | 24.19% | 28.71% |
| Supplemental layouts | 6 | 39.92% | 38.55% | 41.84% | 28.04% | 40.74% |
| Floating-sky composites | 6 | 22.93% | 27.45% | 38.12% | 18.70% | 32.50% |
| Earlier holdout A | 6 | 18.83% | 18.47% | 25.33% | 15.98% | 25.34% |
| Earlier holdout B | 6 | 22.15% | 21.46% | 27.37% | 20.44% | 26.87% |
| New holdout C | 6 | 25.61% | 26.04% | 33.45% | 22.04% | 30.57% |
| New holdout D | 6 | 18.63% | 32.38% | 44.86% | 29.29% | 38.42% |

All entries are strict position F1 after one translation for the whole build. Reference-assisted arms establish method-specific opportunities with the ordinary sampler, atlas and horizontal-face assembler. They change only eligible, already seeded wall charts with a usable complete reference face. Each selected face lies wholly inside the source mask and image. Every native pixel center strictly inside the projected quadrilateral must have that face's frontmost ID. It needs at least 16 such samples and a shortest side of at least four pixels. Recorded camera and corners must agree within 0.0001 pixel. The closest eligible face to any existing seed is chosen; visible area breaks ties.

The exact handoff gives ordered corners so the chart receives its horizontal and upright directions as well as unit scale and phase. The one-view control corrects only the eligible chart with the most proposed cells. Other variants correct one face per eligible chart. The visible-surface control additionally supplies the native surface mask intersected with the source mask. The ordinary variants continue extending across the original source mask. Source-only material matching reruns using the newly projected tiles. Origins and depth relationships between separate charts are left to assembly.

For the precision test, round each reference coordinate to the center of its native image pixel. Four rounded corners alone often extrapolate poorly across a large wall. Keeping the source chart's estimated perspective and using the rounded corners to estimate its local axes, spacing and origin raises the six-sky result from 18.70% to 29.93%. A second widely separated complete face raises it to 32.50%, when given its correct relative integer offset. A source-only two-face variant infers that offset and requires its corner disagreement to be at most 0.30 blocks. The second source face improved some views but also introduced regressions. The two rounded faces in the reference column receive additional information.

## Actual image-only choices

The old Chapel ruler measured internal calcite texture. The new search rectifies the wall using source lines or a predicted normal, proposes quadrilaterals from four edges or an atlas tile, and checks native pixels and neighboring tiles. Its selected stone-brick face overlaps the recorded quadrilateral with IoU 0.85 and corner RMS error 0.064 blocks. Chapel goes from 118 correct positions, 1,431 extra and 19 exact IDs to 107 correct, 58 extra and 39 exact IDs. On Riverside Mill 1 high-right, extras fall from 2,723 to 53, while correct positions fall from 152 to 97. The first effect explains why reporting only pooled F1 would be misleading.

Across 367 measured charts in 86 conditions, 312 had at least one original seed meeting the shared full-face criteria. Source selection increased that to 318, including 15 newly qualified charts and nine lost. Of 183 replacement faces, 170 qualified. A reference chooser could keep original aligned charts and find qualifying offers for 43 more, reaching 355 of 367. Replaying those choices still lowered geometry on one older holdout. Four-corner agreement is a valuable intermediate measure, with a separate end-to-end cost for how far a grid is extended and which chart group is exported.

The atlas verifier improved seed precision and lost recall. Its accepted neighboring tile on Chapel was another crop at the same incorrect scale. A learned logistic gate trained on three prior training layouts also missed the Chapel error. We tested requiring a second independently located source face, refining the single-face fit when a second agrees, and a shape challenge that evaluates the old seed in the proposed new wall plane. The shape challenge retains the original wall if any seed rectifies to an approximately square cell with perpendicular chart-axis assignments (side ratio at most 1.30, each edge within 15° of one axis). It changes only the Chapel export in this round. The sequential verify-then-find variants were assessed after seeing the component results; treat them as exploratory. [Full verifier study](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/c5017d7/benchmark/reviews/floating-face-verifier-01/README.md) · [Replacement finder and per-view diagnostics](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/e1f0218/benchmark/reviews/floating-face-proposals-01/README.md)

## Coverage and reproduction

The first 74 conditions include previously reviewed layouts and six floating-sky composites. The last 12 images use six additional layouts disjoint from those panels. Some new layouts share a family with previous images. The underlying archives contain 144 Visible v2, 144 Visible v3 and 40 Decorative v1 images. The two new panels each use two images per dataset. Production fixed 104 is carried forward at 91.27% strict and 93.42% facade, with zero default delta. Its strict counts are 20,773 correct, 2,929 missed and 1,046 extra. The facade counts are 18,952 correct, 1,624 missed and 1,046 extra. The empty Castle baseline remains a miss. The older Cloister views have a separate compass-transfer check. No floor reference enters the floating reconstruction. [Independent seed, registration and export audit](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/929192a/benchmark/reviews/floating-face-audit-01/README.md)

The companion branches are `experiment/floating-face-audit`, `experiment/floating-face-verifier` and `experiment/floating-face-proposals`. Use their registered worktrees with the stored frozen image and chart handoffs. Run from `experiment/floating-one-face-control` with the benchmark Python:

```powershell
$panels = 'sky','original','fresh','supplemental','holdout','holdout2','holdout3','holdout4'
python -m benchmark.floating_one_face_control @panels
python -m benchmark.floating_one_face_control --precision @panels
python -m benchmark.floating_one_face_control --stabilize @panels
python -m benchmark.floating_two_face_consistency @panels
python -m benchmark.floating_two_face_consistency --shape @panels
python -m benchmark.floating_verify_then_find
python -m benchmark.floating_one_face_scores
python -m benchmark.floating_one_face_review
```
