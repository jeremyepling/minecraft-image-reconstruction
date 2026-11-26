# Floor texel camera experiments

The camera takes a single native-resolution RGB image. It estimates a projection for the visible block grid. Its local origin and horizontal axis names are arbitrary. The evaluator resolves the integer origin and horizontal square-grid symmetries with the recorded camera and a fixed ground ray during scoring. A metric camera alone supplies no block placements.

| Method | Training views with a metric camera | Eligible corners within 2 px | Projected corners within 2 px |
| --- | ---: | ---: | ---: |
| Production floor seams | 0/72 | 0/24,644 | — |
| R1 amplify floor contrast, fit dark seams | 0/72 | 0/24,644 | — |
| R2 fit unit-texel pencils, leave block origin arbitrary | 48/72 | 2,548/24,644 | 2,548/15,548 |
| R3 register both image halves to a vanilla sprite | 72/72 | 24,644/24,644 | 24,644/24,644 |

R2 recovered focal length to a layout-averaged 0.014° vertical-field-of-view error on its supported images. A texture can repeat every block without a dark seam. Its native texel boundaries reveal the pitch, while the atlas identifies where a block starts within those 16 texels. R3 tries narrower foreground bands when the broad band fails. Each half must agree on the 16-by-16 phase, sprite and orientation. The camera retains the original dark-seam projection whenever available.

| Fixed R3 evaluation | Metric views | Eligible corners within 2 px | Mean error on projected corners |
| --- | ---: | ---: | ---: |
| Training, six families | 72/72 | 24,644/24,644 | 0.120 px |
| Cloister validation | 12/12 | 3,286/3,286 | 0.098 px |
| Twinspire evaluation | 12/12 | 4,626/4,626 | 0.108 px |
| Fixed exterior regression | 48/56 | 13,938/14,218 | 0.243 px |

The 128 additional accepted views all matched light gray concrete. The 16 earlier seam-camera results were retained. All eight legacy close-front views abstained. Several screenshots share a floor and camera preset, which limits independent evidence across materials and viewpoints.

[Saved per-image projections, provenance, scores and timings](scores.zip) cover the 72 development frames for R1 and R2 and all 152 for frozen R3 and its hardened replay. R1 was frozen at `28dcc15`, R2 at `ded2c21`, R3 at `1fedad9`, and the hardened reader at `fb9d5b4`. The hardened replay reproduced every original R3 prediction and score, including abstentions. Raw timing varies by run. For R1, the scorer manifest lists the later replay checkout revision. Its candidate file is unchanged from the R1 freeze.
