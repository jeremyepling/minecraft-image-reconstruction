# Calibrating the right Twinspire wall from the left wall

On the high-right Twinspire screenshot, the right tower’s brick side spans **eight columns by ten rows**. The previous grid proposed 143 cells. The source-only correction proposes 80. Each new cell center lands on a distinct recorded visible face. The ordinary assembler exports 110 blocks across the two visible right-tower walls, which share ten corner blocks.

The screenshot supplies a reliable brick chart on the left tower and a shared vertical edge on the right. The two side walls are parallel at different depths. The experiment transfers the left chart’s horizontal direction and block scale in homogeneous coordinates, using the right tower’s front chart for the local vertical scale and the existing seam for the origin. That estimate places the two far corners at 8.65 and 9.01 columns. A small integer-width search fits the right silhouette while leaving the near seam fixed. The median correlation against a full vanilla brick texture is 0.617 at eight columns, 0.311 at nine, and 0.281 at seven. Direct projective transfer scores 0.116. Candidate geometry and material sampling use only the source image, its predicted charts and the vanilla atlas.

| Twinspire view and method | Strict / facade placement F1 | Strict / facade position + material F1 | Correct / missed / extra | Exact IDs at correct positions |
|---|---|---|---|---:|
| High-right, prior grid | 47.11% / 49.22% | 17.99% / 18.79% | 110 / 184 / 63 | 42 |
| High-right, parallel scale | 54.46% / 57.29% | 44.55% / 46.88% | 110 / 184 / 0 | 90 |
| Low-left, prior grid | 57.44% / 60.54% | 16.92% / 17.84% | 112 / 166 / 0 | 33 |
| Low-left, parallel scale | 57.44% / 60.54% | 23.08% / 24.32% | 112 / 166 / 0 | 45 |

The high-right source-only export removes all 63 extras and preserves the 110 correct positions. The left tower remains in a separate component. Scoring with native face IDs after export confirms 80 distinct right-side brick faces at the 80 new cell centers, up from 79 at the 143 prior centers. Median grid area per visible block falls from 1.94 to 1.02 cells. Median absolute corner errors in the near and far image quarters fall from 0.176 and 0.205 blocks to 0.023 and 0.030. The other three brick charts remain unchanged. [Native measurements](native-centers.json)

Reproduce both Twinspire views and the six fixed control exports from the frozen fragment-recovery charts at `5bd7908` using the benchmark Python environment.

```powershell
python -m benchmark.floating_parallel_scale --panel original
python -m benchmark.floating_parallel_scale --panel sky
```

Each [original-view receipt](original) and [six-view receipt](sky) records the source image and prior chart hashes, fitted widths, material correlations for eligible patches, and decoded scores. New charts and packs use `runs/parallel-scale/<panel>`. Keep `experiment/floating-twinspire-parallel-scale` experimental. The rule fired on two views of one previously reviewed building. The six other buildings supplied no qualifying change. The benchmark implementation and paired evidence landed only on this challenger. The separate fixed-104 production measurement carries forward at 91.27% strict and 93.42% facade placement.

## Floating branch scorecard

Six fixed open-sky composites include four Visible Builds v3 and two Decorative Builds v1 views. Floating integration revision `9963dee` carries forward the sparse default. The two experimental rows are measured on the same six images using their expanded chart inputs.

| Method | Strict / facade placement F1 | Strict / facade position + material F1 | Correct / missed / extra |
|---|---|---|---|
| Integration default, carried forward | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Fragment-recovery baseline | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |
| Parallel brick-scale candidate | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |

The fixed-six candidate changes correct positions and extras by zero and all four F1 scores by zero points versus its paired baseline. No behavior landed on the integration branch.
