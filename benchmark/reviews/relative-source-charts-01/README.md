# Independent source-image wall charts

On six frozen structures in the floating-build comparison, the dense atlas arm found 14 independent wall charts on the open-sky images. Each chart maps its own lower-left integer cell corners to source pixels. Positive U goes image-right and positive V goes image-up. The predictions carry observed mask outlines, a learned surface normal, relative inverse depth and an atlas match for each accepted cell. Surface contours approximate exterior boundaries. Cells carry the actual occupancy evidence, including gaps.

| Background | Structures with charts | Charts | Cells | Correct relative center offsets | Exact center material | 85% of samples on one face | Corner RMS ≤10% of edge |
|---|---:|---:|---:|---:|---:|---:|---:|
| Open sky | 5/6 | 14 | 83 | 62 | 51 | 75 | 44 |
| Plain | 4/6 | 9 | 46 | 39 | 33 | 38 | 24 |
| Pavement | 6/6 | 27 | 188 | 185 | 160 | 180 | 176 |

The grader samples a 9 × 9 inset grid in each predicted cell against native frontmost face IDs. For relative center offsets, it counts cells explained by one signed horizontal axis and integer translation per chart, with V constrained to world-up. Corner RMS uses the best cyclic winding of the true visible face's four projected corners, normalized by its mean edge length. The relative-center score gives each chart its own oracle registration. Inter-chart spacing and whole-build F1 require separate assembly evaluation.

Castle had 36/36 internally consistent centers and 36/36 correct material IDs in the open sky. Viaduct had 13/13 in each measure. Pillars had 11/27 consistent centers and 2/27 correct IDs. Walled garden had 1/5 consistent centers and 0/5 correct IDs. Chapel had 1/2 and 0/2. Harbor yielded no verified wall charts. False atlas matches can survive the native correlation checks on repeated textures, and a successful local sheet can still have unknown offsets relative to another.

## Replay

Start from the saved source-handoff directories in the experiment/floating-build-local-grid checkout. The same source hash must appear in the paired fixture manifest, input file and cached Surface Finder / Block Counter handoff. With the calibrated Java 26.3 atlas available, run from this branch:

```powershell
python -m benchmark.relative_source_charts ../../minecraft-image-reconstruction/runs/floating-build-grid-01/fixtures.json ../floating-build-local-grid/runs/floating-build-grid-01 --variants sky plain pavement
python -m benchmark.relative_chart_grade ../../minecraft-image-reconstruction/runs/floating-build-grid-01/fixtures.json runs/relative-face-assembly-01
python -m unittest tests.test_relative_source_charts tests.test_floating_build_local_grid
```

`charts.json` uses whole-surface atlas scale search and the earlier repeated-texture pitch proposals. `dense-charts.json` adds overlapping local search windows and independent wall rectification from a predicted normal, the source's vertical edge pencil, and assumed focal length equal to image height. Predicted patches can also propose rectification by observed surface aspect. Each candidate sheet uses its own unit pitch and local origin. No full 3D camera, source-world coordinates or ground plane enter inference. `counter-charts.json` is a separate exploratory comparison using the Block Counter's pitch and its phase when available. It records fallback material per cell.

The 54 archived handoffs include per-arm image, mask, count and atlas hashes. The two training, two validation and two test structures are regression examples from a previously reviewed cohort. The fixed 104-view suite and expanded cohorts were not rerun in this chart-only experiment. Keep this branch experimental pending whole-build placement and wider evaluation. The default on main remains at its previous verified score.

[Compressed handoffs](chart-handoffs.json.gz) · [Per-wall metrics](chart-metrics.json)
