# Floor-calibrated monocular depth

Keep this branch experimental. A pretrained depth prior helped resolve a few ambiguous planes, but the added inference still put more blocks on hidden structure in the fixed exteriors.

The first trial fitted Depth Anything V2's relative inverse depth to metric floor rays on both sides of the screenshot. It improved a six-frame training pilot from 764 correct placements and 398 extras to 876 and 180. On all 12 Cloister views it fell from the previous default's 1,799 correct and 713 extras to 1,592 and 1,031. Using MoGe point depths gave 974 correct and 1,837 extras. Calibrating again at visible wall-floor contacts gave 1,580 correct and 1,037 extras with Depth Anything.

The second approach lets atlas and outline evidence compete with three depth-prior features in a conditional linear ranker. Each of six training families was held out in turn. Across 72 training views, the selected planes covered 731,756 labeled pixels with RGB features and 732,275 with the additional prior. Three families improved and three regressed. We fitted final weights on all 72 views and froze them at `0c7b955` before scoring exported coordinates on the following suites.

| Images | Atlas plane chooser, correct / extra / hidden | With depth prior, correct / extra / hidden |
|---|---:|---:|
| Cloister, 12 | 2,161 / 340 / 0 | 2,171 / 339 / 0 |
| Twin Spires, 12 | 2,896 / 273 / 16 | 2,921 / 249 / 16 |
| Fixed exteriors, 56 | 7,719 / 1,048 / 76 | 7,725 / 1,032 / 96 |
| Camera-supported native, 12 | 2,909 / 257 / 5 | 2,925 / 254 / 5 |

Hidden is a subset of extras overlapping occupied but unobserved structure. The target contains only blocks with visible face evidence. Ground-truth camera and world data enter during supervised fitting and scoring. The predictor receives only the photograph and its image-derived camera, masks and counts. The image-only floor solve supplies metric scale.

[Frozen weights and per-family validation](model.json) · [Decoded datapacks, coordinates, priors and scores](scoring.zip)

## Reproduce the final comparison

Use the upstream model dependencies from the normal pipeline, the fixed camera and mask inputs from the [benchmark release](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/53b19c0/benchmark/reviews/visible-depth-inputs-01), and the Visible Builds v1 release. Point `MINECRAFT_DEPTH_ONNX` to the project's Depth Anything V2 ONNX file. `MCIR_DEPTH_EXPERIMENT_CACHE` can point to a writable cache for the model fields. From the checkout containing the frozen-input scorer, set `MCIR_METRIC_PLANE_FIT` to this branch's `model.json` and run:

```powershell
python -m benchmark.visible_depth_score --candidate-root PATH_TO_THIS_CHECKOUT --candidate benchmark.metric_plane_training:predict --split validation --out runs/metric-replay
```

Repeat with `test`, `regression` or `native`. For the three direct depth proposals use `benchmark.floor_depth_round:predict`, `:points` or `:anchored`. To refit, supply the training choices from the [atlas depth experiment](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/3f822ae/benchmark/reviews/atlas-plane-depth-01) to `python -m benchmark.metric_plane_training --features PATH_TO_TRAIN_CHOICES --cache runs/depth-priors --out runs/refit.json`.
