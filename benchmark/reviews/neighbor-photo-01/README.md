# Neighboring visible texture phases

The working branch `experiment/visible-rim-recovery` adds a placement test after the existing depth chooser. When an existing cube has a visible face, the method looks one block to either side along that plane. It compares native RGB pixels from the proposed location with a photographed neighboring face at the same sub-block coordinates. To distinguish a texture repeat from a similar solid color, it tries quarter-block shifts on both texture axes.

Every candidate needs its own pixels within a detected subject mask, outside the original reconstruction's projected silhouette. A strong seed needs at least 35% of its projected face observed, at least 24 corresponding samples, mean absolute RGB difference at most 4 on the 0–255 scale, and improvements of at least 2.5 and 0.3 over the two shifted comparisons. A weaker adjacent face can pass with at least 30% coverage and 20 matched samples. The method retests source and target pixels with the new candidates jointly present and preserves the visibility of every original cube. Empty placements abstain. No world data enters this rule.

## Development

We selected the high-coverage seed and adjacent-face rule on the 72 training captures and 12 previously reviewed native views. Each entry counts added correct cells, extra cells and hidden cells among those extras. The evaluator decodes the exported packs and chooses one shared integer translation per structure.

| Rule | Training correct / extra / hidden | Reviewed native correct / extra / hidden |
| --- | ---: | ---: |
| Permissive phase match | 272 / 229 / 21 | 35 / 38 / 15 |
| Medium phase match | 161 / 63 / 6 | 18 / 16 / 5 |
| Strict phase match | 99 / 15 / 0 | 11 / 4 / 1 |
| Joint check with 8% minimum seed coverage | 88 / 10 / 0 | 8 / 2 / 1 |
| Joint check with 35% coverage, seed only | 30 / 1 / 0 | 7 / 0 / 0 |
| Joint check allowing one adjacent match | 30 / 1 / 0 | 9 / 0 / 0 |

An answer-assisted chooser could identify 676 missing targets among 3,673 distinct proposed training coordinates and 84 among 613 proposals on the reviewed native views. At the baseline's alignment, adding only these successes would reach 86.24% and 79.86% F1 respectively. This upper bound assumes perfect selection before proposals compete for image pixels. The test addresses a fraction of the missing geometry.

## Confirmation

Against the `3585aa5` default with the same frozen upstream outputs, the rule added 80 correct coordinates and seven extras across 112 evaluation images, including the 12 reviewed native images. Pooled visible geometry F1 rose from **83.947% to 84.142%**. The eight existing camera abstentions remained empty. One of the new extra coordinates was a hidden block on the fixed low-left masonry shopfront view. No original coordinate was lost.

| Suite | Images | Correct added | Extras added | Hidden added |
| --- | ---: | ---: | ---: | ---: |
| Fixed exterior | 56 | 33 | 3 | 1 |
| Validation | 12 | 5 | 1 | 0 |
| Test | 12 | 12 | 0 | 0 |
| Native six-layout suite | 12 | 17 | 1 | 0 |
| Previously reviewed native | 12 | 9 | 0 | 0 |
| Newer native confirmation | 8 | 4 | 2 | 0 |

Both canopy cohorts recovered three more correctly anchored roof cells. With the roof and the two posts required to share one translation, exact roof/post triples went from 648 to 663 out of 1,823 on seven reviewed canopies, and from 230 to 245 out of 500 on three newer canopies. The newer colonnade and mixed-corner facade each gained one incorrect coordinate.

Recommend testing this rule jointly with the foreground depth changes before promotion. The branch makes the callable available as `engine.neighbor_photo_faces.recover(image, parts, P, cells)`. The baseline and experiment use the same saved screenshots, estimated cameras, detected parts and counts. Actual camera poses and reference worlds are accessed by the scorer only. [Seven cohort manifests, per-image scores, oracle and decoded packs](evidence.zip)

## Replay

Restore the [frozen placement inputs](../visible-depth-inputs-01/README.md) and [current baseline and native captures](../canopy-placement-01/README.md). Use the 26.3 assets associated with that baseline. From the experiment checkout, use the stable checkout for shared inputs and a private output directory. In the commands below, `$stable` is the absolute path of the stable checkout.

```powershell
$baseline = Join-Path $stable runs/canopy-placement/release
python -m benchmark.visible_depth_score --split all --candidate benchmark.neighbor_photo_recovery:live --candidate-root . --baseline $baseline --out runs/neighbor-replay
python -m benchmark.visible_depth_score --split native --candidate benchmark.neighbor_photo_recovery:live --candidate-root . --baseline $baseline --out runs/neighbor-replay
python -m benchmark.canopy_compare --inputs "$stable/runs/visible-depth-rounds/transfer" --baseline "$baseline/old" --candidate benchmark.neighbor_photo_recovery:live --candidate-root . --out runs/neighbor-replay/old
python -m benchmark.canopy_compare --inputs "$stable/runs/canopy-placement/confirmation" --baseline "$baseline/fresh" --candidate benchmark.neighbor_photo_recovery:live --candidate-root . --out runs/neighbor-replay/fresh
```

The branch's unmodified default is used in the `live` adapter. Integrators can call the engine function directly on their own placement output. The archived `live` packs were generated from raw frozen inputs, and all 184 decoded coordinate sets, including the training images, matched a replay with the explicit native-pixel ownership guard at `944d551`. For development ablations use the `predict` adapter with `MCIR_NEIGHBOR_PHOTO_VARIANT` set to `permissive`, `phase`, `strict`, `joint_partial`, `joint_seed` or `joint_growth`. Use a new `MCIR_NEIGHBOR_PHOTO_CACHE` directory for each source revision. After a training and reviewed-native run, pass that cache to `benchmark.neighbor_photo_oracle` with `--scored` pointing to the output directory containing `train` and `old`.
