# Choosing the depth of a visible face

The RGB plane ranker at `3f9974b` improved coordinate precision and recall on every evaluated cohort. Recommend promoting the ranker from `experiment/atlas-plane-depth`, subject to the coordinator's comparison with a calibrated depth-model alternative. This branch contains the standalone candidate. A 26.3 vanilla atlas, the estimated camera, production masks, normals and counts supply its inputs.

For each mask, the ranker considers integer planes near the current estimate and a second estimate from scene-wide apparent ground contacts. It compares which plane places projected grid lines on the visible mask boundary, aligns texel transitions in two image directions, and yields complete visible tile crops matching the calibrated atlas. Neighboring blocks can match different sprites. Twelve numerical features and a conditional logistic model rank competing planes within the mask. The ordinary visibility, occlusion and datapack steps then consume the chosen offset. Missing or damaged model weights or atlas assets return the original placement.

[Stepped Hall example](stepped-hall.jpg) · [Verified manifests, candidate features, trained folds and decoded packs](evidence.zip)

## Rounds on development buildings

| Same six training screenshots, one per family | Correct visible XYZ | Extra XYZ | Extra on hidden structure |
| --- | ---: | ---: | ---: |
| Original | 764 | 398 | 104 |
| Projected mask boundaries | 843 | 260 | 29 |
| Native texel-edge correlation | 790 | 362 | 84 |
| Full-tile atlas matches | 903 | 168 | 16 |
| Wider proposals and weighted combination | 917 | 146 | 16 |

Across all 72 training views the original depth put 413,836 of 834,281 sampled mask pixels on their visible labeled plane. Within ±6 of the initial estimate a perfect selector could place 738,401 samples on their plane. Adding offsets near the estimated scene distance increased that ceiling to 761,970. Holding out each complete building family during fitting, the learned model reached 731,671 and the fixed weighted rule reached 697,448.

We also fitted a model using each visible face's fractional mask coverage so large on-screen blocks carried less training weight. On the six excluded-family pack decodes, the pixel-supervised model recovered 13,563 of 18,322 visible blocks with 3,040 extras and 205 hidden extras. The face-balanced alternative recovered 13,511 with 3,122 extras and 210 hidden extras. The pixel-supervised model won on all three quantities and the final weights then used all 72 training images. The original recovered 10,056 with 7,330 extras.

## Frozen comparison

| Cohort | Original correct / extra / hidden extra | Fixed rule | Learned RGB ranker |
| --- | ---: | ---: | ---: |
| Cloister validation, 12 | 1,799 / 713 / 140 | 2,138 / 361 / 0 | 2,161 / 340 / 0 |
| Twin Spires test, 12 | 2,218 / 1,064 / 518 | 2,840 / 386 / 37 | 2,896 / 273 / 16 |
| Fixed exterior regression, 56 | 5,967 / 3,508 / 978 | 7,405 / 1,423 / 115 | 7,719 / 1,048 / 76 |
| Independent camera-supported native, 12 | 2,760 / 489 / 27 | 2,904 / 284 / 9 | 2,909 / 257 / 5 |

The ranker gained 362 correct blocks and lost zero on Cloister, gained 750 and lost 72 on Twin Spires, gained 1,967 and lost 215 on the fixed suite, and gained 203 and lost 54 on the independent views. All eight fixed buildings gained correct blocks and reduced extras. One native facade family lost 20 net correct and added 14 extras. The largest independent gain came from the corner family with 124 net correct blocks and 190 fewer extras. Position plus the smooth-stone placeholder ID rose from 73 to 85 on the native cohort and remained zero on the other three.

The training targets use only mask samples assigned to visible faces in the supplied image. Evaluation projects the frozen local choices to the recorded reference grid and decodes the exported pack before scoring correct placements or extras. Hidden structure is counted as an extra. Multiple actual depth levels inside one mask still limit a single selected plane; Stepped Hall's highlighted mask had 2,989 samples on the recovered plane and another 633 one block away. Median candidate placement time on supported images varied from 0.61 seconds to 1.19 seconds across cohorts in concurrent runs.

## Replay

### Secondary depths inside merged masks

After freezing the ranker, we tested whether a patch judged to occupy one plane contains a second, visibly offset region. The trial requires at least two good primary atlas matches. It tests offsets one and two units on either side. A secondary match needs a neighboring tile with its own atlas match, stronger texture agreement than the primary reading at the same image location and complete original-mask coverage at the placement raster samples. Reallocated cells still pass through the ordinary occlusion and duplicate checks. Missing atlas data and nonfinite geometry preserve the primary estimate.

For the frozen guarded version `ce4701e`, the comparison below uses the final primary model fit on all 72 training views. These training counts differ slightly from the excluded-family comparison above.

| Versus the primary ranker | Correct gained / lost | Extras added / removed | Hidden extras added / removed |
| --- | ---: | ---: | ---: |
| Training, 72 | 68 / 3 | 1 / 69 | 0 / 20 |
| Cloister, 12 | 0 / 0 | 0 / 0 | 0 / 0 |
| Twin Spires, 12 | 4 / 2 | 1 / 5 | 1 / 5 |
| Fixed exterior, 56 | 19 / 0 | 0 / 20 | 0 / 0 |
| Native camera-supported, 12 | 0 / 0 | 0 / 0 | 0 / 0 |

The gain transferred to Twin Spires and multiple fixed exterior buildings. Recommend adding this refinement if the benefit survives the combined placement pipeline. Replay with candidate `benchmark.atlas_depth_regions` and the primary ranker's output as `--baseline`. [Paired decoded packs and intermediate variants](secondary-depths.zip)

### Commands

Prepare the common frozen inputs with the `review/visible-depth-rounds` harness. Use its checkout to launch `benchmark.visible_depth_score`; its `--candidate-root` argument selects this checkout. For native scoring use `--split native`. Keep all views of one building family together for fitting.

```powershell
python -m benchmark.visible_depth_score --candidate-root ../atlas-plane-depth --candidate engine.atlas_plane_depth:frozen --split validation --out runs/standalone-depth
```

From this experiment branch the following commands rebuild the per-surface training table, fits and excluded-family pack comparison. Use `--cell-target` on the fit for the alternative objective.

```powershell
python -m benchmark.atlas_plane_round --split train --workers 4 --watch 72
python -m benchmark.plane_visible_cell_labels
python -m benchmark.plane_offset_ranker --out runs/plane-depth/pixels-72.json
python -m benchmark.plane_depth_oof --fit runs/plane-depth/pixels-72.json --scorer-root ../visible-depth-rounds --out runs/oof-pixels
python -m benchmark.plane_depth_picture --model runs/plane-depth/pixels-72.json --out runs/stepped-hall.jpg
```
