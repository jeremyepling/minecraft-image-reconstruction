# Correcting photographed sections at the wrong depth

On the 104 whole-structure fixed images the guarded correction removes **214 extra positions** and recovers 12 visible positions while losing one. Relative geometry F1 increases from **88.22% to 88.67%**. All 243 measured canopy gap rays across the fixed and alternate views stay open. Recommend this guarded method from `experiment/plane-section-relocation` for integration with the ordinary placer. Its production entry is `engine.section_decision.recover(image, parts, camera, prior, atlas_path=..., mode="deeper")` at `0479341`. The experiment branch keeps the ordinary placement entry point untouched.

| Photographed section rule | Fixed structures (104) | Alternate views (20) | Two more structures (4) | Blocked fixed / alternate gap rays |
|---|---:|---:|---:|---:|
| Previous production | 88.22% | 86.69% | 82.79% | 0 / 0 |
| Fixed confidence and coverage thresholds | 88.30% | 86.85% | 82.79% | 2 / 5 |
| Two decision heads, all jointly verified changes | 88.68% | 87.49% | 83.01% | 2 / 2 |
| Remove only from that joint result | 88.55% | 86.94% | 83.01% | 0 / 0 |
| **Deeper relocation and high-confidence rejection** | **88.67%** | **86.99%** | **83.01%** | **0 / 0** |

## Photograph and decision

On an already placed, calibrated image, reliably registered visible faces nominate atlas textures, orientations and mip levels. A masked patch covering at most 28 by 28 original pixels searches integer planes through the measured extent of the build. One checkerboard half of the photographed samples selects an observed texture template. The other half checks the match. A patch must beat alternative planes and directions before its samples can vote together for another block position. This can reveal several mispositioned cells at once. The rule consumes RGB, photo-derived masks, estimated camera and prior local cells.

The decision heads weigh the number and spatial spread of supporting samples, their correlation and margin over competing depths, neighboring cells, how much of the old face disagrees, and the size of the newly proposed face. We fitted them on the 72 complete training screenshots, with all members of each of the six families held out in turn for model selection. At .50/.60 the complete unguarded relocation export gained 50 correct and lost two under held-family model inference, adding 14 extras and removing 73. All six families improved in pooled F1. The four hand-designed rules initially tested on those same images gained at most 51 correct positions, with the most permissive rule also losing 45 and introducing 160 extras. [Development outcomes](development.json)

To protect openings, the guarded rule only adds a cell when every native pixel on its projected faces already had geometry at that depth or closer. The original conflicting source must be removed for the new cell to remain visible. A joint rerender verifies the surviving textural witnesses. A highly confident rejection can also remove a wrong old cell when there is insufficient evidence to export its replacement. The .90 rejection threshold scored at least 98% raw precision on each individually held-out training family, 151 correct rejections and one mistaken rejection in total. We introduced the guard after the first joint method closed previously clear gaps in reviewed fixtures. These reviewed cohorts now serve as regressions for the design choice.

## Fixed shapes and transfers

Of 104 fixed photos, 58 improved, 45 tied and the high-right cool Twin Spires 2 view declined by 0.19 percentage points. Correct positions rose from 20,203 to 20,214 and extras fell from 1,894 to 1,680. The number of extras outside the source structure's horizontal X/Z footprint fell from 1,084 to 938. The 110 coincidences with hidden source blocks stayed at 110. All ten measured opening widths and all 294 roof cells correctly anchored to both posts were retained.

Canopy 06 from the left had 29 correct positions and 92 extras before this experiment. The guarded result preserved all 29 correct positions and removed 20 extras, improving F1 from 28.29% to 31.35%. Its depth extent fell from 62 to 56 blocks against ten in the reference. The texture vote generator contained 13 genuinely missing positions on this view, and the original removal head flagged 48 existing positions, all erroneous. The ordinary joint selection had rejected all of the replacements and therefore restored all of these sources. Further work should investigate why the nominated visible roof cells were rejected.

Among the alternate 20 images, nine improved and eleven tied. Eight correct positions were gained and one lost, with 19 extras removed and zero extras added. On the four views of two separate structures, three improved and one tied, all four by preserving every correct position. The method removed two switchback and two loading-dock extras. The latter four photos were reserved for the earlier benchmark and have since been examined. Recheck on fresh structures before treating this transfer result as a general accuracy estimate.

## Frozen results and replay

Each method's archive contains its per-frame scores, screenshot and fixture hashes, both translations, decoded cells and the actual exported packs. Each reported F1 allows one shared integer translation for the complete visible build. The source world and camera truth are read for fitting or scoring only. The evaluator uses all retained dataset views relevant to each split and the framed transfer fixtures. The eight archived close-ups are outside these cohorts. [Guarded packs](deeper/builds.zip) · [Guarded cohorts and roof checks](deeper/summary.json) · [Unguarded joint comparison](learned/summary.json) · [Geometry removal only](trim/summary.json) · [Original fixed-threshold rule](strict/summary.json)

From the branch root, regenerate the 72 training measurements and fit the model with the following commands. The benchmark expects the frozen placement inputs and prior builds from the preceding release.

```powershell
python -m benchmark.plane_section_trial --splits train --workers 4 --out train.json
python -m benchmark.section_decision_fit --workers 4
python -m benchmark.section_decision_validation
python -m unittest tests.test_section_relocation
```

From the `review/framed-depth-round` checkout, score the candidate using the common evaluator. The other two adapters in `benchmark.section_decision_fit` are `postprocess` and `trim`; the fixed-threshold trial uses `benchmark.plane_section_trial:postprocess`.

```powershell
python -m benchmark.framed_depth_eval --candidate-root ../plane-section-relocation --candidate benchmark.section_decision_fit:deeper --out ../plane-section-relocation/runs/sealed-deeper --workers 4
```
