# Structural wall lines and separated atlas repeats

A structural-edge refit on Stable Yard 1 low-left restored six of the eight positions lost by the earlier ruler, lost two other previously correct positions and recovered all 26 original correct position-and-ID pairs. The revised export has **74 correct / 298 missed / 25 extras**, with **44 exact IDs**. The whole-wall original had **78 / 294 / 1,374**, with **26 exact IDs**. An original-fallback rule accepts only this view in the 84-condition panel. Keep the change experimental. All eight matching distant faces lie on one sandstone row, and the trigger has yet to transfer to an unreviewed building.

[Structural lines, candidate grids and native atlas corroboration](structural-lines-and-sprites.jpg) · [Source decisions, per-axis line and texture evidence and decoded scores](conditions.json.gz) · [All cohort totals](summary.json)

## Measurement

Base revision `addbd4e` includes the previous peer-height trigger and source material recovery. Three of 84 existing conditions had independent wall masks disagreeing with the inherited block scale. Use only the observed source RGB and masks, atlas and prior source charts. Detect line segments in the chart's native bounding box. Require substantial length and proximity to the observed mask before comparing them to the two chart axes. Favor long segments and structural outlines. Fit robust vanishing points independently using source segments separated within the image. Candidate projected grid axes point toward those vanishing points. The visible tile center and local unit lengths are held fixed in the line-only control. These lines can still belong to different parallel surfaces.

Next fit peer-sized whole atlas tiles in the left 40% of source-mask pixel centers by horizontal image position. Check the eight strongest shortlisted windows against complete cells in the middle 20%, excluding immediate neighbors. Test the winning grid against the rightmost 40%, which was held out of that search. Then refine local U/V pitch by up to 8% and phase by ±1/16 of a cell, optimizing correlation of the same calibrated sprite on the anchor and two middle tiles. Every candidate must use a visible complete source area. Native boundary checks reject clipped or undersized full-face hypotheses. Baseline segmentation, peer detection and structural lines use the full screenshot. Only the tile matching holds out the far region.

The original whole-mask seam check requires both axes to favor the grid over their half- and double-pitch competitors. It rejects all three conditions. A follow-up local check enumerates all far tile candidates, then compares each same-material whole-face hit with verified competitors at half or double U, V or both. A winning tile must lead competing correlations by at least .05. Require a winning origin and two winning far tiles separated by at least three cells, with at least one far tile distant from the origin in both axes. Require both line families to fit. This checks a visible repeated sprite and its grid, with no requirement to invent hidden geometry. The thresholds were applied after reviewing these failures. A single matching row cannot verify all of a mixed-depth mask.

## Paired results

Each result is correct positions / missed / extras (exact position and ID).

| Stable Yard 1 | Original | Previous peer refit + material recovery | Structural axes only | Coarse refit + materials | Fine refit + materials, forced | Original-fallback local gate |
|---|---|---|---|---|---|---|
| Low-left | 78 / 294 / 1,374 (26) | 70 / 302 / 53 (36) | 72 / 300 / 77 (13) | 75 / 297 / 54 (42) | **74 / 298 / 25 (44)** | **74 / 298 / 25 (44)** |
| High-right | 63 / 291 / 1,198 (16) | 55 / 299 / 234 (6) | 54 / 300 / 281 (5) | 38 / 316 / 78 (1) | 40 / 314 / 84 (18) | **63 / 291 / 1,198 (16)** |

- On low-left 50/51 U lines and 11/11 V lines fit their source vanishing points. The coarse near match was cut sandstone at 3.375×3.375 old chart cells, correlation .889. Increasing both pitches by 4% produced two middle correlations of .756 and .742 and eight complete distant cut-sandstone faces among 101 candidates. They lie at V=5. Four at U=8, 15, 16 and 17 also passed the local harmonic challenge. Whole-mask seam evidence still vetoed the grid. Relative to the previous recovered chart, the accepted result gains four net correct positions, removes 28 net extras and adds eight exact IDs.
- On high-right 36/38 U lines and 3/3 V lines supported the respective vanishing points. The local candidate changed from a pearlescent-froglight motif to cut sandstone. Its near and two middle correlations were .833, .844 and .786 after refinement. No far complete match passed across 67 candidates. Its global export still lost positions. Local evidence cannot assign the correct pitch and position to the entire mixed wall.
- On Pillars, polished diorite matched locally while three far faces matched quartz bricks. The previously forced refit selected another component. The original-fallback arm declines and reproduces the original 69 / 314 / 687 (5 exact IDs).

| Active 84 | Strict | Facade | Strict + ID | Facade + ID | Correct / missed / extras |
|---|---:|---:|---:|---:|---|
| Original large-contraction | 33.54% | 34.01% | 5.51% | 5.61% | 18,323 / 61,187 / 11,420 |
| Local gate, original fallback | **33.96%** | **34.45%** | **5.61%** | **5.72%** | **18,319 / 61,191 / 10,071** |
| Previous experimentally forced peer refit with material recovery | 34.40% | 34.91% | 5.66% | 5.77% | 18,283 / 61,227 / 8,512 |
| Local gate, previous experimental fallback | 34.41% | 34.93% | 5.68% | 5.79% | 18,287 / 61,223 / 8,484 |
| Force fine structural refit on all three, with materials | 34.44% | 34.95% | 5.71% | 5.82% | 18,272 / 61,238 / 8,334 |

The original-fallback evaluation decoded 83 original coordinate-and-ID maps identically. Its only change removes 1,349 extras, loses four net correct positions and adds 18 exact IDs. The candidate was designed on known examples. The pooled gain is due to Stable Yard 1 low-left alone. The independent material-only recovery remains a different handoff and supplies the new chart's material assignments.

On the original-fallback arm, Visible Builds v2 30/144 conditions scored 42.83%, 45.70%, 17.02% and 18.47% in table order. Visible Builds v3 30/144 scored 34.10%, 34.24%, 3.90% and 3.90%. Decorative Builds v1 24/40 scored 26.90%, 27.77%, 6.82% and 7.20%. The separate fixed-104 production trend carries forward 91.27% strict and 93.42% facade, 20,773 / 2,929 / 1,046. No new lighting or building capture was tested.

From this checkout with the frozen sibling archives and mask-scalelsd Python available, run:

```powershell
python -m benchmark.floating_edge_aligned_ruler
python -m benchmark.floating_edge_aligned_ruler --report
python -m benchmark.floating_edge_ruler_sheet
python -m unittest tests.test_floating_edge_aligned_ruler tests.test_floating_peer_ruler_refit tests.test_relative_face_assembly
```

## Floating branch scorecard

The carried-forward six-sky integration at `c0accfd`, runtime `52de2c9`, scores 12.64% strict, 12.90% facade, 4.68% strict + ID and 4.77% facade + ID, with 273 correct / 3,768 missed / 5 extras. On the same six, the frozen large-contraction and local-gate original-fallback arms score 27.50%, 27.73%, 5.96% and 5.99%, with 890 / 3,151 / 1,542. This specific intervention changes zero correct positions and zero extras there. Relative to integration this path gains 617 correct and adds 1,537 extras. The separate forced peer-fallback arm carries 29.73%, 30.14%, 6.59% and 6.69%, with 866 / 3,175 / 919. Only experiment code and evidence were published on `experiment/floating-edge-aligned-ruler`. Integration retains connected-atlas.
