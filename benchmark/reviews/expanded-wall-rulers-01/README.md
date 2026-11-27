# How much do wall coverage, block spacing and origin change the exported build?

On 38 original test images, extending the existing image-derived grid across its observed wall mask raised strict placement F1 from **3.72% to 35.34%**. Keeping original cell materials and ranking added cells against the masked vanilla atlas raised strict position-plus-ID F1 from **1.54% to 4.51%**. The rule uses the original RGB, inferred surface mask, predicted wall normal and predicted grid. It produced 13,895 correct positions, 45,543 misses and 5,302 extras. The ordinary floating assembly had 1,127 correct, 58,311 missed and 107 extra positions.

| Surface masks and intervention | Strict placement | Facade placement | Strict position + ID | Facade position + ID |
|---|---:|---:|---:|---:|
| Ordinary source masks and charts | 3.72% | 3.76% | 1.54% | 1.55% |
| Source masks, resample predicted grids with existing sprite confidence gate | 35.34% | 35.55% | 0.99% | 0.99% |
| Same source placements, retain old IDs and rank new cells with masked atlas | 35.34% | 35.55% | 4.51% | 4.53% |
| Exact native surface masks, ordinary chart extraction | 8.47% | 8.56% | 2.89% | 2.92% |
| Exact masks, replace only numeric width and height in the ordinary Block Counter handoff | 9.19% | 9.30% | 3.76% | 3.81% |
| Exact masks, resample existing predicted grids | 41.18% | 41.57% | 2.68% | 2.71% |
| Same resampling with correct local width and height pitch | 51.88% | 52.33% | 4.85% | 4.90% |
| Also align one local corner to the true grid | 53.32% | 53.78% | 8.21% | 8.28% |

## Source-image results

| Cohort | Views | Strict placement, ordinary → resampled | Strict position + ID, ordinary → retained IDs + masked atlas |
|---|---:|---:|---:|
| Visible Builds v2, twin spires | 12 | 30.90% → 41.90% | 16.87% → 19.42% |
| Visible Builds v3, castles | 18 | 1.57% → 35.89% | 0.31% → 3.62% |
| Decorative Builds v1, walled gardens | 8 | 3.79% → 21.14% | 1.90% → 2.07% |

Strict geometry improved on 35 views, regressed on two and tied on one. The two regressions were twinspire2-cool-low-left (49.32% → 45.71%) and twinspire1-earth-low-left (41.27% → 40.80%). The final material variation improved strict position plus ID on 26 views, regressed on eight and tied on four. The largest losses were twinspire1-earth-low-left (14.29% → 8.05%), walled-garden-2-warm-low-left (4.50% → 0%) and walled-garden-1-warm-high-right (3.63% → 0%).

Resampling visits only cells with a native support-pixel witness. The existing shared-corner rule requires at least 3.5 pixels on the shortest projected edge, a supported center and at least 78% support on a 7×7 grid. It accepts cells even if the texture verifier falls back to smooth stone. This arm extends existing grids over observed support and changes texture sampling and chart confidence. The control is a no-op for the grid matrix only. The source test resampled 182 eligible charts with image-estimated vertical normals; source normals and their errors enter the result. Incorrect masks and grids can still produce unsupported placements.

The material variation keeps the original material, fallback flag and confidence at retained local cell coordinates. On added cells it uses the same masked 16×16 nearest-native sampling and `rank_tile` matcher as the complete-wall-grid trial, using each predicted surface's mask. The chart confidence is frozen. Each of the 38 exports has exactly the same ordered local and registered positions as its source resampling control. Of 19,197 exported cells, 18,888 now received an ID assignment and 309 used the fallback. Correct IDs rose from 389 to 1,773 on those placements. A top-ranked ID is still often wrong.

## Reference-only ruler replay

Replacing only the numeric `counts` field in the ordinary pipeline raised strict placement by **0.72 points**. This run supplied fractional block spans at the source quadrilateral's center for 1,636 eligible wall quads. Only 403 previously had both axes measured. It preserved the predicted quad, phase, texture evidence and confidence fields; the ordinary atlas, row sharing, registration and assembly consumers ran again. Seven views improved, seven regressed and 24 tied. The changes recovered 685 positions, lost 454, and removed 58 extras. Strict placement moved from 31.24% to 28.44% on spires, 6.80% to 7.92% on castles and 8.03% to 7.24% on gardens. This central pitch establishes an exact local scale only; an incorrectly rectified quad can still accumulate error across the wall.

The native mask ladder uses 307 of 310 available charts. Eligible predicted charts belong to visible coplanar vertical full-cube faces. The three other charts pass through verbatim. At an existing integer corner near the center of each predicted chart, the reference projection gives two Jacobian column lengths in native pixels per Minecraft block. The spacing arm rescales only the two predicted basis vectors until they give those lengths while keeping that source corner fixed. It preserves predicted vanishing points and axis directions. The next arm slides the local origin so the anchor lands at the intersection obtained by rounding the reference plane coordinates. This supplies **one** correct local corner; perspective or direction error can give a different phase elsewhere. World coordinates, true normals, lattice indices, true cell occupancy and face IDs do not enter the sampler, assembler or exporter. Both arms retain identical mask coverage and material rules. Every reference wall corner used in the calculation reprojects to within 0.000001 native pixel of its recorded projection.

The predicted-grid control emitted 27,452 eligible chart cells, the spacing arm 32,094, and the origin arm 32,534, before ordinary component selection. Only 51 charts had a predicted local pitch more than 10% wrong on either axis, but a few large castle faces had near-triple pitch on both axes. Exact local spacing gained **10.70 percentage points** over the common resampling control. Adding one reference origin gained **1.44 points**. The separate **32.71-point** step from native baseline to the resampling control includes changed occupancy, materials and chart confidence across accepted charts.

## Scope and decision

The comparison uses all published original test views for these three datasets, 38 images of eight layouts with one whole-build translation. Each datapack is decoded and checked against serialized local coordinates and material IDs. The ordinary masks and native masks share exactly the frozen inputs from revision d4f2e09. Camera or reference data enter only the indicated oracle fields or the established scorer. The historical fixed 104, the other views in the 144-view v3 and 40-view decorative collections, and lighting pairs were not measured here. The source-only variations were designed after inspecting the native-mask results on these same 38 test views. The next independent test should include new layouts, walls with openings and source-mask spill.

Keep `experiment/expanded-wall-rulers` experimental. Source-only geometry and material improved in all three cohorts, while the measured absolute typed score remains low and the resampler adds 5,195 extras over the ordinary export. This branch contains offline experiment runners and evidence. Production and the floating integration branch were unchanged.

Run in this checkout with the shared benchmark environment and the earlier crop-ceiling replay available.

```powershell
python -m benchmark.expanded_wall_rulers --workers 5
python -m benchmark.expanded_wall_counts --workers 8
python -m benchmark.expanded_wall_source_resample --workers 5
python -m benchmark.expanded_wall_source_resample --material --workers 6
python -m benchmark.expanded_wall_ruler_report
```

[Scores, aligned coordinate change counts, original source and decoded export digests](results.json.gz) · [Per-chart pitch, origin and count handoffs](handoffs.json.gz)
