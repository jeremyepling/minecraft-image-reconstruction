# Local visible wall grid and depth choices

The archived production geometry at `028819a` scores 90.414% pooled visible-block F1 on the 104 fixed evaluation photographs. A label-assisted choice among local wall sections reached 91.248%. The learned operating point reached 90.522% and introduced 102 false blocks, including 66 blocks in hidden occupied space. Keep both image-only section selectors experimental.

## Photographic choices and oracle

Reproject the current geometry at native resolution. On a vertical surface mask, group unclaimed image rays into two-by-two-block patches at the wall's dominant observed depth. At each patch try five depth planes from −2 through +2, plus measured nearby levels with sufficient support. Try the ordinary grid origin and four in-plane bin shifts of ±0.375 blocks along either tangent. Each trial bins rays onto integer Minecraft coordinates. The shifts only change which integer cells are proposed.

Check each cell's actual projected face against at least 12 fresh, same-axis native pixels and require at least 88% silhouette support for the newly exposed geometry. Compare the projected texels, atlas phase, neighboring depth planes and nearby blocks. Alternatives can contain multiple blocks. Assign each candidate coordinate to a single section before opening labels and keep the unchanged placement available at every section.

The perfect chooser picks the largest wholly correct alternative per section using visible truth under the control's common XYZ translation. It then recomputes joint visibility. This is an **attained clean result**, with a fixed tie-breaker. Per-section choices are exact before the common render. Another tie-breaker or a search over coupled occlusion could change the final count. The independent union of candidate coordinates is a looser bound.

| Pool | Fixed evaluation, 104 | Development, 72 |
| --- | ---: | ---: |
| Ordinary origin and dominant depth | 134 | 37 |
| Alternative origins, dominant depth | 180 | 52 |
| Ordinary origin, alternative depths | 274 | 187 |
| Both alternatives before joint visibility | 376 | 262 |
| Both alternatives after joint visibility | 348 | 246 |
| Individually reachable missed coordinates | 454 | 315 |

Counts are additional correct visible coordinates. The last row permits incompatible alternatives. On the fixed pool, 606 disjoint candidate sections produced 2,188 nonempty alternatives. The clean chooser also recovered 66 blocks on 20 alternate views and 13 blocks on four new loading-dock and switchback views.

## Deployment tests

The logistic and boosted classifiers were trained only on the 72 development frames. Each cross-validation fold held out all photos from one of the six structural families. Every tested threshold with positive cross-family recovery introduced some false placements. Both zero-error settings therefore abstain. The 0.50 boosted operating point maximized development-fold correct additions minus 0.84 times false additions. The separate transfer rule sends every cell in an alternative to the already frozen vertical atlas verifier at 0.95. Its optional anchor requires a matching sprite on the same plane within three blocks.

| Photo set | Learned operating point, correct / extras / hidden | Frozen verifier, correct / extras | With nearby sprite anchor, correct / extras |
| --- | ---: | ---: | ---: |
| Development 72, after fitting | 115 / 14 / 10 | 3 / 0 | 0 / 0 |
| Fixed evaluation 104 | 129 / 102 / 66 | 1 / 0 | 1 / 0 |
| Alternate views 20 | 47 / 18 / 12 | 0 / 0 | 0 / 0 |
| New loading-dock and switchback 4 | 5 / 2 / 0 | 0 / 0 | 0 / 0 |
| Separate shelter and gate 4 | 0 / 2 / 1 | 0 / 0 | 0 / 0 |

The learned rule improved 38 fixed frames, regressed 22 and tied 44. All 137 marked clear opening rays stayed clear on the ten fixed canopy views. The frozen and anchored selectors each gained a single block on Galleries, masonry high-right. On the 22 additional ordinary-camera challenge views, the frozen verifier left all 22 control exports unchanged. Nineteen of those photographs had no usable upstream camera. Those cases test fallback only.

| Requested photograph | Individually reachable | Clean chooser after render | Learned correct / extra |
| --- | ---: | ---: | ---: |
| Canopy 04 | 3 | 1 | 2 / 1 |
| Colonnade 01 | 0 | 0 | 0 / 0 |
| Dev Relief 00 | 14 | 13 | 3 / 3 |
| Relief 00 | 16 | 14 | 3 / 14 |
| Shopfront | 15 | 14 | 0 / 11 |
| Facade | 15 | 15 | 5 / 0 |

These are the exact representatives from the preceding worst-structure gallery. The coarse depth and texture distribution permits several competing choices on the same rays, particularly on Relief and Shopfront. A projected face inside a real wall mask can still be at the wrong depth.

## Moving occupied wall cells

A second, separate experiment used rays already covered by a placed block. It traced vertical face candidates within two depth planes of that source, using the existing nearby-face sampler. Every alternative had support from three or more image sample points spaced four native pixels apart. The oracle restricted edges to incorrect sources and missing correct destinations under the common control translation. Bipartite maximum matching found 462 disjoint source–destination swaps across the 104 fixed photographs. The existing joint renderer retained 288 swaps, removed 288 extras including 30 hidden blocks, and kept all previously correct cells. That achieved 91.684% fixed F1. The matching is exact before joint occlusion and the post-render count is attained. The two oracles can propose overlapping cells, so their gains cannot be added.

On the 72 development frames, 20 alternate views and four new views, 256, 90 and 14 exchanges respectively survived. Among the requested single-image representatives, the occupied-wall chooser exchanged 23 blocks on Facade, 13 on Shopfront, one on Colonnade 01, and zero on Canopy 04 and both Relief examples. The frozen nearby-face model required at least 0.95 confidence that a destination was right and at most 0.10 confidence that its source was right, alongside its existing thresholds. It approved zero swaps on the 200 photographs. A learned placement correction would need to distinguish two competing depths on the same foreground evidence. For Colonnade 01 these trials expose almost none of the missing geometry, even with correct choices.

On the independent shelter and gate set, the unclaimed-ray section oracle attained three additional visible blocks, two on the low-left gate and one on the high-right shelter. The frozen verifier kept the original geometry on all four.

## Replay

The checkout `experiment/wall-section-grid-depth` holds the candidate implementation. Use the production replay from `../facade-placement-round/runs/production` and the calibrated atlas. The shared evaluator is `benchmark.wall_section_eval` from revision `b17caa8`; the central comparison checkout is `../wall-section-round`. From the experiment checkout run the following commands in order.

```powershell
python -m benchmark.wall_section_oracle --out runs/oracle --workers 6
python -m benchmark.wall_section_fit --source runs/oracle --out benchmark/reviews/wall-section-grid-01
python -m benchmark.wall_section_shelter
python -m benchmark.wall_section_exchange_oracle --workers 6 --out runs/exchanges.json
```

From the shared evaluator checkout replay each of `operating`, `transferred` and `anchored`, with `$method` set to the method name. Use its unchanged `runs/control` replay for comparison.

```powershell
python -m benchmark.wall_section_eval --candidate-root ../wall-section-grid-depth --candidate "engine.wall_section_grid:$method" --out "../wall-section-grid-depth/runs/$method" --workers 5
python -m benchmark.wall_section_eval --transfer-inputs ../framed-depth-transfer/runs/transfer-input --candidate-root ../wall-section-grid-depth --candidate "engine.wall_section_grid:$method" --out "../wall-section-grid-depth/runs/transfer-$method"
```

Back in the candidate checkout, score the frozen verifier on the camera-challenge panels and collect the archive.

```powershell
python -m benchmark.roof_height_transfer_panel --data ../roof-height-round/runs/panel-data --candidate-root . --candidate engine.wall_section_grid:transferred --out runs/panel-transferred
python -m benchmark.roof_height_transfer_panel --data ../roof-height-round/runs/elevated-unpacked --selection ../roof-height-round/runs/elevated-unpacked/elevated.json --candidate-root . --candidate engine.wall_section_grid:transferred --out runs/elevated-transferred
python -m benchmark.roof_height_native_panel --data ../roof-height-new-views/runs/new-height-input --sources ../roof-height-new-views/runs/new-height-input --candidate-root . --candidate engine.wall_section_grid:transferred --out runs/native-transferred
python -m benchmark.wall_section_results
```

[Section oracle and ablations](oracle.json) · [Family-separated fit](training.json) · [Occupied-wall exchanges](occupied.json) · [Shelter and gate oracle](shelter-oracle.json) · [Scored comparisons](results.json) · [Per-frame manifests and changed exports](comparisons.zip)
