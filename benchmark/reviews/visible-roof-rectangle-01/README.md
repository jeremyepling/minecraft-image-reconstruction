# Can two visible rims determine a flat roof?

On 104 fixed full-structure screenshots, two perpendicular rim lengths proposed 1,670 missing block locations. Of those locations, 709 were visible reference blocks, 837 were hidden blocks and 124 were elsewhere. Automatically filling the rectangles would change relative geometry F1 from **90.30% to 90.11%**. A stricter rule accepting cells with photographic evidence on one top surface shared by both rims recovered **24 visible blocks and added zero extras**, reaching **90.36%**. [Three source photographs with edge and placement overlays](three-views.jpg) · [Per-view rectangle audit](box-audit.json)

## What was measured

Each experiment started from the same `1c2f77d` build, surfaces, counts, atlas and image-only camera. The candidate finds a camera-facing corner with two adjoining vertical faces at the same top course. It follows each rim to the final contiguous measured block. Their lengths bound a flat rectangle. Missing cells inside it are proposals. The reference world is opened afterward for evaluation. The automatic-fill row below is a set-composition diagnostic and produced no build.

| Method | Fixed relative F1 | Visible gained / lost | Extras added (hidden) | Alternate 20 F1 | Shelter and gate 4 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous placement | 90.298% | 0 / 0 | 0 | 88.485% | 83.051% |
| All cells inside the measured rim lengths, hypothetical | 90.107% | 709 / 0 | 961 (837) | See audit | — |
| Frontmost horizontal mask pixels inside the silhouette | 90.510% | 95 / 0 | 9 (9) | 88.900% | 83.353% |
| Horizontal mask plus per-block atlas texture and depth margin | 90.330% | 15 / 0 | 2 (2) | 88.527% | 83.051% |
| Require at least four previously uncovered top pixels and at least 85% uncovered | 90.406% | 48 / 0 | 4 (4) | 88.624% | 83.142% |
| Also require both observed rims to touch the same visible top patch | **90.356%** | **24 / 0** | **0** | **88.513%** | **83.192%** |

All four image-only variants retained 83.580% on the separate four loading-dock/switchback views. Every emitted cell needs its own frontmost horizontal pixels after all proposed cells are rendered together. At least 88% of its exposed horizontal face must agree with the measured orientation. New geometry must preserve the existing silhouette to within one native pixel and preserve at least one visible ray for every previously visible block. In the most conservative rule, the same detected horizontal patch must cover two existing upper edge cells along each of the two perpendicular rims and the new cell itself.

The first mask-based rule improved 21 fixed photographs. Of its 104 additions, 103 had at most three pixels of effective thickness (horizontal witness count divided by the longer side of its native bounding box). Nine additions in the colonnade lay at hidden reference positions. An atlas check reduced the number of hidden additions there to two. Requiring previously unclaimed pixels still added four hidden cells there, one on the shelter/gate group and two on the independent low masonry awning. A promising development result was insufficient to resolve this ambiguity.

**Recommendation.** Advance the shared-top-surface rule to integration testing with the other placement changes. Keep the independent-mask, atlas-only and uncovered-pixel variants experimental. The shared-surface rule `a465f08` gained six visible blocks with no extras on 72 development images before the full replay. On 104 fixed photographs it improved six views and tied on 98. It gained 16 blocks in the Cloister, five across Galleries and the two Portico views, one in the Courtyard and two in Canopy 09. It gained two visible blocks on the 20 alternate views and two on the four shelter/gate views. All 18 measured opening widths stayed correct, all 243 clear rays stayed open and anchored roof cells rose from 342 to 344 on the fixed canopies and 311 to 313 on alternate views. All 22 additional original-camera exports matched the previous geometry. The four independent low/tall roof images stayed at 86.46% and the 18 missing camera estimates remained in the denominator. The fixed buildings have been examined in earlier trials. [Full paired results](results.json) · [Scores, hashes and changed exports for all 204](comparisons.zip) · [Independent 22-view results and per-frame records](new-views.zip)

### Open Canopy 00

The preceding build recovered 27 of 50 visible positions with zero extras (70.13% F1). A single measured corner bounded 30 missing candidate locations. The full rectangle would cover the 23 missed visible blocks and seven hidden source cells. In the ordinary estimated-camera projection, 24 of those cells expose a top face somewhere, covering 1,047 native pixels together. Of those pixels, 905 intersect the detected building silhouette but zero intersect a detected horizontal surface. Every witnessed variant therefore abstained. A future image-only test could look for native texture and lighting at the upper silhouette when the predicted horizontal mask is absent. [Initial source and stage overlays](../roof-height-01/worst/structure-01.jpg)

## Replay

Run the shared `benchmark.facade_placement_eval` evaluator at `2e4d3f9` or a descendant using the archived calibration and fixed screenshots. Pass a distinct destination for every method. The 72 development and 104 fixed photographs have separate scores in the manifest. The alternate 20, prior four loading-dock/switchback and four shelter/gate photographs are also separate.

```powershell
python -m benchmark.facade_placement_eval --candidate-root ../visible-roof-rectangle --candidate engine.visible_roof_rectangle:contiguous --out ../visible-roof-rectangle/runs/contiguous --workers 6
python -m benchmark.facade_placement_eval --candidate-root ../visible-roof-rectangle --candidate engine.visible_roof_rectangle:contiguous --transfer-inputs ../framed-depth-transfer/runs/transfer-input --out ../visible-roof-rectangle/runs/transfer-contiguous
```

From the rectangle checkout the geometric audit can be replayed with `python -m benchmark.roof_rectangle_audit --control ../roof-height-round/runs/production --out benchmark/reviews/visible-roof-rectangle-01/box-audit.json`. Use the [native and elevated panel recipes](../roof-height-01/panel/README.md) and the [four newer roofs](../roof-height-new-views/README.md) with the same candidate. All scoring uses one common translation per build. [Proof of the verified current control on the panels](../roof-height-01/new-panel-verification.json)
