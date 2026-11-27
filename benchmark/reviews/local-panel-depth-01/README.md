# Moving a single visible wall panel

The Java 26.3 verifier moved eleven incorrectly placed blocks to eleven visible reference positions on four of the fixed 104 photos, eliminating eleven extras including two hidden source cells. Pooled relative block F1 rose from **90.4142% to 90.4627%** against the ordinary placer at `bec7c94`. Both scores grant one translation to the entire build. The other 100 fixed photos tied. All 72 development, 20 alternate, four new, four shelter/gate, four native canopy and eighteen additional side/elevated photos retained their original exports. Nineteen of the 22 supplementary canopy photos exercised camera fallback.

The fixed set has been used to inspect previous and current failures. The final conflict-resolution rule was developed after inspecting Dev Relief. The unchanged transfer sets establish observed compatibility; they provide no additional examples of a successful depth correction. Running the verifier after the separate rear-pillar correction also preserved all five changed pillar packs.

[Original and shifted panel on Dev Relief](dev-relief.jpg) · [Scores and case results](results.json) · [Frame manifests, candidate options and changed exports](scores.zip)

## Why the photographed return matters

The previous Dev Relief build placed two blocks correctly oriented but one block off along each horizontal axis. The wrong tiles partly covered the same screenshot region as the visible panel. Their replacement changes the side boundary, aligns the front and return textures better, and reduces projected silhouette errors by 27 and normal-direction errors by eight on a two-pixel sampling grid. Both accepted destinations match a vanilla sprite at their own calibrated depth. One old face also matched an atlas texture. Testing each source and destination individually against the photograph supported the shift. This view rises from 380 to 382 correct positions of 480 and drops from 48 extras to 46. Both removed cells were hidden source-world positions.

The other successful decisions recovered five blocks on a masonry Shopfront, two on Twinspire and two on Portico. The plaster Shopfront in the weakest-layout viewer still ties, as do the shown Facade, Canopy 04, Colonnade and the separate Relief 00. The method addresses occupied wall depth; the Colonnade roof lies outside its proposal family.

## Geometry and decisions

1. Project the actual current visible build with its estimated camera. Within each detected mask, divide visible front tiles by plane and connected component. A separate pass splits on observed source color. Per-frame candidates are 2–12 contiguous tiles, with up to 20 patches ranked using visible perpendicular faces. No source labels participate.
2. Translate a patch one block in either direction along its normal. Include a second integer translation when preserving its photographed center also suggests a lateral adjustment. Surrounding cells stay fixed. Recompute the nearest block on every changed ray, including rays that now hit a third or deeper block. Require visible mask support for every new cell and continued visibility of surrounding cells.
3. Rank the remaining choices by projected silhouette and face-axis improvement and the change in atlas-textured color residual on alternating image patches held out from the fit. Limit each picture to one coherent exchange.
4. Check the chosen positions at native resolution. For every new cell, at least one visible face must independently match the calibrated atlas and discriminate its plane from the neighboring depths. A competing match on an old cell triggers another check. Every displaced source and every destination must improve its own shape evidence, and every destination must reduce its held-out texture residual. A native joint visibility pass protects other visible blocks. Missing or unverified atlas data and unresolved cameras leave the preceding result intact.

## What the alternatives cost

| Selector | Fixed correct gained / lost | Fixed extras added / removed | Development correct gained / lost | Development extras added / removed | New-view extras added / removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coherent panel using shape and held-out RGB | 11 / 0 | 0 / 11 | 6 / 3 | 6 / 9 | 5 / 5 |
| Insist on local shape and texture gain for every block | 2 / 0 | 0 / 2 | 0 / 0 | 6 / 6 | 0 / 0 |
| Keep original cells wherever rays still support them | 6 / 0 | 0 / 0 | 6 / 2 | 6 / 1 | 5 / 0 |
| Require destination atlas matches, veto any source match | 9 / 0 | 0 / 9 | 0 / 0 | 0 / 0 | 0 / 0 |
| **Destination atlas matches, local test for disputed moves** | **11 / 0** | **0 / 11** | **0 / 0** | **0 / 0** | **0 / 0** |

The first rule lost three genuine development blocks on Arcbridge and moved already-incorrect cells to other wrong locations on Watchpost and a new Switchback. The native depth comparisons blocked those changes. Keeping original tiles whenever they retained a visible projection added five false Switchback cells. This is evidence that a foreground mask can support several plausible depths.

With the visible reference allowed only for scoring, a one-panel chooser could correct **92 blocks on 21 fixed photos** from 1,559 candidate exchanges. This counts only complete exchanges where every source is wrong, every destination is correct and the native visibility check accepts the whole move at the control's common translation. The same test found 29 blocks on development, 19 on alternate angles and zero on the four new images. Dev Relief had a five-block possible correction, and Relief 00 had a two-block possible correction. The remaining challenge is identifying the correct local interpretation when both positions match the photographed appearance.

## Replay

The original proposal and image-only rule were frozen at `f7572f1`. The endpoint guard, retained-source variant and strict native-depth check were frozen and tested successively. The engine implementation of the final verifier is `cae1a5b`. Use the frozen screenshot releases, the calibrated atlas, and the previous verified control packs in the sibling `wall-section-round` checkout. The first command exports predictions before the oracle inspects the scoring labels. The last command can also take `--transfer-inputs ../framed-depth-transfer/runs/transfer-input`.

```powershell
python -m benchmark.local_panel_trial --out runs/panel-proposals --workers 5
python -m benchmark.wall_section_eval --predictions runs/panel-proposals --out runs/replayed-basic --workers 5
python -m benchmark.wall_section_eval --candidate benchmark.local_panel_atlas:ambiguous --out runs/verified --workers 5
python -m unittest tests.test_local_panel_depth
```

The `local_panel_guard:recover`, `local_panel_retained:recover` and `local_panel_atlas:recover` adapters in the benchmark package replay the other three decisions. `benchmark.local_panel_review` verifies the reports and packs and renders the Dev Relief comparison. Measured incremental process CPU over three in-memory trials per photo ranged from 0.25–0.30 s on Colonnade, 0.58–0.64 s on Facade, 1.61–1.83 s on masonry Shopfront and 1.67–2.39 s on Dev Relief. These timings exclude loading the input photo and upstream placement.
