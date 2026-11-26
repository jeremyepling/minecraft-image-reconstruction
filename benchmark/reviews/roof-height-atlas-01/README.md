# Estimate roof height before assigning individual blocks

The atlas selector projects a detected horizontal surface onto competing integer heights in the screenshot-derived camera frame. It compares the photographed pixels with calibrated vanilla sprites at each height. The score combines pixel correlation across the patch, its two spatial halves, and color agreement. The reported plane is the physical face height; the cell coordinate accounts for seeing a top or underside. On multilevel structures the main roof estimate is the largest confidently matched patch weighted by its pixel correlation. Other observed levels can still produce their own placements.

The independent support fallback requires a clear lead over the runner-up atlas height and two observed vertical support locations at least three grid units apart ending at the patch. These may be two portions of one wall. Every proposed block must individually match photographed texels, win native pixels in a joint render, and pass a silhouette check. Prior placements can disappear only when there is sufficient measured face evidence on their old sightlines.

| Setting | Development photos with an accepted main height | Accepted patches of 256 | Visible blocks gained | Added extras |
| --- | ---: | ---: | ---: | ---: |
| Strict atlas threshold | 35 / 72 | 65 | 9 | 0 |
| Atlas threshold selected for comparison | 46 / 72 | 98 | 20 | 0 |
| Two-support fallback, stricter | 48 / 72 | 102 | 27 | 0 |
| Two-support fallback, selected | 50 / 72 | 108 | 39 | 0 |

The baseline is the ordinary strict roof-strip build at `54c9fee`. Candidate code was frozen at `567709e` for atlas-only, `0e9d174` for the uppermost-existing-layer control and `c534b41` for support corroboration. The two settings of each method were compared on the 72 development images. No previously correct cells were lost in these four development runs. [Per-view development scores](development.json)

On the 104 previously reviewed fixed structure views the atlas-only method reached 90.17% relative geometry F1, with 25 visible cells gained, zero extras added and one extra removed. Support corroboration reached 90.23%, with 59 visible cells gained, 12 extras added and two removed. One new extra fell on a hidden source position. The starting score was 90.10%. On four prior loading dock and switchback photos both variants fell from 83.64% to 83.47% due to three extra placements. On twenty alternate views support corroboration recovered 26 visible positions and added zero extras. Height evidence can therefore help placement but has errors when transferred to a different structure.

## Canopy 06 left

The atlas-ranked physical height was six and the lower coordinate of its visible roof blocks was five. The patch fit had a native pixel correlation of 0.352 and its score led the next height by 0.293. This did not meet the atlas-only threshold. Corroborating visible support locations spanned four grid units, admitting this height in the support variant.

All 52 roof cells missing from the previous build at its two-post alignment receive at least six correctly intersecting mask pixels at this chosen height. Twenty matched the sprite strongly enough on their own native pixels. All twenty would show an observed horizontal face and all twenty pass the silhouette test in the initial joint render. The final build adds none on this photo.

A scoring-only counterfactual inspected the original blocks that would completely disappear if we kept the individually viable additions. There were 27. All 27 were incorrect under the common post alignment. Of their 3,322 original pixel rays, 918 hit horizontal faces in the counterfactual and 2,404 hit vertical faces. The selector verifies photographed horizontal texels; these vertical pixels have no separately verified material match in this method. Their disappearance fails its guard for old placements. These counts use the photographed roof for validation and the source world only for retrospective correctness. The counterfactual is not an exported build. [Native-pixel trace across all 18 canopy views](trace.json)

## Replay

From the `experiment/roof-height-atlas` checkout, with the prior round's production packs in `../roof-span-round/runs/production` and the calibrated Java 26.3 atlas available:

```powershell
python -m benchmark.roof_height_atlas_dev --mode supported_wide --out runs/supported_wide --workers 4
python -m benchmark.roof_height_atlas_trace --prior ../roof-span-round/runs/production --out benchmark/reviews/roof-height-atlas-01/trace.json
python -m unittest discover -s tests -p test_roof_height_atlas.py
```

The shared roof-height review uses the adapters `engine.roof_height_atlas:recover`, `engine.roof_height_atlas:tallest`, and `engine.roof_height_atlas:supported` on the same frozen captures. All geometry changes on this experiment branch are opt-in.
