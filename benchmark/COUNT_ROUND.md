# Counting blocks on a visible surface

The **Surface Finder** uses Regions First to outline a connected wall, roof or underside. The **Surface Straightener** turns its visible pixels face-on and masks openings. We compared three **Block Counters** on those identical cutouts. A separate **Material Matcher** then tried to name each individual block whose four corners the counter could locate. [What B used to do](SURFACE_FINDER.md)

## Which counter worked?

The selected counter looks for repeated texture inside the mask. It can measure one direction independently, use the shape of the straightened surface to check the other direction, and leave a dimension unknown. Across all 56 images it got both dimensions exactly right on **57 of 598** surface appearances, up from **12** with the texture counter taken from B. Using the atlas to infer spacing directly also got **12** right. The texture counter improved the exact count on every building, including on pictures with the lightly marked concrete floor. [Eight-building picture sheet](reviews/count-01/comparison.jpg) · [Results for each picture](reviews/count-01/report.html)

| Counter on the same outlines | Both dimensions right | Gave a wrong pair | No complete pair on a reference surface |
| --- | ---: | ---: | ---: |
| B’s texture counter | 12 | 7 | 579 |
| Repeated texture inside the mask | 57 | 22 | 519 |
| Atlas-assisted spacing | 12 | 8 | 578 |

Each picture counts separately. Shopfront’s 120 surface appearances are summed over its views. On the high-right masonry wing the masked counter read a broad wall as **13 across by 8 down**, exactly right. On the high-right masonry shopfront with the concrete floor it read **14 by 7** on a surface that touches **15 columns and 7 rows**. Even with a known surface and camera, rounding the span of visible pixels disagrees with the count of touched block rows or columns on 166 of 598 appearances. Partly visible blocks and the four-pixel reference raster contribute. [Reference-only extent check](reviews/count-01/extent-diagnostic.json)

## Does the atlas help identify blocks?

With recorded corners, the matcher identified **476 of 672** independently sampled block faces on its first guess. On the 58 faces for which the new counter recovered a complete footprint, we also tried both sets of corners on the exact same blocks. It got **58 of 58** with the recorded corners and **5 of 58** with predicted corners. All 58 were polished andesite. Iron block was the most frequent wrong guess. The broader sample measures the matcher across materials. The paired sample shows how sensitive these polished-andesite results were to the crop. [One block sampled both ways](reviews/count-01/sampling-closeup.png) · [Every match](reviews/count-01/materials.json)

## Does better counting improve the exported build?

The 3D exporter wrote **168 correct visible coordinates** with the selected counter, up from **81** with B’s counter, among 10,626 possible across the suite. It added zero incorrect coordinates with either counter. The gain was concentrated in the two development buildings (31 to 134). On the six other buildings, the export dropped from 50 correct to 34. The new method produced blocks on four images, down from five. Only 16 images had a camera estimate with supported block scale. Surface-local spacing can succeed while camera scale, grid origin or relative depth is unresolved.

Keep Regions First for outlines and masked repetition for surface measurements. Keep atlas identification as a separate pass. Use the existing 3D export as a diagnostic until it improves across buildings. The next measurable target is joint location: check how far the predicted block corners miss the recorded corners and how often that changes the ID. [Readback scores](reviews/count-01/placed-geometry.json) · [Exported smooth-stone datapacks](reviews/count-01/geometry-packs.zip)

## Replay this round

From this checkout on Windows with the benchmark Python dependencies and the vanilla Java 26.3 client installed, run:

```powershell
.\Invoke-BlockCountRound.ps1
```

The wrapper writes to a fresh directory under `runs`. An optional `-Atlas PATH` selects an existing calibrated atlas. The runner verifies each screenshot, the frozen surface proposals, the shared reference archive and the versioned atlas. The normal predictions receive one image and the game version. Saved model-derived masks substitute for repeated MoGe-2 and Depth Anything V2 inference in the suite replay. Only the scorer and labeled diagnostics read the known camera, block states and face corners. Geometry verification decodes the exported Minecraft structures offline.

Settings were frozen on 20 views of shopfront and wing before evaluating the other 36 views. Those six layouts have been reviewed in earlier rounds, so they measure regression on familiar buildings. Selection rewards a correct pair and subtracts a wrong pair, giving each development building equal weight. [Run ID, revisions and settings](reviews/count-01/manifest.json) · [Development decision](reviews/count-01/selection.json) · [Cases, counts and timing](reviews/count-01/cases.json)

An independent run reproduced the per-surface counts, all material rankings and the coordinates and states inside all 112 datapacks. The published replay also matched the separate reference-span calculation. [Independent replay check](reviews/count-01-replay.json)
