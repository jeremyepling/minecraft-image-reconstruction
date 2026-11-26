# From a screenshot to visible surfaces and block counts

The ordinary Java 26.3 build uses **Regions First with split-only zoom** for its **Surface Finder**. It found **516 of 598** eligible surface appearances across 56 views of eight Minecraft buildings. The **Block Counter** got both width and height right for 80 surfaces. [See the prediction and reference outlines](reviews/crop-split-01/comparison.jpg) · [Scores for each picture](reviews/crop-split-01/pictures.md)

## What the Surface Finder sees

1. **Group visible pixels.** MoGe-2 estimates which way the photographed surfaces face. Depth Anything V2 estimates relative depth, which helps distinguish two parallel walls. Regions First keeps disconnected regions separate and preserves the holes in their masks.
2. **Refine walls and thin undersides.** The Wall Separator checks image contrast and relative depth along proposed boundaries. ScaleLSD v2 supplies lines and junctions, M-LSD Tiny supplies line evidence for undersides, and MobileSAM proposes wall outlines. A single outline can divide a wall when the depth samples support a step across it.
3. **Find missed planes.** Pixel-wise Planarity checks the full screenshot for a nearby unclaimed surface or evidence of a split within an existing wall. Splits stay within the parent mask. This step raised the score from 501 to 513, kept every earlier match and covered the same 601 of 16,630 sampled opening pixels. [Method and measured tradeoffs](AFTERPASS_MODELS.md)
4. **Examine walls up close.** The same model inspects nine overlapping crops. A crop can divide an existing surface when two regions have enough support. The resulting pieces exactly cover the original mask, preserving holes. This recovered a narrow Shopfront face and a Workshop face seen in two palettes. [Split-only zoom comparison](CROP_SPLIT.md)

An eligible reference surface is a connected patch with at least four visible block faces and 24 scoring samples. It matches one prediction when their overlap is at least half their combined area. The same wall can count again when seen in another picture. Across the suite, 82 eligible surfaces were missed. There were 204 extra proposals after smaller visible surfaces were accounted for. [Run manifest and scores](reviews/crop-split-01/manifest.json)

## Inspect one screenshot

```powershell
.\Invoke-WallRecovery.ps1 .\screenshot.png -Out .\runs\my-build
```

| Output | What to look for |
| --- | --- |
| `surfaces.json` and `surface-crops/` | Check the actual mask around a wall, recess or opening. Each surface records pixel runs and outer and inner perimeter rings. A transparent crop shows its owned pixels. An RGB crop includes up to eight neighboring source pixels per side so you can check an edge in context. |
| `samples/` | The **Surface Straightener** uses a four-corner chart to produce a face-on sample where possible. Gaps remain masked out. A region without a chart still appears in `surfaces.json`. |
| `counts.json` | The **Block Counter** looks for repeated texture inside the straightened mask, then estimates the grid offset and block span. An opening can lie within the measured width and height. The report records unresolved measurements. |
| `build-manifest.json` | The stages record `mask-first`, `plane-recovery-crop-splits` and `masked-repeat`, plus the settings and hashes of the six model checkpoints. |

Pass `-Through surfaces` to the wrapper to stop after the outlines. In the configured Python environment, `build --through counts --material-match atlas` also asks the **Material Matcher** to rank vanilla block IDs for complete cells. The atlas can name multiple materials on one coplanar wall. On a saved suite image, use the scorer below to check the predicted masks against the reference geometry. [Model setup and Python commands](WALL_RECOVERY_BUILD.md)

```console
python -m benchmark.score_saved_planes --surfaces runs/my-build/surfaces.json --image PATH_TO_ORIGINAL_SUITE_PNG
```

## Compare to older finders

Older sheets call the **Rectangle Finder** “B.” It grouped pixels by direction and color, then required a four-corner fit. Rectangles could span openings and some visible patches failed the fit. Regions First keeps the mask as the outline and uses four corners only when a surface needs a sampling chart. In the original comparison, Regions First matched **394 of 598** surfaces and covered **3.9%** of the sampled openings. The Rectangle Finder matched **281** and covered **29.2%**. Its texture-spacing check was also tested independently against masked repetition. [Original four-method review](reviews/mask-four-01/report.html) · [Block Counter comparison](COUNT_ROUND.md)

Use `.\Invoke-SurfaceRound.ps1 -Round crop-split` to compare the current finder with its 513-surface predecessor. The replay uses saved screenshots and numerical model evidence, and keeps the world geometry on the scoring side. Earlier wall splits and counting comparisons have their own benchmark runs. [Separation replay](SEPARATION_ROUND.md) · [Single-outline replay](SINGLE_WALL_DEPTH.md) · [History](RECONSTRUCTION_LINEAGE.md)
