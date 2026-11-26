# How a screenshot becomes a Minecraft build

The [product intent](PRODUCT_INTENT.md) is visible-only reconstruction for both the current pipeline and proposed learned components. Only blocks supported by visible faces in the supplied input belong in the output. Unobserved backs, interiors and occluded regions stay unknown; geometry consistency must not become a hidden-area completion step.

## Find surfaces and count blocks

The ordinary exterior command on this PC takes one Java 26.3 screenshot. It runs the **Surface Finder**, **Surface Straightener** and **Block Counter** and saves their intermediate results. The selected Surface Finder matched **517 of 598** visible surface appearances across 56 frozen views. The counter got both block dimensions right for **94 of 598**. These scores measure different stages. [Surface comparison](benchmark/reviews/sky-01/pictures.md) · [Crop review](benchmark/CROP_REVIEW.md)

```powershell
.\Invoke-WallRecovery.ps1 .\wall.png -Out .\runs\my-build
```

| Stage | What runs | Where to inspect it |
| --- | --- | --- |
| Surface Finder | **Regions First** groups connected pixels using MoGe-2 surface directions and Depth Anything V2 relative depth. ScaleLSD v2 supplies junctions, M-LSD Tiny helps identify thin undersides and MobileSAM proposes wall outlines. Pixel-wise Planarity checks for nearby missing surfaces and supported splits. | `surfaces.json` contains the masks, hole boundaries and optional sampling corners. `surface-crops/` shows the visible pixels and their surroundings. |
| Surface Straightener | A four-corner chart turns a surface face-on for sampling. Its pixel mask excludes gaps and neighboring surfaces. Regions without a usable chart still have outlines. | `samples/` holds rectified samples from the counting pass. |
| Block Counter | **Masked repetition** measures how often a texture repeats across and down the straightened surface. A supported width or height can be recorded independently. | `counts.json` records the estimated grid and dimensions for each surface. |
| Material Matcher | When requested with `--material-match atlas` on a counts build, compare complete, supported cell footprints with placeable textures from the versioned atlas. | Inspect the cell-level material results in `counts.json`. |

The Surface Finder’s build IDs are `mask-first` for grouping and `plane-recovery-crop-splits` for refinement. After the full-image pass, closer views can split a region while preserving its footprint. The counter’s ID is `masked-repeat`. Each run writes its actual stages, settings, input digest and checkpoint hashes to `build-manifest.json`. The winner is selected on the fixed suite. The previous methods are available through benchmark replays. [Single-image setup](benchmark/WALL_RECOVERY_BUILD.md) · [Stage guide](benchmark/SURFACE_FINDER.md) · [Method history](benchmark/RECONSTRUCTION_LINEAGE.md)

## Verify on the saved pictures

```powershell
.\Invoke-SurfaceRound.ps1 -Round sky
```

The replay applies the selected rules to saved numerical model outputs, runs the same counter and grades the resulting masks against the recorded geometry. The 56 views cover eight buildings captured in vanilla Java 26.3 at clear noon. Settings were chosen on Shopfront and Wing. The other six buildings measure performance on previously reviewed layouts. Run `build --through camera` with an image, version and output directory for the independent camera and projected-grid estimate. [Replay manifest](benchmark/reviews/sky-01/manifest.json) · [Camera test](benchmark/CAMERA_ROUND.md)

## Export blocks and check placement

Assembly may place a full Minecraft cube for a block evidenced by a visible face. It must not add further cells to close a shell, fill an unseen continuation or manufacture concealed supports. An omitted unknown cell is not an instruction to place air in an existing world. The facade builder's legacy `--depth` shell generation is outside this product intent; its existence is not evidence that hidden reconstruction is a goal.

The assembly command uses the facade builder. It measures a projected block grid or accepts one saved for the same image. It ranks atlas textures for those cells, resolves conflicting placements and writes a Minecraft datapack. The six-model surface run currently ends at surface-local measurements. For a measured link between those surfaces and world coordinates, see the Block Counter’s separate export experiment. [Count and placement results](benchmark/COUNT_ROUND.md)

```console
python reconstruct.py analyze wall.png --out runs/grid
python reconstruct.py build wall.png --geometry runs/grid/geometry.json --atlas runs/26.3/atlas --out runs/export
python reconstruct.py verify-build runs/export --version 26.3 --out runs/check
```

| Step | Input and result |
| --- | --- |
| Measure | `analyze` proposes wall outlines, estimates repeating block seams, and projects each occupied cell. `geometry.json` saves the source quad, facing and destination coordinate. `dimensions.json` records observed extents. A supplied `--scene` can provide planes, counts, origins, openings and landmarks. |
| Assemble | A plain `build` with `--atlas` measures its own grid. With `--geometry`, it checks the image fingerprint and reuses the saved quads and coordinates. Each face is compared with textures of the same facing. A calibrated `--library` supplies rendered exemplars. `matches.json`, `placements.json` and `reconstruction.zip` record the result. |
| Verify placement | `verify-build` opens a fresh Minecraft world, places the exported blocks, reads them back and captures a frame. A placement that matches the export can still differ from the source. The geometry and material benchmarks score that separately. |

For atlas creation and fresh in-game reference captures, see the calibration guide. For earlier tests of alternative geometry, lighting, opening detection and voxel occupancy, see the archived comparisons. [Calibration](calibration/README.md) · [Material results](benchmark/COLOR.md) · [Recess depths](benchmark/RECESSES.md) · [Voxel fits](benchmark/VOXEL_FIT.md)
