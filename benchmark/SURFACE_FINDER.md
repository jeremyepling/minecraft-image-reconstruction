# From a screenshot to visible surfaces and block counts

**Surface Finder** answers “which pixels belong to the same flat, connected surface?” **Regions First** is the selected method. It uses two neural models. MoGe-2 estimates which direction each pixel faces. Depth Anything V2 estimates relative depth. The grouping code separates disconnected pieces and can split adjacent regions when depth changes. A wall keeps its actual shape, including visible openings. Relative depth alone supplies no distance in blocks.

**Surface Straightener** turns one visible surface toward the viewer using a four-corner perspective transform. Its output is a face-on image with transparency outside the surface. Pixels from a pillar gap or doorway stay out of the sample. A wall can remain a measured surface even when a sampling transform cannot be established.

**Block Counter** estimates how many block widths lie across and down the rectified sample. A repeated texture can reveal the spacing even when the first pixel is partway through a block. The position of the first grid line (phase) is a separate estimate. The count report distinguishes the dimensions of the visible surface from the span of the sampling chart. A masked wall with a doorway can have an overall width and height without every grid cell being occupied.

**Material Matcher** ranks vanilla block IDs by comparing visible texture against the Java 26.3 atlas. Once a cell's position is supported, the matcher can sample it individually. One large coplanar surface can contain several materials. Blocks with identical visible textures can have tied IDs.

## What came from B?

| Part of the previous experiment | What it did | Use in the current path |
| --- | --- | --- |
| MoGe-2 and orientation grouping | Proposed front, side and underside regions. B could use image color to subdivide a group. | The Surface Finder uses the numerical directions with relative depth and connected regions. |
| Rectangle fitting and filtering | Tried to turn each group into four corners. A region could disappear at this step or a rectangle could bridge an opening. | A four-corner chart can guide rectification. The saved pixel mask defines the outline and survives a failed fit. |
| Vanilla texture-spacing check | Rectified a four-corner chart, sought a repeating 16-texel grid and tested possible joint locations. | Available independently as the Block Counter's `legacy-repeat` baseline. This pass runs without B's grouping, rectangle detection, or 3D assembly. |
| Relative depth ordering | Compared nearby surface edges in camera space or sampled relative depth. | The new grouping uses relative depth immediately. Explicit `--surface-order` requests an extra ordering pass. |
| Camera-space lift and block exporter | Used measured spacing and shared edges to lift charts into 3D. The separate placement benchmark also needed a scaled camera to export occupied cells. | Surface finding, rectification, counting and texture matching can all finish with metric world placement still unresolved. |

On the 56 saved screenshots, Regions First matched **394 of 598** eligible surfaces and B matched **281**. A match requires a predicted mask and a recorded surface to cover at least half their combined area. Each can match only once. Eligible connected surfaces have at least four visible block faces and 24 scoring samples. The reports also grade covered openings, incorrectly assigned pixels and boundary distance. Regions First covered **3.9%** of the sampled opening area, compared with **29.2%** for B. These figures grade the Surface Finder. Correct block counts and material IDs have their own scores. [Per-picture surface results](reviews/mask-four-01/report.html)

## Use the stages on one image

With the pinned models in this checkout and the dependencies from `benchmark/suites/normal-requirements.txt`, this writes one mask per surface in `surfaces.json`. Each mask records its dimensions and half-open pixel runs `[row, start, end]`. `build-manifest.json` identifies the finder, model hashes and settings.

```powershell
python -m benchmark.normal_assets --download
python reconstruct.py build screenshot.png --version 26.3 --through surfaces --out runs/my-surfaces
```

Continue through counting to save the masked, straightened samples and measurements. The first command uses the named default counter. The second independently requests atlas matching on complete cells. A calibrated atlas directory can be given with `--atlas PATH`. Each run records the actual counter used.

```powershell
python reconstruct.py build screenshot.png --version 26.3 --through counts --out runs/my-counts
python reconstruct.py build screenshot.png --version 26.3 --through counts --material-match atlas --out runs/my-materials
```

Select a counter explicitly with `--block-counter legacy-repeat`, `masked-repeat` or `atlas-repeat`. For a comparable surface build using B's original options, use `--through surfaces --surface-outlines normal-regions`. To grade a new default surface prediction on a saved suite image, run `python -m benchmark.score_saved_planes --surfaces runs/my-surfaces/surfaces.json`. The scorer checks the screenshot hash and compares the masks to recorded geometry.
