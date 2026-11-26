# Camera-supported visible block placement

`Invoke-WallRecovery.ps1 image.png -Out runs/build` runs the exterior pipeline through placement. The Python entry point is `python reconstruct.py build image.png --version 26.3 --through placement --out runs/build`. The exported structure uses smooth stone to show geometry. Material choices from the optional atlas, exposure and texture matchers are recorded in the counts report.

The pavement seam estimator must recover a camera with scale in blocks. The placer keeps the strict grid cells from measured surfaces and fits additional planes using surface direction, floor contact and relative depth. A proposed cell requires at least four samples and half its projected face inside a visible surface mask. After combining proposals it checks that each added cell is still visible within a supporting mask. Coordinates in the datapack are local to the placement function. An uncertain camera produces zero placements and records its reason in the build manifest.

## Exact placement checks

Before export, the placer projects a neighboring block's front and side and compares the rectified textures against the proposed second front. Strong side evidence can establish a shared edge. The decision can extend along consecutive rows when the same owner is confirmed on the run, the exposed floor agrees, and removing the column preserves the observed silhouette. Strict placements and blocks used to explain another face are retained. [Seam experiments and source-view comparison](SEAM_OWNERSHIP.md)

Each screenshot has its own visible target coordinates. The evaluator decodes the exported Minecraft structures and uses the saved camera to align the local compass and origin for scoring. Hidden blocks are excluded from the target. A placed block at any other coordinate counts as an extra.

| Saved screenshots | Visible target blocks | Strict correct | Strict extras | New correct | New extras |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original exterior suite, 56 views | 10,626 | 344 | 18 | 2,024 | 621 |
| Additional camera-supported views, 6 layouts × 2 angles | 3,780 | 395 | 5 | 2,753 | 527 |

On the original suite 16 cameras supported metric placement. The strict exporter produced blocks in 9 views and the new exporter in all 16. The other 40 views abstained. On the 12 additional views the new rule retained every strict placement and gained correct locations on every view. The two Corner views added 23 correct blocks and 205 extras. These layouts remain useful tests of depth estimation. Across 70 additional fixture views screened for camera support, 12 qualified for this comparison. Separately, all 96 pilot screenshots on light gray concrete had uncertain cameras. [Additional view scores and input fingerprints](reviews/supported-placement-01/holdout.json)

## Replay the original suite

Use the published Visible Builds v1 regression frames and the pinned sky-gap-01 masks. The replay checks the input digests, scores all 56 screenshots and compares both the exported and registered coordinate sets to the frozen supported-camera experiment.

```powershell
python -m benchmark.supported_placement_round --frames runs/visible-builds-v1/data/regression/frames --out runs/supported-placement-check --verify-experiment
```
