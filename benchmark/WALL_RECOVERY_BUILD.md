# Build surfaces from one screenshot with Wall Recovery

The [product intent](../PRODUCT_INTENT.md) is to reconstruct visible evidence only. A missed visible face is a recovery target; an unseen face is not. Preserve openings and leave occluded regions, hidden backs and interiors unknown. Neither a plausible shape nor a learned prior justifies adding unsupported cells to the exported build. This is the scope for future implementation and evaluation; it does not claim that every existing export path already enforces it.

From this checkout on the Minecraft PC:

```powershell
.\Invoke-WallRecovery.ps1 .\snapshot.png -Out .\runs\my-build
```

This command runs surface finding, counting and visible block placement on one Java 26.3 screenshot. Use `-Through surfaces` or `-Through counts` for the intermediate stages. The build writes visible masks to `surfaces.json`, estimates to `counts.json`, and camera status and placement counts to `build-manifest.json`. The default also writes local smooth-stone coordinates to `placements.json` and their Minecraft datapack to `reconstruction.zip`. [Placement behavior and evaluation](SUPPORTED_PLACEMENT.md)

The Wall Separator starts with MoGe-2 directions and Depth Anything V2 depth. Its masks supply MobileSAM point prompts. ScaleLSD identifies junction-supported cuts, M-LSD Tiny finds thin undersides, and MobileSAM can partition a wall when the depth evidence supports a step. Pixel-wise Planarity checks the full screenshot for an adjacent missing surface or two distinct planes inside one wall. Nine overlapping crops give that model a closer view. Each accepted crop split keeps exactly the original surface area and holes. The Surface Straightener and Block Counter receive those masks. The build manifest calls the selected method `plane-recovery-crop-splits`. [Results on 56 views](CROP_SPLIT.md)

### Model environments

On this PC, the script locates the existing inference installations in the reconstruction worktrees. Other installations can pass `-Python` for the NumPy/Pillow/ONNX/LiteRT environment, and `-ModelPython`, `-ScaleSource`, `-ScaleCheckpoint` and `-MobileSource` for the PyTorch environment and official model checkouts. The MobileSAM checkout must contain `weights/mobile_sam.pt`. The wrapper supports `-Device cuda` or `-Device cpu`. The earlier five-model verification used CUDA for ScaleLSD and MobileSAM and CPU inference for MoGe-2, Depth Anything V2 and M-LSD Tiny.

The same build is available through the normal CLI:

```powershell
python reconstruct.py build .\snapshot.png --version 26.3 --through counts --out .\runs\my-build
```

For direct CLI use, configure `MCIR_RECOVERY_PYTHON`, `MCIR_SCALELSD_SOURCE`, `MCIR_SCALELSD_CHECKPOINT` and `MCIR_MOBILESAM_SOURCE`, or put the model sources and checkpoints in `.deps`. The ONNX and LiteRT paths follow the existing `MINECRAFT_MOGE_ONNX`, `MINECRAFT_DEPTH_ONNX` and `MINECRAFT_MLSD_TFLITE` settings. The default build method stays in `engine/surface_defaults.json`.

For Pixel-wise Planarity, the wrapper accepts `-PlanaritySource`, `-PlanarityCheckpoint` and `-PlanarityUtils3D`. Their environment variables are `MCIR_PLANARITY_SOURCE`, `MCIR_PLANARITY_CHECKPOINT` and `MCIR_PLANARITY_UTILS3D`. The Torch environment also needs the matching scipy and connected-components-3d packages. Set `MCIR_PLANARITY_PYTHONPATH` to their directory when installed separately. Fresh inference checks the upstream Git revisions and checkpoint digest. Split-only zoom computes whole-image and crop fields together. For a cached replay on the 56 suite pictures, run `.\Invoke-SurfaceRound.ps1 -Round crop-split`. [Pinned sources and versions](reviews/afterpass-fields-01/planarity/manifest.json) · [Authors’ checkpoint](https://huggingface.co/alpayozkan/pxwplanar-moge2-planarity)

### Verify the default from a renamed screenshot

```powershell
New-Item -ItemType Directory -Force .\runs\plane-proof | Out-Null
Copy-Item .\benchmark\data\camera-gray-concrete-v1\captures\holdout-courtyard-masonry-high-right.png .\runs\plane-proof\source.png
.\Invoke-WallRecovery.ps1 .\runs\plane-proof\source.png -Out .\runs\plane-proof\build
python -m benchmark.verify_recovery_build --build runs/plane-proof/build --case light_gray_concrete/holdout-courtyard-masonry-high-right
```

On Courtyard, check the final `neighbor-supported-strip` entry in the build manifest. The beam should lose 208 stray pixels before straightening. [Default verification](reviews/default-strip-neighbor-01/)

The earlier sky-filtered Workshop build passed 13 regions to the Block Counter, down from 28 before the filter. Every saved region and block-count result matched the offline replay. The separate verification command reads the finished predictions and recorded building. [Earlier Workshop build](reviews/sky-entry-01/fresh-workshop.json) · [Earlier split-only zoom verification](reviews/default-zoom-workshop-01.json)

### Earlier five-model result

The five-model finder matched 19 of 25 surfaces on the Galleries screenshot and sent 30 regions to the Block Counter. Its original verification contains the exact masks and model hashes. Run `.\Invoke-SurfaceRound.ps1 -Round afterpass` to compare its saved suite-wide predictions against the selected six-model finder. [Five-model verification](reviews/wall-recovery-entry-01.json) · [Suite comparison](AFTERPASS_MODELS.md)
