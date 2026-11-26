# Build surfaces from one screenshot with Wall Recovery

From this checkout on the Minecraft PC:

```powershell
.\Invoke-WallRecovery.ps1 .\snapshot.png -Out .\runs\my-build
```

This command uses the default Java 26.3 Surface Finder and Block Counter. The input is one ordinary screenshot. Use `-Through surfaces` to stop after surface finding. The build writes `surfaces.json` with visible masks, surface crops, `counts.json` with supported dimensions and `build-manifest.json` with the methods, settings and checkpoint hashes.

The Wall Separator starts with MoGe-2 directions and Depth Anything V2 depth. Its masks supply MobileSAM point prompts. ScaleLSD identifies junction-supported cuts, M-LSD Tiny finds thin undersides, and MobileSAM can partition a wall when the depth evidence supports a step. Pixel-wise Planarity then checks the full screenshot for an adjacent missing surface or a pair of distinct planes inside one wall. The Block Counter receives the resulting masks. The build manifest calls this final refinement `plane-recovery`. [Results on 56 views](AFTERPASS_MODELS.md)

### Model environments

On this PC, the script locates the existing inference installations in the reconstruction worktrees. Other installations can pass `-Python` for the NumPy/Pillow/ONNX/LiteRT environment, and `-ModelPython`, `-ScaleSource`, `-ScaleCheckpoint` and `-MobileSource` for the PyTorch environment and official model checkouts. The MobileSAM checkout must contain `weights/mobile_sam.pt`. The wrapper supports `-Device cuda` or `-Device cpu`. The saved build comparison below used CUDA for ScaleLSD and MobileSAM and CPU inference for the other three checkpoints.

The same build is available through the normal CLI:

```powershell
python reconstruct.py build .\snapshot.png --version 26.3 --through counts --out .\runs\my-build
```

For direct CLI use, configure `MCIR_RECOVERY_PYTHON`, `MCIR_SCALELSD_SOURCE`, `MCIR_SCALELSD_CHECKPOINT` and `MCIR_MOBILESAM_SOURCE`, or put the model sources and checkpoints in `.deps`. The ONNX and LiteRT paths follow the existing `MINECRAFT_MOGE_ONNX`, `MINECRAFT_DEPTH_ONNX` and `MINECRAFT_MLSD_TFLITE` settings. The default build method stays in `engine/surface_defaults.json`.

For Pixel-wise Planarity, the wrapper accepts `-PlanaritySource`, `-PlanarityCheckpoint` and `-PlanarityUtils3D`. Their environment variables are `MCIR_PLANARITY_SOURCE`, `MCIR_PLANARITY_CHECKPOINT` and `MCIR_PLANARITY_UTILS3D`. The Torch environment also needs the matching scipy and connected-components-3d packages. Set `MCIR_PLANARITY_PYTHONPATH` to their directory when installed separately. Fresh inference checks the upstream Git revisions and checkpoint digest. An explicitly supplied `-PlanarityFields` directory permits reuse of numeric outputs keyed by screenshot hash. [Pinned sources and versions](reviews/afterpass-fields-01/planarity/manifest.json) · [Authors’ checkpoint](https://huggingface.co/alpayozkan/pxwplanar-moge2-planarity)

### Verify Plane Recovery from a renamed screenshot

```powershell
New-Item -ItemType Directory -Force .\runs\plane-proof | Out-Null
Copy-Item .\benchmark\data\camera-round-01\captures\dev-wing-plaster-high-right.png .\runs\plane-proof\source.png
.\Invoke-WallRecovery.ps1 .\runs\plane-proof\source.png -Out .\runs\plane-proof\build
python -m benchmark.verify_recovery_build --build runs/plane-proof/build --case smooth_stone/dev-wing-plaster-high-right
```

The default build reads one source image, runs six checkpoints and passes the proposed masks to the ordinary Block Counter. The separate verification command compares the finished surfaces with the saved predictions and the recorded building.

### Reproduce the earlier five-model check

```powershell
New-Item -ItemType Directory -Force .\runs\single-image-proof | Out-Null
Copy-Item .\benchmark\data\normal-confirm-v1\captures\holdout-galleries-plaster-low-left.png .\runs\single-image-proof\snapshot-73.png
.\Invoke-WallRecovery.ps1 .\runs\single-image-proof\snapshot-73.png -SurfaceBoundary wall-recovery -Out .\runs\single-image-proof\build
python -m benchmark.verify_recovery_build --build runs/single-image-proof/build --case light_gray_concrete/holdout-galleries-plaster-low-left
```

The build recomputed all five model outputs. The last command then opened the benchmark prediction and reference labels. All 30 predicted region records matched their archived properties, including their outlines' underlying masks, holes, optional sampling charts and IDs. Scoring returned the same 19 of 25 reference surfaces. Four of those matches came from the MobileSAM step. The Block Counter received all 30 proposals. [Saved verification](reviews/wall-recovery-entry-01.json)
