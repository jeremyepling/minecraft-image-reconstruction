# Build surfaces from one screenshot with Wall Recovery

From this checkout on the Minecraft PC:

```powershell
.\Invoke-WallRecovery.ps1 .\snapshot.png -Out .\runs\my-build
```

This opt-in command runs the Java 26.3 Surface Finder and Block Counter. The input is one ordinary screenshot. Use `-Through surfaces` to stop after surface finding. The build writes `surfaces.json` with exact visible masks, surface crops, `counts.json` with any supported dimensions, and a `build-manifest.json` recording the actual stages, inference settings and checkpoint hashes.

The Wall Separator starts with MoGe-2 directions and Depth Anything V2 depth. The original wall masks provide point prompts for MobileSAM. ScaleLSD then supplies junction-supported wall cuts. M-LSD Tiny finds thin undersides. Finally, the MobileSAM proposals can separate a remaining wall when multiple pieces account for most of its pixels. Each successful split keeps the original outline and gaps.

### Model environments

On this PC, the script locates the existing inference installations in the reconstruction worktrees. Other installations can pass `-Python` for the NumPy/Pillow/ONNX/LiteRT environment, and `-ModelPython`, `-ScaleSource`, `-ScaleCheckpoint` and `-MobileSource` for the PyTorch environment and official model checkouts. The MobileSAM checkout must contain `weights/mobile_sam.pt`. The wrapper supports `-Device cuda` or `-Device cpu`. The saved build comparison below used CUDA for ScaleLSD and MobileSAM and CPU inference for the other three checkpoints.

The same build is available through the normal CLI:

```powershell
python reconstruct.py build .\snapshot.png --version 26.3 --through counts --surface-boundary wall-recovery --out .\runs\my-build
```

For direct CLI use, configure `MCIR_RECOVERY_PYTHON`, `MCIR_SCALELSD_SOURCE`, `MCIR_SCALELSD_CHECKPOINT` and `MCIR_MOBILESAM_SOURCE`, or put the model sources and checkpoints in `.deps`. The ONNX and LiteRT paths follow the existing `MINECRAFT_MOGE_ONNX`, `MINECRAFT_DEPTH_ONNX` and `MINECRAFT_MLSD_TFLITE` settings. The default build method stays in `engine/surface_defaults.json`.

### Reproduce the fresh-image check

```powershell
New-Item -ItemType Directory -Force .\runs\single-image-proof | Out-Null
Copy-Item .\benchmark\data\normal-confirm-v1\captures\holdout-galleries-plaster-low-left.png .\runs\single-image-proof\snapshot-73.png
.\Invoke-WallRecovery.ps1 .\runs\single-image-proof\snapshot-73.png -Out .\runs\single-image-proof\build
python -m benchmark.verify_recovery_build --build runs/single-image-proof/build --case light_gray_concrete/holdout-galleries-plaster-low-left
```

The build recomputed all five model outputs. The last command then opened the benchmark prediction and reference labels. All 30 predicted region records matched their archived properties, including their outlines' underlying masks, holes, optional sampling charts and IDs. Scoring returned the same 19 of 25 reference surfaces. Four of those matches came from the MobileSAM step. The Block Counter received all 30 proposals. [Saved verification](reviews/wall-recovery-entry-01.json)
