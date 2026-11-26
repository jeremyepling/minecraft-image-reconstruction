# Build surfaces from one screenshot with Wall Recovery

From this checkout on the Minecraft PC:

```powershell
.\Invoke-WallRecovery.ps1 .\snapshot.png -Out .\runs\my-build
```

This command uses the default Java 26.3 Surface Finder and Block Counter. The input is one ordinary screenshot. Use `-Through surfaces` to stop after surface finding. The build writes `surfaces.json` with visible masks, surface crops, `counts.json` with supported dimensions and `build-manifest.json` with the methods, settings and checkpoint hashes.

The Wall Separator starts with MoGe-2 directions and Depth Anything V2 depth. Its masks supply MobileSAM point prompts. ScaleLSD identifies junction-supported cuts, M-LSD Tiny finds thin undersides, and MobileSAM can partition a wall when the depth evidence supports a step. Pixel-wise Planarity then checks the full screenshot for an adjacent missing surface or a pair of distinct planes inside one wall. The Block Counter receives the resulting masks. The build manifest calls this final refinement `plane-recovery`. [Results on 56 views](AFTERPASS_MODELS.md)

### Model environments

On this PC, the script locates the existing inference installations in the reconstruction worktrees. Other installations can pass `-Python` for the NumPy/Pillow/ONNX/LiteRT environment, and `-ModelPython`, `-ScaleSource`, `-ScaleCheckpoint` and `-MobileSource` for the PyTorch environment and official model checkouts. The MobileSAM checkout must contain `weights/mobile_sam.pt`. The wrapper supports `-Device cuda` or `-Device cpu`. The earlier five-model verification used CUDA for ScaleLSD and MobileSAM and CPU inference for MoGe-2, Depth Anything V2 and M-LSD Tiny.

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

The default build read one renamed Wing image, ran all six checkpoints, found all five scored surfaces and passed eight proposals to the Block Counter. Its predicted masks and stored region properties matched the archived predictions exactly. The separate verification command reads the finished predictions and the recorded building. [Fresh build verification](reviews/plane-recovery-entry-01.json)

### Earlier five-model result

The five-model finder matched 19 of 25 surfaces on the Galleries screenshot and sent 30 regions to the Block Counter. Its original verification contains the exact masks and model hashes. Run `.\Invoke-SurfaceRound.ps1 -Round afterpass` to compare its saved suite-wide predictions against the selected six-model finder. [Five-model verification](reviews/wall-recovery-entry-01.json) · [Suite comparison](AFTERPASS_MODELS.md)
