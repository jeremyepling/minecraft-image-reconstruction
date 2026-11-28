# Minecraft Image Reconstruction

**Expanded Split-Border is the default Floating reconstruction method on `main`.** It builds from the visible surfaces in a Minecraft screenshot and places the result at a local origin. It preserves the relative spacing of the assembled blocks and leaves unseen areas unknown. [Product intent](PRODUCT_INTENT.md) · [Current architecture](ARCHITECTURE.md)

## Run Floating

Build from one screenshot with Python, NumPy, Pillow, OpenCV and ONNX Runtime. Use the prepared Java 26.3 atlas with `--atlas` or set `MCIR_PREPARED_ATLAS_26_3`. The pinned MoGe-2 and Depth Anything V2 checkpoints can live under `.deps` in the checkout or be supplied through `MINECRAFT_MOGE_ONNX` and `MINECRAFT_DEPTH_ONNX`. The benchmark setup shows the existing paths on this PC. [Model filenames and setup](benchmark/FLOATING_SHARED_DATASETS.md#replay-front-100)

```powershell
python reconstruct.py build screenshot.png --out runs/my-build --version 26.3 --atlas $atlas
```

The build runs offline and writes `reconstruction.zip`, source masks and counts under `source-handoff`, and the charts, assembly and guard decisions beside the pack. To place the build, install the ZIP as a Java 26.3 datapack, run `/reload`, stand at the desired origin, and run `/function reconstruction:build`.

## Replay Floating Front 100

The standard benchmark runs the same predictor on all 100 full-block noon images. Select matched morning or afternoon captures for lighting comparisons, or run the separate decorated noon cohort.

```powershell
.\Invoke-FloatingReplay.ps1 -FixtureRoot $fixture -ModelRoot $models `
    -RunDirectory $output -Python $benchmarkPython
```

Use `-Case` with exact frame IDs for a focused comparison. The runner saves the fixture selection, mask and grid evidence, case scores and Java datapacks. See the benchmark guide for fixture restoration, dependencies, and available cohorts. [Floating Front 100 setup and commands](benchmark/FLOATING_SHARED_DATASETS.md)

## Compare results

Use the same screenshots and visible references for the baseline and each candidate. Floating scores a single translation for the entire build, so moving the origin preserves credit while changing the spacing between parts affects the score. Report placement and position-plus-material F1 together with correct, missed and extra blocks. [Baseline and historical comparisons](benchmark/FLOATING_BRANCH.md) · [Round review and promotion](benchmark/ROUND_REVIEW.md)

See the mobile workflow for original masks, reference surfaces, attempted grids, and the blocks that survived export. [Scores and image review](benchmark/MOBILE_REVIEW.md) · [Experiment log](MODEL_CHANGELOG.md)

## Archived floor-and-camera algorithm

The previous algorithm uses pavement and floor texture to establish camera scale. Its code, setup and measured results are archived on `floor-with-camera` at `e3bd220`. [Archived README](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/e3bd220188f224a0a6cf2841c8f495fd0ac9d090/README.md) · [Archived architecture](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/e3bd220188f224a0a6cf2841c8f495fd0ac9d090/ARCHITECTURE.md)
