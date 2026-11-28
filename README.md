# Minecraft Image Reconstruction

**Floating is the default reconstruction algorithm on `main`.** It builds from the visible surfaces in a Minecraft screenshot and places the result at an arbitrary origin. It preserves the relative spacing of supported blocks and leaves unseen areas unknown. [Product intent](PRODUCT_INTENT.md) · [Current architecture](ARCHITECTURE.md)

## Run Floating

Run the Java 26.3 offline replay from `main` or a worktree based on it. Floating Front 100 is the standard benchmark. It defaults to the 100 full-block noon images. Use a separate run for decorated builds, or select matched morning and afternoon captures to test lighting.

```powershell
.\Invoke-FloatingReplay.ps1 -FixtureRoot $fixture -ModelRoot $models `
    -RunDirectory $output -Python $benchmarkPython
```

Use `-Case` with exact frame IDs for a focused comparison. The runner saves the fixture selection, mask and grid evidence, case scores and Java datapacks. See the benchmark guide for fixture restoration, dependencies, and available cohorts. [Floating Front 100 setup and commands](benchmark/FLOATING_SHARED_DATASETS.md)

To place an exported build, install its `reconstruction.zip` as a Java 26.3 datapack, run `/reload`, stand at the desired origin, and run `/function reconstruction:build`.

## Compare results

Use the same screenshots and visible references for the baseline and each candidate. Floating scores a single translation for the entire build, so moving the origin preserves credit while changing the spacing between parts affects the score. Report placement and position-plus-material F1 together with correct, missed and extra blocks. [Baseline and historical comparisons](benchmark/FLOATING_BRANCH.md) · [Round review and promotion](benchmark/ROUND_REVIEW.md)

See the mobile workflow for original masks, reference surfaces, attempted grids, and the blocks that survived export. [Scores and image review](benchmark/MOBILE_REVIEW.md) · [Experiment log](MODEL_CHANGELOG.md)

## Archived floor-and-camera algorithm

The previous algorithm uses pavement and floor texture to establish camera scale. Its code, setup and measured results are archived on `floor-with-camera` at `e3bd220`. [Archived README](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/e3bd220188f224a0a6cf2841c8f495fd0ac9d090/README.md) · [Archived architecture](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/e3bd220188f224a0a6cf2841c8f495fd0ac9d090/ARCHITECTURE.md)
