# Twinspire brick fragments beside the bridge

On the reviewed high-right image, the source-only rule extends five cells into the existing brick grid. Their combined projected footprint contains 1,569 native pixels of brick on the target plane and 1,186 pixels of visible smooth sandstone. All 533 pixels used to authorize the cells belong to the target brick plane. Four cells draw evidence from one visible face each. The fifth straddles two brick faces, with 16 and 12 pixels respectively. Keep this experiment on its branch. It demonstrates that a partly occluded face can have enough visible texture to warrant a cell. The fifth cell still shows grid drift.

The test starts from the previous reviewed row-with-material grid on high-right and the scale grid on low-left. Its homography and existing cells remain fixed. For a brick-dominated chart it learns brick chromaticity from three or more high-confidence brick cells and samples the inner 80% of each missing cell inside the occupied UV bounds. Each candidate must lie within two grid steps of an existing brick and contain at least 12 target-colored mask pixels, with eight connected pixels. Color agreement with the brick samples supplies per-cell evidence. The method does not independently estimate the bridge depth. Native IDs, depths and actual materials enter only after export. The opposite view and all six fixed sky images received no new cells.

Ordinary assembly grew the second-largest Twinspire component from 65 to 70 blocks. It exported the same 173-block component comprising mask-0 and mask-1 on both runs. Correct, missed and extra stayed at 110 / 184 / 63, with strict and facade F1 of 47.11% and 49.22%. Strict and facade position-plus-ID F1 remained 17.99% and 18.79%. Connecting the brick wall to the exported assembly is a separate unresolved handoff.

The experiment branches from 1442344. Reproduce with the saved manifests and chart folders by running `../mask-scalelsd/runs/env/Scripts/python.exe -m benchmark.floating_occlusion_fragments`. Per-case source and chart hashes, native ownership and paired decoded export scores are in [paired.json.gz](paired.json.gz).

## Floating branch scorecard

Six fixed open-sky images, comprising four Visible Builds v3 and two Decorative Builds v1 cases. The integration default at 9963dee carries forward its existing measurements. The other two rows are paired on these same six views.

| Method | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---|---|---|
| Integration default, carried forward | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Large-contraction chart input | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |
| With brick fragment extension | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |

The fragment rule changed strict F1 by zero points and added zero correct positions and zero extras relative to its paired input. The experiment adds benchmark code and evidence only on its branch. The separate fixed-104 production trend carries forward at 91.27% strict and 93.42% facade.
