# Camera-conditioned wall separation

Keep `experiment/camera-conditioned-wall-separation` experimental. On 13 paired Java 26.3 screenshots, the relative inverse-depth gate admitted three of 63 proposed moves. Twelve reconstructions matched the ordinary packs at every decoded coordinate and block ID. Only Harbor 02 kiln high-right changed. Six visible blocks were recovered, six were lost, five extras appeared and twelve extras disappeared. The complete main-branch score remains the previously measured fixed 104-view 91.27% strict / 93.42% facade (20,773 correct, 2,929 missed, 1,046 extras). No production behavior changed. The historical 112-view comparison and paired lighting were unmeasured in this pilot.

## Held source inputs

The camera and masks come from the ordinary pipeline at `e3bd220`. Expanded cases use the exact frozen depth field and the saved ordinary source camera including native floor recoveries. Fixed controls use their released source camera and the pinned Depth Anything v2 Small model for a requested depth decision. The prior largest-wall RGB partition and texture-repeat ranker offer integer plane levels. For each non-parent proposal, compare a one-sample-eroded child with the eroded complement within ten samples. At least 16 samples on each side must survive. Fit relative inverse depth as an affine function of camera-axis ray coefficient over the neighboring parent samples, trimming the largest 20% of residuals. The slope must be positive. Require a child offset in the proposed direction exceeding twice the parent noise and a median absolute residual reduction exceeding one noise unit. Noise is the larger of 1.4826 times the parent median absolute residual, 2.5% of the parent 10th–90th percentile range, or 0.00001 field units. The original rule stayed frozen after development. Each comparison uses one screenshot. The labels enter after export.

Only provisional placement for supported pixels in the largest vertical carrier changes. Public mask ownership, outlines, openings and counted chart fields retain their existing inputs and policies. No extra surface area is proposed. The ordinary visible-cell filter, terminal edits, material assignment and export rerun. In the fixed facade control, one admitted move produced no provisional cell change. On kiln Harbor it added nine and removed twelve provisional cells.

## Decoded results

F1 values are percentages. The fixed pilot used 2 of 104 views, Visible Builds v3 used 9 of 144 and Decorative Builds v1 used 2 of 40. Strict/facade geometry and geometry-plus-ID use the same geometry-selected whole-build translation. Frame filenames identify training, validation and test folds. Harbor and Castle are previously inspected regression families. The market and garden are decorative controls.

| Pilot cohort | Strict / facade placement before → after | Strict / facade placement + ID before → after | Correct / missed / extra after |
|---|---:|---:|---:|
| Fixed 2/104 | 87.02 / 92.52 → same | 85.46 / 90.70 → same | 446 / 101 / 32 |
| Visible Builds v3 9/144 | 51.91 / 55.26 → 51.93 / 55.30 | 31.21 / 34.04 → 31.23 / 34.06 | 5,356 / 7,343 / 2,574 |
| Decorative Builds v1 2/40 | 64.82 / 65.53 → same | 56.79 / 56.99 → same | 444 / 324 / 158 |

| Frame | Admitted / proposed | Strict / facade placement | Strict / facade + ID | Correct gained / lost | Extras added / removed |
|---|---:|---:|---:|---:|---:|
| heldout-facade-oblique-left | 1 / 1 | 91.38 / 92.17 | 89.08 / 89.86 | 0 / 0 | 0 / 0 |
| inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right | 0 / 0 | 77.81 / 93.75 | 77.81 / 93.75 | 0 / 0 | 0 / 0 |
| test-castle-01-ashlar-high-right | 0 / 0 | 75.43 / 75.46 | 42.57 / 42.59 | 0 / 0 | 0 / 0 |
| test-castle-02-ashlar-rear-oblique | 0 / 0 | 56.94 / 57.11 | 39.96 / 40.08 | 0 / 0 | 0 / 0 |
| test-walled-garden-1-warm-high-right | 0 / 12 | 65.49 / 66.39 | 57.40 / 57.54 | 0 / 0 | 0 / 0 |
| train-amphitheater-02-ashlar-high-right | 0 / 6 | 52.85 / 53.46 | 15.81 / 14.64 | 0 / 0 | 0 / 0 |
| train-market-court-1-cool-high-right | 0 / 6 | 63.90 / 64.42 | 55.96 / 56.28 | 0 / 0 | 0 / 0 |
| train-observatory-02-seafoam-low-left | 0 / 7 | 72.54 / 72.86 | 68.14 / 68.44 | 0 / 0 | 0 / 0 |
| train-shrine-01-ochre-low-left | 0 / 1 | 47.50 / 47.89 | 30.23 / 30.48 | 0 / 0 | 0 / 0 |
| train-viaduct-02-riverstone-high-right | 0 / 2 | 82.73 / 82.80 | 62.05 / 62.17 | 0 / 0 | 0 / 0 |
| validation-harbor-02-kiln-high-right | 2 / 13 | 18.22 / 19.44 | 0.70 / 0.97 | 6 / 6 | 5 / 12 |
| validation-harbor-02-seafoam-high-right | 0 / 5 | 20.90 / 22.62 | 7.92 / 7.79 | 0 / 0 | 0 / 0 |
| validation-harbor-03-seafoam-high-right | 0 / 10 | 17.23 / 15.41 | 6.64 / 10.36 | 0 / 0 | 0 / 0 |

Harbor 02 kiln moved from 18.13 / 18.96 placement and 0.55 / 0.77 position-plus-ID. Every other row retained its prior scores. Harbor 03 seafoam vetoed all ten non-parent moves. Its separate reference-partition diagnostic reached 30.69% strict from 17.23%. Neither Castle frame offered a non-parent move from the local repeat ranker.

## What the depth evidence distinguished

All sixteen development moves were vetoed. Six child offsets were inside the two-noise threshold, five had the wrong direction, four interiors were too small and one parent had no usable slope. In Amphitheater the six interiors contained 519–1,406 samples, but every proposed displacement worsened the depth residual. The observed offsets ranged from −0.79 to +0.84 noise units while the proposed displacements ranged from −12.23 to −3.04. This conflicts with the earlier RGB ranker's 52.85% → 69.82% strict improvement on this exact view. In Shrine the one harmful ranked move was correctly rejected on sign. One Viaduct proposal approached the threshold (1.67 noise offset, 0.99 noise gain). Looser positive thresholds alone cannot recover the Amphitheater result. Castle needs different depth proposals before this gate can affect it.

[Per-frame source, field, pack and decoded hashes, counts and scores](summary.json) · [Per-component source measurements and ranker candidates](decisions.json.gz) · [Earlier RGB-only comparison](../wall-color-partition-01/README.md)

For replay, use the workspace benchmark Python and atlas. Set `MINECRAFT_DEPTH_ONNX` to the pinned checkpoint under `normal-round/.deps` and include `surface-round/runs/depth-probe/runtime` on `PYTHONPATH` for fixed controls. Run `python -m benchmark.camera_conditioned_wall_separation FRAME_ID ...`, followed by `python -m benchmark.camera_conditioned_wall_report`. The private run folder contains ordinary exports and component decisions.
