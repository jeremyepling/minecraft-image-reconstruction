# Source-color wall partitions and local depth selection

Keep `experiment/wall-color-partition` experimental. The source-only repeat ranker improves several training buildings, but it regresses Shrine and swaps many correct positions on Harbor. A training-selected confidence margin preserves Shrine while reducing the Harbor gain. Four fixed-suite regression views remain unchanged. This is insufficient evidence for a global default.

The ordinary default remains `main` at `62611ee`. No production engine code changed. The diagnostic and source-only trial are separate: [verified reference partition opportunity](../harbor-placement-evidence-01/README.md).

## What ran

Choose the largest vertical carrier by its source-mask area. Connect neighboring four-pixel samples when their 4×4 median RGB differs by at most 20 per channel. Keep components of at least 16 samples; smaller components retain the parent. For each component, offer the ordinary plane and integer depths within six blocks. The ranked arm uses the existing frozen texture-repeat depth model on each component. The boundary arm accepts a boundary score of at least 0.60 and a gain of at least 0.15. The guarded arm additionally requires a 0.5 ranking-score margin over the parent; this is a score margin, not a calibrated probability.

The first training comparison used Amphitheater ashlar high-right and Viaduct riverstone high-right. Additional training families exposed Shrine’s low-margin failure. The 0.5 guard was selected from training results, not Harbor labels. Harbor has already been inspected and is a regression family, not a fresh holdout. Both source variants preserve the parent when no eligible change is selected. Components assigned the same depth are joined before ordinary visible-cell generation. All later visibility, foreground, material, and export steps rerun. Reference labels enter only after export for scoring.

## Decoded exports

Each row compares identical source images against the verified default. F1 is in percent. Position-plus-ID uses the geometry-selected whole-build integer translation. Gains/losses count actual required or optional visible positions after the permitted global shift, so a tied score can still hide changed cells.

| Source image | Arm | Strict placement, before → after | Facade placement, before → after | Strict / facade position + ID, after | Correct gained / lost | Extras added / removed |
|---|---|---:|---:|---:|---:|---:|
| heldout-facade-oblique-left | guarded | 91.38 → 91.38 | 92.17 → 92.17 | 89.08 / 89.86 | 0 / 0 | 0 / 0 |
| inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right | guarded | 77.81 → 77.81 | 93.75 → 93.75 | 77.81 / 93.75 | 0 / 0 | 0 / 0 |
| legacy-light_gray_concrete-dev-shopfront-plaster-high-right | guarded | 91.97 → 91.97 | 90.07 → 90.07 | 83.21 / 78.15 | 0 / 0 | 0 / 0 |
| legacy-light_gray_concrete-holdout-courtyard-masonry-high-right | guarded | 93.13 → 93.13 | 95.05 → 95.05 | 86.03 / 95.05 | 0 / 0 | 0 / 0 |
| train-amphitheater-02-ashlar-high-right | boundary | 52.85 → 56.58 | 53.46 → 58.24 | 19.65 / 19.50 | 127 / 82 | 147 / 242 |
| train-amphitheater-02-ashlar-high-right | guarded | 52.85 → 69.82 | 53.46 → 71.55 | 28.26 / 28.08 | 298 / 8 | 138 / 369 |
| train-amphitheater-02-ashlar-high-right | ranked | 52.85 → 69.82 | 53.46 → 71.55 | 28.26 / 28.08 | 298 / 8 | 138 / 369 |
| train-factory-01-kiln-low-left | ranked | 66.28 → 70.90 | 66.36 → 70.98 | 53.12 / 53.18 | 19 / 0 | 2 / 24 |
| train-market-court-1-cool-high-right | ranked | 63.90 → 64.71 | 64.42 → 65.25 | 56.75 / 57.09 | 5 / 3 | 9 / 12 |
| train-market-court-1-warm-low-left | ranked | 65.77 → 65.77 | 65.50 → 66.09 | 44.23 / 48.26 | 2 / 2 | 0 / 0 |
| train-observatory-02-seafoam-low-left | ranked | 72.54 → 78.17 | 72.86 → 78.52 | 73.45 / 73.78 | 39 / 3 | 12 / 54 |
| train-pillars-01-marble-low-left | ranked | 88.33 → 88.33 | 91.01 → 91.01 | 61.32 / 63.19 | 0 / 0 | 0 / 0 |
| train-shrine-01-ochre-low-left | guarded | 47.50 → 47.50 | 47.89 → 47.89 | 30.23 / 30.48 | 0 / 0 | 0 / 0 |
| train-shrine-01-ochre-low-left | ranked | 47.50 → 44.02 | 47.89 → 44.38 | 26.41 / 26.63 | 2 / 18 | 9 / 7 |
| train-viaduct-02-riverstone-high-right | boundary | 82.73 → 82.73 | 82.80 → 82.80 | 62.05 / 62.17 | 0 / 0 | 0 / 0 |
| train-viaduct-02-riverstone-high-right | guarded | 82.73 → 83.06 | 82.80 → 83.12 | 62.37 / 62.49 | 3 / 0 | 0 / 4 |
| train-viaduct-02-riverstone-high-right | ranked | 82.73 → 83.45 | 82.80 → 83.51 | 62.75 / 62.87 | 7 / 0 | 0 / 8 |
| validation-harbor-02-kiln-high-right | guarded | 18.13 → 19.96 | 18.96 → 24.67 | 5.35 / 7.28 | 79 / 72 | 134 / 203 |
| validation-harbor-02-kiln-high-right | ranked | 18.13 → 23.17 | 18.96 → 28.77 | 4.89 / 6.64 | 101 / 71 | 113 / 198 |
| validation-harbor-02-kiln-rear-oblique | ranked | 77.49 → 77.49 | 86.13 → 86.13 | 51.01 / 58.05 | 0 / 0 | 0 / 0 |
| validation-harbor-02-seafoam-high-right | guarded | 20.90 → 21.26 | 22.62 → 23.52 | 8.62 / 9.91 | 97 / 97 | 187 / 212 |
| validation-harbor-02-seafoam-high-right | ranked | 20.90 → 20.91 | 22.62 → 23.02 | 7.80 / 8.78 | 94 / 97 | 186 / 212 |

## What to test next

Use the newly calibrated cameras to distinguish two situations: one mask combines genuinely different wall depths, or one flat wall merely changes material. Trace every lost correct cell to its first changed component and depth. Then test independent visible corners, junctions, or depth evidence to authorize individual changes. A global confidence threshold does not reliably distinguish those situations. Do not remove the grounded-depth filter: its isolated bypass introduced too many extras in the diagnostic.

## Placement status

These are carried-forward ordinary scores at `62611ee`, not a full-cohort score for the challenger. The trial has four fixed views, nine v3 views, and two decorative views; arms have different coverage recorded in `summary.json`. No new whole-cohort default was measured.

| Cohort | Verified baseline views | Strict / facade placement | Strict / facade position + ID | Correct / missed / extras |
|---|---:|---:|---:|---:|
| Fixed trend | 104/104 | 91.27 / 93.42 | 84.68 / 88.51 | 20,773 / 2,929 / 1,046 |
| Visible Builds v3 | 144/144 | 21.84 / 22.75 | 14.13 / 14.79 | 28,086 / 186,064 / 14,949 |
| Decorative Builds v1 | 40/40 | 63.46 / 65.23 | 50.51 / 52.56 | 9,240 / 7,686 / 2,957 |

Camera failures are unchanged: 0 fixed, 70 v3, and 0 decorative. Six control exports match the verified baseline coordinates and IDs exactly. Production inputs are frozen source-only model fields and source-derived cameras; the camera stage itself is unchanged in this experiment.

```powershell
python -m benchmark.wall_color_partition train-amphitheater-02-ashlar-high-right train-viaduct-02-riverstone-high-right --modes control ranked boundary --workers 3
python -m benchmark.wall_color_partition train-shrine-01-ochre-low-left validation-harbor-02-kiln-high-right validation-harbor-02-seafoam-high-right --modes guarded --workers 3
python -m benchmark.wall_color_partition_report
```
