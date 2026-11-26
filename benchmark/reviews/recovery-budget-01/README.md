# Where the visible geometry errors occur

On 112 frozen evaluation screenshots the ordinary release reconstructed 18,703 of 24,086 visible blocks, with 1,770 extras and 5,383 misses (83.95% relative geometry F1). One translation per screenshot is free. Every optimal translation for this release was zero. At 95% F1, retaining today's extras would require 4,691 more correct blocks. Halving extras would require 3,890 more correct blocks. Eliminating them would still require 3,090 more.

| Missed visible blocks | Validation | Test | Fixed 56 | Native 12 | Earlier transfer | Fresh transfer | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Camera abstained | 0 | 0 | 384 | 0 | 0 | 0 | 384 |
| No production sampling center hit the block | 56 | 134 | 594 | 364 | 228 | 134 | 1,510 |
| Sampled, outside every surface mask | 6 | 8 | 171 | 44 | 9 | 27 | 265 |
| Sampled in masks, none with the true face axis | 22 | 29 | 186 | 116 | 107 | 22 | 482 |
| Weak matching-axis support | 65 | 95 | 432 | 81 | 52 | 31 | 756 |
| Strong matching-axis support | 146 | 173 | 1,038 | 215 | 267 | 147 | 1,986 |

Strong means at least four correctly oriented samples covering at least half of the block's sampled visible pixels. Every mask comparison uses the reference pixel under the sampling center, so these are diagnostics of visual support. They do not certify that the inferred camera and proposed plane put that block at the correct depth.

Of the 1,510 missed by the four-pixel lattice, 1,181 have at least 16 native visible pixels. The largest visible face points up on 1,095, down on 356 and sideways on 59. A two-pixel lattice starting at native pixel (0,0) hits a matching-axis mask on 663. At full resolution 911 have a matching-axis mask hit and 718 have at least 16 such pixels. This isolates the available evidence for a sampling experiment.

Of the 1,986 strong-support misses, an extra placed in front of the cell covers at least half of its native visible pixels on 857. Across all supported-view misses, 1,057 are at least half blocked and 737 are fully blocked. These counts come from reprojecting decoded extras through the recorded camera at scoring time. They quantify overlap and cannot establish whether the pipeline chose the wrong depth for that particular cell.

| Hypothetical edit to the current output, using perfect reference choices | Relative F1 |
|---|---:|
| Move an extra by one face-adjacent step to a missed block, up to 491 distinct pairs | 86.15% |
| Delete all 1,770 extras | 87.42% |
| Add all 1,986 strong-support misses with the existing extras | 88.90% |
| Add every sampled miss with any mask hit, retaining current extras | 91.78% |
| Add that same set and delete all extras | 95.31% |

The bounds apply only to the stated edits, with the release registration fixed. Plane moves of several blocks, denser sampling, new masks, or another visible-only proposal source can reach a different set of cells. Prioritize testing native or two-pixel sampling on grazing horizontal faces and scoring competing front/back depth assignments on existing masks.

## Reproduce

From the project worktree with restored fixtures, run `python -m benchmark.recovery_budget --out benchmark/reviews/recovery-budget-01/scores.json`. The evaluator verifies each RGB, camera, surface input, packed build and registered coordinates against the frozen relative replay or dataset checksums, and rechecks the reference block counts. Native transfer labels come from the saved camera and world readback. Per-frame metrics and the SHA-256 of the source score archive are in [scores.json](scores.json). No training screenshots enter the table.
