# Testing visible returns before block-grid fitting

Neither source-only variant recovered an additional transition between parallel walls at different depths. The frozen comparison uses two Stable Yard 1 development views and transfers the rule to one view each of Castle 01, Viaduct 02 and Pillars 01. All four buildings had been reviewed in historical rounds. Thresholds were specified in advance and kept identical across buildings. The predictors take the archived photographed support, RGB and relative-depth and normal fields. The five selected masks are a focused regression panel. Material IDs, native owner IDs and camera projection do not enter the predictors.

The first method proposes finite photographic line segments. Each supporting sample stays on its own side of the parent region. Side-specific slopes from 4 to 12 pixels are extrapolated to the seam. The two surfaces must have similar normals and a consistently signed jump above local noise. A line must disconnect the region within three pixels of its measured endpoints. The second method examines appearance regions merged by the existing partition. Across dense normal-to-contour probes at 3, 6 and 9 pixels, the across-boundary change must beat both same-side depth changes. Neither method requires a floor or adds observed mask area.

| Five masks, pooled | Internal boundary precision / recall within 3 px | Pixels assigned to dominant native plane | Parallel transitions recovered | Same-material parallel transitions recovered |
|---|---:|---:|---:|---:|
| Initial depth/normal partition | 90.50% / 6.21% | 54.36% | 565 / 5,630 | 21 / 2,161 |
| Existing partition with appearance and depth contacts | 56.73% / 32.19% | 73.13% | 1,507 / 5,630 | 51 / 2,161 |
| Finite return lines | 56.61% / 32.22% | 73.13% | 1,507 / 5,630 | 51 / 2,161 |
| Dense paired-side gate | 56.73% / 32.19% | 73.13% | 1,507 / 5,630 | 51 / 2,161 |
| Exact native plane IDs, scoring control only | 100% / 100% | 100% | 5,630 / 5,630 | 2,161 / 2,161 |

The exact control groups native frontmost pixels by signed world plane, preserving background as a distinct owner within the input support. It verifies the mask assessor. Its scores establish no placement ceiling. Existing pink and return-line masks covered the same 690,588 source pixels.

Stable Yard low-left offered 32 line/depth candidates, high-right zero, Castle 13, Viaduct six and Pillars zero. Only the Viaduct sky composite accepted a cut, on mask-0 from (762, 477) to (790, 478). The cut added 67 predicted boundary pixels, of which 13 were within three pixels of truth. It recovered 15 more native edge pixels with no additional parallel transition. The dense paired-side rule accepted no cuts. [Viaduct source crop and native boundaries](train-viaduct-02-riverstone-high-right-cut.png)

On the changed mask both partitions went through the same regional tangent-frame and full-face atlas refitter, then ordinary assembly, the existing border-sky filter and decoded pack scoring. The region's 142 locally refitted cells became 129 plus 14. The selected export had identical coordinates and material at every site. The unchanged-package replay matched every ZIP member byte of the Viaduct pack pinned by 51bd1e9. Relative to that pack the common refit gained five correct positions and lost 124, adding 24 extras and removing 98. Rerun with `python -m benchmark.floating_pregrid_visible_returns` followed by the same command with `--build`. The scripts use immutable archived fields and source handoffs in adjacent worktrees. [Decoded placement comparison](train-viaduct-02-riverstone-high-right-placements.json)

Keep both source rules experimental. Benchmark code, receipts and the source crop landed on the experiment branch. The unchanged-mask dense rule required no new placement replay. The fixed-104 production trend remains separately carried forward at 91.27% strict and 93.42% facade placement F1.

## Floating branch scorecard

| Revision and conditions | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---:|---:|---:|
| Sparse integration 9435934, six, carried forward | 12.642% / 12.899% | 4.677% / 4.772% | 273 / 3,768 / 5 |
| Expanded archive 51bd1e9, 84, carried forward | 33.987% / 34.479% | 5.655% / 5.757% | 18,318 / 61,192 / 9,965 |
| Same archived Viaduct sky input, package replay | 43.564% / 43.645% | 6.559% / 6.572% | 352 / 739 / 173 |
| Viaduct, existing partition and regional refit | 32.748% / 32.817% | 4.216% / 4.225% | 233 / 858 / 99 |
| Viaduct, return-line partition and same refit | 32.748% / 32.817% | 4.216% / 4.225% | 233 / 858 / 99 |

The isolated source cut changed placement F1 by zero, correct positions by zero and extras by zero. The common regional refit lost 119 net correct blocks and removed 74 net extras. No integration behavior changed. The remaining archived views and the larger datasets were not rerun.
