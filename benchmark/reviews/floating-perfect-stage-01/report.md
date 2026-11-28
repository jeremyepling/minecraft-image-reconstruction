# Floating evaluation from saved receipts

## Decision for the next round

Baseline — expanded split-border stays the benchmark. Candidate 1 — guarded depth-first refit continues as an experimental challenger. It gained 533 net correct positions and removed 78 net extras on the archived 144 views. Dropped walls accounted for 28 previously correct positions on Arcade 1 and 27 on Terrace 1. Test wall retention and compare on active Combined V1 before promoting. The reporting tools landed on Floating.

**Cohort** Original Visible Builds V1 replay

Stage rows measure different opportunities. Assembly cells can overlap across components.
A fresh atlas fit can pass its local checks and still be left out of the export.
Unresolved nonempty exports receive no correct placements. Paired counts omit unresolved views.

## Visible Builds

| Method | Tested / available | Strict / facade F1 | Strict / facade position + ID F1 | Correct / missed / extra | Unresolved / empty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline — Expanded split-border | 144/144 | 37.64% / 40.20% | 15.11% / 16.42% | 9,623 / 24,831 / 7,052 | 7 / 0 |
| Candidate 1 — Guarded depth-first refit | 144/144 | 39.38% / 41.37% | 17.06% / 18.54% | 10,156 / 24,298 / 6,974 | 7 / 0 |

### Stage measures

| Stage | Measure | Baseline — Expanded split-border | Candidate 1 — Guarded depth-first refit |
| --- | --- | ---: | ---: |
| Surface Finder | Masks kept / masks entering the whole-mask sky check | 2,052 / 2,052 | 2,052 / 2,052 |
| Block Counter | Surfaces with both dimensions / surfaces sent to counting | 412 / 2,052 | 412 / 2,052 |
| Depth-first refit | Split parts with a fresh atlas fit offered / split parts tried | Unmeasured | 165 / 730 |
| Local orientation | Views with oriented charts / views with assembly receipts | 144 / 144 | 144 / 144 |
| Relative joins | Charts in the selected group / all oriented charts | 264 / 733 | 280 / 760 |
| Component selection | Exported blocks / cells across all assembled components | 16,675 / 25,024 | 17,130 / 25,402 |
| Material matching | Exact IDs / correct positions in the decoded export | 3,862 / 9,623 | 4,399 / 10,156 |
| Decoded placement | Correct / visible reference blocks | 9,623 / 34,454 | 10,156 / 34,454 |

### Perfect from this stage

Source-only anchor is **Candidate 1 — Guarded depth-first refit** on each row’s exact views. The reference is supplied at the named handoff. Each row is an independent experiment.

[Open the interactive Perfect Stage comparison](perfect-stage.html).

| Handoff and reference input | Tested | Strict placement F1 | Facade placement F1 | Strict position + material F1 | Facade position + material F1 | Correct / missed / extra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Block Counter · Exact visible full-block wall grids | 144 (Archived comparison) | 39.38% → 82.05% (+42.67 pp) | 41.37% → 85.41% (+44.04 pp) | 17.06% → 63.89% (+46.83 pp) | 18.54% → 70.41% (+51.87 pp) | 24,263 / 10,191 / 427 |
| Local heading · Reference chooser | 144 (Archived comparison) | 39.38% → 41.77% (+2.39 pp) | 41.37% → 44.01% (+2.64 pp) | 17.06% → 18.69% (+1.64 pp) | 18.54% → 20.35% (+1.81 pp) | 10,773 / 23,681 / 6,357 |
| Material assignment · Exact at correctly placed coordinates | 144 (Archived comparison) | 39.38% → 39.38% (+0.00 pp) | 41.37% → 41.37% (+0.00 pp) | 17.06% → 39.38% (+22.32 pp) | 18.54% → 41.37% (+22.82 pp) | 10,156 / 24,298 / 6,974 |
| Pack writing · Exact visible coordinates and IDs | 144 (Archived comparison) | 39.38% → 100.00% (+60.62 pp) | 41.37% → 100.00% (+58.63 pp) | 17.06% → 100.00% (+82.94 pp) | 18.54% → 100.00% (+81.46 pp) | 34,454 / 0 / 0 |
| Surface Finder · Exact native visibility masks | 12 (Held-out test split) | 46.34% → 48.62% (+2.28 pp) | 48.77% → 51.00% (+2.23 pp) | 23.03% → 27.62% (+4.60 pp) | 24.24% → 29.04% (+4.81 pp) | 1,269 / 2,145 / 537 |
| Surface and depth separation · Exact native masks and measured plane fields | 12 (Held-out test split) | 46.34% → 50.67% (+4.33 pp) | 48.77% → 53.50% (+4.73 pp) | 23.03% → 38.67% (+15.64 pp) | 24.24% → 40.90% (+16.66 pp) | 1,178 / 2,236 / 58 |
| Block Counter · Exact visible full-block wall grids | 12 (Held-out test split) | 46.34% → 86.95% (+40.61 pp) | 48.77% → 90.18% (+41.41 pp) | 23.03% → 76.97% (+53.95 pp) | 24.24% → 80.10% (+55.86 pp) | 2,632 / 782 / 8 |
| Local heading · Reference chooser | 12 (Held-out test split) | 46.34% → 46.34% (+0.00 pp) | 48.77% → 48.77% (+0.00 pp) | 23.03% → 23.03% (+0.00 pp) | 24.24% → 24.24% (+0.00 pp) | 1,143 / 2,271 / 376 |
| Material assignment · Exact at correctly placed coordinates | 12 (Held-out test split) | 46.34% → 46.34% (+0.00 pp) | 48.77% → 48.77% (+0.00 pp) | 23.03% → 46.34% (+23.31 pp) | 24.24% → 48.77% (+24.54 pp) | 1,143 / 2,271 / 376 |
| Pack writing · Exact visible coordinates and IDs | 12 (Held-out test split) | 46.34% → 100.00% (+53.66 pp) | 48.77% → 100.00% (+51.23 pp) | 23.03% → 100.00% (+76.97 pp) | 24.24% → 100.00% (+75.76 pp) | 3,414 / 0 / 0 |

The heading row chooses among existing possibilities. The writer row verifies the export path. Geometry and material gains across rows share some of the same errors.


### Component changes

**Candidate 1 — Guarded depth-first refit** changed 14/144 packs. Across 2 views, the selected and prior groups shared no parent chart.

On 137 paired views, correct gains / losses 676 / 143 and extras added / removed 2 / 77. Exact IDs gained / lost 605 / 68.

- `train-arcade1-cool-high-right` dropped mask-2 and added horizontal-mask-0, mask-1, mask-4. Correct gains / losses 154 / 28. Extras added / removed 0 / 0.
- `train-terrace1-cool-high-right` dropped mask-1 and added horizontal-mask-6, mask-0, mask-11, mask-9. Correct gains / losses 42 / 27. Extras added / removed 0 / 0.

## Decorative Builds V1

Unmeasured. A separate Decorative run is available on request.
