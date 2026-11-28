# Floating evaluation from saved receipts

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

### Component changes

**Candidate 1 — Guarded depth-first refit** changed 14/144 packs. Across 2 views, the selected and prior groups shared no parent chart.

On 137 paired views, correct gains / losses 676 / 143 and extras added / removed 2 / 77. Exact IDs gained / lost 605 / 68.

- `train-arcade1-cool-high-right` dropped mask-2 and added horizontal-mask-0, mask-1, mask-4. Correct gains / losses 154 / 28. Extras added / removed 0 / 0.
- `train-terrace1-cool-high-right` dropped mask-1 and added horizontal-mask-6, mask-0, mask-11, mask-9. Correct gains / losses 42 / 27. Extras added / removed 0 / 0.

## Decorative Builds V1

Unmeasured. A separate Decorative run is available on request.
