# Shared rows at visible wall corners

On six saved open-sky composites, a repeated block texture on one wall established the row spacing on a perpendicular neighboring wall. Only Castle 01 produced a transfer. The method is experimental on `experiment/shared-corner-rows` and the production default is unchanged.

## Castle 01

| Intervention | Correct positions | Missed | Extras | Correct position and material |
|---|---:|---:|---:|---:|
| Previous image-only assembly | 26 | 1,445 | 0 | 22 |
| Rebuild the measured donor S03 | 177 | 1,294 | 2 | 20 |
| Keep the original donor cells and add the S04 chart | 55 | 1,416 | 0 | 26 |
| Rebuild S03 and join S04 | 186 | 1,285 | 2 | 20 |

The counter measured S03 at 7 × 27. Source masks and vertical contact supported carrying 27 rows to S04. Mask support admitted 32 receiver face cells in the first two columns. Twenty-three cells shared a block with a donor face. Adding the receiver to the rebuilt donor therefore contributed nine new correct positions and zero extra positions. The source-only grid placed all 32 receiver face centers on the recorded wall and correct vertical row. None had the correct material.

The S01/S04 whole-height transfer was rejected on the difference in the masks’ estimated depth. Recorded geometry shows a genuine shorter contact of 16 rows. S01 groups several depths. A local comparison along that shorter contact could preserve the true junction.

When both walls have independent counts, the rule compares their projected row positions along the overlap. Four synthetic checks covered agreement at equal height, a shorter wall sharing a subset of rows, a clearer conflicting measurement winning, and abstention on an ambiguous conflict. None of the six source views had two qualifying measurements at one candidate seam. The second replay preserved the chart and placement files byte for byte.

Coordinates are scored after a single whole-build translation. Similar castle towers permit multiple plausible registrations. The final score credits a translation onto the other tower. Native pixel-to-face checks independently establish that the transfer measured the intended S04 rows. A reference-only substitution of the exact projected receiver row boundaries reduced the mean boundary error from 2.21 px to zero and yielded 185 correct positions and two extras.

Of 141 candidate contacts, the audit verified 32 along a recorded perpendicular corner for at least two consecutive unit edges within six source pixels. Only one was accepted. Of the other 31, twenty lacked a qualifying repeat count, ten failed the mask-depth check, and one lacked the required width or outline. These counts measure missed junction opportunities. Placement gains on those cases remain unknown.

## Six-view comparison

| Open-sky subset | Views tested / source cohort | Strict placement before → after | Facade placement before → after |
|---|---:|---:|---:|
| Visible Builds v3 | 4 / 144 | 2.48% → 11.61% | 2.51% → 11.70% |
| Decorative Builds v1 | 2 / 40 | 1.77% → 1.77% | 1.91% → 1.91% |
| Pooled open sky | 6 / 6 | 2.35% → 9.78% | 2.40% → 9.98% |

Pooled strict counts moved from 48 correct, 3,993 missed and three extras to 208 correct, 3,833 missed and five extras. Pooled strict position-plus-material F1 fell from 1.37% to 1.22%. No transfer was accepted on Harbor, Viaduct, Pillars, Hillside Chapel or Walled Garden. Across these five views, 57 recorded candidate decisions stopped because neither wall had a confident vertical repeat count. Fixed 104 was not rerun. Its carried-forward production placement is 91.27% strict and 93.42% facade, with 20,773 correct, 2,929 missed and 1,046 extras.

## Replay

From this worktree, run `python -m benchmark.shared_corner_rows` using the benchmark OpenCV environment. Inputs are the frozen sky fixtures and the archived Regions First masks, counts and dense atlas charts. The candidate uses the source image, camera-space normal estimates and a nominal lens prior. Reference files enter only after export.

The six candidate packs, placement lists, grid overlays and seam receipts are in `runs/shared-corner-rows-01/sky`. The donor-only control is in the sibling `donor-only/sky` directory. The independent pack comparison is in `runs/shared-corner-score-01/paired.json`. The native edge audit and Castle controls are in `runs/shared-corner-audit-01`. Run `python -m benchmark.shared_rows_review` with these receipts and the previous stage review present to regenerate `runs/shared-corner-viewer-01/index.html`.
