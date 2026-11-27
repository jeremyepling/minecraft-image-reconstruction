# Cloud proposals in the six frozen open-sky composites

At revision `dd385e6`, 23 of 216 surface proposals owned only background. A source-only check rejected all 23. The foreground reference was the native frontmost face-ID raster of the photographed building. The six original captures share a composited sky style. Another building tests transfer across structures, while a different rendering of the sky still needs its own test.

For each row, the rule requires agreeing blue pixels on the left and right borders. It finds pixels with a compatible mix of sky and white, then takes four-connected regions reaching a side. The whole-mask rule requires every pixel to qualify. A separate follow-up trims the qualifying pixels from mixed masks with at least 99% sky. The whole-mask decision was frozen before Jeremy supplied the four Viaduct inspector labels. The 99% follow-up was tested after inspecting Viaduct S02. The scorer opens the reference after prediction.

| Frame | Split | Masks before → after | Whole masks rejected | Background pixels removed by whole-mask rule | Additional mixed pixels trimmed |
|---|---|---:|---|---:|---:|
| Castle | Test | 21 → 17 | S06, S07, S11, S21 | 21,252 | 0 |
| Harbor | Validation | 34 → 34 | — | 0 | 0 |
| Viaduct | Train | 36 → 33 | S03, S05, S06 | 20,620 | 8,298 from S02 |
| Pillars | Train | 34 → 31 | S10, S33, S34 | 8,660 | 426 from S31 |
| Hillside Chapel | Validation | 30 → 27 | S04, S26, S30 | 7,100 | 0 |
| Walled Garden | Test | 61 → 51 | S09, S15, S19, S24, S32, S42, S47, S55, S57, S59 | 24,264 | 0 |

Of 150,029 unique background pixels covered before filtering, 68,133 remain with whole masks rejected, or 59,409 with mixed trimming. Both arms preserve every covered native building pixel. All 193 masks with any building evidence survive, including 15 masks with at least 64 visible pixels whose red, green and blue values each reach 180. The trimmed Viaduct S02 retains 22 building pixels and Pillars S31 retains two. The small remaining fragments are pixel evidence only.

The six registered-seam chart packets contain 15 charts. Neither arm touches a chart ID, so the inputs to assembly remain identical. The paired baseline has 239 correct positions, 3,802 misses and five extras, 11.16% strict and 11.38% facade placement F1, and 2.89% strict and 2.95% facade position-plus-ID F1. The six views cover four of 144 Visible Builds v3 and two of 40 Decorative Builds v1 views. The fixed 104 historical views were unmeasured here. The previous published production result carries forward at 91.27% strict and 93.42% facade placement F1, with 20,773 correct, 2,929 missed and 1,046 extra positions.

Keep the filter experimental until independent skies and bright building materials under other lighting demonstrate the same behavior. The experiment lives on `experiment/sky-mask-feedback`. Run `python -m benchmark.sky_mask_feedback` from its checkout to generate the per-mask report and six source overlays. Red outlines mark rejected masks and yellow outlines mark partial trims. The [frozen receipt](report.json) lists input hashes, native-pixel counts, discarded IDs, all baseline chart IDs and retained support for the mixed masks.
