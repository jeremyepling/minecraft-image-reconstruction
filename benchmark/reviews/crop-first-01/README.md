# Castle S04 native-crop comparison

The saved masked surface crop preserved horizontal block edges with a median tilt of 5.2°. The supplied counter quadrilateral stretched its right edge from 119 source pixels to the full sample height. Its lower-left corner falls at reference row 27.3 and its lower-right corner at row 8.8. Across 79 full reference faces the transformation preserved handedness, but increased the median horizontal tilt to 33.6°. The source image contains enough information to recognize the materials.

## Counting from the source crop

The original masked-repeat detector, run without a plane-aspect prior, found a weak 15.0-pixel vertical period on S04 and an 11.3-pixel horizontal period. Neither passed its final confidence check. On the saved straightened sample neither direction produced a candidate. This like-for-like check can be repeated with `python -m benchmark.raw_crop_pitch`.

The crop-first challenger in `experiment/surface-crop-first` fits visible line directions and compares individual projected atlas faces with native pixels. Eighteen across-wall and twelve upright segments supplied directions on S04. Locally it measured 11.38 × 14.98 pixels per block. The 38 accepted cells span six columns and eight rows. Independent frontmost-pixel reference checks confirmed all 38 positions and materials, with 0.49 px average corner error. The reference plane contains 87 visible faces in its 7 × 27 envelope. An accepted patch alone gives no full-wall count.

As a separate reference-only diagnostic, all 32 centers of the earlier shared-row S04 chart already landed on the correct source rows and columns. Its corners averaged 3.83 px of error. The existing atlas verifier recognized zero materials using those corners. On exactly the same 32 cells, supplying the recorded corners gave 30 correct materials and two abstentions.

## End-to-end placement

| Six sky views | Correct / missed / extra | Strict / facade placement F1 | Strict / facade position + ID F1 |
|---|---|---|---|
| Previous shared-row assembly | 208 / 3,833 / 5 | 9.78% / 9.98% | 1.22% / 1.25% |
| Replace charts with crop-first charts | 90 / 3,951 / 44 | 4.31% / 4.31% | 0.67% / 0.68% |
| Transfer IDs requiring strong polygon overlap | 208 / 3,833 / 5 | 9.78% / 9.98% | 1.22% / 1.25% |
| Register native cells to the existing joined wall | 239 / 3,802 / 5 | 11.16% / 11.38% | 2.89% / 2.95% |

Substituting only S04's new chart broke its connection to S03. The assembler exported the donor by itself, losing nine correct positions on Castle. Requiring strong polygon overlap to edit IDs preserved all six exports but accepted no changes. The six sky views cover four of 144 Visible Builds v3 views and two of 40 Decorative Builds v1 views. The fixed 104-view production trend was not rerun. It carries forward at 91.27% strict and 93.42% facade, with 20,773 correct, 2,929 missed and 1,046 extra positions.

The updated surface inspector retains original source pixels in the native atlas panel and draws each accepted face at its measured corners. On Castle S04 it also shows the saved sample beside a reference-only straightening of the recorded plane. The local source and evaluation receipts are in `runs/native-crop-audit-01` and `runs/native-crop-score-01`.

The successful combination preserved the old shared corner and aligned the new chart by seven common centers in one column. Four rows proposed an integer offset, and three held-out rows agreed. It added 31 correctly positioned blocks, repaired seven fallback materials and preserved all 23 shared unit edges. All 38 material assignments matched at the pictured wall. Under the whole-build translation to the other tower, the end-to-end scorer credited 36 additional exact IDs. A control requiring agreement in two columns abstained on all six builds. Castle alone changed, so keep this path experimental and test a second joined structure.
