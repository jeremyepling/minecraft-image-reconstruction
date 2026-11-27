# Castle and Viaduct surface feedback

The 13 exported Castle notes and four Viaduct cloud annotations are tied to source, mask and counter fingerprints. Unannotated surfaces have no user label. The six previously reviewed sky composites are a regression set.

## Level rows using source-image edges

The line trial detects segments inside the unwarped surface mask. It requires at least three near-horizontal segments agreeing in slope and majority length support, and applies one vertical shear to level that direction. It keeps the native width and the full masked area. Without a consensus the viewer shows the original crop. Run `python -m benchmark.feedback_row_lines`, then `python -m benchmark.surface_sample_review` to regenerate the samples and viewer.

Of 216 masks, 59 supplied line evidence. There were 26 with a correction, an existing counter sample, and independently measurable horizontal edges on a dominant recorded wall. The median per-mask absolute row tilt fell from 5.28° in the existing sample to 1.82°; 21 improved and five worsened. On those 26 walls, the same texture detector with no aspect prior returned six axis periods after leveling and four on existing samples. Five and three respectively were within 15% of the reference median projected one-block spacing. These are local period estimates. A full wall count also needs the correct ends of the wall.

| Surface | Original tilt | Counter tilt | Trial tilt | Observation |
|---|---:|---:|---:|---|
| Castle S08 | 10.23° | 8.63° | 1.94° | Rows improve; no confident repeat count |
| Harbor S04 | 0.63° | 14.93° | 0.63° | Native perspective already good; recovered horizontal pitch 26.58 px, reference 26.32 px |
| Castle S03 | 5.52° | 1.73° | 5.36° | Current counter warp fits the rows better |
| Castle S09 | 17.16° | 10.58° | 17.16° | No line consensus; the post contains two different visible sides |
| Castle S12 | 19.66° | 24.03° | 19.66° | No line consensus |
| Castle S15 | 6.59° | 13.63° | 6.59° | No line consensus |

The line trial measures sample quality. Its outputs have not replaced chart or placement inputs. Three of four measured Viaduct walls worsened, so choosing between corrections using independent row or atlas evidence on each surface is a better next experiment than applying a shear everywhere.

## Cloud and duplicate-mask checks

The separate `experiment/sky-mask-feedback` source-only rule uses sky color at the image borders and connected background pixels. It removed 23 whole background masks and trimmed two mixed masks. Together the changes removed 90,620 background pixels and zero visible structure pixels on the six synthetic sky views. Viaduct S03, S05 and S06 were pure background. S02 contained 8,298 background pixels and 22 visible structure pixels. The four Castle cloud examples were pure background. Validate real cloud, water and bright-wall views before wider use.

Castle S13 and S14 are disjoint from S03 in source pixels. Their structure pixels belong to blocks already seen in S03. Deleting the entire masks would discard visible wall evidence. In S09 the source mask covers both sides of a three-high post, but the counter quadrilateral retains only 102 of the 378 side-face pixels that the mask found. S05's visible recessed wall lacks a trusted shared-edge measurement from its adjoining masks.

On the earlier shared-row Castle control, supplying an exact local grid to S05 yielded 48 correctly centered cells. Its relative position was unresolved, so the assembler exported none of them. Unioning the S13/S14 fragments with S03 without expanding the grid also left that control unchanged. Expanding the merged grid by one row/column moved correct placements from 186 to 189 and extras from two to four, and lowered exact position-plus-material F1 from 2.41% to 1.80%. Neither diagnostic justifies blanket merging.

## Placement

Filtering clouds preserved all six decoded exports from the registered-seam baseline. The six-view strict/facade placement score is 11.16% / 11.38%, with 239 correct, 3,802 missed, and five extra positions. Position-plus-material is 2.89% / 2.95%. Visible Builds v3 (4/144) scores 13.28% / 13.39% placement; Decorative Builds v1 (2/40) scores 1.77% / 1.91%. The fixed 104 was not rerun; the production trend is 91.27% / 93.42%. The row-angle results alone do not establish a placement change.
