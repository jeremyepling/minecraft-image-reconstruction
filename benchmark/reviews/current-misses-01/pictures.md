# Missed surfaces by screenshot

The Surface Finder matched 501 of 598 scored surfaces across 56 screenshots. 97 did not match.

An R number identifies one reference surface in that picture. A match requires at least 50% overlap, measured as the intersection divided by the union. Each prediction can match one surface.

A merge means one prediction covered at least half of two scored surfaces. A depth step is a merge between surfaces facing the same direction at different depths. The last two columns show where a saved MobileSAM outline could match the missed surface before and after filtering. These checks can overlap.

## Courtyard

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/holdout-courtyard-masonry-close-front` | 1 / 1 | None | — | — | — | — |
| `light_gray_concrete/holdout-courtyard-masonry-high-right` | 7 / 8 | R5 (top) | — | — | — | — |
| `light_gray_concrete/holdout-courtyard-masonry-low-left` | 5 / 5 | None | — | — | — | — |
| `light_gray_concrete/holdout-courtyard-plaster-close-front` | 1 / 1 | None | — | — | — | — |
| `light_gray_concrete/holdout-courtyard-plaster-high-right` | 7 / 8 | R5 (top) | — | — | R5 | — |
| `light_gray_concrete/holdout-courtyard-plaster-low-left` | 5 / 5 | None | — | — | — | — |
| `smooth_stone/holdout-courtyard-masonry-close-front` | 1 / 1 | None | — | — | — | — |
| `smooth_stone/holdout-courtyard-masonry-high-right` | 8 / 8 | None | — | — | — | — |
| `smooth_stone/holdout-courtyard-masonry-low-left` | 5 / 5 | None | — | — | — | — |
| `smooth_stone/holdout-courtyard-plaster-close-front` | 1 / 1 | None | — | — | — | — |
| `smooth_stone/holdout-courtyard-plaster-high-right` | 7 / 8 | R1 (vertical) | — | — | R1 | — |
| `smooth_stone/holdout-courtyard-plaster-low-left` | 5 / 5 | None | — | — | — | — |

## Crossing

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/holdout-crossing-masonry-high-right` | 9 / 11 | R13 (top), R15 (vertical) | R13, R15 | — | — | — |
| `light_gray_concrete/holdout-crossing-masonry-low-left` | 10 / 12 | R4 (vertical), R5 (vertical) | R4, R5 | R4, R5 | R5 | — |
| `light_gray_concrete/holdout-crossing-plaster-high-right` | 9 / 11 | R13 (top), R15 (vertical) | R13 | — | — | — |
| `light_gray_concrete/holdout-crossing-plaster-low-left` | 10 / 12 | R4 (vertical), R5 (vertical) | R4, R5 | R4, R5 | R5 | — |

## Galleries

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/holdout-galleries-masonry-high-right` | 22 / 24 | R7 (vertical), R16 (top) | R7, R16 | R7 | — | — |
| `light_gray_concrete/holdout-galleries-masonry-low-left` | 18 / 25 | R2 (vertical), R3 (vertical), R6 (vertical), R7 (vertical), R13 (underside), R21 (vertical), R22 (vertical) | R2, R3, R7, R21 | — | R21, R22 | R21, R22 |
| `light_gray_concrete/holdout-galleries-plaster-high-right` | 22 / 24 | R16 (top), R23 (vertical) | R16 | — | — | — |
| `light_gray_concrete/holdout-galleries-plaster-low-left` | 20 / 25 | R2 (vertical), R3 (vertical), R6 (vertical), R7 (vertical), R13 (underside) | R2, R3, R7 | — | — | — |

## Portico

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/holdout-portico-masonry-high-right` | 14 / 17 | R5 (vertical), R7 (vertical), R10 (top) | R5, R7 | R5, R7 | R7 | R7 |
| `light_gray_concrete/holdout-portico-masonry-low-left` | 18 / 20 | R2 (vertical), R5 (vertical) | R2 | — | — | — |
| `light_gray_concrete/holdout-portico-plaster-high-right` | 16 / 17 | R16 (vertical) | — | — | — | — |
| `light_gray_concrete/holdout-portico-plaster-low-left` | 17 / 20 | R2 (vertical), R3 (vertical), R5 (vertical) | R2, R3 | — | — | — |
| `smooth_stone/holdout-portico-masonry-high-right` | 15 / 17 | R5 (vertical), R7 (vertical) | R5, R7 | R5, R7 | R7 | R7 |
| `smooth_stone/holdout-portico-masonry-low-left` | 18 / 20 | R2 (vertical), R5 (vertical) | R2 | — | — | — |
| `smooth_stone/holdout-portico-plaster-high-right` | 15 / 17 | R5 (vertical), R6 (vertical) | R5, R6 | R5, R6 | R6 | R6 |
| `smooth_stone/holdout-portico-plaster-low-left` | 17 / 20 | R2 (vertical), R3 (vertical), R5 (vertical) | R2 | — | — | — |

## Registry

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/holdout-registry-masonry-high-right` | 8 / 10 | R6 (top), R9 (vertical) | R9 | R9 | R9 | — |
| `light_gray_concrete/holdout-registry-masonry-low-left` | 6 / 7 | R5 (vertical) | R5 | R5 | R5 | — |
| `light_gray_concrete/holdout-registry-plaster-high-right` | 7 / 10 | R6 (top), R9 (vertical), R10 (vertical) | — | — | R10 | — |
| `light_gray_concrete/holdout-registry-plaster-low-left` | 6 / 7 | R4 (vertical) | R4 | R4 | R4 | R4 |

## Shopfront

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/dev-shopfront-masonry-high-right` | 14 / 14 | None | — | — | — | — |
| `light_gray_concrete/dev-shopfront-masonry-low-left` | 13 / 16 | R9 (vertical), R10 (vertical), R14 (vertical) | R9, R10, R14 | R10, R14 | R9, R10, R14 | R14 |
| `light_gray_concrete/dev-shopfront-plaster-high-right` | 10 / 14 | R5 (vertical), R10 (top), R13 (top), R19 (vertical) | — | — | R5, R10, R19 | — |
| `light_gray_concrete/dev-shopfront-plaster-low-left` | 11 / 16 | R3 (vertical), R10 (vertical), R13 (vertical), R14 (vertical), R15 (vertical) | R10, R13, R14, R15 | R13, R14, R15 | R10, R13, R14, R15 | R10, R14, R15 |
| `smooth_stone/dev-shopfront-masonry-high-right` | 14 / 14 | None | — | — | — | — |
| `smooth_stone/dev-shopfront-masonry-low-left` | 13 / 16 | R9 (vertical), R10 (vertical), R15 (vertical) | R9, R10, R15 | R10, R15 | R9, R10, R15 | R15 |
| `smooth_stone/dev-shopfront-plaster-high-right` | 12 / 14 | R5 (vertical), R6 (vertical) | R5, R6 | R5, R6 | R5, R6 | R6 |
| `smooth_stone/dev-shopfront-plaster-low-left` | 11 / 16 | R3 (vertical), R9 (vertical), R10 (vertical), R15 (vertical), R16 (vertical) | R9, R10, R15, R16 | R9, R10, R15, R16 | R9, R10, R15, R16 | R9, R10, R15, R16 |

## Wing

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/dev-wing-masonry-close-front` | 1 / 1 | None | — | — | — | — |
| `light_gray_concrete/dev-wing-masonry-high-right` | 4 / 5 | R3 (top) | — | — | R3 | — |
| `light_gray_concrete/dev-wing-masonry-low-left` | 3 / 3 | None | — | — | — | — |
| `light_gray_concrete/dev-wing-plaster-close-front` | 1 / 1 | None | — | — | — | — |
| `light_gray_concrete/dev-wing-plaster-high-right` | 5 / 5 | None | — | — | — | — |
| `light_gray_concrete/dev-wing-plaster-low-left` | 3 / 3 | None | — | — | — | — |
| `smooth_stone/dev-wing-masonry-close-front` | 1 / 1 | None | — | — | — | — |
| `smooth_stone/dev-wing-masonry-high-right` | 5 / 5 | None | — | — | — | — |
| `smooth_stone/dev-wing-masonry-low-left` | 3 / 3 | None | — | — | — | — |
| `smooth_stone/dev-wing-plaster-close-front` | 1 / 1 | None | — | — | — | — |
| `smooth_stone/dev-wing-plaster-high-right` | 4 / 5 | R2 (vertical) | — | — | R2 | — |
| `smooth_stone/dev-wing-plaster-low-left` | 3 / 3 | None | — | — | — | — |

## Workshop

| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |
| --- | ---: | --- | --- | --- | --- | --- |
| `light_gray_concrete/holdout-workshop-masonry-high-right` | 14 / 16 | R6 (top), R19 (vertical) | — | — | — | — |
| `light_gray_concrete/holdout-workshop-masonry-low-left` | 6 / 14 | R1 (vertical), R3 (vertical), R4 (vertical), R5 (vertical), R6 (vertical), R8 (underside), R12 (vertical), R13 (vertical) | R1, R3, R4, R5, R6, R12, R13 | R1, R3, R4, R5, R6, R12, R13 | R1, R12, R13 | R13 |
| `light_gray_concrete/holdout-workshop-plaster-high-right` | 12 / 16 | R15 (vertical), R17 (vertical), R18 (vertical), R19 (vertical) | R15, R17, R18, R19 | R15, R17, R18, R19 | R17 | — |
| `light_gray_concrete/holdout-workshop-plaster-low-left` | 7 / 14 | R1 (vertical), R3 (vertical), R4 (vertical), R5 (vertical), R6 (vertical), R8 (underside), R13 (vertical) | R1, R3, R4, R5, R6, R13 | R1, R3, R4, R5, R6, R13 | R13 | R13 |
