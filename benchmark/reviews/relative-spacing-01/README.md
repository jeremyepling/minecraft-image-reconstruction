# Roof position relative to the front posts

We decoded the previous and current builds on the same ten native canopy screenshots. Both methods recovered the opening width on all ten and aligned both front posts on all 52 visible post courses. With these posts as the origin, the current build placed **170 of 459 visible roof blocks** at the correct three-dimensional offset, up from 91. On the original failing canopy the two methods recovered the same four-block opening, but the previous roof had zero of 80 visible blocks in position relative to the posts. The current roof had 41. [Per-image coordinates, digests and measurements](cases.json)

| Visible relationship | Previous `a16668f` | Current `04563e0` |
|---|---:|---:|
| Correct gap width in seven reviewed canopies | 7 / 7 | 7 / 7 |
| Visible roof blocks correctly spaced from both posts, seven reviewed canopies | 74 / 362 | 127 / 362 |
| Correct gap width in three confirmation canopies | 3 / 3 | 3 / 3 |
| Visible roof blocks correctly spaced from both posts, three confirmation canopies | 17 / 97 | 43 / 97 |
| Exact roof-block-to-two-post-block triples across all ten | 470 / 2,323 | 878 / 2,323 |

One post course contains a visible reference block on each of the two front pillars at the same height. Each visible roof block and each complete course defines one triple. All three coordinates must match after one translation of the entire prediction. To use the posts as an origin, the scoring search maximizes matched post courses first, then correctly located roof blocks and finally overlap with the entire visible target. Ties use lexicographic offset order. The gap width is the median of the widest internal span on at least two measured post courses. A build with no matching post pair has unmeasurable width and receives zero roof relationship credit. The diagnostic measures recovery of relationships. The overall build score also penalizes extra geometry. [Native fixture and gap definitions](../canopy-fixtures-01/README.md)

Translating every one of the 20 decoded predictions by `(+42, −17, +31)` preserved every relationship measurement. In a controlled front view, shifting the entire canopy retained all 24 triples. Raising only its roof by one block retained the opening width but lost all 24 triples. The visible reference in this control has one roof course of six blocks and four courses of paired post blocks. [Illustrated control](control.png)

Run `python -m benchmark.relative_canopy_spacing --out runs/relative-spacing` from this checkout with the frozen native screenshots, visible references, prior outputs and released outputs restored in their benchmark locations. The runner verifies each pack's SHA256, decodes the actual template commands, verifies the recorded orientation and coordinates, and uses only reference blocks visible in the source view.
