# Feeding visible material evidence back to the Block Counter

This Java 26.3 experiment starts at reconstruction revision `5818a37` and uses its exported block IDs, estimated camera and current source masks. The first 72 development views set the rules. The candidate definitions were committed at `414b98c` before grading the other 132 views. The 104 fixed views and the other 28 controls retain their existing building groups.

## What the second pass measures

1. Consider a wall with a sampling chart where the ordinary counter left at least one dimension unresolved. Project the final cubes through the estimated camera. Require a single visible wall plane to cover at least 70% of the wall mask.
2. Back-project that mask onto the plane. Its outer extent on each axis must land within 0.28 blocks of an integer boundary. Preserve every dimension already measured.
3. For each supported frontmost face, use its **exported ID** to look up allowed side sprites. Compare source texels with those sprites at the proposed block scale and at half and double that scale. Allow orientation, one-texel shifts and reduced texture resolution. A cube votes only if the unit-scale correlation reaches 0.46 and exceeds the alternative scales by at least 0.085.
4. Compare accepting the plane alone, requiring votes from at least two cubes and half of informative cubes, requiring both ends with at least 60% of the intermediate offsets on each accepted axis, and requiring a vote at **all four corners** of the proposed extent. The ends rule can report one supported axis. No challenger adds a cube to the observed material map.

The scorer matches detected masks to native reference surfaces one-to-one at intersection over union ≥0.5. For each matched visible wall, the target width and height span the extreme observed Minecraft face coordinates, including visible blocks with partial faces. The native face chart gives the across/down ordering. Unmatched proposals and wrong-orientation proposals are counted separately. A predicted width and height describe a wall's span; an opening within that span can still be empty.

## Measured wall counts

On the 104 fixed views, **878 matched wall charts** were eligible for measurement. The ordinary counter already returned both dimensions for 426 and both were exact for 287. The stricter four-corner rule returned both for 496, including 349 exact pairs. Across the 28 independent controls the exact count rose from 66 to 88 among 262 matched wall charts.

| Feedback required | Fixed added correct pairs | Fixed added wrong pairs | Fixed unpaired or wrong direction | Other 28 added correct / wrong / unpaired |
| --- | ---: | ---: | ---: | --- |
| Coplanar final cubes | 185 | 56 | 24 | 68 / 13 / 7 |
| Material supports unit scale | 137 | 35 | 4 | 48 / 11 / 2 |
| Verified end cells per axis | 90 | 12 | 3 | 36 / 5 / 2 |
| Four verified corners | 62 | 8 | 2 | 22 / 0 / 1 |

On the fixed views, four verified corners added **61 correct widths and 3 wrong widths**, plus **59 correct heights and 5 wrong heights**. The per-axis end check added **117 correct widths and 7 wrong**, and **102 correct heights and 13 wrong**. Fifty-five matched proposals from the end check supplied just one new axis and still lacked the other. On development four corners added 41 correct pairs, 2 incorrect, and 3 unpaired. Every new measurement fills a previously unresolved axis.

Shopfront shows the limitation. On the masonry high-right view the feedback correctly measures a 1 × 5 wall. On the plaster low-left view it completes a 1 × 5 chart whose matched reference spans 3 × 6. Visible texture on the four corners of a fragment can be genuine even when its mask stops short of the full surface. In Crossing's plaster high-right view, the ends confirm the correct height of 3 on `mask-6` while withholding a proposed width of 4. The reference width there is 3. [Shopfront source overlays](shopfront-counts.jpg)

[Development scores](development-summary.json) · [Fixed and independent scores](confirmation-summary.json) · [Development surface records](development-cases.json) · [Fixed and independent surface records](confirmation-cases.json)

## Replay

Run `python -m benchmark.material_feedback_grid --set development --workers 5`, then repeat with `--set confirmation`. The runner uses the published typed sites, the frozen stage replays, and native visible face-ID maps. `python -m benchmark.material_feedback_grid_sheet` recreates the source comparison.

