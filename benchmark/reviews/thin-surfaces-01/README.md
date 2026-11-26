# Blocker Picker face eligibility

Only individual block faces with an effective width of at least 5 pixels in the source screenshot enter material matching and its accuracy denominator. For a projected quadrilateral, divide its area by the larger mean length of its two pairs of opposite edges. Exactly 5 pixels qualifies. Record complete, located faces below that threshold separately even when correct identification might have been possible.

In a counts build with material matching enabled, `material_face_counts` reports attempted and under-five totals. Each surface retains the coordinates and widths of its skipped faces in `under_five_px_faces` when any complete cells were evaluated. Surfaces with unresolved grids may have no measurable faces. Surface proposals, openings, dimensions and surface evaluation do not use this material cutoff.

| Frozen round 2 sample | Eligible | Under 5 px | Correct material picks / eligible |
| --- | ---: | ---: | ---: |
| Recorded block corners | 542 | 130 | 462/542 (85.2%) |
| Automatically recovered block faces | 58 | 0 | 58/58 (all polished andesite) |

Separately, 25 of 743 surface proposals are less than five pixels thick in the 56-view surface review. The whole-mask measure uses the narrowest enclosing strip. These are surface observations, counted independently of the block-face samples.

[Scores](material-widths.json) · [Surface examples](surfaces.png) · [Historical block-face examples](individual-faces.png)
