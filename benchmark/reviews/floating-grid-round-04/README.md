# Round three of floating grid feedback

Open [the three annotated comparisons](review.html). Each case includes the source image, native evidence overlay, competing grids, and the recorded visible reference. The Viaduct charts have identical tangent grids because the difference is in assembly. Mill uses the same grid on both sides because the source sky classifier missed the gray clouds.

The 84-condition strict placement F1 was 33.8054% before these experiments. Repeated-texture attachment reached 33.8804% with 53 more correct positions and 18 extra blocks. Geometry-only attachment reached 33.8859% with 57 more correct positions and 20 extras. A reference-bounded attachment reached 33.8936%, with 58 more correct and no extra blocks. The fixed six sky views tied for these attachment experiments. Compare each arm independently with the small-steps baseline.

The narrow projected-sky veto removed 10 extras without losing a correct position across 84. Nine removals came from the fixed six. The broader sky and cloud cue lost 13 net correct positions and removed 65 net extras. Neither addressed the five gray-cloud extras in the annotated Mill view. Next test source-connected gray-cloud regions alongside real light-colored building faces. First prioritize joining Twinspire's detached brick chart with independent evidence for its depth.

The six-view floating integration baseline remains 12.64% strict and 12.90% facade placement, and 4.68% and 4.77% position plus material. These research variants have separate paired baselines. Production and integration were not rerun or modified.

## Pending annotation

[Open the zoomable mobile PNG](mobile-annotation.png). Reply with the IDs below or mark up the PNG. For each grid, record coverage, alignment, and block size separately. If comparing alternatives, record your preference on its own.

| ID | What to mark | Status |
|---|---|---|
| T1 | Mark visible brick fragments to keep, cyan cells spanning two bricks, and any visible edge that could establish the wall connection. | Pending |
| M1 | Trace the roof edge against cloud. Circle pale material to preserve and flag any missed roof edge outside the red boxes. | Pending |
| V1 | Check the second base width and post height or depth. Mark any grid crossing a true opening. | Pending |

## Where the extras land

The six-sky integration replay has five extras. The expanded research baseline has 1,542 on those six and 11,258 across 84. All counts use the same one-translation-per-build visible placement comparison. Each row in the coordinate inventory is a decoded extra voxel. The camera enters only this audit. We project the registered cube center onto the recorded visible face map and compare the depth of the intersected source face with the cube. The resulting location describes where the error lands. Root causes require tracing the originating chart.

| Location in the 84-view expanded baseline | Extras |
|---|---:|
| Behind the visible building face | 3,889 |
| In front of the visible building face | 2,309 |
| Background below the horizon | 1,842 |
| Outside the source image | 1,533 |
| Sky or cloud | 1,467 |
| Intersects the visible face at the same ray | 218 |

2,399 of these coordinates are occupied in the source world readback but have no visible target for that view. Stable Yard 1 contributes 2,563, Castle 02 contributes 2,335, and Castle 01 contributes 1,832. Together they account for 59.8% of expanded baseline extras. Stable Yard 1 starts with four atlas cells on the selected wall in each view. The wall expands to 1,445 and 1,261 exported cells and supplies 1,365 and 1,198 extras. Most land off image or in background. Pillars 01 similarly expands seven cells to 756 exported cells and adds 687 extras. Castle 02 ashlar high-right starts with 10 atlas cells and expands to 1,657 exported cells on one wall. Of its 613 extras, 571 are in front of or behind the visible surface.

The repeated-texture attachment adds 18 extras. In Stable Yard 2, it copies the oak post depth into the two-by-four sandstone panel behind it. The sandstone lands two blocks too far forward and one block sideways. In Walled Garden 1, it places the tuff faces between the green posts at the wrong depth and moves them sideways and down. The lower group lands two blocks too far forward. Two cells by the upper opening land six blocks too far forward. All 18 had wall pixels at all nine source sample points. Sixteen of the final positions lack a recorded block, and two hit a hidden block. The later Viaduct extent fixes remove seven extras on that view, leaving the full column candidate with 11 more extras than its paired baseline.

[Traced source faces for the 18 attachment extras](attachment-extras.jpg) · [Counts by view](extras-by-view.csv) · [All extra coordinates and emitting charts](extra-coordinates.csv.gz)
