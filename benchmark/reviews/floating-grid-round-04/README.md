# Round three of floating grid feedback

Open [the three annotated comparisons](review.html). Each case includes the source image, native evidence overlay, competing grids, and the recorded visible reference. The Viaduct charts have identical tangent grids because the difference is in assembly. Mill uses the same grid on both sides because the source sky classifier missed the gray clouds.

The 84-condition strict placement F1 was 33.8054% before these experiments. Repeated-texture attachment reached 33.8804% with 53 more correct positions and 18 extra blocks. Geometry-only attachment reached 33.8859% with 57 more correct positions and 20 extras. A reference-bounded attachment reached 33.8936%, with 58 more correct and no extra blocks. The fixed six sky views tied for these attachment experiments. Compare each arm independently with the small-steps baseline.

The narrow projected-sky veto removed 10 extras without losing a correct position across 84. Nine removals came from the fixed six. The broader sky and cloud cue lost 13 net correct positions and removed 65 net extras. Neither addressed the five gray-cloud extras in the annotated Mill view. Next test source-connected gray-cloud regions alongside real light-colored building faces. First prioritize joining Twinspire's detached brick chart with independent evidence for its depth.

The six-view floating integration baseline remains 12.64% strict and 12.90% facade placement, and 4.68% and 4.77% position plus material. These research variants have separate paired baselines. Production and integration were not rerun or modified.

## Annotation checklist

[Open the zoomable mobile PNG](mobile-annotation.png). Reply with the IDs below or mark up the PNG. For each grid, record coverage, alignment, and block size separately. If comparing alternatives, record your preference on its own.

| ID | What to mark | Status |
|---|---|---|
| T1 | Twinspire cyan grid beside the sandstone. | Reviewed. Jeremy said the cyan cells look right and should follow the gold reference. No further grid sign-off requested. Follow-up is to trace the placement handoff. |
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

## Twinspire placement follow-up

Jeremy confirmed the cyan grid shown in T1 as the intended grid. It is actual output from the experimental grid stage. The standard export has 173 blocks from the two walls of the right tower. The left tower's two brick walls already join each other on two consecutive grid edges with 1.79 pixels average worst-endpoint error. They form a separate 70-block group.

The missing link is between the two towers' south-facing brick walls. They share the same recorded plane. Projecting four outer cell centers from the left grid into the right grid proposes the same integer offset each time, six rows and 15 columns in the signed local axes. The maximum error is 0.308 block against a 0.13-block limit, so the join never reaches assembly. The subsequent image check would also fail. Its maximum error is 8.05 pixels against a 0.50-pixel limit. There are no depth vetoes or graph conflicts. The east-facing walls face the same way but are 11 blocks apart in depth.

We replayed the ordinary assembler and pack export, supplying only the relative offset from visible reference data. The selected chart packet, whole-build orientation, and translation stayed fixed. The replay on the earlier grid recovered 65 correct blocks. The replay on the approved grid recovered those 65 plus the five new partial cells, with zero correct placements lost and zero new extras. Strict and facade placement rose from 47.11% and 49.22% to 67.04% and 69.63%. Exact position-plus-material rose from 17.99% and 18.79% to 39.11% and 40.62%.

The proposed offset already rounds correctly from the source image. The next source-only test is to validate whether the two separated walls share a plane using an independent cue before accepting the offset. Hold the existing tolerances for cases with no corroboration. The replay is reference-only evidence for this case. The floating default remains the previously scored 12.64% strict and 12.90% facade on six sky images.

[Zoomable before and after](twinspire-join.png) · [Decode receipts](twinspire-join-scores.json)

## Test the right tower's block size

[Open the left-to-right scale comparison](twinspire-scale-test.png) or [see exactly where the extra blocks landed](twinspire-right-extras.png). The left wall where the bridge attaches and the right tower's outer side wall face the same direction, with 11 blocks between their planes. The front-facing brick walls are the coplanar pair. The right outer side has 8 columns and 10 rows. The shared-corner step kept its vertical seam with the right front but replaced its cross-wall scale. Sampling the replacement across the right wall gave 143 cells instead of 80. The 50 extra cells from five surplus columns and the 13 below the wall all landed at the correct wall depth.

The experimental transfer uses the left side wall's block ruler and the right tower's local vertical seam, proposes different widths using the visible right edge, and chooses by how well repeating brick texture matches on the right. It emits 8 columns and 10 rows from the image. The native reference, used to grade the result, finds 80 distinct faces under those 80 projected centers. Before transfer, 143 centers covered 79 distinct faces. We also bypassed the corner override and restored the right wall's own earlier atlas grid using the same downstream sampling and export. That was almost as good. The corner can support vertical alignment without overwriting the independently measured horizontal ruler.

| Twinspire high-right arm | Correct / missed / extra | Strict / facade placement F1 | Exact positions with material |
|---|---:|---:|---:|
| Existing experimental grid | 110 / 184 / 63 | 47.11% / 49.22% | 42 |
| Restore the right wall's earlier atlas grid | 110 / 184 / 0 | 54.46% / 57.29% | 88 |
| Transfer the left wall's ruler | 110 / 184 / 0 | 54.46% / 57.29% | 90 |
| Ruler transfer plus the reference-only tower offset | 180 / 114 / 0 | 75.95% / 79.30% | 153 |

On the opposite Twinspire view, the transfer gained 12 exact position and material matches with the same geometry. On six fixed sky views the rule did not fire. Keep the transfer on the experimental branch until there are positive cases from other buildings.

### Evidence for surface relationships

Keep each surface's direction, visible support, proposed depth and independently measured grid. Depth ordering guides which same-facing patches may share a plane. Test the relationships against the source boundaries and local grid proposals before assembly consumes them.

| Relationship | What assembly can use |
|---|---|
| Adjacent faces sharing a physical edge | Match visible edge endpoints and corresponding block rows. Preserve each face's independent spacing across its width until its own texture or silhouette verifies a candidate. |
| Parallel faces at different depths | Transfer a unit block size after perspective adjustment. Keep the offset between the planes independent. Check the resulting phase and block count against the receiving wall's visible pixels. |
| Coplanar fragments separated by an opening or occluder | Check that the depths, scale, and grid phase agree at multiple visible locations. An accepted hypothesis shares the plane and integer grid offset. Only cells with visible support enter the export. |

The source-only ruler test implements part of the parallel-wall rule. The Twinspire left-to-right placement still uses a reference-only offset in the combined diagnostic. A future candidate can use the same-plane checks to establish that offset from the source image.
