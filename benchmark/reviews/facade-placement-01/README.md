# Placement order, thin roofs and facade accuracy

The ordinary Java 26.3 placer recovered **20,510 of 23,702 visible positions** across the 104 fixed whole-structure views. Together, the shared roof-patch test and the final vertical-face check recovered 48 additional positions, lost none and added zero extras. Sixteen photographs improved and 88 tied. The 1,157 existing extras include 106 hidden source positions. [Annotated facade and roof examples](shipped-visibility/facade-examples.jpg) · [Ten weakest structures, stage by stage](worst/README.md)

## How placement works

1. **Locate surfaces and block spacing.** Regions First and the Wall Separator produce visible surface masks. The Surface Straightener and Block Counter measure each grid from texture repetitions or changes. The camera estimator uses pavement seams or floor atlas texture to establish perspective and metric scale. A failed metric camera leaves the build unresolved.
2. **Fit the surfaces in 3D.** Vertical planes can inform horizontal planes. For each surface the placer tests integer offsets using its image boundary, repeated texels, floor or wall contacts and junctions. It resolves competing planes together and projects the measured grids onto candidate cells.
3. **Assign pixels to blocks.** The placer resolves overlapping claims by depth and the face visible along the camera ray. It removes fully obscured placements, verifies edge and neighboring faces, checks thin horizontal faces against the atlas, and refines competing depths and exposed vertical faces. Several existing stages already use trained decision models.
4. **Check roof height and extent.** If the roof rim and texture both indicate a one-block height correction, the placer rebuilds the affected geometry at that height. It then verifies individual roof faces. Two photographed perpendicular edges can bound a rectangular proposal. To accept a new block, a single visible horizontal patch must touch at least two upper edge cells along each rim and supply pixels for that block. At least four of its support pixels must have been unclaimed, and the build must preserve the observed silhouette and previously visible blocks.
5. **Recheck exposed facade faces and export.** The existing vertical-face model checks the now-resolved scene at a score threshold of 0.95. It runs again only if it added a visible cell, for at most three passes. The exporter writes the accepted local block coordinates. Changing the order in which the final coordinates are written would leave this geometry unchanged.

## What changes the score

The optional-target rules below rescore **the same final exports** after the same single translation per structure. F1 balances correct placements, missed targets and extra blocks. Optional blocks are neutral whether recovered or missed. Every unmatched extra still counts. The 4 × 4 and 8 × 8 rules operate on the complete connected, visible, top-facing reference region at native 1440 × 900 resolution. A block stays required if a vertical face is visible, even when its top is a sliver. These diagnostics do not change the training dataset or authorize hidden placements.

| Score on 104 photos | Previous default | Shipped placer |
| --- | ---: | ---: |
| All visible positions | 90.30% | **90.41%** |
| Optional top-only block when its top region has no 4 × 4 patch | 90.76% | **90.88%** |
| Optional top-only block when its top region has no 8 × 8 patch | 91.53% | **91.64%** |
| Facade, requiring every block with any visible vertical face | 92.48% | **92.54%** |

Use the 8 × 8 result alongside strict geometry for this resolution. It matches the two visibly thin canopies in the examples. It makes 688 of 23,702 targets optional, of which the placer recovered 68. On Open Canopy 00 it changes the interpretation substantially: the build recovered all 27 facade blocks, made no extra placement, and missed 23 top-only blocks. Its strict F1 is 70.13%, and both the 8 × 8 and facade scores are 100%. The preferred threshold at another resolution or apparent building scale needs a separate check. [Per-photo final results](shipped-visibility/frames.json) · [All thresholds and ceilings](shipped-visibility/summary.json) · [Previous default with the same rules](../facade-visibility-01/README.md)

## Which methods earned a place in the default

| Method | Fixed 104 | Alternate 20 | Loading dock / switchback 4 | Shelter / gate 4 |
| --- | ---: | ---: | ---: | ---: |
| Previous default | 90.30% | 88.48% | 83.58% | 83.05% |
| Move vertical verification earlier | 90.20% | 88.27% | 83.76% | 83.12% |
| Final strict vertical verification alone | 90.36% | 88.53% | 83.58% | 83.33% |
| Shared visible top patch across both rims alone | 90.36% | 88.51% | 83.58% | 83.19% |
| **Shared roof patch, then final strict vertical verification** | **90.41%** | **88.55%** | **83.58%** | **83.47%** |

Both final orders gave the same predictions in the full comparison. The default finishes with the vertical check so any newly exposed facade candidates can use the final roof geometry. On the 72 development images the combination gained 17 visible positions. Alternate views gained five and the shelter/gate images gained six. There were no new extras in these groups. Across all 18 measured canopies the openings retained their widths and all 243 sampled sightlines stayed clear. Fixed roof cells at the correct offsets from both posts improved from 342 to 346 of 459. Camera-registered fixed F1 was also 90.41%. The historical 112-image score including eight retired texture controls was 89.66%. [Results and case provenance](results.json) · [Changed exports](comparisons.zip)

The newly captured low/tall canopy panel remained at 86.46% over four views. The 18 additional camera-failure views still yielded empty builds. Those cohorts measure transfer separately. The 104 fixed views have been used to compare revisions, and the tighter rectangle and verifier rules were developed after inspecting failures of the more permissive alternatives. The clean result applies to the tested images. The added geometry work used approximately 0.33–1.81 process CPU seconds per image across three timed examples. [Timings](timing.json) · [Verification of both orders and the ordinary build](verification.json)

**Keep unconditional rectangle filling experimental.** The two measured edge lengths alone proposed 1,670 missing positions. Scoring that hypothetical fill found 709 visible blocks, 837 hidden cells and 124 other errors. Its fixed F1 fell to 90.11%. No such completed roof was exported. Even the less restrictive image-mask variant gained 95 correct blocks but added nine hidden cells on one colonnade. The shared-patch rule is the part of `experiment/visible-roof-rectangle` that shipped. The strict final verifier is the part of `experiment/placement-order-facade` that shipped. Moving the existing verification earlier lost 91 fixed-suite positions while gaining 45. [Rectangle evidence and mechanism](../visible-roof-rectangle-01/README.md) · [Order experiments and illustrated facade gains](../facade-placement-order-01/README.md)

## Decision models and the next placement experiment

We trained two small offline selectors on the 72 development images and checked development transfer by leaving out each of six building families. Both were evaluated on the unchanged fixed set relative to the previous 90.30% default.

- **Choose one complete build per photo.** A one-split decision tree used the proposed additions, removals, neighboring cells and height change. Among the initial rectangle and final-wall variants, training chose the permissive edge-contact build on every image. It scored 90.51% and added nine hidden cells. Even a selector given the correct answer for each frame could achieve only 90.56% with those alternatives. Across five earlier geometry alternatives a learned switch fell to 90.22% on the fixed set. [First placement trials](first-trials-chooser.json) · [Earlier method pool](previous-method-chooser.json)
- **Choose individual proposed blocks.** A depth-two decision tree used which methods proposed a cell, adjacency along the facade and support above or below it. Training penalized a false placement five times as much as a missed addition. There was only one false addition in the development proposal pool, and the fitted tree accepted every proposal. The three initial variants together recovered 123 additional visible blocks, added ten hidden extras and scored 90.57%. Perfect selection among the same additional positions would score 90.60%. The current proposal pool cannot bridge the gap to 95%, even with the answers. [Cell features, fitted tree, family checks and oracle](cell-chooser.json)

**Generate better wall-plane candidates before training another chooser.** Of the remaining 3,192 misses, 1,699 have a visible vertical face, 985 show only a top and 508 only an underside. Of the 1,442 missed blocks with at least 64 visible wall pixels, 1,029 already have a single correctly oriented detected surface covering at least half those pixels (871 at three-quarters coverage). This narrows the next test to the grid origin, row and column count, integer plane depth, and which candidate claimed the pixels. A useful local model would choose among alternate small wall sections or abstain using texture transitions, grid alignment, contact with a neighboring plane and the source pixels it would expose or cover. First measure whether the correct geometry exists among those alternatives using the recorded truth only for scoring. [Wall mask coverage by photo](shipped-wall-masks.json)

Correctly recovering every missing top-only and underside-only block, with existing extras fixed, would bring strict F1 to 93.91%. Perfectly moving each matchable extra one grid step to a distinct missed position would reach 91.96%. Both calculations use known labels to measure the opportunity. [Breakdown and local-move bounds](shipped-visibility/summary.json)

## Replay

```powershell
python -m benchmark.facade_placement_eval --out runs/control --workers 6
python -m benchmark.facade_placement_eval --candidate benchmark.facade_rectangle_order:roof_then_walls --out runs/roof_then_walls --workers 6
python -m benchmark.facade_placement_eval --candidate benchmark.framed_geometry_order:production --out runs/production --workers 6
python -m benchmark.facade_visibility --run runs/roof_then_walls --out runs/visibility --workers 4
python -m benchmark.facade_mask_coverage --run runs/roof_then_walls --out runs/wall-masks.json --workers 4
```

Use `benchmark.facade_rectangle_order:walls_then_roof` for the reverse order. Use `--transfer-inputs ../framed-depth-transfer/runs/transfer-input` for the four shelter and gate photos. The [previous placement review](../roof-height-01/README.md) links the frozen upstream assets, additional 22-view challenge and ordinary screenshot command. The [two method-specific reviews](../visible-roof-rectangle-01/README.md) and [placement-order tests](../facade-placement-order-01/README.md) contain the source revisions, per-frame cases and exported geometry for the alternative methods.
