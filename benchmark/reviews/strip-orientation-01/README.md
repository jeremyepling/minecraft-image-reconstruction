# Thin roof strips, face direction and split surfaces

**For facade-priority evaluation, a visible block is optional when it shows only a top or underside and its individual face has no fully visible 4×4-pixel patch at the native 1440×900 resolution.** A visible wall keeps the block required. Optional correct placements and optional misses are both neutral. Every extra still counts. The reported view continues to supply all the evidence available to the placer. [Colonnade source and measured strip](sliver-colonnade.jpg) · [Rules and threshold sensitivity](slivers.md)

The Java 26.3 default at `edb10a3` scores **92.56%** on this measure and **90.50%** on strict geometry across the same 104 framed photographs. Neutralizing thin horizontal faces accounts for **1,305 of 3,167 misses** (41.2%): 849 top-only and 456 underside-only. It also neutralizes 1,821 correct placements. **24 of 1,146 extras** project predominantly onto those regions. This gives their image location; it does not prove the thin surface caused the extra. Treating every direction this way would neutralize 340 additional wall misses. An area cutoff of fewer than 16 pixels identifies only 427 misses because a long thin strip can have many pixels.

On Colonnade 01 right, **61 of the 65 misses** are top-only blocks that lack even a 2×2 patch on their individual faces. With the 4×4 rule four required misses and all 18 extras remain. The score becomes **88.89%** versus 75.22% strict. The previous 8×8 test operated on an entire connected top surface: one thicker patch kept other cells on that surface required. It neutralized seven misses in this view, giving 76.40%. Both evaluations use identical exports. On the opposite angle, neutralizing 67 top misses leaves two misses and 53 extras, giving 76.60%. Removing tiny surface *detections* in an earlier experiment lost 42 genuine matches and three good placements on the pictured Shopfront. [Earlier pruning experiment](../wall-texture-seams-01/README.md)

## Which face is pointing the right way?

Look at the center of the white circle in the same photograph, recorded geometry and rendered build. The Facade example has a Z wall in each, but the rendered wall is one block away along Z. On the Colonnade roofline the recorded top face becomes a vertical X wall. On the Galleries overhang the recorded underside becomes a Z wall. The rendered first surface is the one that blocks the camera's view along that ray. This is a different diagnostic from what direction the Surface Finder assigns to an image mask earlier in the pipeline. The fixed-set error budget compares the three absolute axes; the examples additionally identify the signed top and underside. [Two examples in the source image](faces/depth-and-direction.jpg) · [All five examples, including a three-face corner](faces/face-directions.jpg)

A photographed block can show one, two or three visible faces. The faces of one block share its coordinates. Their projected corners, texture grids and relative depths can constrain its location, tile spacing and which neighboring row is visible. A change in direction also occurs at an inside corner or where parts overlap at different depths. Across the fixed views, 4,724 perpendicular face contacts with at least four native pixel edges join faces of the same cube; 726 join face pairs on distinct immediately adjacent blocks, and 4,282 join more widely separated block coordinates. Contacts count face pairs observed in each view.

| Visible faces on one block | Placed correctly / visible | Recovered |
| --- | ---: | ---: |
| One | 16,179 / 19,174 | 84.4% |
| Two | 4,207 / 4,376 | 96.1% |
| Three | 149 / 152 | 98.0% |

This association also reflects the amount of evidence and the kinds of blocks visible from each camera. Production already uses neighboring face evidence, shared roof contacts, independently measured pillar planes and native texture checks for panel placements. There are 172 missed blocks showing multiple faces, of which 75 have majority correctly oriented mask support on at least two individual faces. These make a focused set for testing a joint corner fit. [Native contact counts, chosen source pixels and per-view provenance](faces/face-contacts.json) · [Current post and panel geometry](../visible-error-budget-01/README.md)

## Does reinterpreting the same strip recover important blocks?

The image-only experiment tested nearby wall planes and physically visible tops or undersides on the same rays, using texture-atlas agreement on held-out samples, the rim and grid, and visibility after rendering all proposals together. The broad chooser recovered seven visible blocks on fixed: two required and five optional. It added twelve extras, three hidden. Facade-priority F1 fell from **92.561% to 92.539%**. Restricting placement to horizontal winners recovered one optional underside block and added zero extras. Requiring independent corroboration tied. None improved Colonnade 01. [Frozen experiment, held-out results and replay](../strip-face-directions-01/README.md) · [Paired facade-priority scores](priority-directions.json)

The earlier 632 count meant that a detected mask covered most of each missed horizontal-only block's footprint, but predicted horizontal masks did not. **608 of 632** have majority coverage explicitly marked vertical. Under the individual 4×4 rule, **615 are optional**, leaving three substantial tops and fourteen undersides. Ten of these seventeen have majority explicit vertical coverage. For those seventeen, proposed strips touched twelve, the hypotheses contained eleven correct positions, and a chooser supplied with the answers retained one after joint visibility checks. The actual selectors recovered none of the seventeen.

## How much does splitting one surface matter?

A single recorded surface is an edge-connected group on the same plane, with the same signed normal. Material changes can occur within it. Under the standard substantial-fragment rule, **56 of 1,462 surface observations** (3.83%) were split into 117 neighboring masks. They appeared in 42 of 104 pictures. Tightening or loosening the thresholds gave 44–77 affected surfaces. **89** missing block coordinates had a visible face somewhere on a split surface; **57** had source pixels covered by the qualifying pieces; **15** touched the internal seam. These are nested associations over the strict 3,167 misses.

The image-only texture comparison rejoined the false Facade diagonal, but the fixed-suite exports tied. Supplying the correct groups as a diagnostic and running the real counter and placer recovered eighteen required positions and removed fifteen extras while adding two. All fixed gains came from one photographed plaster Shopfront, whose merged surface itself had no missing blocks. Its chart affected placement elsewhere. This is a reason to track joins on Shopfront and similar layouts separately. It would overstate their global importance to attribute every miss on a fragmented plane to the split. [Definitions, real joins, causal intervention and examples](../coplanar-fragments-01/README.md)

## Surface Finder changes in ordinary builds

On the Facade photograph, Surface Finder now removes a false diagonal between two sibling masks when the photographed texture continues across it and the seam runs across both projected wall directions. Its best mask covers **98.35%** of the recorded wall, up from **81.64%**, and mask IoU improves from **73.48% to 88.27%**. The Block Counter now measures the recorded **27 × 11** span instead of **28 × 12**. The union reassigns 31,234 correct wall pixels to the matched mask. It also inherits 598 off-target pixels from the original fragment, so the wall mask still includes some neighboring surfaces. The whole-image footprint, holes and background coverage are preserved. The original and repaired exports contain identical block coordinates and materials. [Before, after and recorded face](wall-extent.jpg) · [Coverage and residual errors](wall-extent.json) · [204-frame validation](../guarded-surface-outlines-01/README.md)

A separate early rule follows an upward-viewed wall to its clear sky edge when the unowned band has continuous wall color. On Watchpost 3 it added **225 previously uncovered pixels**, all belonging to the recorded wall. The normal RGB build produced identical counts and all 384 blocks in the same locations. Each rule fired once across 204 images. The Facade is a previously inspected fixed-suite case and the Watchpost is a development case. The evaluation therefore establishes two specific repairs with limited evidence for how often they will help another building. [Highlighted Watchpost roofline and evaluation](../visible-skyline-01/README.md)

Both guards run before the Surface Finder saves outlines and crops and before counting or placement. Full RGB builds and standalone surface runs produced identical outlines and all 60 source and context crops on the two changed images. All 386 repository tests passed. The broad texture-only merge lost three matched faces on the new-structure test and damaged geometry on the independent shelters. A cross-material rule repaired the pictured Shopfront but lost a correct block and added three extras on a Switchback. These broader rules stay experimental. [Combined ordinary-entry checks](combined-entry.json) · [Measured join trade-offs](../coplanar-fragments-01/README.md)

## Remaining placement work

The 4×4 rule leaves **1,862 required misses**, of which **1,674 have a visible wall** (89.9%). Focus next on wall extent, tile count and local depth on Shopfronts, facades and reliefs; keep the substantial top and underside misses as a secondary test set. On the fixed eight Shopfront views, 215 required misses and 143 extras remain. On the shown Colonnade view, correcting extras is a larger opportunity than placing the four remaining required blocks. The original 104-view ranking and the new ranking use the same grouping of 40 physical layouts. [Facade-priority ranking with all frame IDs](priority-layouts.json) · [Current strict placement budget](../visible-error-budget-01/README.md)

The normal placer consumes the repaired masks and their recomputed counts. The mask improvements tie the recent post and panel geometry scores. The historical comparison including the eight retired texture-only frames is 89.74% strict. The facade-priority diagnostic is reported on admitted structure views only.

## Replay

```powershell
python -m benchmark.strip_orientation_eval --out runs/control --workers 4
python -m benchmark.strip_visibility_audit
python -m benchmark.strip_face_examples
python -m benchmark.facade_priority_ranking
```

The linked challenger reports contain their candidate and transfer commands. Run `benchmark.strip_priority_comparison` with named candidate directories to compute the paired 4×4 scores. All reported world coordinates, source labels and reference-assisted joins are restricted to grading or explicitly labeled diagnostic interventions.
