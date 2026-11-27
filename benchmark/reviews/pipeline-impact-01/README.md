# Which pipeline changes reach the exported build?

[Open the interactive stage impact assessment](viewer.html) to compare the same source images under each intervention. Choose placement, exact material ID, or overall F1 to see which stage can use a correction.

The cohort scorecard now shows the default whole-tile and wider boundary-guided camera fallback: fixed strict/facade placement 91.27%/93.42%, v3 21.84%/22.75% and decorative 63.46%/65.23%. The camera table lists the remaining 70 v3 failures and their latest fallback reasons. Experiment plots retain their own original paired baseline; the joint camera/grid plot uses the new default as its control. GeoCalib, MoGe and joint refinement remain experimental. [Camera-model round and validation](../camera-model-round-01/README.md)

## Dataset coverage

The **104 fixed** score views are part of the established **204-image** companion collection, which includes the 144 whole-structure Visible Builds v2 frames. The 104, the smaller sampled replays and the 28 comparison views are subsets of the established collection. The independent pixel and perfect-export controls cover all 204.

The 144 new Visible Builds v3 views expand the original 204-image collection to 348. The historical stage interventions below used the original collection. The separate decorative material-removal check covered eight of 40 views and lost three real blocks while removing zero extras. We subsequently ran the ordinary pipeline on **all 144 v3 and 40 decorative views**. The scorecard lists camera failures, misses after a supported camera, and strict and facade results alongside the original 104. The capture-lighting pairs remain a separate diagnostic. [Expanded baseline](../expanded-baseline-01/README.md) · [Camera failures by source image](viewer.html#camera-failures) · [Original decorative transfer evidence](../material-feedback-geometry-01/decorative.json)

## Work order

1. **Repair the visible surface and its direction where parts overlap.** Supply competing outlines for a recess, an opening, an occluding pillar, or a mask containing more than one depth. Check which proposal preserves both the front edge and the independently visible rear face. Give the old outline a vote. A single material can span a corner or continue along a parallel wall behind a pillar.
2. **Improve recognition of hard visible faces alongside geometry.** Even with exact coordinates and the recorded camera, the existing matcher produced 1,813 wrong IDs on the fixed suite. Of these, 898 were facade-required and 915 belonged to optional horizontal slivers. The latter can still justify a cube even when very few pixels distinguish its exact material. Test hard negatives under varied lighting on separate buildings. Keep material confidence distinct from confidence that a cube exists.
3. **Connect a verified grid to an addressable wall plane.** Preserve a reliable count, its sampling corners, both grid offsets, the candidate face direction and its depth together. Test whether projecting a neighboring rim or another visible face supports that combination. On Portico a genuine mask repair helped, a changed sampling chart lost part of that gain, and correct counts recovered it. Count-only improvements often leave the export unchanged.
4. **Use camera and depth feedback selectively.** In the 20-view test, perfect camera calibration alone added 0.88 percentage points to overall strict F1. In the smaller depth test, the correct dominant depth was among the candidates for all 149 eligible masks. Compare competing local surfaces before assigning one depth to a region containing a front pillar and a wall behind it.

## Full fixed confirmation of the first priority

Correcting only visible pixel ownership raised strict placement **91.38% → 94.76%**, facade placement **93.52% → 96.55%**, strict overall **84.67% → 87.54%**, and facade overall **88.43% → 91.34%** on all 104 fixed views. It recovered 1,272 correct positions and lost 309, removed 810 extras and added 256. Placement improved on 84 views, tied on seven and regressed on 13. All inherited orientation, grid and depth estimates still had to work with the new masks.

The paired control is the public-context replay. It reproduced 83 of the 104 published packs byte for byte. On just those 83, strict placement still rose **91.85% → 94.52%**. The shipped fixed-suite score remains **91.27% strict geometry** (20,773 correct / 2,929 missed / 1,046 extra), **93.42% facade geometry**, **84.68% strict overall** and **88.51% facade overall**. [All 104 paired exports and shape summaries](../reference-surface-feedback-01/fixed-confirmation.json)

## Correct an earlier stage, then actually rerun later stages

Twenty source images were selected in advance by source hash, two from each of ten splits. Twelve are from the fixed evaluation, eight are comparison views including two development views. Each row below reruns from the stated handoff through a decoded material pack. Directions, chart corners and counts build on the preceding surface intervention. Camera alone retains the original surfaces. Placement and overall F1 both use the same relative alignment for each exported build.

| Information supplied | Strict placement F1 | Strict overall F1 | Facade overall F1 |
| --- | ---: | ---: | ---: |
| Paired baseline | 89.95% | 86.42% | 89.32% |
| Correct visible surface masks | 95.19% | 91.79% | 94.38% |
| Also correct face directions | 95.88% | 92.50% | 94.61% |
| Also correct sampling-chart corners, let the counter measure | 95.49% | 92.07% | 94.40% |
| Also give the counter the correct grid dimensions | 95.92% | 92.52% | 94.62% |
| Recorded camera alone | 90.87% | 87.30% | 89.87% |
| All the above together | **97.13%** | **93.47%** | **95.34%** |

The combined intervention still leaves plane depth, candidate generation, visibility checks and material identification to the pipeline. It improved placement on all 20 views. Eighteen control packs reproduced the published pack byte for byte. Restricting the comparison to those 18 gives strict placement **89.76% → 94.87%** for masks alone and **97.04%** for the combined intervention. Private geometry context differs from the paired replay on some views. Percentage-point changes use the corresponding replay baseline. [Per-view controls and interventions](../reference-surface-feedback-01/README.md) · [Portico source-image comparison](../reference-surface-feedback-01/portico.jpg)

The counter got both visible spans right on **52 of 322** reference surfaces at baseline, **132** with exact masks, **134** with exact directions too, and **172** when also given the exact projected sampling chart. Thus the counter still missed a dimension or returned an incorrect one on 150 perfectly charted surfaces. The placer can often recover visible cubes via other evidence. A correct count does not automatically locate them in depth. [Counter handoffs](../reference-surface-feedback-01/count-handoffs.json)

Canopy 04 illustrates why each handoff matters. Replacing the masks while inheriting the estimated face directions took the build from **41 correct positions to 32**. Supplying the correct directions as well brought it to **47**. With the full set of early corrections it reached **52 correct and zero extra**. The drop identifies a direction assignment to investigate.

## How exact were the answers?

We independently cast a ray from every native pixel center through the recorded camera and intersected it with the recorded Minecraft unit cubes. Across **204 frames and 264,384,000 pixels**, the first-hit block and face agreed with the reference at every pixel. The check covered 37,273,981 foreground pixels, 47,633 visible block appearances and 57,685 visible face appearances. The visible XYZ sets and block IDs matched the world readbacks. Re-encoding all 3,243 reference surfaces as masks and reading them back changed zero pixels.

We also passed the exact visible positions **and** block IDs through the ordinary pack writer, decoded the resulting Minecraft packs, and transformed them back to world coordinates. All **47,633** visible block placements across 204 views returned with the correct ID, zero omissions and zero extras. Both strict and facade overall F1 were **100%**. This positive control measures the exporter and scorer. A stage supplied with perfect masks or corners still has to infer the remaining information. [Per-frame perfect export controls](../material-feedback-surfaces-01/perfect-export-control.json)

Using the capture's raw projection matrix, we also reprojected the 3D corners independently. Maximum disagreement was below **0.000000001 source pixels** for both individual faces and bounding charts. Every chart's ordered across/down extent matched its integer span. A chart bounds the visible face cells on a plane. It can surround an opening or extend to the edge of a partially visible cube. The native mask still governs pixel ownership.

These checks establish exactness **against the recorded opaque-cube reference**. In the mask-only arm, grouping and coverage are exact while direction and depth still come from a predicted donor. In the chart arm the counter can revise its input. In the chart-plus-count arm the exact chart and counts are supplied at placement entry. Placement still samples image evidence at its normal resolution and must recover depths. In the exact-position material run, the scorer verified the registered local-to-world XYZ round trip and decoded each typed pack. The actual source-image methods get none of the oracle's world records. [Independent ray and projection audit](../material-feedback-surfaces-01/oracle-coordinate-audit.json)

The audit caught a **half-source-pixel mismatch in the original depth diagnostic**. The placer casts its coarse rays at source coordinates `4k + 2.0`. Reading every fourth native label at index `4k + 2` instead retrieves the ray through `4k + 2.5`. We rerendered the visible reference at the actual placement rays and reran every reported depth intervention. Moving the production rays as a separate test improved pooled strict placement 90.25% → 90.48%, but lost 25 correct positions and recovered seven. Keep the original production sampling while testing where both placements disagree. [Exact-ray audit, calibration and paired exports](../material-feedback-geometry-01/component-bypasses.md)

## Separate material errors from geometry errors

These runs cover **all 104 fixed views**. The recorded camera in this table changes only where the final matcher reads source pixels. Existing-position interventions use the frozen geometry alignment. When given true positions, the matcher has to identify all 23,702 visible blocks, including hard ones the placer had missed.

| Input to the final matcher or exporter | Strict placement F1 | Strict overall F1 | Gain vs default | Facade overall F1 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary pipeline | 91.27% | 84.68% | — | 88.51% |
| Recorded camera for reading materials | 91.27% | 85.88% | +1.20 pp | 89.54% |
| Exact IDs on currently placed blocks, arithmetic ceiling | 91.27% | 91.27% | +6.59 pp | 93.42% |
| Exact visible coordinates, current matcher and estimated camera | 100% | 90.93% | +6.25 pp | 94.16% |
| Exact visible coordinates and camera, current matcher | 100% | 92.35% | +7.67 pp | 95.64% |

The same last intervention reached **97.24% strict overall F1** on the separate 28 views, versus 86.79% for their ordinary pipeline. The gains in this table overlap. These are measured outcomes for the current algorithms and data. Perfecting one stage does not guarantee the same gain from an approximate image-only fix. [Full material interventions](../material-feedback-surfaces-01/README.md)

## Should a sub-stage be removed?

A negative result from supplying an earlier stage with truth can expose an incompatible handoff. A removal test keeps the rest of the path identical and measures a real bypass.

| Sub-stage bypassed | Paired impact | Decision |
| --- | --- | --- |
| Score texture repetitions when choosing the depth of a plane | On ten views, strict ranked placement fell 90.25% → 76.55% and overall F1 fell 86.66% → 73.20%. The bypass lost 307 correct positions while gaining 18. | **Keep.** It already makes many of the correct depth choices. |
| Arbitrate depths jointly across surfaces | On those ten views, lost five correct positions and added four extras. | **Keep.** Local choices can conflict at an opening. |
| Reject fully occluded candidate cells | Added 37 extras and recovered no correct positions. | **Keep.** |
| Recover verified edge cells | Lost six correct positions and added two extras. | **Keep.** |
| Supply counted spacings to primary placement | On five ordinary builds, suppressing 30 spacing rows eliminated 78 cells supplied by that particular path. Other paths recovered all 78. All five final packs tied, including IDs. Supplying 28 perfect wall-dimension corrections also tied. | **Investigate redundancy** on more builds before pruning this handoff. Counting also serves other stages and its own output. |
| Retry an incomplete count with texture-guided straightening | On 20 ordinary-input replays, bypassing the retry changed one pack; strict overall F1 rose 86.42% → 86.45%. With true sampling charts all 20 packs tied and the number of fully correct measured grids fell from 172 to 171. | The retry did not explain the drop caused by feeding perfect chart corners through the ordinary counter. Scope any gate to the cases where the retry demonstrably changes a later decision. |

Across all **104 fixed views**, turning off the counting retry raised strict placement **91.38% → 91.51%** and facade overall **88.43% → 88.56%**. Fifty-six handoffs changed. Three exports improved in facade placement and the remaining 101 were identical. Full ordinary builds reproduced all three changed packs. On 939 matched charts, disabling the retry cut fully correct dimensions from **288 to 233**. On Twinspire, all three retry counts (4×11, 11×1 and 6×2) were correct, yet using the original charts raised facade placement **88.82% → 97.77%**. **Keep the retry for counting. Test a placement choice between the original and adjusted charts**, scored on projected corners and independently visible adjoining edges. [All 104 handoffs](../material-feedback-grid-01/retry-bypass-fixed.json) · [Three complete ordinary exports](../material-feedback-grid-01/retry-bypass-ordinary.json)

The full command reproduced the released Colonnade and Loading Dock control packs. Removing verified edge recovery tied on Colonnade and lowered Loading Dock strict placement from **81.80% to 80.08%**. Removing repeat-based depth selection lowered Loading Dock to **69.81%**. The ten-view depth and visibility comparisons run through the ranked placer and material exporter. The five-view spacing comparison runs through the complete ordinary build. Neither sample is a blanket all-building removal result. [Placement bypasses](../material-feedback-geometry-01/component-bypass-summary.json) · [Count handoff bypass](../material-feedback-grid-01/count-only-oracle.json) · [Texture retry bypass](../material-feedback-surfaces-01/retry-bypass.json)

## What happened when the feedback came from actual predictions?

| Proposed feedback | Fixed result | Decision |
| --- | --- | --- |
| Project materials onto touching masks and merge matching IDs | Surface F1 fell from 75.56% to 66.74%. Even true material IDs on both sides gave 67.07%. | Material continuity can support a seam hypothesis. It cannot establish a single plane. |
| Require projected material, plane, normal, depth and seam agreement | It repaired one real Courtyard seam. Recounting and placing that corrected surface still added one extra cube. It falsely merged two Loading Dock surfaces on a comparison view and cost four correct positions. | Keep outline repair, sampling geometry and placement acceptance separately measurable. |
| Use verified final cubes at the four corners of a wall to fill a missing count | Exact dimension pairs rose from 287 to 349 among 878 matched fixed wall charts. The rule also added eight wrong pairs and two unpaired or wrong-direction proposals. | Use as an additional measured candidate. Validate its projection before changing placement. |
| Reproject the front cube and rear plane, remove a front cube with worse texture fit | Removed 16 extras and one correct cube on the fixed suite, raising overall F1 from 84.68% to 84.75%. On eight decorative views it removed three real blocks and zero extras. | Keep as a review signal; require independent rim or rear-plane evidence. |
| Give the ranker the dominant true depth on ten views | It already selected that depth for 139 of 149 eligible masks. Forcing the ten corrections reduced ranked placement F1 from 90.25% to 90.09%. Nine individual changes tied and one regressed. | Split mixed-depth regions and check their occlusion ordering. This diagnostic scores ranked placement. |

[Surface joins and closed-loop builds](../material-feedback-surfaces-01/README.md) · [Counting and count-only oracles](../material-feedback-grid-01/README.md) · [Depth, foreground and placement experiments](../material-feedback-geometry-01/README.md)

The production pipeline still uses the source image and versioned game assets. Reference cameras, masks, coordinates and IDs in this assessment are evaluation instruments. All corrections apply to the visible build.
