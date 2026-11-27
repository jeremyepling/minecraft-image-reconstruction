# When several wall depths share one mask

A scanned wall mask can cover most of the right pixels and still put several distinct sections of the building at the same depth. On the Harbor kiln high-right image, correcting the depth partition of just the largest mask raised strict visible placement F1 from 18.13% to 28.24%. Replacing that mask's one depth with the depth that covers its largest visible area gave 18.13%. The results use reference geometry to expose the opportunity. The source-only step tested here is the partition itself.

**Decision.** Keep `experiment/harbor-placement-evidence` experimental. This branch adds diagnostic runners, source-image overlays and scored receipts. The production default remains on `main` at the measured revision `62611ee`. Train Amphitheater and Viaduct are development examples. Harbor is an inspected validation-layout regression. This round contains no fresh test-layout result.

## Follow an exported block back to the source

On Harbor kiln high-right, source ray (1134, 286) hits a polished blackstone face at world (15024, 84, 4). The surface mask covers that ray and the selected axis is correct. The placer gives all of mask-0 local Z=38. The ordinary pack exports mud at local (−6, 19, 38), registered at world (15024, 84, 2). Even after the one whole-build scoring shift, it is an extra at (15024, 84, 3). Assigning that photographed patch its own observed depth puts the cube at local (−6, 19, 36), the correct position under both registration and scoring. Material assignment still returns mud. This example separates the wrong carrier depth from the remaining wrong block ID. [Source overlay](validation-harbor-02-kiln-high-right.jpg) · [Ray, cells and decoded IDs](validation-harbor-02-kiln-high-right-ray-trace.json)

The largest kiln high-right mask covers 51,572 labeled native pixels. Its largest single plane accounts for 11,852, at local Z=34. The selected Z=38 accounts for 3,298. Z=35, 36 and 37 account for another 26,301. Kiln rear has one dominant mask with 155,980 of 159,419 native building pixels on its already-correct X=10. The public surface artifacts omit a placement-only mask in both high-right palettes. Counts below use the actual pre-inset placement input. [High-right carrier inventory](validation-harbor-02-kiln-high-right.json) · [Rear source overlay](validation-harbor-02-kiln-rear-oblique.jpg)

| Kiln view | Required correct / missed / extras | Misses without a mask | Mask but no correct axis | Correct axis but wrong level | Correct plane already present | Counted charts with both spans | Exact IDs / strict positions |
|---|---:|---:|---:|---:|---:|---:|---:|
| High-right | 98 / 616 / 222 | 17 | 177 | 344 | 78 | 0 | 4 / 131 |
| Rear-oblique | 503 / 107 / 55 | 0 | 57 | 31 | 19 | 3 | 341 / 518 |

For each missed required cube, its strongest evidence on **any** native frontmost pixel of **any** visible face sets its category. A one-pixel hit suffices. The final column counts materials on correctly positioned cubes including thin visible faces. High-right produced zero strict chart cells and its 353 exported blocks all received source-atlas IDs. The mask-based placement path can emit blocks when the Block Counter finds no full span. The rear produced 573 cubes. The score receipts report complete typed geometry and assignment counts.

## Interventions on one carrier

The control returned exactly the independently scored decoded baseline packs on all three replayed views. We chose the single carrier with the largest labeled source overlap for diagnosis. A source-only reproduction must select a carrier by source-mask area or other source evidence. The original source, inferred camera, counts, ordinary depth ranking, secondary depth adjustment, frontmost-ray filters, downstream geometry edits, material assignment and pack export still run. The main intervention replaces only that carrier's primary candidate cells inside `visible_placement._estimate_plane_cells`. The scorer registers the camera axes and chooses one integer translation for the whole build. Its shift may differ between arms.

| View and intervention | Correct / missed / extra | Strict / facade placement F1 | Strict / facade position-and-ID F1 | Correct IDs |
|---|---:|---:|---:|---:|
| Amphitheater ashlar high-right, ordinary | 839 / 938 / 559 | 52.85 / 53.46 | 15.81 / 14.64 | 251 |
| One reference level | 899 / 878 / 620 | 54.55 / 55.50 | 14.99 / 14.04 | 247 |
| Reference partition and levels | 1,279 / 498 / 150 | 79.79 / 82.94 | 39.49 / 40.77 | 633 |
| Source RGB partition, reference levels | 1,232 / 545 / 233 | 76.00 / 78.31 | 34.67 / 34.92 | 562 |
| Harbor kiln high-right, ordinary | 131 / 961 / 222 | 18.13 / 18.96 | 0.55 / 0.77 | 4 |
| One reference level | 128 / 964 / 192 | 18.13 / 16.97 | 0.42 / 0.61 | 3 |
| Reference partition and levels | 195 / 897 / 94 | 28.24 / 35.77 | 6.81 / 9.55 | 47 |
| Source RGB partition, reference levels | 185 / 907 / 126 | 26.37 / 33.33 | 6.13 / 8.33 | 43 |
| Harbor kiln rear-oblique, ordinary | 518 / 246 / 55 | 77.49 / 86.13 | 51.01 / 58.05 | 341 |
| One reference level | 518 / 246 / 55 | 77.49 / 86.13 | 51.01 / 58.05 | 341 |
| Reference partition and levels | 518 / 246 / 53 | 77.60 / 86.28 | 51.09 / 58.15 | 341 |
| Source RGB partition, reference levels | 517 / 247 / 55 | 77.40 / 86.13 | 51.05 / 58.05 | 341 |

The exact partition arm divides same-axis visible source rays by the integer level of their reference face. It preserves off-axis and unlabeled evidence on the selected ordinary level. The single-level arm uses the same-axis level receiving the most rays. In the source-RGB arm each whole source component receives its most common same-axis reference level. Small components retain the original level. These are attainable diagnostic outputs within the tested choices. Selecting the dominant level by pixel count does not optimize whole-build F1. Given only the option of keeping the ordinary build or taking the source-RGB/reference-level output, a reference chooser prefers the latter on four of these five views and retains the ordinary rear view.

On high-right kiln the RGB partition plus reference depth gained 121 correct cells, lost 67, removed 201 extras and introduced 105. On Amphitheater the counts were +402, −9, −400, +74. It also improved Harbor seafoam high-right from 20.90% to 26.67% strict placement and from 7.92% to 13.48% strict position-plus-ID, and Viaduct riverstone high-right from 82.73% to 83.89% and from 62.05% to 63.20%. These figures do not measure source-only depth selection. [Kiln paired result](validation-harbor-02-kiln-high-right-rgb-oracle-score.json) · [Seafoam paired result](validation-harbor-02-seafoam-high-right-rgb-oracle-score.json) · [Training Viaduct result](train-viaduct-02-riverstone-high-right-rgb-oracle-score.json)

Each replay checks the projection of every recorded face corner to less than 0.0001 native pixel. The reference faces are rasterized at the actual production rays (4k+2); a second raster at 4k+2.5 agrees exactly with the decoded native frontmost face IDs sampled at those pixel centers. A complete cube for a visibly supported face is the extent of the intervention. Each downstream reprojection and export uses only the actual frame's mask evidence.

## First source-only experiment

Use the source partition in `benchmark.harbor_partition_probe.bounded_rgb_regions(image, mask)`. We fixed four-pixel stride, median RGB over each 4×4 cell, a maximum per-channel difference of 20 between four-connected neighbors, and at least 16 samples per component. A small component belongs to the residual. Without access to geometry labels, the partition does not know a component's integer depth.

| Largest carrier | Same-axis visible rays | Correct at current level | Best single level | RGB components | Best independent levels | Best levels within ±6 |
|---|---:|---:|---:|---:|---:|---:|
| Train Amphitheater | 6,973 | 602 | 863 | 9 | 5,465 | 4,249 |
| Train Viaduct | 8,285 | 7,631 | 7,631 | 4 | 7,832 | 7,832 |
| Harbor kiln high-right | 2,751 | 205 | 743 | 23 | 2,313 | 2,214 |
| Harbor seafoam high-right | 2,806 | 81 | 743 | 17 | 2,173 | 2,073 |
| Harbor kiln rear-oblique | 9,767 | 9,730 | 9,730 | 8 | 9,734 | 9,730 |

The last two columns maximize correct **same-axis source rays**, one level per actual component. They define a ceiling for that intermediate objective. The global Planarity labels on Amphitheater and kiln high-right admit 1,685 and 1,573 correct rays respectively. Existing atlas scores also offer little leverage on this case. For the largest high-right masks, neither palette has two adjacent qualifying cells at any tested alternative from Z32 through Z44, even if the primary atlas-evidence gate is bypassed. [Training partitions](train-amphitheater-02-ashlar-high-right-partition-probe.json) · [Kiln per-level atlas evidence](validation-harbor-02-kiln-high-right-atlas-probe.json)

For a bounded source-only challenger, rank the parent level and integer levels within six blocks for each sufficiently large RGB component using native projected boundary fit, one-block repeat evidence and floor contacts where the source shows one. Test a margin over the parent on development buildings; preserve the parent when evidence is ambiguous. Generate candidates once per selected level from the union of its components with the ordinary mask coverage and partial-face checks. Keep the public surface ID and counter unchanged; the child assignment only changes the observed cells proposed on the parent's axis before the common visibility checks. Test solid and stepped carriers on separate buildings and decode the packs. The RGB partition changes even where materials differ on a flat wall, so a meaningful unchanged-parent choice matters.

The true bypass of the grounded depth filter gave kiln high-right +44 correct cells and −1 lost, but it also introduced 113 extras and removed only two. Rear gained four correct cells and 58 extras, lowering strict F1 from 77.49% to 74.62%. Amphitheater fell from 52.85% to 52.66%. Keep the ordinary filter. [Kiln bypass](validation-harbor-02-kiln-high-right-bypass-score.json) · [Rear bypass](validation-harbor-02-kiln-rear-oblique-bypass-score.json)

## Placement status and reproduction

The ordinary scores below carry forward the independently verified `62611ee` controls. These partial diagnostic replays do not change production behavior. No whole-cohort challenger was run.

| Cohort | Tested / available | Strict / facade geometry F1 | Strict / facade position-and-ID F1 | Correct / missed / extra |
|---|---:|---:|---:|---:|
| Fixed exterior | 104 / 104 | 91.27 / 93.42 | 84.68 / 88.51 | 20,773 / 2,929 / 1,046 |
| Visible Builds v3 | 144 / 144 | 21.84 / 22.75 | 14.13 / 14.79 | 28,086 / 186,064 / 14,949 |
| Decorative Builds v1 | 40 / 40 | 63.46 / 65.23 | 50.51 / 52.56 | 9,240 / 7,686 / 2,957 |

Run from the branch checkout with the saved v3 datasets, ordinary handoffs and released packs. Use `--modes rgb-oracle` for the RGB depth replay on the seafoam and Viaduct rows. Each score receipt records source and pack fingerprints, the registered-frame and translated scores, the actual shift and coordinate deltas. [Verified controls](../camera-model-controls-01/independent-matched-summary.json)

```powershell
$id = 'validation-harbor-02-kiln-high-right'
python -m benchmark.harbor_carrier_audit --overlay $id
python -m benchmark.harbor_partition_probe $id
python -m benchmark.harbor_atlas_probe $id
python -m benchmark.harbor_plane_replay $id --modes control single split rgb-oracle bypass
python -m benchmark.harbor_ray_trace
```
