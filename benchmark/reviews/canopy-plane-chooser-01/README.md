# Depth choices for elevated faces

Promote the unrestricted per-surface ranker from `experiment/canopy-plane-chooser` for the coordinator's combined evaluation. The prediction was frozen at `783380a`. The fallback and repaired-asset tests at `a7b2a0c` preserve its decisions. The predictor uses a single image, the image-derived camera, parts, counts and calibrated native atlas. It adds no cells behind an observed face.

The previous method considered integer depths within six units of two estimated planes. A distant false floor contact can put both estimates far beyond a roof edge. We tested offsets within 18 units of the estimates and up to 24 units from the eye on the measured side. An input pixel projected onto a candidate plane is translated by one block along each surface axis. The ranker compares its RGB change to the change at neighboring noninteger displacements. This lets repeated art support a depth even when the mask has no complete tile crop. Distinct atlas sprites, texel edges and boundary alignment still contribute evidence. The 17 coefficients were fitted on the 72 original training views.

| Set | Ordinary correct / extra / hidden extra | Unrestricted | Conservative evidence gate |
|---|---:|---:|---:|
| Six excluded training families, 72 frames | 13,769 / 2,458 / 160 | 14,244 / 1,574 / 87 | 13,996 / 1,855 / 116 |
| Earlier native transfer development, 12 | 1,638 / 859 / 42 | 1,667 / 359 / 43 | 1,667 / 344 / 42 |
| Cloister, 12 | 2,171 / 126 / 0 | 2,179 / 126 / 7 | 2,171 / 126 / 0 |
| Twin Spires, 12 | 2,925 / 154 / 14 | 2,971 / 123 / 8 | 2,928 / 150 / 11 |
| Fixed exteriors, 56 | 7,780 / 979 / 46 | 7,800 / 834 / 25 | 7,775 / 976 / 45 |
| Earlier camera-supported native views, 12 | 2,926 / 256 / 5 | 2,956 / 227 / 5 | 2,927 / 255 / 5 |

The ordinary training row uses the released weights fitted on all 72 training views. Each challenger in that row uses a separate fit excluding all twelve views of the family being scored. The primary unrestricted model was preferred on development before reporting the remaining suites. It matched more sampled visible mask pixels on five of the six excluded families. The conservative gate keeps the previous plane unless the proposed plane has strong repeat and texel-edge agreement or multiple strong atlas matches.

## Where a perfect chooser helps

Across 834,281 training mask samples, the accurate-plane ceiling rose from 761,970 to 773,628 by widening the proposed planes. The unrestricted model selected 756,235 correctly in grouped validation. The earlier transfer suite had 132,348 mask samples. The original model aligned 112,025, the earlier ranker with wider choices aligned 118,705, and the new ranker aligned 119,671. The corresponding per-surface oracle ceilings were 114,014 for the old choices, 118,411 for radius 18 alone and 121,815 when the closer candidates were included.

For the answer-assisted pack diagnostic, the oracle chooses the level with the most correctly assigned visible face pixels for a surface's detected axis. It abstains when all proposals have zero supporting reference pixels. The remaining image-based face tests and exporter still run. This is a pixel optimum for the given choices. Different depth choices interact in the final visibility test. On old transfer the narrow and expanded oracles produced 1,716 correct / 218 extras and 1,729 correct / 202 extras, respectively.

For the seven old canopies specifically, correct blocks with visible roof edges rose from 63 to 92 of 101. Correct roof-top blocks rose from 60 to 89 of 348. A block with both visible faces contributes to both categories. All 170 previously correct supports were retained in aggregate. The expanded oracle recovered 101 roof-edge blocks and 137 roof-top blocks. These high-angle captures have zero visible underside target blocks. The failing canopy 09 right view improved from 22 correct and 301 extras to 39 and 47.

[Training fold results](development.json) · [Per-canopy roof results](roof-cells.json) · [All proposed levels and visible votes, decoded packs, manifests and coordinates](evidence.zip)

## Replay

Restore the [common inputs and ordinary outputs](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/a16668f/benchmark/reviews/visible-depth-01) and earlier native fixture. From this branch, regenerate the development trials with `python -m benchmark.canopy_depth_choices --split train`, repeat for `transfer`, then run `python -m benchmark.canopy_depth_fit`. Extracting the evidence archive beneath `runs` restores these tables. Run `python -m benchmark.canopy_oracle_replay` to index the development photographs. The optional `narrow` and `wide` oracle adapters deliberately read these evaluation labels. The `family_oof` adapter replays each excluded family's fit.

For an image-only check, run `python -m benchmark.visible_depth_score --split test --candidate engine.plane_repeat_evidence:place --baseline PATH_TO_RELEASED_CELLS --out runs/my-test`. The conservative adapter is `engine.plane_repeat_evidence:guarded`. For the earlier transfer cases, use `benchmark.canopy_compare` from `review/canopy-placement` with this branch as its candidate root.
