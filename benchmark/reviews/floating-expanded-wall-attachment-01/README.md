# Detached wall fragments on the adopted expanded benchmark

**57 more correct positions, 51 more exact IDs and 20 more extras** across 84 conditions. No previously correct positions or exact IDs were lost. Five exported builds changed. The best source-only arm attached compatible tangent grids while preserving explicit depth partitions. On the six fixed sky views it matched the adopted build exactly. Keep this as the leading experimental attachment candidate for the user's correct-position priority. The 20 extras on three views and previously reviewed buildings limit the generalization claim.

[Every offered chart on its source image](detached-source-charts.jpg) · [Per-image source fits, placements, comparison roots and fingerprints](conditions.json.gz) · [Pooled and cohort scores](summary.json)

## What was compared

Start from the adopted expanded package `ac69f22` plus split-border guards `4d9aca1`. Each case resolves its exact chart and actual assembly using the frozen split and projected-sky decisions. The runner confirms chart, source, pack and reference fingerprints. It reruns the full ordinary assembly on all 84 unchanged inputs. Local positions, material IDs, registered positions, joins and scores match the adopted build. This verifies the no-op even when multiple input chart packets can yield the same exported blocks.

The frozen source candidate and texture functions were copied from `f088877`. They propose attaching a non-selected, observed wall chart to a selected chart with the same signed horizontal and vertical axes and outward normal. Both need at least four eligible cells. The child must span both tangent dimensions. Map child centers to parent UV coordinates and round the median displacement to integers. Require a 90th-percentile residual no greater than .30 blocks, a maximum no greater than .45, no more than 10% overlap and at least two neighboring cell edges. Select the parent with lowest residual and then most contacts.

The tangent arm uses the parent plane. The repeated-texture arm requires at least four native tile matches across two corresponding rows. It transfers a majority-voted parent normal offset for each child row. An additional stricter arm requires half the child's tiles to match and at least three contacts. All present parent offsets are zero. Texture evidence can reject an offer but cannot recover a missing relief step on this benchmark. The earlier model had explicit parent depth steps.

The candidate gets the actual selected component and solved chart origins. Joins and orientation continue to use intact source charts. Both the candidate evidence and cell insertion honor export membership, including the projected-sky veto. A mapped child cell cannot reintroduce a rejected parent UV. New child voxels fill only unoccupied coordinates in the selected group, leaving its previous blocks and material IDs intact. Ordinary pack writing, decoding, whole-build registration and visible-reference scoring follow. The partition-preserving arm further rejects attachments between children of an explicitly split source mask. Such pieces inherit the same original ruler; tangent agreement alone does not reverse their depth separation.

## Results

| Active 84 | Strict | Facade | Strict + ID | Facade + ID | Correct / missed / extras | Gross correct +/− | Gross extras +/− |
|---|---:|---:|---:|---:|---|---|---|
| Adopted / no-op | 33.987% | 34.479% | 5.655% | 5.757% | 18,318 / 61,192 / 9,965 | 0 / 0 | 0 / 0 |
| Tangent | 34.065% | 34.559% | 5.745% | 5.849% | 18,375 / 61,135 / 9,997 | 57 / 0 | 32 / 0 |
| Repeated texture, or stricter majority | 34.056% | 34.549% | 5.738% | 5.842% | 18,369 / 61,141 / 9,997 | 51 / 0 | 32 / 0 |
| Tangent, preserve partitions | **34.069%** | **34.563%** | **5.746%** | **5.850%** | **18,375 / 61,135 / 9,985** | **57 / 0** | **20 / 0** |
| Repeated texture, preserve partitions | 34.060% | 34.553% | 5.739% | 5.843% | 18,369 / 61,141 / 9,985 | 51 / 0 | 20 / 0 |

The reference-only whole-export chooser over the no-op and all measured source arms matches the partition-preserving tangent result. Its options cannot selectively prune incorrect cells inside an accepted child. The best arm has 3,099 exact IDs against 3,048, 27,769 material-assigned exports against 27,694 and 591 fallbacks against 589.

| Offered source chart | Added correct / extra / exact IDs | Observation |
|---|---|---|
| Viaduct 1 riverstone low-left | 20 / 4 / 19 | The same 24-cell left column maps at (−15, −1) with four contacts. The earlier depth-refit parent supplied a +1 step on four rows' cells and recovered all 24. On the flat adopted parent those four are misplaced. |
| Riverside Mill 2 warm low-left | 24 / 0 / 24 | The flat parent already carries the appropriate depth. Fifteen contacts and 23 source texture witnesses support the same shift (6, −2). All 24 positions and IDs transfer. |
| Cloister 2 cool low-left | 6 / 0 / 4 | Three contacts fit. No repeated texture witness passes, so imposing the texture veto loses six correct positions and saves no extras. |
| Walled Garden 1 cool low-left | 3 / 8 / 0 | Three contacts and nine texture witnesses pass on a flat plane. |
| Stable Yard 2 warm low-left | 4 / 8 / 4 | Four contacts and nine witnesses pass. Most added positions are extras. |
| Pillars sky low-left | 0 / 0 / 0 | A split child agrees exactly with its sibling's inherited ruler. Ignoring their shared partition history adds 12 extras and no correct positions. |

The recommended arm scored 43.21% strict / 46.15% facade / 17.57% strict + ID / 19.07% facade + ID on 30/144 Visible Builds v2 conditions. On 30/144 Visible Builds v3 it scored 34.15% / 34.29% / 3.94% / 3.95%. On 24/40 Decorative Builds v1 it scored 27.24% / 28.13% / 7.16% / 7.58%. No newly captured structure was evaluated. The separate fixed-104 production result carries forward 91.27% strict and 93.42% facade, 20,773 / 2,929 / 1,046.

Using the benchmark Python and saved inputs in the sibling package, package-rechecks and column-recovery worktrees, run:

```powershell
python -m benchmark.floating_expanded_wall_attachment --workers 5
python -m benchmark.floating_expanded_wall_attachment --report
python -m benchmark.floating_attachment_sheet
python -m unittest tests.test_floating_expanded_wall_attachment tests.test_relative_face_assembly
```

## Floating branch scorecard

Sparse connected-atlas integration at `9435934` carries forward **12.64% strict, 12.90% facade, 4.68% strict + ID and 4.77% facade + ID**, with **273 correct / 3,768 missed / 5 extras** on the six fixed sky composites. The adopted split-border expanded benchmark and partition-preserving tangent attachment both score **27.588%, 27.826%, 5.983% and 6.005%**, with **890 / 3,151 / 1,521**. The attachment contributes zero correct positions and zero extras on those six. This expanded path gains 617 correct and adds 1,516 extras against sparse integration. Without the partition guard, the attachment adds 12 extras and no correct positions. Only the experiment and its evidence landed on `experiment/floating-expanded-wall-attachment`. The adopted expanded benchmark retains the selected package plus split-border.
