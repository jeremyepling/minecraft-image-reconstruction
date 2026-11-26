# Reconstructing partially observed Minecraft faces

Experiment branch `experiment/visible-face-reprojection`, code revision `2609924`, Minecraft 26.3. The strongest repeatable addition to the depth ranker was checking partial faces against the texture atlas. Across 92 single-image evaluations it recovered 126 previously missed visible blocks, lost five previously correct placements, added 13 extras and removed nine extras. Each cohort gained visible blocks. On Twin Spires it also added two hidden extras. These results support incorporating the partial-face check after the depth ranker. The branch records experiments. Deployment and fallback verification belong to the combined-default review.

## What was held fixed

Every variant receives the same frozen RGB screenshot, image-derived camera, surface masks and count estimates for each frame. The placement stage returns complete blocks only for proposed faces. Evaluation independently registers the decoded exported coordinates in the source world. A correct coordinate has an observed face in the source image. An extra is any other placed coordinate. A hidden extra is an extra occupied in the source world. Unknown space has no positive label. The packs use smooth stone to isolate placement. The material atlas supplies recognition evidence here; the resulting numbers measure geometry.

The depth ranker and cell models were fitted on 72 training views across six structure families. The six-view pilot is a training screen. Validation has 12 Cloister views, test has 12 Twin Spires views, the exterior regression suite has 56 views across eight buildings, and the native-camera suite has 12 supported views across six layouts. The eight unsupported regression inputs emit empty packs and count toward recall. The test and regression outputs are shared comparisons across methods, so a policy selected after looking at those results needs new examples for an independent confirmation.

## From face candidates to coordinates

The ranker chooses an integer plane for an observed surface using projected atlas alignment, border and edge evidence. `art` relaxes the usual 50% mask coverage to 22% with at least four low-resolution mask samples **only if the observed portion matches a vanilla texture**. The projected face must span at least nine pixels on both axes. At least 42 projected texel samples from 20 distinct image pixels must survive the surface mask. Two interleaved groups must independently match the same atlas face and orientation. The match must exceed a quarter-tile shift on either axis. Without an atlas match the ordinary coverage rule still applies.

On the six training views, same-orientation reprojection and the occlusion retry each scored 917 correct placements and 146 extras, including 16 hidden. Accepting all partial faces scored 940 correct and 192 extras, including 23 hidden. Requiring atlas evidence scored 924 correct and 148 extras, including 16 hidden. On validation the retry lost one correct placement and removed one extra compared with the ranker. That trade did not justify carrying the retry forward.

## Cohort results

Each entry gives **correct / extras / hidden extras**. The historical baseline is the ordinary placement route before depth ranking. Rank alone is the immediate comparison for the partial-face experiment.

| Method | Validation · 12 | Test · 12 | Regression · 56 | Native · 12 |
|---|---:|---:|---:|---:|
| Historical baseline | 1799 / 713 / 140 | 2218 / 1064 / 518 | 5967 / 3508 / 978 | 2760 / 489 / 27 |
| Rank alone | 2161 / 340 / 0 | 2896 / 273 / 16 | 7719 / 1048 / 76 | 2909 / 257 / 5 |
| Rank + atlas partial faces | 2171 / 340 / 0 | 2923 / 278 / 18 | 7786 / 1048 / 73 | 2926 / 256 / 5 |
| Rank + tree | 2160 / 111 / 0 | 2896 / 244 / 16 | 7511 / 655 / 33 | 2811 / 177 / 0 |
| Partial faces + tree | 2169 / 111 / 0 | 2923 / 249 / 18 | 7578 / 655 / 32 | 2828 / 177 / 0 |
| Partial faces + agreement of tree and occlusion | 2171 / 158 / 0 | 2923 / 272 / 18 | 7771 / 1003 / 49 | 2926 / 256 / 5 |
| Partial faces + occlusion | 2171 / 126 / 0 | 2923 / 158 / 18 | 7764 / 999 / 46 | 2926 / 256 / 5 |
| Partial faces + family-constrained rule | 2158 / 326 / 0 | 2917 / 274 / 18 | 7679 / 1002 / 73 | 2907 / 246 / 5 |
| Partial faces + conjunctions | 2167 / 170 / 0 | 2913 / 253 / 18 | 7669 / 852 / 68 | 2901 / 193 / 5 |

**Paired coordinate changes.** Each pair counts newly correct / previously correct lost, or new extras / previous extras removed, comparing actual world coordinates within each image.

| Atlas partial faces vs rank alone | Correct gained / lost | Extras added / removed | F1 before → after |
|---|---:|---:|---:|
| Validation | 11 / 1 | 1 / 1 | 86.84% → 87.07% |
| Test | 28 / 1 | 5 / 0 | 87.98% → 88.37% |
| Regression | 69 / 2 | 5 / 5 | 79.61% → 80.02% |
| Native | 18 / 1 | 2 / 3 | 83.76% → 84.06% |

Compared with the historical baseline, the ranker plus partial faces gained/lost 372/0 correct on validation, 778/73 on test, 2,033/214 on regression, and 221/55 on native. Extra placements added/removed were 20/393, 65/851, 206/2,666, and 48/281 respectively. Most of that improvement came from choosing the plane. Per-method paired changes against both references are in [cohorts.csv](cohorts.csv) and [paired-rank.csv](paired-rank.csv).

## Which filter carries forward

The `invisible` filter renders the frontmost faces of the **predicted** occupied cells from the inferred camera. It removes a cell only when it owns zero pixels in both the coarse and native-pixel renders. A misplaced foreground cell can hide a correctly placed cell in those renders even though the source screenshot shows it. This explains why the filter can remove source-visible targets. Source-world visibility is used only to score the result.

Relative to partial faces alone, that filter removed 214 extras with zero correct lost on validation, 120 with zero lost on test, and 49 with 22 lost on regression. Of those 49 regression extras, 27 were hidden. On native it changed no cells. F1 for the resulting four cohorts was 90.97%, 90.01%, 80.09% and 84.06%, respectively. This is the strongest broadly applicable geometric filter here. The regression losses warrant inspection when deciding whether to enable it across camera types.

The learned tree after partial faces removed 229 extras for two correct lost on validation and 29 extras for zero lost on test. On regression it removed 393 extras for 208 correct lost. On native it removed 79 extras for 98 correct lost, dropping F1 from 84.06% to 83.36%. Keep unconditional tree rejection experimental. Requiring both tree and predicted occlusion to reject a cell preserved seven additional correct regression placements compared with occlusion alone, but left 114 additional extras on test. The family-constrained rule and conjunction policies also lost correct coordinates in every cohort compared with simply verifying partial faces.

## Evidence and replay

The [evidence archive](evidence.zip) contains 876 decoded placement packs, their local/world coordinate tables, frame-level scores and cohort manifests, including the historical and rank-only references. All packs were independently decoded and checked against their local coordinate tables. Every stored pack and case list matched its recorded SHA-256, and all appearances of each image shared identical RGB and input hashes. The archive SHA-256 is `566333c11fe56bf8d4658c27097528070dfd2d535af2420aef0228150123bba9`. The original manifests preserve candidate and scorer digests and source revisions. Some runs recorded a dirty source tree during development; the archives retain the bytes scored.

For replay use the coordinator's frozen scorer and fixtures, this branch's candidate, the published 72-view plane weights, and the frozen cell decisions branch at `567be25` for policies that filter cells. From the scorer checkout with adjacent named worktrees:

```powershell
$env:MCIR_OFFSET_FIT = (Resolve-Path .\engine\plane_offset_weights.json).Path
$env:MCIR_CELL_GATE_ROOT = (Resolve-Path ..\visible-cell-decisions).Path
python -m benchmark.visible_depth_score --split validation --candidate-root ..\visible-face-reprojection --candidate benchmark.face_reprojection_round:art --out runs/replay-face-art
```

Use the entry points named in the manifests to replay a different policy. The original captures, source pose and source block state belong to the dataset and scorer. Model fitting and inference use distinct inputs from final placement scoring.
