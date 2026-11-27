# Iterative scoring of floating surface crops

The baseline is the six open-sky views at `52de2c9`. Every source-only run starts from the same cloud-filtered proposals, normal and depth fields. Corrections cover only faces visible in that source. A missing or decorative target remains outside the full-cube reference control. Fixed 104, full v3 and full decorative production results were not rerun.

## Repeating the experiment

A candidate supplies native row runs for each revised proposal in each frame. The same frozen target is used throughout the comparison. Missing proposal entries retain their source support. The original surface-grid resolution is restored by nearest pixel-center sampling before counts, and corrected supports receive freshly fitted sampling quads. Direction and depth stay at the original proposal estimates unless the diagnostic explicitly introduces new proposals.

From this experimental checkout, the following command takes a candidate through the pixel scorer, source crops, counting, shared rows, native atlas registration, chart assembly and decoded placement evaluation. Files are written under a separate arm. Use the registered reference checkout alongside this worktree and the existing benchmark Python environment.

```powershell
../mask-scalelsd/runs/env/Scripts/python.exe -m benchmark.sky_crop_round learned ../sky-crop-chooser/runs/sky-crop-chooser/learned.json
```

1. Fit edit rules on development buildings only. For each proposal, record native extra and missing pixels, other surface ownership, and the building and surface ID. Learn on the masks the current rule actually produces, including its own mistakes.
2. Freeze the predictor, stopping rule and action choices. Measure every pass, including regressions, on buildings outside fitting. Keep an exact-reference chooser or simulated annotator as a separately labeled diagnostic.
3. Check both per-surface overlap and the inventory of distinct target surfaces. Clipping a merged mask to one exact target can erase the only evidence for another surface. A new proposal or a split can preserve both.
4. Rerun counts, shared joins and placement. Compare decoded positions and materials under a single whole-build translation. Treat new variants chosen after inspecting these six results as exploratory until evaluated on previously untouched buildings.

## Results on the six sky views

| Crop handoff | Strict placement F1 | Facade placement F1 | Strict position + material F1 |
|---|---:|---:|---:|
| Floating baseline and no-change replay | 12.64% | 12.90% | 4.68% |
| Learned one-pass chooser, all selected masks applied | 11.11% | 11.33% | 4.34% |
| Learned chooser, alternatives must preserve baseline block arrangements | 12.69% | 12.94% | 4.81% |
| Zoomed SAM ViT-B, alternatives considered one measured surface at a time | 13.25% | 13.52% | 5.13% |
| Best available crop per existing proposal using reference IoU | 14.78% | 15.04% | 8.15% |
| Exact visible target substituted per matched proposal, ordinary quad refit | 12.09% | 12.33% | 3.25% |
| New proposals from all exact full-cube target masks; normal and depth estimated from image | 14.46% | 14.75% | 3.86% |

The last control offers all 393 connected full-cube targets; 369 survive sampling onto the existing analysis grid. It also excludes partial-block geometry and changes the proposal inventory. It measures a specific intervention rather than an upper bound on an ideal pipeline. On Castle the exact-per-proposal refit lost the selected component's corner join. On 73 baseline-matched rectangular wall proposals with IoU at least 0.5, rounded width or height was correct on 7 of 146 axes for the original counter, 18 for the learned chooser and 41 for exact target crops. The respective wrong/missing counts were 5/134, 8/120 and 16/89. This subset scores rectangular visible wall extents; it does not score slivers, irregular walls or decorative geometry.

The zoomed-SAM paired-chart result gained on four buildings, including both decorative buildings. Across v3 (4/144 views) strict/facade placement rose from 13.87%/13.98% to 14.14%/14.25%; position-plus-ID rose from 4.17%/4.20% to 4.28%/4.31%. Across decorative (2/40) geometry rose from 7.36%/7.94% to 9.47%/10.20%, with position-plus-ID rising from 6.87%/7.41% to 8.74%/9.41%. This policy was examined on these same six regression views; keep it experimental. Fixed 104 is the historical production trend of 91.27%/93.42% geometry and 84.68%/88.51% position-plus-ID.

## Does planning beat correcting the largest immediate error?

On each of 188 matched crops the diagnostic enumerates keeping, trimming the current mask to any of ten candidates, or extending it by any candidate. Each second action acts on the edited mask. Only this diagnostic can read the answer to choose actions. On the 124 matched crops from four held-out buildings, mean IoU was 42.30% initially, 55.43% after the best immediate edit, 58.39% after repeating greedy selection and 58.82% after exhaustive two-step planning. Constraining the first edit to maintain or improve IoU gives 58.80%. Six held-out crops have a strictly higher optimum if an initially harmful move is allowed. A two-step advantage for this limited action set measures planning headroom, not the performance of a trained reinforcement-learning policy.

[Per-view decoded scores and cohort metrics](paired-placements.json.gz) · [Count and join checks](counts-and-joins.json.gz) · [Two-edit traces and input hashes](two-edit-oracle.json.gz)
## Choosing a training method for repeated corrections

The exact native visible mask supplies a label for each erroneous pixel on development buildings. A supervised correction policy can learn to add missing pixels and remove extras using the source image, current mask, previous edits and competing segmentation proposals. At inference it sees only information derived from the source image. Training on the policy's own intermediate masks exposes it to the drift that occurs when an earlier correction was wrong.

For a model-assisted diagnostic, simulate an annotator who can inspect the target, place a corrective click in an erroneous region, and score the changed mask. Record both rejected and accepted attempts and freeze the stopping rule. A separate source-only run must choose edits and decide when to stop without comparing to the reference. Compare the two trajectories at the same edit budgets. The exact evaluator can name whether extra pixels belong to sky, another face direction, a parallel plane at a different depth, or decorative geometry. This feedback is more precise than grading a text description of how the images differ.

Keep reinforcement learning as a follow-up for a richer action space: split a merged proposal, keep both child surfaces, connect a visible corner, transfer a row count, or stop. On development buildings, compare a greedy chooser with a short look-ahead search that scores assignments and internally aligned geometry after assembly. Investigate a learned long-horizon policy if a material planning advantage recurs across unseen buildings and survives the cheaper supervised baseline. The present trim/extend two-edit experiment measures only that limited action space. Source-only reprojection and texture consistency are useful inputs for unlabeled screenshots; a visually plausible reprojection alone need not preserve internal block depth.

The iterative-mask approach follows prior work that conditions a prediction on the previous mask and simulated corrections. Automatic click prediction is another supervised baseline for these experiments. [RITM](https://arxiv.org/abs/2102.06583) · [PseudoClick](https://arxiv.org/abs/2207.05282)

## Sequential correction results

These results use the 124 proposals on four buildings outside fitting that have a visible full-cube target. Five additional held-out proposals lack a comparable target. All six input images and all 193 proposals remain in end-to-end placement.

| Starting crop and correction policy | Initial mean IoU | Pass 1 | Pass 2 | Pass 3 | Pass 4 |
|---|---:|---:|---:|---:|---:|
| Ordinary crop, source-only learned boundary edits | 42.30% | 45.73% | 46.83% | 47.31% | 47.43% |
| Learned one-pass chooser, same frozen boundary edits | 50.70% | 51.06% | 51.10% | 51.02% | 50.84% |

The simulated annotator with access to the exact mask starts at 42.30% on the same 124 crops. It reaches 59.52% after one click, 76.22% after five and 79.55% by twenty. It chooses the click, compares SAM ViT-B masks and accepts or rejects edits using the reference. The source-only forest learns add/remove decisions along a five-pixel collar on two development buildings. Its four-pass stop was fixed before scoring the other buildings. Starting at the ordinary mask it improves 89 of 124 held-out crops and worsens 35. Chapel S11 improves from 8.1% to 81.2%; Garden S34 grows into neighboring pixels and drops from 80.6% to 48.7%.

On six decoded builds, direct replacement with the source-only four-pass masks scores 11.25% strict / 11.48% facade placement and 3.55% / 3.62% position-plus-ID. Direct replacement after chooser initialization scores 4.12% / 4.21% placement and 3.15% / 3.22% position-plus-ID. On Castle this last configuration loses the earlier arrangement and drops from 217 correct positions to 24. With arrangement-preserving alternative charts, the two strict placement scores are 12.73% and 12.86%. Even the reference-guided SAM result scores 12.57% / 12.82% placement on direct substitution. Keep all sequential policies experimental. Compare foreground crop masks separately from the measurement chart, row interval, and shared-edge constraint. [Decoded trajectory endpoints](iterative-placements.json.gz)
