# Getting Surface Finder outlines closer to the recorded surface

**The ordinary Surface Finder now uses the photographic edge adjustment with stable block counts.** Look up to four source pixels around the current outline for a stronger color boundary, preserving holes and overlapping masks. If the move changes an existing complete count or creates a complete count from an unresolved one, retry the frame within two pixels, then keep the original outline if needed. The counter supplies its own estimate during prediction. Recorded block counts enter only during scoring.

This improved average outline error on every building and found two additional surfaces. The strict safeguard was chosen after seeing four errors on Portico. Keep the ordinary default while checking this frozen rule on previously unexamined buildings. The accuracy gains below are regression results on eight familiar structures.

| Across 56 saved views and 598 eligible surface appearances | Mean edge error (pixels) | Surface matches | Exact block sizes | Incorrect block sizes | Opening samples incorrectly covered |
|---|---:|---:|---:|---:|---:|
| Shipped Surface Finder | 3.824 | 517 | 95 | 50 | 511 |
| Plane model, add-only, 2-pixel grid | 3.809 | 517 | 95 | 50 | 511 |
| Photo edges, protect existing complete counts | 3.712 | 519 | 95 | 54 | 481 |
| Photo edges followed by add-only plane model | 3.705 | 519 | 95 | 54 | 481 |
| Photo edges, also protect unresolved counts | 3.720 | 519 | 95 | 50 | 482 |

Edge error is the mean bidirectional distance between outlines at source resolution, including holes, pooled over the same 517 baseline-matched surface appearances. Surface matches and opening coverage use the established suite scorer. Correct and incorrect dimensions concern matched eligible surfaces. The scorer reports complete guesses on unmatched proposals separately.

## What the iterations taught us

1. **Four-pixel sampling explains little of the model gap.** Using the gold labels solely for diagnosis, we selected the single best full-view plane segment for each eligible surface. On all 56 views, mean overlap was 64.53% before sampling, 64.15% after four-pixel sampling, and 78.42% for the best region in the shipped multi-model finder. On development, two-pixel sampling and four-pixel sampling were within 0.05 percentage points. Different surfaces may select the same region in this diagnostic. The 95% pixel overlap in the paired boundary test uses previously matched surfaces and a different denominator. [Stage-by-stage measurements](reviews/plane-detail-selection-01/diagnostic.json)
2. **More model detail did not improve block counts.** On development buildings, moving both sides of an edge to a two-pixel-grid plane mask discarded 71,641 correctly owned pixels. Allowing additions only was safer. Increasing the model budget from 1,600 to 2,500 tokens gave better raw segments on average but one fewer exact count in the development trial. In a controlled add-only comparison, the existing four-pixel grid slightly beat the two-pixel grid (3.804 vs 3.809 pixels across 56 views). Neither added a match or an exact count. [Settings, checkpoints, and per-picture results](reviews/plane-detail-selection-01/README.md)
3. **Nearby photo edges gave the largest useful improvement.** On Shopfront and Wing we tried two-, three-, and four-pixel adjustments with the same color evidence. Four pixels moved outlines closest but disturbed an existing block count. A whole-frame fallback protected existing counts on development. Across all views it gained Shopfront R3 and Workshop R19 with no lost match. Adding the frozen plane-model pass afterward improved error by only another 0.007 pixels, with identical totals for surfaces and block sizes. [Development rounds](reviews/edge-detail-dev-01/README.md) · [Combined method](reviews/edge-detail-stack-01/scores.json)
4. **An unresolved count also needs protection.** All four extra wrong counts came from two low-left Portico views. One real 18 × 1 surface went from unresolved to a 36 × 3 guess. The stricter rule kept these cases unresolved and retained both new matches. Across the original 517 paired surfaces it added 17,427 correct and 2,131 incorrect pixels, and removed 17,319 incorrect and 1,485 correct pixels. This is an average improvement with some locally worse edges. [All four count transitions](reviews/edge-detail-stable-01/count-transitions.json)

On 24 fresh Cloister and Twin Spires views the frozen rule preserved all 250 matches and 59 exact sizes and reduced opening samples covered from 686 to 653. Production reproduced all 56 approved regression outputs. [Fresh-building results](reviews/detail-transfer-01/README.md) and [ordinary build verification](reviews/default-photo-edges-01/suite.json).

## Review and replay

[Stage comparison for all eight buildings](reviews/detail-01/comparison.jpg) · [Shopfront, Workshop, and Portico close-ups](reviews/edge-detail-stable-01/comparison.jpg) · [Final scores by building](reviews/edge-detail-stable-01/scores.json) · [Final per-picture measurements](reviews/edge-detail-stable-01/cases.json)

The first-round finalists were selected on 20 Shopfront and Wing views before evaluation on the remaining 36 views. The strict follow-up uses the same images after inspecting the Portico regression. All predictions consume only one image and its image-derived model fields. The suite has 56 frame IDs and 52 unique image hashes. The model revision, checkpoint, token settings, field hashes, model timing and scoring revision are in the [first comparison manifest](reviews/detail-01/manifest.json) and [final comparison manifest](reviews/edge-detail-stable-01/manifest.json).

Reproduce the recommended comparison and its close-up sheet from the repository root using a fresh output directory:

```powershell
.\Invoke-SurfaceRound.ps1 -Round boundary-detail -Out runs/my-boundary-detail
```
