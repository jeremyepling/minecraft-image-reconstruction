# Deciding which proposed visible blocks to keep

Branch `experiment/visible-cell-decisions` tests an accept-or-abstain rule on the Java 26.3 geometry exporter at `651ef4f`. Each candidate starts with the exact cells produced by the current placement stage. It can remove cells. The frozen shallow tree is a candidate for use after plane selection on the atlas-floor domain. Keep its use on seam-camera views experimental. On the independent native 12 it removed 110 correct visible placements and 95 extras. [Per-frame predictions, decoded packs and feature rows](scored-decisions.zip)

## Training and selection

The official 72 training screenshots cover six building families. Ordinary placement proposed 17,386 blocks, of which 10,056 matched visible truth and 7,330 were extras. A perfect rejection decision on this pool would keep those 10,056 and remove all 7,330 extras. It could recover no additional visible coordinates. Among the extras, 1,878 landed on hidden structure cells.

Each inference row uses the source image, its estimated camera, visible surface masks, model-estimated normals and proposed cells. For the three possible camera-facing sides of a block we projected a 16 × 16 grid of points, checked which ones reached the camera, and compared supported pixels against calibrated vanilla sprites across possible rotations and reflections. The model also measured agreement on other tiles of the same plane, agreement with predicted surface normals, size in source pixels, ground contact and evidence that a ray sees the floor. The latter used the source image's lower left and lower right ground strips to estimate a texture for that view. Membership in the view-specific visible set is the training answer. Unobserved occupancy and building IDs never enter the features.

Each development fold holds out an entire building family. Round one compared one-feature rules. The rule selected on all development frames removed extras with few lost correct coordinates but transferred unevenly across family folds. Round two tested depth-four CART with minimum leaf size proportional to training-set size. Leaf probabilities with Laplace smoothing allowed the tree to abstain only on its least-supported leaves. In round three we tested agreement among trees trained while dropping different families, two conjunctions selected under family loss limits, and a single-feature rule under a 0.5% family loss cap. We also tested a direct rendering rule that drops a cell only if no native pixel sees it in the proposed reconstruction.

Selection required every family fold to lose at most 3% of correct blocks and remove at least twice as many extras as correct blocks. For deployment eligibility the aggregate allowed at most 1% lost correct and required five removed extras per loss. Among those candidates we maximized removed extras minus five times lost correct. The shallow tree won with a probability cutoff of 0.12. It removed 1,730 extras, including 600 hidden-structure placements, and lost 75 of 10,056 correct coordinates in grouped out-of-fold predictions. The tree and its competitors were frozen at `070581e` before validation, test or regression evaluation. [Model](model.json) · [Whole-family folds and selection](training.json)

## Decoded evaluation

Entries give **correct lost / extras removed / hidden-structure extras removed**. Hidden extras are included in the middle count. All candidates use the fixed visible-only inputs from `review/visible-depth-rounds`.

| Rule | Cloister validation, 12 | Twin Spires test, 12 | Regression, 56 |
| --- | ---: | ---: | ---: |
| Single feature | 14 / 60 / 0 | 13 / 92 / 1 | 219 / 261 / 9 |
| Shallow tree selected on development | **1 / 490 / 131** | **17 / 183 / 88** | **238 / 910 / 167** |
| Family-omission tree consensus | 0 / 271 / 61 | 6 / 85 / 31 | 121 / 667 / 113 |
| Family-constrained single feature | 6 / 41 / 0 | 4 / 73 / 0 | 81 / 134 / 5 |
| Family-constrained conjunctions | 1 / 197 / 0 | 8 / 71 / 1 | 86 / 300 / 7 |
| Direct occlusion in final proposed build | 1 / 214 / 0 | 0 / 113 / 0 | 49 / 49 / 26 |

The selected standalone tree retained 1,798 correct blocks with 223 extras on validation, 2,201 with 881 extras on test and 5,729 with 2,598 extras across the fixed 56. On the native 12 its precision rose from 85.0% to 87.1% and recall fell from 73.0% to 70.1%. F1 fell from 0.785 to 0.777 and exact coordinate-and-material matches fell from 73 to 59. All 72 official training views used the atlas-registered light gray concrete camera. The 12 native views used the older seam camera. [Numeric summary](evaluation.json) · [Independent native screen and per-frame results](scored-decisions.zip)

The deployable tree evaluator uses six features and exports its two rejecting paths. Most decisions depend on agreement across the proposed plane. A single weakly matching tile seldom justifies removing a block. The ordinary atlas loader verifies the texture assets. Without a usable atlas, verified model or finite camera the filter keeps its input proposals. Its decoded sets reproduced the research tree on 152 of 152 frames, comprising 27,439 blocks. The parent integration evaluates it again on the actual plane-ranked candidates. [Standalone evaluator](../../../engine/visible_evidence_filter.py)

## Replay

Use the shared cache files under `runs/visible-depth-rounds/inputs` in the stable checkout. They contain the image-derived camera, production masks and counts. Fetch and extract the Visible Builds v1 shards and prepare the ordinary 26.3 atlas. From the experiment checkout run the following commands.

```powershell
python -m benchmark.visible_cell_decisions prepare --splits train --workers 4
python -m benchmark.visible_cell_decisions floor --splits train --workers 4
python -m benchmark.visible_cell_decisions rays --splits train --workers 4
python -m benchmark.visible_cell_decisions train
python -m benchmark.visible_cell_decisions prepare --splits validation test regression --workers 4
python -m benchmark.visible_cell_decisions floor --splits validation test regression --workers 4
python -m benchmark.visible_cell_decisions rays --splits validation test regression --workers 4
python -m benchmark.visible_cell_decisions evaluate --splits train validation test regression
```

The shared placement scorer can invoke `benchmark.visible_cell_decisions:predict_tree` for the original tree or `predict_exported_tree` for its deployment equivalent.
