# Ground contacts and excessive depth

A learned rejection step at `f054f4a` removed 545 extra coordinates while losing 63 correct ones on the 104 admitted fixed views. Relative visible-block F1 increased from **88.22% to 89.13%**. A stricter threshold removed 238 extras at a cost of four correct positions. All comparisons decode the actual Minecraft datapack and allow one shared integer translation for the entire build.

| Retrieval method | Fixed 104 | Alternate 20 | Four newer photos |
| --- | ---: | ---: | ---: |
| Prior default (`acfe46b`) | 88.22% | 86.69% | 82.79% |
| Whole-cell rejection, probability ≥ .85 | 89.13% | 87.64% | 83.10% |
| Whole-cell rejection, probability ≥ .98 | 88.68% | 86.83% | 82.79% |
| Same-course neighbor vote | 89.17% | 87.84% | 82.90% |

The .85 rule improved 52 fixed views, tied 41 and reduced F1 on 11. It removed seven hidden-source extras. On the alternate views it removed 121 extras and 26 correct cells. On the four newer photos it removed seven extras and one correct cell. All 18 measured opening widths and 243 clear rays remained correct. Two of 294 previously correct anchored roof cells were lost in the ten fixed canopy views, and one of 232 in the eight alternate views. [Paired per-frame measurements](apply/cases.json) · [Cohorts and roof-to-post checks](apply/summary.json) · [All 200 scored packs](apply/packs.zip)

## What the decision uses

The lower edge of a detected vertical surface has to meet photographically similar floor pixels. The intersections of those rays with the estimated floor must stay close to one grid plane along a contiguous stretch. Visible contact positions supply an observed footprint. They can establish that part of a structure meets the ground; they cannot establish where a legitimate overhang ends.

A cell decision combines distance to the contacts, local and connected occupancy, how much of the candidate owns pixels with the detected face direction, and how well projected texels match the calibrated vanilla atlas compared to shifted tiles. Only current placements can be removed. The check retains the prior build if its camera, contact evidence, atlas or frozen weights are unavailable. The predictor never receives a world readback, reference position, measured margin or frame identity.

On all 72 v2 training images, 12,158 cells had the required contacts and calibrated art. Of these, 1,220 were incorrect. The 72 training photos span six families, each held out in turn. Hard footprint rules relying on distance and weak texture removed 585–840 errors but also deleted 1,134–2,253 valid positions. The selected 72-tree depth-three forest removed 443 errors and 36 valid cells on these family-held-out predictions. A higher threshold removed 120 errors and three valid cells. The trained artifact used all 72 for the frozen cohort replay. [Rule, threshold and family comparisons](family-holdouts.json)

Canopy 06 left kept all 29 previously correct positions. The .85 model reduced extras from 92 to 46 and the depth span from 62 to 52 blocks against a ten-block reference. The six correctly anchored roof-level positions remained in place. The remaining depth and missing roof are still significant. The .98 rule preserved all aggregate measured roof-to-post counts but removed just four of this photo's 92 extras. [Conservative results](conservative/summary.json)

We also tested combining each cell's estimated probability with the median for its touching neighbors on the same horizontal course. On held-out training families the gate removed 445 wrong and 27 correct cells. On the fixed evaluation it removed 570 extras and lost 67 correct, four more losses than the cellwise gate. The small gain changed direction on the newer views. Keep that smoothing experimental. [Neighbor results](neighbors/summary.json) · [Training comparisons](component-train.json)

## Trying the newly exposed pixels

After rejecting cells, the atlas rechecked the original RGB for horizontal faces. A new cell had to own at least four original photo pixels inside a detected structure mask that a rejected cell previously hid. The rejected cells themselves were ineligible. On the 72 development views this recovered 42 correct cells with zero extras, 36 absent from a simple third atlas pass. The horizontal-mask fallback without atlas verification recovered 11 and added two extras. [Development controls](exposed-train.json)

Compared with pruning alone, the frozen exposed-atlas pass recovered 27 correct positions at a cost of two extras on the 104 fixed images. Twenty-two were on three Portico photos. It added two extras and zero correct positions on the 20 alternate views and changed nothing on the four newer images. Roof-to-post and clear-opening results also matched pruning alone. Keep this variant experimental until it transfers to more structures. The .85 pruning gate has the broader standalone gain and merits combined validation. [200-frame comparison](exposed/summary.json) and [decoded outputs](exposed/packs.zip).

## Replay

Use the trained weights from `experiment/grounded-depth-envelope` and the evaluator from `review/framed-depth-round` in registered worktrees. Restore Visible Builds v2 and the documented native fixtures. The placement adapter uses the installed calibrated Java 26.3 atlas. The optional scoring diagnostics operate only on approved complete-structure frames. The prediction stage uses screenshot-derived inputs. From the evaluator checkout:

```powershell
python -m benchmark.framed_depth_eval --candidate-root ../grounded-depth-envelope --candidate benchmark.grounded_depth_round:apply --out runs/grounded-cellwise --workers 3
```

For the .98 comparison use the `conservative` adapter from the same module; for the same-course vote use `neighbors`. To reproduce the training comparisons, use the experiment checkout and run `python -m benchmark.grounded_depth_round features`, `python -m benchmark.grounded_depth_round fit` and `python -m benchmark.grounded_component_trial`. The fit step writes the learned weights in that checkout. The branch exposes experimental callables for combined validation and keeps the ordinary entry point at its baseline.

The post-pruning trial uses the scorer adapter `benchmark.grounded_depth_exposure:apply` with the same arguments. Four focused tests cover missing inputs, missing or damaged weights, excluding rejected cells, and requiring a horizontal face on photographed mask pixels. Run `python -m unittest tests.test_grounded_depth_envelope`.
