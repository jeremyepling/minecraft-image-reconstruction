# Can a per-surface confidence switch improve the widened depth chooser?

Keep `experiment/plane-switch-confidence` experimental. A confidence switch choosing between the old narrow RGB plane ranker and the widened repeat-aware plane ranker failed to improve the wider chooser when tested on excluded building families. This comparison measures the selected planes before downstream wall-junction decisions, rasterization and placement.

Across 72 training screenshots from six buildings, 1,165 surface masks had candidate planes. The two rankers disagreed on 97 masks. The wider ranker better explained visible samples on 58, the older ranker won on 20, and 19 tied. The wider chooser gained 26,775 correctly assigned visible samples while losing 2,211. A switch with access to the answer could therefore recover at most 2,211 further samples across these masks.

| Family-excluded decision | Correctly assigned visible surface samples | Change from wider chooser |
|---|---:|---:|
| Narrow RGB depth ranker | 731,671 | −24,564 |
| Wider repeat-aware ranker | 756,235 | 0 |
| Perfect choice per mask | 758,446 | +2,211 |
| Fixed confidence margin | 754,578 | −1,657 |
| Learned switch at 50% probability | 756,228 | −7 |
| Learned switch at 75% probability | 756,232 | −3 |

The 50% switch returned to the narrow choice seven times. It recovered 23 visible samples and discarded 30. The 75% switch reverted two choices, costing three samples with none gained. The fixed confidence rule returned 11 surfaces to the old plane, costing 1,657 samples.

## Training and scoring

All comparisons used the stored train-only, image-derived candidate features, the previously frozen six excluded-family fits for **both** rankers, and the visible face labels for the corresponding single screenshot. A sample counts as correct only when its detected surface axis and chosen integer plane match its visible labeled face. Multiple sampled pixels can belong to the same placed block. No decoded-pack gain is claimed.

For each outer family, a second family was held out in turn from the other five. Both rankers were fitted on the remaining four to produce switch-training examples on that second family. A regularized logistic model learned only from disagreements with distinct visible-sample counts. The loss weights each disagreement by the square root of its visible-sample difference. Features include the two rankers' score margins, differences in atlas, repeat, texel-edge and outline evidence, whether the level retained the baseline, surface orientation and mask size. The model and standardization were fitted independently for each outer family. The fixed margin rule reverted when the widened score advantage was below 0.5, the old score favored its own choice by more than 1.0 and the baseline flag favored or tied the old choice.

The comparison was retired on grouped development results. It did not use the Shopfront images, the Cloister/Twin Spires test suites or either independent native suite for fitting or testing. A subsequent attempt would need stronger independent evidence of mask merges or occlusion.

[Per-family scores, every changed plane, selector probabilities and checksums](nested.json) · [Wider ranker's training baseline](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/783380a/benchmark/reviews/canopy-plane-chooser-01/development.json)

```powershell
python -m benchmark.plane_switch_confidence --features ../canopy-plane-chooser/runs/plane-choices --old ../atlas-plane-depth/runs/plane-depth/pixels-72.json --new ../canopy-plane-chooser/benchmark/reviews/canopy-plane-chooser-01/development.json --out runs/plane-switch/nested.json
```
