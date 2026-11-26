# Recovering visible faces at mask edges

The image-derived camera and surface depths sometimes leave source pixels unclaimed at the edge of the reconstructed build. For each proposed block face, we projected the vanilla atlas texel centers, checked each against the photograph and original surface mask, and required matching sprite and orientation on two separate parts of the patch. Accepted candidates must own image samples when projected together with the other blocks.

The first trial used the fitted surface depth. The guarded variant required a neighboring block on the same plane and stronger texture agreement. The third trial let an observed patch choose a plane up to two blocks farther or nearer if it had an independently corroborating neighbor. The final trial also checked adjacent cells across a detected mask boundary, requiring 56 observed texel samples and a stronger sprite match. Every proposed face needs direct image evidence.

| Variant added to the prior default at `a16668f` | Train72 correct gained / extras added | Old native transfer12 correct gained / extras added |
|---|---:|---:|
| Ordinary depth | 23 / 2 | 3 / 1 |
| Neighbor and stronger match | 18 / 2 | 1 / 0 |
| Corroborated nearby depths | 54 / 2 | 3 / 1 |
| Include across-mask matches | 80 / 3 | 3 / 1 |

The corroborated nearby-depth rule added 20 correct and three extra positions on the fixed exteriors, seven correct and two extras on Twin Spires, four correct and zero extras on the earlier native views, and no blocks on Cloister. One new Twin Spires extra overlapped hidden structure. These standalone variants retain all previously placed cells. The production comparison also rechecks occlusion after adding new faces.

Recommend incorporating the across-mask and nearby-depth evidence where it improves the composed system. The final decision and composed results appear in the main repo's canopy placement review. [Frozen standalone per-image scores and decoded packs](scoring.zip)

From a checkout containing `benchmark.canopy_compare` and the standard frozen-input scorer, use `benchmark.visible_depth_score --candidate-root PATH_TO_THIS_CHECKOUT --candidate benchmark.canopy_edge_round:crease --split train --baseline PATH_TO_PREVIOUS_SHIPPED --out runs/edge-replay` with `python -m`. Use `:broad`, `:guarded` or `:alternate` for the other rows. For native transfer use `python -m benchmark.canopy_compare --inputs PATH_TO_TRANSFER --baseline PATH_TO_TRANSFER/results/production --candidate-root PATH_TO_THIS_CHECKOUT --candidate benchmark.canopy_edge_round:crease --out runs/edge-native-replay`.
