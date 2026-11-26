# Should we accept more of the planes the model already sees?

**Keep the six-model Surface Finder on main.** Both challengers recovered real surfaces. The single-inset rule improved only on the buildings used to choose its settings. The two-model rule found the most surfaces and gave the most correct block dimensions, but erased two correctly detected walls and increased the pixel assignment error on Crossing from 8.4% to 19.5%. [Zoomable sheet](reviews/plane-acceptance-01/comparison.jpg) · [Results for every picture](reviews/plane-acceptance-01/pictures.md)

| On 56 views | Main default | Single inset | Two-model agreement |
| --- | ---: | ---: | ---: |
| Surfaces found out of 598 | 513 | 517 | 523 |
| Newly found / previous matches lost | — | 4 / 0 | 12 / 2 |
| Extra guesses after pairing smaller visible surfaces | 204 | 205 | 220 |
| Correctly assigned visible surface pixels | 94.47% | 94.65% | 94.28% |
| Incorrectly covered opening samples | 601 | 601 | 601 |
| Correct width and height in blocks | 80 | 81 | 88 |
| Undersides found out of 18 | 14 | 14 | 14 |
| Development views, Shopfront and Wing | 137/156 | 141/156 | 144/156 |
| Six previously reviewed buildings | 376/442 | 376/442 | 379/442 |

**Single inset** looks inside an existing wall. A connected region must have a consistent difference in depth or direction from its neighbors, and the cut must leave a connected wall. The three settings all found 141 development surfaces. The strictest did so with the fewest cuts. It recovered four Shopfront surfaces. On Galleries it added two guesses without recovering a wall.

**Two-model agreement** lets a Pixel-wise Planarity outline cross existing surface boundaries when MobileSAM independently supports it. We compared agreement alone, agreement with a local depth check, and partial agreement with that same check. The selected setting recovered **Workshop R13** in both low-left views. In the masonry view its overlap rose from 4% to 79%. At **Crossing**, the two models agreed across a real boundary. R3 went from 68% overlap to 46%, and R6 went from 90% to 49%. Both fell below the 50% needed to count as found. A focused next test would protect an existing boundary when a proposed outline takes substantial area from two separately supported walls.

The 56 views use the frozen vanilla Java 26.3 screenshots and reference geometry. The predictor consumes one screenshot’s saved model evidence. The runner picks settings on the 20 Shopfront and Wing views, writes the selection to disk, and then grades both candidates on all 56. It runs the ordinary texture-based Block Counter on the resulting masks. This round compares surface finding and block dimensions. It reuses model inference and requires no Minecraft launch. [Frozen settings](reviews/plane-acceptance-01/development-freeze.json) · [Run manifest](reviews/plane-acceptance-01/manifest.json) · [Scores across the eight buildings](reviews/plane-acceptance-01/scores.json)

```powershell
.\Invoke-SurfaceRound.ps1 -Round acceptance
```
