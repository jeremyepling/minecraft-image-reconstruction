# Can a textured part of one wall measure its quieter end?

[Zoomable comparison by building](comparison.jpg) and [selected regions and measurements](comparison.json).

Keep this experiment out of ordinary builds. Using a texture patch added two correct sizes and one wrong size on the 56 familiar views. The 24 Cloister and Twin Spires views supplied no new count. All existing counts survived, and the number of unmatched proposals with complete counts stayed at 30 across the two suites.

| Views | Promoted counter, exact / wrong | With local patch, exact / wrong | Eligible without complete count after patch |
|---|---:|---:|---:|
| 20 development | 22 / 14 | 22 / 14 | 120 |
| All 56 familiar | 121 / 62 | 123 / 63 | 412 |
| 24 additional | 73 / 40 | 73 / 40 | 185 |

The run froze settings at `f6578ae`, evaluated development, and then ran both suites. For an incomplete result, the predictor selects a local patch inside the rectified surface mask by texture contrast, fits a repeat using that patch, and extends its spacing over the original chart. It checks the opposite end for independent repeat evidence. A textureless end instead requires a distinctive measured joint in the donor patch that lands on the visible far boundary. The plane aspect converts a measured span into the other axis. Existing partial counts and noninteger spans can veto the estimate. No other surface or model is consulted.

On Portico's plaster low-left view with smooth stone ground, two recorded cobblestone pillars measure 1 by 5. The patch completed R18 at 1 by 5 but reported R17 as 1 by 6. On Workshop's high-right masonry view with light gray concrete ground, the same rule completed a polished-andesite surface at 6 by 1. All three estimates had repeated texture on both ends.

Among 473 matched surfaces with an incomplete baseline result, 56 contained multiple recorded materials. Only five of those yielded a fitted donor pitch. One had different dominant materials in the donor and opposite sampling boxes, with spruce planks on one side and white concrete on the other (Crossing R6). It failed the integer-extent check. A synthetic test demonstrates transfer to a flat material when the far boundary lands on a grid line and rejection when that visible boundary shifts seven pixels. We have no successful example of this transfer across materials in the real runs.

Per-picture guesses, accepted and rejected patches, crop coordinates and envelope corners are in [cases.json](cases.json). Scoring projects sampled pixels into the recorded plane and filters them with the quarter-resolution visible-surface map to estimate materials at either end. These labels never enter the predictor. [Scores](scores.json), [settings and fixture hashes](manifest.json), [development](development/scores.json).

```powershell
python -m benchmark.within_wall_round --out runs/my-within-wall --dataset runs/visible-builds-v1
python -m unittest discover -s tests -p test_within_wall_grid.py
```
