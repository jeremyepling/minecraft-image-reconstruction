# Expanded Split-Border tower depth split

## Decision for the next round

Retain Expanded Split-Border on main. Retire this bypass as a challenger for the tower. It left every decoded coordinate and material ID unchanged across the tower, the six preselected holdout views, and both views where the stock depth split activated. No reconstruction behavior was promoted.

## Tower trace

The supplied tower screenshot SHA-256 is `787c1407910301b91d0fcc6b053650b4ca944d6b4de32779b438f5da90201731`. The ordinary build and the bypass consumed the same verified source fields. The 66 source masks and Block Counter output also match the historical `4bcc191` run. No surface had a complete two-axis count. Expanded Split-Border formed two wall charts, compared with zero certified grids on the historical path.

Both new runs assembled 112 blocks from the broad right wall, with one selected chart and no joins. Orientation dropped the 218-cell left-wall chart. Its estimated grid directions were both steep in image coordinates, with a verticality gap of 0.105 against the required 0.20. The depth partition never activated. The tower has no visible reference world, so positional and material accuracy are unmeasured.

## Comparison

The [replay script](../../floating_tower_partition_bypass.py) intercepts the post-expansion chart partition. The candidate passes the original cells and chart IDs to the same border checks and assembly. The stock split is computed for the audit, so this comparison measures placement behavior only. A synthetic two-part chart verified the stock wrapper and bypass outputs.

The six full-block noon holdouts from Floating Front 100 repeat the previous round's exact case IDs. They comprise Windmill 013, 055 and 097 and Vault 014, 056 and 098. None requested a split. To exercise the same intervention when active, the candidate also ran on Water Tower 040 (validation) and Skybridge 073 (train), chosen from the saved baseline's split flags before their candidate outputs were inspected. Those views are an activation diagnostic and remain separate from the holdouts. Both diagnosis packs remained exact matches. Stock selected two partition siblings and joined them in each view. The bypass selected their original parent chart and produced the same blocks without that join.

The fixture release SHA-256 is `f3e416acdbd6e0fa171260997fcdf62a86bc0c42c001d3a16d13e3f0e3c01187`. Scores use the ordinary Expanded Split-Border decoder, visible references and one whole-build translation. Decorative views and the fixed-104 production trend were outside this round.

## Floating branch scorecard

Measured implementation revision `b0b6e3b`, present on current main at `c33f3b0`. Counts are correct / missed / extra.

| Matched cohort | Method | Strict / facade placement F1 | Strict / facade position + material F1 | Strict counts | Facade counts |
|---|---|---:|---:|---:|---:|
| Six full-block noon holdouts | Baseline — Expanded Split-Border | 26.48% / 27.00% | 3.78% / 3.87% | 301 / 1,317 / 354 | 300 / 1,268 / 354 |
| Six full-block noon holdouts | Candidate 1 — Depth split bypass | 26.48% / 27.00% | 3.78% / 3.87% | 301 / 1,317 / 354 | 300 / 1,268 / 354 |
| Two activation diagnostics | Baseline and Candidate 1 | 5.66% / 5.67% | 0% / 0% | 89 / 94 / 2,872 | 89 / 91 / 2,872 |

Paired geometry gained and lost zero correct blocks, added and removed zero extras, and changed zero material IDs. The next discriminating tower test would correct only the left-wall orientation and measure whether assembly can retain it.
