# Workshop horizon outlines

The horizontal line on each side of Workshop is the distant sky meeting terrain. On the high-right plaster source image, the right-hand proposal contains 1,916 background pixels. The left contains 775 background pixels and four pixels of the roof. The detector discards the right-hand proposal and removes 773 pixels from the left. Its six-pixel remainder preserves all four observed roof pixels.

The rule tests a narrow, horizontally elongated proposal without a usable sampling chart. Both image margins must support a near-constant blue sky. The detector follows the sky-to-ground transition from an open margin and checks the native source colors below that edge. A conflicting foreground pixel or an interruption in the sky limits the extent of the trim. A genuine counted platform keeps its chart. [Horizon rule](../../../engine/horizon_continuity.py) · [Controls for real detail, platforms and disconnected background](../../../tests/test_horizon_continuity.py)

| Evaluation | Views that changed | Background support pixels removed | Structure pixels removed | Entire unmatched proposals removed |
| --- | ---: | ---: | ---: | ---: |
| Horizon only, fixed 104 | 10 | 13,054 | 0 | 4 |
| Horizon only, additional 100 | 8 | 7,635 | 0 | 2 |
| Exterior foreground followed by horizon, fixed 104 | 12 | 18,260 | 0 | 6 |
| Exterior foreground followed by horizon, additional 100 | 14 | 27,843 | 0 | 9 |

The combined audit measures both refinements on the same native pixels. Both retained every matched reference surface and all paired foreground pixels. At Workshop the exterior pass independently removes 2,330 pixels from a proposal above the plank awning. Its mask does not overlap either of the horizon arms. The combined methods remove 9,013 background pixels across the two affected Workshop views, 1,406 on one Galleries view and 202 on one Shopfront view. [Horizon cases](cases.json) · [Combined cases](combined-cases.json)

## Fitting and ordinary export

The original strips help constrain relative plane fits even though their distant terrain lies outside the build. The ordinary horizon pass keeps those original supports internally for fitting. The public surface outlines and crops use only the retained support. A fully discarded strip has no visible-surface row in the public block-count report and no material samples. On the reported view, the 18 public surface and count IDs agree. Every retained surface dimension and material group matched the earlier measurement; the decoded export retained all 230 previous block positions and materials. [Ordinary build verification](ordinary.json)

With the frozen Java 26.3 inputs, run `python -m benchmark.workshop_horizon_eval` for the isolated comparison. To reproduce the combined audit, run `python -m benchmark.workshop_horizon_eval --combined` with the exterior-foreground implementation available. For separate experiment checkouts, pass the sibling `engine/exterior_foreground.py` with `--foreground-source`. Run `python -m benchmark.workshop_horizon_eval --ordinary <placement-output>` to verify a horizon-only ordinary build. The per-view JSON files record each source image hash and the IDs of every trimmed proposal.
