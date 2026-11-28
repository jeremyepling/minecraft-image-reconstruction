# Verify a visible wall ruler before expanding

The reference-assisted control replaces one wall chart in each Stable Yard 1 view. It fits a homography to one fully visible recorded block face nearest an existing seed. Each corner is verified against the recorded camera to 0.0001 native pixel. Face ownership is checked at native pixel centers. Ordinary resampling, atlas ranking, chart assembly, export and whole-build alignment then run on the same screenshot. The first control keeps the original Surface Finder mask. The second also restricts the chart to the single recorded plane that owns the anchor.

| Stable Yard 1 view | Arm | Correct / missed / extra | Exact position and material |
|---|---|---|---:|
| Low left | Frozen large-contraction input | 78 / 294 / 1,374 | 26 |
| Low left | Exact ruler, original mask | 76 / 296 / 51 | 32 |
| Low left | Exact ruler, one visible reference plane | 25 / 347 / 0 | 25 |
| High right | Frozen large-contraction input | 63 / 291 / 1,198 | 16 |
| High right | Exact ruler, original mask | 55 / 299 / 54 | 45 |
| High right | Exact ruler, one visible reference plane | 24 / 330 / 0 | 24 |

Both original masks cover several surfaces. The dominant vertical reference plane accounts for about 22% of each source mask. Replacing the ruler while leaving both masks alone removes 2,467 net extras, loses ten net correct blocks, and gains 35 exact material placements. Restricting the chart to the one plane removes the remaining extras and discards other visible surfaces. The diagnostic points to verifying the ruler first and giving the other surfaces their own charts.

[Zoomable Stable Yard comparison](stable-yard-ruler-control.png) · [Paired measurements](stable-yard-ruler-control.json)

Run the two reference controls with `python -m benchmark.floating_stable_yard_ruler_control` using the frozen archives in the registered floating worktrees.

## Source-only comparisons

Both experiments use the same 84 archived large-contraction inputs. Sixty-seven of those packs match the later depth-refit archive. The other 17 retain their frozen large-contraction outputs for this comparison. The six fixed sky inputs are included in the 84.

| Expanded research arm | Correct / missed / extra | Strict / facade placement F1 | Strict / facade position plus material F1 |
|---|---|---|---|
| Frozen baseline | 18,323 / 61,187 / 11,420 | 33.54% / 34.01% | 5.51% / 5.61% |
| Restore a verified adjacent pair | 18,322 / 61,188 / 11,343 | 33.56% / 34.04% | 5.55% / 5.65% |
| Verify source anchors only | 17,741 / 61,769 / 11,165 | 32.73% / 33.23% | 5.51% / 5.62% |
| Keep only verified regions | 6,263 / 73,247 / 2,562 | 14.18% / 14.36% | 4.21% / 4.33% |
| Require agreement from nearby wall rulers before trimming | 18,200 / 61,310 / 8,236 | 34.36% / 34.88% | 5.61% / 5.73% |

The two-wall rule considers 13 restorable corners across 10 views. It can restore a pair when the receiver's independent repeat count disagrees with the shared width, distant atlas repeats support the original grid, and the reassembled pair keeps the previously exported charts and joins. It selects O2 and keeps the other 83 views. O2 removes 77 extras, preserves 84 facade-required correct placements, loses one strict-only underside, and gains 19 exact position and material matches. On the sparse connected-atlas path, O2 currently exports the other tower. The retention rule abstains from replacing that group, and all six sparse controls reproduce the integration result.

The narrower wall guard changes just three expanded views. Distant texture evidence must be absent and at least two nearby wall charts must predict a block height at least twice as large. It cuts Stable Yard low left from 1,374 extras to 28 and high right from 1,198 to 14. Correct placements fall from 78 to 41 and 63 to 22. Both Stable Yard views pick a new whole-build translation and gain a different set of positions. Pillars loses 45 correct positions and 654 extras. The six expanded sky scores rise from 27.50% / 27.73% to 29.27% / 29.59%, with 845 correct, 3,196 missed, and 888 extras. Integrated connected atlas carries forward 12.64% / 12.90%, 273 / 3,768 / 5.

Keep the two-wall rule as an expanded-pipeline candidate. Keep all clipping variants experimental. The guarded trim identifies oversized expansions. To recover the missing visible blocks, the next test should replace the suspect ruler and check its predictions across the wall. The two Stable Yard masks also need separate charts for different visible depths. Flat brightness or sparse atlas evidence by itself offers too little support to trim a wall.

[Zoomable source-only comparison](source-only-results.png) · [Corner repair branch](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-corner-pair-repair) · [Wall expansion branch](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-verified-wall-expansion)
