# Stable Yard. Find a full block and fit each visible depth first

[Full-block size audit](whole-block-pitch.png) · [Source, atlas fit, and lost positions](ruler-stages.png) · [Structural edge and finer ruler comparison](finer-ruler.png) · [Pink depth evidence and masks](depth-masks.png)

## Automatic audit of the very large exports

Scanning all 84 active expanded views found exactly two at or above 1,000 extras. Both are Stable Yard 1. No view fell in the 850–999 range. The next largest is Pillars at 687. The later small-steps archive has the same two outliers, at 1,365 and 1,198. The six integrated sky views have five extras in total.

| Stable Yard view | Extras | True width / old width | True height / old height | Extras with a recorded ruler, same source mask |
|---|---:|---:|---:|---:|
| Low left | 1,374 | 3.44× | 3.57× | 51 |
| High right | 1,198 | 3.35× | 3.38× | 54 |

Four atlas seeds per view grew to 1,452 and 1,261 cells. Nearly all projected centers landed on building pixels. They reused pixels from only 169 and 134 distinct recorded block owners. The native control verifies full face corners and uses the old mask throughout, isolating the ruler as the main cause. The old masks still mix the upper bands, projecting uprights, recessed walls, and foreground detail. [Decoding, native corner and owner receipts](high-extra-audit.json.gz)

## Where the earlier refit lost geometry and materials

Each view lost eight facade-required positions. On low left, five corrected grid centers missed the predicted mask and three covered only 65%, 76%, or 67% of their cell against a 78% threshold. Verified native corners put six of those eight above the existing coverage check. Giving the same atlas matcher the actual visible target crops raises low-left correctness from zero to 58 out of 70 and high-right from four to 54 out of 55. These are reference diagnostics. On low left all 26 original exact materials were on blocks that remained correctly placed, and every retained block got an atlas label. The failures are bad sampled footprints and correspondence to the wrong source region. The high-right fitting window proposed froglight but its visible pixels belong to a pale oak trapdoor. [Native faces, fitted anchor and material transitions](forced-native-trace.json)

A separate reference replay fixes only the earlier atlas grid's directions and perspective while keeping its chosen local width and height. That lowers low-left from 70 correct and 53 extras to 69 correct and 90 extras. High-right goes from 55 correct and 234 extras to 54 correct and 285 extras. Correct skew is necessary for coherent projected tiles, and local size and phase must fit the new directions too. [Fixed-pitch direction control](direction-control.json)

## Use real edges, then challenge the repeat

The source-only runner finds long structural line segments and fits two direction families before searching for atlas windows. It uses the neighboring walls to bracket block sizes and tests independent height and width, rather than locking a source crop to the old homography. On a rectified region, use repeated texture and gradient peaks as candidate spacings, then project real tile windows back into native pixels. Challenge half- and double-size alternatives independently on both axes. Require the selected phase to repeat on widely separated visible material patches. This gives a tiny rectangle from inside one texture a different test from a complete block face.

The edge-aligned low-left fit restores six of the eight missing positions from the first atlas refit and drops one other, reaching 75 correct and 42 exact IDs with 54 extras. The finer fit uses the same direction evidence and searches nearby sizes more densely. A source gate verifies at least two separated same-material atlas matches that beat their half- and double-width or height controls. Low-left supplies four qualifying distant cut-sandstone hits, all on one row, plus the original anchor. It reaches 74 correct with 25 extras and 44 exact IDs. High-right has an acceptable local tile but no far match passes, so it keeps the original. [Earlier edge fit](edge-alignment.png) · [Per-coordinate edge fit audit](edge-aligned-check.json)

| Low-left replay | Correct / missed / extra | Exact position and material |
|---|---|---:|
| Original | 78 / 294 / 1,374 | 26 |
| Initial atlas ruler | 70 / 302 / 53 | 0 |
| Material consensus on the same initial ruler | 70 / 302 / 53 | 36 |
| Structural edge fit and material consensus | 75 / 297 / 54 | 42 |
| Finer source ruler and repeated full-face validation | 74 / 298 / 25 | 44 |

The final source selection uses the original large-contraction export everywhere it abstains. On 84 it selects only the low-left view. Strict/facade F1 rises from 33.54%/34.01% to 33.96%/34.44%, with 18,319 correct, 61,191 missed, and 10,071 extras. Exact geometry plus material F1 rises from 5.51%/5.61% to 5.61%/5.72%. The six expanded sky cases match 27.50%/27.73% and 890/3,151/1,542. The one row supports this measured sandstone scale, with whole-mask verification still unresolved, so the experiment stays isolated.

## What the pink depth check separates

The source depth residual panel subtracts a fitted wall slope. Pink marks nearer pixels and violet farther pixels relative to that estimated slope. The region mask then uses source normals, depth breaks, and appearance continuity, before any texture ruler is fitted. The reference panel shows visible physical planes, colored independently. The low-left source depth still blends the stacked roof steps, while isolating several pillars from recesses. The high-right split separates more roof bands. The charts on the two sides remain different because the evidence differs by view. [Zoomable depth masks](depth-masks.png)

The initial depth-first partitions produced 13 low-left and 24 high-right regions. Low-left plane agreement decreased from the earlier color-depth mask's 75.9% to 61.5%. High-right improved from 83.7% to 85.1%. Eight complete source atlas confirmations supported a 21-cell high-right eave front. Recorded face IDs verify 21 distinct sandstone faces and all 21 local material IDs. The component joins could not attach that eave to the selected 993-cell group. Forcing the depth-first fit loses 27 correct placements across 84, so it stays experimental. Preserving long visible straight edges in a second source split raises low-left plane agreement to 69.6%. The previous color-and-depth split still scores higher, and the new variant did not fit a verified ruler there. Outer, recessed, and step faces each require their own independent visible evidence.

[Edge-aligned ruler results](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/a0e67a7/benchmark/reviews/floating-edge-aligned-ruler-01/README.md) · [Depth-first experiment](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-depth-before-atlas)

The six integrated sky exports still score 12.64% strict / 12.90% facade, with 273 correct, 3,768 missed, and five extras. The automatic high-extra audit is now part of the first-pass review instructions.
