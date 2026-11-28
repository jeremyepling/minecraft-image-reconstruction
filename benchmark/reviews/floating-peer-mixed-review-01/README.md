# Refit the ruler or separate the wall first

The two source trials share the frozen 84-condition expanded large-contraction baseline. Three views qualified for a peer ruler search because independent charts disagreed with the existing unit size. All comparisons decode ordinary exported blocks and evaluate one whole-build translation. The baseline has 18,323 correct, 61,187 missed, and 11,420 extra placements across 84. On the six fixed sky views it has 890 correct, 3,151 missed, and 1,542 extras. The sparse connected-atlas integration is a separate carried-forward result.

## Refit from nearby walls

The edge-first challenger uses the neighboring wall pitches to bracket a source-image search. The atlas-first version searches for a full tile at those scales before fitting the rest of the wall. Both fit one side of each crop and hold out a second side. Agreement on both axes, a margin over half- and double-pitch candidates, and whole-tile evidence are required before switching. Both verified arms abstained on every chart.

| Paired expanded cohort, 84 views | Correct / missed / extra | Strict / facade F1 | Strict / facade position plus material F1 |
|---|---|---|---|
| Baseline or verified peer fit | 18,323 / 61,187 / 11,420 | 33.54% / 34.01% | 5.51% / 5.61% |
| Force the edge-first fit | 18,226 / 61,284 / 8,281 | 34.38% / 34.91% | 5.62% / 5.74% |
| Force the atlas-first fit | 18,283 / 61,227 / 8,512 | 34.40% / 34.91% | 5.59% / 5.71% |

The atlas-first ruler on Stable Yard low-left matches a local cut-sandstone tile and cuts extras from 1,374 to 53. It keeps 70 of 78 baseline correct positions and has zero exact material matches at those positions. On high-right it keeps 55 of 63 correct positions and cuts extras from 1,198 to 234. The forced cases regress exact material IDs on both views. The two forced methods also switch the exported component on Pillars. [Source comparison](atlas-ruler-comparison.png) · [Independent replay receipts](peer-replay-audit.json) · [Peer experiment](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/614c72b/benchmark/reviews/floating-peer-ruler-refit-01/README.md)

## Combine source regions with the proposed atlas ruler

Apply the same atlas-first source homography to each source-only part, resample its visible support and rerun ordinary assembly. This isolates whether separating the mask makes the proposed ruler easier to verify. The parts still share the offered orientation and pitch. Independent depth and direction estimation for each part would be a different experiment.

| Focused view | Prior | Forced atlas ruler | Forced ruler with source regions |
|---|---|---|---|
| Stable Yard low left | 78 / 294 / 1,374 | 70 / 302 / 53 | 27 / 345 / 5 |
| Stable Yard high right | 63 / 291 / 1,198 | 55 / 299 / 234 | 42 / 312 / 96 |
| Pillars low left | 69 / 314 / 687 | 45 / 338 / 64 | 45 / 338 / 64 |

Only one complete sprite confirmed within any combined part. No part confirmed the ruler on two separated pieces of evidence. Cropping to finer native regions with a shared proposed homography removed additional support and recovered no new placement advantage. The combined arm was exploratory and did not pass the source gate. [Combined packet and decoded scores](combined-source.json)

## Reference-assisted placement opportunity

The recorded visible full-face corners show the old Stable Yard ruler subdivides each real block into roughly 3.5 steps in both directions. Correcting only its local pitch while preserving its previous perspective puts 107 correct blocks across the two views with 97 extras. Correcting local grid offset too gives 113 correct and 89 extras. A whole reference ruler on the original masks gives 131 correct and 105 extras. This isolates the effect of the image projection, beyond local pitch calibration alone.

Using exact frontmost visible ownership to divide the same source masks into four signed planes gives accurate separate charts. Ordinary assembly keeps one selected component and places 46 and 45 correct blocks with zero extras. Adding only their known relative transforms to ordinary assembly gives 121 and 98 correct blocks with zero extras and exact materials. All 219 were sampled from visible native faces inside original source masks. This is a reference replay. There are still 507 missing blocks across the two views. The four planes face the same way at different depths, so a perpendicular edge-join rule has no corresponding connection for them. [Reference plane and placement comparison](reference-planes.png) · [Local pitch replay](pitch-only-control.json) · [Multi-plane decoded receipts](mixed-ruler-reference.json)

## Split the source mask before refitting

A separate source-only trial decomposes each suspect mask using material boundaries and the local source depth field. Across 84 conditions the trigger supplies 40 part charts on three images. The three eligible views had already been reviewed. Verified complete-face fitting accepts none. A partial-patch variant finds one repeatable sandstone fit on the low-left roof in two separate visible patches. It replaces the region's 362 old cells with 21 well-sized cells. Native grading verifies 21 distinct recorded cut-sandstone faces at those centers. The ordinary assembler leaves the 21-cell chart in a separate group and selects a larger group with the old ruler.

| Expanded cohort | Correct / missed / extra | Strict / facade placement F1 | Strict / facade position plus material F1 |
|---|---|---|---|
| Original charts | 18,323 / 61,187 / 11,420 | 33.54% / 34.01% | 5.51% / 5.61% |
| Allow the partial-patch ruler | 18,320 / 61,190 / 11,049 | 33.65% / 34.13% | 5.48% / 5.59% |

The strict split/refit stage ties both Stable Yard views. It removes 12 extras with zero correct positions lost on Pillars in the six expanded sky inputs. Allowing the single partial-patch ruler loses three correct Stable Yard positions and all 26 existing exact material matches on that view. The split reference helps identify planes while the local chart fit identifies a usable ruler. Connecting those accurately measured charts across visible depth steps is the next constraint to solve.

None of the new refit variants was promoted. The six fixed sky integration result remains 12.64% strict / 12.90% facade with 273 correct, 3,768 missed and five extras, carried forward from the prior verified packs. On the six expanded views, the verified peer arms match the 27.50% / 27.73% baseline and the split removes 12 extras, reaching 27.55% / 27.79%.

[Mixed-wall experiment](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-mixed-wall-refit) · [Independent source checks](mixed-candidate-audit.json)
