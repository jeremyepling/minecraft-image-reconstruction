# Can the visible mask give the crop a better perspective?

For each unresolved crop, the candidate fits four robust lines to the edges of the detected mask in the existing chart coordinates. It samples a new quadrilateral only when that quadrilateral overlaps the mask well and enough pixels remain inside the mask. The ordinary Block Counter then estimates dimensions on the new chart. A counted repeat must also be detectable on two disjoint perpendicular bands. Previously complete sizes pass through. The method sees only the existing masks and screenshot. It does not know which example is Portico or which block texture is present. Parameters were frozen before the 20 development views were scored.

| On the same 56 pictures | Shipped count | Mask-envelope crop and band check |
|---|---:|---:|
| Correct sizes among eligible surfaces | 95 | 116 |
| Incorrect sizes | 50 | 61 |
| Eligible surfaces unresolved or missed | 453 | 421 |
| Complete guesses on unmatched proposals | 10 | 14 |
| Correct individual axes among matched surfaces | 238 | 280 |
| Incorrect individual axes | 84 | 95 |

The method accepted 36 new complete sizes out of 714 proposals. Of these, 21 were correct, 11 wrong and four unmatched. All previously correct sizes stayed correct. The two development buildings gained two correct and two wrong sizes. The other six previously reviewed buildings gained 19 correct and nine wrong. Workshop increased from 6 to 14 correct with one additional wrong size. Portico increased from 20 to 26 correct with five additional wrong sizes.

**The original Portico examples remained unresolved.** On its 18 × 1 spruce beam, 93% of the newly fitted chart was usable, compared with 69% in the shipped crop. The new chart still gave no convincing full count. On the 1 × 5 cobblestone face the usable fraction increased from 67% to 71%. The 6 × 3 white concrete patch still gave no measurable repeat. Good coverage alone does not establish the correct perspective or fix within-tile motifs.

Retain this as an experimental proposal for better texture measurement. Its additional wrong sizes and unmatched proposals make it a poor replacement for the default. A follow-up could require an independent square-tile or atlas-scale check before publishing new complete guesses. The two band scores confirm the period repeats throughout a crop; a within-tile motif can pass the same test.

[Development scores](development-scores.json) · [Full suite and building scores](scores.json) · [Per-picture guesses](cases.json) · [Settings, inputs and replay manifest](manifest.json)

```powershell
python -m benchmark.texture_count_round --method rectify --out runs/my-mask-envelope-count
```

### Does atlas confirmation weed out the wrong refits?

We withheld both estimated spans on each of the 36 newly complete refits, remeasured the same predicted quadrilateral, and included its measured repeats among the atlas hypotheses. The frozen atlas matcher challenges half, full, and double pitches and tests its best texture scale and phase on separate regions of the crop. A confirmation requires the same rounded width and height as the refit.

The atlas confirmed **three** refits. Two correct 5 × 7 counts belong to the same Courtyard masonry surface R3 from the high-right angle on both floors; both ranked tuff first. One wrong 2 × 2 count belongs to Crossing R1 on concrete (actual 2 × 3), where stone bricks ranked first. The other 33 checks abstained. Final exact sizes increased from 95 to 97, wrong sizes increased from 50 to 51, and unmatched complete guesses stayed at 10. None of the three confirmations were on development layouts.

Of the 21 new correct sizes, two passed (9.5%). Of the 11 new wrong sizes, one passed (9.1%). None of the four unmatched proposals passed. The confirmation reduced output volume, but the accepted fraction for wrong and right proposals was nearly the same. Keep this test diagnostic. [Every proposal, confirmation, and atlas score](atlas-confirmation.json)

```powershell
python -m benchmark.refit_atlas_check --out runs/my-refit-atlas-check.json
```
