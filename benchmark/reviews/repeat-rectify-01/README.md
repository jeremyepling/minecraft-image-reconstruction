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
