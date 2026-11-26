# Can the tile atlas confirm repeat spacing?

The default counter leaves an axis unresolved when scanline repeats are weak. This experiment tries missing chart spans from cropped pixel repeats, their half- or double-scale alternatives, and the relative shape of the straightened crop when available. It ranks a color shortlist of 20 vanilla textures at up to 38 candidate grids. Each grid searches eight orientations and all cyclic 16 × 16 texel offsets. One set of observed patches determines the best texture, pitch, and offset; the same orientation and phase must fit a disjoint set of patches. A competing half- or double-scale match must score at least 0.10 lower. Existing complete counts are retained.

The frozen test made zero new complete estimates on the 20 development views. Across all 56 it scored 95 exact and 51 incorrect, against 95 exact and 50 incorrect for the ordinary counter. Of 142 proposals with a scored template, one passed: it found the correct stone-brick texture on Crossing R1 and estimated 2 × 2 for a 2 × 3 surface. A valid material and plausible spacing do not guarantee that the sampled window covers the correct span.

On the 82 matched appearances whose recorded visible blocks were all cobblestone, the default got 19 complete sizes right, seven wrong and left 56 unresolved. Workshop R15, a 14 × 5 cobblestone wall, was correct on its predicted chart. With a synthetic 6 × 4 cobblestone panel rotated 90°, shifted, darkened and partly occluded, the ordinary counter returned 6 × 4. The atlas check, exercised by withholding the baseline counts and retaining the crop-derived pitch candidates, also returned 6 × 4 and named cobblestone.

### Portico diagnosis

All examples below use the same low-left plaster screenshot on the smooth-stone floor. For the diagnostic only, change where the image is straightened (four projected corners) and which pixels are admitted (surface mask), retaining the predicted normal and the ordinary pixel-repeat counter. Geometry and material labels are used only in scoring and in these marked diagnostic runs.

| Recorded surface | Normal input | Recorded mask alone | Recorded corners alone | Both |
|---|---|---|---|---|
| 1 × 5 cobblestone side R8 | ? × ? | ? × ? | 1 × 5 | 1 × 5 |
| 6 × 3 white concrete R13 | ? × ? | ? × ? | ? × ? | ? × ? |
| 18 × 1 spruce beam R20 | ? × 5 | ? × 5 | 18 × 1 | 18 × 1 |

The four-corner change alone allows the existing counter to read both textured surfaces. The measured charts for cobblestone were 20 × 148 image pixels from the finder and 20 × 164 using the recorded corners. For the spruce beam these were 588 × 51 and 586 × 38. A separate diagnostic fitting the atlas on recorded-corner crops identified cobblestone and spruce with the correct spacing. With one known face per patch, all five cobblestone faces and all 18 spruce faces were named correctly. White concrete stayed unresolved. More generally, the finder mask can still have errors even in a case where the corner substitution corrected the count.

**Decision.** Keep the atlas fallback experimental. The next candidate should compare alternative *image-derived* chart corners by how consistently texture motifs repeat across different parts of the same crop. Use a separate confidence check for texture-poor faces. [Cases](cases.json) · [Controls](controls.json) · [Dev scores](dev-scores.json) · [All scores](scores.json)

```powershell
python -m benchmark.texture_count_round --method atlas --split development --out runs/atlas-dev
python -m benchmark.texture_count_round --method atlas --out runs/atlas-56
python -m benchmark.portico_atlas_summary --run runs/atlas-56 --out runs/atlas-summary.json
python -m benchmark.portico_atlas_controls --out runs/atlas-controls.json
```
