# What a repeating patch can tell us about a Portico crop

The ordinary counter already compares masked scanlines at different offsets, checks plausible multiples against 16-texel spacing, and can transfer one measured axis through the estimated plane aspect. A periodic peak is evidence for a texture feature. We need to establish whether it represents one full block, a smaller motif, or multiple blocks.

We tested one crop-only fallback. It compares translated shading and fine texel patterns across small 2D offsets and three bands of the same crop. It looks for a common translation, rejects unrelated strong periods, and preserves already complete counts. Across the frozen 56 views this raised exact sizes from 95 to 99, but wrong sizes rose from 50 to 57. Ten additional axes were right and 28 additional axes were wrong. On 20 development views it added no exact size and one wrong size; on the remaining 36 previously reviewed views it added four exact sizes and six wrong ones. Keep the fallback experimental.

| View and actual material | Actual dimensions | What the image-only crop shows | Same screenshot with recorded corners and visible mask (diagnostic only) |
|---|---:|---|---|
| Portico R20 spruce plank beam | 18 × 1 | A 32.46-pixel horizontal repeat gives the correct 18 across. The vertical peak is at 9.42 pixels, about a quarter of the true 37.97-pixel block pitch. The competing axes disagree with the estimated chart aspect, so the fallback retains the previous partial estimate of ? × 5. | The ordinary counter returns 18 × 1. The horizontal peak sharpens from .67 to .84. The small vertical feature still repeats four times inside that one block. |
| Portico R8 cobblestone side face | 1 × 5 | In the 20 × 148 crop, 67% of pixels are usable and the vertical chart spans about 4.53 true blocks. Neither method finds a reliable repeat. | With the recorded straightening and visible mask, usable area rises to 89%. The 32.94-pixel repeat implies 4.98 vertical blocks, and the ordinary counter returns 1 × 5. |
| Portico R13 white concrete wall | 6 × 3 | No repeat clears the threshold. The crop spans about 6.07 × 2.35 recorded blocks. | Even with recorded surface geometry supplied to the cropper, neither counter finds a supported count. |
| Shopfront R11 cobblestone face | 2 × 4 | A direct vertical shift of 41.16 pixels gives 4.01 repeats. The ordinary counter returns 2 × 4. | The measured period is 41.17 pixels and the ordinary counter again returns 2 × 4. |

The positive Shopfront control and three Portico examples use the smooth-stone floor and the low-left plaster view. Reference materials and projected block spans are opened for scoring only. Crops with recorded geometry change both the sampling chart and support mask; this diagnostic measures their joint effect.

**Could we return the 18 across separately?** As a review hint, yes. In a retrospective check, reporting the independent translation on each previously unresolved axis produced 45 additional hypotheses on the 517 matched surfaces. Only 12 were right and 33 were wrong (8 of 21 horizontal, 4 of 24 vertical). Keep these as scored alternative pitches until a second source confirms their scale. A good next test is to refine straightening before testing spacing again. For ambiguity inside one tile, use rendered atlas texture repeats to check which within-block motifs occur at half or quarter spacing. Purely uniform or sub-resolution evidence can still require abstention.

[Per-example chart evidence](charts.json) · [All independent one-axis hints](partial-hints.json) · [Development](development-scores.json) · [Whole suite](suite-scores.json)

Replay in the repository, using a fresh output directory:

```powershell
python -m benchmark.texture_count_round --method repeat --out runs/my-patch-repeats
python -m benchmark.portico_repeat_diagnostic --out runs/my-portico-charts.json
python -m benchmark.portico_partial_repeats --run runs/my-patch-repeats --out runs/my-partial-hints.json
```
