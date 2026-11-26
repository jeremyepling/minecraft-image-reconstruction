# Plane detail experiments

The first trial asked how much the four-pixel registration costs. A fresh run of the pinned Pixel-wise Planarity model retained its 800 × 500 segment IDs before registration on all 52 distinct frozen screenshots. With 1,600 tokens, four-pixel registration exactly reproduced all 52 previously cached relaxed segmentation maps after renumbering the segment IDs. A second inference run changed only the token budget to 2,500. Compressed PNGs contain the numerical segment IDs. The predictors use those IDs with the shipped single-image predictions.

On the 20 development views the best available raw segment per recorded surface averaged 62.76% overlap with recorded geometry. Registering the same segmentation at two and four pixels produced 62.84% and 62.79%. The shipped Finder averaged 77.19% on the same test. This diagnostic allows each recorded surface to select any output region, including one selected by another recorded surface. The selections do not enter any predictor. The ordinary one-to-one surface-match score is separate.

| On development | Correct dimensions | Incorrect dimensions | Paired outline error, px | Paired pixel overlap |
|---|---:|---:|---:|---:|
| Shipped Finder | 17 | 13 | 5.081 | 95.618% |
| Follow nearby plane boundary, four pixels | 17 | 14 | 5.048 | 94.967% |
| Follow nearby plane boundary, two pixels | 17 | 14 | 5.137 | 94.844% |
| Add only, two pixels | 17 | 13 | 5.062 | 95.644% |
| Add only, two pixels and 2,500 tokens | 16 | 14 | 5.069 | 95.627% |
| Add only, four pixels | 17 | 13 | 5.050 | 95.659% |

**Why the test changed.** Replacing both sides of an edge with two-pixel plane evidence removed 71,641 correct pixels while removing 9,748 wrong ones. Additions were more reliable at 2,740 correct vs 668 wrong. The second trial kept the first surface intact and allowed only additions within the union of already proposed visible pixels. It tested 1,600 vs 2,500 tokens and froze the original budget for regression; the higher budget gave the wrong dimensions on an additional surface. The final four-pixel control isolated whether these gains required finer sampling.

**Four-pixel control on all 56 views.** Mean paired native boundary distance fell from 3.824 to 3.804 pixels, pooled overlap rose from 95.142% to 95.191%, and 517 surface matches and 95 correct block sizes held. One incorrect guess became unresolved. On Wing plaster high-right with stone floor, the method added 1,096 correct pixels to R2 without adding a wrong pixel. At the lower-left angle on the same floor it added 386 wrong pixels to R3 without adding a correct pixel. The eight buildings have previously been reviewed.

[Every tested setting and every picture](trials.json) · [Raw, sampled and shipped diagnostic per surface](diagnostic.json) · [Original token budget and fields](../plane-detail-fields-01/manifest.json) · [Higher token budget and fields](../plane-detail-fields-2500/manifest.json)

Replay this series from the committed numerical model fields with a fresh output directory:

```powershell
python -m benchmark.plane_detail_selection --out runs/plane-detail-replay
```

To regenerate fields, use the model environment configured for Pixel-wise Planarity and run the following twice with distinct output directories. Use `--tokens 1600` and `--tokens 2500`, respectively. The assets directory contains the pinned upstream source, utils3d, and model.pt.

```powershell
python -m benchmark.plane_detail_fields --assets <assets-directory> --tokens 1600 --out runs/plane-fields-1600
```
