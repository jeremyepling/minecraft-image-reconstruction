# Counting on Cloister and Twin Spires

The 24 saved views cover six layouts across two families, two angles and two palettes. Their labels were used to evaluate Surface Finder previously. Settings for these Block Counter trials were frozen before they ran on these families. All 24 ordinary count totals and pairings by count reproduced the archived photo-edge scores. Predictions used each screenshot and its saved model-derived surfaces, with no Minecraft launch.

| Method | Exact size | Wrong size | Eligible without a complete size | Unmatched proposals given complete sizes |
|---|---:|---:|---:|---:|
| Current counter | 59 | 34 | 205 | 15 |
| Previous outline-based crop refit | 70 | 41 | 187 | 16 |
| Texture-guided crop with a reserved strip | 73 | 40 | 185 | 17 |
| Same crop, requiring atlas confirmation | 62 | 34 | 202 | 15 |
| Scale transferred between surfaces | 60 | 38 | 200 | 15 |

Surface Finder matched 250 of 298 eligible surface appearances. All methods kept the 59 previously correct sizes. With full atlas confirmation the three new correct results were a 2 × 6 Cloister surface and two Twin Spires surfaces measuring 6 × 1 and 8 × 10.

**Shared scale.** Five transfers passed on Twin Spires, all between parallel surfaces. Four crossed a two-block depth difference, including all four transfers between different materials. Prismarine bricks supplied the right 6 × 1 dimensions for a light gray concrete face despite the two-block depth difference. The one genuinely coplanar pair correctly put one polished diorite beam eight blocks across and two up from the donor, but undercounted its length at 5 × 1 instead of 6 × 1. No transfer got both the plane and the dimensions right. Depth differences here were measured from the scoring geometry after prediction. The predictor returned an offset on its proposed plane. It did not estimate an independent 3D position.

**Replay.** Keep the two released Visible Builds v1 validation and test ZIPs together. Each matching prediction and each transfer assessment appears in the per-picture results. Raw count evidence, four corners or transferred grid, tile comparisons and the repeat-only result are stored per proposal in the compressed prediction files. [Scores by family and layout](scores.json) · [Per-picture evidence](cases.json) · [Manifest with frozen settings and hashes](manifest.json)

```powershell
python -m benchmark.count_transfer_check --dataset runs/visible-builds-v1 --out runs/count-transfer-replay
```
