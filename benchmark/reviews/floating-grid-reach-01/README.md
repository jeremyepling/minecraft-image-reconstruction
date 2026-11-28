# How far can an inferred wall grid extend from a matched atlas tile?

Keep the whole-wall experiment's existing extent. Four-cell limits on sparsely anchored walls fixed the principal chapel sky error, but reduced strict placement on all three original-RGB panels. Two genuinely new observatory frames showed the same failure as previously reviewed castles.

The handoff under test is chart cell membership in local integer wall coordinates. We froze two- and four-cell Chebyshev radii in commit `d6133fe`. Each cell must already have passed the whole-wall RGB support rule. A bounded variant additionally requires a detected tile within the specified radius. Two variants apply the same limits only when the original chart has at most three distinct tiles or all tile coordinates are collinear. The cutoff retains the original supported tiles and their assigned IDs. Every retained cell keeps its original material, fallback and confidence. No chart homography, confidence, origin or coordinate changes. The ordinary assembler reruns, including its component selection, followed by export and native visible-only scoring. The ordinary and whole-wall controls reproduce their previous registered and local positions and IDs exactly.

The original 38, first 12 additional frames and six sky composites were previously reviewed. The independent six in the last row represent two original-lighting views per new layout from published train splits. All four rules were frozen before measuring any of these runs. There is partial building overlap between sky composites and original-RGB panels. Keep them separate. The fixed historical 104, remaining v3 and Decorative v1 views, and paired lighting were unmeasured in this branch.

Each entry gives strict position / facade position / strict position-plus-ID / facade position-plus-ID F1 in percent. Each export's decoded blocks and material IDs matched its serialized local coordinates, and the registered strict hits and IDs matched visible labels.

| Panel | Frames | Whole wall | Radius 2 | Radius 4 | Sparse radius 2 | Sparse radius 4 |
|---|---:|---|---|---|---|---|
| original | 38 | 35.34 / 35.55 / 4.51 / 4.53 | 6.97 / 7.05 / 2.35 / 2.38 | 9.95 / 10.05 / 2.95 / 2.98 | 25.51 / 25.68 / 4.39 / 4.41 | 26.17 / 26.33 / 4.39 / 4.41 |
| fresh | 12 | 27.25 / 28.39 / 5.63 / 6.04 | 12.79 / 14.04 / 4.72 / 5.18 | 16.13 / 17.40 / 4.68 / 5.01 | 26.76 / 28.49 / 5.42 / 5.85 | 26.97 / 28.69 / 5.43 / 5.87 |
| sky | 6 | 22.93 / 23.07 / 4.40 / 4.40 | 16.75 / 16.95 / 4.54 / 4.63 | 19.83 / 20.04 / 4.99 / 5.09 | 25.65 / 25.87 / 4.92 / 4.93 | 26.42 / 26.64 / 4.97 / 4.98 |
| supplemental | 6 | 39.92 / 43.08 / 9.27 / 10.00 | 24.20 / 26.57 / 6.47 / 7.10 | 29.69 / 32.47 / 7.81 / 8.54 | 29.64 / 32.32 / 7.61 / 8.30 | 32.02 / 34.86 / 8.34 / 9.08 |

## What changed

On the chapel sky image, two measured tiles seeded 1,549 whole-wall cells. Sparse radius four produced 89. It removed 1,404 extra placements and lost 56 correct positions. Remaining counts were 62 correct, 290 missed and 27 extra. Strict placement rose from 12.41% to 28.12%, while the number of exact material placements fell from 19 to five; typed F1 rose from 2.00% to 2.27% on the much shorter output. All other sky exports tied on strict geometry and typed F1.

Across the original 38 frames that same rule recovered 493 correct positions but lost 4,870. It removed 1,597 extras and added 89. Twelve images regressed, two improved and 24 tied. The castle-02 limestone rear-oblique view alone lost 47.1 strict placement points. On the first additional 12, the rule lost 130 correct positions, recovered 61, removed 429 extras and added 95. On the disjoint six it lost 257 correct positions, recovered six, removed 212 extras and added eight. The two observatory views fell from 36.48% to 15.77% strict placement. The two watchpost and two mill views tied. Few matched atlas tiles can support a correct grid over a large wall, so their count or collinearity alone cannot identify overextension.

[Chapel and observatory source overlays](reach-overlay.jpg) · [Original 38 per-frame, layout and cohort scores](original.json.gz) · [First 12 additional frames](fresh.json.gz) · [Six actual sky composites](sky.json.gz) · [Six new original-RGB frames](supplemental.json.gz)

## Reproduce

Use the benchmark Python environment with the saved original, fresh and six-sky whole-wall inputs in their registered worktrees. Run these commands from this branch.

```powershell
python -m benchmark.floating_grid_reach original fresh sky supplemental --workers 5
python -m benchmark.floating_reach_report original fresh sky supplemental
```
