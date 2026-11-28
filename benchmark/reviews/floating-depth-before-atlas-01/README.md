# Depth separation before local atlas fitting

Stable Yard’s high-right eave produced 21 cut-sandstone assignments on 21 distinct visible faces at height 71. The ordinary assembler exported the larger component and omitted the eave. The low-left view did not produce a verified ruler. Keep this candidate experimental. No integration change landed.

The benchmark begins with the three previously flagged wall expansions among 84 active archived inputs. It measures surface normals and signed local inverse-depth steps on the source screenshot before fitting any new atlas unit. The old atlas expansion supplies eligibility and the unchanged baseline for unconfirmed cells. This experiment does not rerun the whole raw-image surface finder. Each observed support pixel belongs to exactly one piece. Native face ownership enters only scoring. No extra positions are inferred behind an observed surface.

Parallel slopes are estimated from neighboring pixels. Direction and relative-depth bands supply initial regions. Candidate appearance borders are retained when adjacent pixels confirm a consistent depth jump or a normal turn. Locally measured source periods and atlas phase are tested against distant visible patches. A second version preserves long straight photographic borders and recovers vertical tangents from paired horizontal vanishing directions. The comparison requiring complete faces uses whole source tiles with phase-shift controls. The partial variant allows independently checked fragments of the same texture. The guarded version requires complete-face evidence and a single connected set of replacement charts. Ordinal depth alone does not give an integer displacement between independent grids.

| Source | Earlier → depth parts | Dominant native plane share | Precision / recall of internal boundaries within 3 px | Full refit, strict correct / missed / extra |
|---|---:|---:|---:|---:|
| Stable Yard 1, low-left | 17 → 13 | 75.9% → 61.5% | 61.4% / 42.2% → 74.6% / 17.9% | 78 / 294 / 1,374 |
| Stable Yard 1, high-right | 14 → 24 | 83.7% → 85.1% | 58.0% / 61.9% → 60.3% / 63.7% | 61 / 293 / 932 |
| Pillars 01, low-left | 9 → 6 | 95.1% → 95.1% | 40.8% / 54.7% → 71.5% / 47.6% | 44 / 339 / 92 |

Coverage of the original observed mask is 100% in every row. The source correctly ordered 156 of 157 sampled low-left and 255 of 255 high-right contacts where both visible faces were parallel and at different depths. Many true boundaries produced no source cut. The low-left masks recovered 522 of 1,203 native neighboring front-to-top pixel transitions within three pixels, compared with 2,535 of 3,256 on high-right. The counts describe adjacent pixel pairs and include multiple samples per border.

Low-left’s roof fronts merged into a 60,521-pixel piece, so the previous 21-cell roof ruler lost its source witnesses. Preserving long straight borders recovered 69.6% plane purity and 28.6% boundary recall, still below the earlier 75.9% and 42.2%. On high-right, piece 24 verified eight complete eave tiles. Its fitted native area was 1.172 cells per true block. All 21 centers and local material IDs were correct. The newly measured 21-cell component remained separate from the exported 993-cell component. The partial plinth grid hit 11 different real cut-sandstone faces and labeled two correctly. Pillars yielded 31 different true face centers with 24 correct local IDs, at 1.013 cells per native block. Complete-face refitting still lost 25 correct placements on Pillars and two on high-right. Forced local refits change which cells the ordinary assembler exports. The guarded version abstained in all three reviewed cases. The other 81 packs were unchanged by the source eligibility rule. No additional building passed this eligibility rule.

Use `python -m benchmark.floating_depth_before_atlas` for the main pass and add `--straight` for the straight-border comparison. The benchmark requires the frozen archived field, image and chart inputs from the adjacent worktrees. The comparison used the ordinary assembler, decoded packs and one whole-build translation. [Per-frame receipts, mask audits and accepted grids](conditions.json.gz) · [Pooled metrics](summary.json) · [Straight-border comparator](../floating-depth-before-atlas-straight-01/summary.json)

## Floating branch scorecard

| Revision and inputs | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---:|---:|---:|
| Integration `9435934`, sparse six, carried forward | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Expanded six, baseline or guarded candidate | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |
| Expanded six, complete / partial forced | 29.56% / 29.91% | 6.42% / 6.52% | 865 / 3,176 / 947 |
| 84 expanded, baseline or guarded candidate | 33.54% / 34.01% | 5.51% / 5.61% | 18,323 / 61,187 / 11,420 |
| 84 expanded, split only | 33.55% / 34.02% | 5.51% / 5.61% | 18,323 / 61,187 / 11,400 |
| 84 expanded, complete forced | 33.77% / 34.25% | 5.53% / 5.64% | 18,296 / 61,214 / 10,559 |
| 84 expanded, partial forced | 33.82% / 34.30% | 5.54% / 5.65% | 18,296 / 61,214 / 10,403 |
| 84 expanded, straight borders and complete forced | 33.62% / 34.10% | 5.51% / 5.61% | 18,321 / 61,189 / 11,142 |

The main complete refit lost 27 correct placements and removed 861 net extras. Partial refitting lost the same 27 and removed 1,017 net extras. Split-only retained all correct placements and removed 20 extras. The fixed-104 production trend carries forward separately at 91.27% strict and 93.42% facade placement F1.
