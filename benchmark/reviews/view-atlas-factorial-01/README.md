# View-conditioned atlas and full-block lattice, separately and together

Keep `experiment/view-conditioned-atlas` and `experiment/view-atlas-full-block-grid` experimental. The final matcher decreased exact-ID scores on the fixed and decorative cohorts with either placement method. Visible Builds v3 gained a few IDs, nearly all of the standalone gain also appeared with point sampling, and most v3 images still lacked a usable camera. The revised lattice removed 347 net extras from the original lattice on the fixed suite, where it still exported 3,955 more extras than the ordinary build. This review adds benchmark evidence. The measured reconstruction on `main` is `1ca8822`.

## Same frames, cameras and exports

The independent grader checks source images, native truth, building splits, baseline payloads, and the final matcher implementation. It reads all of the exported Java 26.3 datapacks and regrades them against per-view visible truth. Each placement arm receives one geometry-selected integer translation. A material-only arm must reuse its parent's camera and contain exactly the same occupied cells and translation. Every row includes its own pack and source hashes. The exact-ID score grades IDs at visible coordinates. Decorative block states and partial model shape remain outside that score. [Pooled counts and splits](summary.json) · [Per-view counts and fingerprints](cases.json.gz)

| Arm | Export that was scored |
| --- | --- |
| C | Ordinary coordinates and final IDs at `1ca8822`. The fixed 104 reproduce the released packs. |
| A | Ordinary coordinates with the view-conditioned final ID vote. |
| G | Original full-block lattice at `1c6af5c`, with its original IDs and fallback policy. |
| G+A | Identical G coordinates and camera with the same final ID vote as A. This is the matched combination. |
| I | Revised lattice with view-conditioned anchors, scale and phase search, and subface-depth proposals. IDs are the original proposals from this revised lattice. |
| I+A | Identical I coordinates and camera with the same final ID vote as A. This is the integrated alternative. |

## Full-cohort results

Values are pooled F1 percentages. A shares the C placement result, G+A shares G placement, and I+A shares I placement. All 288 images are included, including every empty build.

| Cohort | Arm | Placement strict / facade | Position and ID strict / facade | Correct / missed / extra positions |
| --- | --- | ---: | ---: | ---: |
| Fixed 104/104 | C | 91.27 / 93.42 | 84.68 / 88.51 | 20,773 / 2,929 / 1,046 |
| | A | 91.27 / 93.42 | 82.01 / 87.31 | 20,773 / 2,929 / 1,046 |
| | G | 83.51 / 84.59 | 77.40 / 80.07 | 20,826 / 2,876 / 5,348 |
| | G+A | 83.51 / 84.59 | 76.52 / 79.38 | 20,826 / 2,876 / 5,348 |
| | I | 84.10 / 85.25 | 77.95 / 80.69 | 20,826 / 2,876 / 5,001 |
| | I+A | 84.10 / 85.25 | 77.05 / 79.99 | 20,826 / 2,876 / 5,001 |
| Visible Builds v3 144/144 | C | 8.08 / 8.51 | 6.16 / 6.50 | 9,299 / 204,851 / 6,859 |
| | A | 8.08 / 8.51 | 6.21 / 6.56 | 9,299 / 204,851 / 6,859 |
| | G | 8.28 / 8.70 | 6.15 / 6.49 | 9,606 / 204,544 / 8,189 |
| | G+A | 8.28 / 8.70 | 6.21 / 6.56 | 9,606 / 204,544 / 8,189 |
| | I | 8.26 / 8.69 | 6.16 / 6.50 | 9,577 / 204,573 / 8,152 |
| | I+A | 8.26 / 8.69 | 6.23 / 6.58 | 9,577 / 204,573 / 8,152 |
| Decorative Builds v1 40/40 | C | 54.33 / 56.11 | 43.43 / 45.34 | 7,120 / 9,806 / 2,162 |
| | A | 54.33 / 56.11 | 41.51 / 43.38 | 7,120 / 9,806 / 2,162 |
| | G | 53.94 / 55.57 | 41.67 / 43.42 | 7,287 / 9,639 / 2,806 |
| | G+A | 53.94 / 55.57 | 39.87 / 41.59 | 7,287 / 9,639 / 2,806 |
| | I | 53.90 / 55.54 | 41.61 / 43.34 | 7,289 / 9,637 / 2,829 |
| | I+A | 53.90 / 55.54 | 39.83 / 41.51 | 7,289 / 9,637 / 2,829 |

The ordinary fixed result carries forward the same verified trend score, a 0.00-point delta. The older 112-view strict placement score is 90.50%. The tested/available split counts are in the summary. Camera-registered placement for C/G/I is 91.27%/83.51%/84.10% strict on fixed, 6.63%/6.61%/6.61% on v3, and 54.33%/53.56%/53.67% on decorative. [Expanded baseline](../expanded-baseline-01/README.md) · [Original lattice](../full-block-lattice-01/README.md)

## Final material vote on fixed coordinates

The point and finite-pixel arms share the photographed source pixels, candidate sprites, lighting cost and vote rule. Point sampling reads the sprite at the pixel center. The finite version averages nine samples across the pixel footprint. The table compares exact correct IDs on unchanged occupied cells.

| Cohort | Eligible cells on C / G / I | A minus C | A minus point on C | G+A minus G | G+A minus point on G | I+A minus I | I+A minus point on I |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 19,088 / 18,116 / 18,076 | −608 | −290 | −220 | −61 | −222 | −61 |
| Visible v3 | 13,207 / 13,527 / 13,522 | +65 | +2 | +70 | −4 | +79 | −6 |
| Decorative | 8,519 / 8,553 / 8,548 | −252 | −38 | −244 | −35 | −240 | −32 |

The final vote applies only to usable camera views with occupied cells. It evaluated eligible cells in 104 fixed, 36 v3 and 30 decorative ordinary views. With those ordinary camera groups held constant, C/A strict position-and-ID F1 was 84.68%/82.01% fixed, 33.24%/33.55% v3, and 51.78%/49.49% decorative. The v3 standalone gain was concentrated in the training layouts. On 18 v3 validation views A lost 18 exact IDs, and all 18 v3 test views kept empty ordinary exports. On eight decorative Walled Garden test views, A lost 167 exact IDs, lowering strict position-and-ID F1 from 40.95% to 35.50%. `test-walled-garden-2-cool-high-right` alone lost 88 exact IDs from C to A and 11 against the point control. `test-twinspire3-earth-low-left` lost 65 and 34 by the same comparisons. [Per-view scores and matcher participation](summary.json)

G+A versus G changed strict/facade position-and-ID F1 by −0.882/−0.686 points fixed, +0.060/+0.066 v3 and −1.806/−1.826 decorative. G+A versus A changed it by −5.489/−7.933, −0.003/−0.005 and −1.638/−1.792. The descriptive difference between the final-vote effects on G and C is +1.789/+0.512 points fixed, +0.004/+0.004 v3 and +0.117/+0.125 decorative. It describes how the measured vote effect changes after the lattice changes visible cells, pixel ownership and eligible counts. Both fixed effects are losses. The matched point-to-finite difference is smaller on G than C for fixed and decorative, and it crosses from +2 to −4 IDs on v3.

## Revised anchor and depth proposals

I versus G retained all correct fixed positions while removing 351 extras and adding four. It gained/lost 174/203 correct positions on v3 and 41/39 on decorative after the geometry-selected shift. It added/removed 63/100 extras on v3 and 63/40 on decorative. Its source depth step reassigned 458/49/32 fragment owners across the fixed/v3/decorative cohorts. Those fragment updates are upstream evidence. The position changes above count actual decoded exports. I+A versus G changed strict/facade position-and-ID F1 by −0.354/−0.077 points fixed, +0.079/+0.085 v3 and −1.843/−1.908 decorative. I+A versus A changed it by −4.961/−7.324, +0.015/+0.014 and −1.675/−1.874.

The lattice producer also ran a point control on all 288 images at the **upstream** proposal stage, using the same search and gates as I before the final matcher. Its point-to-finite strict/facade position F1 increased by 0.043/0.048 points fixed and 0.023/0.024 v3, and dropped by 0.065/0.079 points decorative. V3 finite versus point gained 28 relative correct positions and nine extras while losing one correct position after camera registration. In Walled Garden, the original G, same-search point and finite variants respectively joined zero, one and two distinct visible subfaces of a true partial block. The whole decorative cohort recovered 430 partial coordinates with C, 458 with G and 461 with I. Each arm reported zero exact partial IDs because the calibrated catalog lacks those IDs. [Verified upstream point-control counts](summary.json)

### Rescue only

Apply the selected lattice and its IDs only when the ordinary camera rejected the frame. With all original views in the denominator, fixed scores tie C since all ordinary fixed cameras work. V3 G+A and I+A produced respectively 31/108 and 33/108 nonempty rescues, with 278/250 correct positions, 52/68 extras and 33/52 exact IDs on the rescued cohort. V3 placement was 8.31%/8.73% strict/facade with G+A rescue only and 8.28%/8.71% with I+A rescue only. Joint scores were 6.18%/6.52% and 6.19%/6.54%. On decorative rejects, both rescued 4/10, with 34/27 correct positions, 10/6 extras and 14/10 exact IDs. Decorative placement was 54.50%/56.27% and 54.47%/56.24%, with joint scores 43.46%/45.37% and 43.45%/45.36%. As a separate ID-only check, adding the final finite vote to G's rescued positions gained five v3 IDs and tied decorative. Adding it to I's rescued positions gained 14 v3 IDs and one decorative ID. [Rescue and camera groups](summary.json)

## Reproduce

Use the ordinary saved builds and the original lattice, standalone atlas and revised lattice run directories from the sibling worktrees. The scorer reads saved placements and reference maps, and performs no new inference. It reconstructs the point-only C pack from the per-coordinate ID decisions saved by the standalone runner and verifies its score against the producer's receipt.

```powershell
$python = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $python -B -m benchmark.view_atlas_factorial_scores --workers 3
```
