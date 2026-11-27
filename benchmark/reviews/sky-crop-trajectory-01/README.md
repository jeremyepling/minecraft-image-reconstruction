# Consecutive Sky crop corrections

The same 193 saved crops and published native face owners used in the [initial comparison](../sky-crop-reference-01/README.md) establish the fixed pairing. Each follow-up pass starts from the preceding crop. The report recomputes native crop IoU and the added and removed pixels against the fixed target at every pass. A saved snapshot stores the complete state for a pass. Proposal IDs omitted from the snapshot use the ordinary saved crop.

| Condition | Initial mean IoU | Final mean IoU | Initial → final ≥95% IoU | Held-out matched mean IoU |
|---|---:|---:|---:|---:|
| SAM with reference feedback | 44.53% | 78.21% | 2 → 31 of 193 | 42.30% → 79.55% |
| Source-only policy from saved crop | 44.53% | 52.22% | 2 → 21 of 193 | 42.30% → 47.43% |
| Source-only policy from chooser | 52.79% | 54.80% | 33 → 33 of 193 | 50.70% → 50.84% |

The SAM diagnostic uses the exact target to locate the next erroneous region, select a proposal and keep it only when measured IoU rises. It attempted 1,459 prompts and accepted 563. Pass 1 brought 17 crops above the threshold, pass 3 brought 27, and pass 9 brought 31. The held-out buildings finished with 16 of 129 above threshold and the development buildings with 15 of 64. Of the 157 matched crops below threshold, 75 primarily lacked target pixels, 69 primarily included another cube surface, nine primarily included decorative faces, and four primarily included background. Five proposals had no positive full-cube overlap at baseline.

On Viaduct S15, successive positive prompts raised IoU from 1.87% to 36.95%, 61.49% and 89.82%. A later negative prompt trimmed a differently oriented neighboring face and the crop stopped at 90.73%. On Viaduct S01, a negative prompt on a parallel plane removed 121,293 off-target pixels in one edit and also removed 7,117 target pixels. Native edit panels show these tradeoffs, including prompts that leave the crop unchanged.

The source-only policy applies the next update to its current mask. The development-selected stop is pass 4. Starting from the saved crop, held-out improvement persisted across all four passes and 8 of 129 held-out crops reached the threshold. Starting from the chooser, held-out matched IoU peaked at 51.10% in pass 2, then fell to 50.84%. Held-out threshold passes fell from 19 to 17. The chooser-start policy improved mean IoU on the development split. Its held-out result shows limited transfer beyond the stronger starting mask.

A crop can gain fixed-pair IoU by removing pixels belonging to another surface. Across all masks the SAM diagnostic reduced coverage of the complete set of visible cube pixels from 94.93% to 72.58%. Use the paired IoU and union coverage together. Keep these results experimental. This audit does not substitute crops into the Block Counter or assemble a new placement.

## Replay

Run from the Sky crop reference checkout after the three saved trajectory directories exist.

```powershell
$teacher = '../sky-crop-corrections/runs/sky-crop-corrections'
$policy = '../sky-crop-correction-policy/runs/sky-crop-correction-policy'
python -m benchmark.sky_crop_trajectory --trajectory "SAM reference feedback=$teacher" --trajectory "Source-only policy from saved crop=$policy" --trajectory "Source-only policy from chooser=$policy/from-chooser"
```

Open `runs/sky-crop-corrections-audit-01/index.html` for each building’s examples and scores. The report records native four-way edit counts, fixed-target scores, target coverage, threshold crossings, producer receipts and source hashes. It checks the saved SAM point against the preceding mask and exact target, checks the reported before and after IoU, and recomputes the source-only edit totals. The initial comparison review remains available in its own output directory.
