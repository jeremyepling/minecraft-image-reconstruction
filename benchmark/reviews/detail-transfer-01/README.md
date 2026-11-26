# Frozen photo edge rule on two additional building families

The previous 56-picture comparison chose a four-pixel edge adjustment with a two-pixel fallback when an existing complete Block Counter estimate changes or a new complete estimate appears. We froze that code at `eefbe59` and ran its next test on all 24 saved validation and test views in Visible Builds v1. Cloisters and Twin Spires each contain three layouts with two palettes and two angles.

| Across all 24 frames | Ordinary neighbor cleanup | With frozen photo rule |
|---|---:|---:|
| Eligible surfaces matched | 250 / 298 | 250 / 298 |
| Same previously matched surfaces lost | 0 | 0 |
| Correct / incorrect complete sizes | 59 / 34 | 59 / 34 |
| Native paired edge error | 4.494 px | 4.402 px |
| Native paired pixel overlap | 88.575% | 88.838% |
| Incorrectly occupied opening samples | 686 | 653 |

**Recommendation.** Promote the frozen photo adjustment after verifying its ordinary entry point. On the twelve Cloister views, edge error was 8.709 → 8.644 px, with all 76 matches and 6 correct / 10 wrong sizes preserved. On the twelve Twin Spires views it was 1.678 → 1.573 px, with all 174 matches and 53 correct / 24 wrong sizes preserved. Overall, native edge error improved on 19 pictures, stayed the same on four, and increased by 0.006 px on one. One Cloister view gained an incorrectly occupied opening sample. Both family totals improved (141 → 136 and 545 → 517). The rule added 7,214 correct and 663 incorrect pixels and removed 9,095 incorrect and 921 correct pixels on fixed matched surfaces.

Each image went through the ordinary six-model build using its RGB pixels and Java 26.3. The comparison reads its finished regions and reproduces all ordinary count records before grading the two methods. Stored ordinary outputs include the build manifest hash and model hashes. Recorded face corners, camera and the native face-ID raster enter only in scoring. We verified every reprojected surface pixel against the native dataset target. The comparison uses the original 50% one-to-one matching protocol, four-face and 24-sample eligibility requirements, and the same native edge measure used for the eight-building regression suite. This transfers a fixed choice to two further layout families; all claims are limited to these full-block daytime fixtures.

[Zoomable stage comparison on six layouts](comparison.jpg) · [Scores by split and layout](scores.json) · [Per-picture scores, gains and losses](cases.json) · [Manifest, code revision and artifact hashes](manifest.json)

Replay with the two released ZIPs in a directory and a new output directory:

```powershell
python -m benchmark.detail_transfer --dataset runs/visible-builds-v1 --out runs/transfer-replay
```

To regenerate the ordinary inputs from RGB, first export the PNGs, run the build command on each PNG into a per-frame directory, and point the scorer at those builds.

```powershell
python -m benchmark.detail_transfer --dataset runs/visible-builds-v1 --out runs/transfer-rgb --inputs-only
Get-ChildItem runs/transfer-rgb/*.png | ForEach-Object { .\Invoke-WallRecovery.ps1 $_.FullName -Through counts -Out (Join-Path runs/transfer-builds $_.BaseName) }
python -m benchmark.detail_transfer --dataset runs/visible-builds-v1 --builds runs/transfer-builds --out runs/transfer-fresh
```

Use the manifest's baseline revision for regeneration when the ordinary default has moved.
