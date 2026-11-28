# Floating horizontal-cell transfer check

On six views from three previously unused layout families, whole-wall resampling raised strict placement F1 from **3.87% to 18.85%** and moved strict position-plus-ID F1 from **3.49% to 3.39%**. The published horizontal-cell arm added four unsupported placements on Stable Yard 2 high-right and left the correct-position and exact-ID counts unchanged. Keep both methods experimental. The earlier gains justify retaining their runners for comparison, and the new holdout shows weak transfer from the horizontal cells.

| Frozen panel | Views | Method | Strict position | Facade position | Strict position + ID | Facade position + ID | Correct / missed / extra positions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Earlier test images | 38 | Ordinary | 3.72% | 3.76% | 1.54% | 1.55% | 1,127 / 58,311 / 107 |
| | | Whole-wall | 35.34% | 35.55% | 4.51% | 4.53% | 13,895 / 45,543 / 5,302 |
| | | Horizontal cells | 35.64% | 35.82% | 4.63% | 4.65% | 14,065 / 45,373 / 5,422 |
| First fresh images | 12 | Ordinary | 6.75% | 7.39% | 4.31% | 4.76% | 263 / 7,251 / 12 |
| | | Whole-wall | 27.25% | 28.39% | 5.63% | 6.04% | 1,409 / 6,105 / 1,418 |
| | | Horizontal cells | 30.43% | 31.11% | 6.85% | 7.36% | 1,600 / 5,914 / 1,401 |
| Supplemental images | 6 | Ordinary | 14.22% | 15.70% | 3.94% | 4.35% | 249 / 3,004 / 0 |
| | | Whole-wall | 39.92% | 43.08% | 9.27% | 10.00% | 900 / 2,353 / 356 |
| | | Horizontal cells | 39.92% | 43.08% | 9.27% | 10.00% | 900 / 2,353 / 356 |
| Floating sky composites | 6 | Ordinary | 12.64% | 12.90% | 4.68% | 4.77% | 273 / 3,768 / 5 |
| | | Whole-wall | 22.93% | 23.07% | 4.40% | 4.40% | 901 / 3,140 / 2,915 |
| | | Horizontal cells | 22.93% | 23.07% | 4.40% | 4.40% | 901 / 3,140 / 2,915 |
| New holdout | 6 | Ordinary | 3.87% | 4.45% | 3.49% | 4.06% | 91 / 4,522 / 0 |
| | | Whole-wall | 18.85% | 20.13% | 3.39% | 3.71% | 700 / 3,913 / 2,116 |
| | | Horizontal cells | 18.83% | 20.12% | 3.39% | 3.70% | 700 / 3,913 / 2,120 |

The two views of each new layout use the same palette and contrasting elevations and sides. The three layouts come from dataset-train splits and were selected before running the comparison. Their layout families did not appear in the 62 older conditions. [Frozen source and scoring fingerprints](frames.json)

| New layout | Dataset | Tested / available | Horizontal cells strict / facade position | Horizontal cells strict / facade position + ID |
| --- | --- | ---: | ---: | ---: |
| Arcbridge 2 | Visible Builds v2 | 2 / 144 | 18.04% / 21.03% | 7.62% / 9.29% |
| Amphitheater 02 | Visible Builds v3 | 2 / 144 | 17.62% / 18.98% | 2.25% / 2.43% |
| Stable Yard 2 | Decorative Builds v1 | 2 / 40 | 25.15% / 25.91% | 7.19% / 8.30% |

Stable Yard 2 high-right is the only new view that produced horizontal charts. Two charts proposed seven cells, and the assembly exported four additional positions. Scoring counted all four as extras. Amphitheater 02 low-left isolates a separate whole-wall failure with 132 correct positions, 1,540 extras and seven exact IDs. Its ordinary export had 32 correct positions, all with exact IDs. The six-view whole-wall arm increased exact IDs from 82 to 126 while its much larger output lowered typed F1.

## Second transfer check

The six second-panel views use different screenshots and layouts from all 68 preceding conditions. Stepped Hall 2 and Shrine 02 are new layout families. Market Court 1 is a new layout in the Market Court family. All five Decorative Builds v1 families appeared earlier, including Market Court 2 in the first 12 fresh views. Keep this panel separate when checking post-hoc refinements. [Second-panel frozen inputs](holdout2-frames.json)

| Method | Strict position | Facade position | Strict position + ID | Facade position + ID | Correct / missed / extra positions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ordinary | 8.27% | 8.73% | 2.16% | 2.37% | 134 / 2,972 / 0 |
| Whole-wall | 22.15% | 23.16% | 4.93% | 5.27% | 499 / 2,607 / 901 |
| Horizontal cells | 22.15% | 23.16% | 4.93% | 5.27% | 499 / 2,607 / 901 |

| New layout | Dataset | Tested / available | Horizontal cells strict / facade position | Horizontal cells strict / facade position + ID |
| --- | --- | ---: | ---: | ---: |
| Stepped Hall 2 | Visible Builds v2 | 2 / 144 | 53.31% / 54.49% | 16.43% / 16.79% |
| Shrine 02 | Visible Builds v3 | 2 / 144 | 13.18% / 14.43% | 3.09% / 3.39% |
| Market Court 1 | Decorative Builds v1 | 2 / 40 | 22.47% / 21.86% | 2.37% / 2.48% |

All six views exported blocks and resolved a direct camera orientation. Market Court 1 high-right proposed a six-cell horizontal chart. The two arms exported identical coordinates and block IDs on all six views. [Second-panel decoded receipts](holdout2-results.json.gz)

## Export audit

The 62 older views were re-scored from the existing decoded packs. Published whole-wall and horizontal-cell grades matched the archived `6f5df52` results. The six new views ran the existing wall stages from `ae27198` and the published horizontal-cell runner from `6f5df52`. Predictions used one original source image and its verified image-derived model fields with the Java 26.3 atlas. Reference camera orientation, frontmost labels and visible blocks were reserved for scoring. Each exported build used one XYZ translation for all geometry and material grades.

All six new views exported blocks and resolved a direct camera orientation. The earlier Castle 03 ashlar high-right view produced no blocks in any of the three arms and counted as a complete miss. The two Cloister 02 whole-wall and horizontal-cell exports lost the direct orientation anchor. The shared-chart compass transferred the ordinary orientation and reproduced the published grades in both views. [Per-view receipts and dataset totals](results.json.gz)

Fixed 104 production scores are carried forward at **91.27% strict** (20,773 correct, 2,929 missed, 1,046 extra) and **93.42% facade** (18,952 correct, 1,624 missed, 1,046 extra). This experiment has no production delta.

## Replay

Run from this checkout with the published `floating-horizontal-bridges` worktree, frozen datasets and baseline artifacts available. The producer writes private outputs to `runs/floating-patch-validation`. The scorer decodes inventories, recomputes each selected whole-build shift and compares visible geometry and material counts with its original score receipt.

```powershell
$worktrees = (Resolve-Path ..).Path
$env:PYTHONPATH = "$worktrees/surface-round/runs/depth-probe/runtime;$worktrees/surface-round/runs/mlsd-probe/lib"
$runtime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$setup = 'import sys; sys.path.append(r"../mask-scalelsd/runs/env/Lib/site-packages"); '
& $runtime -B -c ($setup + 'from benchmark.floating_patch_validation import main; main()') --workers 3
& $runtime -B -c ($setup + 'from benchmark.floating_patch_scores import main; main()') --workers 3
& $runtime -B -c ($setup + 'from benchmark.floating_patch_validation import main; main()') --holdout2 --workers 3
& $runtime -B -c ($setup + 'from benchmark.floating_patch_scores import main; main()') --panel holdout2 --workers 3
```
