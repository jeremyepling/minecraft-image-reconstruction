# Precision and recall after replacing the floating wall ruler

Keep the source face proposer experimental. On the original 86 views it removed 5,212 net extra placements and lost 486 net correct placements. Strict position F1 rose from 31.47% to 32.11%, with gains on 14 views, declines on 23 and ties on 49. Two extra-heavy views supplied 4,043 of the 5,212 fewer extras. On the other 84 views, position F1 fell from 32.34% to 32.08% and exact position-plus-ID F1 fell from 4.82% to 4.80%.

## Paired counts

The first value is the whole-wall sampler with horizontal cells. The second uses the source-discovered face before the same horizontal assembler. Each export is decoded and receives one geometry-selected integer translation for the whole build. Precision counts correct positions among all placed positions. Recall counts correct positions among all visible positions. Exact IDs count placements that match both position and block ID using that same registration.

| Panel | Views | Correct positions | Misses | Extras | Exact IDs | Precision | Recall | Position F1 | Position + ID F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 38 | 14,065 → 13,824 | 45,373 → 45,614 | 5,422 → 5,101 | 1,828 → 1,792 | 72.18% → 73.05% | 23.66% → 23.26% | 35.64% → 35.28% | 4.63% → 4.57% |
| First additional layouts | 12 | 1,600 → 1,575 | 5,914 → 5,939 | 1,401 → 1,349 | 360 → 342 | 53.32% → 53.87% | 21.29% → 20.96% | 30.43% → 30.18% | 6.85% → 6.55% |
| Supplemental | 6 | 900 → 854 | 2,353 → 2,399 | 356 → 324 | 209 → 246 | 71.66% → 72.50% | 27.67% → 26.25% | 39.92% → 38.55% | 9.27% → 11.10% |
| Floating sky | 6 | 901 → 888 | 3,140 → 3,153 | 2,915 → 1,542 | 173 → 193 | 23.61% → 36.54% | 22.30% → 21.97% | 22.93% → 27.45% | 4.40% → 5.96% |
| Holdout A | 6 | 700 → 685 | 3,913 → 3,928 | 2,120 → 2,119 | 126 → 121 | 24.82% → 24.43% | 15.17% → 14.85% | 18.83% → 18.47% | 3.39% → 3.26% |
| Holdout B | 6 | 499 → 419 | 2,607 → 2,687 | 901 → 380 | 111 → 106 | 35.64% → 52.44% | 16.07% → 13.49% | 22.15% → 21.46% | 4.93% → 5.43% |
| Holdout C | 6 | 1,815 → 1,817 | 5,973 → 5,971 | 4,573 → 4,350 | 145 → 168 | 28.41% → 29.46% | 23.31% → 23.33% | 25.61% → 26.04% | 2.05% → 2.41% |
| Holdout D | 6 | 513 → 445 | 1,501 → 1,569 | 2,979 → 290 | 164 → 112 | 14.69% → 60.54% | 25.47% → 22.10% | 18.63% → 32.38% | 5.96% → 8.15% |
| Original 86 | 86 | 20,993 → 20,507 | 70,774 → 71,260 | 20,667 → 15,455 | 3,116 → 3,080 | 50.39% → 57.02% | 22.88% → 22.35% | 31.47% → 32.11% | 4.67% → 4.82% |
| New holdout E | 6 | 335 → 344 | 1,494 → 1,485 | 1,490 → 1,487 | 205 → 204 | 18.36% → 18.79% | 18.32% → 18.81% | 18.34% → 18.80% | 11.22% → 11.15% |

The view counts use independent comparisons after registration. Each placement can be gained or lost even when the number of correct positions in the view stays constant. Break-even uses one unit of cost for a missed position and a variable cost for an extra. Where both quantities improve, the proposer wins at any nonnegative extra cost.

| Panel | Correct gained / lost | Extras added / removed | Exact IDs gained / lost | Geometry F1 better / worse / tied | ID F1 better / worse / tied | Net extras saved per net hit lost | Break-even extra cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 605 / 846 | 276 / 597 | 254 / 290 | 4 / 10 / 24 | 5 / 9 / 24 | 1.33 | 0.751 |
| First additional layouts | 52 / 77 | 89 / 141 | 107 / 125 | 3 / 4 / 5 | 3 / 3 / 6 | 2.08 | 0.481 |
| Supplemental | 18 / 64 | 2 / 34 | 102 / 65 | 1 / 3 / 2 | 2 / 2 / 2 | 0.70 | 1.438 |
| Floating sky | 10 / 23 | 1 / 1,374 | 32 / 12 | 1 / 1 / 4 | 1 / 0 / 5 | 105.62 | 0.0095 |
| Holdout A | 0 / 15 | 1 / 2 | 0 / 5 | 0 / 2 / 4 | 0 / 2 / 4 | 0.07 | 15.000 |
| Holdout B | 7 / 87 | 11 / 532 | 5 / 10 | 1 / 1 / 4 | 1 / 1 / 4 | 6.51 | 0.154 |
| Holdout C | 62 / 60 | 0 / 223 | 23 / 0 | 1 / 0 / 5 | 1 / 0 / 5 | — | — |
| Holdout D | 56 / 124 | 33 / 2,722 | 32 / 84 | 3 / 2 / 1 | 2 / 3 / 1 | 39.54 | 0.0253 |
| Original 86 | 810 / 1,296 | 413 / 5,625 | 555 / 591 | 14 / 23 / 49 | 15 / 20 / 51 | 10.72 | 0.0932 |
| New holdout E | 11 / 2 | 0 / 3 | 21 / 22 | 2 / 0 / 4 | 1 / 3 / 2 | — | — |

On the original 86 views, 10 strictly improved and nine strictly declined on the joint correct/extra counts, with 49 ties and 18 tradeoffs. Outside the two largest outliers the proposer saved 1,169 extras for 420 fewer correct placements, or 2.78 per hit lost. That puts break-even at 0.359 of a miss per extra.

## Where the totals move

- **Hillside Chapel, low-left sky composite.** Correct positions fell from 118 to 107, extras fell from 1,431 to 58, and exact IDs rose from 19 to 39. The other five sky frames saved zero extras and lost two net hits. Their strict F1 fell from 26.29% to 26.23%.
- **Riverside Mill 1, high-right.** Correct positions fell from 152 to 97, extras fell from 2,723 to 53, and exact IDs rose from five to six. The other five holdout D views fell from 33.50% to 32.78% strict F1 and from 159 to 106 exact IDs. The low-left Mill view alone fell from 52 to eight exact IDs. Across all six, correct IDs among position matches fell from 31.97% to 25.17% even as position-plus-ID F1 rose.
- **Arcade 1 and Arcbridge 1, new holdout E.** The two high-right views gained eight and one correct positions respectively. The former lost seven exact IDs while the latter gained eight. Four other exports retained their position counts. Whole-wall and horizontal placement lists were byte-identical on all six views. Amphitheater 01 low-left still had 1,479 extras and 69 correct placements with either method.

## Holdout E

The six frames were selected before replay using unused layout IDs from the three packaged suites. The deterministic selection used the alphabetically first two nonlegacy unused Visible Builds v2 layouts from distinct families and the first unused Visible Builds v3 layout. Each supplied high-right and low-left from its existing palette. All ten Decorative Builds v1 layout IDs were already represented in the first 86 views. Arcade, Arcbridge and Amphitheater have different layouts in earlier panels, so this is a layout holdout within previously tested families. Available pack sizes are 144 Visible v2, 144 Visible v3 and 40 Decorative v1. The original run measured 26, 36 and 24 conditions respectively, including 35 distinct Visible v3 and 23 distinct Decorative images. The new holdout adds four Visible v2 and two Visible v3 images. Its key is `holdout5` in the machine-readable replay. [Frozen RGB and native reference hashes](holdout5-frames.json)

The baseline used `7958722` with the published horizontal assembler from `6f5df52`. The proposer was frozen at `a4ce17a` and replayed through the `e1f0218` adapter. The audit verified the local and world-registered block coordinates and IDs for each pack, recomputed one whole-build translation under the recorded compass rule, and matched the published per-view scores and gross changes on the original 86. The scorer reads visible face labels and material IDs after prediction. The three archived compass transfers include two Cloister horizontal exports and one Cloister source-face export, and they agree with the stored registration. [Raw paired score and pack receipt](paired-audit.json.gz)

## Floating default scorecard

The actual floating default is connected-atlas on `experiment/floating-visible-builds`. Runtime was measured at `52de2c9` and verified here under documentation revision `c0accfd`. Each row uses the same six sky composites. Parentheses show percentage-point change from connected-atlas.

| Six-sky method | Correct / missed / extra | Exact IDs | Strict position F1 | Facade position F1 | Strict position + ID F1 | Facade position + ID F1 |
|---|---:|---:|---:|---:|---:|---:|
| Connected-atlas default | 273 / 3,768 / 5 | 101 | 12.64% | 12.90% | 4.68% | 4.77% |
| Whole-wall or horizontal candidate | 901 / 3,140 / 2,915 | 173 | 22.93% (+10.29) | 23.07% (+10.18) | 4.40% (−0.27) | 4.40% (−0.37) |
| Source face proposer | 888 / 3,153 / 1,542 | 193 | 27.45% (+14.80) | 27.68% (+14.78) | 5.96% (+1.29) | 5.99% (+1.22) |

Only this benchmark and audit land on `experiment/floating-extra-audit`. The floating default is unchanged. The fixed 104 production score is carried forward at 91.27% strict and 93.42% facade F1, with strict counts 20,773 / 2,929 / 1,046 and facade counts 18,952 / 1,624 / 1,046. The production delta is zero. [Floating integration baseline](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/52de2c9/benchmark/FLOATING_BRANCH.md)

## Replay

Run from this checkout with the frozen registered sibling worktrees. All new outputs go under this checkout's `runs` directory.

```powershell
$worktrees = (Resolve-Path ..).Path
$env:PYTHONPATH = "$worktrees/surface-round/runs/depth-probe/runtime;$worktrees/surface-round/runs/mlsd-probe/lib"
$runtime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $runtime -B -c 'import runpy,sys; sys.path.append(r"../mask-scalelsd/runs/env/Lib/site-packages"); runpy.run_path("benchmark/floating_extra_holdout.py",run_name="__main__")' baseline
$py = '../mask-scalelsd/runs/env/Scripts/python.exe'
& $py -B benchmark/floating_extra_holdout.py proposer
& $py -B -m benchmark.floating_extra_audit --holdout5
```
