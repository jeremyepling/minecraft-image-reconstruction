# Whole-wall recovery on fresh floating-build layouts

Extending predicted grids across their observed wall masks raised strict placement F1 from **6.75% to 27.25%** on 12 original screenshots of six layouts. Retaining existing material IDs and matching newly covered cells raised strict position-plus-ID F1 from **4.31% to 5.63%**. The larger export recovered 1,146 net correct positions and added 1,406 net extras. [Per-frame scores and fingerprints](results.json.gz)

## Primary 12-view comparison

| Source-only arm | Strict placement F1 | Facade placement F1 | Strict position + ID F1 | Facade position + ID F1 | Correct / missed / extra positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary floating connected atlas | 6.75% | 7.39% | 4.31% | 4.76% | 263 / 7,251 / 12 |
| Whole-wall resampling with sprite confidence gate | 27.25% | 28.39% | 2.09% | 2.26% | 1,409 / 6,105 / 1,418 |
| Same placements with retained IDs and masked atlas | 27.25% | 28.39% | 5.63% | 6.04% | 1,409 / 6,105 / 1,418 |

| Frozen source frames | Views | Strict placement F1 | Facade placement F1 | Strict position + ID F1 | Facade position + ID F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dataset validation layouts | 6 | 9.50% → 28.39% | 10.89% → 30.22% | 6.78% → 9.45% | 7.88% → 10.54% |
| Supplemental dataset training layouts | 6 | 4.88% → 26.48% | 5.20% → 27.23% | 2.63% → 3.05% | 2.81% → 3.20% |
| Visible Builds v2 | 4 of 144 | 27.06% → 36.19% | 29.12% → 38.62% | 14.85% → 17.15% | 15.98% → 18.30% |
| Visible Builds v3 | 4 of 144 | 2.62% → 23.66% | 2.79% → 24.46% | 2.17% → 2.64% | 2.38% → 2.77% |
| Decorative Builds v1 | 4 of 40 | 5.21% → 33.10% | 5.86% → 34.66% | 3.51% → 8.19% | 3.95% → 9.06% |

The final arm gained strict geometry on 11 views and lost 2.99 percentage points on Cloister 02 high-right. Strict position plus ID improved on nine views, regressed on Harbor 02 low-left and Factory 02 low-left, and tied on Arcade 02 high-right. Across both Factory 02 views geometry rose from 3.67% to 28.53% while typed F1 moved from 3.10% to 2.99%. Exact IDs across the first 12 frames increased from 168 to 291, while accuracy at recovered positions moved from 63.88% to 20.65%. The final arm assigned 2,767 of 2,827 exported blocks and used 60 fallbacks. The ordinary arm assigned 209 of 275 and used 66 fallbacks.

Each layout contributed a low-left and high-right view from one palette. The dataset validation layouts are Cloister 02, Harbor 02 and Hillside Chapel 2. Arcade 02, Factory 02 and Market Court 2 are a separate check from the dataset training split. Both groups were scored with the frozen runner at `f389a08` and were held out from tuning this whole-wall rule. Harbor 02 and Hillside Chapel 2 share families with Harbor 01 and Hillside Chapel 1 in the historical six sky captures. Their layouts and original screenshots differ. The old 38 test views and historical fixed 104 were outside this replay. [Frozen frames and input checksums](frames.json)

## Additional six-view transfer check

Watchpost 02, Observatory 02 and Riverside Mill 2 contributed one low-left and one high-right source view each. These three dataset training families are disjoint from the primary 12, the old 38 and the six sky captures. The six were frozen as an independent comparison for choices made using the primary 12.

| Source-only arm | Strict placement F1 | Facade placement F1 | Strict position + ID F1 | Facade position + ID F1 | Correct / missed / extra positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary floating connected atlas | 14.22% | 15.70% | 3.94% | 4.35% | 249 / 3,004 / 0 |
| Whole-wall resampling with sprite confidence gate | 39.92% | 43.08% | 4.26% | 4.60% | 900 / 2,353 / 356 |
| Same placements with retained IDs and masked atlas | 39.92% | 43.08% | 9.27% | 10.00% | 900 / 2,353 / 356 |

The final arm improved strict placement and strict position plus ID on all six views. It gained 663 correct positions, lost 12 and added 356 extras. Exact IDs rose from 69 to 209. Every arm had resolved camera yaw in all six views. [Frozen additional frames](supplemental-frames.json) · [Decoded score audit](supplemental-results.json.gz)

## Prediction and scoring

Each prediction receives the one original RGB image, the pinned RGB-derived normal and depth fields, and the Java 26.3 atlas. Visible Builds v2 fields were computed from RGB. The other cohorts reused the verified fields in the expanded control cache. The ordinary crop, count, connected atlas and assembly stages ran in their published order. The published whole-wall resampling and material arms ran with unchanged settings. Reference masks, world blocks and camera data were reserved for scoring. A clean Chapel replay reproduced all three chart hashes, decoded block inventories and grades. The export archive hashes include ZIP timestamps and differed between the two runs.

Each headline uses one XYZ translation per whole exported build after upright compass registration. In the resampled Cloister views the ordinary camera-anchor choice changed and the archived evaluator searched upright yaw. Every shared oriented chart with identical homography and source normal gave the same change of basis back to the ordinary compass. The low-left turn was +1 from three shared charts and the high-right turn +3 from five. Those independently derived rotations matched the fallback grades exactly. The 10 views with directly resolved yaw in all arms also improved from 3.99% to 25.31% strict geometry and 2.31% to 3.75% strict position plus ID.

Keep `experiment/floating-fresh-validation` experimental. The transfer checks recover more photographed structure, with a high rate of unsupported placements and incorrect IDs on the added blocks. The experiment branch carries the offline evidence and replay runner.

## Replay on this PC

Use the bundled Python with the installed ONNX runtime and OpenCV. Run from this checkout. Generated handoffs and decoded exports go in `runs/floating-fresh/`.

```powershell
$worktrees = (Resolve-Path ..).Path
$env:PYTHONPATH = "$worktrees/surface-round/runs/depth-probe/runtime;$worktrees/surface-round/runs/mlsd-probe/lib"
$runtime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$entry = 'import sys; sys.path.append(r"../mask-scalelsd/runs/env/Lib/site-packages"); from benchmark.floating_fresh_validation import main; main()'
& $runtime -B -c $entry --workers 6
& $runtime -B -c $entry --supplemental --workers 6
```

Add `--score-only` for a decoded replay of existing outputs or `--manifest-only` to resolve verified source paths without starting reconstruction.
