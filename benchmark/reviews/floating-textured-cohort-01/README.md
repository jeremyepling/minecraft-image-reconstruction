# Active floating layouts with visible texture

Use the [versioned cohort manifest](../../../training/floating-active-cohort-v1.json) to select active train, validation, and test captures for floating reconstruction. It excludes every angle and palette of a layout when one of its palettes has at least 80% concrete or unglazed terracotta among its visible building pixels. Pool the frontmost label pixels across the recorded camera angles for that palette. Sky, the staging floor, and occluded faces contribute zero pixels. Glazed terracotta and concrete powder have distinct textures and remain eligible. The review also explicitly excludes Amphitheater 01 and Amphitheater 02. The published datasets, original splits, and production benchmark are preserved.

| Dataset | Active train | Active validation | Active test | Active total |
|---|---:|---:|---:|---:|
| Visible Builds v2 | 68 / 72 | 12 / 12 | 12 / 12 | 92 / 96 |
| Visible Builds v3 | 72 / 108 | 18 / 18 | 18 / 18 | 108 / 144 |
| Decorative Builds v1 | 24 / 24 | 8 / 8 | 8 / 8 | 40 / 40 |
| Total | 164 / 204 | 38 / 38 | 38 / 38 | 240 / 280 |

| Excluded layout | Palette with highest smooth coverage | Visible share | Removed captures |
|---|---|---:|---:|
| Watchpost 3, v2 | Cool | 82.97% | 4 |
| Amphitheater 01, v3 | Orchid | 76.99% | 6 |
| Amphitheater 02, v3 | Orchid | 82.70% | 6 |
| Amphitheater 03, v3 | Orchid | 84.45% | 6 |
| Factory 02, v3 | Chalk | 81.30% | 6 |
| Factory 03, v3 | Chalk | 81.34% | 6 |
| Observatory 03, v3 | Verdigris | 82.99% | 6 |

Observatory 02, whose reviewed fallback established a plausible block scale, stays active with 76.64% coverage in its highest palette. The four historical Factory 02 and 03 test conditions use the kiln palette and are excluded with their full layouts.

## Select and verify captures

Pass case records carrying `dataset` and `id` to `list(select_active_cases(cases))` from `training.floating_cohort`. The selector preserves case order and validates the release identity and any provided source capture metadata. Use `frame_index()` to read the active flags, original splits, and input hashes for training. For a composited image condition, keep its own source-image hash as well as the release `rgb_sha256`.

Regenerate the selection from the original release ZIPs and their manifests. Future release versions require an explicit manifest hash and review of their split and palette coverage. These commands run from the experimental worktree on this PC. Use the benchmark Python environment for scoring.

```powershell
python -m training.floating_cohort --archives-root ../../minecraft-image-reconstruction/runs
../mask-scalelsd/runs/env/Scripts/python.exe -m benchmark.floating_cohort_scores --out benchmark/reviews/floating-textured-cohort-01/scores.json.gz
```

The [verified scores](scores.json.gz) bind the active manifest and the archived evaluation data from `e97dd44`. Across 92 image conditions from 90 release captures, all four archived arms were decoded and rescored from the saved yaw and shared translation against the original visible-only truth. All receipt counts replayed exactly. The cohort policy retains 84 conditions from 82 captures. Measured conditions cover 30 of 92 active v2 captures, 29 of 108 active v3 captures, and 23 of 40 active Decorative captures. Repeat image conditions explain the difference between unique captures and condition counts. This is a paired retrospective comparison on previously reviewed cases.

| Method on the 84 active conditions | Strict F1 | Facade F1 | Position + ID F1 | Facade + ID F1 | Correct / missed / extra |
|---|---:|---:|---:|---:|---:|
| Connected-atlas panel replay | 5.29% | 5.44% | 2.20% | 2.27% | 2,164 / 77,346 / 124 |
| Whole-wall challenger | 32.26% | 32.69% | 5.19% | 5.28% | 18,411 / 61,099 / 16,203 |
| Unrestricted full-face ruler | 33.08% | 33.54% | 5.40% | 5.49% | 17,934 / 61,576 / 10,988 |
| Large-contraction rule | 33.54% | 34.01% | 5.51% | 5.61% | 18,323 / 61,187 / 11,420 |

Compared on those same conditions, the large-contraction rule gains 1.28 strict placement points over the whole-wall challenger, gives up 88 correct blocks, and removes 4,783 extras. Compared with unrestricted full-face replacement, it gains 389 correct blocks and adds 432 extras.

## Floating branch scorecard

Integration revision `7033891` carries forward the connected-atlas default on all six eligible fixed sky composites. The six exports were decoded again for this report. The candidate rows use the same six source images and frozen scoring rule. Production fixed 104 carries forward separately at 91.27% strict and 93.42% facade with 20,773 correct, 2,929 missed, and 1,046 extras.

| Method on the six sky conditions | Strict F1 | Facade F1 | Position + ID F1 | Facade + ID F1 | Correct / missed / extra |
|---|---:|---:|---:|---:|---:|
| Integration connected atlas | 12.64% | 12.90% | 4.68% | 4.77% | 273 / 3,768 / 5 |
| Whole-wall challenger | 22.93% | 23.07% | 4.40% | 4.40% | 901 / 3,140 / 2,915 |
| Unrestricted full-face ruler | 27.45% | 27.68% | 5.97% | 5.99% | 888 / 3,153 / 1,542 |
| Large-contraction rule | 27.50% | 27.73% | 5.96% | 5.99% | 890 / 3,151 / 1,542 |

The contraction rule places 617 more correct blocks and 1,537 more extras than integration on the six sky views. No reconstruction behavior was promoted by the cohort experiment. Land the cohort policy on floating integration and keep the branch separate from production `main`.
