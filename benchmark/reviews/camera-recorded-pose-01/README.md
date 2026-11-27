# Placement with the recorded camera

This Java 26.3 diagnostic measures how much the ordinary pipeline can recover when it receives the recorded projection on views where its own camera rejected the screenshot. It uses the production code from `1ca8822` and the frozen [expanded baseline](../expanded-baseline-01/README.md). The 104 fixed views and 66 already-supported expanded views keep their original exports, verified by ZIP hash. The replay covers the 108 rejected Visible Builds v3 views and 10 rejected Decorative Builds v1 views.

The replay supplies a 3 × 4 projection through the ordinary metric-camera provider. Local Y = 0 is the top of the staging floor and the integer X/Z origin is the block below the recorded eye. Inset detection, subsequent outline and surface refinements, counting, placement, and material export run through the ordinary entry point. The same RGB, archived normal and depth fields, line detections, recovery masks and planarity evidence feed the build. Prompt equality is checked before archived recovery masks are reused. The replay checks the final placement projection and the baseline atlas and model digests. Reference block locations and identities enter the grader after export.

On all 118 rejected views, **245,460** native visible vertices reproject within a maximum error of **1.12 × 10⁻¹⁰ px**. Three no-op replays through the same hook reproduce the decoded placement and block IDs, every ZIP member byte, and the full placement status. Ordinary ZIP entry timestamps vary by run.

## Scoring results

All 118 recorded projections reached placement and produced nonempty exports. The ordinary baseline and recorded-camera arm are paired on every view.

| Cohort | Views | Usable cameras | Strict position F1 | Facade position F1 | Strict position + ID F1 | Facade position + ID F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 104 | 104 | 104 → 104 | 91.27% → 91.27% | 93.42% → 93.42% | 84.68% → 84.68% | 88.51% → 88.51% |
| Visible Builds v3 | 144 | 36 → 144 | 8.08% → 51.72% | 8.51% → 53.26% | 6.16% → 29.06% | 6.50% → 30.08% |
| Decorative Builds v1 | 40 | 30 → 40 | 54.33% → 64.96% | 56.11% → 66.87% | 43.43% → 51.98% | 45.34% → 54.09% |

The fixed 104 carry forward 20,773 correct, 2,929 missed, and 1,046 extra positions under strict scoring. Facade-priority scoring counts 18,952 correct, 1,624 missed, and 1,046 extra. Both F1 deltas are 0.00 percentage points.

The following counts include only views where the ordinary camera rejected the image. The baseline exported zero blocks on those views.

| Cohort | Replayed and nonempty | Strict correct | Missed | Extra | Strict position F1 | Facade position F1 | Strict position + ID F1 | Facade position + ID F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Visible Builds v3 | 108 | 80,862 | 106,794 | 37,473 | 52.85% | 54.34% | 28.48% | 29.43% |
| Decorative Builds v1 | 10 | 2,394 | 1,832 | 689 | 65.51% | 66.47% | 52.59% | 53.93% |
| Castle test subset | 18 | 26,269 | 26,133 | 7,963 | 60.64% | 60.66% | 29.03% | 29.05% |

Across all 118 repaired views, the exported packs recover 83,256 correct positions, miss 108,626, and place 38,162 extras. Strict geometry F1 is 53.15% and facade F1 is 54.62%. Strict position-plus-ID F1 is 29.04% and facade position-plus-ID F1 is 30.00%.

### Expanded splits

| Cohort | Split | Views | Usable cameras | Strict position F1 | Facade position F1 | Strict position + ID F1 | Facade position + ID F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visible Builds v3 | train | 108 | 26 → 108 | 9.78% → 49.97% | 10.30% → 51.41% | 7.77% → 29.90% | 8.21% → 30.92% |
| Visible Builds v3 | validation | 18 | 10 → 18 | 16.01% → 36.76% | 19.95% → 42.24% | 9.61% → 20.81% | 11.99% → 24.85% |
| Visible Builds v3 | test | 18 | 0 → 18 | 0.00% → 60.64% | 0.00% → 60.66% | 0.00% → 29.03% | 0.00% → 29.05% |
| Decorative Builds v1 | train | 24 | 18 → 24 | 55.45% → 63.70% | 57.42% → 65.63% | 45.05% → 50.71% | 47.14% → 52.82% |
| Decorative Builds v1 | validation | 8 | 4 → 8 | 46.02% → 76.70% | 47.57% → 78.33% | 41.10% → 67.48% | 42.51% → 69.17% |
| Decorative Builds v1 | test | 8 | 8 → 8 | 57.46% → 57.46% | 59.10% → 59.10% | 40.95% → 40.95% | 42.80% → 42.80% |

## What the export errors look like

[Castle, Pillars and Market Court examples](examples.jpg) show the same native screenshot and aligned export errors. Castle 1 limestone low-left recovered 929 correct positions, missed 463, and placed 113 extras. Its strict geometry F1 is 76.34% and its strict position-plus-ID F1 is 46.75%. The orange patches flag correct locations with different material IDs. Red highlights extras visible in the reprojection, while the count includes every extra in the datapack.

All 393 native pixels of the missed Castle block at `(16128, 90, 18)` are inside a detected wall mask. The Block Counter returned no dimensions for that mask. The depth selector put its south-facing plane one block behind the recorded plane, and the export contains a block at `(16128, 90, 17)`. The recorded camera exposes an error that remains after successful surface coverage.

## Reproduction

From the experiment checkout, use the archived `full-expanded-controls/runs/expanded-baseline-01` builds, the frozen expanded captures under the stable checkout's `runs` folder and the prepared Java 26.3 atlas.

```powershell
python -m benchmark.camera_recorded_replay --workers 5 --no-op train-pillars-01-marble-low-left train-pillars-01-marble-high-right train-market-court-1-warm-high-right
python -m benchmark.camera_recorded_examples --out benchmark/reviews/camera-recorded-pose-01/examples.jpg
```

The [summary](summary.json) reports pooled relative placement and position-plus-ID counts, dataset splits, and the Castle family. The [paired cases](cases.json.gz) include each export and reference fingerprint, scoring alignment, and the retained-pack status. The primary scorer selects one whole-build integer translation by strict visible geometry and applies it to all cells and material IDs. Facade-priority F1 treats small horizontal-only targets as optional and charges every extra. The registered score is also recorded for comparison.

The earlier floor-chart diagnostic supplied the true ground chart to the atlas matcher, which accepted 80 of the 118 rejected views. This replay supplies the recorded projection and measures actual exported blocks. Keep the branch experimental because the replay uses recorded per-view poses. The branch contains the benchmark runner and evidence.
