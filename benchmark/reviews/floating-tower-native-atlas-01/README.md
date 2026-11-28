# Native atlas wall fallback

## Decision for the next round

Retain default Floating at measured revision `4bcc191`. Keep the native atlas wall fallback experimental. The six-view candidate recovered 11 visible positions with no geometry losses or new geometric extras, but none had the correct material ID. The supplied tower remained empty. No behavior was promoted.

## Tower screenshot

Both runs used the original tower screenshot, SHA-256 `787c1407910301b91d0fcc6b053650b4ca944d6b4de32779b438f5da90201731`. Neither arm received hand-marked windows, scale, or other geometry. Regions First retained 66 masks, of which 40 passed the wall-facing filter. The Block Counter formed 55 sampling charts and returned zero axis counts. The 110 lines supplied to orientation did not yield a three-axis fit. Without that fit the native sheet fallback received no wall candidates. The planar path also verified no grids. The two Java 26.3 packs were byte-identical and decoded to zero placements. The supplied image has no reference world, so accuracy was unmeasured.

The earlier hand-seeded grid suggests a follow-up that proposes a local facade orientation independently of the three-axis fit. Verify an independent local grid on visible seams before joining any other face.

## Reproduction

The only candidate change was enabling the existing `--native-fallback` option in `benchmark.floating_build_local_run`. Both arms consumed the same source and handoffs. The run used the pinned Floating Front 100 full-block noon release at `f3e416acdbd6e0fa171260997fcdf62a86bc0c42c001d3a16d13e3f0e3c01187` and atlas at `49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950`. The six held-out IDs were frozen before inspecting candidate outputs.

- `front-013-windmill-full-noon` and `front-014-vault-full-noon`
- `front-055-windmill-full-noon` and `front-056-vault-full-noon`
- `front-097-windmill-full-noon` and `front-098-vault-full-noon`

Decoded packs were scored against visible references using one whole-build translation. Three views gained visible blocks, with six correct positions on Vault 014, two on Windmill 055, and three on Vault 098. No previously exported coordinate or ID changed. Both arms kept 215 of 217 masks and counted both axes on five masks. Usable orientations for scoring increased from one to four views and empty exports fell from five to two. There was no matched Perfect input for this pair. Decorative views were unmeasured.

## Floating branch scorecard

Measured integration revision `4bcc191`, six of 100 full-block noon views. Counts are correct / missed / extra.

| Method | Strict / facade placement F1 | Strict / facade position + material F1 | Strict placement counts | Facade placement counts |
|---|---:|---:|---:|---:|
| Baseline — default Floating | 0.98% / 1.01% | 0.00% / 0.00% | 8 / 1,610 / 1 | 8 / 1,560 / 1 |
| Candidate 1 — native atlas wall fallback | 2.32% / 2.39% | 0.00% / 0.00% | 19 / 1,599 / 1 | 19 / 1,549 / 1 |

Paired geometry gained 11 correct positions, lost zero correct positions, and added and removed zero extras. Position-plus-material extras increased from nine to 20 because the new blocks had incorrect IDs. The fixed-104 production trend was outside this comparison.
