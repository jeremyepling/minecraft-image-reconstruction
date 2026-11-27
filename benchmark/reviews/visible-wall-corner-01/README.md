# Visible wall corner alignment

On the 13-view pilot, **no eligible alternate exports** survived the frozen corner rule. Keep `experiment/visible-wall-corner-alignment` experimental. The trial found photographed corners but did not identify a reliable one- or two-block correction on these buildings. Main at `e3bd220` supplies the default.

The rule examined 329 vertical masks after ordinary foreground-conflict reconciliation and before visible face candidate generation. All paired normals had to be perpendicular. A measured seam needed at least nine neighboring sample rows, one facing interval per row, at least 80% continuity and each mask ending near an endpoint to suppress T junctions. We projected the candidate planes' vertical intersection into the same photo, measuring the 80th percentile distance from the observed seam samples. One wall had to have a concentrated floor contact or at least two atlas votes and an atlas score of 0.68. The other wall could move up to two integer levels using options already scored for that mask. The rule required residual at most 5 px, improvement at least 5 px and a 2 px margin over the next choice. We froze it before the nine regression and control views.

| Frozen source measurement | Development (4) | Regression and controls (9) | Total |
|---|---:|---:|---:|
| Measured boundaries | 15 | 36 | 51 |
| No anchored wall | 6 | 18 | 24 |
| Two anchored walls | 0 | 8 | 8 |
| Exactly one anchored wall | 9 | 10 | 19 |
| Changed plane levels | 0 | 0 | 0 |

Across all nine development seams with one anchor, the incumbent projected within 1.0–3.7 pixels and the best shift increased the residual. On the other buildings two candidate shifts reduced residual, ending at 78.7 px on Colonnade and 80.6 px on Castle 01. Harbor 03 had 41 vertical masks and no boundary meeting the seam criteria. Of four measured Castle 02 boundaries, three had no anchor and the incumbent best fit the fourth. [Photographed seams](source-seams.jpg)

The intervention ran on the ordinary primary private placement masks and frozen screenshot-derived camera for each frame. The runner stopped after the source gate declined; it retained the exact default pack on all 13 views. The release verifies the ZIP and decoded coordinate-and-ID fingerprints for all 13. Correct positions gained or lost, extra positions added or removed, and changed IDs all totaled zero. The retained pack was the sole eligible choice per view. Placement headroom of rejected plane hypotheses is unmeasured.

| Pilot cohort | Checked / available | Strict placement F1 | Facade placement F1 | Strict position + ID F1 | Facade position + ID F1 | Correct / missed / extra |
|---|---:|---:|---:|---:|---:|---:|
| Fixed historical | 2 / 104 | 87.02% | 92.52% | 85.46% | 90.70% | 446 / 101 / 32 |
| Visible Builds v3 | 9 / 144 | 51.91% | 55.26% | 31.21% | 34.04% | 5,356 / 7,343 / 2,581 |
| Decorative Builds v1 | 2 / 40 | 64.82% | 65.53% | 56.79% | 56.99% | 444 / 324 / 158 |

Visible Builds v3 development scores were 64.40%, 65.16%, 40.02% and 40.43% in the same column order. The five Harbor/Castle regression views scored 44.99%, 49.15%, 26.33% and 30.09%. The fixed cohort's full default trend at `e3bd220` was 91.27% / 93.42% placement and 84.68% / 88.51% with IDs, with 20,773 correct, 2,929 missed and 1,046 extra. Full cohort default scores are carried forward; this trial evaluated 13 selected views. Lighting pairs and the historical 112 were not rerun. [Default score manifest](../camera-native-recovery-controls-01/independent-default-summary.json)

Replay with the local Python runtime and source-field environment from `benchmark/Invoke-ExpandedBaseline.ps1`. These commands load saved RGB-source fields from the full-expanded-controls worktree. They leave Minecraft closed.

```powershell
python -m benchmark.visible_wall_corner_run dev
python -m benchmark.visible_wall_corner_run hold
python -m benchmark.visible_wall_corner_report
```

[Per-frame masks, row measurements, fingerprints and scores](cases.json.gz) · [Pilot summary](summary.json)

