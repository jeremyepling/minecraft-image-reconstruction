# Horizon and floor-repeat camera experiment

The ordinary camera gets the first decision on each source image. The experiment runs on its 118 abstentions. The free and boundary arms use scalar-frequency shortlists and independently checked 2-D phase gradients. The free arm sweeps possible horizons. The boundary arm limits the search to horizons near a long transition at both image borders.

The source arm measures the repeated 16-texel motif directly. It rectifies the lower image at trial horizons, finds two independent brightness translations, and refines both across separated patches. A border transition adds hypotheses to a free grid. Every acceptance requires the ordinary same-sprite, orientation and origin check across both halves, and a metric camera lift. The repeat rule was frozen after training-source probes on Factory 01, Market Court 1, and Viaduct 01. Each newly supported camera was sent through the ordinary whole build and its exported ZIP was decoded. Recovered cameras were scored against the recorded pose and visible corners only after prediction. The test uses vanilla Java 26.3 concrete staging floors.

| Cohort | Arm | Camera supported | Strict geometry F1 | Facade geometry F1 | Strict position + ID F1 | Facade position + ID F1 | Correct / missed / extra positions |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed-104 | baseline | 104/104 | 0.9127 | 0.9342 | 0.8468 | 0.8851 | 20773 / 2929 / 1046 |
| fixed-104 | free | 104/104 | 0.9127 | 0.9342 | 0.8468 | 0.8851 | 20773 / 2929 / 1046 |
| fixed-104 | boundary | 104/104 | 0.9127 | 0.9342 | 0.8468 | 0.8851 | 20773 / 2929 / 1046 |
| fixed-104 | source | 104/104 | 0.9127 | 0.9342 | 0.8468 | 0.8851 | 20773 / 2929 / 1046 |
| fixed-104 | repeat_no_boundary | 104/104 | 0.9127 | 0.9342 | 0.8468 | 0.8851 | 20773 / 2929 / 1046 |
| visible-builds-v3 | baseline | 36/144 | 0.0808 | 0.0851 | 0.0616 | 0.0650 | 9299 / 204851 / 6859 |
| visible-builds-v3 | free | 42/144 | 0.0925 | 0.0970 | 0.0667 | 0.0701 | 10777 / 203373 / 8192 |
| visible-builds-v3 | boundary | 43/144 | 0.0943 | 0.0991 | 0.0682 | 0.0718 | 11008 / 203142 / 8407 |
| visible-builds-v3 | source | 66/144 | 0.1899 | 0.1988 | 0.1249 | 0.1312 | 23798 / 190352 / 12644 |
| visible-builds-v3 | repeat_no_boundary | 66/144 | 0.1900 | 0.1990 | 0.1249 | 0.1312 | 23818 / 190332 / 12694 |
| decorative-builds-v1 | baseline | 30/40 | 0.5433 | 0.5611 | 0.4343 | 0.4534 | 7120 / 9806 / 2162 |
| decorative-builds-v1 | free | 36/40 | 0.5889 | 0.6063 | 0.4535 | 0.4727 | 8166 / 8760 / 2641 |
| decorative-builds-v1 | boundary | 36/40 | 0.5887 | 0.6060 | 0.4526 | 0.4719 | 8158 / 8768 / 2631 |
| decorative-builds-v1 | source | 40/40 | 0.6346 | 0.6523 | 0.5051 | 0.5256 | 9240 / 7686 / 2957 |
| decorative-builds-v1 | repeat_no_boundary | 40/40 | 0.6335 | 0.6510 | 0.5033 | 0.5237 | 9231 / 7695 / 2987 |

## Split changes for the repeat-pattern fallback

| Cohort | Split | Cameras recovered | Correct placements gained | Extra placements added | Strict F1 before → after | Strict position + ID F1 before → after |
|---|---|---:|---:|---:|---:|---:|
| visible-builds-v3 | test | 2 | 1515 | 391 | 0.0000 → 0.0558 | 0.0000 → 0.0309 |
| visible-builds-v3 | train | 28 | 12984 | 5394 | 0.0978 → 0.2344 | 0.0777 → 0.1569 |
| visible-builds-v3 | validation | 0 | 0 | 0 | 0.1601 → 0.1601 | 0.0961 → 0.0961 |
| decorative-builds-v1 | test | 0 | 0 | 0 | 0.5746 → 0.5746 | 0.4095 → 0.4095 |
| decorative-builds-v1 | train | 6 | 1054 | 511 | 0.5545 → 0.6264 | 0.4505 → 0.4970 |
| decorative-builds-v1 | validation | 4 | 1066 | 284 | 0.4602 → 0.7232 | 0.4110 → 0.6316 |

## What the border measurement contributed

The free grid and the grid with border hypotheses supported the same 40 views. 8 had identical camera matrices. The largest difference in fitted horizon was 0.019 px. Their decoded, translated block positions and IDs were identical in 18 of 40 recovered builds. See the repeat_no_boundary arm for the complete placement effect.

The recorded-camera atlas diagnostic had passed on 80 of the 118 failed views. The repeat-pattern arm recovered 40 of those. In 40 of 40 remaining views, at least one metric chart reached bilateral registration but every candidate failed it. Of 38 views where the true chart failed registration, 38 stayed abstentions.

The two held-out Castle 01 views are sensitive to the evaluation origin near the horizon. Moving the evaluation-only integer ground registration by at most one block per horizontal axis reduced the 90th percentile corner error from 20.72 to 0.85 px on limestone and from 11.46 to 0.29 px on ashlar. All tables use the ordinary registration and placement scoring.

Keep this branch as an experiment. The repeat search recovered placements on held-out buildings in both expanded cohorts, which supports integrating it as an abstention-only fallback. The benchmark injects accepted cameras at the production handoff and does not install a default camera policy. The default branch has received no change from this experiment.

All original source-supported build exports were carried forward and verified by SHA-256. The fixed 104 received no fallback attempts. Every abstention stays in the overall denominator. See summary.json for frozen splits, matched-position material accuracy, and conditional camera errors. cases.json.gz contains every camera and decision. The example overlays estimated and recorded visible corners on the original input.

Run these commands from this checkout with the frozen cohorts in the stable runs directory and the full-expanded-controls worktree beside it.

```powershell
python -m benchmark.camera_horizon_run predict
python -m benchmark.camera_horizon_run build
python -m benchmark.camera_horizon_run repeat-predict
python -m benchmark.camera_horizon_run repeat-build
python -m benchmark.camera_horizon_run grid-predict
python -m benchmark.camera_horizon_run grid-build
python -m benchmark.camera_horizon_report
```
