# Floor-free wall depth verification

Keep this experimental. First-hit reprojection kept every incumbent export, including with the recorded projection. The perfect chooser recovered 37 net correct positions across 18 conditions. Neither depth model recovered that gain on the held-out buildings. No reconstruction behavior changed on Floating integration or production `main`.

## The pink split and this test change different handoffs

The guarded pink depth split uses source-image depth to divide a Surface Finder mask that appears to cover multiple planes. It runs before separate chart fitting and assembly. Its output changes which pixels and cells belong to each chart. This experiment starts with the saved `source-refit-with-small-steps` exports. It freezes those patch boundaries, grids, block IDs and surrounding chart transforms. It changes only the signed integer offset of one existing patch and uses first-hit projection to check the result in the source image. Both work without a ground reference. A missed split, an incorrect grid or a disconnected chart remains a problem after the depth check.

`experiment/floating-local-reprojection` started from Floating integration `9435934c63c126a629469a307673f78c0c9fb117`. Its expanded patch inputs come from `experiment/floating-depth-refit` at `47d6e7c1c6b8d6eba3d20b9990c7ca3f498c3dfa`. The newer expanded split-border research benchmark was pinned separately at `51bd1e9`. That benchmark measures a different intervention and cannot supply the same saved signed patch handoffs.

## First, export the options

The 18 saved Java 26.3 conditions have 21 eligible saved patches. For each image the candidate set contains the incumbent and an individual move by one block in either signed normal direction for each eligible patch. Patches do not move in combination. Each of 60 whole-build options was assembled, exported to a Java datapack and decoded through the same path. All 18 newly exported no-op controls matched the archived local and registered coordinates and IDs. All option assemblies preserved the selected chart origins and joins. Scoring uses a single whole-build integer translation for geometry and IDs, with the usual Floating yaw registration. Paired changes use the separately registered coordinates of the selected exports. The perfect chooser maximizes strict placement F1 and keeps the incumbent on ties.

| Development source image | Best offered option | Correct gained / lost | Extras added / removed | Exact IDs gained / lost |
| --- | --- | ---: | ---: | ---: |
| Castle 01 ashlar high-right, original | 87-cell patch from −1 to 0 | 12 / 0 | 75 / 87 | 5 / 0 |
| Riverside Mill 1 warm low-left | Incumbent | 0 / 0 | 0 / 0 | 0 / 0 |
| Viaduct 01 riverstone low-left | Incumbent | 0 / 0 | 0 / 0 | 0 / 0 |

On the nine textured transfer conditions, Stepped Hall improves by flattening one patch (12 gains and seven losses), while Terrace improves by moving one patch from +1 to +2 (26 gains and six losses). Stepped Hall loses four net exact IDs even with its higher strict placement score. Stable Yard's high-right saved candidate offers six cells on an unselected chart. The separate 21-cell eave is absent from the options. The low-left incumbent also wins. None of the six fixed sky images has a saved eligible patch. These are no-op regression cases.

Green marks a newly correct visible position. Orange marks a lost one. The marks follow recorded frontmost face pixels and show the reference chooser's result, including any change in the one whole-build registration. [Zoomable comparison](placement-deltas.png)

## Then choose from the same exports using the source image

The Depth Anything and MoGe arms independently detrend inverse depth against nearby stable parent-wall cells. MoGe uses the point head's Z values and validity mask from independent inference. The two fields vote on whether the patch is at the same relative depth, nearer or farther. They cannot distinguish a one-block step from two blocks of the same sign, so the incumbent wins a tie.

The first-hit arm fits a 3×4 projection to source-grid corners on two nonparallel faces in the selected local group, anchored on one visible grid corner. It excludes moving UVs and their one-cell fringe. Held-out grid corners on those faces check fit consistency. Native grayscale gradients at the projected rim and internal block seams supply a separate source-image check. The pose stays fixed for all options. The rasterizer from `engine/visible_placement.py` measures which voxel and axis the ray reaches first. A candidate must show adequate source-mask support for every moved block, avoid collisions, preserve neighboring visible voxel faces and avoid adding coverage in source support gaps. It needs independent photographed rim or seam agreement as well as improved mask overlap. The combined arm adds the two depth votes without relaxing these checks.

The recorded-projection diagnostic uses the same first-hit scorer. It rebases the recorded projection with the archived incumbent's reference-derived yaw and whole-build translation. A fixed but imperfect incumbent registration can still misalign a local surface. No source-only choice uses the recorded camera, positions or labels. The source rule was frozen on development before the family holdout and checked again without adjustment after the fresh no-op exports.

### Decoded results

| Cohort and selector | Strict / facade placement F1 | Strict / facade position + material F1 | Correct / missed / extra | Correct gained / lost | Extras added / removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development, incumbent, Depth Anything, reprojection, combined or recorded-camera diagnostic | 49.314% / 49.465% | 19.871% / 19.944% | 881 / 1,308 / 503 | 0 / 0 | 0 / 0 |
| Development, MoGe | 49.426% / 49.577% | 19.871% / 19.944% | 883 / 1,306 / 501 | 2 / 0 | 0 / 2 |
| Development, perfect chooser | 49.986% / 50.141% | 20.151% / 20.225% | 893 / 1,296 / 491 | 12 / 0 | 75 / 87 |
| Held out, incumbent, reprojection, combined or recorded-camera diagnostic | 17.717% / 18.142% | 6.265% / 6.326% | 509 / 1,996 / 2,732 | 0 / 0 | 0 / 0 |
| Held out, Depth Anything | 17.682% / 18.205% | 5.813% / 5.850% | 508 / 1,997 / 2,733 | 2 / 3 | 41 / 40 |
| Held out, MoGe | 17.543% / 18.060% | 5.674% / 5.705% | 504 / 2,001 / 2,737 | 2 / 7 | 45 / 40 |
| Held out, perfect chooser | 18.587% / 19.150% | 6.544% / 6.613% | 534 / 1,971 / 2,707 | 38 / 13 | 11 / 36 |

All three development projections passed the two-face fit check. Eight of nine held-out conditions, including both perfect-chooser wins, lack a second selected plane and fail that fit. On Castle, moving the 87-cell patch back to the parent plane increases overlap with its source-grid footprint from 69% to 89%, but the fitted projection finds source-mask pixels on only 82 of the 87 moved cells. The recorded projection finds pixels on 78. The whole patch fails the visible support gate with either projection. On Viaduct and Mill there was no offered gain for a stronger verifier to recover. The tested verifier did not demonstrate a correct source-only move, so the separate conditional Stable Yard attachment test was not run.

The transfer group spans Arcbridge, Market Court, Sawtooth, Stable Yard, Stepped Hall, Terrace and Twin Spire. The 18 conditions contain six Visible Builds v2, six Visible Builds v3 and six Decorative Builds v1 views. This was a targeted cohort, not a replay of full v3 or full decorative. Those full scores were not recomputed.

## Replay

Use the registered `floating-depth-refit` checkout at the pinned revision with its saved Java 26.3 sources, chart/patch artifacts, depth fields and native scoring labels. The model is the local checksum-verified MoGe ONNX in the `normal-round` checkout. The runner uses the saved local ONNX Python 3.12 environment for that model and the regular benchmark environment for assembly. All runs were offline, with Minecraft closed. The runner writes only into `runs/local-reprojection-options` in the experiment worktree, with separate per-frame and per-option paths.

```powershell
python benchmark/floating_local_reprojection.py --stage options --workers 4
python benchmark/floating_local_reprojection.py --stage depth --group development
python benchmark/floating_local_reprojection.py --stage projection --group development
python benchmark/floating_local_reprojection.py --stage depth --group held-out
python benchmark/floating_local_reprojection.py --stage projection --group held-out
python benchmark/floating_local_reprojection.py --stage depth --group six-sky
python benchmark/floating_local_reprojection.py --stage projection --group six-sky
python -m benchmark.floating_local_reprojection_report
```

The archived conditions identify source and reference hashes, patch UVs, integer offset choices, decoded export fingerprints, fit failures, and complete paired scores. [Frozen conditions and receipts](conditions.json.gz) · [All pooled counts](summary.json)

## Floating branch scorecard

| Revision and cohort | Strict / facade placement F1 | Strict / facade position + material F1 | Correct / missed / extra |
| --- | ---: | ---: | ---: |
| Sparse connected-atlas integration `9435934`, six fixed sky views, carried forward | 12.642% / 12.899% | 4.677% / 4.772% | 273 / 3,768 / 5 |
| Archived depth-patch incumbent and all local-reprojection arms, same six | 27.499% / 27.734% | 5.963% / 5.986% | 890 / 3,151 / 1,542 |
| Expanded split-border research benchmark pinned at `51bd1e9`, same six | 27.588% / 27.826% | 5.983% / 6.005% | 890 / 3,151 / 1,521 |
| Expanded split-border, all 84, carried forward | 33.987% / 34.479% | 5.655% / 5.757% | 18,318 / 61,192 / 9,965 |

The six fixed sky scores tied across every local-reprojection arm. The split-border pin landed as benchmark evidence on Floating integration. Local reprojection landed only on its experiment branch. The separate fixed-104 production trend carries forward at **91.27% strict / 93.42% facade** (20,773 correct / 2,929 missed / 1,046 extra).
