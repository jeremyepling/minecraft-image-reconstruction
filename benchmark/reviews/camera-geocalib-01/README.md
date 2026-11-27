# GeoCalib proposals for a metric Minecraft camera

Keep the learned camera on `experiment/camera-geocalib`. Relative to repeat-only, GeoCalib recovers ten more cameras, including three on validation Harbor. A source-image boundary search with the same refinement budget recovers those same three Harbor cameras and five of the seven training cameras. GeoCalib's two unique recoveries are palette variants of one training Shrine view. Neither method recovers an additional held-out Castle camera.

## What the model supplies

The official [GeoCalib pinhole checkpoint](https://github.com/cvg/GeoCalib/releases/tag/v1.0) predicts image focal length and camera-frame up for a single RGB screenshot. The model receives no vertical FOV, pose, map, image neighbors or recorded world data. Its gravity vector and focal length specify a horizon. Two *full-tile* translations in pavement determine metric scale and the horizontal directions. A separate atlas test must identify the same one-block origin in both screen halves. Model inference ran on all 184 expanded screenshots. Routing only invokes the new search on 78 remaining abstentions after the original camera and repeat-only searches. All 104 historical cameras are supported before either fallback.

Both additional searches test seven horizon offsets up to ±8% of the image height and fit up to 12 distinct pixel translation pairs. Learned candidates use agreement with predicted focal length to rank alternatives. The no-model control centers on the source sky/ground boundary and ranks by correlation and horizon displacement. When the boundary is unavailable it searches seven evenly spaced horizons. Both policies preserve earlier supported cameras. The same settings ran on training before validation and test, without per-building tuning. Raw learned parameters, accepted fitted matrices and native visible-corner errors are distinct fields in the per-view records.

Evaluation decodes the final `reconstruction.zip` and checks its coordinates and IDs against the ordinary placement inventory. One geometry-selected integer translation applies to the entire exported build. Every abstaining view still contributes its misses. Fixed and unchanged expanded exports are carried forward by SHA-256. Exact visible cameras on all 118 original failures previously raised v3 facade F1 to 53.26% and decorative to 66.87%. [Recorded-camera replay](../camera-recorded-pose-01/README.md)

## Decoded placement results

| Cohort and policy | Views | Failed cameras | Strict / facade placement F1 | Strict / facade position + ID F1 | Correct / missed / extra positions |
|---|---:|---:|---:|---:|---:|
| Fixed 104, all policies | 104/104 | 0 | 91.27% / 93.42% | 84.68% / 88.51% | 20,773 / 2,929 / 1,046 |
| Visible Builds v3, ordinary | 144/144 | 108 | 8.08% / 8.51% | 6.16% / 6.50% | 9,299 / 204,851 / 6,859 |
| Visible Builds v3, repeat-only | 144/144 | 78 | 18.99% / 19.88% | 12.49% / 13.12% | 23,798 / 190,352 / 12,644 |
| Visible Builds v3, repeat + boundary search | 144/144 | 70 | 21.84% / 22.75% | 14.13% / 14.79% | 28,086 / 186,064 / 14,949 |
| Visible Builds v3, repeat + GeoCalib | 144/144 | 68 | 22.36% / 23.25% | 14.59% / 15.24% | 28,857 / 185,293 / 15,069 |
| Decorative Builds v1, ordinary | 40/40 | 10 | 54.33% / 56.11% | 43.43% / 45.34% | 7,120 / 9,806 / 2,162 |
| Decorative Builds v1, all repeat policies | 40/40 | 0 | 63.46% / 65.23% | 50.51% / 52.56% | 9,240 / 7,686 / 2,957 |

The no-model wider search adds 4,288 correct positions and 2,305 extras over repeat-only. On v3 its strict placement F1 moves from 23.44% to 26.58% on training and 16.01% to 22.77% on Harbor validation; Castle test stays at 5.58%. All eight newly accepted cameras have a per-image P90 visible-corner error at or below 1.07 pixels and absolute vertical FOV error below 0.11�. The largest adverse count is training Amphitheater ashlar rear-oblique, which exports 561 extras alongside 260 correct blocks. Harbor kiln rear-oblique gains 518 correct positions and 55 extras, including 341 exact IDs.

GeoCalib adds 5,059 correct and 2,425 extra positions over repeat-only. Relative to the no-model control it gains six correct positions, three exact IDs and zero net extras on validation. It finds two additional training Shrine cameras. Fixed and decorative exports are identical between the three repeat policies. This branch contains the experimental runners and report. The ordinary runtime and main branch were unchanged by these commits.

## Native-camera diagnostics

On the 78 views offered to the new search, GeoCalib's raw vertical FOV, camera-frame up and center-horizon mean absolute errors are respectively 4.26°, 0.87° and 10.23 pixels on the ten accepted fits; they are 4.73°, 0.85° and 12.69 pixels on the 68 remaining failures. P90 native visible-corner error after floor refinement on Harbor is 0.87 px on kiln high-right, 1.07 px on seafoam high-right and 0.59 px on kiln rear-oblique. The corresponding strict placement F1s are 18.13%, 21.31% and 77.59%. On kiln high-right, 131 correct positions include only four correct material IDs. Improving the camera did not resolve the other placement and material errors on that view.

The upper image shows a recovered Harbor view. The lower image shows a Castle abstention. Orange marks the learned horizon, cyan marks the recorded horizon and, when a metric fit exists, recorded visible corners. Magenta crosses show the fitted projections on those native corners. [Source overlays](source-examples.jpg)

## Reproduction

Measured code revision `867c0cb` on base `d9eb377`. Model source `1dedb29f94fc11b896c5f572005978aeebccd67f`, checkpoint SHA-256 `86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac`, Kornia v0.6.12 `bfb2a8370de15813ddaa4846979c47cd2198f8b3`, PyTorch 2.8.0+cu128. The official model source and Kornia can be checked out under `runs/geocalib/upstream` and `runs/geocalib/kornia`, with `geocalib-pinhole.tar` saved as `runs/geocalib/pinhole.tar`. Inference requires PyTorch, torchvision, Kornia, OpenCV and Matplotlib. From this checkout with the frozen sibling baseline and repeat runs present:

```powershell
$env:PYTHONPATH = 'runs/geocalib/upstream;runs/geocalib/kornia'
../mask-scalelsd/runs/env/Scripts/python.exe -m benchmark.camera_geocalib_infer
$python = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $python -m benchmark.camera_geocalib_run predict train
& $python -m benchmark.camera_geocalib_run predict heldout
& $python -m benchmark.camera_geocalib_run build all
& $python -m benchmark.camera_geocalib_run predict all --matched-control
& $python -m benchmark.camera_geocalib_run build all --matched-control
& $python -m benchmark.camera_geocalib_report
```

[All 288 per-view decisions, proposal errors and decoded pack hashes](cases.json.gz) · [Pooled and split scores](summary.json)
