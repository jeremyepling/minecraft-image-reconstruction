# Component removals and coordinate audit

Ten source images were selected using the lowest source SHA-256 in each of the ten fixed dataset partitions. Each ordinary source-driven component was disabled separately. Every following ranked placement refinement still ran, including downstream steps whose ordinary mode is gated on joint placement. The ordinary Material Matcher then assigned each resulting coordinate, and the scorer decoded the material datapack. Five additional full raw-image command runs verified the behavior with the final local build-level corrections included.

## Actual switch removals

| Bypass | Strict placement F1 | Facade placement F1 | Strict position + ID F1 | Facade position + ID F1 | Correct positions gained / lost | Extras added / removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 90.25% | 93.41% | 86.66% | 91.45% | 0 / 0 | 0 / 0 |
| Independent repeat-based depth choice | 76.55% | 79.02% | 73.20% | 77.30% | 18 / 307 | 238 / 47 |
| Joint foreground depth arbitration | 90.00% | 93.12% | 86.41% | 91.16% | 0 / 5 | 4 / 0 |
| Native fully occluded-cell rejection in the primary placer | 89.35% | 92.30% | 85.79% | 90.36% | 0 / 0 | 37 / 0 |
| Atlas-verified edge recovery inside the primary placer | 90.02% | 93.14% | 86.42% | 91.18% | 0 / 6 | 2 / 0 |

Bypassing the independent repeat choice returns the initial baseline depth at that decision, while preserving the generated alternatives, junction choice, joint shortlist and all later checks. Bypassing joint arbitration keeps each independently ranked level while leaving joint-mode downstream recovery enabled. Disabling occlusion rejection bypasses its primary-placer checks before and after adding verified edges. The later neighbor-face guard still runs. Disabling edge recovery skips that individual addition pass; nearby and horizontal refinements still run.

**Keep all four.** Nine of ten view packs changed when the independent repeat choice was skipped. Among these ten, joint arbitration, occlusion rejection and edge recovery affected Loading Dock only. A tie on the other nine views does not establish that these checks are redundant across the suite. [Paired changes and input/pack fingerprints](component-bypass-cases.json) · [Pooled scores](component-bypass-summary.json)

The ten-view table stops at the end of ranked geometry and material assignment. Colonnade and Loading Dock were also rebuilt from their raw source images using `Invoke-WallRecovery.ps1`, with the selected switch applied and every stage through ordinary export enabled. Both control outputs reproduced their previously released canonical typed-pack hashes. Colonnade tied at 128 correct, 63 missed and 10 extra positions when edge recovery was removed. On Loading Dock, removing the edge pass reduced strict geometry and typed F1 from 81.80% to 80.08% (209/47/46 to 203/53/48). Removing the repeat choice reduced both to 69.81% (163/93/48). [Full ordinary pack checks](full-entry-bypasses.json)

## Exact native pixels, corners and coarse sampling

With the recorded camera, the ray through the center of **every one of the 1,666,936 labeled native pixels** intersected the labeled face plane at the correct world block coordinate. All **2,863 distinct visible in-frame corners** projected to the stored location within 0.000000001 pixel. This is a numerical round trip of the deterministic capture-label contract. It checks the conventions used by the benchmark. The supplied coordinates and stored projections come from the same recorded camera.

Repeating that intersection at each native pixel with the **estimated** camera and the same supplied face planes returned 1,651,398 correct cells (99.07%). All 2,863 corners were within one pixel and 2,728 within half a pixel. The largest offset was 0.792 pixels. Thus even providing the correct planes, outlines and world grid does not grant exact pixel registration to an estimated-camera run. [Per-view native ray and corner checks](reference-coordinate-audit.json)

The primary placer samples the downscaled grid at source positions `4k + 2.0` in both axes. Looking up `face_ids[2::4, 2::4]` indexes native pixel centers at `4k + 2.5`. Pairing that label with the primary ray lost 4,040 world-cell agreements out of 104,151 even under the recorded camera. The corrected **oracle** re-rasterizes the recorded world at the exact primary-ray positions. As a cross-check, shifting the raster back to native centers reproduced **every native face ID** on that coarse subset. At primary positions, 4,661 coarse face IDs changed across the ten frames. We left 178 rays that hit a face with zero native source pixels outside the oracle's visible-target set. The [plane choice intervention and scores](README.md#what-if-the-earlier-depth-ranker-had-the-correct-answers) use these corrected labels for all three oracle variants.

As a separate **source-only calibration trial**, we moved only the primary plane ray-generation grid by +0.5 source pixels on both axes. The estimated camera, downstream native texel checks and visibility raster kept their original coordinates. This is a sensitivity test of the primary ray choice. Correcting evaluator labels does not require this runtime change. Pooled strict placement F1 rose from 90.25% to 90.48%, facade placement F1 from 93.41% to 94.33%, strict typed F1 from 86.66% to 86.89%, and facade typed F1 from 91.45% to 92.35%. It recovered seven correct positions, lost 25, added three extras and removed 34. Adjacency 26 alone lost 11 correct positions. Keep the source-only shift experimental. [Per-view shifts and errors](primary-ray-calibration-cases.json) · [Calibration totals](primary-ray-calibration-summary.json)

## Replay

```powershell
python -m benchmark.material_feedback_bypasses --workers 6
python -m benchmark.material_feedback_coordinate_audit --workers 4
python -m benchmark.material_feedback_bypasses --calibration --workers 6
```

To reproduce an ordinary raw-image run on this PC, set `MCIR_BYPASS_PYTHON` to the ordinary approved Python executable, select `MCIR_BYPASS_METHOD` from the names in the JSON report, and pass the small entry shim to the normal wrapper. The five combinations are listed in `benchmark/material_feedback_full_checks.py`. Run the checker after placing each in its specified private output directory.

```powershell
$env:MCIR_BYPASS_PYTHON = (Get-Command python).Source
$env:MCIR_BYPASS_METHOD = 'skip_edges'
.\Invoke-WallRecovery.ps1 -Image $sourceImage -Out runs/full-bypass/loadingdock-skip_edges -Python (Resolve-Path benchmark/material_feedback_entry.cmd)
python -m benchmark.material_feedback_full_checks
```
