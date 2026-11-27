# Pass the measured back-wall depth to placement

Across 204 images, the strict jamb-and-sill rule emitted 28 recessed backs on 17 images. Each measured depth agreed with the held-out world labels, both relative to its own wall and in the registered camera frame. The ordinary placer independently chose the same plane 25 times. Reusing the three remaining measured depths gained 24 correct positions, lost 10, and removed 13 extras. No extra position or hidden block was added relative to placement on the **same masks**. The other 201 final coordinate sets were identical. All unrelated surfaces kept their ordinary chosen plane level, although rerendering can still change nearby block ownership.

| View with a changed export | Previously missed blocks gained | Previously correct blocks lost | Extras removed |
|---|---:|---:|---:|
| Fixed train-recess-02-right | 9 | 1 | 6 |
| Alternate heldout-facade-left | 7 | 2 | 5 |
| Alternate train-recess-02-left | 8 | 7 | 2 |

The fixed-suite geometry F1 goes from 90.4964% at `c589f1a` to 90.7677% with the strict return masks and 90.7990% with the depth handoff. On the named heldout Facade right, the ordinary placer already chose all three measured back levels. The lower-performing alternate train-recess-02-left still has an end-to-end regression relative to its pre-inset reconstruction. The handoff's eight gained and seven lost cells there make the remaining placement disagreement visible. Native surface area and counts are identical between handoff and no-handoff.

The handoff applies only to a back mask produced by the two-return fit. It checks the image-derived projection fingerprint, supporting host, jamb and sill IDs, wall axis, inward sign, and an integral separation of one to four blocks. The solver uses the measured back plane while running its normal foreground, first-hit, occlusion and photographic checks. The surface mask constrains where it can place blocks. A camera mismatch or lost supporting surface falls back to the existing chooser. The 187 pictures without fitted recesses passed through without any handoff.

**Reproduce.** From the coordinator's `facade-inset-round` checkout, run `python -m benchmark.facade_inset_round --candidate-root ../inset-depth-handoff --candidate benchmark.inset_depth_handoff_trial:refine --geometry --out ../inset-depth-handoff/runs/handoff --workers 5`. From the candidate checkout, run `python -m benchmark.inset_depth_handoff_report` for the direct comparisons and `python -m unittest tests.test_inset_depth_handoff tests.test_inset_return_grid` for sign, camera, missing-support and shape checks. The truth data is read after prediction. [Recorded planes and geometry by image](comparison.json)

The RGB check used `Invoke-WallRecovery.ps1` on `workspace/builds/shape-benchmark-26.3/captures/train-recess-02-right.png` with `-Through placement` and `-Out runs/rgb-handoff-recess`. The launcher’s Python wrapper forwards the ordinary command to `python -m benchmark.inset_depth_rgb`, which inserts the candidate at the existing post-join surface stage. This regenerates all upstream model fields from the sole input screenshot. Run `python -m benchmark.inset_handoff_rgb_check` to verify the exact exported blocks and compare with the identical surfaces and counts after dropping the plane annotations. [RGB input, model, export fingerprints and paired scores](rgb-cli.json)
