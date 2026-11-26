# Canopy placement confirmation fixtures

The previous native screen identified 20 camera-supported occupancy shapes independent of Visible Builds v1 (152 screenshots) and the native camera suite (70 screenshots). The first transfer round used 12 of these shapes. Before scoring placement in this round, we reserved all eight remaining shapes and picked the supported image with the lowest RGB SHA256 for each. Three are canopies, one a colonnade, two corners and two facades. The earlier transfer12 is a regression set for this experiment.

The source screen SHA256 is `691708e501e0647365f15793d78f5001b59a9b35815b0052ad6586d0eba5a866`. It checks shape identity under translation, reflection and horizontal rotation. The eight selected images and frozen camera estimates have manifest SHA256 `6cc3d9ff8e00167efb78ae8941358c67fad9d5791d5379cda36dbd3feb0d5898`. The model receives only RGB and the ordinary camera, mask and count outputs from the same image. The reference readback and pose are scoring inputs.

The geometry default at `a16668f` placed 1,014 of 1,431 visible blocks with 482 extras, including 21 on hidden structure. That gives F1 of 0.6929 on these eight camera-supported images. Its three canopies had 74 correct of 167 visible blocks and 261 extras. The project used the 164 previously released normal-placement packs and the preceding 12 transfer packs as controls. Their saved packs and decoded coordinates matched their published archives byte for byte. [Previous baseline](../visible-depth-01/README.md) · [Selection and source checks](../visible-depth-transfer-01/README.md) · [Sealed baseline outputs and diagnostic scores](baseline.zip)

## Roof, post and opening diagnostics

The ten canopies across the previous 12 and new eight use a one-course roof and two front posts. We recorded their authored measurements in [canopies.json](canopies.json) and checked each gap width against its native readback. These values enter only after decoding each datapack. The face metric requires the *same block and face orientation* to remain visible when rendering the decoded blocks at the native pose.

| Baseline | Visible roof top faces recovered | Visible roof rim faces recovered | Visible front post faces recovered | Filled clear rays to the front post plane | Filled background rays through the full build | Measurable widths / total |
|---|---:|---:|---:|---:|---:|---:|
| Seven reviewed canopies | 60 / 348 | 68 / 108 | 144 / 144 | 0 / 106 | 0 / 75 | 7 / 7 |
| Three new canopies | 15 / 82 | 16 / 43 | 64 / 64 | 0 / 31 | 0 / 22 | 3 / 3 |

Every measured width was exact. To measure width, the evaluator takes the largest internal span between reconstructed blocks at each height of the authored front-post slice, requiring at least two measurable rows and returning the median in blocks. Clear rays test whether decoded blocks obstruct a world-verified line to the center of each visible gap cell. The background-only subset also checks whether the reconstructed silhouette incorrectly fills its pixel. A real back wall seen through a gap can fill the image without blocking a ray to the front-post slice.

The selected oblique canopy angles contain zero visibly downward-facing roof faces. The fixed release supplies underside coverage on 36 of the 80 held or regression views. The ordinary baseline recovers 525 of 1,060 downward-facing blocks with their underside still visible in the fixed exteriors, 202 of 382 in Twin Spires and 16 of 30 in Cloister.

## Frozen challenger comparisons

| Eight new shapes | Correct of 1,431 | Extras | Extras on hidden structure | F1 |
|---|---:|---:|---:|---:|
| Previous default `a16668f` | 1,014 | 482 | 21 | .6929 |
| Junction creases `5219558` | 1,066 | 174 | 6 | .7982 |
| Repeated-patch evidence `783380a` | 1,070 | 139 | 6 | .8106 |
| Combined plane rules `af24549` | 1,070 | 139 | 6 | .8106 |
| Combined rules plus verified neighboring edges `af24549` | 1,070 | 139 | 6 | .8106 |

The repetition and both combined methods returned exactly the same registered coordinates on these eight pictures. Against the baseline, the combined method gained 70 correct placements, lost 14, removed 359 extras and added 16. On the three new canopies each challenger recovered 100 of 167 visible blocks with four extras, gaining 26 correct placements, losing zero and removing 257 net extras. On the larger positive facade the combined method lost three correct placements and added one extra. On the colonnade it gained 21 net correct placements and removed 75 net extras.

The combined method raised visible rim recovery from 16 to 43 of 43 and roof top recovery from 15 to 28 of 82 on the three fresh canopies. On the seven reviewed canopies, rim recovery rose from 68 to 99 of 108, and top recovery from 60 to 113 of 348. The methods retained all 208 front post faces, all ten exact gap widths and all 137 clear sightlines. On the broader 80-view downward-face test the two combined rounds recovered 776 and 778 of 1,472 visible downward faces, compared with 743 for the baseline. [Per-image predictions, roof and gap audits, and underside scores](challengers.zip)

### Why the top surface remains difficult

In the new canopies, 79 of 82 roof top faces occupy at most two rows of the source image. Of 326 four-pixel-grid samples that actually hit the roof tops, 314 landed in an automatic surface mask but only 64 in a mask assigned to the horizontal axis. Only 29 of the 82 visible top faces received even one sample. All 43 roof rim faces had at least four sampled pixels and at least half of those pixels in a mask with the correct axis. On the seven reviewed canopies, 297 of 348 top faces also occupied at most two source rows. The next depth and placement evidence worth testing is a native-resolution strip sampled along the thin roof surface. The extent and orientation figures here are scoring diagnostics against recorded visible faces. Recompute them with `benchmark.canopy_fixture_coverage --inputs PATH_TO_INPUTS --out runs/coverage.json`.

## Replay the native confirmation

Download the 11,792,186-byte input archive and verify SHA256 `6f5097f69727125448cd0615d191fdd3b2f348edd9101745a798c4c0711aafe7`. It contains eight original PNGs, recorded poses, native block readbacks, and the ordinary upstream masks, counts and build manifests. [Native confirmation fixture](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/download/canopy-confirmation-v1/canopy-confirmation-v1.zip)

```powershell
gh release download canopy-confirmation-v1 --repo jeremyepling/minecraft-image-reconstruction --pattern canopy-confirmation-v1.zip --dir runs
Expand-Archive runs/canopy-confirmation-v1.zip runs/canopy-confirmation-v1
$env:MCIR_NATIVE_TRANSFER_FIXTURES = (Resolve-Path runs/canopy-confirmation-v1/fixtures).Path
$fixture = (Resolve-Path runs/canopy-confirmation-v1/confirmation).Path
python -m benchmark.canopy_compare --inputs $fixture --candidate-root PATH_TO_BASELINE --candidate engine.atlas_plane_depth:place_ranked --out runs/canopy-baseline
python -m benchmark.canopy_fixture_metrics --inputs $fixture --predictions runs/canopy-baseline --out runs/canopy-openings.json
```

The candidate must provide `method(image,version,parts,counts,camera)`. Supply the baseline output directory with `--baseline` and a private candidate `--out` directory to obtain paired gains and losses. On this PC, `benchmark.canopy_fixture reserve` and `benchmark.canopy_fixture build` reconstruct the manifest and upstream outputs from the original captures. For the fixed-release underside diagnostic run `python -m benchmark.canopy_fixture_undersides --predictions PATH_TO_FULL_EXPORTS --out runs/undersides.json`.
