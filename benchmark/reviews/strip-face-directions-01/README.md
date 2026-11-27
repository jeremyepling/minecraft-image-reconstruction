# Wall, top or underside on the same photographed strip

Keep the three placement decisions experimental. Against the Java 26.3 placer at `edb10a3`, exporting all winning face directions recovered seven visible blocks on the 104 fixed structure views but introduced twelve extras, including three hidden placements. Only two recovered positions remain required under a per-block 4×4 horizontal-evidence rule. The horizontal-only export added one correctly placed underside block on Twinspire and no extras. That block is a sliver and lies outside the 632-block target cohort. [Paired results](results.json) · [Frozen input manifests and changed exports](comparisons.zip)

## Method

Use the estimated camera, source image, predicted surface masks, calibrated 26.3 texture atlas and decoded provisional build. On a two-native-pixel grid, find unclaimed mask samples near the projected edge of an existing block. Split elongated connected strips into local windows of at most 40×40 native pixels. The existing predicted mask direction groups samples and does not determine the winning direction.

For each strip, nominate horizontal and vertical planes at nearby integer levels. Test the two possible wall axes, an upward face below the camera or a downward face above it. Ray direction and plane position must allow the source pixels to see the proposed face. As a result a top and an underside cannot both explain the same non-horizon ray from one camera. Each physically feasible plane is evaluated on the same photographed strip. Require a visible brightness or color boundary where the strip meets the adjoining render or the mask exterior. Fit atlas texture, rotation, mip level and phase on alternating spatial samples; compare the held-out samples and the projected grid's alignment with the rim. Abstain on weak held-out agreement or insufficient margin over competing axes and levels.

Then re-render proposed cells together. Check for enough visible pixels on each proposed face, estimated silhouette spill beyond the existing build and loss of a previously visible cell. A valid local atlas match can still produce a false 3D placement when the initial silhouette or neighboring depth is wrong. The horizontal-only rule keeps horizontal winners; a third rule also asks a visible coplanar neighbor for the same within-tile texture sample. Surface labels, reference occupancy and recorded cameras enter only scoring.

## Results on matched views

| Decision | Development 72, correct / extras added | Fixed 104, correct / extras added | Fixed F1 | Alternate 20, correct / extras added |
| --- | ---: | ---: | ---: | ---: |
| Current default | 0 / 0 | 0 / 0 | 90.4964% | 0 / 0 |
| All winning directions | 13 / 4 | 7 / 12 | 90.4894% | 0 / 4 |
| Export only horizontal winners | 3 / 0 | 1 / 0 | 90.4989% | 0 / 0 |
| Horizontal with neighboring texture confirmation | 1 / 0 | 0 / 0 | 90.4964% | 0 / 0 |

All three matched the default on the four new views and separate four shelter/gate photos. Development wall outputs accounted for all four extras there. The two narrower rules were frozen after development and before the fixed comparison. On fixed views the all-directions rule added two correct blocks on Physical Shade left. It also added five extras on the masonry Shopfront high-right view and a hidden extra on Portico masonry low-left. The ordinary right Colonnade 01 tied for all rules. Its top-only slivers are a different subset from the 632 cases here: none of the pictured Colonnade's misses met *all* conditions for that cohort.

## Where orientation failed

There were 1,317 elongated source patches on the fixed images. On 748 the search could place both a vertical and a horizontal plane at plausible nearby levels. The remaining patches lacked a valid competition or sufficiently supported projected cells. On the subset of those 748 with one recorded label covering strictly more than half of all test pixels, raw best scores produced the following counts before confidence/visibility filtering.

| Recorded majority | Predicted wall | Predicted top | Predicted underside |
| --- | ---: | ---: | ---: |
| Wall | 363 | 33 | 7 |
| Top | 73 | 23 | 0 |
| Underside | 64 | 0 | 13 |

This is a patch count on previously unclaimed pixels. It measures the proposed orientation choice, separately from exported block placement. The atlas frequently matched a vertical plane on a real horizontal strip. Neighbor confirmation helped avoid false placements but also rejected the remaining fixed-set gains. [Source hashes and per-view results](opportunity.json)

## How much of the 632-block target could be recovered?

The audited baseline cohort consists of top-only or underside-only missed blocks whose photographed footprint is mostly uncovered by the build, mostly covered by a detected mask, and lacks majority coverage in the correct orientation. Reference labels define membership only after generating predictions. For the local tolerance comparison, a horizontal-only block remains required if a 4×4 native square belongs to that same block. There were 17 such blocks in the 632-block cohort.

| Stage | All target blocks | Target blocks with a 4×4 patch |
| --- | ---: | ---: |
| Missing in cohort | 632 | 17 |
| A strip with a scored option sampled this block | 291 | 12 |
| Some alternative contained the correct cell and had a same-axis target witness within the strip | 154 | 11 |
| Clean, jointly verified per-cell selection using the reference answers | 85 | 1 |
| All-directions image-only rule | 3 | 0 |
| Both horizontal-only image rules | 0 | 0 |

The oracle permits answers from incompatible plane/strip proposals to nominate separate cells, then re-renders their union and applies the original visibility and opening checks. Its 85 are an attained clean selection under those checks. The one retained non-sliver target was on Twinspire 2 earth low-left; the actual selector abstained on it. The idealized experiment across *all* missed positions, including walls, yielded 253 clean additions, 145 required under the local tolerance. These are opportunity counts and are excluded from the exported result.

## Replay

The method commits before scoring were `a9856ae` for the horizontal rules and its parent for the initial all-directions development trial. Use the orientation-audit checkout containing `benchmark.strip_orientation_eval` with the shared inputs restored and use this experiment checkout for candidate code.

```powershell
foreach ($m in 'conservative','horizontal','corroborated') {
    python -m benchmark.strip_orientation_eval --candidate-root ../strip-face-directions --candidate "benchmark.strip_face_directions:$m" --out "../strip-face-directions/runs/$m" --workers 4
    python -m benchmark.strip_orientation_eval --transfer-inputs ../framed-depth-transfer/runs/transfer-input --candidate-root ../strip-face-directions --candidate "benchmark.strip_face_directions:$m" --out "../strip-face-directions/runs/transfer-$m"
}
cd ../strip-face-directions
python -m benchmark.strip_direction_opportunity --workers 4 --out benchmark/reviews/strip-face-directions-01/opportunity.json
python -m benchmark.strip_direction_results
```
