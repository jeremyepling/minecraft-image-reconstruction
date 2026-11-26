# Depth correction on two new Minecraft structures

Four previously unscored whole-structure views improved from **77.61% to 82.41%** pooled visible-geometry F1. The previous engine was `acfe46b`; the challenger was frozen at `0e0c229` before seeing these outcomes. It gained 17 correct positions and lost two, giving 677 correct of 764 visible blocks. It removed 79 extras and introduced one, leaving 202 extras. Known hidden placements increased from nine to ten. All four per-photo F1s improved. [Paired packs and block coordinates](scores.zip) · [Scores and hashes](summary.json)

| View | Previous correct / extra | New correct / extra | Previous F1 | New F1 |
| --- | ---: | ---: | ---: | ---: |
| Platform shelter, low-left | 179 / 17 | 181 / 16 | 87.53% | 88.29% |
| Platform shelter, high-right | 204 / 35 | 217 / 21 | 84.12% | 89.67% |
| Offset gate, low-left | 101 / 22 | 101 / 20 | 82.11% | 82.79% |
| Offset gate, high-right | 178 / 206 | 178 / 145 | 62.90% | 70.50% |

The rightmost gate view recovers 178 of 182 visible positions with both methods. Its reconstructed depth falls from 42 to 39 blocks against a 12-block visible reference. The new output still includes 113 blocks outside the source structure's horizontal bounding rectangle, down from 174. Of these, 105 occur at the lower lintel and side roof's height, and eight at the upper lintel height. The frozen gate-low view gained one error at a hidden source position. The camera estimated all 1,250 checked visible corners to within two native pixels over the four photographs. An ordinary command-line run on the gate-high photo exactly reproduced all 323 block positions and materials in the frozen new export. The previous default also reproduced the 196 exported blocks on the low-left platform. [Source and top-down gate comparison](gate-footprint.png) · [Relative extents and hidden extras](extents.json) · [Normal CLI checks](cli-checks.json)

## Capture and reuse

Two occupancy recipes, material palettes and camera pairs were frozen at `c0fc6a8`. A vanilla Java 26.3 client produced the native RGB photos and recorded camera and block observations. It verified 5,766 server readbacks, and the visible-only labeler found 1,057 face appearances. Every complete structure meets the 2% border requirement on every side. The smallest observed or projected clearance is 18.73%. Neither occupancy matches any of 91 previous shapes, accounting for rotations, translations and reflections, and the four RGBs differ from the 362 unique prior photos. [All four input photos](sources.jpg) · [Framing, camera and per-file fingerprints](framing.json) · [Shape audit](novelty.json)

The portable fixture includes the original images, poses, native readbacks, per-image visibility and depth labels, and frozen screenshot-only surface and count outputs. The ZIP contains 53 verified members, is 8,155,724 bytes, and has SHA-256 `76ea86bda7ebd6c96768d869f3e371840d8f8056f2d4c037f7d95d5a05d6e00a`. Keep these images in the evaluation split. [Download the fixture](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/tag/framed-depth-confirmation-v1)

From a checkout with the calibrated Java 26.3 atlas and project Python dependencies:

```powershell
gh release download framed-depth-confirmation-v1 --repo jeremyepling/minecraft-image-reconstruction --dir runs --pattern framed-depth-confirmation-v1.zip
Expand-Archive runs/framed-depth-confirmation-v1.zip runs/transfer-input
$env:MCIR_NATIVE_TRANSFER_FIXTURES = (Resolve-Path runs/transfer-input).Path
python -m benchmark.canopy_compare --inputs runs/transfer-input --candidate-root PATH_TO_ACFE46B --candidate benchmark.visible_depth_score:integrated --out runs/baseline
python -m benchmark.canopy_compare --inputs runs/transfer-input --candidate-root PATH_TO_0E0C229 --candidate engine.atlas_plane_depth:place_ranked --baseline runs/baseline --out runs/final
```
