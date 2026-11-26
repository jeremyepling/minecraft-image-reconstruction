# New-structure confirmation

Two authored Minecraft Java 26.3 structures were reserved at `434f44b` before placement scoring. The switchback has two stair flights around an open well, an overhanging landing and a rear lookout. The loading dock has two recess depths, an offset platform and a beam above two front posts. Each received one low-left and one high-right camera view. The screenshots use bricks, polished andesite and birch planks on the switchback, and spruce planks, stone bricks and oxidized copper on the dock. [Native photographs](preview.jpg)

The disposable vanilla client captured all four requested framebuffer images in clear weather at noon. The fixture checked 6,024 server state readbacks, including air and floor probes. The scoring labels keep visible building blocks per photograph. The fixture retains the authored space, unfiltered block states, measured camera, and per-pixel face ID and depth. The shape hash accounts for translations, horizontal rotations and reflections. Neither new shape matches any of the 89 earlier shape hashes. None of the four image hashes matches the 358 earlier unique images checked. The screenshot-only camera produced a metric estimate on all four views. All 1,127 checked visible reference corners projected within two native pixels. [Novelty checks](novelty.json) · [Camera accuracy](camera-audit.json) · [Frozen frame hashes and cameras](screen.json)

## Replay

The release asset is 8,817,257 bytes with SHA-256 `f1d53989e6d87f085491e0c3bff6e6bf373eaccf8eed855d0dcf9cc6baad7320`. All 53 archived files were read back and compared with the scoring inputs. [Portable fixture](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/tag/visible-geometry-new-shapes-v1)

```powershell
gh release download visible-geometry-new-shapes-v1 --repo jeremyepling/minecraft-image-reconstruction --pattern visible-geometry-new-shapes-v1.zip --dir runs
Expand-Archive runs/visible-geometry-new-shapes-v1.zip runs/blind-input
$env:MCIR_NATIVE_TRANSFER_FIXTURES = (Resolve-Path runs/blind-input).Path
python -m benchmark.canopy_compare --inputs runs/blind-input --candidate-root PATH_TO_CEBBFFA --candidate benchmark.visible_depth_score:integrated --out runs/blind-baseline
python -m benchmark.canopy_compare --inputs runs/blind-input --candidate-root PATH_TO_FINALIST --candidate benchmark.visible_depth_score:integrated --baseline runs/blind-baseline --out runs/blind-final
python -m benchmark.blind_shape_results --root runs/blind-input --predictions runs/blind-final --out runs/final-parts.json
```

Prepare the calibrated 26.3 atlas in each candidate checkout before scoring. From its root run `python reconstruct.py atlas --version 26.3 --out runs/26.3/atlas` if an existing validated atlas is unavailable. The two challenger-only endpoints in `benchmark.blind_shape_endpoints` use this fixture branch for the original placer. Set `MCIR_BLIND_OVERLAY` to the matching experiment checkout, use this fixture branch as `--candidate-root`, and pass `benchmark.blind_shape_endpoints:nearby` or `benchmark.blind_shape_endpoints:thin`. Each method consumes identical screenshot, camera, mask and count inputs. The scorer opens camera and world truth after export.

## Placement results

The final production entry at `36d406a` recovered 20 additional visible blocks across four images. It retained all 598 previously correct positions and added 18 incorrect ones. Its F1 improved by 0.59 percentage points across the 747 visible targets. Every method placed zero cells at known hidden source blocks. Each photograph was reconstructed independently. Scoring allows one translation for the whole prediction and gives every authored part the same translation. The table measures position only. [Full-precision scores and revisions](summary.json) · [Per-photo exports, decoded cells and part scores](scores.zip)

| Frozen method | Correct | Extra | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior default (`cebbffa`) | 598 | 110 | 84.46% | 80.05% | 82.20% |
| Second horizontal pass (`f478162`) | 600 | 119 | 83.45% | 80.32% | 81.86% |
| Nearby depth/face model alone (`5452523`) | 599 | 113 | 84.13% | 80.19% | 82.11% |
| Thin-face atlas model alone (`3b7b69c`) | 609 | 115 | 84.12% | 81.53% | 82.80% |
| Combined, one thin-face pass (`e634403`) | 613 | 125 | 83.06% | 82.06% | 82.56% |
| Combined, two thin-face passes (`36d406a`) | 618 | 128 | 82.84% | 82.73% | 82.79% |

| Source photograph | Visible targets | Correct, prior → final | Extra, prior → final | F1, prior → final |
| --- | ---: | ---: | ---: | ---: |
| Switchback, low-left | 98 | 86 → 87 | 20 → 21 | 84.31% → 84.47% |
| Switchback, high-right | 129 | 102 → 103 | 39 → 44 | 75.56% → 74.64% |
| Loading dock, low-left | 264 | 215 → 221 | 8 → 10 | 88.30% → 89.29% |
| Loading dock, high-right | 256 | 195 → 207 | 43 → 53 | 78.95% → 80.23% |

The high-right switchback gained one correct cell and five extras. Across the two dock images, the final method missed 33 of 125 visible roof cells, 20 of 118 deck cells and 14 of 38 recessed-bay cells. On the switchback lookout it missed 14 of 37. The final combination and thin-only model finished within 0.02 F1 points of each other. Four views of two structures can confirm that some gains transfer. A larger set of independently authored shapes would be needed to distinguish such a small difference.

The fixture and report can be merged as confirmation coverage. These photographs were kept sealed through selection and were first disclosed after the `36d406a` configuration was frozen.

The lowest final score was the high-right switchback. The orange faces include the upper brick pier and lookout platform. Many extra centers project into the background, close to the horizon on either side of the lookout. The right panel projects each extra block center using the recorded native camera, so dots at different depths can overlap. [Visual comparison](switchback-high-errors.png) · [Original, uncropped screenshot](switchback-high-source.png)

The ordinary entry point reproduced the scorer's exported positions and materials for all 106 prior-default blocks in the low-left switchback and all 260 final-default blocks in the high-right loading dock. [Original CLI check](cli-check.json) · [Final CLI check](final-cli-check.json)
