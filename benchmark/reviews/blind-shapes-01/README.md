# New-structure confirmation

Two authored Minecraft Java 26.3 structures were reserved at `434f44b` before placement scoring. The switchback has two stair flights around an open well, an overhanging landing and a rear lookout. The loading dock has two recess depths, an offset platform and a beam above two front posts. Each received one low-left and one high-right camera view. The brick/andesite and timber/copper palettes are fixed per structure; the screenshots use bricks, polished andesite and birch planks on the switchback, and spruce planks, stone bricks and oxidized copper on the dock. [Native photographs](preview.jpg)

The disposable vanilla client captured all four requested framebuffer images in clear weather at noon. The fixture checked 6,024 server state readbacks, including air and floor probes. The scoring labels keep visible building blocks per photograph. Separate archives retain the authored space, unfiltered block states, measured camera, and per-pixel face ID and depth. The shape hash accounts for translations, horizontal rotations and reflections. Neither new shape matches any of the 89 earlier shape hashes. None of the four image hashes matches the 358 earlier unique images checked. The screenshot-only camera produced a metric estimate on all four views. All 1,127 checked visible reference corners projected within two native pixels. [Novelty checks](novelty.json) · [Camera accuracy](camera-audit.json) · [Frozen frame hashes and cameras](screen.json)

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

The two challenger-only endpoints in `benchmark.blind_shape_endpoints` use this fixture branch for the original placer. Set `MCIR_BLIND_OVERLAY` to the matching experiment checkout, use this fixture branch as `--candidate-root`, and pass `benchmark.blind_shape_endpoints:nearby` or `benchmark.blind_shape_endpoints:thin`. Each method consumes identical screenshot, camera, mask and count inputs. The scorer opens camera and world truth after export.
