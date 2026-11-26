# Roof height on six additional structures

The original height panel uses 12 native 1440 × 900 screenshots of six distinct Minecraft Java 26.3 canopies. It contains an open canopy and a rear-walled canopy at each of three roof courses, measured five, six and seven blocks above the base course. All six have a left and a right view. Across the preceding 204 images we checked 62 distinct full-occupancy shapes, canonicalized for translation, rotation, horizontal reflection and material palette. No geometry or source-image hash overlaps. Four, ten and twelve unused eligible shape pairs were available at the three respective heights. We selected by structure type and occupancy hash before running camera estimation. Every complete structure meets the 2% admission rule on all four edges. [Selection and per-view clearance](selection.json)

## Ordinary single-image build

All 12 images failed metric camera estimation. The seam fitter found too few supported repeats, and the floor atlas registration lacked enough texel coverage. The current build at `54c9fee` and all seven candidate adapters returned empty builds on these images. That gives **0 of 954 visible blocks recovered**, all 12 widths unresolved and zero primary roof-height decisions. These camera failures remain in the denominator. [All scores](scores.json) · [Decoded exports, stage choices and per-frame scores](scores.zip)

## Supplying only the recorded camera

We ran a separate diagnostic using the true camera matrix with a local origin fixed at ground height beside the camera. The ordinary Surface Finder and Block Counter outputs, source pixels, texture atlas and geometry algorithms stayed the same. The previous placement stage ran again with the supplied camera and created each diagnostic prior. Neither the recorded roof height nor the reference surface masks entered these decisions. The evaluator used them after exporting the build.

| Method with supplied camera | Primary on the correct roof and course | Wrong course / missing | Correct of 954 visible | Extras | Relative F1 | Correct roof positions from both front posts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous default | No explicit choice | 0 / 12 | 909 | 8 | 97.17% | 497 / 522 |
| Atlas patch | 4 / 12 | 3 / 5 | 934 | 8 | 98.52% | 522 / 522 |
| Atlas with support fallback | 4 / 12 | 3 / 5 | 934 | 8 | 98.52% | 522 / 522 |
| Post and rim | 0 / 12 | 2 / 10 | 909 | 8 | 97.17% | 497 / 522 |
| Early atlas override | 4 / 12 | 3 / 5 | 914 | 8 | 97.44% | 502 / 522 |
| One-course consensus | 0 / 12 | 0 / 12 | 909 | 8 | 97.17% | 497 / 522 |
| Consensus and atlas patch | 0 / 12 | 0 / 12 | 934 | 8 | 98.52% | 522 / 522 |
| Thin vertical rim band | 11 / 12 | 0 / 1 | 917 | 8 | 97.61% | 505 / 522 |

With the supplied camera, both the highest occupied layer and the largest elevated layer of the previous build matched the roof course on all 12 views. Each of its 12 opening widths was correct and all 305 sampled sightlines were clear. The atlas method's 25 additional correct placements were on the four course-five photos. On all four course-six photos it abstained. It selected a wrong course three times on the four course-seven photos and abstained once. The thin vertical rim band recovered the primary height on all eight taller side views and on three of the four course-five views.

The viewing geometry explains why the three heights have different evidence. In the eight course-six/seven side views the camera eye sits between the top and underside of the roof block. The native reference raster contains **zero visible horizontal roof pixels** on those eight views. The vertical rim is still visible. Each of the four course-five photographs has about 9,900 horizontal roof pixels. [Visibility by face, view and course](visibility.json)

## Elevated view of each structure

The saved elevated photo of each of the same six structures passed the clearance check. All six show a broad roof top, with 74,895 to 168,569 labeled top pixels per image. The ordinary pipeline failed camera estimation on every photo and recovered zero of 552 visible blocks. [Frozen elevated views](elevated.json) · [Face pixels by view](visibility.json)

| Method with supplied camera | Side views, correct primary height | Elevated views, correct primary height |
| --- | ---: | ---: |
| Atlas patch, support fallback, early atlas override | 4 / 12 | 6 / 6 |
| Thin vertical rim band | 11 / 12 | 0 / 6 |
| Post and rim | 0 / 12 | 1 / 6 |
| Consensus, consensus and atlas patch | 0 / 12 | 1 / 6 |

With an elevated camera and a visible roof top the atlas selected the correct roof and course on both photos at each height. None of the seven candidates changed the previous placement. Its tallest and largest elevated layer already gave the correct course on all six. Geometry F1 was just 9.89% because 9,830 extra cells accompanied 540 correct placements. Each of the two height-five photos produced 2,413 cells entirely at roof height. The rear-walled height-six build produced 5,198 ground-course cells. Four of six opening widths were correct, two were unresolved, and 14 of 80 sampled clear sightlines were filled. The two course-seven pictures recovered 194 correct cells with zero extras. Inspect camera recovery first, then the visible boundary used to limit a surface's extent. A correct height alone left these errors intact.

## Replay

The versioned fixture contains the original native screenshots, recorded camera and block readback files, frozen image-derived masks and counts, prior ordinary exports and checksums. The 86-member ZIP is 17,251,703 bytes and has SHA-256 `3c17aeb1b2636a57cd1bb54cb1c55628fe2dac633dbf864b484dcccb9149a85c`. Its published GitHub asset digest matches the local archive. [Download the fixture](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/tag/framed-canopy-heights-v1) · [Input checksum](inputs.json)

```powershell
gh release download framed-canopy-heights-v1 --repo jeremyepling/minecraft-image-reconstruction --pattern framed-canopy-heights-v1.zip --dir runs
Expand-Archive runs/framed-canopy-heights-v1.zip runs/panel-data
python -m benchmark.roof_height_transfer_panel --data runs/panel-data --out runs/panel-control
python -m benchmark.roof_height_transfer_panel --data runs/panel-data --candidate-root ../roof-height-atlas --candidate engine.roof_height_atlas:recover --out runs/panel-atlas
python -m benchmark.roof_height_transfer_panel_pose --data runs/panel-data --previous-root ../roof-span-round --out runs/panel-pose-prior
python -m benchmark.roof_height_transfer_panel --data runs/panel-data --pose-diagnostic runs/panel-pose-prior --candidate-root ../roof-height-atlas --candidate engine.roof_height_atlas:recover --out runs/panel-pose-atlas
```

The previous checkout for the pose diagnostic must resolve to `54c9fee`. Image-only runs use the ordinary estimated camera stored in the archived CLI placement manifest. Missing estimates remain missing.

The elevated supplement on the same release is 12,651,738 bytes with 44 members. Its verified SHA-256 is `966102401dde711a9d6bca4ccb7007f364c0a7317b33c4e101fe455858c86acc`. An unpacked copy reproduced the ordinary control's per-frame score checksum. For the elevated commands replace `--data runs/panel-data` with `--data runs/elevated-unpacked --selection runs/elevated-unpacked/elevated.json`. Give the elevated camera-prior run and each score run its own output directory. [Elevated archive checksum](elevated-inputs.json) · [Download the two fixtures](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/tag/framed-canopy-heights-v1)
