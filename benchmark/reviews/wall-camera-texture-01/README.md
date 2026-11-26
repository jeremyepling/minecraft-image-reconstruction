# Wall texture camera experiment

Keep `experiment/camera-wall-atlas` experimental. Both frozen wall methods added zero camera estimates after the floor texture camera on the 152 saved frames. Of the eight remaining close-front frames, six lacked two usable horizontal vanishing points and two lacked sufficient visible line evidence.

## Three rounds

Each predictor received an RGB image and the Minecraft version. It estimated its own focal length and camera axes. An independent scorer loaded view-masked labels after prediction and registered the arbitrary local horizontal axes and integer origin. All choices, including the additional confidence rule in round three, used the official 72 training images. Both round-three variants were frozen before validation, test and fixed-suite scoring. The old production seam camera ran first on all splits.

| Training method | Cameras / 72 | Vertices within 2 px / 24,644 | Frames with full projection and p90 ≤2 px |
| --- | ---: | ---: | ---: |
| Rectified wall intervals, 7dbbe35 | 20 | 431 | 0 |
| Adjacent different atlas textures, e319d30 | 8 | 109 | 0 |
| Native atlas consensus, e6709f9 | 36 | 8,288 | 20 |
| Consensus with two grounded contacts, 2b942bb | 10 | 3,572 | 9 |

Round one fitted straight wall edges and proposed metric scale from likely one-block intervals. Integer multiples produced severe errors. Round two searched for neighboring, independently atlas-matched materials with a color change, without requiring a repeated texture. A one-pair acceptance rule allowed shading and partial tiles to masquerade as material boundaries.

Round three jointly refined perspective pencils using long building edges, sampled native 16×16 atlas texels at projected block centers, and required a separate, nonadjacent texture match on the same wall chart. Its camera can use either repeated or different materials. A common failure was anchoring a floating ledge to the ground, which produced a one- or two-block error in camera height even with strong texture matches. The guarded variant requires a contact shadow and a consistent camera height and horizontal grid phase at two spatially separated visible wall contacts.

## Frozen evaluation

| Split | Broad cameras / images | Broad full-frame p90 ≤2 px | Guarded cameras / images | Guarded full-frame p90 ≤2 px | Guarded visible XYZ precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cloister validation | 8/12 | 5 | 2/12 | 2 | 89.12% / 10.26% |
| Twinspire test | 6/12 | 3 | 4/12 | 2 | 69.91% / 26.27% |
| Fixed exterior regression | 33/56 | 18 | 20/56 | 17 | 75.32% / 25.28% |

XYZ came from decoded packs using the frozen surface masks and the same geometry-export path. The geometry diagnostic uses a fixed material. Counts include the old seam camera wherever it already returned a pose.

## Which apparent changes were real?

We projected the two predicted face centers into the saved per-pixel visible-face map and compared both atlas texture associations to the actual block IDs. Each frame contributed its strongest claimed different-material pair. In round two, all eight accepted pairs had the same true material on both sides. Four pairs did land on different adjacent coplanar block positions. In round three, both center IDs and adjacency were correct on two of four training examples and one of four fixed examples. The single Twinspire transition was a false match. Cloister had no accepted distinct-material pair. Other supported views used same-material repeats.

The true pairs included cobblestone beside polished diorite, terracotta beside mud bricks and polished andesite beside stone bricks. This measures the identities at the projected sample centers. It does not certify the entire quadrilateral or the metric camera for those images.

## Reproduction

Run the scorer in the checkout containing `benchmark.texture_camera_eval`, with this branch as the candidate root. Use Python with NumPy, Pillow and OpenCV. The experiment used the local ScaleLSD virtual environment with OpenCV 4.12, the release's verified 26.3 vanilla texture atlas and one BLAS thread per worker.

```powershell
python -m benchmark.texture_camera_eval --candidate-root ../camera-wall-atlas --candidate benchmark.wall_atlas_guard --split validation --out ../camera-wall-atlas/runs/r3guard-validation --workers 4
```

Use `benchmark.wall_atlas_consensus` for broad round three, `benchmark.wall_atlas_transition` for round two and `benchmark.wall_texture_camera` for round one. The manifests record split, source and prediction digests. The compressed cases are byte-exact copies of the scorer outputs after decompression. The separate transition audit reads the generated camera cases, RGB and saved visible-face ownership. The placement manifests identify the parallel export evaluation.

