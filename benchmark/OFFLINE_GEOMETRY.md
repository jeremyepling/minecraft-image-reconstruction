# Reusing the same buildings without starting Minecraft

All 56 screenshots of the eight buildings have a saved map of their visible surfaces. The model still sees one screenshot. After it draws outlines, the scorer opens that screenshot's map and checks the guesses. The review sheet uses the same map to mark correct, extra and missed outlines. [Saved maps](data/visible-geometry-v1.zip)

To score the existing predictions from disk, run this from the checkout. It reads the archived outputs and maps without starting a model or a game.

```console
python -m benchmark.score_saved_planes
```

The saved map labels every fourth pixel in each direction. At the benchmark's 1440 × 900 resolution that gives a 360 × 225 grid. Zero is background. Every other number refers to a connected flat surface. Each surface has its direction, world plane, visible block corners, extent in the picture, and whether it meets the size threshold for scoring. The camera and individual block faces remain in the original fixture files. The map can be used for outline, block-spacing and depth comparisons.

The ZIP contains a JSON file per screenshot, keyed by floor and frame ID. Its index records the generating code fingerprint and hashes for the original screenshot, camera, face labels, block states and derived map. A checksum mismatch on a known frame raises an error. For a new fixture without a saved reference, the scorer can still compute the surface map from its local labels.

To rebuild the maps after a deliberate change to the geometry definition, run the command below. It checks the frozen fixture release and writes a complete replacement atomically. Running it without `--rebuild` only verifies the release.

```console
python -m benchmark.surface_reference --rebuild
```

For this release all 56 maps matched freshly calculated geometry on Windows, and all 168 per-image outline score records from the published round agreed exactly. The archive is 417,162 bytes. Over five alternating single-process passes on this PC, loading all 56 maps with source checks took a median of 0.30 seconds. Projecting them again from the saved face labels took 1.72 seconds. These times measure preparation of the scoring reference. [Original predictions and scores](reviews/normals-01/manifest.json)
