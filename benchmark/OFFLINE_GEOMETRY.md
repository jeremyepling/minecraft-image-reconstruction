# Reusing the same buildings without starting Minecraft

For surface-matching reviews, start with `.\Invoke-SurfaceRound.ps1 -Round detection-sheet`. It prints the path to a portrait sheet drawn entirely from saved screenshots, predictions and recorded geometry. The per-surface pairings are saved beside it. The default source is `separation-01`. Pass `-SourceRun benchmark/reviews/perimeters-01` to regenerate the earlier sheet. [Current review layout](reviews/separation-01/comparison.jpg)

Use `.\Invoke-SurfaceRound.ps1 -Round separation` to repeat the wall and underside comparisons and the downstream dimension check on all 56 pictures. It runs picture jobs in parallel and reuses frozen model evidence by screenshot hash. [Inputs, selection and scores](SEPARATION_ROUND.md)

To investigate merged walls and the undersides of overhangs, compare saved outlines with the raw line, surface direction and depth evidence at the same spot. [Close-ups and six-method audit](OVERHANG_REVIEW.md)

Use `.\Invoke-SurfaceRound.ps1 -Round perimeters` to score the three Surface Finder methods and review the surface through cropping, straightening, counting and atlas matching. The experiment reuses the same frame IDs and the model's numerical depth cache. An optional `-Fields` path can point to a different compatible cache. [Review and results](PERIMETER_ROUND.md)

Run `.\Invoke-MaskRound.ps1` to replay the four-method comparison of the Rectangle Finder, model regions, ScaleLSD v2 and DeepLSD. Its per-picture report and comparison sheet use the 56 saved images. [Reproduction and cache setup](MASK_REPLAY.md)

Run `.\Invoke-BlockCountRound.ps1` to straighten the same surfaces, count blocks, compare individual texture crops and score the coordinates decoded from the exported datapacks. It reuses the frozen Regions First masks from the four-method experiment. [Count comparison](COUNT_ROUND.md)

All 56 screenshots of the eight buildings have a saved map of their visible surfaces. The model still sees one screenshot. After it draws outlines, the scorer opens that screenshot's map and checks the guesses. The review sheet uses the same map to mark correct, extra and missed outlines. [Saved maps](data/visible-geometry-v1.zip)

To score the existing predictions from disk, run this from the checkout. It reads the archived outputs and maps without starting a model or a game.

```console
python -m benchmark.score_saved_planes
```

On this PC, `.\Invoke-SurfaceRound.ps1 -Round saved` finds the installed Python and runs the same check. To score a new prediction, use `python -m benchmark.score_saved_planes --surfaces OUTPUT/surfaces.json`. Its build manifest identifies the screenshot. For a prediction without a manifest, also pass `--image PATH/TO/FROZEN.png`. This scores the visible outlines on that one image and reports the surfaces found, the ones missed and the extra outlines.

To make the [eight-building sheet](reviews/normals-01/buildings-b.jpg) from the archived pictures, model colors and scores, use `.\Invoke-SurfaceRound.ps1 -Round building-sheet`. Its eight camera views stay fixed to the original review. The saved color previews are for display. Future methods can make their own predictions from the original screenshots.

## Try a new combination in one conversation

Ask the coordinator to compare the baseline with one or two alternatives on the same saved buildings. It assigns the independent methods to parallel agents, keeps code edits in separate worktrees and gives each run a private output folder. All agents use the same version, image IDs, geometry maps and split. One coordinator checks the manifests and returns a single comparison. A useful request is “Try both ways of finding wall edges on the saved shopfront and wing pictures. Show me which finds more walls and which fills more openings.”

For example, these first two commands generate fresh predictions from the 20 saved development images. They can run at the same time. Each invocation needs a new output directory. The last command reads the finished results. For a learned method, `-Fields` points to verified, read-only model outputs derived from the screenshots. When testing a changed model, generate its own outputs from those same pictures. Separate worktrees can use absolute paths to the shared fields and completed run directories.

```powershell
.\Invoke-SurfaceRound.ps1 -Round candidate -Name baseline -Outlines edges+shade -Out runs/edge-baseline
.\Invoke-SurfaceRound.ps1 -Round candidate -Name regions -Outlines normal-regions -Fields runs/normal-fields -Out runs/regions-challenger
.\Invoke-SurfaceRound.ps1 -Round compare -Runs runs/edge-baseline,runs/regions-challenger
```

The comparator checks that the jobs used the same frames, scoring code and saved geometry with a single stage under comparison. Each manifest records the code, methods, settings and artifact paths. Use `-Split all` to run a frozen choice across all 56 pictures. The original camera benchmark has its own 40-image suite, and material tests use the saved image crops, versioned atlas and exemplar library described in their guide. [Camera replay](EXTERIOR_SUITE.md) · [Material replay](TEXTURES.md)

## When Minecraft opens

The routine experiment commands set `MCIR_IMAGE_ONLY=1` for themselves and their workers. The project’s Minecraft launcher checks that flag before setting up a game session. Missing images or stale inputs surface as data errors. Restore the saved fixture for that version. Opening the game is a separate operation for creating or editing a reference structure, or for placing and photographing a completed build. Exporting a datapack can be done from files.

The saved map labels every fourth pixel in each direction. At the benchmark's 1440 × 900 resolution that gives a 360 × 225 grid. Zero is background. Every other number refers to a connected flat surface. Each surface has its direction, world plane, visible block corners, extent in the picture, and whether it meets the size threshold for scoring. The camera and individual block faces remain in the original fixture files. The map can be used for outline, block-spacing and depth comparisons.

The ZIP contains a JSON file per screenshot, keyed by floor and frame ID. Its index records the generating code fingerprint and hashes for the original screenshot, camera, face labels, block states and derived map. A checksum mismatch on a known frame raises an error. For a new fixture without a saved reference, the scorer can still compute the surface map from its local labels.

To rebuild the maps after a deliberate change to the geometry definition, run the command below. It checks the frozen fixture release and writes a complete replacement atomically. Running it without `--rebuild` only verifies the release.

```console
python -m benchmark.surface_reference --rebuild
```

For this release all 56 maps matched freshly calculated geometry on Windows, and all 168 per-image outline score records from the published round agreed exactly. The archive is 417,162 bytes. Over five alternating single-process passes on this PC, loading all 56 maps with source checks took a median of 0.30 seconds. Projecting them again from the saved face labels took 1.72 seconds. These times measure preparation of the scoring reference. [Original predictions and scores](reviews/normals-01/manifest.json)
