# Reusing the same buildings without starting Minecraft

Apply the [visible-only product intent](../PRODUCT_INTENT.md) to prediction, training and scoring. A single-image run reconstructs only what is visible in that frame; sibling views and source-world labels must not supply hidden output geometry. Visibility masks distinguish observed surfaces and clear sightlines from unobserved cells. Unobserved is unknown, not an empty-space label. Score visible recovery and unsupported extra guesses or placements separately, without rewarding completion of unseen regions. Preserve the original definitions when replaying archived scores.

For surface-matching reviews, `.\Invoke-SurfaceRound.ps1 -Round detection-sheet` redraws the latest comparison from its saved predictions. Use `-Round afterpass` to rebuild predictions from archived Planarity and ECT fields, score all 56 pictures, and run the ordinary Block Counter. Use `-Round single-wall` for the earlier depth-voting experiment. [Current sheet](reviews/afterpass-models-01/comparison.jpg) · [Experiment and default](AFTERPASS_MODELS.md)

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

## Stage-specific comparisons and review sheets

When investigating merged surfaces, score exact reference IDs gained and lost, unpaired guesses, correctly assigned pixels and covered openings. For an overhang, distinguish its vertical front from its downward-facing underside. Count genuine underside matches separately from the fraction of a seam a line model could trace. For model-input comparisons, hold the splitting logic fixed, freeze the development choice before scoring the regression layouts and include close-ups at angles where the target is visible. Run the Block Counter on the selected masks before changing an ordinary build default. [Separation benchmark](SEPARATION_ROUND.md)

Make the pipeline comparison sheet one full-resolution portrait image for zooming. Give every building a fixed column across the top and every stage a row down the left. Keep additional examples from the same building within its column. Follow the same surface through the clean screenshot, Surface Finder perimeter, neighboring-pixel crop, perspective-corrected masked surface, supported block-grid overlay and material sample with its atlas guess. Add the source-angle reconstruction when assembly was tested. Mark unresolved or unrun stages explicitly. Use the shared renderer in `benchmark/surface_pipeline_sheet.py` for exterior reviews. A four-corner sampling chart belongs in a labeled diagnostic. Draw actual outline rings including holes, and keep recorded geometry in a separate overlay. Preserve the frame, proposal, native crop and stage with reviewer feedback.

Run `.\Invoke-BlockCountRound.ps1` for a block count or material-sampling comparison on the frozen Regions First outlines. Grade the visible surface dimensions, recovered cell crops and exported coordinates separately. [Count protocol and results](COUNT_ROUND.md)

Use the [surface-detection sheet](reviews/afterpass-models-01/comparison.jpg) as the first visual when debugging surface matching. Keep one zoomable portrait image with the same building columns and camera views across rounds. Read down each column from the clean screenshot through all predictions, recorded surfaces, scored misses, extra guesses, and paired reference and prediction close-ups at the same location and scale. Keep the status colors consistent. Show how many surfaces were found, scored, actually visible, below the size cutoff and proposed in each picture. A percentage beside a selected outline grades that one shape. Regenerate the sheet from saved predictions with `.\Invoke-SurfaceRound.ps1 -Round detection-sheet`. Use the pipeline sheet when reviewing what happens to a surface in later stages.

For this project, select the surface method by accuracy. Record model count and runtime separately. When two depth models grade a proposed split, freeze the source mask, sample positions and acceptance rule and change only the numerical depth field. Track true gains, lost surfaces and cuts across flat walls. Grade correct block dimensions separately. [Current experiment, selection and replay](SINGLE_WALL_DEPTH.md)

For a new surface candidate, use `.\Invoke-SurfaceRound.ps1 -Round candidate`. Pass `-Round compare` the finished run directories. The runner and its workers inherit image-only mode, which blocks project Minecraft capture and registry launches. See the offline guide for a two-agent command example.

For changes after Wall Recovery, replay `.\Invoke-SurfaceRound.ps1 -Round afterpass`. Pixel-wise Planarity supplies full-image plane proposals to the Surface Finder. When an outline is missing, record the best overlap of a raw proposal and the overlap of the accepted surface at the same location. When testing edge confirmation, grade real depth changes and flat-wall seams first, then apply the cuts and grade whole surfaces, openings and block dimensions. An edge may be correctly classified on a wall that an earlier pass has already split. [Frozen model comparison](AFTERPASS_MODELS.md)

From this checkout, run `python -m benchmark.exterior_suite` for the Stage 1 regression comparison. During development, `--split dev` and `--floor light_gray_concrete` isolate the development weak-ground case. Compare a changed method on both floors before reporting a result. See [the suite guide](EXTERIOR_SUITE.md) for reference scores, release updates, review records and replay instructions.

The earlier surface round uses `python -m benchmark.surface_round`. Replay the fixed 56-frame normals experiment with `.\Invoke-SurfaceRound.ps1 -Round normals`. Select a method on the saved development buildings and report generalization separately on the remaining layouts. All eight reviewed buildings now serve as fixed regression fixtures. [Methods and reference results](NORMAL_ROUND.md)

For work on overlapping posts, facade relief, Shopfront outlines and thin Colonnade roofs, the [six reported issue regions](reviews/visible-error-budget-01/README.md) provide repeatable local checks. Use them to distinguish a better outline from a better first-hit block and inspect both the affected region and its neighbor. These reviewed examples complement the fixed suite and separate transfer cohorts.
