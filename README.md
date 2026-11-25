# Minecraft Image Reconstruction

Reconstruct a Minecraft Java build from a screenshot and place it in a real world. The project measures the visible block grid independently of material identification. A version calibration ties the atlas, in-game examples, scoring benchmarks and structure exporter to a specific vanilla client.

[Architecture, call paths, defaults and experimental modes](ARCHITECTURE.md)

## Single-image exterior benchmark

Run `.\Invoke-WallRecovery.ps1 .\screenshot.png -Out .\runs\my-build` to use the full surface-recovery sequence on a new image. It writes the outlines, crops and block count estimates and records which models ran. [Single-image build and local models](benchmark/WALL_RECOVERY_BUILD.md)

The **Wall Separator** found 423 of 598 scored surfaces on 56 saved screenshots. Joining short wall edges, looking under overhangs and splitting merged walls with MobileSAM brought that to **471 of 598**. The review shows what each individual change added and where the full sequence still missed a wall. [Zoomable surface comparison](benchmark/reviews/recovery-overview-01/comparison.jpg) · [Results for all eight buildings](benchmark/RECOVERY_ROUND.md)

Read down each building’s column for the source, recorded surfaces, predictions and same-location close-ups. Run `.\Invoke-SurfaceRound.ps1 -Round recovery-summary` to rescore all six combinations and rerun the ordinary Block Counter. The [underside sheet](benchmark/reviews/recovery-01/comparison.jpg) enlarges the low angles. The earlier [depth and line comparison](benchmark/SEPARATION_ROUND.md) tested which model should supply each kind of evidence.

Open the [pipeline comparison](benchmark/reviews/perimeters-01/comparison.jpg) to follow the same surface through the original photograph, Surface Finder outline, context crop, perspective correction, block-grid overlay and texture match. The [three close-ups](benchmark/reviews/perimeters-01/edge-closeups.jpg) show why the old courtyard, workshop and registry diagonals were misleading. Reproduce the review with `.\Invoke-SurfaceRound.ps1 -Round perimeters`. [Surface and crop results](benchmark/PERIMETER_ROUND.md)

Regions First supplies the visible outlines in the **Surface Finder**. The **Surface Straightener** then removes perspective from each masked surface. The **Block Counter** measures repeated textures across and down, and the **Material Matcher** compares individual cells with the vanilla atlas. Run `.\Invoke-BlockCountRound.ps1` to grade the saved screenshots. [Stage names and the legacy Rectangle Finder](benchmark/SURFACE_FINDER.md) · [Block-count comparison](benchmark/COUNT_ROUND.md) · [Per-picture report](benchmark/reviews/count-01/report.html)

Run `.\Invoke-MaskRound.ps1` for the four-way comparison on the 56 saved views. It compares the Rectangle Finder, preserved regions, and the same regions refined with ScaleLSD v2 or DeepLSD. The review shows each building from the same angle, with the recorded geometry in its own column. [Comparison sheet](benchmark/reviews/mask-four-01/comparison.jpg) · [Per-picture results](benchmark/reviews/mask-four-01/report.html) · [How the experiment works](benchmark/MASK_ROUND.md)

Use the same 56 saved screenshots and their geometry whenever you test a new surface method. Change the method, run it on the images, then compare its outlines against the same Minecraft surfaces. A coordinator in one chat can give the baseline and each challenger to a different agent, run them in parallel and return one comparison. [How to run a new combination](benchmark/OFFLINE_GEOMETRY.md)

Open the [eight-building picture sheet](benchmark/reviews/normals-01/buildings-b.jpg) to see each original screenshot beside the MoGe-2 surface directions and the Rectangle Finder’s checked outlines. Run `.\Invoke-SurfaceRound.ps1 -Round saved` to compare the Rectangle Finder’s saved outlines against the recorded buildings entirely offline. [Saved geometry and replay commands](benchmark/OFFLINE_GEOMETRY.md)

The visible-plane experiment compares the deterministic outline method with two local vision alternatives on eight Minecraft structures. The review includes clean source screenshots, the effects on openings, block scale and relative depth, and one command to reproduce it. [Measured results and execution path](benchmark/NORMAL_ROUND.md)

Two Windows runs produced identical predictions for all 56 frames. Two independent Linux replays reproduced every plane match count and the default selection. The experimental relative-depth pass varied across runners. Using normal regions on the original six layouts, Windows proposed 274 near/far relations and the final Linux replay proposed 277. Both verified 41 correct relations there and 14 on the blind buildings. [Windows run manifest](benchmark/reviews/normals-01/manifest.json) · [First Linux replay](https://github.com/jeremyepling/minecraft-image-reconstruction/actions/runs/45600701874) · [Final Linux replay](https://github.com/jeremyepling/minecraft-image-reconstruction/actions/runs/45601327150)

Run the shared regression suite with `python -m benchmark.exterior_suite`. It replays 40 original screenshots from two clear-noon worlds, pairs tiled and weakly marked floors at the same pose, and compares the results with the recorded camera reference. [Suite, worlds and review](benchmark/EXTERIOR_SUITE.md)

Round one measures the projected camera and block grid on newly captured vanilla Java 26.3 exteriors. To reproduce the experiment on its frozen inputs, run `python -m benchmark.camera_round` from this checkout. To use the selected camera on a single screenshot, run `python reconstruct.py build IMAGE --through camera --version 26.3 --out runs/camera-build`. [Round design and review](benchmark/CAMERA_ROUND.md)

## Create a calibration world for a new version

The saved benchmarks above run with Python and their listed dependencies. Use the following live procedure when building new reference displays for a game version. It needs an installed vanilla client and compatible JDK. Live verification has run on Windows with JDK 25 and Minecraft Java 26.1, 26.2 and 26.3.

```console
python -m pip install -r requirements.txt
python reconstruct.py calibrate --version 26.3 --out runs/26.3
```

The command reads the game's block registry and texture/model assets. It creates a flat calibration world, builds three shuffled displays of the registry plus walls of touching blocks, and takes labeled screenshots at clear noon from different angles. It chooses the matching method on validation images, scores held-out placements and grid boundaries, then reconstructs a separate example wall. A second game session places the exported structures and reads back the resulting blocks. Compare `evaluation.json` and `independent-check.json` when evaluating releases or algorithm changes. Use a fresh output directory for each run.

For a new release, substitute its Java Edition version ID. If the client is missing, add `--install-dir runs/minecraft-cache` to download the matching official assets and libraries. A release that changes the game's driver API or save format can require an adapter. The compile and live-placement checks expose those incompatibilities. [Runtime and installer](runtime/README.md) · [Calibration and metrics](calibration/README.md)

For an offline atlas and a first reconstruction without launching Minecraft, run `python reconstruct.py atlas --version 26.3 --out runs/26.3/atlas` and omit `--library` below. Use `--jar PATH` for a client outside the usual installation or Gradle cache. The generated atlas PNG, texture metadata, and direction-specific placement index stay on your machine.

## Reconstruct an image

```console
python reconstruct.py analyze examples/repair-wall/wall-only.png --out runs/wall-grid
python reconstruct.py build examples/repair-wall/wall-only.png --geometry runs/wall-grid/geometry.json --atlas runs/26.3/atlas --library runs/26.3/face_textures/softened-library.npz --out runs/wall
```

Replace the example PNG with a JPEG, PNG, or another Pillow-readable image. To run both image stages in one command, omit `--geometry`. Review `grid.png` for the detected outline and spacing. `geometry.json` records the image fingerprint, camera-plane transforms, source pixels and destination coordinates. You can reuse this file with another version's atlas and library. `dimensions.json` reports block measurements. `matches.json` records the closest texture candidates, distances and margins. Each visible cell receives its best eligible block when building. A small margin means the decision is ambiguous. Reviewer labels belong in test cases.

The result is a Java datapack, `reconstruction.zip`, containing tiled structure templates. File export runs with Minecraft closed. When the task calls for actual placement and an in-game screenshot, run `python reconstruct.py verify-build runs/wall --version 26.3 --out runs/wall-world`. Its save is under `client/saves/Palette Lab` and the screenshot is under `captures`. To place it in an existing world, put the ZIP in that world's `datapacks` folder, run `/reload`, stand at the desired origin and run `/function reconstruction:build`.

## Angled views and hidden geometry

The automatic pass proposes a main wall or a front and return from image structure and looks for periodic seams. Busy backgrounds, stepped roofs and repeated window bays can produce the wrong outline or block pitch. For those images, edit the generated `proposed-scene.json` and rebuild using `--scene PATH`. Geometry hints are independent of block identities. This example defines a six-by-six wall by its four pixel corners and world-relative location.

```json
{"planes":[{"name":"front","quad":[[80,40],[250,55],[247,211],[82,216]],"columns":6,"rows":6,"origin":[0,0,0],"facing":"south"}]}
```

Order corners clockwise from top left. The origin is the block at the bottom left of that plane. Supply a plane per visible surface with shared coordinates at adjoining edges. Directions include north, south, east, west, up and down. A south wall runs toward positive X on the right. On an upward-facing roof, the image top runs toward negative Z. Omit both counts to estimate the lattice from seams. Use `holes` containing lists of image polygons to exclude openings or occlusions. Alpha transparency also excludes cells. For detailed parallel façades, the geometry engine also supports shared vanishing directions, approximate block scales and per-plane anchors; the `dimensions` section can join measured adjoining bands across setbacks.

By default, the pack contains the observed surfaces. For one wall, `--depth 8` builds a hollow eight-block-deep shell from its side, roof, floor and back edges. Depth supplied this way is an assumption. Hidden rooms, backs of buildings, stair/slab states, and identical-looking block IDs cannot generally be recovered from one view. Reconstruction currently places full cubes. The catalog also measures exact-ID recognition for irregular blocks as a separate research metric. Use `--white-balance` for conservative neutral-patch correction on a shader image.

## Verification

Minecraft can drop sand, gravel and concrete powder when a reconstructed face has a gap below it. For known vanilla gravity blocks, the exporter checks support within the reconstructed shape. Where support is missing it takes the highest-ranked self-supporting alternative and records the coordinate, original material and replacement in `placements.json` under `substitutions`. The original texture rankings remain in `matches.json`. This check assumes full-block supports and does not model later player edits or redstone changes.

| Java release | Registry IDs tested | Held-out labeled faces correct | Touching-wall material IDs correct | Auto-detected independent wall |
| --- | ---: | ---: | ---: | ---: |
| 26.1 | 1,168 | 393/400 | 233/240 | 6 × 6, 33/36 IDs |
| 26.2 | 1,196 | 401/409 | 229/240 | 6 × 6, 29/36 IDs |
| 26.3 | 1,286 | 413/417 | 234/240 | 6 × 6, 31/36 IDs |

The contiguous-wall benchmark supplies a coarse wall outline and grid seed, then tests the fit against independent camera-projected seams. All 448 tested seams per version were within one tenth of a block. The independent automatic case is a cropped in-game wall with nine masonry materials. The versioned packs also passed live readback on a 36-block reconstruction from an oblique view. Whole-building accuracy on arbitrary scenes remains unmeasured. [Detailed client results](tests/results/client-validation.json)

The Java 26.3 building benchmark adds canopies, stepped towers, recessed openings, and flat seams beside real corners. The experimental `--geometry-method texel` improved mean silhouette overlap from 0.195 to 0.829 across 60 held-out views. The optional `--match-exposure` improved material accuracy on aligned faces from 61.2% to 75.2%. Combining the options reduced end-to-end accuracy on adjacent-material panels. [Four-family benchmark, screenshots and reproduction steps](benchmark/README.md)

Run `python -m unittest discover -s tests` for the release-format and geometry regressions. The client JARs, vanilla texture sheets, trained libraries, generated worlds and runtime data are produced locally.

The gallery color experiment compares fixed geometry with three material matchers in fresh Minecraft worlds. Searching for repeated atlas details recovered 1,147 of 1,479 material observations on four previously seen views. Sharing a palette produced the closest rendered colors. [In-game comparison and material options](benchmark/COLOR.md)

Use `--geometry-method remap` to estimate a common elevation grid from texture repetition and test visible openings for depth. The measured front view recovered five recess depths on a freshly arranged wall and left a flush section in place. [How the engine uses shadow, angled edges and projected rims](benchmark/RECESSES.md)

For an angled wall, `python reconstruct.py inspect-openings IMAGE --out REVIEW` writes an annotated image and the evidence for each depth decision. This experimental command is separate from export. [How a setting earns adoption, and what happened on two new walls](benchmark/ENTRANCES.md)

To trace the general form first, run `python reconstruct.py inspect-form IMAGE --out REVIEW`. It draws the visible outline and major edges, then inspects smaller line junctions for inward and outward relief. It also works on images without visible pavement. [Line tracing, screenshot comparison and measured accuracy](benchmark/LINE_FORM.md)

Add `--cube-turns` to fit three mutually perpendicular world directions to the image. The preview marks supported outline turns and shows how the same axis projects differently across the frame. Use `--geometry-method cube` with `analyze` or `build` to use the corrected silhouette for block occupancy. [Projected block directions and test results](benchmark/CUBE_TURNS.md)
