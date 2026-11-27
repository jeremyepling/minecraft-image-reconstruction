# Expanded Minecraft capture datasets

The datasets use Minecraft Java 26.3 native framebuffer PNGs. Each screenshot is one reconstruction input. The camera, world readback and visible geometry are evaluation and supervision data. Every new structure view includes the complete build with at least 2% clearance on each edge.

| Dataset | New images | Layouts | Train / validation / test |
| --- | ---: | ---: | --- |
| Visible Builds v3 | 144 | 24 | 108 / 18 / 18 |
| Decorative Builds v1 | 40 | 10 | 24 / 8 / 8 |

## Visible Builds v3

The expansion adds pillar arrays, viaducts, factories, amphitheaters, shrines, observatories, harbors and castles. Each layout has two palettes and three viewpoints. A palette assigns separate materials to masonry, facing, trim, roofing, infill and accents. The largest castle measures 100 blocks wide, 50 high and 36 deep. The largest pillar array measures 88 wide and 42 high. Harbors are the validation family and castles are the test family.

The full benchmark now spans 348 images. The original 204-image companion archive keeps the 104 fixed score views and 100 development and transfer views with their original image hashes and cohort membership. The original cases have already been reviewed in previous rounds. Report their scores separately from the 144 new views.

## Decorative Builds v1

The pilot uses market courts, stable yards, riverside mills, hillside chapels and walled gardens. Each layout has two palettes and two oblique views. Chapels are the validation family and gardens are the test family.

Opaque full cubes appear together with walls, fences, fence gates, trapdoors, slabs and stairs. Connected pieces, open and closed gates, and front-mounted shutters exercise overlapping objects. The labeler resolves the exact state-selected opaque geometry from the vanilla client assets. A pixel belongs to its nearest modeled face. In every example, a foreground decoration hides some pixels of a full block that retains visible pixels elsewhere in the same image. Report decorative blocks and full blocks separately when evaluating the pilot.

The two datasets use the same frame layout. See [the v2 data contract](DATASET.md) for camera axes and basic label fields. Decorative faces use the coordinates of the actual model quadrilateral. Each block records its geometry kind. The dense maps store `face_id` and `depth_z`. A depth is the distance along the camera forward axis in blocks. Occluded cells stay unknown. The recipe and world readback remain available for the scorer and capture replay.

## Capture, package and verify

Run each command from a checkout containing the corresponding recipes. Capture opens a private client. Packaging and archive verification operate on files.

```powershell
python -m training.capture_expanded full --out runs/visible-builds-v3-capture
python -m training.package_expanded --capture runs/visible-builds-v3-capture --dataset visible-builds-v3 --out runs/visible-builds-v3
python -m training.package_expanded --verify-only --out runs/visible-builds-v3

python -m training.capture_expanded decorative --out runs/decorative-builds-v1-capture
python -m training.package_expanded --capture runs/decorative-builds-v1-capture --dataset decorative-builds-v1 --out runs/decorative-builds-v1
python -m training.package_expanded --verify-only --out runs/decorative-builds-v1
```

The manifest indexes checksummed split archives. Each archive contains its own index and per-frame PNG, native camera, readback, visible block labels, and dense raster. A separate archive contains the capture recipes. `preview.jpg` shows a source image from each layout.

With the previous benchmark fixtures installed, rebuild its companion archive from the frozen source and evaluation indices.

```powershell
python -m training.package_placement_reference --out runs/visible-builds-v3-original-204
```

## Visible Builds v1 time-of-day pairs

The early-morning and late-afternoon archives replay every original Visible Builds v1 camera and build at fixed times of 400 and 11600 ticks. Each variant contains 152 frames. The eight historical close-ups have a dedicated diagnostic shard. Each native RGB frame carries its original ID for matching, both camera records and the preserved visible label files. Verification checks the full camera projection and block states against the archived noon capture. The initial checks at 2500 and 9500 ticks found identical structure pixels on the sampled views, so the released variants use times closer to the ends of the day.

```powershell
python -m training.replay_v1_lighting --source runs/visible-builds-v1 --jar PATH_TO_26_3_CLIENT --out runs/lighting-native
foreach ($cohort in Get-ChildItem runs/lighting-native -Directory) {
    python -m runtime.capture capture --version 26.3 --run $cohort.FullName --jar PATH_TO_26_3_CLIENT
}
python -m training.package_v1_lighting build --original-data runs/visible-builds-v1/data --captures runs/lighting-native --lighting morning --ticks 400 --out runs/visible-builds-v1-early-morning
python -m training.package_v1_lighting build --original-data runs/visible-builds-v1/data --captures runs/lighting-native --lighting afternoon --ticks 11600 --out runs/visible-builds-v1-late-afternoon
python -m training.package_v1_lighting verify --release runs/visible-builds-v1-early-morning
python -m training.package_v1_lighting verify --release runs/visible-builds-v1-late-afternoon
```
