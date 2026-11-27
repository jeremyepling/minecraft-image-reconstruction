# Exact camera and signed block coordinates

This experiment isolates the ranked placement handoff. The source-only control for each photograph saves its actual surface masks, normals, quadrilaterals, relative inverse depths and counts. All exact-camera arms consume that same handoff. The upstream Surface Finder and Counter may have used the ordinary source camera before their handoff. After ranked placement the assay runs the normal material assignment and datapack exporter, decodes the pack and scores every visible block under one whole-build integer XYZ translation. The additional edits normally performed by `surface_model.build` between ranked placement and material assignment are excluded equally from every arm.

| Arm | Projection and starting distance |
| --- | --- |
| `exact-floor-base`, `exact-floor-roof` | Native recorded projection, ordinary floor intersection and nonnegative-height policy. |
| `exact-floating-base`, `exact-floating-roof` | Same projection; skip assumed floor contacts and allow signed heights. Start from the existing fallback `focal / (image_height / 24)`, approximately 23.05 blocks here. |
| `exact-distance-sky` | Base-origin bypass with one extra reference scalar: the median axial depth of the visible block centers. |

All cameras use native image pixels and block units. We retain the recorded layout's horizontal origin. The base origin uses the lowest observed block's Y coordinate; the roof origin uses one above the highest observed block. The evaluation uses reference coordinates to choose the two origins. Plane fitting receives the translated matrix and its declared distance input. Native corner checks compare each eligible recorded face's four stored corners to the projected local vertices, including partly occluded faces. The gauge comparison adds the known integer origin back to every emitted cell and compares sites *and* IDs without independent score alignment. The standard score also reports its ordinary pavement-based registration; `projection_checks.exact_registered_correct` provides the direct world-coordinate check for these exact cameras.

## Six-structure paired result

| Exact-camera placement arm | Pavement | Plain | Synthetic sky |
| --- | ---: | ---: | ---: |
| Grounded, base origin | 76.34% | 71.67% | 71.55% |
| Grounded, roof origin | 14.21% | 18.97% | 12.33% |
| Floating, base origin | 40.15% | 34.44% | 36.76% |
| Floating, roof origin | 40.15% | 34.44% | 36.76% |
| Floating with reference median distance, base origin | — | — | 62.08% |

Each cell is pooled strict geometry F1 over six structures (4/144 Visible Builds v3, 2/40 Decorative Builds v1); these small combined results do not update the fixed 104 default. Every arm has supported exact projection. Three of the 18 roof-origin grounded runs emitted no blocks. The 18 base/roof floating pairs returned identical world-coordinate positions and material IDs (4,886 positions counted across the 18 views). Exact reprojection agreed with the native saved corners to within 1.12e-10 pixels. A no-op replay of the ordinary RGB-estimated camera on harbor pavement reproduced all 230 saved ranked cells.

On sky composites, supplying the extra distance increased correct sites from 1,050 to 2,454 and extras from 621 to 1,411. Strict site-plus-ID F1 reached 44.83%. Harbor's measured reference distance was 39.0 blocks and viaduct's 52.6; the shared fallback placed both too close. Keep this branch experimental. The ordinary floor prior aligns well with the photographed source layouts, which really sat on the recorded plane, even when the surrounding scenery is replaced. A build-only distance estimate and independently observed depth anchors are still needed.

## Floor assumptions

The experimental policy disables the column-bottom intercepts in provisional plane placement and junction proposals, exposed-floor seam ownership, the grounded depth envelope, and the roof-rim intersection of assumed ground and a wall. It removes the Y=0 eligibility gate in ranked plane options, primary/secondary face enumeration, edge recovery, neighbor recovery, section moves, pillar cells, roof strips, roof rectangles and local panels. The roof search starts at the provisional observed vertical extent. The section classifier receives height above the current provisional inventory instead of absolute Y. Geometry from sampled charts and photo-verified horizontal faces remains eligible, including undersides. The visible-sheet test requires the two photographed faces and checks the gap between them at three integer origin heights.

The outer pipeline also includes a ground-derived inset host plane, aperture-versus-pavement filtering, a counted ground corner and a floor-height exemption in exterior occlusion. These modules recognize the optional camera flag when invoked with a floating camera. The paired handoff assay does not exercise these outer hooks. Existing sky/horizon classifiers are outside this placement comparison.

## Replay

With the primary workspace Python, NumPy, Pillow and the calibrated 26.3 atlas, run these commands in this checkout after the frozen inputs and baseline handoffs exist:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m benchmark.floating_exact --workers 6 --wait
python -m benchmark.floating_exact_report
python -m unittest tests.test_floating_grid
```

Fixtures and baseline handoffs come from the stable checkout's `runs/floating-build-grid-01` directory. Each arm writes a real pack, decoded scores and coordinate inventories to this checkout's `runs/floating-build-grid-01-verified` directory. Each score includes the original evaluator hashes, the actual variant RGB hash, the frozen handoff hash and the pack hash.
