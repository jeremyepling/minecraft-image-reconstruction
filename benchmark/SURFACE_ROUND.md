# Visible planes and block spacing

From PowerShell run `.\Invoke-SurfaceRound.ps1`. On other platforms run `python -m benchmark.surface_round` with the dependencies in `benchmark/suites/requirements.txt`. Each command creates a new run under `runs/`. For an ordinary one-image prediction use `python reconstruct.py build IMAGE.png --version 26.3 --through surfaces --out OUTPUT`. Open `surfaces.json` for image corners, repeat evidence, shared edges and 3D coordinates in a right/down/forward camera frame. The build manifest records the methods used. Only connected components anchored by a local pitch estimate have block units.

## Frozen inputs

The 40 paired-floor images in exterior-v1 cover flat and returning walls, recessed shop bays, a courtyard and a pillared portico. Surface-confirm-v1 adds eight native screenshots on light gray concrete, two palettes and two oblique angles of each new layout. The workshop has a stepped roof, annex and cut loading bay. The crossing has a bridge, exposed underside, stairs and a partially backed passage. Masonry uses stone bricks, tuff and polished andesite. Plaster uses white concrete, cobblestone and spruce planks. All six layouts use default textures and full blocks in clear weather at noon. All angles, palettes and floors of each underlying layout belong to one split. [Confirmation index](data/surface-confirm-v1/index.json) · [Playable world and teleport points](worlds/surface-confirm-v1/world-index.json)

Run `python -m benchmark.surface_fixture --out FRESH` to acquire the eight views in a fresh vanilla Java 26.3 world. Run `python -m benchmark.surface_world` to decode and verify the frozen playable save. The benchmark verifies the screenshot release hashes and uses the archived frames for comparisons.

## What the comparison measures

| Comparison | Fixed input | Measurements |
| --- | --- | --- |
| Ground-contact charts, connected edges, edges with shading | One screenshot | One-to-one precision and recall at intersection-over-union ≥ 0.5 against occlusion-tested coplanar patches. Opening spill is the fraction of background pixels bracketed by the building to the left, right and above that an outline covers. |
| Legacy texel count and texture repeat | The same plane footprint for each method | Both spans within half a block. Predicted and true visible intersections within three native pixels in both directions. Run first on labeled rectangles inset by fractional blocks, then on matched automatic quads. |
| 3D lift | The same automatic footprints, pitch estimates and joins | Camera-space normal error and optical depth in blocks. Compare an image-estimated focal length with a diagnostic using recorded intrinsics. Score join consistency by intersecting both matched true planes at rays through the proposed endpoints. |

The outline scorer combines 4-connected unit faces with the same normal and plane offset. It includes a patch with at least four labeled faces and 24 quarter-resolution pixel centers. A rectangle can match only once. The pitch diagnostic requires a fully visible rectangle of at least 2 × 2 blocks. The same scoring code handles noninteger spans and cropped borders. The report includes outputs that abstain and surfaces the detector misses.

The texture method rectifies each plane, checks for repeated texture detail across scanlines and measures the native texel spacing against the default 16-texel block face. A distinct dark joint can establish the boundary phase in each direction. Smooth surfaces, repeated half-courses and unresolved texels can leave the block crossings uncertain. No block ID or texture atlas enters this stage.

For geometric proposals, the detector follows contrast boundaries and junctions. The shading alternative changes how those same proposals are scored. It measures a broad luminance step that persists along an edge after subtracting local brightness ramps. A fitted lens and compatible shared edge endpoints establish relative depth. Texture pitch can anchor their scale. A projected contact can also be an occlusion, and the lift can accept skewed rectangles. Treat each resulting surface as a hypothesis.

## Reference result

The development rule selected edges with shading and texture repeat before confirmation scoring. On the two fresh layouts the old contact method returned zero outlines. Edges alone matched 16 of 106 eligible patches from 97 proposals. Edges with shading matched 11 from 79. Their masks covered 0.92% and 0.94% respectively of the measured opening pixels. The development winner remains recorded as the reproducible experimental default. Edges alone is the challenger to beat on new layouts.

On the same 62 labeled plane crops, repeat returned 26 counts and got all 26 within half a block on both axes. The legacy method returned 62 and got three right. Repeat recovered both grid phases on two crops. Automatic outlines yielded one accurate repeat count among 11 matched patches. On the original floor pairs the repeat method produced the same number of accurate diagnostic counts for each floor, 14 of 44 development crops and 18 of 50 regression crops.

The crossing exposes a camera failure. A detected bridge patch measured 12.18 × 1.68 blocks against 12.10 × 1.68 from the recorded geometry. Its estimated depth was 9.21 blocks against a measured 20.48. With the recorded lens the same outline and count gave 20.20 blocks. Its adjoining patch was 7.55 blocks away in the estimated reconstruction, 16.55 in the recorded-lens diagnostic, and 19.05 in the world. Only one of the three proposed joins connected two matched true planes. Its two endpoint depth discrepancies were 0.32 and 0.11 blocks.

The six layouts give a small controlled development set. For the next validation milestone, target 18 independent layouts with six reserved for fresh confirmation, balanced across strongly textured and smooth walls, flush palette transitions, recesses with visible backs, projecting edges, soffits and through-gaps. Two palettes and two oblique angles per layout give 72 weak-floor images. Additional floor pairs and close crops can share the same layout split. Publish a new fixture version for new layouts and use the existing six for regression.

[Comparison sheet](reviews/surfaces-01/comparison.jpg) · [Fixture gallery](reviews/surfaces-01/fixtures.jpg) · [Run manifest](reviews/surfaces-01/manifest.json) · [Per-floor scores](reviews/surfaces-01/floor-scores.json) · [Opening and junction scores](reviews/surfaces-01/topology.json)

Use `python -m benchmark.camera_feedback RUN --stage outlines --floor light_gray_concrete --image IMAGE_ID --region X0 Y0 X1 Y1 --preferred edges --note "Your observation"` to attach a visual preference. The journal accepts native-image coordinates and keeps reviewer entries independent of the numerical scores. The comparison's example coordinates are in `review-regions.json`.
