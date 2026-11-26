# Which parts of an exterior can a local vision model recover?

Run `normals-01` compares three ways to outline visible planes from one vanilla Java 26.3 screenshot. Grouping pixels by the surface direction predicted by MoGe-2 found 223 of 466 eligible planes on the six existing structures. The published contrast-and-shading method found 78. On two newly captured buildings the figures were 58 of 132 and 15 of 132. A matched plane is a connected flat surface that overlaps its recorded visible footprint by at least 50%, with one prediction per match. This is a plane detection score. Block counts, openings and depth have their own measurements. [Comparison sheet](reviews/normals-01/comparison.jpg) · [Original screenshots and palettes](reviews/normals-01/fixtures.jpg) · [Run manifest](reviews/normals-01/manifest.json)

## Same pictures, three methods

Start with the eight-row picture sheet. Each row shows the original building, B’s map of which way its surfaces face, and B’s rectangles over the same screenshot. B uses its map to group neighboring pixels, then uses brightness changes in the screenshot to adjust the edges. The recorded Minecraft surfaces supply the green, pink and gold scoring marks. The sheet uses B’s best whole-building view for each layout, balancing missed surfaces and extra outlines. Those eight views are selected from the original run and stay fixed when reviewing later results. [What B saw in each building](reviews/normals-01/buildings-b.jpg)

The shopfront has 14 counted surfaces from the high-right angle and 16 from the low-left angle. Four screenshots show each angle, giving 120 chances to find a surface. The last four columns below add up all views of a building. The two development layouts determined the choice of challenger. The four regression layouts had appeared in earlier experiments. The two blind layouts were held back until the methods were frozen.

| Building | Split | Frames | Surfaces in one picture | Total across frames | Contrast + shading | Contrast ranked by normals | Regions grouped by normals |
|---|---|---:|---:|---:|---:|---:|---:|
| Shopfront | Development | 8 | 14–16 | 120 | 10 | 16 | 57 |
| Wing | Development | 12 | 1–5 | 36 | 18 | 19 | 28 |
| Courtyard | Regression | 12 | 1–8 | 56 | 14 | 18 | 36 |
| Portico | Regression | 8 | 17–20 | 148 | 25 | 28 | 67 |
| Workshop | Regression | 4 | 14–16 | 60 | 6 | 11 | 16 |
| Crossing | Regression | 4 | 11–12 | 46 | 5 | 5 | 19 |
| Galleries | Blind | 4 | 24–25 | 98 | 9 | 14 | 37 |
| Registry | Blind | 4 | 7–10 | 34 | 6 | 7 | 21 |

| Cohort | Contrast + shading F1 | Ranked by normals F1 | Grouped by normals F1 | Precision, baseline → grouped | Recall, baseline → grouped |
|---|---:|---:|---:|---:|---:|
| Six existing layouts, 48 frames | 19.1% | 29.5% | 55.4% | 22.2% → 65.8% | 16.7% → 47.9% |
| Four previously reviewed regression layouts alone | 18.9% | 28.8% | 52.0% | 22.9% → 62.4% | 16.1% → 44.5% |
| Two fresh layouts, 8 frames | 14.4% | 24.1% | 54.2% | 19.7% → 70.7% | 11.4% → 43.9% |

The ranking rule gave each development layout equal weight. Mean layout F1 was 26.8% with contrast and shading, 40.3% with normal ranking and 67.1% with normal regions. The reporting table above pools plane instances. On the historical frames the baseline matched all previously published per-frame match, proposal, eligibility and background-spill counts. [Selection](reviews/normals-01/selection.json) · [Per-layout, floor and cohort scores](reviews/normals-01/scores.json)

## How the pipeline ran

1. **Capture and freeze.** The vanilla client placed full blocks, locked the clock at 6,000 ticks, disabled the time and weather cycles, cleared rain and thunder, and captured 1440 × 900 frames with default textures and the HUD hidden. The runner checked the fixture, world and image hashes. All views, floors and palettes of a building stayed in one split. Each predictor received one RGB image and the Java version. The normal and depth maps are reusable local CPU inference outputs keyed by the image, model hashes, preprocessing and library versions. [Capture conditions](suites/exterior-v1.json) · [New world, coordinates and teleport points](worlds/normal-confirm-v1/world-index.json)
2. **Propose the visible planes.** The baseline found contrast boundaries and corners and used shading to score rectangular outlines. The first challenger used MoGe-2 ViT-S predicted normals to rescore contrast-derived outlines, checking the interior direction and changes across the boundary. The second used the same normal map to estimate orthogonal scene directions, group connected similarly oriented pixels, fit straight edges and refine the corners with local image contrast. It saved both the four corners and the pixels belonging to each component. MoGe supplied surface directions. Its metric depth and camera estimates did not enter the experiment. [Outline implementations](../engine/normal_regions.py) · [Normal ranking](../engine/normal_rank.py)
3. **Count blocks and lift outlines into 3D.** All three methods used the same deterministic texture-repeat code. It rectified each proposed plane and looked for recurring texture detail and seam phase along both axes. A supported pitch set the local scale in blocks. A shared observed edge could carry scale to a neighbor. Other components retained relative units. The lens estimate used perpendicular outline directions with a recorded 70° vertical-field-of-view fallback when needed. No block IDs entered this step. [Texture-repeat estimator](../engine/surface_pitch.py) · [Camera and shared-edge geometry](../engine/surface_model.py)
4. **Compare near and far.** On the same predicted outlines the runner compared two signals. The geometric version compared depths where the planes shared a coordinate scale. The learned version sampled the relative inverse-depth map from Depth Anything V2 Small on either side of a candidate boundary. Both needed at least four sample pairs and 70% agreement. A depth ordering says which surface appears closer. The model supplied no distance in blocks. [Ordering code](../engine/surface_order.py)
5. **Score the completed predictions.** The evaluator projected saved Minecraft blocks and visible faces using the recorded camera, matched planes one-to-one, measured gap coverage and evaluated scale and ordering separately. Eligibility required at least four visible block faces and 24 quarter-resolution sample points per connected plane. Two labeled diagnostics provided extra information only for their measurement: the actual focal length for lifting the same corners, and actual occlusion boundaries for testing the depth signal. The 56 compressed prediction files preserve the unassisted outputs for inspection. [Evaluation](normal_round.py) · [Per-frame scores](reviews/normals-01/cases.json)

## Where it still breaks

**Openings decide the default.** On the fresh buildings, filling the complete rectangles from normal regions would cover 1,754 of 4,022 measured opening pixels (43.6%). The baseline covered 120 (3.0%). Using just the pixels assigned to each normal region covered 118 (2.9%) and still matched 58 planes. On the six established buildings the corresponding opening fill was 24.6% for rectangles and 4.6% for their pixel masks, against 7.7% for the baseline. An opening pixel is visible background bracketed by the structure above and on both sides, measured once at quarter resolution. This can also count a space between building parts. It does not establish the depth of that space.

In the high-right masonry Galleries view, the normal method found 7 of 24 eligible surfaces. Its fitted rectangles crossed 213 of 828 opening samples. The visible masks crossed five. The current 3D lifting step consumes the rectangle corners. The promotion rule, frozen before confirmation, allowed at most a three-percentage-point increase in opening fill. The recommended ordinary surface default therefore stays `edges+shade`, with `repeat` for block spacing and `geometry` for ordering. For the next geometry experiment, carry the pixel masks through plane adjacency, counting and block placement, then check the resulting occupied cells around openings on fresh layouts. [Labeled gallery example](reviews/normals-01/comparison.jpg)

**Absolute scale is sparse.** Across the original six layouts, normal regions returned both spans on 12 of 223 matched planes. All 12 were within half a block on both axes, and one had an accurately registered grid. The baseline returned six span estimates, five within tolerance. On the fresh pair, normal regions returned no span on its 58 matched planes. Of its 15 estimated metric depths on the existing suite, seven were within one block. Supplying the recorded lens raised that to 14 of the same 15. On the fresh buildings there was no normal-region metric depth to check. Direction was easier: on matched fresh planes the median direct MoGe normal error was 4.6°, while the median error of the direction derived from the quad and estimated lens was 33.2°. The learned method found two of four eligible soffits in the fresh set.

**Depth ordering needs a reliable boundary.** With the recorded boundary locations, Depth Anything picked the nearer surface on 342 of 346 old-suite pairs and 117 of 118 fresh pairs. Starting from the normal-region outlines, it emitted 71 relations on the fresh set. Fourteen were verified correct, two reversed, and 55 could not be associated with a distinct eligible true boundary in the scoring set. Geometry emitted none on those outlines. On the original six layouts the same automatic depth comparison gave 41 correct, one reversed and 232 unverified. The frozen rule required at least half of all emitted relations to verify as correct on both development and blind. Ordinary builds retain geometric ordering. The separately tested M-LSD Tiny line detector covered 72.7% of true boundary pixels on the 48 existing frames, compared with 90.9% for the deterministic line detector at similar pooled precision. It ran only in the line comparison. [Line experiment](reviews/normal-lines-01/README.md) · [Model research and original sources](LOCAL_VISION_MODELS.md)

## Floors, palettes and future comparisons

Eight independent layouts supply 56 frames. Masonry uses stone bricks, tuff and polished andesite. The second palette uses white concrete, cobblestone and spruce planks. There are 20 smooth-stone-floor frames and 36 light-gray-concrete-floor frames. For the 20 exact pose-and-palette pairs on the original four layouts, group-by-normal F1 was 60.8% on smooth stone and 58.8% on concrete. Both floors yielded six measurable spans. Only the smooth-stone group yielded a registered grid. All eight fresh frames use concrete. [Fixture gallery](reviews/normals-01/fixtures.jpg)

Keep these eight as frozen regression layouts after review. The earlier sample target is 18 independent layouts, six reserved for blind confirmation, with at least two palettes and two oblique views per layout. This gives 72 weak-floor images before paired floor controls. Include low-texture planes, palette changes across flush walls, visible backs at different depths, thin soffits and through-gaps. More views of one wall measure viewpoint sensitivity. More independently built layouts test generalization. [Fixture design](SURFACE_ROUND.md)

## Reproduce and review

For a quick check, run `.\Invoke-SurfaceRound.ps1 -Round saved`. It compares B’s saved rectangles against the saved Minecraft surfaces in all 56 screenshots. To redraw the eight-building sheet, run `.\Invoke-SurfaceRound.ps1 -Round building-sheet`. Both commands run offline from the files in the checkout. [Saved geometry and scoring a new prediction](OFFLINE_GEOMETRY.md)

From the checkout on this PC, run one command. It verifies both downloaded checkpoints, regenerates any missing image maps and creates a new timestamped run with its own sheet. On another machine, install the pinned Python dependencies in `benchmark/suites/normal-requirements.txt` first. [Pinned models and download sources](normal_assets.py)

```powershell
.\Invoke-SurfaceRound.ps1 -Round normals
```

For an ordinary single-image surface build, use the first command. The second explicitly runs both learned components from this experiment. Each build writes `surfaces.json` and a `build-manifest.json` naming the applied methods. Material recognition, voxel placement and Minecraft recapture are separate from this geometry round.

```console
python reconstruct.py build IMAGE.png --version 26.3 --through surfaces --out runs/visible-default
python reconstruct.py build IMAGE.png --version 26.3 --through surfaces --surface-outlines normal-regions --surface-order depth-anything --out runs/visible-models
```

Use `benchmark.camera_feedback` to record a visual preference against the run, source frame, floor, native-pixel rectangle and stage. Review entries go into the run's journal alongside the frozen scores. They do not feed into predictions. [Review regions](reviews/normals-01/review-regions.json) · [Feedback command](camera_feedback.py)
