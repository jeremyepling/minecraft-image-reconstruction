# Scoring the build with a free world origin

Give the reconstruction one integer shift in X, Y and Z and apply it to every visible block in that screenshot. A translated copy earns full credit. Moving a roof relative to its supports changes the score even when their individual shapes are perfect. Scale and orientation stay fixed after the usual camera-axis registration.

**Keep the current geometry placer.** Across 112 evaluation screenshots with 24,086 visible block appearances, the change gave the current default exactly the same score. The previous default gained four correct positions on one image. The current default still recovered 245 more visible positions with 1,082 fewer extras. The primary score is pooled visible-position F1, which gives weight to both missing and extra blocks.

| Method | Correct with camera registration | Correct with a free origin | Extras with a free origin | Relative F1 |
|---|---:|---:|---:|---:|
| Previous default, `a16668f` | 18,454 | 18,458 | 2,852 | 81.32% |
| Current plane chooser before the edge pass | 18,667 | 18,667 | 1,768 | 83.86% |
| Current default, `04563e0` | 18,703 | 18,703 | 1,770 | 83.95% |

The suite contains 104 nonempty current predictions and eight images where placement abstained. Its cohorts are Cloister (12), Twin Spires (12), the fixed exteriors (56), the earlier supported native views (12), the reviewed native builds (12) and the additional native confirmation set (8). On the separate 72 training images, the current default retained 14,356 correct positions. On the common 104 evaluation images available for the two chooser variants, the unrestricted depth chooser also retained its lead over the guarded variant. [All 904 decoded-pack checks, input hashes, offsets and per-image scores](scores.zip)

## Roof-to-pillar relationships

Both defaults reproduced all ten measured canopy opening widths. With both front pillars serving as a shared origin, correct roof positions rose from **91 to 170 of 459**. On canopy 09, both methods got the four-block opening right. The previous roof had **0 of 80** visible roof blocks at the correct offsets from the pillars. The current roof had **41 of 80**. A single global shift cannot repair a roof at the wrong depth relative to two correctly placed pillars.

Under the whole-build overlap score, canopy 09 was the previous default's only improved evaluation case: translating its entire prediction by (+30, +1, +19) raised its overlap from 22 to 26 of 112. The current build matched 65 of 112 with seven extras at zero shift. The Shopfront high-right regression also persisted: the previous and current methods matched 137 and 56 of 216, respectively, even after granting each its best shift. [Per-canopy part relationships and opening widths](../relative-spacing-01/README.md)

In a control with two pillars and a roof, shifting the entire build preserved all **24 of 24** tested roof-to-post relationships. Raising only the roof by one block left the opening width correct but broke all 24 relationships. We also translated each of the 20 real canopy outputs by (+42, −17, +31) and reproduced every part-relationship measurement. [Illustrated control](../relative-spacing-01/control.png)

## Scoring and replay

The original evaluator already maps the datapack's local axes and origin into the reference world using the measured and recorded cameras. The new evaluation additionally chooses the translation with the most overlap between the complete predicted set and the image's visible target. Each target-minus-prediction coordinate difference casts one vote. The highest vote count is the exact maximum overlap, without a distance cutoff. Ties prefer the shortest squared displacement, then lexicographic XYZ. All extra blocks still count as errors. The count of extras coinciding with hidden structure uses the selected alignment and is a secondary diagnostic. Evaluation offsets never enter inference or change a datapack.

The ordinary placement benchmark manifest now identifies `summary.relative_geometry.f1` as its primary metric and keeps the camera-registered metrics for comparisons with previous releases. Per-frame records include the scoring offset for each method and its baseline. For paired gains and losses each method receives one independent whole-build alignment to the same reference. The canopy relationship diagnostic anchors the whole build to its two front posts and then measures visible roof blocks in that frame.

To reproduce the archived comparison, restore the frozen inputs and decoded packs from the [canopy evaluation](../canopy-placement-01/README.md), the [previous release](../visible-depth-01/README.md) and the native fixtures they reference. The replay verifies each pack hash and registered coordinate set. Use `--runs`, `--data` and the native fixture root options for other storage locations. The optional `--chooser-runs` also compares the guarded and unrestricted model variants.

```powershell
python -m benchmark.relative_geometry_replay --out runs/relative-geometry.zip
python -m benchmark.relative_canopy_spacing --out runs/relative-spacing
python -m unittest tests.test_relative_geometry
```
