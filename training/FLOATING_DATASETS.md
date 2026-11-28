# Floating datasets

Use [Floating Front 100](../benchmark/FLOATING_SHARED_DATASETS.md) for Expanded Split-Border replays. The standard score uses all 100 full-block noon images. The other three shards supply matched morning, afternoon and decorative noon images. Each frame retains the original source RGB and the scoring references, including its measured camera, clock, readback positions and states, and per-pixel face and metric depth maps.

The companion `floating-front-100-v1-structures.zip` is a Java 26.3 datapack. It contains one full-block template and one decorated template per layout. The three full-block lighting views share the same structure. `structures.json` maps each template to a capture ID, native resource name, SHA-256, dimensions and exact world origin. The complete world is for scene recreation and label verification. Ordinary predictors receive one screenshot.

To place a saved build, install the ZIP into the world's `datapacks` directory and reload. In an empty area, use `/place template <resource> <x> <y> <z>` with the resource and origin from `structures.json`. Each template includes all occupied cells, including hidden parts. The recorded empty void provides the background. To recreate the original scene, the ZIP also contains each batch's starting save, original placement commands, layout, requested views, client settings and block registry. The frame shard carries the measured camera and resulting world readback.

For a fresh acquisition of the first build, run `python -m training.capture_front --out runs/front-recapture-001 --first 1 --last 1` with Java 26.3 and the benchmark dependencies. Omit the range to generate all 100. Use the checksum-verified original PNGs and label maps whenever reproducing a published score. GPU, driver and renderer timing can affect newly captured pixels.

## Historical Combined Floating dataset

The [combined manifest](floating-combined-v1.json.gz) points to the exact saved Java 26.3 screenshots and their original labels. It retains **240 active views** from **51 layouts in 20 architectural families**. Development and comparison cohorts can use the same index. The active split is **164 train, 38 validation and 38 test**. Every family's original assignment is preserved. The frozen 84-condition Floating benchmark remains a named score comparison for existing runs.

The combined index holds 200 historical Visible Builds views and 40 historical Decorative Builds V1 views. Select `visible` or `decorative` in the [sparse replay launcher](../benchmark/FLOATING_SHARED_DATASETS.md) to reuse those inputs. Use the [recorded method and revision](../benchmark/FLOATING_SHARED_DATASETS.md#replay-historical-methods) when reproducing historical scores.

## Capture sources

| Source | Saved coverage | Use in the combined dataset |
| --- | --- | --- |
| Visible Builds v1 | 152 records across 32 layouts | Original pilot and legacy regression renders. The 144 whole-structure views exactly match V2. Eight close-ups failed framing and are excluded. |
| Visible Builds v2 | 144 views, including 48 regression | Canonical copy of the admitted V1 RGB and visible references, with a checksummed full-build framing audit. Its pilot contributes 92 active views across arcades, stepped halls, sawtooth roofs, terraces, watchposts, arched bridges, cloisters and twin spires. |
| Visible Builds v3 | 144 new views across 24 layouts | Three viewpoints per layout, two palettes, new large structures, and per-frame framing, dimensions and palette metadata. It contributes 108 active views. |
| Decorative Builds v1 | 40 new views across 10 layouts | All 40 are active. Shape-aware labels distinguish full cubes from slabs, stairs, walls, fences, gates and trapdoors. Face corners and depth follow the actual block-state geometry. |
| Original-204 placement reference | 204 records | 144 repeat V2. The other 60 images cover 30 historical geometries. They are evaluation-only. Sixteen geometries recur under different old transfer split labels. |
| V1 paired time-of-day releases | Four releases with 152 records each | 144 matched whole-structure views at each of 400, 2500, 9500 and 11600 ticks. These add 576 distinct RGBs with the original view and visible targets. Sixteen inherit the excluded Watchpost 3 layout. Each release also contains eight archived close-ups. |
| Floating background fixtures | Six matched triplets | Each original screenshot has a plain backdrop and a synthetic sky variant. Twelve new images reuse the parent's visible geometry. |

V3 adds pillars, viaducts, factories, amphitheaters, shrines, observatories, harbors and castles. Decorative adds market courts, stable yards, riverside mills, hillside chapels and walled gardens. The original-204 reference accompanies V3 and is a separate archive from its 144 new captures. Original V1/V2 are publicly tagged in this repository. The other capture archives and synthetic images are available in the saved workspace runs. [V1 release](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/tag/visible-builds-v1) · [V2 data contract](DATASET.md)

The index contains 976 distinct RGB fingerprints across all listed roles. That includes 240 active, 48 V2 regression, 60 additional historical, 560 eligible lighting and 12 background views, plus 40 excluded core and 16 excluded lighting entries for audit. The 560 lighting views contain 368 eligible active-parent views and 192 regression-parent views. Select them as paired robustness checks, with the same split and parent ID as the noon capture. Sample by parent or layout if adding them to training so repeated lighting does not dominate. Early morning and late afternoon give the strongest contrast. In each, the median share of visible building pixels with at least an 8-level RGB channel change from noon is 92%. At 2500 and 9500 ticks, 137 of 140 eligible pairs per release have zero pixels above that threshold. The 48 regression views include 16 same-view pairs with separately captured floors. The sky and plain fixtures change the background by composition while preserving visible structure pixels.

## Terracotta and concrete review

The active filter uses concrete and unglazed terracotta coverage among **visible building pixels**, pooled over the viewpoints of each layout and palette. A palette at 80% excludes the whole layout, including its other palette and angles. Orchid is retired explicitly. The 40 excluded core captures all belonged to train.

| Excluded layout | Main reason |
| --- | --- |
| Watchpost 3 | Cool palette, 83.0% concrete or unglazed terracotta |
| Amphitheater 01, 02 and 03 | Orchid retired. Orchid coverage is 77.0%, 82.7% and 84.5% respectively. |
| Factory 02 and 03 | Chalk palettes, 81.3% and 81.3% |
| Observatory 03 | Verdigris palette, 83.0% |

Watchpost 2 earth (78.9%), Observatory 02 verdigris (76.6%), and Factory 01 chalk (76.4%) are the closest active palettes to the cutoff. Factory 01 chalk rear-oblique (89.6%) and Watchpost 2 earth low-left (81.4%) exceed it in individual views. These stay active because the threshold applies to the pooled palette. A broader texture audit also flags Shrine 03 ochre, at 92.5% when smooth sandstone, smooth stone and smooth quartz join the existing list. The broader category is a diagnostic because smoothness within these materials and lighting conditions varies.

## Labels and access

All sets share exact RGB and visible block coordinates and materials, recorded camera, a per-pixel frontmost face-ID raster and metric camera-depth raster. The face table supplies normals, projected corners, per-corner visibility and surface IDs. V1/V2 also carry the benchmark `reference-faces.json` with roles and coverage. V3 records larger structure dimensions, third viewpoints and more detailed palette metadata. Decorative includes exact block-state geometry and block-kind labels, which support scoring full and partial blocks separately. Lighting releases preserve the original visible labels and store both camera records, clock tick and measured RGB changes. Synthetic background pairs link to the parent labels.

The predictor receives one source image and the game version. Use camera and block references for evaluation and supervision. Target only the visible faces and their owning blocks. Background and occluded space remain distinct.

The manifest stores each image's fingerprint, its archive and member path, and label fingerprints. Paths are relative to the saved runs root. Rebuild it from the pinned ZIPs and saved background fixture manifest with:

```powershell
python -m training.floating_combined --archives-root C:/code/minecraft/minecraft-image-reconstruction/runs
```

Use `training.floating_combined.load()` to load the index. `training.floating_combined.select_active_cases()` filters runner fixtures to the active set. Choose regression, lighting and background records explicitly for their matched comparisons. The current combined index has not been scored as a new benchmark.
