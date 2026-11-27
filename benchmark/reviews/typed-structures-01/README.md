# Visible placement and exact block materials

The material pass takes the final visible block coordinates, the estimated camera and the source image. It uses the Minecraft 26.3 texture atlas to select a block ID for each exported coordinate. An unreadable placement retains the smooth stone fallback. Every pack in this review was decoded from its Minecraft templates before scoring. No source world coordinates, reference materials, layout names or reference camera poses enter prediction.

## Results

| Saved views | Geometry strict | Geometry facade | Material accuracy | Joint strict | Joint facade |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 104 | 91.27% | 93.42% | 92.78% (19,274 / 20,773) | 84.68% | 88.51% |
| Other 100, including 72 training views | 90.89% | 93.17% | 97.24% (20,429 / 21,008) | 88.39% | 91.43% |
| Other 28, excluding training | 89.16% | 89.49% | 97.35% (4,814 / 4,945) | 86.79% | 88.24% |

On the fixed 104, the previous smooth stone export scored 1.20% material accuracy, 1.10% joint strict F1 and 0.49% joint facade F1. All 204 views improved on each of these three metrics. Every exported coordinate stayed the same. On the fixed set the matcher assigned 21,809 of 21,819 placements. Nine of the ten fallback placements landed on true blocks. Three happened to be smooth stone. The fixed joint scores ceiling with perfect IDs on the current placements is 91.27% strict and 93.42% facade.

The five training-only refinements scored **79.36%, 88.52%, 94.09%, 94.22% and 97.21%**. The fourth improved overall material accuracy slightly but hurt joint facade F1 on training. The fifth combined sparse and native samples and weighted the source image's palette votes by observed texture coverage. This raised training joint facade F1 from 89.84% to 92.35%. Selection was then frozen. On the fixed 104, the selected method scored 92.78% compared with 82.87% for the face-only alternative, 86.58% using the simpler palette vote and 89.98% using the unweighted blend. [Development scores](development.json) · [Frozen comparisons](variants.json)

### Where material matching fails

The fixed images contain 1,499 wrong block IDs at correctly placed coordinates. Five of the 40 structures account for 793 (52.9%) of these errors. Portico contributed 216, Crossing 163, Adjacency 26 and Shopfront 152 each, and Registry 110. Twelve images account for half the errors. Forty-nine images and twenty complete structure groups had perfect IDs at their matched coordinates. The weakest single view by number of wrong IDs was `test-adjacency-26-left`, with 129 wrong among 344 positioned blocks. [Error counts for every structure, image and ID](errors.json)

| True material | Exported material | Wrong IDs, fixed 104 |
| --- | --- | ---: |
| Polished andesite | Stone bricks | 234 |
| Tuff | Light gray wool | 147 |
| Polished deepslate | Tuff | 137 |
| Stone bricks | Polished andesite | 103 |
| Smooth stone | Stripped pale oak wood | 101 |
| White concrete | Iron block | 98 |
| White concrete | Polished andesite | 93 |

White concrete, tuff, polished deepslate and smooth stone together caused 840 of the 1,499 errors (56.0%). The 72 development views contained no matched coordinates of these four materials. The 28 independent controls included only 17 distinct truth IDs, versus 37 on the fixed 104, and contained no white concrete. On the fixed 104 all wrong ID pairs had different calibrated opaque tile art on every like-facing side. Small projections, lighting and imperfect geometry can still make those textures look alike in a source image. [Per-split material confusions](confusions.json)

A follow-up experiment should use separate development structures with white concrete, tuff, polished deepslate, smooth stone and their common lookalikes together. Vary orientation, shade and apparent block size. Compare aligned texture rows and boundaries across neighboring faces before extending a palette decision. This tests the largest errors while preserving the 104-view evaluation and these frozen predictions for comparison.

## Reading the scores

- **Geometry F1** measures visible occupied coordinates. A single whole-build translation comes from the frozen geometry benchmark. Both material methods use that translation for all scores.
- **Material accuracy** measures exact namespaced block IDs among placements at correct visible coordinates. It includes correctly exported fallback IDs if smooth stone happens to be the truth. The separate evidence counts tell how many such coordinates the matcher actually classified. Other block states, such as a log's axis, are outside this block-ID score.
- **Joint strict F1** requires both the correct coordinate and the correct block ID. A wrong ID on a true coordinate adds one miss and one extra. A correct guess elsewhere cannot offset it.
- **Joint facade F1** applies the same rule to the established facade-priority target. The existing optional visible horizontal slivers are neutral when placed at their true coordinates, regardless of ID. Any placement outside the visible reference is still an extra.
- **Assignment coverage** is `assigned / (assigned + fallback)` across all emitted coordinates. Among those correctly located, `assigned_correct / assigned_positions` measures the accuracy of decisions supported by matcher evidence.

The fixed set includes 104 images of 40 distinct structures. The 100 other views include 72 training views used to choose the matcher, and 28 additional controls. Source image hashes and visible geometry validation tie each prediction to its original capture. All aggregate ratios pool block counts before division. The four distinct structures named Facade keep separate shape hashes.

## Inspect or replay

[Scores by structure and split](summary.json) · [Every camera view](cases.json) · [Per-material confusion counts](confusions.json) · [Final method comparisons](variants.json) · [Training-only iterations](development.json) · [Training candidate source snapshots](development-matchers.zip) · [Typed Minecraft packs](typed-exports.zip) · [Overlay sites and material evidence](sites.json.gz)

`cases.json` includes the source SHA-256, geometry pack SHA-256, typed pack SHA-256, selected translation and all three score groups per view. In `sites.json.gz`, each frame has rows of `[local_x, local_y, local_z, namespaced_block_id, evidence]`. These coordinates are directly comparable to the decoded Minecraft templates. Each confusion triple is `[true_id, exported_id, count]` at matching positions. The diagonal contains the correct IDs. `summary.json` fingerprints the checked release files, matcher, projection code, atlas loader, texture sheet and block index.

From a checkout with the fixed source data and released atlas available, run the following commands. The first checks the 204 source hashes and reads the archived final geometry. The second runs prediction from the resulting source-only records. Worker counts can be adjusted independently.

```powershell
python -m benchmark.typed_structures prepare --workers 6
python -m benchmark.typed_inference --split all --methods face palette blend auto --workers 8 --out runs/typed-structures/selected
python -m benchmark.typed_structures release --predictions runs/typed-structures/selected --out runs/typed-structures/replayed
python -m benchmark.typed_structures verify --out runs/typed-structures/replayed
```

The release script also accepts repeated `--development-run` options for scored training directories. Use `python -m benchmark.typed_structures development --predictions <training-directory>` to create their summaries. The independent verifier checks the archive inventory, hashes, IDs, fallback markers, complete geometry and per-image material scores.
