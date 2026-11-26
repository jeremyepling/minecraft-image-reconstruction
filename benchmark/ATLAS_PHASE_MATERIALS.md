# Do better block corners produce the right material IDs?

**Yes, on the walls where the counter could already measure spacing.** The restricted atlas method added 51 correctly located and named face appearances. It kept every previously correct face and added zero incorrect block IDs. With production revision `a12741d`, exact material identifications rose from 103 to 154 across the 56 saved screenshots. The atlas located the first seam; Minecraft's ordinary Texture matcher named the resulting face crop. [Comparison sheet](reviews/atlas-phase-materials-01/comparison.jpg) · [Production-revision scores](reviews/atlas-phase-materials-01/current/scores.json)

| Screenshots | Eligible visible faces | Ordinary crop + Texture | Atlas offset + Texture |
| --- | ---: | ---: | ---: |
| Development, 20 views, production revision | 3,072 | 19 | 35 |
| Six regression buildings, 36 views, production revision | 7,366 | 84 | 119 |
| Twin Spire variant 1, four test views | 824 | 0 | 0 |
| Cloister variant 1, four validation views | 644 | 0 | 0 |

The last two rows are prespecified families from Visible Builds v1. We ran the pinned experiment's image-only Surface Finder on their original images. The production-revision check used the same fixed 56 screenshots as the previous round with the newer verified outlines and engine. These are paired replays of 56 images, plus eight distinct family images. All were clear-noon vanilla Java 26.3 captures.

## What the atlas changed

The method accepts a rectified surface only if the ordinary counter already has both positive, finite spacings and at least one missing grid offset. It registers visible RGB against the pinned game atlas at those spacings. It fills a missing offset only when sufficiently strong texture matches agree on the location. Measured spacings, known offsets, visible support masks and the material scorer are fixed between arms.

On the preceding round's outlines at `d95c350`, 155 of 714 proposed surfaces already had both spacings. The restricted method located 22 additional axis offsets. This raised correctly located and named faces from 100 to 151, including 82 to 117 on the six regression buildings. An independent run through the production count and material code verified face order, grouping, chosen source face, corners and IDs of the ordinary arm. A separate check decoded all 56 fixed-suite and four Twin Spire geometry-only packs and found identical coordinate and state sets. The measured gain is local face geometry and material identity. [Independent material control](reviews/atlas-phase-materials-01/control/summary.json) · [Geometry and export control](reviews/atlas-phase-materials-01/geometry-control.json)

All 51 gains came from seven views of three buildings. Wing added 16 stone-brick appearances, Courtyard added 18 stone bricks and six spruce planks, and Portico added 11 stone bricks. Two pairs of close-front floor variants have byte-identical PNGs. Deduplicating by screenshot hash, face direction and world position leaves 34 added face identifications across five distinct images. On `a12741d`, each gain matched the same frame, position, facing and block ID as on the earlier outlines. One named crop missed the geometry tolerance with either method. Exact identifications require a one-to-one match to a fully visible face, at least 0.5 intersection over union, all four corners within a quarter block and the exact top-choice block ID. Unsupported guesses and missed faces are reported separately. [Per-face predictions](reviews/atlas-phase-materials-01/current/cases.json)

The matcher grouped supported faces, selected the widest source face with a five-pixel minimum, and propagated the chosen ID to other faces in the same block group. Running the simpler atlas-only classifier on the identical crops gave 13 → 64 exact IDs on the production revision, with 90 wrong IDs among aligned faces in both arms. The Texture matcher gave 103 → 154 with zero wrong IDs on aligned faces. The main comparison uses Texture. [Matcher settings and asset hashes](reviews/atlas-phase-materials-01/current/manifest.json)

## Why the family check matters

Both methods produced zero complete crops on the eight Twin Spire and Cloister views. On Cloister the finder proposed 39 surfaces and the counter measured both spacings on 11. The atlas increased surfaces with both offsets from one to three; none yielded a whole cell inside the observed mask. On Twin Spire, 30 of 68 proposals had both spacings, and the atlas raised the number with both offsets from two to four. Those also yielded zero complete cells.

To isolate the material stage, we sampled eight visible reference faces per frame independently of the predictions and ran the same matcher with recorded corners. Texture got **27 of 32** on Twin Spire and **29 of 32** on Cloister. These are answer-assisted diagnostics. In ordinary single-image use the matcher must first receive a supported crop. [Twin Spire results](reviews/atlas-phase-materials-01/twinspire/scores.json) · [Cloister results](reviews/atlas-phase-materials-01/cloister/scores.json)

Across the seven surfaces with both offsets, **21 candidate cells fit inside the rectified charts**. Every cell had some pixels outside the usable mask. As a geometry diagnostic we compared their predicted corners to the recorded visible faces, without feeding the rejected crops to the matcher. A 90% supported-pixel cutoff would admit 20 cells, of which only eight matched the correct corners. At 98% it would admit five, all misaligned or unmatched. The supported fraction alone is a poor way to decide whether a partial crop belongs to one block. [Per-cell support and corner check](reviews/atlas-phase-materials-01/family-coverage.json)

**Keep this as an opt-in experiment.** It has a repeatable benefit for the pictured stone-brick and spruce-plank walls. The next useful test is to score candidate corner alignment for partially visible cells before passing their supported pixels to the material matcher. The 21 recorded examples supply an initial check, including nearly full crops with the wrong corners.

## Try it on a screenshot

From the experiment checkout, use a Java 26.3 screenshot, a calibrated atlas and the repository's usual benchmark Python environment. The first command creates image-only surfaces. You can reuse a saved Surface Finder run for that exact screenshot. The second command writes `materials.json` with continuous grid measurements, original and refined offsets, projected face corners and namespaced block IDs.

```powershell
.\Invoke-WallRecovery.ps1 -Image .\wall.png -Through surfaces -Out runs/wall-surfaces
python -m benchmark.phase_material_image wall.png runs/wall-surfaces --out runs/wall-materials
```

An optional `--atlas PATH` selects the calibrated 26.3 atlas directory. We ran both commands on the original low-left Portico masonry screenshot. All five predicted faces were stone bricks, with the same corners and IDs as the fixed-suite verifier. [Saved example](reviews/atlas-phase-materials-01/portico-example/materials.json)

## Replay the tests

Download the validation and test shards of Visible Builds v1 to `runs/visible-builds-v1`. Each new evaluation directory must be unused. To repeat the compatibility test, use a checkout of the recorded production revision with its verified photo-edge archive. [Dataset details](../training/DATASET.md) · [Evidence manifest](reviews/atlas-phase-materials-01/manifest.json)

```powershell
python -m benchmark.phase_material_round --out runs/materials-fixed
python -m benchmark.phase_material_round --suite twinspire1 --shard runs/visible-builds-v1/visible-builds-v1-test.zip --surfaces benchmark/reviews/block-seam-01/family/surfaces.zip --out runs/materials-twin
python -m benchmark.phase_material_round --suite cloister1 --shard runs/visible-builds-v1/visible-builds-v1-validation.zip --surfaces benchmark/reviews/atlas-phase-materials-01/cloister-surfaces.zip --out runs/materials-cloister
python -m benchmark.phase_material_current --checkout PATH/TO/a12741d --out runs/materials-current
python -m benchmark.phase_material_sheet runs/materials-fixed/cases.json runs/materials-sheet
python -m benchmark.phase_crop_coverage runs/visible-builds-v1 benchmark/reviews/atlas-phase-materials-01 runs/family-coverage.json
```
