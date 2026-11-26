# Selecting visible surface depths from floor contacts and roof junctions

The Java 26.3 placement experiment on `experiment/canopy-junction-depth` starts from the ordinary placer at `a16668f`. I recommend merging the support-anchored plane search and the two-wall crease rule if their gains survive the combined pipeline. Keep one-wall height overrides experimental. The conservative challenger was frozen at `5219558` before scoring the fixed and newly reserved buildings. The ordinary build on this branch still uses its original entry point. Benchmark adapters select the experiments explicitly.

## What the photograph contributes

On canopy 09 viewed from the right, visible roof pixels extend over empty space in the photograph. Treating their bottom outline as a floor contact placed the primary search for the roof front around local grid plane 24. The correct plane was −5. Several other visible wall masks reached the actual floor, on planes −1 and 3 along the same axis. Ray intersections across each of those contact lines were consistent to a fraction of a block.

1. Project observed vertical-mask footpoints to the estimated floor. Keep a wall contact if at least five samples and 15% of its supported width reach the bottom image silhouette, and 80% of their coordinates land within 0.32 blocks of one integer plane. With at least two such contacts, expand the historical plane candidates around the observed horizontal footprint. Overriding an existing placement outside the original windows requires two distinct projected tiles matching the atlas, improved texture agreement, aligned texel boundaries and a stronger overall visual score. Each tile can show a different material.
2. Check an individual wall's own grounded plane against the existing winner when the original atlas evidence is weak and texel-edge alignment improves. In the conservative variant a weak average atlas score can override the occasional misleading pair of good tiles. Each selected wall passes its existing mask and occlusion checks.
3. Use a vertical wall as a roof-height reference if its visible ground contact or projected atlas tiles corroborate its depth. Where its mask meets a horizontal mask, intersect nearby image rays with the wall. Accept a height observation when at least eight raster samples spanning six sample steps agree to within 0.23 blocks at the 80th percentile. Changing the horizontal height requires two perpendicular wall planes agreeing on the same integer height and weak competing atlas evidence. The same rule applies to visible undersides. It emits only cells supported by the horizontal mask.

On canopy 09 the front and side independently put the top of the roof at level 6. The previous model chose level 5. The full proposal corrected the two front planes to −5 and −4 and recovered 63 visible positions with seven extras, compared with 22 and 301 previously. Seven current extras were added and all 301 old extras were removed. No visible reference face, true camera or block coordinates are inputs to the method.

## Development comparison

Each triplet gives correct positions / extras / extras on occupied hidden structure. Ground truth is limited to blocks visible from the supplied single image. The twelve previous native transfer images are development/regression examples in this experiment. All settings were chosen using these and the existing 72 training views.

| Method | Training 72 | Previous transfer 12 | Canopy 09 correct / extras |
|---|---:|---:|---:|
| Previous default | 13,769 / 2,458 / 160 | 1,638 / 859 / 42 | 22 / 301 |
| Add support-anchored atlas planes | 13,801 / 2,414 / 156 | 1,667 / 344 / 42 | 39 / 47 |
| Also allow a strong footpoint match against a low-vote old plane | 13,801 / 2,414 / 156 | 1,667 / 344 / 42 | 39 / 47 |
| Corroborated creases and weak-art footpoint correction | 13,819 / 2,397 / 138 | 1,691 / 304 / 42 | 63 / 7 |
| Allow a single wall to set height | 13,934 / 2,171 / 128 | 1,711 / 225 / 42 | 63 / 7 |

The two-wall variant gained 51 correct coordinates and lost one on training, all on Arc Bridge. On previous transfer it gained 53 with none lost, adding seven extras and removing 562. Although the single-wall variant raised pooled training F1, it lost 115 previously correct coordinates and introduced 170 extras as it corrected other parts of the builds. An additional experiment requiring corroborating photo or outline evidence for a lone wall reproduced the conservative result on both development sets.

## Frozen comparison

Across Cloister 12, Twin Spires 12, fixed exteriors 56 and the earlier supported native 12, every exported coordinate matched the previous default. Eight independently chosen supported native shapes had 1,431 visible targets. The previous default placed 1,014 correctly with 482 extras, including 21 on hidden structure. The conservative challenger placed 1,066 correctly with 174 extras, including six on hidden structure. Its 58 gains and six losses raised pooled position F1 from 69.3% to 79.8%. All three fresh canopies improved. The six lost correct blocks were on Colonnade, which gained 32 and removed 51 extras overall.

## Replay

[Frozen development, regression and independent scores, coordinates and datapacks](scoring.zip) · [Input restoration for Visible Builds v1 and native geometry](../visible-depth-inputs-01/README.md) · [Previous native transfer fixture](../visible-depth-transfer-01/README.md)

From this branch, select `engine.junction_depth:creases` with the frozen placement scorer and compare against the previous ordinary placer. The other adapters are `art`, `contacts`, `single` and `supported`.

```powershell
python -m benchmark.visible_depth_score --candidate benchmark.visible_depth_score:integrated --split all --out runs/previous
python -m benchmark.visible_depth_score --candidate engine.junction_depth:creases --split all --baseline runs/previous --out runs/junction
```

Run both commands with `--split native` for the earlier native suite. For transfers, run `python -m benchmark.canopy_compare` from the coordinator's `review/canopy-placement` checkout, with `--candidate-root PATH_TO_THIS_BRANCH --candidate engine.junction_depth:creases --inputs PREPARED_TRANSFER --baseline PREVIOUS_DECODED_RESULTS --out PRIVATE_OUTPUT`. The output includes SHA256 checksums and the individual decoded coordinates.
