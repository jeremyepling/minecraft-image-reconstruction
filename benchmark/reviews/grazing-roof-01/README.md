# Native pixels on grazing-angle roof faces

Keep `experiment/grazing-roof-evidence` experimental. The strict texture check recovered seven correct visible blocks in the earlier 104 evaluation views and none on eight independently reserved shapes. The seven gains came from a recess and a single Twin Spires screenshot. None of the seven development canopies gained a block under the strict check.

## Visibility diagnostic

On the seven earlier canopy screenshots, the combined depth and edge baseline at `af24549` missed 235 of the 348 blocks with a visible roof-top face. A missed top face had a median of 59 native source pixels. Only 13 had fewer than 14. Of those 235 targets, 213 had at least 14 true source pixels inside some predicted surface mask. After rendering the baseline geometry, 122 still had at least 14 unclaimed mask pixels. An answer-assisted pass removing all extra coordinates first raised that to 164. This identifies 42 blocks where correcting the foreground could restore usable image evidence.

We intersected the remaining native rays with nearby horizontal planes at detected face heights, including near-horizontal planes at observed block edges. Each proposed block required at least 14 native samples near an existing visible cell. Restricting proposals to masks classified as horizontal reached 50 of the 235 missed roof tops. Including all detected masks reached 114. The expanded set also included 167 other candidate coordinates across these views.

## Decisions tested

We inferred the vanilla sprite from the native-ray UV coordinates at three texture resolutions. Distinct subsets of the observed texels had to agree on one sprite and rotation, and the projected phase had to beat shifted texture samples. On the seven development canopies this initial per-cell check found 30 otherwise missing top cells and 14 incorrect coordinates. We then required a second local tile independently matching the same sprite. The last visibility pass required the candidate horizontal face to own at least 14 of its actual evidence pixels in the combined render.

| Candidate | 72 training images, correct gained / extras added | 12 earlier native transfer images, correct gained / extras added |
|---|---:|---:|
| Allow any detected mask, corroborate neighboring native texture matches | 4 / 13 | 8 / 0 |
| Also require the same pixels to match a horizontal projection more strongly than competing vertical faces | 4 / 2 | 2 / 0 |

The first version gained six canopy blocks across three development screenshots but added an error on hidden structure in training. The stricter version at `435d558` avoided that hidden error. On the separately scored 12 Twin Spires screenshots it added five correct blocks and zero extras. It changed no coordinates on the 12 Cloister, 56 fixed exterior, or 12 earlier supported native views. On the eight fresh shapes, including three canopies, it preserved all 1,070 correct and 139 extra coordinates. Neither rule lost an existing correct block in its scored comparisons.

The existing geometry in front of a roof can block its image evidence during reprojection. On canopy 09 right, 38 of the 39 missing top faces had at least 14 native pixels covered by an inferred mask, but only nine kept that much unclaimed evidence. Examining conflicting depths along those rays is a better next experiment than relaxing these texture thresholds.

[Per-canopy observability and oracle counts](oracle.json) · [Case metrics, hashed coordinates and 280 decoded datapacks](evidence.zip)

## Replay

Use the frozen earlier transfer, original Visible Builds v1, and the combined placement baseline from `review/canopy-placement`. Run `python -m benchmark.grazing_roof_oracle` to reproduce the source-pixel counts. Under `benchmark.grazing_roof_round`, the experimental adapters are `horizontal`, `union` and `oriented`; the latter is the final frozen variant. From this checkout:

```powershell
python -m benchmark.visible_depth_score --split test --candidate benchmark.grazing_roof_round:oriented --baseline PATH_TO_COMBINED_EDGE_OUTPUT --out runs/grazing-test
python -m benchmark.canopy_compare --inputs PATH_TO_NATIVE_TRANSFER --candidate-root . --candidate benchmark.grazing_roof_round:oriented --baseline PATH_TO_NATIVE_COMBINED_EDGE_OUTPUT --out runs/grazing-transfer
```
