# Native visible surface crop reference

This scorer reviews the 193 active crops from the six Sky handoffs at `52de2c9`. The frozen fixture manifest SHA-256 is `b42a3b350090efa0e4eed7e980120de12ee3fc3d177960afeffbb5953ebeed33`. It contains four of 144 Visible Builds v3 views and two of 40 Decorative Builds v1 views. The fixed 104-view suite was outside this crop comparison.

## Baseline

| Building | Split | Crops | Cube targets | Mean crop IoU | Targets without a primary crop | Targets with no crop overlap | Cube pixels touched |
|---|---|---:|---:|---:|---:|---:|---:|
| Castle | Test | 17 | 73 | 46.6% | 61 | 2 | 96.6% |
| Harbor | Validation | 34 | 80 | 45.4% | 49 | 1 | 82.2% |
| Viaduct | Train | 33 | 113 | 51.2% | 83 | 7 | 93.8% |
| Pillars | Train | 31 | 43 | 53.6% | 19 | 0 | 96.9% |
| Hillside Chapel | Validation | 27 | 21 | 24.2% | 8 | 0 | 97.0% |
| Walled Garden | Test | 51 | 63 | 44.2% | 32 | 0 | 97.2% |
| All | | 193 | 393 | 44.5% | 252 | 10 | 94.9% |

92 crops reached 50% IoU, 46 reached 75%, and zero exactly matched. Pixel-pooled IoU was 32.7%. Extra support summed across the paired crops included 433,709 pixels from a different cube plane or component, 67,832 pixels from partial blocks and decorative quads, and 68,133 background pixels. The last group measures pixels without a visible structure face in the reference. The 193 masks already touched 1,240,623 of 1,306,887 visible cube pixels, including pixels in nonprimary surfaces. Viaduct S01 alone includes 139,575 pixels from neighboring cube surfaces in addition to its best reference. A plane split can recover a missing surface even where its pixels already lie in an active crop.

The exact replacement control substituted the whole visible target for each of 188 positively matched crops and preserved five unmatched baseline proposals. It reached 97.4% mean and 99.5% pixel-pooled IoU, on fixed per-proposal targets. That replacement arm touched 67.6% of all visible cube pixels. Each active crop supplies at most one target in that arm, and 47 positive pairings duplicate a target. Use the control to diagnose the mask handoff on active proposals. A missing proposal, a duplicated pairing, and a new target at the wrong physical offset need separate treatment.

## Target and pairing

The reference uses the native frontmost `face_id` raster, grouped by signed direction, integer normal-axis face coordinate and four-neighbor edge connection in the camera-visible full-cube face grid. This is the grouping used by [the oracle chart generator](../../relative_oracle_charts.py). A target can have separated source-pixel patches because occlusion interrupts its image footprint. Two visible patches separated by a gap in observed world cells stay separate targets, even at equal plane offset. Every saved crop was checked against its source RGB and native support alpha. The two decorative frames contribute 71,190 pixels of non-cube reference faces. Those pixels have known structure ownership and are counted separately outside the connected-cube target; the scorer assigns no exact target for a proposal that touches only such pixels or background.

Each active crop's positive highest-IoU target is fixed for all paired arms, with ties resolved by lowest native face ID. Stable S labels sort by the original support area and break ties by baseline manifest order. T labels use the smallest native face ID per connected component. Per-target union coverage includes overlap from any crop. Each target's best candidate IoU and count of primary predictions are also measured with independent matching to track shifts in which target a candidate best describes. Repeated references contribute once to the union and once per proposal to the paired metrics.

## Replay

With the saved fixture and the original Sky Surface Finder handoffs available:

```powershell
python -m benchmark.sky_crop_reference --output runs/sky-crop-reference-01
```

Open `runs/sky-crop-reference-01/index.html` for all six buildings, the 193 same-coordinate crop rows, details of reference surfaces without a primary crop, and links to the native images. Green shows the intersection, orange shows the saved mask outside the reference, and blue shows reference pixels outside the mask. `report.json` records the resolved pairs, per-pixel counts, input hashes, and separate building, split and dataset results. `exact-target-masks.json` is the reference-only control.

Use `--fixtures` and `--handoffs` to point to restored inputs elsewhere. Score prospective changes with `--candidates name=path/to/candidates.json` and `--scores-only` when the per-pixel figures already exist. Candidate packets use `{"frames":{"frame-id":{"plane-id":{"size":[1440,900],"runs":[[y,left,rightExclusive]]}}}}`. The scorer accepts other raster dimensions and samples them as the actual crop does. Omit a proposal ID to use its active baseline support. Empty runs veto a proposal. Keep label-derived packets and source-only inference separate.

The scorer is an experimental diagnostic. It changes no ordinary build options. The six-frame connected-atlas placement score is carried forward from the floating baseline at 12.64% strict F1 and 12.90% facade F1, with 273 correct positions, 3,768 missed and five extras. The fixed 104-view scores are carried forward at 91.27% strict F1 and 93.42% facade F1. Candidate crop changes require a separate downstream replay. Keep this branch experimental while the crop refinements and downstream effects are measured.
