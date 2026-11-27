# Smooth wall boundaries against pavement

The Surface Finder tests a connected near-uniform patch next to an existing mask. It compares its source colors and contrast with all 448 registered Java 26.3 atlas sprites under variable shading. A second, uncensored 8 × 8 window must agree inside the existing mask. If an equally smooth patch at the same height along either frame margin also matches the source color within three levels per RGB channel, the finder abstains. This classifies local smoothness; several materials can explain the same color. Thirty-three atlas sprites have at most three levels of within-tile standard deviation on every channel.

The selected rule leaves a two-pixel safety margin around the outside of the combined host and proposed pixels. Shared edges stay joined. The new mask controls the saved outline and source crop. Each edited surface also stores its original measured support for block counting and placement. If the atlas is unavailable, the existing mask passes through. [Implementation](../../../engine/smooth_atlas_outline.py)

## What did the atlas add?

| Fixed 104, added pixels | Correct host | Neighbor | Background |
| --- | ---: | ---: | ---: |
| Raw matching color | 84,401 | 20 | 312 |
| Atlas check on patch and host | 82,923 | 20 | 0 |
| Atlas plus same-height margin sample | 82,923 | 20 | 0 |
| Atlas plus margin sample and two-pixel interior | **77,329** | **0** | **0** |
| Earlier broad same-height chromaticity veto | 15,542 | 0 | 0 |

On the 100 transfer views, the raw proposal included 580 correctly owned and 2,449 background pixels. The atlas alone kept 67 correctly owned and 507 Watchpost background pixels. Comparing the same-height margin removed the 507. The two-pixel interior retained seven correctly owned Stepped Hall pixels. The earlier broad veto retained 569 correctly owned transfer pixels, including 502 on Offset Gate that the atlas rejects because a textured sprite fits better. Applying the atlas after the broad veto also discards the strong Registry and Shopfront gains blocked by that veto. [Atlas comparisons and fingerprints](summary.json) · [Per-proposal competing materials and abstentions](cases.json)

The one- and two-pixel interior checks were designed after inspecting the remaining wrong-neighbor pixels. One pixel still includes four on Shopfront. The two-pixel choice adds 77,329 correctly owned pixels across 14 fixed views and seven in one transfer view. Across 204 views it recovers two one-to-one surface matches at native IoU ≥ 0.5, loses none, and reduces unmatched guesses by two. Correctly paired area rises by 94,193 pixels on the fixed set because two original masks now also qualify for pairing. Median diagnostic time for the proposals, atlas checks and three interior variants was 0.47 s per view (95th percentile 1.11 s with five concurrent workers, excluding upstream surfaces and scoring). [Current-case grades, source hashes and native matched IDs](interior.json)

| Named layout | Views | Views changed | Correct pixels added | Local result |
| --- | ---: | ---: | ---: | --- |
| Shopfront | 8 | 3 | 9,567 | 2,615 pixels in the central 60 × 55 plaster low-left patch; no ground or neighboring surface included. |
| Registry | 4 | 2 | 23,969 | 15,393 pixels added on the plaster high-right wall. The finder previously missed 15,833 there. |
| Portico | 8 | 2 | 3,641 | One additional correctly matched visible plane. |
| Courtyard | 8 | 0 | 0 | The nearby host sample favors a textured sprite. The finder abstains from a proposed 65-pixel extension. |

Two public surface builds verified the repaired Shopfront and Registry rings, source-crop alpha and stored original supports. A fresh public Registry placement build reproduced all measured dimensions and all 241 exported block positions and material IDs. This outline change restored the full prior measurement masks on all 204 replayed views. The prior 104-view placement score carried forward is 90.8126% strict F1 (20,619 correct, 3,083 missed, 1,089 extra) and 92.9145% facade-priority F1 (18,798 correct, 1,778 missed, 1,089 extra). One placement export was rerun for this outline experiment. [Public CLI verification](ordinary.json)

## Replay

Run these commands from the experiment checkout with the frozen inputs available. The first command reuses the independently frozen full-image proposal manifests and their source hash checks. The second computes fresh proposals on all 204 images after the default eave repair and jagged joins.

```powershell
python -m benchmark.smooth_boundary_atlas --proposal-checkout ../shopfront-outline-bites --out benchmark/reviews/smooth-boundary-atlas-01
python -m benchmark.smooth_boundary_interior
python -m unittest tests.test_smooth_atlas_outline tests.test_reconstruction_cli
```

For the public checks, run `Invoke-WallRecovery.ps1` on the two frozen plaster source images named in `ordinary.json`, using `-Through surfaces` and a prepared 26.3 atlas. Run Registry once more with `-Through placement`. Supply their three output directories to `python -m benchmark.smooth_boundary_public`.
