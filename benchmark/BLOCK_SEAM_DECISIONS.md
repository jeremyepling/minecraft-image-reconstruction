# Which decisions help locate visible blocks?

**Matching the visible texture to the vanilla atlas was the strongest way to locate individual faces once spacing was known.** On six regression buildings it found 35 additional accurately aligned block faces. All 35 came from Courtyard and Portico. Its gains across the full suite came entirely from surfaces where the ordinary counter had already measured both horizontal and vertical spacing. The learned seam detector was the only method to recover an exact size on a surface that had lacked a complete candidate. [Visual comparison](reviews/block-seam-01/comparison.jpg) · [Numerical results](reviews/block-seam-01/summary.json)

## The comparison

We used the 56 vanilla Java 26.3 screenshots, frozen image-derived Surface Finder outlines and straightened crops from the [perfect chooser test](GRID_CHOOSER.md), revision `7189cb7`. Shopfront and Wing are development buildings. The other six are previously reviewed regression buildings, covering 36 views and 442 eligible surface appearances. Every method received one screenshot's evidence. Recorded dimensions, corners and world positions entered the scorers after prediction. Weights for the learned method came from the separate Visible Builds training split.

| On the six regression buildings | Exactly sized surfaces | Wrong sizes | Unresolved sizes | Aligned / proposed face crops | Correct / extra exported coordinates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current Block Counter | 78 | 37 | 327 | 82 / 83 | 122 / 18 |
| Visible outline edges | 78 | 37 | 327 | 100 / 101 | 122 / 18 |
| Learned seam detector | 80 | 38 | 324 | 106 / 110 | 122 / 18 |
| Atlas texture registration | 78 | 38 | 326 | 117 / 118 | 122 / 18 |

A correct size requires the exact width and height of the visible surface. The table counts wrong guesses on matched outlines. There were also 10 complete guesses on unmatched outlines for the default, outline and learned methods, and 11 for the atlas method. An aligned face is a fully supported predicted block footprint paired one-to-one with a fully visible reference face, with at least half intersection over union and all four corners within a quarter block. The 36 regression screenshots contain 7,366 eligible fully visible face appearances. A geometry-only export uses smooth stone and its own camera-derived origin. We decoded the actual structure packs and compared their coordinate sets frame by frame. All 36 regression packs had identical coordinates across methods. [Scores, grids and decoded packs](reviews/block-seam-01)

- **Visible outline edges** kept the existing spacing and looked for two long, separated parallel edges that agreed on the offset. It added 18 aligned faces, all on the low-left masonry Crossing screenshot with the concrete floor. It introduced zero additional alignment failures and changed zero dimensions. [Per-frame results](reviews/block-seam-01/boundaries/placement/cases.json)
- **Learned seam detector** fitted a regularized logistic classifier to 15 RGB line measurements, including color changes and bright or dark joints at three scales. It used 60 screenshots to fit the classifier and 12 Watchpost training-split screenshots to set confidence thresholds. Visible face labels supplied positive examples and internal texture edges supplied negatives. This CPU model added 24 aligned faces but also three additional footprints that failed the alignment test. It got two of the 295 previously candidate-free, usable surface crops exactly right. Both were regression cases. Only one exported build changed on the entire suite, a Shopfront development image that gained five correct coordinates and three extras. [Training weights and frame hashes](data/learned-block-seam-v1.json) · [Dimension scores](reviews/block-seam-01/learned/dimensions.json)
- **Atlas texture registration** compared candidate spacings and offsets with repeated vanilla texture tiles. A strong correlation and agreement among near-tied matches were required to return an offset. It raised aligned faces from 100 to 151 on all 56 views, while the number failing the alignment test stayed at one. The 51 additional aligned faces came from seven screenshots across three buildings, including repeated views over different floors. Its broad search for missing spacing also added one wrong size on a matched outline and one size on an unmatched outline. The material match supplied evidence for grid registration. We did not score material identity. [Atlas decisions and paired scores](reviews/block-seam-01/template)

## What happened on a new building?

Before scoring the challengers, we selected all four images of Twin Spire layout 1 from the Visible Builds test split (two palettes and two camera angles). The ordinary Surface Finder ran its image-only models on the original framebuffer images. Its 68 proposals matched 58 of 60 eligible surfaces. On these four images every method got **21 sizes right, nine wrong and 30 unresolved**; none produced a complete face crop among 824 eligible fully visible faces. The atlas changed continuous spacing or offset on two proposals. Every image-derived camera reported uncertain scale, and the decoded exports contained zero blocks out of 852 visible block appearances. These four views are one layout in one family. [Fresh-family scores](reviews/block-seam-01/family/scores.json) · [Saved image-only surfaces](reviews/block-seam-01/family/surfaces.zip)

**Recommendation.** Keep the three challengers experimental. Our adoption rule required aligned-face gains on both the regression buildings and the separate family while protecting wrong sizes, failed footprints and exported coordinates. The strongest follow-up is to use atlas registration only for the offset on surfaces with both spacings already measured. That is where all 51 observed gains occurred. The next end-to-end test should give those measured corners to the material sampler or placement stage and grade its actual output. The exporter tested here reads the continuous spacing and obtains its global origin from the camera, so a better local seam offset alone cannot change a world coordinate.

## Replay

Run from this experiment checkout with the repository's Python benchmark dependencies and Java 26.3 atlas. The three fixed-suite commands have independent outputs and can run concurrently. The learned weights are included. To refit, run `python -m benchmark.train_block_seam PATH/TO/visible-builds-v1/data/train`. [Visible Builds download and checksums](../training/DATASET.md)

```powershell
python -m benchmark.block_seam_round --method boundaries --out runs/seam-boundaries
python -m benchmark.block_seam_round --method learned --out runs/seam-learned
python -m benchmark.block_seam_round --method template --out runs/seam-template
python -m benchmark.block_seam_review --boundaries runs/seam-boundaries --learned runs/seam-learned --template runs/seam-template --out runs/seam-review
Expand-Archive benchmark/reviews/block-seam-01/family/surfaces.zip runs/seam-family-surfaces
python -m benchmark.seam_family_check --shard runs/visible-builds-v1/visible-builds-v1-test.zip --surfaces runs/seam-family-surfaces --out runs/seam-family-replay
```

The fixed review stores per-surface predictions, per-frame scores, source fingerprints, and decoded pack archives. The family archive preserves the Surface Finder outputs and original build manifests, allowing a repeat check without running the upstream models again. [Artifact manifest](reviews/block-seam-01/manifest.json)
