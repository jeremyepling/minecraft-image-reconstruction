# Placement on 12 additional native Minecraft structures

The production placement method at `659539b` recovered 1,638 of 2,359 visible blocks and placed 859 extras on 12 previously unused layouts. The previous default recovered 1,479 and placed 1,718 extras. Precision rose from 46.3% to 65.6%, recall from 62.7% to 69.4%, and F1 from 0.532 to 0.675. The score improved on 11 of 12 images. Every pack in this test uses smooth stone as its geometry placeholder. Exact material and location matches rose from 23 to 35.

## Selection and evidence

The screen considered saved Minecraft Java 26.3 screenshots from 16 fixture directories named in the runner. Each retained frame had a vanilla resource pack, camera record, ground at Y=64, opaque full-block structure and a readback matching an authored layout page. We excluded RGB duplicates and the occupancy shapes of all 152 Visible Builds v1 frames and the earlier native 70. The shape comparison accounts for translation, cardinal rotations and reflections. Of 140 eligible images depicting 38 distinct shapes, the ordinary image-only camera returned a metric estimate on 40 images of 20 shapes. Within each supported group we sorted by screenshot SHA256, sorted groups by the lowest hash and selected the first image from the first 12 groups before evaluating any placement. This produced seven canopies, two relief structures, one recess, one facade and one bay structure, all using stone-family materials. Historical capture names such as `train-canopy` identify an older shape benchmark. None of these images or occupancy shapes entered the placement fit.

Each candidate received the same screenshot, frozen image-only camera, ordinary Surface Finder masks and Block Counter outputs. The runner used the recorded pose and block states after decoding the datapack to align the local coordinates and score its visible targets. Hidden extras count placements at genuinely occupied source coordinates with no visible face. All other extras also count against precision. The separate native-12 comparison covers the earlier six layouts with a supported camera. Both comparisons measure placement conditional on camera support.

| Method | Fresh correct | Fresh extras (hidden) | Fresh F1 | Earlier native correct | Earlier extras (hidden) | Earlier F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous default | 1,479 | 1,718 (68) | .5324 | 2,760 | 489 (27) | .7853 |
| Texture-ranked planes | 1,601 | 894 (48) | .6597 | 2,909 | 257 (5) | .8376 |
| Ranked planes + partial-face evidence | 1,632 | 865 (48) | .6722 | 2,926 | 256 (5) | .8406 |
| Partial-face method + per-cell tree | 1,624 | 825 (48) | .6755 | 2,828 | 177 (0) | .8336 |
| Partial-face method + single-feature family rule | 1,629 | 833 (48) | .6758 | 2,907 | 246 (5) | .8386 |
| Partial-face method + tree confirmed by zero visible rays | 1,632 | 865 (48) | .6722 | 2,926 | 256 (5) | .8406 |
| Partial-face method + conjunction rules | 1,629 | 748 (48) | .6879 | 2,901 | 193 (5) | .8441 |
| Partial-face method + zero-ray filter | 1,632 | 865 (48) | .6722 | 2,926 | 256 (5) | .8406 |
| Production with secondary plane refinement | 1,638 | 859 (42) | .6746 | 2,926 | 256 (5) | .8406 |
| Production with final zero-ray filter disabled | 1,638 | 859 (42) | .6746 | 2,926 | 256 (5) | .8406 |

The conjunction rules improved aggregate F1 and removed 117 extras on the fresh frames relative to ranked planes with partial-face evidence. They also discarded three correct cells there and 25 on the earlier native images. On the earlier suite a standalone tree applied to the previous default lost 110 correct blocks to remove 95 extras (F1 0.7767). Keep the learned per-cell vetoes experimental. The final zero-ray filter changed zero coordinates across these 24 images. Secondary depth refinement replaced six hidden cells with six visible ones across the two fresh relief images.

The biggest remaining error is the elevated canopy in `train-canopy-09-right`. The previous method found 26 of 112 visible blocks with 264 extras. Production found 22 with 301 extras. Its F1 fell from .129 to .101. At the other extreme, `dev-canopy-18-left` improved from 22 correct with 208 extras to 36 with zero extras. A post-score comparison against recorded camera geometry placed **3,257 of 3,257 visible corners within two pixels** across the 12 images. Per-frame 90th-percentile error ranged from 0.25 to 0.54 px and measured 0.44 px on the regressing canopy. These results support examining surface masks and depth hypotheses on thin elevated faces.

## Replay

The versioned 17,458,053-byte input ZIP contains the exact 12 native framebuffer PNGs, source block readbacks, recorded poses, upstream masks/counts, build manifests and image-only screen. Its SHA256 is `8c8de8a6b0f29d07cd01d76f0df86dcca0772f97f2639b0092bc93566fd40209`. The data does not change the Visible Builds v1 split. [Download native-transfer-v1.zip](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/download/visible-depth-native-transfer-v1/native-transfer-v1.zip)

The compact archive includes decoded datapacks, local and registered coordinates, per-frame metrics, candidate and input hashes, the complete camera screen, post-score corner diagnostics and frozen fit parameters. Its SHA256 is `2102f18d5345e90dc9b80e93528a6e8825175979bb06d8a4d78c1f431c0c149a`. The screen initially named seven columns. The eighth, the zero-ray filter, was added before any placement results were opened. The production replay includes the secondary-depth change developed independently on the other cohorts. [Scoring evidence](evidence.zip)

With the repository's pinned Python dependencies and Java 26.3 atlas available, extract the fixture to a private directory and rerun from a checkout containing the transfer scorer and final engine:

```powershell
gh release download visible-depth-native-transfer-v1 --repo jeremyepling/minecraft-image-reconstruction --pattern native-transfer-v1.zip --dir runs
Expand-Archive runs/native-transfer-v1.zip runs/native-transfer-v1
$env:MCIR_NATIVE_TRANSFER_FIXTURES = (Resolve-Path runs/native-transfer-v1/fixtures).Path
$run = (Resolve-Path runs/native-transfer-v1/transfer).Path
python -m benchmark.visible_depth_transfer --stage score --out $run --candidate-root . --method prior
python -m benchmark.visible_depth_transfer --stage score --out $run --candidate-root . --method production
python -m benchmark.visible_depth_transfer --stage score --out $run --candidate-root . --method production_without_ray
python -m benchmark.visible_depth_transfer --stage camera-audit --out $run
```

To replay experimental scoring, use a candidate worktree at `2609924` for the face reprojection methods, set `MCIR_OFFSET_FIT` to the extracted `fitted/plane-offset.json` and set `MCIR_CELL_GATE_ROOT` to a checkout of `567be25` for its frozen cell gates. Pass `--candidate-root` with the face reprojection worktree and `--method all`. Run IDs and hashes for every adapter are stored alongside its pack scores. The full historical source directories and prior suite are needed only to regenerate eligibility and the initial camera screen; the ZIP supplies the selected images and upstream outputs for placement replay.
