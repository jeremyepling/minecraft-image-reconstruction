# Which surfaces do protected edges and closer views recover?

**Keep the 513-surface finder as the default.** Protecting existing edges found the most surfaces and improved block sizes, but still broke one Crossing wall. Closer views revealed many more surfaces in the model output. Automatically accepting new regions from those views filled real openings. [Zoomable comparison sheet](reviews/guard-zoom-01/comparison.jpg) · [Scores for all 56 pictures](reviews/guard-zoom-01/pictures.md)

| Across 56 pictures | Main default | Protected edges | Closer views |
| --- | ---: | ---: | ---: |
| Surfaces found out of 598 | 513 | 524 | 516 |
| Newly found / earlier matches lost | 0 / 0 | 12 / 1 | 3 / 0 |
| Extra guesses after accounting for smaller visible surfaces | 204 | 218 | 230 |
| Surface pixels assigned correctly | 94.47% | 94.36% | 94.54% |
| Opening samples incorrectly covered out of 16,630 | 601 | 601 | 953 |
| Correct surface connections out of 606 | 418 | 420 | 424 |
| Mean boundary / corner error in source pixels | 4.23 / 13.30 | 4.10 / 13.55 | 4.20 / 13.25 |
| Undersides found out of 18 | 14 | 14 | 14 |
| Surfaces with both block dimensions correct | 80 | 88 | 80 |
| Shopfront and Wing, development | 137/156 | 144/156 | 138/156 |
| Six previously reviewed buildings | 376/442 | 380/442 | 378/442 |

### Protecting an edge between walls

The earlier agreement test used a plane outline when Pixel-wise Planarity and MobileSAM agreed. It found 523 surfaces. It also erased two walls in the high-right plaster view of Crossing. This experiment started with its frozen proposals. Where a proposal took pixels from neighboring walls, the new rule checked the facing direction and sampled relative depth across the shared edge. When those measurements supported keeping the edge, it trimmed the proposal. Each of three settings reached 144 development matches. The frozen choice tested a boundary when a proposal took at least 5% of each neighboring surface. [Earlier 523 comparison](PLANE_ACCEPTANCE.md) · [Development settings](reviews/guard-zoom-01/candidate-runs/guard/development-freeze.json)

The safeguard changed one picture compared with the earlier 523 result. In Crossing it brought R3 back from 46% to 68% overlap, recovering the match. R6 remained at 49% overlap, just below the 50% match threshold. Its earlier outline on `main` reached 90%. The new method kept the Workshop ledge R13 in both low-left pictures. Across Crossing, 82.71% of visible surface pixels were assigned correctly, compared with 91.60% on `main`. [Per-picture surface gains and losses](reviews/guard-zoom-01/candidate-runs/guard/pictures.md)

### Giving the same model a closer look

The ordinary Planarity pass sees a 1,440-pixel-wide screenshot resized to 800 pixels. For each of the 52 distinct screenshots, the test also ran the same checkpoint on nine overlapping crops. Each used roughly half the original frame in each dimension. The full-image camera supplied the surface directions and 3D points used to compare neighboring regions. Crop regions touching an artificial crop edge were excluded from acceptance. The experiment kept the existing rules for adding or splitting a surface. For overlapping crops, each parent received at most one split, chosen by the number of supporting samples. Repeating the acceptance pass with the original whole-image evidence, once or with nine identical copies, reproduced all 513 matches and 80 sizes. [Crop settings and diagnostic controls](reviews/guard-zoom-01/candidate-runs/zoom/manifest.json)

Of the 85 baseline misses, the original Planarity output contained a matching outline for 25. The complete crops had one for 49. Thirty were visible in the crops alone and six in the whole image alone. Of the 51 misses where neither the original Planarity pass nor MobileSAM had a matching outline, **22** had one in the crops. These counts use the recorded geometry to inspect the raw evidence after prediction. [Raw overlaps for each miss](reviews/guard-zoom-01/raw-opportunities.json)

All three final gains came from splitting walls. They were Shopfront R15 and Workshop R1 in two palettes. The 26 newly added regions yielded zero matches. Four additions accounted for all 352 extra covered opening samples. The worst was Registry, where a single addition claimed 260 samples of exposed floor at the base of two walls. In the low-left masonry view of Galleries, the best crop covered the outline of underside R13 with only 3.5% overlap. The next test I recommend is using crop evidence only to split existing regions. Whether that rule keeps the gains and preserves the openings needs its own run. [Actions and opening coverage by picture](reviews/guard-zoom-01/cases.json)

### Read and reproduce the review

Each building has a column. The rows show the original, reference surfaces, raw model outlines, finished predictions and scored matches. The detail rows revisit the same eight close-ups as the earlier acceptance review. Below each close-up, the sheet shows the straightened sample and block grid for each method. It identifies the closest guess when the surface was missed. The final rows show the worst increase in covered opening samples per building, with the exact source view labeled. [Sheet selections and measurements](reviews/guard-zoom-01/sheet-views.json)

```powershell
.\Invoke-SurfaceRound.ps1 -Round guard-zoom
```

This replays both methods, grades their predicted masks and reruns the ordinary texture-based Block Counter. All 56 Java 26.3 clear-noon fixture entries use their original PNGs and scoring labels. Development used Shopfront and Wing. All eight buildings have been reviewed previously. The report measures visible surfaces and block dimensions. It does not measure placed blocks or material IDs.

The original crop inference used 468 crop forwards on an RTX 4080. It took 60.22 seconds for crop inference, 8.81 seconds for full-image geometry, 6.65 seconds for crop grouping and 4.56 seconds to load the model. With cached model evidence, the boundary safeguard took 0.20 seconds beyond the earlier agreement proposals, and processing the crops took 10.65 seconds, summed across the 56 views and excluding scoring. The one-command replay uses the saved numerical fields and runs with Minecraft closed. To regenerate the fields with the local model installations, use `python -m benchmark.zoom_planes_round --out runs/fresh-zoom --workers 3` in a fresh output directory. [Run manifest, code revisions and artifact hashes](reviews/guard-zoom-01/manifest.json) · [Checkpoint and inference timings](reviews/guard-zoom-01/candidate-runs/zoom/crop-fields/manifest.json)
