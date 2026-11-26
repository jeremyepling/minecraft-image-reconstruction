# Can keeping the region shapes recover more of the building?

In this archived comparison, **B** means the **Rectangle Finder**. It grouped pixels by their predicted surface direction and fitted a four-corner outline. The Surface Finder now keeps region masks. [Current stage names](SURFACE_FINDER.md)

[Eight-building comparison sheet with close-ups](reviews/mask-four-01/comparison.jpg) · [Results for all 56 pictures](reviews/mask-four-01/report.html) · [Run manifest and hashes](reviews/mask-four-01/manifest.json)

## Result

Keeping the model regions recovered **394 of 598 visible surfaces** across the 56 screenshots. B recovered **281**. The improvement appeared in 37 pictures, 14 tied and five lost matches. Region masks covered **3.9%** of the sampled opening area, down from **29.2%**. They also made **206** proposals with no eligible match, up from **140**. Of the 206, 21 matched real surfaces below the scoring size and 85 covered mostly background.

| Method | Surfaces recovered | Correctly assigned surface pixels | Mean edge error on matched surfaces | Real connections recovered out of 606 |
| --- | ---: | ---: | ---: | ---: |
| Current B | 281 | 77.71% | 25.26 px | 88 |
| Regions first | 394 | 84.69% | 6.94 px | 271 |
| Regions + ScaleLSD v2 | 393 | 84.59% | 7.01 px | 264 |
| Regions + DeepLSD | 394 | 84.70% | 6.92 px | 271 |

On the **36 pictures held back from this round’s tuning**, B recovered 196 of 442 surfaces, regions alone recovered 287, ScaleLSD recovered 286 and DeepLSD recovered 287. The largest building-level gains were the portico (67 → 106) and galleries (37 → 66). The registry tied at 21. Each count adds up over views, palettes and available floors. [Every building and picture](reviews/mask-four-01/report.html)

**Where the gain came from.** Region masks kept 253 of B’s 281 matches and recovered 141 surfaces B missed. These included 17 of the 23 surfaces that reached B’s fitter and were rejected, nine of the 12 that B’s rectangle distorted, and 70 of the 175 where the directions alone gave B too little separation. On a plaster shopfront, B submitted a recessed wall with 62% overlap to its fitter, then emitted no rectangle touching that wall. Directions agreed on most of the pixels of 536 surfaces. That number alone overstates how many distinct walls a predictor can find. [Enlarged stage-by-stage trace](reviews/mask-four-01/loss-trace/stages.jpg) · [Recoveries by failure stage](reviews/mask-four-01/recovered-by-stage.json)

**Did better outlines give better blocks?** B exported 129 correct visible block coordinates and regions alone exported 81, out of 10,626 across all views. Both made zero extra blocks. The camera could establish metric scale in only 16 pictures. Given the true camera and planes in a separately labeled diagnostic, B exported 4,056 correct coordinates and 341 extras. Regions exported 4,940 correct and 317 extras.

The masonry wing on smooth stone shows why this matters. The new method found all five eligible surfaces, up from four. Its largest region lacked reliable block spacing. Correct exported blocks dropped from 108 to 10. Keeping the new masks and reusing B’s original measuring charts recovered 102. That places the main loss in this picture in deriving countable geometry from the charts. The masks already contain the wall. [Wing source image](data/camera-round-01/captures/dev-wing-masonry-high-right.png) · [Same view in the comparison sheet](reviews/mask-four-01/comparison.jpg)

**Recommendation.** Use regions alone for the next visible-surface experiment. ScaleLSD lost one match and DeepLSD reduced the average matched boundary error by just 0.02 native pixels. Keep the existing ordinary build defaults. The full building path still needs reliable block spacing and depth on the surfaces the model retains.

**Runtime.** Reprocessing the saved numerical fields took 17.4 s for B and 12.2 s for regions across 56 frames. Refinement added 5.2 s for ScaleLSD or 4.7 s for DeepLSD. In a separate pass over the 52 distinct PNGs, MoGe-2 took 69.4 s and Depth Anything V2 took 45.6 s on the local CPU. Both reproduced the cached numerical arrays exactly. Original line inference took 3.9 s for ScaleLSD on an RTX 4080 and 366.4 s for DeepLSD on a hosted CPU. The different hardware and timing scopes preclude a direct speed ranking. [Per-image times and sample counts](reviews/mask-four-01/report.html)

## How to read the experiment

Each prediction starts with one saved 1440 × 900 screenshot from vanilla Java 26.3, taken at noon in clear weather. The same shapes appear with different palettes, angles and, for the original four layouts, two floor materials. The suite contains 56 frames from eight buildings and 52 distinct PNGs. Four close-up pairs are pixel-identical. Each frame counts once in the picture scores. Shared neural inference time counts each distinct PNG once.

We tuned the mask method on 20 shopfront and wing pictures, froze its settings, and showed its first result before implementing the line challengers. We then applied identical line-refinement rules and a development-only selection rule. The other 36 pictures cover six buildings that had appeared in previous reviews. They measure regression performance on layouts excluded from this round’s tuning. This suite contains no previously unseen building. [Initial mask comparison](reviews/mask-first-01/comparison.jpg) · [Development selection](reviews/mask-four-01/selection.json)

1. **Current B.** MoGe-2 predicts a direction for each pixel. B groups nearby directions, uses the screenshot to suggest subdivisions, fits four corners and filters the proposed rectangles. We replayed the implementation against the archived predictions before using the saved outputs, including its original texture counts. The raw direction preview is a rendering of a numerical field. Its colors name directions, so two separate walls can have the same color.
2. **Regions first.** MoGe-2 directions suggest which way a surface faces. Depth Anything V2 supplies relative depth. Connected regions are found separately for each direction. A sustained depth change can separate parallel surfaces, and disconnected pieces have separate identifiers. The predictor retains pixel masks and their openings. Some regions have an optional four-corner chart for sampling texture and measuring scale. A region without a chart still contributes to the surface score.
3. **Regions plus ScaleLSD v2.** The official model reads the source screenshot and emits segments and junctions. Its 512 × 512 input is scaled back independently in x and y. The wrapper verified native coordinates by running the first image at both coordinate scales. [Official implementation](https://github.com/ant-research/scalelsd) · [Author's checkpoints](https://huggingface.co/cherubicxn/scalelsd) · [Frozen evidence](data/mask-lines-v1/scalelsd/manifest.json)
4. **Regions plus DeepLSD.** The official detector with its generic MegaDepth weights reads the same source screenshot. The long side is resized to 960 pixels. Endpoints are mapped back into the original screenshot coordinates. Nearby endpoints of nonparallel segments supply candidate junctions. [Official implementation and generic checkpoint](https://github.com/cvg/DeepLSD) · [Frozen evidence](data/mask-lines-v1/deeplsd/manifest.json)

Both line methods can move a boundary between two existing masks if the line follows that boundary and the directions or relative depth on its two sides support a separation. A line crossing a single flat wall provides no such evidence. The refiner moves at most three samples on the 720-wide mask. It preserves the union of occupied pixels, region connectivity and holes. It tests boundary refinement. Opening coverage and the number of proposals therefore stay equal across the three mask methods by construction. The enlarged line panels highlight pixels that changed owners in pink.

## What the scores measure

A surface is one connected set of visible, coplanar block faces. Repeated views of a wall count separately. A reference surface enters the main score with at least four visible block faces and 24 scoring samples. One predicted surface can match one reference surface at overlap of at least 50% of their combined area. The report distinguishes unmatched predictions on small real faces from fragments, mixed regions and mostly-background proposals. Pixel ownership includes failures to recover the corresponding surface.

All methods use the same scoring raster, with one sample per four original pixels in each direction. For boundaries and corners, the scorer traces the perimeter, including holes, and measures distances in original screenshot pixels. It reports both distance on matched surfaces and coverage across all surfaces. A claimed connection occurs where two predicted regions touch in the picture. The scorer checks whether the corresponding recorded planes meet there in 3D. The opening test samples empty space that has building pixels above and on both sides, including gaps open to the floor.

The downstream check reuses the seam-fit camera, texture repetition, shared-edge depth calculation and Minecraft structure exporter. Each method supplies its own charts to the same code. We record abstentions when camera, scale or plane depth remain unresolved. For block occupancy a full visible cell needs at least four supported samples, and a hole vetoes that cell. Texture counting still samples the chart. The recorded camera and states enter after export to align the arbitrary world origin and score the coordinates actually decoded from the pack. A separate diagnostic supplies the true camera and the plane of a matched region. All exported blocks use smooth stone so the score measures geometry. This replay does not test materials or live placement. [Block results](reviews/mask-four-01/blocks.json) · [Exported packs](reviews/mask-four-01/block-packs.zip)

## Repeat and review

Run `.\Invoke-MaskRound.ps1` in this checkout. It verifies source images and line evidence, reuses numerical normals and depth, runs all four predictions, scores the held-back buildings and exports the selected geometry. It writes a new run directory each time. The manifest records the methods, checkpoints, hashes, environment, scores and artifact paths. [Cache and environment](MASK_REPLAY.md)

A second offline run independently reproduced all 224 method-by-picture predictions byte for byte, as well as the scores and selection. Decoding the two sets of 280 packs produced the same coordinates and block states for every frame and diagnostic. Each cached replay took about 171 seconds on this PC. [Independent verification record](reviews/mask-four-01/independent-verification.json)

Visual feedback can be recorded with `python -m benchmark.camera_feedback`. Use the run directory, image ID, floor, native-pixel rectangle, stage (`regions` or `block-placement`), preferred method and note. Entries go in the run’s feedback journal alongside the measured scores. [Review images and crop coordinates](reviews/mask-four-01/review-regions.json)
