# Reading a surface from its outline

The [pipeline sheet](reviews/perimeters-01/comparison.jpg) is one portrait image you can zoom into. Read down a building's column to follow its original screenshot, detected outline, context crop, straightened surface, predicted block grid and atlas lookup. Read across a stage to compare all eight buildings. The Wing column also follows a second surface with a located grid and a sampled block. Dark areas in the straightened sample have no accepted pixels. “Grid origin unknown” means the counter can estimate spacing but cannot locate the first block boundary.

## Where the three diagonals came from

The [surface-detection sheet](reviews/perimeters-01/surface-detection.jpg) shows every prediction and every recorded visible building surface in these eight pictures. Green regions were found, amber ones were missed, and pink marks predictions left without a match. The recorded geometry contains **132 visible surfaces**. The finder recovered **75 of 107** in the regular score and paired **2 of the 25** smaller surfaces in a separate full-resolution check. It proposed 101 regions, leaving 24 without a match after both checks. The bottom two rows compare the same area for the largest scored miss in each picture. Wing shows its weakest match because it recovered all five scored surfaces. [Surface pairings and close-up coordinates](reviews/perimeters-01/surface-detection.json)

We verified the original PNG hashes and reused the same native screenshot crops. The previous count sheet drew a four-corner sampling chart on top of a pixel mask. The generated line joined distant corners across steps and recesses. The new Surface Finder emits the actual perimeter and a tight transparent copy of the full surface. It preserves every hole, step and disconnected component in its prediction. Recorded geometry enters only during scoring and when explicitly shown in gold. [Enlarged comparisons](reviews/perimeters-01/edge-closeups.jpg) · [Individual crops and scores](reviews/perimeters-01/report.html)

| Same pictured surface | Old chart treated as the outline | Original region pixels | Current Surface Finder | What remains |
| --- | ---: | ---: | ---: | --- |
| Courtyard rear wall | 72% | 95% | 96% | The outline follows the exposed wall around both foreground blocks. Mean border error is about two source pixels. |
| Workshop stepped front | 37% | 51% | 51% | The steps are visible. The finder includes the white front one block behind the gable. |
| Registry upper wall | 77% | 79% | 79% | The finder includes part of the footing one block behind the wall. |

Overlap means intersection divided by union for one predicted and one recorded surface. The four-corner chart in the first column was the drawing on the sheet. In the earlier method the scorer already used the pixel mask. Tracing the mask fixes the drawing. Only a change in pixel ownership can separate a second wall.

The original chart left out about 12%, 18% and 2% of the three respective predicted masks. Full-resolution Finder crops now retain all owned pixels. The straightened counting sample still has its own chart. The counter could not determine both dimensions on any of these three highlighted surfaces.

## How well did Surface Finder improve?

| Method on the same 56 screenshots | Surfaces found out of 598 | Mean border error | Correctly assigned surface pixels | Both block counts right |
| --- | ---: | ---: | ---: | ---: |
| Original Regions First | 394 | 6.94 px | 84.69% | 57 |
| Align nearby borders to image contrast | 398 | 6.83 px | 84.77% | 56 |
| Split parallel walls with a supported depth step | 406 | 6.27 px | 86.12% | 59 |

The development rule selected contrast alignment on the 20 shopfront and wing pictures. This is the ordinary Surface Finder refinement for this branch. Depth splitting gave the strongest overall result, driven by the other six previously reviewed buildings. It kept the two named workshop and registry walls joined on these exact views. Each alternative ran independently on the same frozen outputs and screenshots. The original MoGe-2 directions and Depth Anything V2 depth supplied the region proposals. Both refinements are deterministic. The depth challenger uses changes in relative depth and corroborating image edges. [Development choice](reviews/perimeters-01/selection.json) · [Scores by picture](reviews/perimeters-01/report.html)

## Should the crop be larger?

The source crop now includes a separate RGB view with **eight native pixels per side** by default. You can set `--surface-context-px` on an ordinary surface or count build. The tight transparent crop identifies owned pixels, and the padded RGB shows adjacent scenery at its photographed brightness. Measurements for the 0, 4, 8 and 16 pixel review margins cover every matched surface. Eight pixels exposed 1,875 more missed target samples than the tight box. Widening from 8 to 16 exposed another 227. Each scoring sample represents a four-by-four area in the native screenshot. [Crop measurements](reviews/perimeters-01/context-margins.json)

We also tested whether the extra pixels help estimate the grid. With the same surface masks and counter, the tight crop gave **56 correct sizes**. Allowing similar-colored neighboring pixels to vote gave **55 at 8 pixels** and **55 at 16 pixels**. Using the same expanded charts with the added RGB hidden gave 57 at either size. At 8 pixels the number of matched surfaces with both grid origins located stayed at ten. Keep the context for inspection and keep measuring the owned pixels. [Context evidence and controls](reviews/perimeters-01/count-context/scores.json)

## Repeat or inspect a single image

From this checkout run:

```powershell
.\Invoke-SurfaceRound.ps1 -Round perimeters
```

The replay checks the 56 saved Java 26.3 frames, their model-derived data and reference geometry. It reuses the versioned atlas from the vanilla client and the direction/depth cache where available. The benchmark dependencies are in `benchmark/suites/normal-requirements.txt`. For an individual build, the [stage guide](SURFACE_FINDER.md) covers model setup and the ordinary build command. Each run writes to a fresh output directory. [Run manifest](reviews/perimeters-01/manifest.json) · [Native coordinates and individual stage results](reviews/perimeters-01/pipeline-rows.json)
