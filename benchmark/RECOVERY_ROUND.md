# Finding the missing surfaces

The most accurate sequence found **471 of 598 surfaces** across 56 screenshots, up from 423 with the Wall Separator. It kept all 423 previous matches, recovered 48 more, and made seven additional guesses that could not be paired with a visible surface. Gains came on 24 pictures covering seven of the eight buildings. [Zoomable comparison](reviews/recovery-overview-01/comparison.jpg) · [Every picture](reviews/recovery-summary-01/pictures.md)

## Which changes helped?

Every row starts with the same Wall Separator. Each scored surface is a connected visible patch on a plane in one picture. A prediction has to overlap at least half the combined area of its mask and the recorded surface in a one-to-one pairing. The suite grades 598 such appearances with sufficient visible area. Smaller surfaces are paired separately before counting extra guesses.

| After the Wall Separator | Development, 20 pictures | Reviewed regression, 36 pictures | Found / 598 | Undersides / 18 | Extra guesses | Both block dimensions right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No additional pass | 114 / 156 | 309 / 442 | 423 | 8 | 182 | 64 |
| Underside Finder with M-LSD Tiny | 117 | 313 | 430 | 14 | 174 | 64 |
| Joined edges with ScaleLSD v2 | 117 | 324 | 441 | 8 | 184 | 68 |
| Wall splits with MobileSAM | 117 | 327 | 444 | 8 | 190 | 68 |
| Joined edges, then underside | 120 | 328 | 448 | 14 | 176 | 68 |
| Joined edges, underside, then MobileSAM | **122** | **349** | **471** | **14** | **189** | **73** |

The three isolated tests answer different questions. M-LSD Tiny’s short strips picked up six real undersides. Joining ScaleLSD lines where they meet let the finder separate walls that share an outline. It gained 19 surfaces and lost one. MobileSAM was the strongest single addition. It found 21 more vertical surfaces and kept every previous match. It could separate a pillar from a wall that faces the same direction behind it. The last two rows are follow-ups using the frozen rules in sequence.

On the eight views in the sheet, look at the Shopfront pillar, where MobileSAM took overlap from 25% to 77%, and the Galleries pillar, which went from 17% to 91%. Courtyard still shows a failure. The recovered outline of its ledge covers only 15% by the same measure. Separate rows show the prediction and the recorded shape at the same magnification. [Enlarged Galleries example](reviews/mobile-walls-01/prediction-crop.jpg)

**Doubling MoGe-2's detail did not help the underside pass.** The same checkpoint, run at a higher token count, dropped from 430 to 429 surfaces and took about 2.3 times as long per image. In one low Portico view it split a previously recovered underside into two pieces covering roughly 40% each. [Paired 56-picture experiment and inference timing](reviews/direction-resolution-01/manifest.json)

## Where did the gains come from?

| Building, all variants | Wall Separator | Full combination | Scored |
| --- | ---: | ---: | ---: |
| Courtyard | 48 | 51 | 56 |
| Crossing | 34 | 35 | 46 |
| Galleries | 68 | 80 | 98 |
| Portico | 109 | 119 | 148 |
| Registry | 23 | 26 | 34 |
| Shopfront | 82 | 90 | 120 |
| Wing | 32 | 32 | 36 |
| Workshop | 27 | 38 | 60 |

The full combination improved correctly assigned surface pixels from 87.7% to 92.0%. Correctly recovered connections rose from 303 to 377 of 606, while total proposed connections rose from 893 to 994. All methods covered 601 of the same 16,630 sampled opening pixels, leaving 12 of 98 gaps at least half filled. Mean border error among matched faces went from 5.56 to 5.71 pixels and mean corner error from 15.73 to 15.99 pixels. Finding more faces helped coverage more than edge alignment. [All scores and layout breakdowns](reviews/recovery-summary-01/scores.json)

Block counting improved from **64 to 73** surfaces with both dimensions correct, keeping every previously correct size. Wrong dimensions went from 30 to 36. The other 489 surfaces had an incomplete or missing estimate. Coordinates and in-game placement require a separate check.

## How each screenshot ran

1. MoGe-2 supplied pixel directions and Depth Anything V2 supplied relative depth. Regions First and the Wall Separator grouped pixels into connected surface masks. Every comparison started from exactly the same saved masks.
2. Three independent image-only additions consumed those masks. The junction test joined short source-image lines and required evidence from appearance and depth before cutting. The underside test looked for supported shaded strips. MobileSAM took automatic point prompts from a grid inside the building and from the initial regions. Its rule divided a wall when two or more confident shapes largely fit inside it and together explained at least 80% of it.
3. The combination ran junctions, undersides and MobileSAM in that order. The MobileSAM prompts still came from the starting wall masks. The original support constrained every split so an outline could not silently fill an opening. The unchanged Block Counter straightened each resulting crop and looked for repetitions.
4. The scorer compared saved predictions with the recorded geometry. It counted newly found and lost identities, including below-cutoff pairings, per picture. The development layouts were Shopfront and Wing. All six other buildings had been viewed in earlier work and serve as regression data. The development selection for the original junction-versus-underside test favored the underside option. The final combination is the best observed across the frozen suite. [Original selection](reviews/recovery-01/development-selection.json) · [Checkpoints, settings and run manifests](reviews/recovery-summary-01/manifest.json)

## How to close the remaining 127

The Wall Separator originally missed 175 surfaces. Of those, 138 were merged with another face, including 116 merged with a parallel face at another depth. After all three additions, **95 of the remaining 127 are still merged, including 75 parallel walls at different depths**. In the filtered MobileSAM suggestions, a correct candidate already exists for 41 of the remaining misses. Of those 41, 38 are among the parallel-wall merges. These counts use the recorded answer after prediction to locate the opportunity. [Original miss audit](reviews/recovery-01/remaining/summary.json) · [Remaining IDs by picture](reviews/recovery-summary-01/cases.json)

1. **Test accepting one supported wall boundary.** The current rule requires two agreeing MobileSAM shapes. Allow one confident shape to cut its parent when the saved depth map shows a consistent step across the shared boundary. Keep the remainder and constrain both pieces to the old wall. Compare against the 471-surface control. Check how many of those 41 become real automatic gains and how many flush walls with a texture change acquire a false split.
2. **Test a sharper depth model on exactly those candidate boundaries.** Swap only Depth Anything V2's boundary vote for Apple's Depth Pro in the preceding test. Keep the MobileSAM suggestions, depth sampling positions, acceptance rule and original surface finder fixed. Compare the ordering of the two sides, surfaces gained or lost, and new false cuts. The authors specifically evaluated sharp depth boundaries. [Depth Pro paper and code](https://machinelearning.apple.com/research/depth-pro)
3. **Test joining adjacent underside fragments.** Use the existing Tiny lines and MoGe directions. Join two touching thin pieces only when their directions agree, their depths continue smoothly, and the union preserves the gap. Grade the four remaining undersides individually and count every pillar gap the merge fills. The Portico detail-resolution failure provides a concrete regression case.

## Replay

Use the optional **Wall Recovery** build when surface accuracy is the priority. It loads five pinned checkpoints. The lighter Wall Separator remains the ordinary default. A direct build of a renamed Galleries screenshot reproduced all 30 saved mask records and the same 19 of 25 matches before passing the surfaces to the Block Counter. [Single-image command and setup](WALL_RECOVERY_BUILD.md) · [Fresh build verification](reviews/wall-recovery-entry-01.json)

From this checkout, regrade every saved prediction, run the Block Counter and regenerate the zoomable portrait sheet together.

```powershell
.\Invoke-SurfaceRound.ps1 -Round recovery-summary
```

The command prints a new run directory with `scores.json`, `pictures.md`, the per-surface results in `cases.json`, and `review/comparison.jpg`. Run `-Round detection-sheet` to redraw the picture alone. Run `-Round recovery` to regenerate the two isolated line refinements from numerical model evidence. [MobileSAM inference and both split replays](reviews/mobile-regions-01/README.md) · [Frozen interaction replay](reviews/junction-underside-01/manifest.json)
