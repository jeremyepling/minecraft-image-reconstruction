# Where the Surface Finder merges visible faces

Open the [standard detection sheet](reviews/perimeters-01/surface-detection.jpg) for the counts, then the [evidence close-ups](reviews/overhang-01/evidence.jpg) to inspect a miss. Both use the same eight screenshots in the same order. The close-ups separate the real surface, three proposed outlines, the two raw line detectors, surface direction and relative depth. When the target is a thin strip, the white box locates an enlarged stretch. The overlap percentage always grades the entire surface.

## What was merged?

Of 32 scored misses on the standard sheet, 29 are vertical faces, two are tops and one is an underside. For 28 of the 32, one proposal covered at least half of the missed surface and half of another scored surface. For 23 of those, a second surface faced the same way at a different depth. The eight views show three scored undersides and two smaller ones. Six views show no underside at all. [Miss measurements](reviews/overhang-01/misses.json)

Across all 56 saved pictures, there are 598 scored surfaces, including 18 undersides. There are 18 more undersides below the regular scoring cutoff. None of the six methods matched one of those smaller undersides in the separate full-resolution check.

| Saved outlines | All scored surfaces found | Scored undersides found | Found that the current finder missed | Lost that the current finder found |
| --- | ---: | ---: | ---: | ---: |
| Current finder, regions with contrast alignment | 398 | 8 | — | — |
| Regions alone | 394 | 8 | 0 | 4 |
| Regions with ScaleLSD | 393 | 8 | 0 | 5 |
| Regions with DeepLSD | 394 | 8 | 0 | 4 |
| Regions with depth splitting | 406 | 8 | 12 | 4 |
| Earlier B rectangles | 281 | 3 | 28 | 145 |

All five region methods found exactly the same eight undersides. Picking from all six sets of finished outlines with the recorded answer in hand would find 436 surfaces. The 38 additions over the current finder are 36 vertical faces and two tops. That is a ceiling for choosing among existing matches. Adding all guesses would also add duplicates and false surfaces. [Per-surface pairings and overlaps](reviews/overhang-01/results.json) · [Split and orientation counts](reviews/overhang-01/summary.json)

## Three places to inspect

- **Galleries, masonry low-left, R13.** The recorded underside occupies 2,559 pixels. None of 144 sampled MoGe directions pointed within 30 degrees of downward. The longest shared seam has a median brightness difference of 33 on a 0–255 scale. DeepLSD followed 606 of 606 tested positions on that seam, ScaleLSD 588. Along *all* shared borders of R13 they followed 76% and 67%, respectively. The closest finished outline from any method had only 9% overlap.
- **Portico, masonry high-right, R17.** The current outline includes a neighboring face. Splitting on a supported depth change raised overlap with this vertical side face from 27% to 87%.
- **Workshop, plaster high-right, R4.** MoGe got the direction right on 99% of sampled pixels. The front a block forward has the same direction, and both ended up in one prediction. A fitted B rectangle scored 73% overlap with R4, compared with 45% for the current finder. Its rectangle also covers the door opening, which is why it makes a better hint for a dividing edge than a replacement mask. [Original crops and predictions](reviews/overhang-01/evidence.jpg)

## Are the raw lines worth testing?

The previous line refinements moved an existing border between two masks. They could not introduce a missing surface. We measured the raw lines in source pixels wherever an underside borders another visible building surface. A sample counts when a segment is within three pixels and 15 degrees of the locally straight seam. Silhouettes against the sky or ground and ambiguous corners are excluded.

| Raw evidence on the ten missed scored undersides | Appearances with at least half the seam traced | Fraction of all tested seam samples |
| --- | ---: | ---: |
| ScaleLSD | 7 / 10 | 54.7% |
| DeepLSD | 9 / 10 | 71.4% |
| Either detector at each location | 10 / 10 | 82.1% |

These ten misses are eight pictures of four previously examined buildings, all from low-left views. The control sampled emitted lines about every two pixels along their length. Of the samples falling on a building, 80% for DeepLSD and 42% for ScaleLSD were more than a five-pixel square radius inside a single surface. An interior line can be a brick seam or a change in color or lighting. [Seam and control measurements](reviews/overhang-01/lines.json)

## Two tests worth running

1. **Separate parallel faces.** Start from the current mask. Let the depth splitter or a side of B's rectangle suggest an internal dividing edge. Fit the two sides independently using relative depth and visible corners or T-junctions. Keep the original holes. Compare with ordinary depth splitting and record gained surfaces, lost surfaces, spurious splits and covered openings.
2. **Recover thin undersides.** Let a line from either detector propose a new narrow face inside or just beside an existing mask. Look for a coherent strip with two edges or an edge that meets an opening or support, with a sustained change in brightness across it. Compare DeepLSD alone with either detector allowed to propose an edge under the same acceptance rule. Count new underside matches, flat walls split by mistake and gaps filled. The missing Galleries strip makes correct predicted surface direction a poor mandatory gate.

Use the original 20 development pictures to choose settings, then freeze them before scoring the other 36 regression pictures. There are only four scored underside appearances on development, so report counts by building and keep the paired palettes or floors together. Treat the gains in the table as observations on the frozen suite. This audit adds no predictor or placed-block result.

## Replay

Use the suite's Python environment. With the saved numerical field cache on this PC, run the following in the review checkout.

```powershell
python -m benchmark.overhang_review --out runs/overhang-repeat
```

Use a new output directory for each replay. Pass `--fields <directory>` if the compatible MoGe and Depth Anything cache is elsewhere. The review reads the saved captures, predictions, line evidence and reference geometry. [Input revisions and output hashes](reviews/overhang-01/manifest.json)
