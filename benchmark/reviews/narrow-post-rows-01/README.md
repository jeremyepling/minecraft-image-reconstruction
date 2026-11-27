# Read block rows on a partly hidden pillar

On the left view of Canopy 04, a front pillar and a narrow exposed strip of a rear pillar shared a surface mask. Comparing their native texture profiles finds the seam. The downstream Block Counter could measure the front but could not fit a complete tile across the rear strip. We now fit the vertical repetition independently on either side. [Block courses over the source detail](canopy-04.jpg) · [All 200 image-only estimates and evaluated depths](rows.json)

For each candidate integer depth we project the photographed texels onto alternating block rows. We fit atlas appearance on even rows and test it on odd rows, then reverse the folds. Separate depth planes are permitted only when the atlas scores on both masks exceed 0.65 with at least a 0.18 margin over the next depth. These are correlation and margin thresholds. A single plane still explains coplanar material changes and triggers no additions. Every emitted block must project onto at least 24 sampled mask pixels, have at least 24 native frontmost pixels on the measured face and keep at least 75% of that exposed face in the measured mask. We re-render the scene to reject occluded proposals, loss of a previously visible block, or newly occupied background beyond the allowed contour tolerance. Empty masks, an unresolved camera and an unavailable or uncalibrated atlas leave the input placements intact.

The pictured Canopy 04 produced local planes −3 and +1. Across all sampled pixels on the two masks the selected depths put 99.25% and 84.4% respectively on the exact recorded first block. The strip includes a few pixels outside the pillar at its foot. Simply placing every touched cell would have gained five correct positions and two extras on the narrow side. Photographed foreground support, source-ray visibility and rejection of negative local heights removed the extras in the exported build.

| Tolerance on the photographed outer contour | Fixed 104, correct added | Alternate 20, correct added | New extras or hidden additions |
| --- | ---: | ---: | ---: |
| Exact mask | 1 | 1 | 0 |
| One native pixel | 10 | 7 | 0 |
| Two native pixels | 14 | 8 | 0 |

The two-pixel candidate recovered five correct blocks on Canopy 04, five on Canopy 18 and four on Canopy 05. On alternate angles it recovered four on Canopy 06 and four on an open Canopy 00. The trial was refined after inspecting Canopy 04. The gains are concentrated in overlapping posts. Development, independent new layouts and shelter/gate tied. The four additional native canopy views and eighteen camera-challenge views also tied. The latter set has eighteen unresolved cameras. The forty-four blocks in the complete Canopy 04 screenshot export match the candidate, including all five new correct blocks. Sampled incremental CPU cost on three photographs was about 0.11–0.84 seconds with assets warm. [Timing samples](timing.json)

The found seam, row counter and two-pixel contour tolerance were selected for ordinary 26.3 placement. The outer limit accounts for rasterization of a thin silhouette. At least three quarters of the exposed tile still has to fall inside its own original mask. The historical surface-mask join and shade rules retain their separate benchmark entry points.

## Replay

```powershell
python -m benchmark.narrow_post_rows --out runs/rows --workers 4
python -m benchmark.wall_section_eval --candidate benchmark.narrow_post_trial:two_pixels --out runs/two_pixels --workers 4
python -m benchmark.wall_section_eval --candidate benchmark.framed_geometry_order:production --out runs/production --workers 6
```

The last command uses the ordinary geometry stages of the checked-out revision. [Verified combined placement and release archives](../visible-error-budget-01/README.md)
