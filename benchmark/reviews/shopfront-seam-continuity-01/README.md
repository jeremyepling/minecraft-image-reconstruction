# Jagged surface seams on Shopfront and Registry

On the Shopfront plaster high-right source image, a smooth seam divides one visible side wall into two masks. Joining the masks assigns another 5,139 pixels to the correct surface. The resulting Block Counter estimate on one axis is 35, although the entire recorded visible plane spans only 8 × 7 blocks. The exported block coordinates tie. On the low-left source image, the apparent hump above the sill has a different cause. All 3,300 pixels in the inspected rectangle belong to front plane 12. One mask covers 624 pixels and the remaining 2,676 have no support to join.

## What was tested

From `a95816d`, the Surface Finder challenger compares nearby pieces whose predicted normals agree. It requires a common boundary with at least 18 samples, extending 16 pixels or more and deviating at least 3.5 pixels at the 90th percentile from its fitted straight line. A smooth seam must have a 75th-percentile RGB difference of at most 5/255 after light smoothing, no more than two levels above the adjacent interior comparison. The alternate texture test projects the two masks using the image-derived camera and independently fits a repeating tile to each one. Both must select the same integer plane with a margin and agree on the shared texels. The proposed mask is the exact union of the two supported areas. A seam with multiple potential partners is left alone. The world readback and reference masks are opened after prediction.

| Frozen rule and images | Accepted pairs | Exact reference surfaces gained / lost | Change in correctly paired pixels |
|---|---:|---:|---:|
| Smooth or repeating, fixed 104 | 8 | 1 / 2 | −71,910 |
| With junction guard, same 104 | 7 | 1 / 0 | +30,344 |
| With junction guard, additional 100 | 5 | 0 / 1 | +9,474 |

The initial rule merged two different Registry walls in the plaster low-left view. They looked smooth at the join. Both carry existing junction evidence. The guard protecting those earlier splits was added after examining this result. All four Registry source images then retained their original masks. The 100 additional images include the 72 training examples, 20 alternate views, four new-shape views and four shelter/gate views. On alternate Canopy 08 the guarded rule still merged two real surfaces. It lost four required blocks and added two extras. Smooth seams alone cannot distinguish two similarly shaded faces at different depths. The repeating-texture test accepted four pairs across the 204 source images, all on the same recorded plane; all four builds tied. It recovered the Portico masonry high-right roof as an exact match using the combined masks. It did not fire on Shopfront.

## Rightmost Registry wall

The plaster high-right view is the one displayed in the lowest-score viewer. The far-right wall, ID 10, has 31,895 visible pixels. Existing masks cover 16,062. Its largest mask includes 16,039 pixels of that wall and 6,473 on a different right-facing wall, ID 9. In the masonry high-right view the existing masks cover 31,877 pixels of ID 10, and the two walls again share one mask. A split at the depth transition and recovery of unsupported wall pixels are separate tasks.

In the low-left camera the rightmost upper front wall, ID 6, has 55,599 visible pixels. Its coverage is 55,575 in masonry and 55,590 in plaster. The broad rule incorrectly joined plaster planes 2 and 6. [Per-source coverage, contributing masks and evaluated pairs](guarded.json)

## Placement

In a separate repeating-texture-only replay, both sides were tested for plane and texel agreement even if their colors changed smoothly. It made the same four joins across 204 images, with 19,660 more correctly paired pixels and five fewer unpaired masks. The Cloister union counts 9 × 6 blocks, matching the extent of its 44 majority-covered visible faces. The Facade union preserves the correct 25 × 8 span. Arcbridge and Portico return unknown dimensions on their joined masks. The four new block sets match their controls exactly. [All repeat-only proposals](phase-only.json) · [Verified counts, reference spans and placement](phase-placement.json)

For the guarded test all 12 affected images reproduced the saved production block sets before intervention. Both variants were recounted and placed. The seven fixed-suite changes exported exactly the same coordinates. Strict placement stays at 90.8126% (20,619 correct, 3,083 missed, 1,089 extra) and facade-priority stays at 92.9145% (18,798 correct, 1,778 required misses, 1,089 extra), both a zero-point change. The historical 112-view strict score with the eight retired wall views remains 90.0511%. The transfer Canopy 08 regression is outside the fixed 104. Retain the broad and junction-guarded rules as experiments. [Per-image dimensions and placements](placement.json)

## Replay

Run from the experiment checkout with the frozen fixtures and Java 26.3 atlas available. The broad and guarded tests write separate files. The placement pass reads the guarded case manifest.

```powershell
python -m benchmark.shopfront_seam_continuity
python -m benchmark.shopfront_seam_continuity --protect-junctions --all-cohorts
python -m benchmark.shopfront_seam_placement
python -m benchmark.shopfront_seam_continuity --protect-junctions --phase-only --all-cohorts
python -m benchmark.shopfront_seam_placement --phase-only
```

The case manifests include the source image and native visible surface-map SHA-256 for each frame. Matched surfaces use one-to-one native-mask IoU of at least 0.5. The two-input purity flag requires at least 80% of both masks on the same recorded visible plane. [Original fixed-suite trial](cases.json) · [Guarded replay](guarded.json)
