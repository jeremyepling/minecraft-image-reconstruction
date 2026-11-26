# Finding surfaces the first pass missed

Pixel-wise Planarity recovered 12 of the 97 surfaces the previous Surface Finder missed. It kept all 501 previous matches, improved six of the eight buildings and covered exactly the same number of opening samples. ECT could recognize many real depth edges, but it was much less reliable at turning an outline into a whole surface. [Zoomable eight-building sheet](reviews/afterpass-models-01/comparison.jpg) · [All 56 pictures](reviews/afterpass-models-01/pictures.md)

## What improved

| On all 56 views | Previous Surface Finder | Add Pixel-wise Planarity | Add ECT-ranked outlines |
| --- | ---: | ---: | ---: |
| Surfaces found out of 598 | 501 | **513** | 511 |
| Previously found surfaces lost | — | 0 | 0 |
| Extra guesses after small visible surfaces are accounted for | 201 | **204** | 424 |
| Visible surface pixels assigned to the right surface | 93.33% | **94.47%** | 87.78% |
| Opening samples incorrectly covered out of 16,630 | 601 | **601** | 1,755 |
| Correct connections out of 606 | 406 | **418** | 397 |
| Undersides found out of 18 | 14 | 14 | 14 |
| Mean boundary / corner error in source pixels | 4.32 / 13.11 | 4.23 / 13.30 | 4.19 / 13.24 |
| Surfaces with both block dimensions right | 78 | **80** | 81 |
| Surfaces with a wrong size / no paired size | 41 / 479 | 44 / 474 | 42 / 475 |

On Shopfront and Wing, where we picked settings, Planarity found **137/156** and the previous finder found **132/156**. On the six previously reviewed buildings, Planarity found **376/442**, ECT found **373/442** and the previous finder found **369/442**. On those six buildings Planarity’s fraction of guesses that matched dipped from 68.8% to 68.6%, while the combined precision and recall score rose from 75.5% to 76.0%. This is a regression comparison on familiar buildings. Performance on a new building is still unmeasured. [Scores by building and split](reviews/afterpass-models-01/scores.json)

| Building, across all its views | Previous | Planarity | ECT |
| --- | ---: | ---: | ---: |
| Shopfront | 98/120 | 102 | 104 |
| Wing | 34/36 | 35 | 34 |
| Crossing | 38/46 | 42 | 38 |
| Courtyard | 53/56 | 53 | 54 |
| Galleries | 82/98 | 82 | 82 |
| Portico | 130/148 | 131 | 131 |
| Registry | 27/34 | 28 | 29 |
| Workshop | 39/60 | 40 | 39 |

The denominator counts surfaces across pictures. The same wall can appear in several views or palette and floor variants. In the sheet each building has a fixed column, with the **count for that particular picture** beneath each prediction. Gold shows recorded geometry. Each predicted outline can match one recorded surface at 50% overlap or better. [Prediction and reference overlays](reviews/afterpass-models-01/comparison.jpg)

## What the models did

1. The previous Surface Finder ran first. MoGe-2 supplied surface directions and Depth Anything V2 supplied relative depth. MobileSAM supplied candidate masks, ScaleLSD supplied line and junction evidence, and M-LSD helped find thin undersides. We reproduced its 501 matches before comparing the new models.
2. **Pixel-wise Planarity** took the full original screenshot. Its trained model estimated planarity, surface direction and depth. Fixed rules grouped adjoining pixels, added a nearby unclaimed surface, or split an existing mask when two sufficiently different planes explained it. We tried standard additions, standard additions with splits, a thin-surface setting and a more permissive surface-direction setting. The first three each found 134 surfaces on development. The last found 137 and was frozen. On all 56 views it added eight regions and split nine existing masks. The original mask constrained every split. [Development settings and scores](reviews/afterpass-models-01/candidate-runs/planarity/development-freeze.json) · [Authors’ checkpoint](https://huggingface.co/alpayozkan/pxwplanar-moge2-planarity)
3. **ECT** took the original screenshot independently. It estimated edges caused by depth, surface direction, reflectance and lighting. We used those signals to reconsider MobileSAM outlines that the previous finder had passed over. We compared plain RGB contrast, geometric edge strength, and geometric strength with a penalty for appearance edges. Each could return one, three or six extra proposals per image. At six, they found 137, 137 and 138 surfaces on development. On all 56, the selected ECT setting found 511 vs 508 with the same RGB control. Only 39 of the original 97 misses had a matching raw MobileSAM outline. The common filters kept 24 of those opportunities and ECT selected ten. [ECT settings and scores](reviews/afterpass-models-01/candidate-runs/ect/development-freeze.json) · [ECT model and checkpoint](https://github.com/Daniellli/ECT)
4. The unchanged **Block Counter** used the screenshot and each proposed surface to measure width and height from texture repetition. We graded its outputs against recorded dimensions after prediction. Planarity improved exact sizes by two. Actual block coordinates, material IDs and in-game placement were outside this surface experiment.

### Could ECT just confirm a wall cut?

We scored proposed edges before asking whether cutting along them improved a whole wall. On 48 clear cases from the six reviewed buildings, ECT recognized **15/23** real depth changes and flagged **5/25** flat walls. Plain contrast found **4/23** and flagged **6/25**. Subtracting appearance strength more aggressively found **7/23** and flagged **2/25**. Four ambiguous edges were recorded separately. [Every graded boundary](reviews/afterpass-models-01/edge-diagnostic/boundaries.json)

After seeing those diagnostic results, we applied the frozen thresholds to the finished walls. Only 26 of the 66 original candidate cuts across the suite still had an intact wall to split. ECT cut five, found Workshop R13, and lost Registry R6. The total stayed **501**, with **206** extra guesses. Plain contrast also cut five and finished at **502**, with **207** extras. Both preserved the opening coverage. [Actual cut results](reviews/afterpass-models-01/wall-cuts/scores.json)

## Where to zoom

- **Wing R2, plaster viewed from high right.** The old prediction overlapped just 14% of the recorded wall. The raw Planarity region overlapped 97%. Its accepted outline reached 77%, enough to recover that wall.
- **Workshop R13, masonry viewed from low left.** The raw Planarity region overlapped 79% of the missed surface. The automatic pass kept the old outline at 4% overlap. The separate ECT wall-cut test recovered it. This is a place to improve how we accept an outline already visible in the model output.
- **Galleries R13, masonry viewed from low left.** The raw Planarity labels had zero overlap with the recorded underside. All three main predictions stayed at 36%. Improving the acceptance rule alone cannot extract this underside from those labels.

Across the suite, raw Planarity labels overlapped **37 of the 97** previous misses by at least half. Automatic processing recovered 12. Planarity and ECT recovered three of the same surfaces, nine only Planarity found and seven only ECT found. A blind union would also inherit ECT’s unwanted outlines. [Each earlier miss and its outcome](reviews/afterpass-models-01/recovered.json)

## Reproduce the round

```powershell
.\Invoke-SurfaceRound.ps1 -Round afterpass
```

The command rebuilds both candidate predictions from saved numeric model fields, re-scores the previous finder, runs the exact same block counter, and regenerates the sheet and both ECT diagnostics. The 56 fixture entries contain 52 distinct screenshots, all from vanilla Java 26.3 at clear noon. Image hashes and model-field hashes are checked before scoring. Camera poses and recorded geometry are available to the scorer. The ordinary prediction receives one image. [Round manifest](reviews/afterpass-models-01/manifest.json)

We also ran the ordinary build on a renamed Wing screenshot. All six models ran from the source pixels, the build recovered the missing wall, and all eight region records matched the saved experiment. [Fresh-image verification](reviews/plane-recovery-entry-01.json) · [Ordinary build command](WALL_RECOVERY_BUILD.md)

Inference on the RTX 4080 took 3.30 seconds for Planarity and 7.49 seconds for ECT across the 52 distinct screenshots. Planarity also took 3.02 seconds to load and 1.51 seconds to group all three sets of plane labels. ECT took 10.47 seconds to load. Timings for applying each frozen method to all 56 cached inputs are in the per-model manifests. [Planarity fields, settings and checkpoint hash](reviews/afterpass-fields-01/planarity/manifest.json) · [ECT fields, settings and checkpoint hash](reviews/afterpass-fields-01/ect/manifest.json)
