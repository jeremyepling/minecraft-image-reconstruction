# Can a corner rescue a missed ledge, or two pieces recover a wall?

The current Surface Finder found **501 of 598 surfaces**. Asking a visible corner to confirm a ledge gave the same 501 predictions that matched. Joining wall pieces recovered zero surfaces and lost 12. Keep the current Surface Finder as the ordinary default. [Portrait comparison](reviews/ledge-join-01/comparison.jpg) · [All 56 pictures with surfaces gained and lost](reviews/ledge-join-01/pictures.md)

| Across the same 56 views | Current finder | Ledge cuts | Wall-piece joins |
| --- | ---: | ---: | ---: |
| Surfaces found on development (156 scored) | 132 | 132 | 132 |
| Surfaces found on six previously reviewed buildings (442 scored) | 369 | 369 | 357 |
| All surfaces found (598 scored) | 501 | 501 | 489 |
| Extra guesses after allowing small-surface matches | 201 | 201 | 203 |
| Scored surface pixels assigned correctly | 93.33% | 93.33% | 92.25% |
| Opening samples incorrectly covered (16,630 scored) | 601 | 601 | 601 |
| Correct connections between surfaces (606 scored) | 406 | 406 | 393 |
| Undersides found (18 scored) | 14 | 14 | 14 |
| Both dimensions correct in blocks | 78 | 78 | 75 |
| Wrong dimensions / unresolved | 41 / 479 | 41 / 479 | 41 / 482 |

## What happened in the pictures?

**Workshop R13.** The ledge rule looked at the three candidate cuts across the suite where depth nearly agreed. ScaleLSD traced 56 pixels along the Workshop ledge, but the nearest crossing lines were 14 and 29 pixels from its ends. The frozen rule required a junction within 8 pixels. Neither Shopfront candidate had a long enough aligned line. The ledge rule made zero cuts. All 56 sets of output regions were identical to the baseline. [Per-picture line, endpoint and depth evidence](reviews/ledge-join-01/candidate-runs/ledge/cases.json)

**Shopfront R5.** The best individual guess still had 42% overlap with the wall around the bays. Earlier diagnostic selection showed two pieces that together scored 70%. The automatic test rejected the pair. They differed enough in brightness and estimated depth to fail the frozen checks, and the second supporting model outline overlapped their union by only 47.5% against the required 58%. On Shopfront and Wing the conservative join rule tied the baseline at 132 matches, while a looser rule fell to 130. The comparison used the conservative rule on the other six buildings. [Development settings and scores for joining](reviews/ledge-join-01/candidate-runs/join/development-freeze.json)

**Portico.** The joiner accepted ten pairs in nine views across the suite. Every pair combined pieces whose pixels mostly belonged to two different recorded surfaces. On the smooth-floor, plaster, low-left Portico view, R17 and R18 had the same recorded direction and depth. So did R11 and R12. Each pair still belonged to separate connected surfaces. Both joins preserved the gap pixels, but the two pairs produced just one surface match between them. Portico lost seven matches across its views, Workshop and Galleries lost two each, and Registry lost one. The remaining four buildings kept their totals. [Per-surface gains and losses](reviews/ledge-join-01/pictures.md)

For a future join test, look for a route along visible wall pixels connecting the two pieces around a recess. Treat unbroken sky or ground between two pillar faces as evidence for keeping them separate. For the Workshop ledge, the open question is whether its original pixels support the short missing corner where the long model lines stop.

## Replay

```powershell
.\Invoke-SurfaceRound.ps1 -Round ledge-join
```

The two tests start independently from the saved five-model Surface Finder. The ledge test keeps the original depth threshold for a step and checks for an aligned ScaleLSD line ending in a corner or T for votes with 60% to under 70% agreement. The joining test fits a relative-depth plane to each proposed wall piece, checks direction and color, and asks whether at least two distinct MobileSAM prompts support their union. Each accepted join keeps exactly the original visible pixels and all holes. The reference geometry enters during development grading, final scoring and labeled overlays. The initial model inference, camera and Block Counter are shared. The replay uses the frozen Java 26.3 clear-noon captures and runs without Minecraft or new model inference.

The published run spent **0.019 seconds** on the added ledge checks and **7.64 seconds** on the added joining checks summed across 56 views. Those figures exclude reading archives, shared preparation, original model inference and scoring. The independent control replay also regenerated the default's last wall-cut pass from the numerical inputs and reproduced the saved proposals in every view. The portrait sheet returns to the four targets from the previous review at their exact saved coordinates. [Run manifest, versions and hashes](reviews/ledge-join-01/manifest.json) · [Full scores by building](reviews/ledge-join-01/scores.json)
