# Should Surface Finder trim a narrow strip?

Courtyard B has a false tail below the right end of a 6 × 1 beam. Shopfront B has a genuine 1 × 4 side face. Both rules removed all 184 pixels in the lower tail and 208 incorrect pixels across the beam proposal. They retained all 4,554 correctly assigned beam pixels and preserved the Shopfront face. The Block Counter still could not determine the dimensions of either example.

[Review both B examples](reviews/attached-strip-01/strip-comparison.png) · [See a genuine extension that got cut](reviews/attached-strip-01/strip-failure.png) · [Visual feedback](reviews/attached-strip-01/feedback.jsonl)

| On the same 56 views | Current Surface Finder | Trim by shape | Trim by shape and color |
|---|---:|---:|---:|
| Matched surfaces out of 598 | 517 | 518 | 518 |
| Previous matches lost | 0 | 1 | 0 |
| Exact block dimensions | 94 | 98 | 95 |
| Incorrect block dimensions | 51 | 46 | 49 |
| Opening samples claimed as a surface | 511 | 470 | 492 |
| Correctly assigned eligible surface pixels on the scoring grid | 94.539% | 94.415% | 94.532% |

The shape rule looks for a thin extension attached at one end to a broader region. The photo-assisted rule also checks whether the colors on the extension match its immediate surroundings more closely than the parent surface. Both operate on the frozen finder output, before the ordinary crop and counter. Recorded geometry is used for scoring and display.

**How often did this happen?** The photo-assisted rule changed 22 of 714 proposals across 17 views. Of the 18 changed proposals matched to an eligible recorded surface, eight lost only incorrect pixels. The other ten lost a mix of correct and incorrect pixels. Four changed proposals had no eligible match for this check. Across those 18 matches the rule removed 2,390 incorrect pixels and 1,678 correct pixels at native resolution. Estimating the prevalence of all stray strips would also require identifying strips the rules missed.

**Decision.** Keep the current Surface Finder. The shape-only rule lost a Galleries surface. The photo-assisted rule kept every previous match, but on Courtyard plaster, low-left, concrete floor (R5), 587 of the 596 pixels it removed belonged to the real wall. On development buildings, current, shape and photo-assisted methods each matched 138 surfaces and got 17, 19 and 17 block sizes right respectively. On the previously reviewed regression buildings they matched 379, 380 and 380 surfaces and got 77, 79 and 78 sizes right.

[Per-picture changes](reviews/attached-strip-01/cases.json) · [Per-building scores](reviews/attached-strip-01/scores.json) · [Run manifest](reviews/attached-strip-01/manifest.json)

Reproduce from the repository root using a fresh output directory.

```powershell
.\Invoke-SurfaceRound.ps1 -Round attached-strip -Out runs/my-strip-review
```

### Checking for a neighboring surface

The follow-up kept the shape and color tests and added one requirement. A single, independently predicted surface must run within three pixels of at least 90% of the strip, cover at least 75% of each quarter of its length at that distance, and have similar colors. The trim removes pixels from the original surface. It adds no pixels to the neighboring surface.

| Across the same 56 views | Default | Shape and color | Also require a neighbor |
|---|---:|---:|---:|
| Correct surfaces out of 598 | 517 | 518 | 517 |
| Incorrectly assigned pixels removed from matched surfaces | 0 | 2,390 | 516 |
| Correctly assigned pixels removed from matched surfaces | 0 | 1,678 | 0 |
| Correct block sizes | 94 | 95 | 95 |
| Incorrect block sizes | 51 | 49 | 50 |

The neighbor check changed five surfaces across five pictures. On Courtyard B, another predicted surface ran alongside all 208 pixels removed from the beam. It had zero direct overlap with them. On the genuine Courtyard R5 wall extension, the closest surface reached just 168 of 596 candidate pixels. The rule kept that extension intact, including all 587 correctly assigned pixels. It also preserved the genuine 1 × 4 Shopfront face. The reviewed beam itself still had no complete block count. The newly correct count came from Registry R2.

**Recommendation.** Promote the neighbor check as a conservative Surface Finder cleanup. It fixed the requested example and kept all previously matched surfaces and exact sizes. The current default still uses the original finder. These eight buildings have been reviewed before, so the observed zero collateral damage applies to this suite. On development buildings, the default and neighbor check each had 138 matches and 17 exact sizes. On the other reviewed buildings, both had 379 matches and exact sizes increased from 77 to 78.

[Zoomable comparison](reviews/strip-neighbor-01/comparison.png) · [Per-picture results](reviews/strip-neighbor-01/cases.json) · [Scores and splits](reviews/strip-neighbor-01/scores.json) · [Manifest](reviews/strip-neighbor-01/manifest.json)

```powershell
.\Invoke-SurfaceRound.ps1 -Round strip-neighbor -Out runs/my-neighbor-review
```
