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
