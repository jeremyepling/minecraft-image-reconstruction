# Can the finder recover the Workshop ledge and keep the Crossing wall?

**Split-only zoom was selected for the ordinary build on `main`.** The one-wall split found one additional surface on Portico, and the Block Counter measured it correctly. It also created six extra guesses. The two Workshop ledge views remained misses. [Zoomable sheet](reviews/contained-01/comparison.jpg) · [All 56 pictures](reviews/contained-01/pictures.md)

| On the same 56 saved pictures | Split-only zoom | One-wall split |
| --- | ---: | ---: |
| Matched surfaces out of 598 | 516 | 517 |
| Extra guesses | 204 | 210 |
| Opening samples covered out of 16,630 | 601 | 601 |
| Surfaces with both block dimensions correct | 80 | 81 |
| Correct connections out of 606 | 424 | 425 |
| Undersides found out of 18 | 14 | 14 |
| Development, Shopfront and Wing | 138 / 156 | 138 / 156 |
| Six previously reviewed regression buildings | 378 / 442 | 379 / 442 |

### Where the decision came from

- **Portico.** The split separated the narrow R5 face in the high-right plaster view. Its overlap with the reference rose from 9% to 68%, and its previously unresolved size became 1 × 5 blocks, exactly right. Splitting the original surface into four pieces also produced two extra guesses.
- **Shopfront.** In the low-left plaster view with the concrete floor, one accepted cut divided a surface into five pieces. That produced four extra guesses and gained zero matches. These two cuts account for all the changed predictions.
- **Workshop.** Both proposed ledge outlines fit inside one parent wall. They overlapped the recorded 6 × 1 face by 79% and 81%. Along their edges, zero samples supported a change in facing direction. Relative depth agreed on a step at 61% and 64% of sampled crossings, below the frozen 70% threshold. The finder kept the original wall. The nearby long face recovered by split-only zoom stayed matched in both views.
- **Crossing.** The plaster high-right wall retained its 90% overlap. The earlier protected-edge approach had reduced it to 49%. [Per-picture matches, rejection reasons and pixel audits](reviews/contained-01/cases.json)

### How the test ran

Both methods started with the same split-only zoom masks. The challenger considered the earlier outlines supported by Pixel-wise Planarity and MobileSAM. A proposed cut had to fit inside one existing surface within a one-cell, 5% registration tolerance, and show a change in direction or relative depth across the edge. Accepted pieces retained exactly the original surface pixels, including its holes. The settings were fixed before this run. Reference geometry was used afterward for grading and the labeled close-ups. [Frozen settings](reviews/contained-01/development-freeze.json)

Across 56 pictures the rule considered 25 outlines. It accepted two, rejected six that crossed a second surface, rejected one that left too little area on one side, and rejected 16 whose edge lacked geometric support. Every original match survived and opening coverage was identical in every picture. Extra guesses exceeded the predeclared limit. All eight buildings have been reviewed before. This round measured outlines and block dimensions on saved screenshots. [Run manifest, model hashes and input revisions](reviews/contained-01/manifest.json)

### Reproduce

```powershell
.\Invoke-SurfaceRound.ps1 -Round contained
```

The command reuses the frozen Java 26.3 clear-noon images and numerical model outputs with Minecraft closed. It writes fresh predictions, scores, a comparison sheet and a manifest under `runs/`. On the portrait sheet, Workshop and Crossing occupy adjacent columns. The Portico close-up follows the recovered face through straightening and block counting. The Shopfront close-up shows where the wall broke into pieces. [Frames and crop coordinates](reviews/contained-01/sheet-views.json)
