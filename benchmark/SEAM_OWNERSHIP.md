# One block at a visible corner

On the right view of Adjacency 26, the previous exporter sometimes filled a gap with a second cube to explain a side face. The recorded camera and server readback identified 26 extras one block east of an already placed wall block. For each one, the east face of that wall block accounted for the largest share of the projected extra's pixels. Both the wall block and its side were visible in the screenshot.

The placement stage compares the observed texture at three projected faces: the owner's front, its exposed side in the one-cube explanation, and the proposed second front. The comparison allows a horizontal flip and lighting differences. The owner's texture comes from the same screenshot. The separate atlas and material selection stages can still rank block identities. A corroborated vertical run may extend the decision beyond an individual texture match. It requires the same owner direction, a ground contact, matching textured floor samples and a check on how much observed surface would disappear. Ambiguous ownership and strict placements are protected. The placement manifest records how many cells were removed and which checks supported them.

## Decoded placements

| Matched comparison | Correct before and after | Extras before | Extras after |
| --- | ---: | ---: | ---: |
| Original review, Adjacency 26 right | 257 | 67 | 51 |
| Ordinary current build, Adjacency 26 right | 258 | 63 | 46 |
| Current surface masks, 56 exterior views | 2,066 | 580 | 580 |
| Current surface masks, 12 additional supported views | 2,760 | 506 | 487 |

The combined rule removed 16 of the 26 audited cells on both versions of the source masks. On the current masks one further off-target cell was removed from that view. In the whole suite, two other extras were removed, one on Adjacency left and one on Physical Shade right. Each decoded coordinate set was compared with its own before-build and its own screenshot-visible target. Source-angle silhouette overlap on the current right view increased from 94.94% to 95.52%.

The standalone texture check removed 11 of the 26 duplicates. An early repeated-edge rule removed 16 but also trimmed a misplaced wall in Corner 29 right, reducing its source-view overlap from 88.50% to 84.75%. We used that regression to design the corroboration requirement. The combined version preserves both Corner exports exactly. These are regression views used during development. [Source-angle comparison](reviews/seam-ownership-01/adjacency-right.jpg) · [Scores and frozen input fingerprints](reviews/seam-ownership-01/results.json)

## Replay

Run the normal exterior build and compare it with the previous export for the same screenshot. Use the recorded fixture root containing `captures`, `views.json` and `layout.json`. The camera and server states enter only during evaluation.

```powershell
.\Invoke-WallRecovery.ps1 "$fixture/captures/test-adjacency-26-right.png" -Out runs/adjacency-seam
python -m benchmark.seam_ownership_round --capture $fixture --case test-adjacency-26-right --build runs/adjacency-seam --baseline $previous/reconstruction.zip --out runs/adjacency-seam-score
```
