# What a thin strip contributes to the placement score

In the pictured Colonnade 01 view, **61 of the 65 missed blocks show only a top face**. No individual top face on those 61 blocks contains a fully visible, axis-aligned 2 × 2 native pixel square. Many have more than a few pixels along a long strip. The previously published rule checks the **whole connected top surface** for an 8 × 8 square. Neighboring blocks can jointly provide that square. It neutralizes seven misses in this view. Checking each horizontal-only block for a 4 × 4 square instead neutralizes all 61. Four wall misses and all 18 extra blocks still count, giving **88.89%** F1, up from 75.22% strict or 76.40% under the existing whole-top rule. Only 19 of the 65 misses have an area below 16 pixels. [Original image, building crop and colored roof strip](sliver-colonnade.jpg)

The opposite Colonnade view in the alternate cohort has 69 misses and 53 extras. Of its misses, 67 satisfy both the whole-top 8 × 8 and individual horizontal-face 4 × 4 rules. Its F1 goes from 65.92% strict to 76.60% under either criterion. This second view is outside the fixed-104 totals.

## Same exports, different visible-target priorities

These are pooled results on the 104 full-structure photographs from Java 26.3, all at 1440 × 900. The strict build has 20,535 correct placements, 3,167 misses and 1,146 extras. Each photographed block is counted once. We keep the same one whole-build translation as strict scoring. An optional block contributes neither a reward when recovered nor a penalty when missed. **Every extra remains an error in every row.** Projected extras below are only a location diagnostic: more than half the extra block's first-hit pixels must land in the optional area's native footprint. This overlap does not establish cause.

| Visible evidence made optional | Correct made neutral | Misses made neutral | Extras mostly projecting there | Visible-block F1 |
| --- | ---: | ---: | ---: | ---: |
| None, strict | 0 | 0 | 0 | 90.50% |
| Whole connected top region lacks 4 × 4, top-only blocks | 8 | 232 | 2 | 90.96% |
| Whole connected top region lacks 8 × 8, top-only blocks | 68 | 620 | 10 | 91.72% |
| Whole connected top or underside region lacks 8 × 8, horizontal-only blocks | 82 | 768 | 10 | 92.02% |
| Individual top or underside face lacks 2 × 2, horizontal-only blocks | 248 | 959 | 16 | 92.36% |
| Individual top or underside face lacks 4 × 4, horizontal-only blocks | 1,821 | 1,305 | 24 | 92.56% |
| Individual top or underside face lacks 8 × 8, horizontal-only blocks | 2,636 | 1,459 | 29 | 92.62% |
| All horizontal-only blocks, facade score | 2,789 | 1,493 | 29 | 92.64% |

Under the local 4 × 4 rule, **1,305 of 3,167 misses (41.2%)** have only a thin horizontal face: 849 top and 456 underside. This leaves 1,862 required misses. If the same threshold applied to *every* photographed face of a block, another 340 misses with a visible wall would become optional (1,645 total, 51.9%). Many thin wall faces provide useful placement evidence for posts and small recesses. A pure area cutoff below 16 source pixels instead includes just 427 misses of all orientations (13.5%). The native square test measures whether there is a solid patch of the specified side, so occlusion, fragmentation or the angle of a diagonal edge can affect it as well as thickness.

The local test changes the priority of the earlier 632-block orientation problem. Those 632 are horizontal-only misses where the build leaves a majority of photographed pixels uncovered, yet a predicted mask covers half or more and a correctly oriented mask covers less than half. The whole-top 8 × 8 criterion neutralizes 374 of them. Extending the whole-region 8 × 8 criterion to undersides neutralizes 486. Under one 4 × 4 square per individual horizontal face, **615 become optional**: 394 of 397 tops and 221 of 235 undersides. This leaves **17** required blocks in that specific cohort. On 524 of 632, the face lacks even a 2 × 2 square.

The 632-block screen tests insufficient horizontal-mask coverage. A second measurement checks explicit wall assignments. On **608 of 632** blocks, a vertical mask covers at least half the footprint, even after excluding any pixel also covered by a horizontal mask. The other 24 have some vertical-mask coverage below half. Within the 17 still required by the local 4 × 4 rule, ten have a majority explicitly labeled wall; seven have a mix with only partial vertical coverage.

## Geometry cues and tradeoffs

One physical cube can supply distinct front, side and top evidence. Using the reference first-hit pixels, 19,174 visible blocks show exactly one face (2,995 misses), 4,376 show two (169 misses) and 152 show three (three misses). A single photographed pixel suffices to establish a face for these counts. They measure the available evidence and do not imply that an adjacent perpendicular mask belongs to the same cube; the shared projected corner, block grid and depth need to agree.

The published whole-top rule changed how the same geometry was evaluated. It never removed thin surfaces from training or from the build. A separate historical attempt to delete small or thin masks reduced redundant outlines but also lost 42 real matched surfaces on 104 fixed views and three correct blocks on the pictured Shopfront. Keeping a low-priority face available as evidence is useful even if its block will be neutral in the facade-oriented metric. [Earlier pruning and shade tests](../wall-texture-seams-01/README.md)

## Reproduce

The read-only scorer verifies the frozen screenshot, visible reference and export hashes, measures the largest tested square of identical first-hit face or surface IDs, and reuses the current camera and one whole-build scoring translation. Coplanar, edge-connected visible unit faces with the same signed normal share a surface identifier, including across changes in material. Unknown and hidden geometry are excluded from required targets. The breakdown includes threshold sensitivity, all per-view counts and both Colonnade cases. [Measured counts and provenance](sliver-counts.json)

```powershell
python -m benchmark.strip_visibility_audit --run ../visible-error-budget/runs/primary-production --out benchmark/reviews/strip-orientation-01 --workers 4
```

For another frozen prediction run with the same layout, provide its run directory and a fresh output directory. A comparer with recorded labels can call `horizontal_slivers(labels, ids, side=4)` to obtain the optional world coordinates; setting `whole_surface=True, top_only=True, side=8` reproduces the whole-top selection.
