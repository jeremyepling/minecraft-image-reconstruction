# Can placed blocks repair a surface boundary?

The 204 saved source images use the same foreground masks, estimated camera and decoded material exports as the 26.3 release at `5818a37`. We tested a single feedback pass. An existing color-based method supplies disjoint pairs of masks with a visible seam. The challenger projects the final cubes back through the estimated camera. It checks both sides within 12 source pixels of the seam, requires at least two observed cubes on each side and 75% projected coverage, and requires an 80% majority for each attribute being tested. Export fallbacks have no vote. An accepted merge takes the union of the two observed masks. The procedure never fills a hole or adds an unseen block.

## Fixed surface comparison

| Rule for joining two candidates | Matched reference surfaces | Surface F1 | Joins between separate surfaces |
| --- | ---: | ---: | ---: |
| Released finder | 1,093 | 75.56% | — |
| Color similarity | 789 | 63.09% | 300 |
| Agreement of projected block IDs | 884 | 66.74% | 197 |
| Agreement of projected face direction and plane coordinate | 1,090 | 76.30% | 6 |
| Agreement of projected plane and ID | 1,090 | 76.12% | 5 |
| Also require the original source normal, relative-depth and seam guards | 1,093 | 75.59% | 0 |
| Supply the true material on each side of each seam | 886 | 67.07% | 196 |
| Supply the true visible connected surface on each side of each seam | 1,094 | 76.64% | 1 |

Each surface has at least 50% pixel intersection-over-union with a unique reference surface to count as a match. Surface F1 balances matches with the number of predicted and reference surfaces. The plane-only rule reduced redundant predictions enough to raise F1 while losing three matches. The reference rows deliberately use the recorded face map. Even a perfect local surface label can disagree with the dominant surface of the entire mask when a proposal already spans a boundary.

The strictest source-only rule triggered on two of 204 views. Courtyard's two pieces belonged to the same wall and the merge recovered 151 paired pixels. On a separate Loading Dock image the estimated planes and materials agreed across **different** reference surfaces, costing a match and 16,995 paired pixels. The unmodified finder remains the default. [Courtyard source and surface overlay](courtyard.png) · [Three cohort summaries](summary.json) · [Every join, input hash and surface score](cases.json)

**Rerunning after the merge exposed a second failure.** We rebuilt the sampling chart from the union, reran the counter, placed and refined the visible cubes, assigned materials and decoded each pack. Both unmodified control runs reproduced the published placement counts. The genuine Courtyard surface repair retained 162 correct positions but changed extras from 13 to 14. On Loading Dock the false merge changed 209 correct / 47 missed / 46 extra to 205 / 51 / 48. Neither gained a complete grid. An improved outline still needs its own trusted sampling chart and a successful depth fit before it changes the right exported cubes. [Closed-loop scores](closed-loop.json)

## What perfect geometry buys the Material Matcher

The four arms use the same matcher and vanilla atlas. The camera-only arm replaces the projection used to read material. The position-only arm replaces the cube set with exactly the blocks having a visible face in that source image. The joint arm supplies both. The recorded camera is transformed into the export's local coordinates using its frozen registration and whole-build translation. Every new Minecraft pack is decoded and scored against the same visible targets.

| Perfect input supplied to the matcher | Fixed strict overall F1 | Fixed facade overall F1 | Separate 28 views, strict overall F1 |
| --- | ---: | ---: | ---: |
| Neither, released output | 84.68% | 88.51% | 86.79% |
| Camera used for reading textures | 85.88% | 89.54% | 87.52% |
| Visible positions | 90.93% | 94.16% | 96.56% |
| Visible positions and camera | **92.35%** | **95.64%** | **97.24%** |

With both inputs supplied, 21,889 of 23,702 fixed IDs are correct. Of 1,813 material mistakes, 915 are at optional horizontal slivers and 898 at required facade positions. Perfect positions make more hard-to-read blocks enter the material denominator. With exact inputs, 35 cells still used the smooth-stone fallback. This intervention measures capacity on this dataset and does not give the runtime access to the capture camera or the reference world. [Pooled scores](material-oracle-summary.json) · [Per-view scores and decoded-pack fingerprints](material-oracle-cases.json)

Run `python -m benchmark.material_surface_feedback`, then `python -m benchmark.material_surface_recount` to recheck the two accepted joins through the counter and placement stages. Run `python -m benchmark.visible_geometry_material_oracle` for the camera and position interventions. All three commands use saved source images.
