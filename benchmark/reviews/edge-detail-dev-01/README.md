# Development decisions for native outline alignment

The ordinary Surface Finder at `79365b9` supplied the same starting surfaces in every run. These are per-proposal native masks. Existing overlaps, sky and enclosed openings are protected during the photographic adjustment. Curves can move inside a narrow band only when local photographic contrast supports the new edge. The original connectivity and holes of the label map are preserved.

| 20 Shopfront and Wing views | Native boundary error (px) | Matched surfaces | Correct sizes | Wrong sizes |
|---|---:|---:|---:|---:|
| Shipped finder | 5.0809 | 138 | 17 | 13 |
| 2-pixel photo collar | 5.0509 | 138 | 17 | 13 |
| 4-pixel photo collar | 5.0035 | 139 | 16 | 14 |
| 3-pixel photo collar | 5.0152 | 139 | 17 | 13 |
| 4-pixel collar with count fallback | 5.0056 | 139 | 17 | 13 |
| Selected collar, then Planarity additions | 4.9839 | 139 | 17 | 13 |

The first two settings tested whether a native image edge could improve the mask after the last Surface Finder filter. The broader collar reduced the most native edge error, but changed a previously correct Wing masonry count at high-right on concrete. Its changes added 6,407 correct and 1,169 wrong pixels while removing 558 correct and 6,435 wrong pixels on the fixed matched IDs. Most wrong additions crossed into a neighboring recorded plane (1,065), with 104 on background. [First round](edge-first/scores.json) · [Pixel breakdown](native-edit-types.json)

The second round therefore tested an intermediate 3-pixel collar and a fallback for the 4-pixel collar. If the unchanged Block Counter fails to reproduce a previously resolved width and height for any changed proposal, retry the entire frame with the 2-pixel collar, then keep the original frame if necessary. A whole-frame fallback keeps shared boundaries consistent. Both challengers had 139 matches and preserved the exact counts. The 4-pixel fallback had the smaller native edge error, higher paired overlap and one fewer extra proposal than radius 3. Its 33 incorrectly occupied opening samples were the same as the baseline. This method was frozen before the 36 remaining views were scored. [Second round](edge-second/scores.json)

One extra development trial added the frozen two-pixel Planarity addition rule after the selected photographic change. The model saw the original screenshot. The Planarity rule could add only pixels already owned by a visible proposal. On the same 20 dev pictures this improved native edge error by another 0.022 pixels and improved overlap from 95.752% to 95.780%, while preserving all matched IDs and exact sizes from the photo candidate. These settings were fixed before running this combination on the full suite. [Composition development](edge-plus-plane-dev/scores.json) · [Frozen full comparison](../edge-detail-stack-01/scores.json)

Each run folder includes the per-picture results, prediction archive, score summary and manifest with its exact replay command. The additional native-pixel breakdown is reproducible with:

```powershell
python -m benchmark.edge_detail_dev --settings native2 native4 --out runs/edge-pass1.json
```
