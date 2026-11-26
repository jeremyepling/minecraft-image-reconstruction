# Should close-up planes be allowed to split an existing wall?

**Recommend split-only zoom for Surface Finder.** It found 516 of 598 scored surface appearances across the same 56 pictures and kept every one of the default’s 513 matches. The occupied area in each picture stayed identical to the default. The ordinary build still selects the 513-surface finder pending the decision to switch. [Zoomable comparison sheet](reviews/crop-split-01/comparison.jpg) · [Results by screenshot](reviews/crop-split-01/pictures.md)

| On the 56 saved pictures | Current default | Split-only zoom |
| --- | ---: | ---: |
| Surfaces found | 513 | 516 |
| Extra guesses | 204 | 204 |
| Incorrectly covered opening samples out of 16,630 | 601 | 601 |
| Visible surface pixels assigned correctly | 94.47% | 94.54% |
| Correct connections between surfaces out of 606 | 418 | 424 |
| Mean boundary / corner error in source pixels | 4.23 / 13.30 | 4.20 / 13.25 |
| Undersides found out of 18 | 14 | 14 |
| Surfaces with both block dimensions correct | 80 | 80 |
| Shopfront and Wing, 20 development views | 137/156 | 138/156 |
| Six previously reviewed buildings, 36 views | 376/442 | 378/442 |

### What changed in the pictures

- **Shopfront, plaster from low left.** The narrow five-block-high side face R15 reached 78% outline overlap, up from 17%. It counts as one additional match.
- **Workshop, masonry and plaster from low left.** The long one-block-high face R1 went from 33% to 70% overlap on masonry and from 32% to 75% on plaster. These are two pictures of the same physical face. The Block Counter still could not recover both dimensions of the newly found faces.
- **Registry, masonry from high right.** Earlier unrestricted zoom claimed 260 of the 482 scored opening samples in this view, including exposed floor between the walls. Split-only zoom and the default both claimed zero. The last rows of the sheet show all three results at the same coordinates.
- **The other 53 pictures** retained their original predictions. The troublesome Crossing wall survived. The Galleries underside still needs better evidence. [Per-picture outcomes and pixel-ownership checks](reviews/crop-split-01/cases.json)

### How the test worked

The candidate used the existing six-model finder to produce the starting masks from one screenshot. Its Pixel-wise Planarity checkpoint also viewed nine overlapping crops of that screenshot. The full-image prediction supplied the camera-space geometry. A crop could divide an existing mask if two regions met the same depth, direction and coverage rules used in the preceding round. Each original surface could split once. The rule was fixed from the earlier review before this replay. Both paths used the same Block Counter. [Frozen settings](reviews/crop-split-01/development-freeze.json) · [Earlier splits and additions](GUARD_ZOOM.md)

The runner generated the splits from numerical model outputs. It then checked that the children of each split shared no pixels and covered exactly the original parent. It also checked the combined surface footprint against the default in **every** frame. Each of the three new matches came from assigning already visible pixels to the right face. The score measures visible surfaces. All eight layouts have been reviewed before, and placement accuracy on a new building remains unmeasured. [Run manifest and input hashes](reviews/crop-split-01/manifest.json)

### Checks from the original screenshots

We ran the full candidate on two renamed screenshots and scored the output afterward. On Workshop it found 7 of 14 surfaces, up from 6 in a separate ordinary build of the same image. On Registry it found 8 of 10 and left all 482 opening samples clear. Every saved region and every Block Counter result on these two images matched the offline replay. The candidate builds took 39.9 and 43.7 seconds on an RTX 4080. Within each build, running and grouping the nine additional crops took about 0.84 seconds. [Workshop candidate](reviews/crop-split-entry-01/workshop-proof.json) · [Workshop default](reviews/crop-split-entry-01/default-workshop-proof.json) · [Registry candidate](reviews/crop-split-entry-01/registry-proof.json)

### Reproduce

```powershell
.\Invoke-SurfaceRound.ps1 -Round crop-split
```

The command uses the saved Java 26.3 clear-noon screenshots, cached numerical fields, and recorded geometry. It independently re-scores every image, runs the Block Counter, and writes a new portrait sheet and manifest under `runs/` with Minecraft closed. The overview keeps one building per column. Two close-ups revisit the newly separated faces, and the opening crops use the same views and coordinates as the preceding zoom review.
