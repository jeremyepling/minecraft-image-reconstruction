# Finding the edge of a smooth wall

[Explore the four layouts in the interactive viewer](../shared-face-round-01/viewer.html). Choose a building under **Compare smooth walls**. The **Smooth-color proposals** layer highlights unassigned pixels that pass the initial color-continuity check. **Surface Finder** shows the accepted outlines. **Outline issues** marks unusually jagged outlines in pink. **Projected build** shows the final block placements on the source image.

## What the colors can tell us

On the previous default, the 14 plaster views of Shopfront, Registry, Portico and Courtyard missed 452 required blocks and scored 86.27% facade-priority F1. The 14 masonry views of the same layouts, angles and floors missed 116 and scored 94.17%. Repeated texture provides block spacing and depth evidence. Pale concrete supplies little of either. A cut inside the wall can follow a slight lighting change, and an outer edge can disappear where the horizon meets a similar gray wall.

| Layout | Fixed views | Plaster facade F1 before → after | Masonry facade F1 | Required blocks recovered |
| --- | ---: | ---: | ---: | ---: |
| Shopfront | 8 | 84.48% → 84.78% | 86.27% | 3 |
| Registry | 4 | 86.09% → 91.25% | 95.81% | 43 |
| Portico | 8 | 84.12% → 84.58% | 97.54% | 5 |
| Courtyard | 8 | 89.67% → 89.67% | 97.00% | 0 |

All scores pool visible coordinates under one whole-build translation per source image. Roof slivers falling under the facade-priority threshold are optional. All extras count. The eight changed exports across the entire fixed suite added **77 required positions**, lost zero and added zero extras. The other 100 checked views retained their placements. [Decoded exports, scores and source hashes](../smooth-local-placement-01/summary.json)

## Where the pipeline should draw a line

**Shopfront.** In the central gray patch beneath the wooden eave, 2,676 of 3,300 visible wall pixels originally had no mask. There was no second fragment to merge in that gap. The broad smooth-color trial reached all 2,676. Separate orange specks inside the narrow wooden underside responded to a different check. Where the ring of underside pixels was continuous and the texture agreed across the hole, the finder recovered 128 correct underside pixels across the two Shopfront floor variants. Some equally small surfaces elsewhere are genuine. [Central plaster patch](plaster-gap.jpg) · [Eave close-ups](../shopfront-eave-01/eave-closeups.png)

**Registry.** In the plaster high-right view, the upper right wall lost 15,833 of its 31,895 visible pixels. The orange line extending beyond its silhouette mostly followed distant pavement. The lower right wall sits one block farther in, so joining both right-facing walls would erase a real ledge. The broad growth trial reached 15,788 of the missing upper-wall pixels, and the local placer recovered 43 required blocks in this image. [Right wall close-up](registry-right.jpg) · [Masks and recorded same-plane boundaries](shopfront-registry.jpg)

**Portico.** The smooth-wall trial helped the plaster high-right view. A separate repeating-texture check joined two pieces of the masonry roof and recovered an exact surface match. Both sides of the seam had to agree on the camera-projected texture pattern. A broader rule based on smooth appearance alone lost a real surface on Canopy 08. [Texture-seam trials](../shopfront-seam-continuity-01/README.md)

**Courtyard.** On the high-right smooth-stone floor, the plaster view missed 33 required blocks and the matching masonry view missed none. The large rear wall has 67,771 visible pixels. On plaster, 34,660 were outside every original mask. On masonry, only 205 were outside. The teal region in the close-up marks an existing piece of that wall. The orange region shows where a future candidate needs to extend while keeping the nearer lip separate. [Rear wall with both palettes](courtyard-high-right-wall.png)

## Which edits reach the build

1. The finder flags masks with repeated short, sharp turns. Across the original 104 views it flagged 46 of 1,439 masks. It caught genuine wall fragments as well as background guesses. Registry's false straight cutoff illustrates why this warning alone cannot find every bad boundary.
2. A smooth-color candidate starts at observed mask pixels and proposes a connected, similarly colored patch of unassigned pixels. The material check compares both the patch and its neighboring source pixels with smooth and textured blocks in the Java 26.3 atlas. A separate comparison samples the frame margins at the same height for pavement or sky. Distant pavement passed the atlas check alone on Watchpost.
3. For placement, the ordinary build stays in place while a second candidate proposes cubes. A new cube needs at least 16 first-visible source samples and at least 60% of its visible samples on newly supported pixels with the expected face axis. Rebuilding the entire layout from expanded masks lost 41 required blocks on the other plaster Registry angle. The local rule preserves all of them.

[Controlled smooth-extent experiments and replay](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/experiment/shopfront-outline-bites/benchmark/reviews/shopfront-outline-gaps-01/README.md) · [Eave repair and placement controls](../shopfront-eave-01/README.md)

