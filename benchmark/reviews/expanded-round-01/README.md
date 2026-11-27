# Geometry and material trials across the three datasets

The control is the ordinary Java 26.3 image-to-build pipeline at `1ca8822`. Compare the same source image, visible targets and decoded export in each arm. Each comparison uses all 104 fixed views, all 144 Visible Builds v3 views, and all 40 Decorative Builds v1 views unless an individual diagnostic specifies a smaller sample. The reports also separate the development and held-out building families.

The fixed replay matched all **104 released material packs byte for byte**, including the 21 that differed in the earlier public-context replay. Restoring the original inset and exterior context and running the downstream placement checks reproduced the ordinary builds. The reproducible command is `python -m benchmark.expanded_fixed_handoffs --workers 4`.

## What reached the exported builds?

[Compare the source-image gains, score changes and oracle choices](../pipeline-impact-01/viewer.html#experiment-gains). The primary geometry score is facade-priority placement F1. The strict score includes every visible position. Position-plus-ID F1 requires both the block and its material. Each build gets one integer translation for relative placement.

| Experiment | What the measured gain tells us | Decision |
| --- | --- | --- |
| Surface boundaries and face directions | A pillar in Stable Yard was two blocks in front of a wall facing the same direction. Splitting at the visible seam improved facade placement 76.01% → 79.03%. Both decorative gains were on training images. V3 validation fell 19.95% → 19.89%. | Keep experimental. Test whether a visible return face or occlusion edge confirms a discontinuity before splitting. [Boundary trial](../visible-boundary-direction-01/README.md) |
| Choosing a placement grid | Keeping the count and changing only the grid corners reproduced the Twinspire gain. The source-only chooser missed that gain and took the Workshop loss. All expanded exports tied across the offered charts. | Keep the existing placer and counter retry. Additional training needs distinct decisions that can actually change the build. [Grid trial](../placement-grid-choice-01/README.md) |
| Estimating a shared 3D grid from full textures | A whole sprite can measure a slab's width. It leaves open which cell owns the slab's fractional-height edge. Only eight of 1,630 partial targets offered fragments from different source masks that could be linked; the grouping rules correctly joined none. | Keep the occupancy and chart feedback experimental. Evaluate more independently localized partial faces and explicitly compare same-cell and adjacent-cell explanations. [Original grid and partial-face tests](../full-block-lattice-01/README.md) |
| Using the structure grid only after the pavement camera fails | This follow-up recovered 278 correct relative positions on v3 and 34 on decorative, with 52 and ten new extras respectively. Some sparse fragments match another repeated panel after whole-build alignment. | Preserve as a separate comparison. Check the identity of photographed anchor blocks and the projected junction in addition to relative F1. [Paired rescue example](../full-block-lattice-01/floor-rejection.jpg) |

The per-image perfect chooser may use reference labels to select among the packages the method actually produced. Its score estimates opportunity within that choice set. Optimizing facade placement alone can give a lower material score. The exact pooled geometry ceiling is separately identified where it was computed. Each ordinary source method uses only the source image and versioned game assets.

## Reading a failed build

A rejected camera or missing projection stops the placer. Keep all its visible targets in the overall missed count and identify the camera failure on that image, including the atlas fallback reason when present. Show camera-supported placement separately. A usable camera can still leave an empty export when downstream evidence is insufficient. Execution errors require a retry and must be distinguished from a completed empty build.
