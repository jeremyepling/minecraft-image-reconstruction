# Where the reconstruction work lives

`main` is the checkout for ordinary builds. The development line published as `feature/single-wall-depth` carried the previous camera, plane, perimeter and count rounds forward. Its name describes one experiment along the way. The permanent commit links below identify the snapshots even as branches move.

| Work | Where to find it |
| --- | --- |
| Earlier deterministic camera, projected grids, visible geometry and material tests | [Architecture](../ARCHITECTURE.md) and [camera round](CAMERA_ROUND.md) |
| Region masks, the historical Rectangle Finder (B), and the two line-model comparisons | [Four-way review](MASK_ROUND.md) |
| Stage names, perspective correction and counting repeated textures | [Surface Finder](SURFACE_FINDER.md) and [Block Counter](COUNT_ROUND.md) |
| True mask perimeters, holes and review sheets | [Perimeter round](PERIMETER_ROUND.md) |
| Five-model Wall Recovery, then the 501-surface baseline | [Depth comparison](SINGLE_WALL_DEPTH.md) |
| Six-model Surface Finder, 513 matched surfaces on 56 views | [Tested default at 1b00db1](https://github.com/jeremyepling/minecraft-image-reconstruction/commit/1b00db13aecdd170e5867601d987987560da0d72) and [Planarity and ECT comparison](AFTERPASS_MODELS.md) |

The Planarity, ECT and comparison-sheet experiments were parallel assignments collected on the former `feature/single-wall-depth` branch. Their code and reviews live on `main`. The registered `feature/normal-round` checkout holds the frozen exterior suite on this PC. The `feature/exterior-suite-default` branch is the source of the original fixture pull request. [Fixture pull request](https://github.com/jeremyepling/minecraft-image-reconstruction/pull/1)

Use the linked benchmark runners to revisit an experiment on the saved images. For an original command such as `inspect-form`, `inspect-openings` or `fit-voxels`, use the snapshot immediately before those commands were retired. The same snapshot supports historical build flags printed in older reports. [Prototype CLI and source at ca0a60c](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/ca0a60cfd03fb9b6bfddfbbd09d7db40da678710)

The separate model-research conversation proposed line-model experiments. The four-way review records their measured outcomes. The separate image-rebuild work tested masonry textures on different images. Its exact texture scores and the Surface Finder’s visible-surface counts measure different things.

The local prototype commits for streaming capture, tracing lines and matching masonry were integrated along with the model work. Their history includes `5b9fc3c`, `06655c6` and `cf87010`. Experimental masonry matching remains available in the build that supports it. The ordinary Surface Finder and texture-based Block Counter retain their own measured defaults.

The single-inset and two-model agreement trials were parallel assignments in this chat. Their scores and settings live in the [acceptance review](PLANE_ACCEPTANCE.md). Run them from `main` with the benchmark command in that review.
