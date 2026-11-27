# Surface boundaries, face directions and placement grids

The control is the ordinary Java 26.3 image-to-build pipeline at `1ca8822`. Compare the same source image, visible targets and decoded export in each arm. Score each proposed change separately on the fixed 104, the 144 new Visible Builds v3 views, and the 40 Decorative Builds v1 views. Report development and held-out families separately within each new dataset.

The fixed replay matched all **104 released material packs byte for byte**, including the 21 that differed in the earlier public-context replay. Restoring the original inset and exterior context and running the downstream placement checks reproduced the ordinary builds. The reproducible command is `python -m benchmark.expanded_fixed_handoffs --workers 4`.

## Questions for the two comparisons

| Change | What the source image must resolve | How to measure the opportunity |
| --- | --- | --- |
| Surface boundary and face direction | At an overlapping pillar, rim or recess, does the surface need to split, face another direction, or both? Test the existing interpretation alongside the alternatives. | Compare the best offered interpretation with the control after placement and material export. Check whether source edges, depth changes or independently visible faces select that interpretation on other buildings. |
| Grid supplied to placement | For a surface with a texture-guided count, which projected grid places its corners on the visible wall? Keep the measured counts available, including when the original grid is used for placement. | Compare the original and adjusted placement charts on matching axes. Score the projected corners and adjoining edges in the source image. Report separately what choosing with the visible reference could recover. |

Use facade-priority placement F1 as the primary comparison. Report strict placement, material ID accuracy on matched positions and both position-plus-ID F1 scores. Each view permits one whole-build integer translation. Report distinct correct positions gained and lost and extras added and removed where the positions are available. A reference chooser estimates the opportunity among its offered options. Only the source-image decision is eligible for an ordinary build.

## Reading a failed build

A rejected camera or missing projection stops the placer. Keep all its visible targets in the overall missed count and identify the camera failure on that image, including the atlas fallback reason when present. Show camera-supported placement separately. A usable camera can still leave an empty export when downstream evidence is insufficient. Execution errors require a retry and must be distinguished from a completed empty build.
