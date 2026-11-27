# Harbor surface review

Eight new per-surface annotations were verified against the saved source image and both handoff hashes. The cumulative export also included the 13 Castle annotations already on this branch. The audit compares native atlas polygons with the recorded frontmost face at each predicted center. Material counts in the audit are visible pixels within each mask.

## What the labels establish

- On S01 all six proposed copper faces occupy only 23–26% of the associated true face area. On S02 the twelve proposals occupy 3–9%. Both masks also contain prominent mangrove planks and multiple south-facing planes. Agreement on direction alone cannot establish a common depth.
- S04 supplies a positive control. All four atlas proposals match the recorded prismarine-brick ID and exceed 50% overlap with the recorded face. The review identifies more visible blocks beyond that patch.
- S05, S06 and S07 each contain both white concrete and dark prismarine. The matcher abstained before accepting a pitch on either pair of axes. Preserve the user's S07 preference in its original comment: the counter warp looks better on this surface.
- S08's two partial cut-copper proposals cover about half the true face area. They did not pass the neighbor requirement. The visually matching copper variant and exact waxed block ID should be recorded separately in diagnosis.

## Smooth-material context

Jeremy observes that a smooth block can be hard for a human to identify or count without looking at the surrounding blocks. Test using a verified textured neighbor or visible boundary to carry grid spacing onto the smooth region. Check surface continuity and depth before transfer. Track confidence in the geometry separately from confidence in the material. A visually ambiguous cell can support a placement even when its exact material remains uncertain.

The recorded materials and coordinates were used for this audit only. No placement behavior changed.
