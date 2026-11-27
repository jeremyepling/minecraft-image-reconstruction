# Planned Minecraft reconstruction training

Use the [reconstruction adjustment log](../MODEL_CHANGELOG.md) when choosing training examples and evaluation cases. It tracks observed failures and measured changes from the existing pipeline.

The active training dataset is available in [Visible Builds v2](DATASET.md). It contains exact screenshots, per-frame visible blocks and camera calibration. The proposed learning experiment below uses the same [visible-only product intent](../PRODUCT_INTENT.md) as the existing pipeline.

## Objective

The origin of a reconstructed build is arbitrary. Evaluate visible positions after one shared translation of the complete build. Preserve the measured offsets between roofs, walls, posts and disconnected pieces. Keep source-world coordinates in the dataset for visibility, camera calibration and reproducibility. [Shared placement criterion](../PRODUCT_INTENT.md#training-and-evaluation)

Improve measurements needed to reproduce the observed build: visible surface boundaries, projected block-grid alignment, block dimensions, depth in block units, relative surface placement and material identity. Start by evaluating a Minecraft-specific block-grid predictor against the existing Block Counter. Adapting a pretrained vision encoder is a proposed first experiment, not an implemented or validated result.

Do not train this project to complete backs, interiors, occluded continuations, off-image regions, assumed thickness or other hidden structure. Whole-building occupancy prediction, generative completion and hidden-area inpainting are outside the objective. A full Minecraft block may be placed for its observed face; additional unseen cells must not be generated to close or support it.

## Training data

Use the whole-structure framing rule for every training task, including texture and block counting. A full screenshot must contain the entire build and the required margin on every edge. A local measurement may sample a surface within an admitted screenshot while retaining its parent view and spatial context.

Source worlds can contain complete structures. For each example, capture the image together with aligned camera parameters, visible depth, normals, surface IDs, projected block corners and visible block-state labels. Use full world data to determine visibility and generate correct labels; restrict reconstruction targets to evidence visible in that example's supplied input view(s).

Represent visible occupied cells, observed free space and unknown space separately. Unknown cells must not become air labels or hidden occupancy targets. Different camera views may form separate single-image examples, but each target must use only its own visibility. A multi-image example can use the union of its explicitly supplied views. Keep all views, palettes and close structural variants of a building in one split. Held-out world state and camera truth belong to the scorer, not ordinary prediction inputs.

Use known world labels as ground truth when available. Other models can provide image-derived features or candidate predictions; agreement alone is not proof of correctness. Human review should verify capture alignment and selected failures rather than approve every automatically labeled block.

## Proposed first comparison

1. Audit aligned labels on a small collection of full-cube structures with openings, overhangs and varied dimensions.
2. Use the pilot's family-level training, validation and test assignments. Add player-authored examples in a future version before interpreting performance on player builds.
3. Train a small grid-prediction module over a frozen pretrained encoder; consider selective fine-tuning only after the baseline is measured.
4. Compare against the current counter using identical input images and surface evidence. Grade grid-corner error, exact visible dimensions and correct and incorrect exported cells. Report teacher-feature and camera assumptions.
5. Check the combined image-to-build result. Verify that export preserves visible openings and adds no unsupported cells. Report unknown measurements explicitly; do not fill gaps to improve appearance.

Use the existing eight-building suite as a regression collection. Its repeatedly reviewed buildings are not an untouched generalization test. GPU time, dataset size and expected accuracy remain estimates until measured on the selected architecture and capture system.

The [reported placement issues and region scorecard](../benchmark/reviews/visible-error-budget-01/README.md) supply examples for future depth and orientation training. Useful comparisons include coplanar changes of material versus overlapping posts, recesses versus protrusions, and a photographed roof edge versus a shaded wall. Evaluate the selected depth on the first visible block along the affected rays. Keep these reviewed structures in their existing splits and score generalization on separate building families.

## Acceptance

Progress means more correct visible blocks, fewer incorrect or unsupported placements, better dimensions and material identification, and preserved openings on new buildings. Hidden world truth is excluded from required recovery. Any unsupported added cell is an error even if it makes a building look plausible. No completion model or full-world occupancy objective should enter this plan as an automatic next stage.

Read [AGENTS.md](AGENTS.md) before implementing the training work and [the current architecture](../ARCHITECTURE.md) before integrating a model.
