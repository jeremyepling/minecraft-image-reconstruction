# Planned learned reconstruction work

Read [the shared product intent](../PRODUCT_INTENT.md), [the learning plan](README.md), [the published dataset contract](DATASET.md) and the repository's [AGENTS.md](../AGENTS.md). Preserve the v1 screenshots, checksums and family assignments when extending the dataset. Model training is a subsequent experiment.

The user explicitly excludes predicting hidden areas. Train and evaluate recovery of visible surfaces and blocks only. Derive each target from visibility in the actual supplied image(s), even when the complete source world and other views are available. Keep unknown cells separate from observed air. Do not introduce completion, inpainting, hidden thickness, shell closure or extra unseen support blocks. Placing a whole Minecraft cube for an observed face is valid.

Keep source worlds and camera metadata available for labeling and scoring, without leaking them into ordinary evaluation predictions. Preserve building-level splits. Compare a learned stage with the existing stage on identical inputs, and evaluate the resulting visible block placements as well as intermediate measurements. Count unsupported added cells as errors and retain unresolved outputs rather than filling unseen space.

Use the shared `minecraft-algorithm-benchmark` skill for controlled comparisons. Preserve historical results and mark proposed work, diagnostics and implemented behavior accurately.
