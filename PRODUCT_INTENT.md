# Visible-only image-to-Minecraft reconstruction

Jeremy's scope decision, confirmed November 26, 2026, applies to the current reconstruction pipeline and the planned learned approach documented in [training/README.md](training/README.md).

## Intended result

Recover the visible parts of a Minecraft build from the supplied image or images, accurately enough to place the observed blocks in Minecraft and compare the result from the source view. Estimate visible surface boundaries, block grids, dimensions, relative placement and materials from image evidence. Report unresolved measurements and ambiguous identities.

Reconstruct only cells supported by visible block-face evidence in the supplied input. A partly visible face can support its block when the grid and placement are established. Minecraft places whole blocks, so placing a full cube for its observed face is expected; that does not authorize adding neighboring cells whose faces were never observed.

A full input frame showing only a surface texture, with no usable view of how parts of a build relate, is a texture and grid example. Leave its structure placement unresolved. For a surface extracted from a larger building screenshot, use the surrounding view for orientation and placement. On the extracted surface, compare the perspective-corrected texture against the complete atlas tile to measure block spacing and alignment. An individual brick or mortar course can occupy only part of one Minecraft block.

## Unseen areas remain unknown

Do not predict or generate hidden backs, interiors, thickness behind an observed face, occluded continuations, off-image regions or other unobserved geometry. Do not close shells, extrude a facade to an assumed depth, mirror a visible facade onto an unseen side, inpaint missing structure or add hidden supports. Shape or symmetry priors may help interpret visible evidence, but they must not create unobserved cells.

Preserve observed openings and clear sightlines. Keep unknown space distinct from observed air. Omitting an unseen cell from an export means no placement is specified there; it does not establish that the source cell was empty or authorize clearing it in an existing world.

For one-image reconstruction, visibility means visibility in that image. If several images are explicitly supplied together, their combined visible evidence can support reconstruction. Other views held by a dataset or evaluator do not expand a single-image target.

## Training and evaluation

The build may be placed anywhere in the world. Its internal geometry is the objective: block dimensions, distances, heights, openings and the offsets between visible parts. For the primary placement score, give each complete prediction one integer XYZ translation that maximizes overlap with the visible target. Apply the same shift to every component, including disconnected pieces. Keep scale and the registered axes fixed. Absolute coordinates remain useful for diagnosing camera registration.

Full source worlds, block states, camera parameters and renderer outputs may provide training labels and scoring truth. Derive the target from the visibility of each example's actual input image(s). Exclude hidden geometry from reconstruction supervision and from the set of blocks the model is expected to recover. Do not label unobserved cells as air or train full-world occupancy/completion as this project's objective. Source metadata is not an inference input unless the task explicitly declares it.

Score visible surface and grid accuracy, block dimensions, material identity and exported block placements. Count unsupported added blocks as errors, preserve openings, and allow unresolved predictions. A partial reconstruction with accurate visible blocks satisfies the intent better than a completed building containing invented hidden structure. Do not reward visual resemblance alone when unsupported blocks have been added. Keep every view and close variant of a building in the same dataset split.

## Existing behavior and future changes

This document records product intent; it does not certify that every historical command already enforces it. The legacy facade builder's explicit `--depth` shell construction is outside this reconstruction scope. Do not recommend or enable it in reconstruction workflows. Archived experiments and their original results remain historical evidence rather than current requirements.

Apply this scope to future defaults, datasets, model selection, skills, documentation and acceptance checks. See [README.md](README.md), [AGENTS.md](AGENTS.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for the current implementation and [training/README.md](training/README.md) for the proposed learning work.
