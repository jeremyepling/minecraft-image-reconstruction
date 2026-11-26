# Version calibration

Follow the [visible-only product intent](../PRODUCT_INTENT.md) when reusing captures for learned reconstruction. Full world states and exact cameras can generate aligned labels, but the target for each example must be masked to the evidence visible in its supplied input. Other captured angles must not turn a single-image example into a hidden-geometry target. Keep unobserved cells unknown and outside occupancy/material losses; confirmed clear sightlines are distinct from occluded or out-of-frame space. No hidden-area completion objective is part of calibration or planned reconstruction training.

Use a new output directory for each client version and calibration run. The main project CLI orchestrates the steps. To run them individually from the repository root:

```console
python -m runtime.capture registry --version 26.3 --out runs/26.3/registry.json
python -m calibration.fixture --version 26.3 --jar PATH_TO_CLIENT_JAR --registry runs/26.3/registry.json --out runs/26.3
python -m runtime.capture capture --version 26.3 --run runs/26.3
python -m calibration.evaluate runs/26.3 --atlas PATH_TO_SAME_VERSION_ATLAS_JSON
```

The pack function selector is `palette_lab:page_N`, where `N` is a page in `views.json`. The fixture contains one independently shuffled physical placement per registry ID in each split, plus a contiguous 20 × 12 masonry panel per split. The train camera is frontal. Validation and test have distinct positions and oblique angles. Minecraft supplies a rendered frame, actual camera and block-state readbacks at frozen clear noon. Default unmodified vanilla assets are the calibration target.

Evaluation checks the jar fingerprint, atlas pixels, version in each frame, runtime block registry, split disjointness and readbacks. Unstable or invisible states are recorded separately. The whole-block ID task scores every visible default-state ID. The face-texture task combines equivalent faces of different IDs and limits training to model-resolved untinted south faces. Partial-model ID crops include their mounting board. Both tasks have separate libraries.

For stage one, the contiguous panels provide a known plane outline and approximate scale. The grid estimator starts with errors in spacing and origin. Independent camera projection supplies truth at each internal seam. `evaluation.json` reports the fraction within one tenth of a block, the median and 95th-percentile pixel error, and recognition accuracy from the fitted crops. Panels include runs of identical material and adjacent contrasting material. This tests within-panel geometry after locating a wall.

For stage two, validation selects among fixed color, aligned-pattern, multiscale and mottling combinations using real frames and a softened, tinted version. The independent test set records all predictions. `atlas-comparison.json` compares rendered exemplars and local atlas sprites on the same opaque candidates with the same descriptor. An exact default ID can share its appearance with another ID. Library labels and the jar's default cube-face model mappings keep ID and texture ambiguity explicit. `face_textures/softened-library.npz` includes the selected method, training tiles, class labels, target version and atlas digest.
