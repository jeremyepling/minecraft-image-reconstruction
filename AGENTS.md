# Single-image exterior experiments

Use the versioned fixture release in `benchmark/suites/exterior-v1.json` for camera, visible-face/depth, material and placed-build comparisons. Reuse its individual frame IDs, floor pairs and layout groups. The reference captures are vanilla Java 26.3, clear weather and noon (6000 ticks). Each ordinary prediction receives one framebuffer and the selected game version. Label any run using recorded poses, face labels or true endpoints as a diagnostic.

From this checkout, run `python -m benchmark.exterior_suite` for the Stage 1 regression comparison. During development, `--split dev` and `--floor light_gray_concrete` isolate the development weak-ground case. Compare a changed method on both floors before reporting a result. See [the suite guide](benchmark/EXTERIOR_SUITE.md) for reference scores, release updates, review records and replay instructions.
