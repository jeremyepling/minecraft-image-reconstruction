# Exterior regression suite

Run `python -m benchmark.exterior_suite` from the repository root. This replays 40 frozen Java 26.3 screenshots from the two floor conditions, checks the inputs and capture conditions, scores the current Stage 1 method, and produces a paired comparison sheet. It runs offline after Python and its dependencies are installed. The published reference used CPython 3.12 and the packages in `benchmark/suites/requirements.txt`. [Release manifest](suites/exterior-v1.json) · [Recorded reference run](reviews/ground-floor-01/manifest.json) · [Paired comparison](reviews/ground-floor-01/comparison.png)

## What is fixed

The release is `exterior-v1`. Four layouts each have two palettes and two oblique views. The wing and courtyard also have a close frontal view per palette. Each of these 20 source views appears once on smooth stone and once on light gray concrete. Every native frame is 1440 × 900. The saved clock is noon (6000 ticks) with weather clear, thunder off and vanilla resources. The world generator and the client fix time and weather before capture. The loader checks each frame's recorded conditions before running a method.

The release records SHA-256 fingerprints of both indexes, every referenced image, camera pose, visibility label and state readback, plus the cross-world pairing audit. Pairs use stable frame IDs. Block states, light readings, true camera and visible labels match across the two floors. The four facade-only controls were pixel-identical. The weakly marked concrete surface preserves normal rendering and building shadows. [Pair verification](data/camera-gray-concrete-v1/pair-verification.json)

These four layouts are now regression examples. `dev` and `holdout` retain their original round-one meanings for historical comparisons. They no longer provide a blind test of improvements inspired by these images. Confirm such improvements on additional, previously unseen layouts, grouping all their floors, angles and palettes together.

## Run a variation

```console
python -m benchmark.exterior_suite
python -m benchmark.exterior_suite --floor light_gray_concrete --split dev
python -m benchmark.exterior_suite --method tile-lines
```

Each invocation creates a fresh directory under `runs/exterior/`. Use `--out NEW_DIRECTORY` for a named run. The manifest records the actual method, settings, code revision, source fingerprint, suite digest, environment, cohort and per-floor scores. `delta.json` compares the same cohort, floors and metric definitions with the committed reference. Each predicted camera receives one RGB framebuffer and the selected game version. Paired labels and alternative views stay with the evaluator.

The unit-grid scores measure absolute placement in blocks. The labeled-endpoint diagnostic measures width divided by height from predicted perspective and three supplied image points on the same wall. Its estimate does not use the known block lengths or true camera. The separate automatic ratio uses the wall extent the predictor located itself. The summary counts all eligible ratios, returned estimates and estimates within 10%, and gives error on returned estimates only. Each layout has equal weight in the conditional mean. Some layouts can have zero returned estimates.

Use one baseline and at most two challengers per review. Compare one stage at a time with the same release and scoring protocol. Stages for faces/depth, material choice and placement can import `benchmark.exterior_set.load_suite()` and evaluate the same registered cases as they become available. For future capture additions, give the expanded corpus a new release ID and keep `exterior-v1` available to reproduce published scores. A change in metric definition gets a new protocol ID and its own reference scores.

### Initial floor comparison

The original Stage 1 default, `seam-fit`, ran on each of the 20 views in each world. On the two concrete-floor masonry close-ups it recovered eight supplied-endpoint ratios within 10%. On one low oblique concrete-floor view it returned eight more, averaging 115% error. It abstained on the remaining 200 labeled ratios. The concrete control exposes both absent scale and occasional incorrect relative perspective.

| All four layouts | Smooth stone | Light gray concrete |
| --- | ---: | ---: |
| Absolute camera returned | 16/20 views | 0/20 views |
| Width-to-height ratios returned with endpoints supplied | 208/216 | 16/216 |
| Supplied-endpoint ratios within 10% of truth | 208/216 | 8/216 |
| Fully automatic wall ratios returned | 30/36 | 0/36 |
| Fully automatic wall ratios within 10% of truth | 2/36 | 0/36 |

The GitHub Actions workflow replays the suite when benchmark or predictor code changes and uploads the run artifacts. A change in score appears in the comparison, including lost coverage. Visual feedback stays in the run's `feedback.jsonl`, keyed by run, method preference, condition, source image, native-pixel rectangle and stage. Use `python -m benchmark.camera_feedback --help` to record it.

## Recreate or open the worlds

Both built worlds and their layout coordinates are listed in the [world archive index](worlds/exterior-v1/world-index.json). To create another independent capture, use the matching JDK 25, installed Minecraft assets and a fresh directory.

```console
python -m benchmark.camera_fixture --floor light_gray_concrete --out runs/fresh-concrete --paired-with benchmark/data/camera-round-01
```

Use the frozen screenshots for score comparisons. A fresh capture is useful for auditing acquisition, since cloud motion or renderer differences can change pixels even with noon and clear weather locked.
