# Reuse Floating stages on original Visible Builds V1

Use `python -m benchmark.floating_v1_run --selective` for verified checkpoint replay. A complete frame has seven checkpoints. The source-prediction stages receive the original image and source-model products. The scorer loads the visible reference after the selected Java pack is written and decoded. The path remains opt-in.

| Stage | Saved output | Examples of changes that invalidate it |
|---|---|---|
| fields | Normal and relative-depth arrays, source/model hashes and provider | RGB, model weights, preprocessing, producer code |
| surfaces-counts | Surface masks, crops, source counts, build manifest | Surface settings and model code |
| native-charts | Dense atlas, shared corners, native and connected atlas seeds | Atlas, native matcher or shared-row code |
| expanded | Expanded and border-filtered chart packets | Contraction, edge ruler, pair, sky or split rules |
| pregrid | Fresh complete and partial source grids, depth partitions and evidence | Regional refit, minimum wall area |
| assembly | Baseline and fitted local Java packs, selected source component and assignment evidence | Edge tolerance, joining, visibility filter, export or component selection |
| score | Strict/facade placement and position-plus-ID, orientation resolution and registered positions | Reference content, scoring code or orientation policy |

Source masks and counts are one ordinary producer. Dense grids, seams, native registration and seed extension likewise run together. Use the listed stage names to force either group. Confidence-based component selection remains part of assembly.

Each frame has its own OS writer lock. A successful stage publishes its immutable output generation and its content hashes. The next consumer reads that generation. A failed build can resume from the last verified checkpoint. The last completed pipeline pointer is published atomically. Warm runs re-hash files and compare source/settings, actual executed project functions and module-level configuration, runtime versions, model and atlas provenance, and parent fingerprints. Invalidating a stage also reruns all later consumers, including after a forced rebuild that reproduces the old bytes. Old generations remain available for comparison.

An explicit field import accepts the original V1 source field arrays only when the published fingerprints, frozen inference source code, model weights, preprocessing and RGB all match. `--rerun-from fields` forces real inference. Existing unverified source charts and old per-frame score rows are never used as cache hits. Changed contents of an immutable V1 fixture stop the run at validation. A new fixture requires its own admission record.

## Run a frame or inspect a handoff

Use the benchmark Python (3.12+) and prepared Java 26.3 atlas. `--release` can point to the original checksum-verified V1 folder when its installation has moved. `--field-archive` imports saved source fields without running ONNX when eligible.

```powershell
$frame = 'train-arcade1-cool-high-right'
$common = @('--selective','--case',$frame,'--release',$v1Release,'--field-archive',$v1Fields)
python -m benchmark.floating_v1_run @common --profile runs/cold.json
python -m benchmark.floating_v1_run @common --profile runs/warm.json
python -m benchmark.floating_v1_run @common --rerun-from assembly --profile runs/assembly.json
python -m benchmark.floating_v1_run @common --through pregrid
```

Repeat `--case` for more frames and set `--workers` for parallel case execution. The default corpus for this **original V1 reproduction entry point** is the archived set of 144 eligible whole-structure views. Use the combined Visible Builds-only index for ordinary development. This tooling does not change that default. No Decorative or Orchid image was used in the checks below.

## Measured replay

The verification used an archive of integration revision `640dea9` plus the 36 new benchmark files listed below. Both frames used original source RGB and known source-only fields. Every subsequent source stage ran from scratch. Arcade 1 cool high-right verified a fresh chart and selected the partial refit. Terrace 2 cool low-left retained expanded split-border, including its vetoed sky cell. Both local Java pack dictionaries and all registered positions and IDs matched the published candidate.

| Two-frame check | Stages reused | Stages rebuilt |
|---|---:|---:|
| Cold | 0 | 14 |
| Identical warm run | 14 | 0 |
| Change scorer orientation policy | 12 | 2 |
| Force assembly including downstream score | 10 | 4 |
| Raise minimum wall area above both source masks | 8 | 6 |
| Restore default minimum wall area | 8 | 6 |

On Arcade the high minimum area changed the winner from 154 correct to 28. Restoring the default recovered the published 154. The per-stage profiles include elapsed seconds for debugging. No timing benchmark was performed. Focused tests also check modified output bytes, interrupted steps, a shorter prefix run, settings/code fingerprints and the per-frame lock.

[Profiles, source hashes, local and registered export comparisons](proof.json.gz) · [Portable integration file list](integration-files.txt). Every path in the list is new relative to `640dea9`. The optional export-UV adapter filters only the explicit V1 replay input. Integration's existing solver, cohort code, reports and model defaults remain in place.

## Floating branch scorecard

| Method and cohort | Correct / missed / extra | Strict / facade placement F1 | Strict / facade position + ID F1 |
|---|---:|---:|---:|
| Baseline — archived Arcade 1 and Terrace 2, two original V1 frames | 187 / 220 / 16 | 61.31 / 61.92% | 57.70 / 58.28% |
| Selective replay on those same two frames | 187 / 220 / 16 | 61.31 / 61.92% | 57.70 / 58.28% |
| Baseline — published guarded pre-grid, original V1 144, carried forward at `01d51c0` | 10,156 / 24,298 / 6,974 | 39.38 / 41.37% | 17.06 / 18.54% |
| Baseline — expanded split-border, floating integration 84, carried forward at `640dea9` | 18,318 / 61,192 / 9,965 | 33.99 / 34.48% | 5.66 / 5.76% |

The measured two-frame tool check has the same scores as the identical archived frames, with zero change in correct blocks, extras or exact IDs. Recommend integrating the benchmark-only replay tooling into Floating. The model behavior and full-cohort results above are carried forward.
