# Sparse worktrees and shared original V1 fixtures

The selective Floating V1 replay passed from a full checkout, a sparse checkout with private fixture copies, and a sparse checkout with one read-only fixture bundle. The three layouts used `9ccf8138cbb7d84ddd2ae5ea4d1582a407fac9a5`, identical code and settings, and the same two original V1 development views. The historical replay at `5a62761fb218005eab3d7d5ca66901cfdf7dcfa3` also passed with the old revision's own scorer and the same fixture bundle. [Input hashes, per-view scores and decoded placement digests](reviews/floating-worktree-fixtures-01/receipt.json)

## What each entry point consumes and produces

| Entry point | Reads | Writes |
| --- | --- | --- |
| Selective replay | Admitted frame records, each RGB, the verified normal and depth field arrays and receipts, archived field provenance, the prepared Java 26.3 atlas, the checked-in calibrated placement index, and reconstruction settings. It prefers the legacy sibling `fixtures.json` when present and verifies its digest. The checked-in gzip produces the same records when the sibling is absent. | Frame-local writer locks, checkpoint receipts, stage generations and handoffs under `--cache`, including imported field copies, masks, crops, counts, chart packets, assembly records and Java packs. `--profile` writes a separate run summary. |
| Selective scorer | Decodes the selected pack, reads its assembly and orientation receipt, then reads `frame.json`, `camera.json`, `labels.json` and `labels.npz`. The frozen visible references determine the targets and the facade allowance. The scorer uses one whole-build integer shift and checks exact block IDs at the aligned positions. | `score.json` under the score checkpoint, including the registered position-to-ID rows and strict and facade placement and position-plus-ID results. |
| Stage report | The supplied `conditions.json.gz`, archived score receipts referenced by its pack or run paths, and optional source handoffs, per-frame depth audits and Perfect Stage receipts. Embedded summaries cover some missing handoffs. | `report.md`, `report.json` and optional `perfect-stage.html` under `--out`. |
| Visual review | The report and conditions, the archived packs and their score and placement receipts, RGB and native visible-face labels from `--frame-root` or `--shard`, and source masks for dropped-chart overlays. Pack and frame hashes are checked. Archive paths referenced by conditions must resolve. | Native-pixel overlay PNGs, `queue.md` and `receipts.json` under `--out`. |

Read and write paths are implemented in [the V1 runner](floating_v1_run.py), [selective replay](floating_selective_replay.py), [checkpoint cache](floating_stage_cache.py), [reference loader](expanded_scores.py), [relative scorer](relative_face_scoring.py), [stage report](floating_stage_report.py) and [visual review](floating_visual_review.py). The pilot regenerated the archived 144-view stage report and four selected visual-review cases from the sparse checkout using the existing archived outputs. A two-view replay bundle contains only the inputs for those two views. Older nonselective runners also use fixed checkout `runs` paths and sibling artifacts. Floating Combined V1 has its own release/shard index and evaluation path. Use the matching checkout and fixture layout for those workflows. [Combined index builder](../training/floating_combined.py) · [Legacy local runner](floating_build_local_run.py)

## Use the tested layout

For the original V1 selective replay, retain code, `benchmark/data`, and the two small archive receipts. Start with a fresh worktree so sparse setup never touches a worktree containing review edits. These patterns keep 1,497 tracked files at the measured revision.

```powershell
git sparse-checkout set --no-cone -- '/**' '!/benchmark/reviews/**' `
  '/benchmark/reviews/floating-visible-builds-v1-01/fixtures.json.gz' `
  '/benchmark/reviews/floating-visible-builds-v1-01/conditions.json.gz'
```

Give the shared fixture root a versioned name and retain its SHA-256 manifest with the release. Use `release/data/<split>/frames/<id>` for captures and visible references, `fields/<id>` for the verified field triplets, and `atlas` for the prepared Java 26.3 atlas. Restore each component from its pinned release or verified donor. The receipt includes the exact hashes used here, including the original release manifest `a702f1d1…`. Mount the fixture read-only or make its files read-only on Windows. Keep the checked-in surface defaults and calibrated placement index with the revision that consumes them.

Run with a benchmark Python that can import NumPy, OpenCV and Pillow. Choose a fresh output directory outside the fixture root for each invocation, including concurrent jobs using the same checkout.

```powershell
.\Invoke-FloatingV1Replay.ps1 -Python $benchmarkPython -FixtureRoot $fixture `
  -RunDirectory $newRunPath -Case train-arcade1-cool-high-right,train-terrace2-cool-low-left
```

On Jeremy's PC the shared fixture root is `C:\code\minecraft\workspace\artifacts\reconstruction-fixtures\visible-builds-v1-a702f1d1-java26.3`. It contains all 144 admitted original V1 frames and their verified source fields. Its `sha256.json` lists the 1,156 input files. When using the bundled Codex Python, add `C:\code\minecraft\workspace\artifacts\reconstruction-fixtures\runtime\opencv-4.12.0.88` to `PYTHONPATH` for OpenCV.

The launcher forwards existing replay arguments and sets `MCIR_PREPARED_ATLAS_26_3` for its subprocess. The single path override uses the existing atlas and calibrated-index validation. An explicit missing atlas fails before the loader searches other locations. For an older revision with selective replay, use its own entry point and atlas discovery. The historical check staged the atlas at that revision's existing shared workspace lookup and passed the release, field archive, cache and profile explicitly. Each run used `charge` orientation, area 5,000, edge tolerance 2.5 px, one worker and all seven stages through scoring. Verified source-field import rebuilt every later stage from scratch.

For original V1 selective replay, use the versioned read-only fixture bundle and the two-file sparse allowlist for replay-only worktrees. Freeze frame IDs, input and scorer fingerprints, and settings in the run receipt. Give each job a private cache, pack and report directory. Use the stage or visual-review entry point with the archived inputs it names. For historical runs, resolve assets against the target revision and give runners with fixed output locations their own worktree or private working copy.

## Measured storage and concurrency

| Layout | Tracked files per checkout | Logical storage for two checkouts and their fixtures | NTFS file allocation |
| --- | ---: | ---: | ---: |
| Full with private fixtures | 6,814 | 1,073.63 MiB | 1,099.77 MiB |
| Sparse with private fixtures | 1,497 | 186.34 MiB | 192.11 MiB |
| Sparse with one shared fixture bundle | 1,497 | 174.68 MiB | 180.41 MiB |

The tracked checkout fell from 525.15 to 81.51 MiB in logical bytes, saving 443.64 MiB per worktree. The two-view bundle is 11.67 MiB logical and 11.70 MiB allocated. Removing its second copy brings the two-worktree savings to 898.95 MiB logical and 919.37 MiB in NTFS file allocation. The table uses the measured file sizes for each layout and excludes the common Git object database, directory metadata and private outputs.

The two stock jobs used separate caches and overlapped for 27.625 seconds. Two jobs through the patched launcher overlapped for 25.297 seconds. SHA-256, inventory, byte sizes, modification times and read-only attributes matched before and after for all 20 shared files. The benchmark commands ran with Minecraft closed. A directory junction failed with access denied on this PC, so the launcher uses the explicit atlas path. The two-view benchmark used archived source fields. It did not measure live inference or total storage for all 144 frames.

## Floating branch scorecard

| Layout and cohort | Correct / missed / extra placement, strict | Strict / facade placement F1 | Strict / facade position + ID F1 |
| --- | ---: | ---: | ---: |
| Full checkout baseline, two original V1 views at `9ccf813` | 187 / 220 / 16 | 61.31% / 61.92% | 57.70% / 58.28% |
| Sparse checkout and both shared-fixture runs, same two views at `9ccf813` | 187 / 220 / 16 | 61.31% / 61.92% | 57.70% / 58.28% |
| Historical `5a62761` and both launcher verification runs, same two views | 187 / 220 / 16 | 61.31% / 61.92% | 57.70% / 58.28% |

Arcade 1 cool high-right exported 154 local positions and selected the partial refit. Terrace 2 cool low-left exported 49 and selected the baseline. Every local position and ID in every assembly arm, every registered position and ID, and all per-view score fields matched apart from ZIP byte hashes. Both views resolved orientation. The layout changes moved zero correct blocks, misses or extras and changed zero block IDs. The launcher is available on `experiment/floating-visible-builds` for original V1 selective replay.
