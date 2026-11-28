# Floating Perfect Stage comparison

[Open the interactive chart](perfect-stage.html) or [read the scored tables](report.md). Each row compares the original source-only export with an independent reference-assisted replay. Choose **Archived comparison** for 144 original Visible Builds V1 views or **Held-out test split** for the same 12 Twinspire views across handoffs. Decorative Builds V1 and the newer 200-view Floating Combined V1 were outside this replay.

The source-only anchor is Candidate 1, the guarded depth-first refit from the [frozen V1 comparison](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/01d51c0/benchmark/reviews/floating-visible-builds-v1-01). The source-only evaluation table also retains its original split-border Baseline. Reference-assisted rows do not change the reconstruction default.

| Replay | What it establishes |
| --- | --- |
| Native masks | Supply recorded frontmost surface ownership at each pixel. Keep source-model plane estimates and run counting and placement. |
| Masks and measured planes | Also supply recorded facing and relative optical depth per visible plane. This leaves integer inter-plane offsets to the ordinary joins. |
| Visible wall grids | Supply exact projected corners, local count and visible full-cube wall cells. Material matching still reads the source, and assembly still has to determine orientation, relative offsets, joins and component selection. |
| Local heading chooser | Let reference scoring choose among four upright headings for the seven nonempty unresolved source-only exports. This tests the options actually available, with every pack's geometry held fixed. |
| Exact IDs at correct coordinates | Replace material IDs where the decoded source-only export already placed a correct visible coordinate. Geometry stays fixed. This tests the material gap on the existing geometry. |
| Writer check | Pass exact visible coordinates and IDs to the Java 26.3 pack writer and decode the result. A 100% result verifies serialization. |

Exact wall grids gave the largest measured placement gain. A mask alone still leaves the Block Counter to recover scale and pitch. Even exact wall grids leave 10,191 required visible coordinates missing on the strict score, so the results support looking at both local grid quality and which valid charts reach the export. The individual effects of perfect relative joins and component selection are unmeasured. Deltas overlap across rows.

## Saved evidence and refresh

- `controls.json.gz` contains the late-control input checksums, per-view scores, code fingerprints and pack receipts. `perfect-packs.zip` contains each generated pack.
- The [early replay archive](../floating-perfect-stage-inputs-01/README.md) contains actual masks or wall grids, downstream score receipts, per-view placements and decoded pack inputs. Its provenance pins the archived candidate separately from the Floating reporting revision.
- The generated [report.md](report.md) can be reproduced without inference or export. Checksum mismatches require a new control for the changed anchor. A visual or report edit can reuse the existing receipts.

From the Floating checkout, with the original V1 release restored in its neighboring worktree:

```powershell
$release = '../floating-visible-builds-v1/benchmark/reviews/floating-visible-builds-v1-01'
$review = 'benchmark/reviews/floating-perfect-stage-01'
python -m benchmark.floating_stage_report "$release/conditions.json.gz" --run-root ../floating-visible-builds-v1/runs/floating-v1 --cohort-label "Original Visible Builds V1 replay" --visible-available 144 --baseline-name "Expanded split-border" --candidate-name "Guarded depth-first refit" --perfect "$review/controls.json.gz" --perfect-inputs benchmark/reviews/floating-perfect-stage-inputs-01 --out runs/perfect-stage-report
```

Refresh the late replay with `python -m benchmark.floating_perfect_export $release --source-root <restored-V1-release-root> --out runs/v1-perfect`. Use the same private output folder to reuse matching decoded controls. The optional source root contains `data/<split>/frames/<id>`. For the native mask, plane and grid replays, use the pinned command in the early archive. Keep the visible references outside ordinary source-only reconstruction.
