# Floating-build reconstruction

`experiment/floating-visible-builds` is the long-lived integration branch for reconstructing visible blocks with no ground reference. One translation can place the entire build anywhere. All visible components retain their relative spacing. Production `main` at the fork was `e3bd220`.

## Baseline to beat

The connected-atlas baseline transfers shared rows across supported corners, registers native atlas matches to the grid, and proposes grids on additional visible wall patches where neighboring atlas observations agree. It scores **12.64% strict and 12.90% facade placement F1** on six open-sky composites. Strict/facade position-plus-material F1 is **4.68% / 4.77%**. It places 273 correct blocks, misses 3,768, and adds five extras. The preceding registered-seam baseline placed 239 correctly with five extras. The gains came from Walled Garden and Harbor. [Garden experiments and source overlays](reviews/garden-feedback-01/README.md)

Four views are from Visible Builds v3 and two from Decorative Builds v1. Use the fixed 104 production trend (91.27% strict, 93.42% facade) for historical context. Assess the gap to production using the same buildings, backgrounds, inputs and scoring rule on both branches.

The default sky check rejected 23 of 216 proposals across six views, removing 81,896 background pixels from the combined surface masks. All 193 retained supports and their source crops matched their prior versions. The six decoded placement exports matched in coordinates and materials. [Filtered pipeline replay](reviews/sky-mask-feedback-01/filtered-pipeline-replay.json.gz)

### Reproduce the six exports

From this checkout, using the benchmark Python environment and the saved open-sky fixtures, source masks/counts, and dense atlas chart handoffs on this PC:

```powershell
python -m benchmark.floating_build_local_run ../../minecraft-image-reconstruction/runs/floating-build-grid-01/fixtures.json --sky-handoffs-only
python -m benchmark.shared_corner_rows
python -m benchmark.native_surface_counter
python -m benchmark.native_corner_registration
python -m benchmark.garden_anchor_extension --arm connected --workers 3
python -m benchmark.relative_face_round connected-atlas
```

The floating Surface Finder refines a proposal, then tests its native support against a sky mask built from the source image. Each row requires agreeing blue samples at both side borders. Sky and cloud-colored pixels must connect back to a border. A proposal is rejected only when every supported pixel qualifies. Mixed proposals retain their full support. The gate runs before outlines, crops and counts. Each build manifest records its rejected proposal IDs. The sky-handoff command uses the frozen normal and depth fields to reproduce the six inputs. The following stages consume outputs within this checkout. Decoded and scored exports are under `runs/floating-cloud-gate-01/relative-face-assembly/connected-atlas/sky`. This is the benchmark entry point for the floating branch; the ordinary build CLI still runs the production-style pipeline.

## Which work belongs here

| Branch history included | Use |
|---|---|
| Floating local grids, source charts and relative assembly/scoring | Inputs, visible-grid assembly, single-translation evaluation |
| Relative face junction control | Reference-only estimate of how well exact visible grids join |
| Shared corner rows and surface crop first | Previous source-only registered-seam baseline; weaker variants retained for comparisons |
| Sky mask feedback, line leveling and the surface inspector | Experiments and review tools. The whole-mask sky check now runs in the floating Surface Finder. Line leveling helped several surfaces and worsened others. Neither improvement is included in the baseline score as a new placement gain. |

The `experiment/floating-build-floor-bypass` worktree remains a historical diagnostic of the older placer, including supplied-camera interventions. Future work should start here and use sibling worktrees for parallel candidates. Keep the current inspector and annotation exports available at their original paths so existing feedback links keep working.

## Next annotations

Prioritize a second building with a visible corner that the assembler failed to join, a normal-looking crop where straightening breaks the texture rows, and a similar-looking wall where the current counter is already right. Note the surface IDs, the edge or rows you followed, and whether the two surfaces share the full height or only part of it. Mark a block count only when you can actually see it. Cloud examples from real skies, especially beside pale blocks, would extend the synthetic-sky check. Keep development annotations separate from a fresh held-out building used to judge promotion.

The floating benchmark runner selects the active [240-view training and optimization cohort](../training/floating-active-cohort-v1.json) by default. Use `--historical-cohort` with the manifest runner to replay a historical comparison. The six fixed sky composites are all active.
