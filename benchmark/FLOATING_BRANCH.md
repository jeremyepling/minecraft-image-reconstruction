# Floating-build reconstruction

`experiment/floating-visible-builds` is the long-lived integration branch for reconstructing visible blocks with no ground reference. One translation can place the entire build anywhere. All visible components retain their relative spacing. Production `main` at the fork was `e3bd220`.

## Baseline to beat

The source-only baseline measures local face grids, transfers shared rows across a supported corner, then registers native atlas matches onto that existing grid. The last step preserves the corner join. The six open-sky composites score **11.16% strict placement F1**, **11.38% facade placement F1**, and **2.89% / 2.95%** strict/facade position-plus-material F1. There are 239 correct positions, 3,802 missed positions and five extras. The crop-first registration gain came from Castle. Four views are from Visible Builds v3 and two from Decorative Builds v1. Use the fixed 104 production trend (91.27% strict, 93.42% facade) for historical context. Assess the gap to production using the same buildings, backgrounds, inputs and scoring rule on both branches.

### Reproduce the six exports

From this checkout, using the benchmark Python environment and the saved open-sky fixtures, source masks/counts, and dense atlas chart handoffs on this PC:

```powershell
python -m benchmark.shared_corner_rows
python -m benchmark.native_surface_counter
python -m benchmark.native_corner_registration
python -m benchmark.relative_face_round registered-seam
```

The first stage reads frozen source handoffs from the `floating-build-local-grid` and `relative-face-source-charts` worktrees. The following stages consume outputs within this checkout. Decoded and scored exports are under `runs/relative-face-assembly-01/registered-seam/sky`. This is the benchmark entry point for the floating branch; the ordinary build CLI still runs the production-style pipeline.

## Which work belongs here

| Branch history included | Use |
|---|---|
| Floating local grids, source charts and relative assembly/scoring | Inputs, visible-grid assembly, single-translation evaluation |
| Relative face junction control | Reference-only estimate of how well exact visible grids join |
| Shared corner rows and surface crop first | Winning source-only registered-seam baseline; weaker variants retained for comparisons |
| Sky mask feedback, line leveling and the surface inspector | Experiments and review tools. The sky filter cleaned background masks on the composites. Line leveling helped several surfaces and worsened others. Neither improvement is included in the baseline score as a new placement gain. |

The `experiment/floating-build-floor-bypass` worktree remains a historical diagnostic of the older placer, including supplied-camera interventions. Future work should start here and use sibling worktrees for parallel candidates. Keep the current inspector and annotation exports available at their original paths so existing feedback links keep working.

## Next annotations

Prioritize a second building with a visible corner that the assembler failed to join, a normal-looking crop where straightening breaks the texture rows, and a similar-looking wall where the current counter is already right. Note the surface IDs, the edge or rows you followed, and whether the two surfaces share the full height or only part of it. Mark a block count only when you can actually see it. Cloud examples from real skies, especially beside pale blocks, would extend the synthetic-sky check. Keep development annotations separate from a fresh held-out building used to judge promotion.
