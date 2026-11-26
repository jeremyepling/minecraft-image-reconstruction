# Joint foreground depth decisions

The conservative rule improves visible geometry from 83.947% to 84.399% relative F1 on the 112-frame evaluation. Eleven views improve, 100 tie and one loses three correct blocks. A single integer translation is free per screenshot. The measured inputs are frozen screenshot-derived cameras, masks and counts. Each score comes from decoding the exported Minecraft datapack.

| Development rule | Conflict strength | Penalty for changing a surface depth | F1 on 72 training + 12 reviewed native images |
|---|---:|---:|---:|
| Previous independent rule | — | — | 83.257% |
| Conservative | 1.6 | 0.50 | 83.967% |
| Balanced | 3.2 | 0.28 | 83.821% |
| Assertive | 5.0 | 0.15 | 83.285% |

The independent chooser proposes a depth for each observed surface using its texture atlas, repeat, boundary and junction evidence. The joint rule retains that choice and the four best repeat-ranker candidates. It projects each candidate's observed face tiles and sides against neighboring masks. Overlap with a different photographed surface penalizes a candidate only when a ray hits it in front of that surface. Atlas match quality weights the conflicting evidence. Up to two sweeps update the depths. The existing visibility checks still decide which blocks reach the export.

| Evaluation set | Images | Correct gained / lost | Extras added / removed | Relative F1 |
|---|---:|---:|---:|---:|
| Cloister validation | 12 | 9 / 3 | 8 / 7 | 91.315% |
| Twin Spires test | 12 | 1 / 0 | 0 / 10 | 91.499% |
| Fixed exteriors | 56 | 110 / 0 | 7 / 29 | 81.894% |
| Earlier native | 12 | 0 / 0 | 0 / 0 | 84.972% |
| Reviewed native | 12 | 3 / 0 | 13 / 33 | 77.989% |
| Additional native confirmation | 8 | 11 / 0 | 0 / 8 | 81.801% |

The reviewed native cohort also participated in development. The separate 72 training images contributed no blocks to the 112-frame evaluation. The winning rule found 18,834 of 24,086 visible target blocks, with 1,711 extras. This includes 95 extras where the source world contains a block hidden from the screenshot, up from 94. Eight unsupported views produced empty packs under both rules. In the ten canopy probes, both rules preserved the correct ten opening widths and put 170 of 459 visible roof blocks at the correct offsets from the posts. The perfect image-level choice between the default and conservative packs would reach 84.407% F1.

Commit 3267164 froze the selected adapter before the remaining five cohorts ran. The public placement path in 041c6c2 reproduced the frozen candidate's decoded coordinates in all 112 images. It records the actual foreground method and number of changed planes in the build manifest. Use the previous independent rule with the placement switch `--independent-plane-depth` or `-IndependentPlaneDepth` in the PowerShell wrapper.

Run with the shared frozen inputs and the stable checkout's `runs/canopy-placement/release` baseline. The release baseline's cohort named `old` holds the reviewed views and `fresh` holds the additional native confirmation.

```powershell
python -m benchmark.visible_depth_score --split regression --candidate-root . --candidate engine.foreground_conflicts:conservative --baseline $release --out runs/conservative
python -m benchmark.canopy_compare --inputs $confirmationInputs --baseline "$release/fresh" --candidate-root . --candidate engine.foreground_conflicts:conservative --out runs/conservative/fresh
```

Replace the adapter with `benchmark.visible_depth_score:integrated` to replay the public entry. [Per-image counts, checked input and pack fingerprints, decoded coordinates and packs](scores.zip) · [Development comparison, frame-level perfect chooser and ten roof probes](summary.json)
