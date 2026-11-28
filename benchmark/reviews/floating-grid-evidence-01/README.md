# Can visible texture and wall seams correct the block spacing?

Keep the existing whole-wall grid spacing. Selecting half or double pitch reduced pooled strict placement from 35.34% to 31.79% on 38 original images and from 27.25% to 26.22% on 12 new images. A conservative second selector returned the existing coordinates and IDs on all 50. Retain this branch as an offline experiment.

The first rule froze at `2d064e7`, and the conservative rule froze at `6bd69bb` before fresh evaluation. Both are fixed heuristics. Inputs are the single native source image, its inferred supported vertical wall charts and the vanilla Java 26.3 atlas. Existing whole-wall exports preserve original material assignments and rank added cells on the masked atlas. We replay that input unchanged as the current-pitch control. Half/double changes scale jointly along both local chart axes about the existing integer seed closest to their median. The phase and vanishing directions stay anchored. Candidate cells must have source support, a supported center, at least 78% covered subpoints, and at least a 3.5-pixel shortest edge. Changed grids resample all material IDs at the new physical locations; chart confidence is fixed.

For each pitch we visit up to 32 shared native pixel witnesses, deduplicate their local cells, compare masked atlas fit and compare RGB differences across grid boundaries with equal-distance interior differences. Boundary samples may include openings in the inferred support. We accept the best median score when it exceeds the current one by .06 using at least four sampled cells. The guarded variant considers only finer proposals, protects any existing nonfallback sprite of confidence .90 or higher and vetoes growth of more than 256 cells per chart. On both chart halves it requires independent new-boundary contrast and atlas gain measured on different tile parity. The finite oracle selects among the five actually exported frame options (current, half, double, selected, guarded), maximizing pooled strict position F1 separately for each reported cohort. Oracle choices never enter inference. Other columns show those same packs. The ceiling excludes arbitrary combinations of chart pitches, anisotropic scale and rephasing.

Values in each cell give strict position / facade position / strict position-plus-ID / facade position-plus-ID F1 in percent. Scores use one whole-build translation under the established orientation rule. For every arm and frame, the pack decoder verified all local coordinates and IDs and the scorer checked the registered strict hits against native visible block labels.

| Cohort | Images | Current | Image-selected | Guarded | Finite oracle |
|---|---:|---|---|---|---|
| Original test | 38 | 35.34 / 35.55 / 4.51 / 4.53 | 31.79 / 31.96 / 3.86 / 3.88 | 35.34 / 35.55 / 4.51 / 4.53 | 43.25 / 43.50 / 5.22 / 5.24 |
| Twin spires v2 | 12 | 41.90 / 44.06 / 19.42 / 20.44 | 36.59 / 38.34 / 15.00 / 15.78 | 41.90 / 44.06 / 19.42 / 20.44 | 42.81 / 45.06 / 18.82 / 19.85 |
| Castles v3 | 18 | 35.89 / 35.82 / 3.62 / 3.60 | 32.33 / 32.28 / 3.33 / 3.33 | 35.89 / 35.82 / 3.62 / 3.60 | 44.52 / 44.48 / 4.56 / 4.54 |
| Walled gardens v1 | 8 | 21.14 / 22.76 / 2.07 / 2.24 | 17.22 / 18.66 / 0.43 / 0.47 | 21.14 / 22.76 / 2.07 / 2.24 | 21.54 / 23.32 / 2.27 / 2.47 |
| Fresh combined | 12 | 27.25 / 28.39 / 5.63 / 6.04 | 26.22 / 27.27 / 5.12 / 5.49 | 27.25 / 28.39 / 5.63 / 6.04 | 27.44 / 28.31 / 5.43 / 5.83 |
| Fresh validation | 6 | 28.39 / 30.22 / 9.45 / 10.54 | 25.79 / 27.35 / 8.58 / 9.61 | 28.39 / 30.22 / 9.45 / 10.54 | 28.39 / 30.22 / 9.45 / 10.54 |
| Fresh supplemental train layouts | 6 | 26.48 / 27.23 / 3.05 / 3.20 | 26.49 / 27.21 / 2.94 / 3.09 | 26.48 / 27.23 / 3.05 / 3.20 | 26.81 / 27.13 / 2.75 / 2.90 |
| Fresh visible-builds-v2 | 4 | 36.19 / 38.62 / 17.15 / 18.30 | 34.81 / 36.79 / 14.65 / 15.48 | 36.19 / 38.62 / 17.15 / 18.30 | 37.42 / 40.03 / 17.65 / 18.88 |
| Fresh visible-builds-v3 | 4 | 23.66 / 24.46 / 2.64 / 2.77 | 23.66 / 24.46 / 2.64 / 2.77 | 23.66 / 24.46 / 2.64 / 2.77 | 23.66 / 24.46 / 2.64 / 2.77 |
| Fresh decorative-builds-v1 | 4 | 33.10 / 34.66 / 8.19 / 9.06 | 28.46 / 29.63 / 6.42 / 7.17 | 33.10 / 34.66 / 8.19 / 9.06 | 33.15 / 33.47 / 7.15 / 7.93 |

On the original frames, the rule changed 17 of 182 eligible charts, nine finer and eight coarser. It recovered 1,024 correct registered positions and lost 677, while adding 11,303 extras and removing 698. The castle-02 ashlar rear-oblique view alone added 4,710 extras for 26 recovered positions. Finer spacing across all walls of castle-03 raised placement to 42.75% from 27.13% on its six views. The image rule reached 26.24% there. The finite oracle achieved 19,418 correct, 40,020 missed and 10,941 extra positions across the original frames.

Of nine finer proposals, the guard rejected four for excessive cell growth and two for existing trusted sprites. The other three failed separated evidence checks. Across the fresh frames the first rule changed seven of 53 charts and recovered 37 positions, lost 83, added 194 extras and removed 92. Validation chapel placement fell from 40.10% to 32.26%. The fresh validation six had zero strict geometry headroom within the finite options. Both selectors used frozen settings on all six fresh layout families. The supplemental six frames come from published train layouts. All views are original lighting. The fixed 104-view historical suite, lighting pairs and remaining dataset views were unmeasured. Across both runs the tested coverage is 16 views of v2, 22/144 of v3 and 12/40 of Decorative Builds v1.

[Original decoded score, manifests, switches, guard reasons and paired counts](original.json.gz) · [Fresh decoded score and decisions](fresh.json.gz) · [Source overlays for the two largest extra-cell increases](pitch-overlay.jpg)

## Replay

Use the benchmark's Python environment with OpenCV, Pillow, NumPy and the existing calibrated atlas. Restore the source and source_material packs from the expanded-wall-rulers and floating-fresh-validation runs in their registered sibling worktrees. From this checkout run the following commands.

```powershell
python -m benchmark.floating_grid_evidence --workers 5
python -m benchmark.floating_grid_guard original
python -m benchmark.floating_grid_report original
python -m benchmark.floating_grid_evidence --fresh --workers 4
python -m benchmark.floating_grid_guard fresh
python -m benchmark.floating_grid_report fresh
```
