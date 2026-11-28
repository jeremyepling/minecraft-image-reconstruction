# Stable Yard eave attachment after a depth-first fit

**Reference-assisted, one saved Floating view.** One rigid attachment recovered 21 correct visible eave blocks and 21 exact material IDs, with zero previously correct positions lost and zero extras added. Keep this as a diagnostic. The source-only pipeline still needs a way to recover the one-block relief step and attach the eave at the right place.

The saved high-right `mask-0~p24` eave had its own pink depth region before fitting its cut-sandstone grid. Its independently verified grid covers 21 distinct frontmost recorded faces. The frozen expanded split-border export gives the inherited chart for that region the same normal coordinate as the surrounding wall and reconstructs zero of those eave blocks. The earlier ordinary full refit leaves the independently fitted grid detached.

The following replays preserve the frozen chart packet and ordinary assembler, pack writer and decoder. The reference determines only a rigid relative transform for the 21 already-evidenced source cells. No reference blocks, block IDs, or hidden backing cells are inserted. The flat control uses the measured tangent coordinates but offers the incumbent's flat depth. The attach arm keeps competing inherited cells. The replace arm removes the 260 inherited grid cells whose source centers lie on the measured eave region. The detached no-op reproduced the exact archived export.

| Stable Yard 1 high-right | Strict / facade placement F1 | Strict / facade position + material F1 | Correct / missed / extra | Exact eave positions + ID |
| --- | ---: | ---: | ---: | ---: |
| Detached or same-plane offer | 7.84% / 7.93% | 1.99% / 2.02% | 63 / 291 / 1,190 | 0 |
| Attach at recorded relief step | 10.32% / 10.44% | 4.55% / 4.60% | 84 / 270 / 1,190 | 21 |
| Attach and replace inherited cells | 8.92% / 9.04% | 1.32% / 1.33% | 61 / 293 / 953 | 0 |

The replace arm loses two net correct positions and seven net exact IDs. It changes the optimal whole-build translation from X=1547 to X=1585. At the incumbent translation it keeps the 21 attached eave blocks and directly removes 26 correct positions and 234 extras. This distinguishes the incorrect depth from the additional decision to remove all inherited cells. The perfect chooser selects attachment with existing cells preserved.

The source image overlay marks the fitted pink region and its verified visible eave faces in green. [Zoomable eave overlay](verified-eave.png) · [Decoded scores and source, reference and pack checksums](results.json)

Reproduce offline using the benchmark Python environment and the saved inputs in the adjacent `floating-depth-before-atlas`, `floating-selected-package` and `floating-package-rechecks` worktrees. Run `python -m benchmark.floating_pregrid_eave_attachment`. Each arm writes its own pack under `runs/pregrid-eave-attachment`. Score each entire decoded export with one integer translation.
