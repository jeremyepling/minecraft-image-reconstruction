# Remove a false diagonal between sibling wall masks

On `heldout-facade-oblique-right`, the two regions that the junction finder cut diagonally are part of one south-facing wall. Rejoining them from RGB and first-stage normals moves the wall’s best mask IoU from **73.48% to 88.27%**. The chart’s rounded block span changes from 28 × 12 (plus an unresolved fragment) to **27 × 11**, equal to the recorded visible span on both axes. The output covers the original two fragments exactly. Only 218 unit faces in that bounding rectangle have visible support in the reference.

| Measured on the photographed facade | Before | After |
|---|---:|---:|
| Regions emitted | 15 | 14 |
| Proposals unmatched at 0.5 IoU | 3 | 2 |
| Distinct reference surfaces matched out of 20 | 12 | 12 |
| Reference wall pixels captured by its best matching mask | 152,607 | 183,841 |
| Recall of the wall’s 186,920 pixels | 81.64% | 98.35% |
| Correctly paired pixels across all surfaces | 194,340 | 225,574 |
| All background pixels claimed | 745 | 745 |
| Blocks in the decoded geometry datapack | 309 | 309 |

**Gate.** Two neighboring candidates must share the same split parent and near-identical wall normals (dot ≥ 0.985). The camera-normal constraint for an upright image and a nominal 70° vertical FOV provide the wall’s vertical and horizontal vanishing directions. A shared boundary must have ≥96 distinct native pixels, a line variance ratio ≥15, and a direction at least 18° from both physical wall directions. Across-gap color and interior texture contrast are sampled on each sibling’s actual support. The median RGB step must be ≤6 and the mean cross-edge contrast ≤max(4, 1.5 × within-side contrast). The facade passes with 261 distinct boundary pixels, 24.72°, a median step of 4 and contrast 11.60 vs 8.70. This compares local photograph colors and texture variations. It does not fit a repeating atlas tile or estimate hidden wall cells.

**Controls and limits.** Exactly one join was accepted among the locked 204 images (72 development, 104 fixed evaluation, 20 alternate views, four new shapes, four shelter/gate). Every decoded block position and material ID equaled its paired control. Fixed strict F1 remained 90.4964% and facade-priority F1 remained 92.5611%. The independent full RGB CLI checks regenerated the original and new predictions from the same screenshot. The export uses smooth stone for its geometry. Neither count pass produced any positionable material groups, even when explicitly supplied the prepared atlas, because the available texture evidence did not resolve tile phase. The result establishes a correction on one surface, with no measured geometry or material-identification gain. Tests also cover a real color step, axis-aligned edge, non-sibling, different face directions, a hole in the wall support, and old corner links after the shared edge disappears.

The mask-stage rule can land as a guarded Surface Finder correction if its limited scope is acceptable. A separate photograph with a real diagonal occlusion between same-parent parallel surfaces would be a valuable harder negative, especially if the surfaces use similar materials. Retain other fragments until their own evidence supports a join.

## Reproduce on the shared PC

In the candidate checkout at `792a38a`, run `python -m unittest tests.test_guarded_wall_outlines`. In the frozen evaluator checkout `strip-orientation-audit`, use the following commands with Python from the project environment.

```powershell
python -m benchmark.strip_orientation_eval --candidate-root ../guarded-surface-outlines --candidate benchmark.guarded_wall_outlines_trial:predict --out ../guarded-surface-outlines/runs/primary-verified --workers 5
python -m benchmark.strip_orientation_eval --candidate-root ../guarded-surface-outlines --candidate benchmark.guarded_wall_outlines_trial:predict --out ../guarded-surface-outlines/runs/shelter-verified --transfer-inputs ../framed-depth-transfer/runs/transfer-input
```

Run the normal `Invoke-WallRecovery.ps1` from control `edb10a3` and candidate `792a38a` separately. Supply the photograph from `workspace/builds/entrance-holdout-26.3/positive/captures/heldout-facade-oblique-right.png` and `-Through placement`. Write to `runs/control-cli-facade` and `runs/verified-cli-facade` under the candidate worktree respectively. From the candidate, run `python -m benchmark.guarded_wall_outlines_report` to compare every decoded pack, check the native RGBA crops against source pixels, verify the outputs with and without explicit count-stage atlas matching, and grade against held-out visible face labels. [Paired 204-frame manifest](frames.csv) and [CLI and wall evidence](result.json).
