# Visible block faces behind the floating wall grids

Across the 74 previously scored conditions, 3,093 of 3,846 atlas seeds aligned with a single visible full block face. Of those aligned seeds, 1,961 also had the correct block ID. The 48 wall charts without an aligned seed produced 14,942 of the 39,052 proposed wall cells. These charts supplied 6,637 of 13,115 extra exported positions and 6,167 of 18,665 correct positions. Rejecting a whole wall on its seed evidence could discard many correct placements.

The Chapel sky chart `mask-0` had two seeds measuring 0.410 × 0.276 and 0.426 × 0.281 native block units along their proposed axes. They overlapped their best visible block-face quads at 0.038 and 0.098 IoU. Both called calcite pearlescent froglight. The chart supplied 1,549 exported positions, of which 118 landed on visible blocks, 19 had the correct ID and 1,431 were extra. [Chapel source overlay](../floating-patch-controls-01/chapel-ruler.jpg)

## Seed alignment and chart coverage

For each source-only seed, the diagnostic projects its four corners and matches the positively visible native full-cube face with highest convex IoU in the same wall or horizontal class. It checks matched corners against the recorded camera projection to within 0.0001 pixel. Native face visibility is the share of its projected face area with frontmost pixel ownership. Occlusion is measured with native pixel centers, with the shared rasterizer's area estimate if a face crosses the viewport. Corner error is the lowest RMS over all rotations and reflections, measured in the reference face's unit-square coordinates. Alignment requires IoU ≥ 0.65, visible fraction ≥ 0.70, corner RMS ≤ 0.20 block and at least 55% of the seed's in-frame pixel centers owned by the best native face. Material identity is measured independently. The receipt keeps continuous measurements and the frame, chart and seed key.

The growth count is the number of measured wall cells before assembly. Chart masks or their center samples can cover the same native face more than once across charts in a view. Attribution uses the archived selected chart origins, orientation and cell precedence, and reproduces the decoded export position-for-position and ID-for-ID before counting the chart that owns each block. [Archived seed labels](seed-audit.json.gz) · [Third holdout seed labels](seed-audit-holdout3.json.gz) · [Fourth holdout seed labels](seed-audit-holdout4.json.gz)

| Panel | Views | Aligned seeds / seeds | Correct IDs on aligned seeds | No aligned seed / charts | Cells grown from those charts / all grown | Correct positions / extras from those charts |
|---|---:|---:|---:|---:|---:|---:|
| Original | 38 | 1,859 / 2,286 | 1,185 | 24 / 182 | 7,038 / 23,445 | 5,131 / 1,458 |
| Fresh | 12 | 455 / 543 | 346 | 9 / 53 | 1,030 / 3,697 | 187 / 504 |
| Supplemental | 6 | 292 / 318 | 129 | 0 / 19 | 0 / 1,485 | 0 / 0 |
| Sky composites | 6 | 215 / 394 | 128 | 8 / 29 | 3,684 / 5,627 | 206 / 2,139 |
| Holdout 1 | 6 | 117 / 126 | 114 | 2 / 17 | 2,483 / 3,147 | 508 / 1,975 |
| Holdout 2 | 6 | 155 / 179 | 59 | 5 / 21 | 707 / 1,651 | 135 / 561 |
| Holdout 3 | 6 | 134 / 189 | 84 | 4 / 18 | 3,018 / 6,671 | 209 / 2,795 |
| Holdout 4 | 6 | 181 / 225 | 161 | 3 / 28 | 2,939 / 3,861 | 152 / 2,723 |

Of the 753 archived seeds outside the alignment thresholds, 405 have an edge-length scale error, including 274 undersized on both axes, 27 oversized on both axes and 104 with at least one edge length outside 0.80–1.25 blocks. Another 327 have lengths in range and poor phase or shape, 14 have a partly occluded match, and seven overlap no owned pixel of their best geometric match. A single good seed establishes local scale near itself. Factory 03 low-left and high-right expose the limit of growing that grid across the entire wall. On those two views charts with aligned seeds still supplied 1,766 extras and 1,546 correct positions.

## Decoded baseline and fixtures

All 74 archival horizontal exports reproduce their published scores after independent datapack decoding and recomputation of one whole-build integer translation. The baseline uses the same translation for position, material and facade results. Two old Cloister views use the archived shared-chart yaw transfer. One Castle view exports zero blocks and remains a complete miss. All 12 new images exported blocks with fixed yaw. Across the 86 conditions there are 36 Visible Builds v3 conditions over 35 of 144 unique views, 24 Decorative Builds v1 conditions over 23 of 40 unique views, and 26 Visible Builds v2 conditions. The sky composites are displayed separately. The fixed 104 production scores are carried forward at 91.27% strict and 93.42% facade from `baff1db`, with 20,773 / 2,929 / 1,046 strict and 18,952 / 1,624 / 1,046 facade correct / missed / extra. This diagnostic has zero production delta. [Decoded baseline and dataset breakdown](horizontal-baseline.json.gz)

| Panel | Strict F1 | Facade F1 | Strict position + ID F1 | Facade position + ID F1 | Strict correct / missed / extra |
|---|---:|---:|---:|---:|---:|
| Original | 35.64% | 35.82% | 4.63% | 4.65% | 14,065 / 45,373 / 5,422 |
| Fresh | 30.43% | 31.11% | 6.85% | 7.36% | 1,600 / 5,914 / 1,401 |
| Supplemental | 39.92% | 43.08% | 9.27% | 10.00% | 900 / 2,353 / 356 |
| Sky composites | 22.93% | 23.07% | 4.40% | 4.40% | 901 / 3,140 / 2,915 |
| Holdout 1 | 18.83% | 20.12% | 3.39% | 3.70% | 700 / 3,913 / 2,120 |
| Holdout 2 | 22.15% | 23.16% | 4.93% | 5.27% | 499 / 2,607 / 901 |
| Holdout 3 | 25.61% | 25.97% | 2.05% | 2.02% | 1,815 / 5,973 / 4,573 |
| Holdout 4 | 18.63% | 18.92% | 5.96% | 6.05% | 513 / 1,501 / 2,979 |

The third panel uses Sawtooth 2 cool, Factory 03 kiln and Stable Yard 1 warm. The fourth uses Terrace 3 cool, Viaduct 01 riverstone and Riverside Mill 1 warm. Each layout contributes two angles from its packaged training split, frozen for out-of-layout evaluation relative to the earlier panels. Sawtooth and Terrace are new families. Factory, Stable Yard, Viaduct and Riverside Mill have other layouts in prior panels. Source images and native references are pinned by hash. [Third holdout manifest](holdout3-frames.json) · [Fourth holdout manifest](holdout4-frames.json)

## Replay

Keep the audit on `experiment/floating-face-audit` as a diagnostic of one handoff and how its charts contribute to the existing export.

Use the benchmark Python and the registered artifact worktrees. The scorer requires `floating-patch-validation` at `7958722` and verifies the published horizontal score archive from `6f5df52`. The native audit runs entirely on saved images and exports.

```powershell
python benchmark/floating_face_baseline.py --verify-holdout holdout3 holdout4
python -m benchmark.floating_face_audit original fresh supplemental sky holdout holdout2
python -m benchmark.floating_face_audit holdout3 --output seed-audit-holdout3
python -m benchmark.floating_face_audit holdout4 --output seed-audit-holdout4
```

The new baseline panels can also be generated from scratch with the locally installed ONNX runtime and line detector. The outputs are private to this worktree under `runs/floating-holdout3` and `runs/floating-holdout4`.

```powershell
$worktrees = (Resolve-Path ..).Path
$env:PYTHONPATH = "$worktrees/surface-round/runs/depth-probe/runtime;$worktrees/surface-round/runs/mlsd-probe/lib"
$runtime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $runtime -B -c 'import runpy,sys; sys.path.append(r"../mask-scalelsd/runs/env/Lib/site-packages"); runpy.run_path("benchmark/floating_face_baseline.py",run_name="__main__")' --produce-holdout holdout3 holdout4 --verify-holdout holdout3 holdout4
```
