# MoGe surface-first camera and grid experiment

Keep `experiment/camera-moge-surface-first` experimental. Starting from the verified pavement fallback, five extra v3 views returned builds with 2,145 additional correct positions, 3,969 additional extras, and 443 additional correct position-and-ID pairs. Four of those five were development images. The single held-out Castle camera placed zero blocks correctly at the registered coordinates. Whole-build translation recovered 478 correct positions there.

## Camera measurement and placement

The only changed handoff was the native 3×4 projection in block units returned by `visible_placement.seam_camera`. A source-only MoGe proposal was available on ordinary-camera failures. On acceptance, the ordinary single-image surface detector, counting, placement, material matching and datapack export ran with the frozen source-model evidence. All scores include failed views and use a single integer translation selected on geometry. Each cube requires visible evidence from that input. No part of this method completes unseen surfaces.

| Cohort | Arm | Cameras failed | Strict / facade placement F1 | Strict / facade position + ID F1 | Correct / missed / extra positions |
| --- | --- | ---: | --- | --- | --- |
| Fixed 104/104 | All four | 0 | 91.27 / 93.42 | 84.68 / 88.51 | 20,773 / 2,929 / 1,046 |
| Visible Builds v3 144/144 | Ordinary | 108 | 8.08 / 8.51 | 6.16 / 6.50 | 9,299 / 204,851 / 6,859 |
| | Whole-tile repeat | 78 | 18.99 / 19.88 | 12.49 / 13.12 | 23,798 / 190,352 / 12,644 |
| | MoGe after ordinary | 98 | 10.15 / 10.67 | 6.45 / 6.78 | 12,164 / 201,986 / 13,299 |
| | MoGe after repeat | 73 | 20.21 / 21.14 | 12.54 / 13.15 | 25,943 / 188,207 / 16,613 |
| Decorative Builds v1 40/40 | Ordinary | 10 | 54.33 / 56.11 | 43.43 / 45.34 | 7,120 / 9,806 / 2,162 |
| | Whole-tile repeat and MoGe after repeat | 0 | 63.46 / 65.23 | 50.51 / 52.56 | 9,240 / 7,686 / 2,957 |
| | MoGe after ordinary | 7 | 55.55 / 57.20 | 43.77 / 45.64 | 7,470 / 9,456 / 2,501 |

The fixed 104 and the 66 ordinary-supported expanded exports are carried forward byte-for-byte. MoGe ran on 118 images, accepting 11 of 88 development proposals and two of 30 outside development. Every accepted camera received a fresh export. The independent pavement fallback already solved eight of the 13. Per-view hashes, placements, native corner errors, full split metrics and block-ID assignment counts are in the [proposal records](proposals.json.gz) and [pooled scores](scores.json).

| Split | Images | Repeat → repeat then MoGe strict geometry | Strict position + ID | Camera failures |
| --- | ---: | --- | --- | --- |
| v3 development | 108 | 23.44 → 24.56 | 15.69 → 15.33 | 54 → 50 |
| v3 Harbor validation | 18 | 16.01 → 16.01 | 9.61 → 9.61 | 8 → 8 |
| v3 Castle test | 18 | 5.58 → 7.26 | 3.09 → 4.30 | 16 → 15 |
| Decorative development | 24 | 62.64 → 62.64 | 49.70 → 49.70 | 0 → 0 |
| Decorative validation | 8 | 72.32 → 72.32 | 63.16 → 63.16 | 0 → 0 |
| Decorative test | 8 | 57.46 → 57.46 | 40.95 → 40.95 | 0 → 0 |

## Why the scale gate accepted wrong cameras

MoGe-2 supplied relative 3D points, normals, foreground validity, and a global metric-scale output. The last output was saved for provenance. Block scale came entirely from matching one-block sprites and texture repeats in source pixels. A robust centered perspective fit estimated native focal length and the shift along Z. Signed model normals plus robust plane fits to source masks supplied an up vector and a pair of horizontal Manhattan directions. The floor samples estimated ground elevation in arbitrary model units. The calibrated Java atlas supplied candidate block spans on the structure. The gate required separated sprite matches, phase agreement, compatible scale factors on two different masks, and agreement between the predicted ground and texture grid phase. It allowed ±12% around the selected scale and ±0.22 block of ground phase. Mask fragments can still sample the same physical wall, and periodic sprites can match an incorrect multiple. Mask agreement by itself leaves these ambiguities unresolved.

The fitted planes returned three axes for 118/118 failures. Five met both a 2° worst-axis error and 5% focal-error threshold, compared with 39 from the previously measured long-edge scaffold. Median worst-axis error was 5.53°. Among the 13 source-accepted cameras, only seven of 20,420 native visible corners reprojected within 2 pixels. Median per-frame corner error was 69.65 pixels. All 13 images improved the ordinary empty build in strict F1, but moving a build into alignment can conceal severe metric camera error.

The Castle rear view accepted mutually consistent texture measurements even though the predicted height was 47% too low. Across 5,643 visible corners, the median error was 481.7 native pixels. Its build contained 600 blocks. The whole-build-translated evaluation recovered 478 positions (341 with the correct ID), while the registered evaluation recovered zero. On the development Amphitheater view, the camera was 31% too high, the median corner error was 44.9 pixels, and the export added 868 extras for 258 correct positions. The overlays draw source atlas matches and a deterministic sample of native corner projections. [Castle source overlay](test-castle-03-limestone-rear-oblique.jpg) · [Amphitheater source overlay](train-amphitheater-02-ashlar-rear-oblique.jpg)

The earlier correct-camera replay reached 53.26% facade placement on v3 and 66.87% on Decorative. That measures the opportunity at this handoff. It supplies no source camera proposals. [Recorded-camera control](../camera-recorded-pose-01/README.md)

## Reproduce

Base revision `d9eb377`, Java 26.3, fixed cohort IDs and expanded source hashes from the controls experiment. The [checkpoint](https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/blob/e50ffda41565591092adea54c6ac83d6212e1e23/model.onnx) has SHA-256 `24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0`. The graph exports raw points, normals, mask, and scale in one inference. CPU ONNX Runtime 1.23.2, 600 tokens, 448 maximum input side, Lanczos resize with no crop. Camera coordinates use +X right, +Y down and +Z forward. Field NPZs preserve full-resolution outputs of that resized inference. Each sister JSON records native dimensions, model provenance, input hash and the pixel mapping. The source proposal is stored separately from grading.

Run the following from the experiment checkout in the saved benchmark environment. Build outputs and large model arrays are private under `runs/camera-moge-surface-first-01/<dataset>/<id>/`. Shared baseline fields and repeat exports are read from their registered worktrees. The historical 112-view comparison was not rerun.

```powershell
python -m benchmark.camera_moge_fields
python -m benchmark.camera_moge_surface --split train
python -m benchmark.camera_moge_replay --split train
python -m benchmark.camera_moge_surface --split held
python -m benchmark.camera_moge_replay --split held
python -m benchmark.camera_moge_report
python -m unittest discover -s tests -p test_camera_moge_surface.py
```
