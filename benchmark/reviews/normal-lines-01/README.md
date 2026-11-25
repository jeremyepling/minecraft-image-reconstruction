# Physical boundary coverage on 48 Minecraft frames

M-LSD covered 72.7% of visible boundary pixels with 2,992 segments. The structural detector covered 90.9% with 30,427 segments. Their pooled line precisions were 34.4% and 33.8%, respectively.

Both detectors saw one Java 26.3 screenshot per prediction. This fixed comparison reuses the six layouts and their original splits from the first surface round. Counts pool pixels within each group. Floor and palette variants of a layout are repeated observations. The concrete group includes eight workshop and crossing views available only on that floor.

Precision is the fraction of unique predicted line pixels within five native pixels of a physical boundary, evaluated inside the building and its five-pixel exterior band. Coverage is the fraction of true visible boundary pixels reached by those predictions. The scorer counts turns, depth changes and silhouettes, and excludes coplanar block seams.

| Group | Frames | M-LSD segments | Precision | Coverage | Structural segments | Precision | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 48 | 2,992 | 34.4% | 72.7% | 30,427 | 33.8% | 90.9% |
| courtyard | 12 | 904 | 15.5% | 83.0% | 6,656 | 21.2% | 94.1% |
| crossing | 4 | 206 | 70.4% | 70.4% | 2,834 | 41.3% | 92.3% |
| portico | 8 | 497 | 80.9% | 71.5% | 5,760 | 64.8% | 87.4% |
| shopfront | 8 | 539 | 81.5% | 70.2% | 5,760 | 56.9% | 89.7% |
| wing | 12 | 709 | 15.4% | 86.5% | 6,537 | 15.4% | 96.8% |
| workshop | 4 | 137 | 65.6% | 57.4% | 2,880 | 38.0% | 90.3% |
| light_gray_concrete | 28 | 1,440 | 37.2% | 71.3% | 17,912 | 33.3% | 93.8% |
| smooth_stone | 20 | 1,552 | 31.0% | 75.0% | 12,515 | 34.9% | 86.3% |

The manifest contains pooled pixel counts and method settings. The frames file contains the source and evaluator hashes and both pixel denominators for every paired comparison.

Run from the repository root using the previously downloaded checkpoint and private runtime.

```powershell
& 'C:/Users/jerem/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m benchmark.normal_line_round --runtime-dir ../surface-round/runs/mlsd-probe/lib --checkpoint ../surface-round/runs/mlsd-probe/M-LSD_512_tiny_fp16.tflite --out runs/normal-lines-01
```
