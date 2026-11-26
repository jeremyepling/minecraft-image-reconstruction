# Two-pixel sampling for visible horizontal faces

The guarded placement pass recovered **487 additional visible blocks**, including 210 whose visible pixels missed the ordinary four-pixel lattice. It retained every block in the input prediction. On 112 evaluation images, relative geometry F1 rose from 83.947% to **85.064%**. The 560 additions comprise 487 correct and 73 extra blocks. One extra coincided with an occupied but unseen world cell. Three dimensional placements use screenshot-derived evidence only.

| View set | Frames | Ordinary F1 | Guarded F1 | Newly correct | New extras |
|---|---:|---:|---:|---:|---:|
| Validation | 12 | 91.198% | 91.701% | 29 | 8 |
| Test | 12 | 91.342% | 92.012% | 43 | 3 |
| Fixed exteriors | 56 | 81.126% | 82.477% | 251 | 42 |
| Supported native | 12 | 84.972% | 86.285% | 85 | 6 |
| Earlier transfer | 12 | 77.549% | 78.855% | 53 | 9 |
| Fresh transfer | 8 | 81.061% | 82.067% | 26 | 5 |

Whole-build translation is free for each screenshot. The metric covers visible block coordinates. The input predictions came from the originally released packed builds, independently decoded. The candidate emits another pack, which the scorer decodes before reading the recorded pose, visible labels or source world. A frame with no supported camera yields the same empty build as its input.

## What changed

The candidate proposes a cell adjacent to at least two placements at its elevation. It projects the horizontal face at actual native pixel centers spaced two pixels apart. At least four centers and 85% of the exposed face must agree with a horizontal image mask where the current rendering has a different face axis or background. All candidate cubes compete for the frontmost rays. At least 90% of the visible cube samples must be inside a detected surface mask, and 85% must agree on the face axis. Those criteria apply to every exposed side of the proposed cube. Neighboring blocks alone never authorize a placement.

The final guard checks whether any accepted addition would eliminate the last native-pixel sightline to a previously visible block. It discards the blocking additions and repeats the visibility check if needed. The guarded pass improved 55 views and tied on 52. Five individual views declined, with the largest decline under 0.5 percentage points. The median time for the added placement pass was 0.109 seconds per image in a concurrent three-worker run.

## Controls and decision

On the 12-frame development screen (two views of the first layout of each of the six training families), increasing density for unclaimed-mask coverage alone added one correct block and five extras at the original area threshold. Scoring only the actually exposed fraction of the horizontal face recovered more blocks. Across all 72 training images, requiring two neighbors and checking every visible axis produced 515 additional correct placements and 215 extras. Training F1 rose from 83.988% to 85.182%.

The mask-based selector was frozen at `0ee69ee` on `experiment/native-sample-density` before the 112 evaluation images were scored. The matched four-pixel control scored 84.397%, with 294 newly correct and 165 extra placements. The two-pixel version scored 85.070% before the final protection check. Adding the native visibility guard at `7991728` retained all 487 gains and all input blocks. This standalone result supports testing the guarded rule in the combined default. A composed run needs a separate score because earlier placement stages change which faces are visible.

## Reproduce

Run from the experiment checkout with the saved release data and native fixtures restored. The per-image score archive includes pack and source fingerprints, coordinates recovered outside the coarse lattice, relative scores, and placement-only timings. [Download the score archive](scores.zip).

```powershell
python -m benchmark.native_density_score --sets train --diagnostic-train --modes four64 two64 two16 --out runs/sample-density/train-smoke
python -m benchmark.native_density_score --sets train --diagnostic-train --modes visible16 interior16 visible64 --out runs/sample-density/train-occlusion
python -m benchmark.native_density_score --sets train --modes interior4pitch interior16 interior_verified --out runs/sample-density/train
python -m benchmark.native_density_score --sets validation test regression native old fresh --modes interior_verified4 interior_verified --out runs/sample-density/holdout
python -m benchmark.native_density_score --sets validation test regression native old fresh --modes production_guard --out runs/sample-density/guard
python -m unittest discover -s tests -p test_visible_horizontal_faces.py
```
