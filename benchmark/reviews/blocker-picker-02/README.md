# Blocker Picker, round 2

[Review the same 20 faces](feedback.png) or [compare the stronger eave correction](eave-comparison.png). The marked source quadrilateral supplies the visible input. Coordinates supplied for diagnostic crops and recorded materials are evaluation data. The 58 automatically positioned faces are the same physical-material class in multiple views.

| Method | Predicted crops / 58 | Same faces at recorded corners / 58 | Independent sample at recorded corners / 672 |
|---|---:|---:|---:|
| Atlas | 5 | 58 | 476 |
| Rendered examples with exposure adjustment | 40 | 50 | 483 |
| Shape and color | 58 | 58 | 479 |
| Shape and color with local shade correction | 58 | 58 | 481 |
| Shape and color with additional cross-plane correction | 58 | 58 | 477 |

The previous crop-adjustment experiment scored 30/58. All round 2 challengers use the original cell corners. The shape matcher scores the placement of light and dark markings, directional contrast and aligned bands. It tests quarter turns, reflections and a small shift, and reports color disagreement, score margin and ambiguity. Glass uses its alpha mask to predict how markings composite over the local background. The local shade setting uses nearby comparable patches on the same surface. The cross-plane experiment also considers possible undersides and nearer surfaces overhead.

The combined local method corrected cases 01–08, 10, 12 and 15. It gained 36 and lost 38 of the 672 recorded-corner picks compared with exposure alone. The cross-plane experiment brightened undersides 14, 16 and 18 and missed all three. The registry validation gallery supplied 17 different glass types and seven opaque controls. The matcher found the glass family on nine glass samples, identified seven exact glass IDs and matched all seven opaque controls. See [individual registry picks](registry-check.json). [Development glass diagnostics](glass-controls.json) include examples inspected while developing the glass rule.

On this PC, from the experiment checkout, use this optional matcher on one screenshot:

~~~powershell
.\Invoke-WallRecovery.ps1 .\picture.png -Out .\runs\picker -MaterialMatch shape-shadow -Atlas C:\code\minecraft\workspace\builds\mcir-port-validation\final-cli-26.3\atlas
~~~

Replay the two shade experiments on the frozen count-01 faces with the benchmark Python environment:

~~~powershell
$cal = 'C:\code\minecraft\workspace\builds\mcir-port-validation\final-cli-26.3'
python -m benchmark.blocker_picker_shadow $cal runs/local/shadow.json --shape-module engine/blocker_shape.py
python -m benchmark.blocker_picker_shadow $cal runs/planes/shadow.json --shape-module engine/blocker_shape.py --plane-comparison
python -m benchmark.blocker_picker_feedback benchmark/reviews/blocker-picker-01/sheet-cases.json runs/local/shadow.json runs/local/shadow-maps "$cal/atlas" runs/local/feedback.png
~~~

[Local predictions and frame fingerprints](shadow-final.json.gz) · [Cross-plane predictions and frame fingerprints](plane-shadow-final.json.gz)
