# Separating walls and recovering overhangs

The **Wall Separator** became the default Surface Finder refinement on this branch. It recovered 25 additional vertical surfaces across eight buildings and lost none of the earlier matches. The separate **Underside Finder** recovered six additional undersides across Shopfront, Portico and Workshop. The thin Galleries underside still fell short of a match in both palettes. [Zoomable eight-building sheet](reviews/separation-01/comparison.jpg)

## Which models helped?

Each row starts from the same contrast-aligned Regions First masks. With the Wall Separator, only the source of relative depth changes. With the Underside Finder, only the source of the line segments changes. The ten extra underside appearances previously traced by raw line models were clues to try a split. Six became independent surface matches in this test.

| Surface Finder method | Found / 598 | Undersides / 18 | Unpaired guesses | Correctly assigned surface pixels | Mean border error on matched surfaces |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous default, contrast-aligned regions | 398 | 8 | 181 | 84.77% | 6.83 px |
| Wall Separator with Depth Anything V2 | 423 | 8 | 182 | 87.67% | 5.56 px |
| Wall Separator with MoGe-2 depth | 423 | 8 | 183 | 77.55% | 15.89 px |
| Underside Finder with M-LSD Tiny | 404 | 14 | 174 | 84.89% | 6.72 px |
| Underside Finder with DeepLSD | 404 | 14 | 175 | 84.89% | 6.73 px |
| Underside Finder with ScaleLSD v2 | 404 | 14 | 175 | 84.88% | 6.73 px |
| Underside Finder with both DeepLSD and ScaleLSD v2 | 404 | 14 | 176 | 84.88% | 6.73 px |

All four line inputs found the same six additional scored undersides. DeepLSD and ScaleLSD each matched one additional *below-cutoff* surface compared with Tiny. Tiny left the fewest unpaired guesses and is the optional underside build method. MoGe-2 and Depth Anything V2 tied on the number of matched surfaces, but Depth Anything assigned more of each face to the right region. Its border error fell by 1.27 source pixels relative to the previous default. All methods covered the same 601 of 16,630 sampled opening pixels. Faces below the regular size cutoff are paired separately before counting extra guesses. [Per-picture scores and exact changes](reviews/separation-01/cases.json)

## Where did it work?

| Underlying layout | Previous finder | Wall Separator with Depth Anything V2 | Underside Finder with M-LSD Tiny |
| --- | ---: | ---: | ---: |
| Shopfront | 77 | 82 | 79 |
| Wing | 30 | 32 | 30 |
| Courtyard | 43 | 48 | 43 |
| Portico | 107 | 109 | 109 |
| Workshop | 23 | 27 | 25 |
| Crossing | 31 | 34 | 31 |
| Galleries | 66 | 68 | 66 |
| Registry | 21 | 23 | 21 |

Counts are visible surfaces per picture, summed over that layout’s views and palettes. On the 20 development pictures of Shopfront and Wing, the previous finder, Wall Separator and Underside Finder recovered 107, 114 and 109 of 156 surfaces. After settings and input choices were frozen, the same methods recovered 291, 309 and 295 of 442 on the 36 previously reviewed regression pictures. Neither selected challenger lost an existing scored match. The Wall Separator introduced three new instances of splitting an already matched face into substantial fragments. The two refinements were tested independently. [Development selection](reviews/separation-01/development-selection.json) · [All layout and split scores](reviews/separation-01/scores.json)

The ordinary Block Counter got both dimensions right for 64 surfaces with the Wall Separator, up from 56. Nine previously wrong or missing sizes became correct and one previously correct size was lost. Wrong sizes increased from 23 to 30 while missing sizes fell from 519 to 504. The Underside Finder stayed at 56 correct. These measurements cover detection and counting. World coordinates and in-game placement were outside this round. [Count check](reviews/separation-01/counts/scores.json)

## How a picture was processed

1. MoGe-2 estimated surface directions and Depth Anything V2 estimated relative depth. Regions First grouped connected pixels; the earlier contrast-alignment step refined their boundaries. These masks were frozen for all comparisons.
2. **Wall Separator.** The model supplies relative depth. Image-processing code proposes a split from a change in color, a change in depth, or a side or footprint from the legacy **Rectangle Finder**. It checks that a depth change and visible image edge support the division. The rectangle is only a hint; the new regions retain the original holes. Older reports call the Rectangle Finder **B**.
3. **Underside Finder.** A line model proposes edges from the source pixels. The strip-finding code checks nearby parallel edges, shade and surface-direction evidence. A supported strip can become a new region. The previous line experiment could only move a border between already existing regions.
4. **Scoring.** Predictions are saved before consulting the recorded surface geometry. A scored surface needs a one-to-one match with at least 50% shared outline area divided by combined area. The report also records false proposals, gaps, fragmented walls and pixel accuracy. The camera, block states and true outlines are evaluation inputs.

The alternate MoGe-2 depth test uses the installed checkpoint’s 3D point output. The adapter estimates its forward offset from the projection of the predicted points, then takes reciprocal depth. It does not use a recorded camera or the model’s metric scale. This comparison measures that relative-depth adapter under the same wall rules. [Official MoGe ONNX output documentation](https://github.com/microsoft/MoGe/blob/main/docs/onnx.md)

## Replay or build one image

```powershell
.\Invoke-SurfaceRound.ps1 -Round separation
```

The round is image-only. It verifies the 56 captures and 52 distinct screenshot hashes, uses the frozen line segments, reads compatible cached MoGe-2 and Depth Anything V2 fields and caches the MoGe-2 point-head result. Its CPU picture jobs run in parallel. Use `-Round detection-sheet` for the quicker replay of just the saved comparison sheet. [Model checkpoints, settings, runtime and artifacts](reviews/separation-01/manifest.json)

```powershell
python reconstruct.py build screenshot.png --version 26.3 --through surfaces --out runs/walls
python reconstruct.py build screenshot.png --version 26.3 --through surfaces --surface-boundary strip-mlsd --out runs/undersides
```

The first build uses the default Wall Separator and pinned local MoGe-2 and Depth Anything V2 models. The optional underside build also needs the M-LSD Tiny 512 FP16 checkpoint and `ai-edge-litert==2.4.0`. Use `--surface-boundary color-edges` for the previous contrast-aligned baseline. [Local models and checkpoints](LOCAL_VISION_MODELS.md)
