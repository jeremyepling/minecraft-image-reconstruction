# Learning sequential crop corrections

The student learned to revise crop boundaries from labeled mistakes on Pillars and Viaduct. On four held-out structures, its four edits lifted ordinary crops from **42.30% to 47.43% mean fixed-target IoU**. A crop chooser already scored **50.70%** on these same targets. Applying the identical student after that chooser reached **50.84%** and reduced Harbor's score. Keep `experiment/sky-crop-correction-policy` experimental. The new student partly learns corrections already captured by the chooser and can keep expanding a crop in the wrong direction.

## Measured crop edits

Every pass sees the current mask. It considers adding or removing source pixels within five native pixels of its boundary. Training uses frontmost visible connected coplanar full-cube reference targets locked to the ordinary proposal IDs. Inference uses only the image, frozen source model outputs and the mask resulting from the preceding pass. It carries every accepted and mistaken edit forward.

| Initial mask and cohort | Matched crops | Initial IoU | Pass 1 | Pass 2 | Pass 3 | Pass 4 |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary, development | 64 | 52.33% | 59.94% | 62.94% | 64.50% | 65.56% |
| Ordinary, held out | 124 | 42.30% | 45.73% | 46.83% | 47.31% | **47.43%** |
| Chooser, development | 64 | 60.95% | 63.57% | 64.95% | 65.95% | 66.75% |
| Chooser, held out | 124 | 50.70% | 51.06% | 51.10% | 51.02% | **50.84%** |

| Structure | Ordinary start → pass 4 | Chooser start → pass 4 |
|---|---:|---:|
| Pillars, development | 53.58 → 67.67% | 63.06 → 70.08% |
| Viaduct, development | 51.16 → 63.58% | 58.97 → 63.63% |
| Harbor, validation | 45.42 → 51.75% | 54.74 → 53.23% |
| Hillside Chapel, validation | 24.15 → 31.89% | 37.46 → 38.44% |
| Castle, test | 46.61 → 50.97% | 50.95 → 52.75% |
| Walled Garden, test | 49.05 → 52.05% | 55.40 → 55.65% |

With ordinary inputs, 89 held-out crop IoUs improved and 35 declined between the first and final masks. With chooser inputs 60 improved, 53 declined and 11 tied. Five additional Garden crops had no positive reference match under the full-cube scoring definition. The raw student edited all 193 proposals blind. All five unpaired IDs pass through as their original ordinary masks in the separately archived scope-matched packet. The 188-crop table omits those IDs at every step and for both initial masks.

**Hillside Chapel S11:** ordinary crop IoU moved 8.1 → 29.9 → 61.4 → 76.5 → 81.2%. The first pass removed 696 wrong pixels and added 647 correct ones. The next three passes added 1,011, 486 and 150 correct pixels without adding incorrect pixels.

**Walled Garden S34:** IoU fell 80.6 → 78.9 → 66.5 → 56.3 → 48.7%. The first pass added 105 correct and 168 incorrect pixels. Passes three and four grew onto 326 and 329 incorrect pixels, respectively, without recovering additional target pixels. The same surface regressed from its stronger chooser initialization.

Across the held-out ordinary inputs, the first pass removed 43,530 wrong and 5,364 correct pixels and added 11,214 correct and 6,986 wrong pixels. By pass four, wrong additions outnumbered correct additions, 5,343 to 4,503. This explains why continued editing gave progressively smaller gains in mean IoU. Per-crop intersection, extras, misses and all four edit outcomes are retained for every step. [Ordinary traces and score receipts](ordinary-scores.json.gz) · [Chooser-initialized score receipts](chooser-scores.json.gz)

## Training and freeze

The classifier is a 32-tree OpenCV random forest with depth limit seven. It learns inside/outside target membership from current-boundary samples, using a deterministic budget of 1,200 samples per crop and training state. Training first fits the original development errors, rolls that policy out twice, and fits the final student to the original states and resulting residual errors. Each prediction recomputes local mask coverage, interior/exterior RGB and relative depth evidence, median normal and affine inverse-depth consistency. Fixed source evidence includes four prompted SAM supports, native RGB gradient and border-connected sky. Pixels are added at positive vote fraction ≥0.70 and removed at ≤0.30.

Using just the two development buildings, the learner was fitted twice, holding out an entire building each time. The best mean cross-building score occurred at pass four of four tested passes. On held-out Viaduct the cross-development sequence was 51.16 → 58.22 → 60.58 → 61.43 → 62.02%. On held-out Pillars it was 53.58 → 58.40 → 60.03 → 60.77 → 61.25%. The final model was fitted to both development buildings. The same model, thresholds and four-pass stopping rule were used for all raw inference, also when changing the initial mask to the prior source-only chooser. [Frozen policy and hashes](freeze.json) · [Trained forest](forest.xml.gz)

The reference mask is needed only by the training and scoring entry points. To replay the experiment from this checkout, use the installed OpenCV benchmark interpreter:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$python = '../mask-scalelsd/runs/env/Scripts/python.exe'
& $python -m benchmark.sky_crop_correction_policy train
& $python -m benchmark.sky_crop_correction_policy infer --start ordinary
& $python -m benchmark.sky_crop_correction_policy infer --start chooser
& $python -m benchmark.sky_crop_correction_report
```

To run inference without refitting, copy the archived forest and freeze JSON to `runs/sky-crop-correction-policy/` before invoking either inference command. The four hash-checked source mask packets come from `experiment/sky-crop-models`. Fitting reads the two development reference maps using `experiment/sky-crop-reference`. The final scorer verifies the native edit counts against the source-only trace.

The trajectory files contain round 0 and every attempted native support packet, plus added/removed pixel counts including the five unpaired crops. All packets are sparse relative to ordinary supports. [Ordinary mask trajectory](ordinary-trajectory.json.gz) · [Chooser mask trajectory](chooser-trajectory.json.gz) · [Final ordinary-start supports](ordinary-selected.json.gz) · [Final chooser-start supports](chooser-selected.json.gz)

## Placement scope

This branch archives the learned student, inference and crop score evidence. It does not select a new default. Placement and exact-ID evidence for the six sky exports belongs to the coordinator's downstream replay. Four sky pictures come from Visible Builds v3 and two from Decorative Builds v1. No full cohort was rerun for this correction policy. Production scores carried forward from `e3bd220` are below. [Published cohort results](../camera-native-round-01/README.md)

| Cohort | Full-cohort placement runs here / available | Prior strict / facade placement F1 | Prior strict / facade position + ID F1 |
|---|---:|---:|---:|
| Fixed trend | 0 / 104 | 91.27 / 93.42% | 84.68 / 88.51% |
| Visible Builds v3 | 0 / 144 | 46.63 / 48.04% | 28.04 / 29.08% |
| Decorative Builds v1 | 0 / 40 | 63.46 / 65.23% | 50.51 / 52.56% |
