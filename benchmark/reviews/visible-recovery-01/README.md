# Visible geometry after foreground and horizontal-face recovery

The three changes together recovered **19,408 of 24,086 visible block positions**, up from 18,703, on the fixed 112-image comparison. With 1,794 extras, pooled relative-geometry F1 rose from **83.95% to 85.71%**. Precision went from 91.35% to 91.54% and recall from 77.65% to 80.58%. The scorer decodes the exported Minecraft packs and permits one integer translation for the entire visible structure. Every offset on the final evaluation was zero. This evaluates geometry. The ordinary geometry pack uses smooth stone for placement.

| Compared on the same photographs | Correct positions | Extras | Relative F1 |
|---|---:|---:|---:|
| Previous default, `3585aa5` | 18,703 | 1,770 | 83.95% |
| Conservative foreground depth decisions alone | 18,834 | 1,711 | 84.40% |
| Guarded neighboring texture alone | 18,783 | 1,777 | 84.14% |
| Guarded two-pixel sampling alone | 19,190 | 1,843 | 85.06% |
| Foreground and neighboring texture together | 18,914 | 1,718 | 84.59% |
| All three in the public placement path, `eea027b` | **19,408** | **1,794** | **85.71%** |

The final rule gained 708 correct positions and lost three. It added 111 erroneous positions and removed 87. Extras coinciding with hidden reference structure rose from 94 to 96. Sixty-five photographs improved, 43 tied and four declined. Eight camera abstentions remain in the denominator. There are 108 unique RGB hashes in the 112 rows because a few close-ups are pixel-identical across ground variants. The separate 72 training images were used in development. The 12 previously reviewed transfer views also informed some rule choices. The fixed suites have been examined in earlier experiments. [Frame-level scores, input hashes and all 184 exported packs](scores.zip) · [Cohort summary](summary.json)

| Cohort | Screenshots | Previous F1 | New F1 | Visible targets |
|---|---:|---:|---:|---:|
| Cloister validation | 12 | 91.20% | 91.85% | 2,476 |
| Twin Spires test | 12 | 91.34% | 92.30% | 3,414 |
| Fixed exteriors | 56 | 81.13% | 83.52% | 10,626 |
| Earlier supported native views | 12 | 84.97% | 86.47% | 3,780 |
| Reviewed transfer views | 12 | 77.55% | 79.47% | 2,359 |
| Additional transfer views | 8 | 81.06% | 82.91% | 1,431 |

## Confirmation on additional views

Before scoring, we reserved all 20 remaining camera-supported photographs from the native archive after excluding every RGB hash in the fixed suite and the earlier transfer selections. The camera views were new to this comparison. The 16 structures had already appeared from other angles. Pooled F1 rose from **79.58% to 81.94%**. Correct positions rose from 2,986 to 3,173 of 4,098. Extras rose from 420 to 474, including two more on hidden structure. Precision went from 87.67% to 87.00% and recall from 72.86% to 77.43%. Twelve views improved, five tied and three declined. Eight alternate canopy views recovered 131 of 398 roof blocks at the right offsets from both posts, up from 112. Six opening widths were correct and two were unresolved under both methods. [Reserved photographs, input hashes and paired builds](../visible-recovery-extra-views-01/README.md)

## What changed in an exported build

The foreground decision compares plausible surface depths. If a proposed block would project in front of another independently observed surface, a depth that preserves both observations can win. The rule keeps the prior decision when evidence for switching is weak. The next pass checks whether the visible part of an adjacent face matches the source photograph at the same position within a texture tile. The last pass tests projected tops and undersides every two image pixels. Two neighboring blocks nominate a location. Visible face pixels and their measured orientations must support placement. Both recovery passes preserve native-pixel visibility of earlier placements. [Foreground depth alternatives](../foreground-conflicts-01/README.md) · [Neighboring texture controls](../neighbor-photo-01/README.md) · [Two-pixel and four-pixel controls](../native-sample-density-01/README.md)

On ten canopies all ten opening widths stayed correct and all 137 sampled clear rays stayed open. Of 459 visible roof blocks, **194** were placed at the correct offsets from both front posts, up from **170**. All 52 paired post courses stayed aligned. Across the 80 validation, test and regression exteriors, genuinely visible downward faces recovered rose from **778 to 899 of 1,472**. The two-pixel samples helped particularly on faces that occupy long, thin strips in the original photograph. [Source photographs, visible target and both builds](comparison.jpg) · [Roof and post measurements](spacing/cases.json) · [Visible undersides](undersides.json)

Keep `experiment/front-wall-atlas-grid` experimental. It recovered 262 of 384 positions on the eight close-up rows. Identical pixels could also come from looking straight down at a textured floor. The synthetic controls demonstrated that the image can support an exact tile count while leaving the plane's orientation unresolved. [Close-up experiment and orientation controls](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/62de481/benchmark/reviews/front-wall-grid-01/README.md)

## Ordinary builds and remaining misses

On the smooth-stone Shopfront high-right photo, a fresh command-line build recovered 157 of 216 visible blocks with 39 extras. The rollback command recovered 56 with 37 extras. One surface changed depth. The following passes added two texture-matched blocks and 18 horizontal-face blocks. Default and rollback each exactly reproduced the coordinates and materials of their corresponding frozen replay. A second full command-line check on Canopy 00 also reproduced both packs. [Command-line checks](cli-checks.json)

At the new default, 4,678 visible positions are still missing. Even moving every extraneous cube onto a different missing position would yield only 93.63% F1. The next useful tests are local depth and face-direction alternatives on already detected pixels and new proposals for narrow photographed tops and undersides. There are 1,670 misses with strong support in the existing orientation masks, and 494 misses between the original sample points with at least 16 native pixels in a matching mask. [Remaining errors and limits of each correction](../post-placement-budget-01/README.md)

## Replay

Restore the [frozen upstream inputs](../canopy-placement-01/README.md) and [native transfer fixtures](../canopy-fixtures-01/README.md) first. The baseline below is the released placement before this round. The model and atlas fallbacks still apply. To explicitly run the previous placement rule on a new screenshot, add `-IndependentPlaneDepth` to `Invoke-WallRecovery.ps1`.

```powershell
$prior = 'runs/canopy-placement/release'
$dest = 'runs/visible-recovery-replay'
$method = 'benchmark.visible_depth_score:integrated'
python -m benchmark.visible_depth_score --split all --candidate $method --baseline $prior --out $dest
python -m benchmark.visible_depth_score --split native --candidate $method --baseline $prior --out $dest
python -m benchmark.canopy_compare --inputs runs/visible-depth-rounds/transfer --out "$dest/old" --baseline "$prior/old" --candidate-root . --candidate $method
python -m benchmark.canopy_compare --inputs runs/canopy-placement/confirmation --out "$dest/fresh" --baseline "$prior/fresh" --candidate-root . --candidate $method
python -m benchmark.relative_canopy_spacing --previous $prior --current $dest --out runs/visible-recovery-spacing
python -m benchmark.visible_recovery_report --predictions $dest --out runs/visible-recovery-report
```
