# How much does a barely visible roof cost?

On the 104 framed views scored at `1c2f77d`, the strict visible-build score is **90.30% F1**. Making barely visible top surfaces optional gives **90.76%** if the surface cannot contain a 4 × 4 pixel patch, or **91.53%** for 8 × 8. A facade score that requires every block with a visible vertical face reaches **92.48%**. These are three evaluations of the identical exported blocks. [Three annotated photographs](facade-examples.jpg) · [Full totals](summary.json) · [Each photo](frames.json)

| Optional reference blocks | Optional total | Already recovered | Currently missed | F1 when optional | Strict F1 if all these misses were correctly added |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 0 | 0 | 0 | 90.30% | 90.30% |
| Top-only block with fewer than 16 native pixels | 251 | 18 | 233 | 90.76% | 90.86% |
| Top-only block on a surface lacking a 2 × 2 patch | 192 | 5 | 187 | 90.67% | 90.75% |
| Top-only block on a surface lacking a 4 × 4 patch | 240 | 8 | 232 | 90.76% | 90.86% |
| Top-only block on a surface lacking an 8 × 8 patch | 688 | 68 | 620 | 91.53% | 91.78% |
| Top-only block on a surface lacking a 16 × 16 patch | 1,706 | 925 | 781 | 91.53% | 92.16% |
| All top-only blocks | 3,302 | 2,293 | 1,009 | 91.47% | 92.69% |
| All top- and underside-only blocks (facade score) | 4,282 | 2,765 | 1,517 | 92.48% | 93.85% |

Every photo is 1440 × 900. The reference face ID at each native pixel identifies the first visible block, its face direction and its connected surface. Surface membership follows edge connectivity in a coplanar layer across materials. A surface passes the patch test when an axis-aligned square of that size fits anywhere inside pixels owned by that surface. An isolated partly exposed block on a broad roof can therefore have few pixels while its surface passes. The test can classify small islands as well as thin ribbons. The thresholds describe image support. They have not been calibrated to human recognition or other resolutions.

The evaluator decodes each exported pack and uses its previously scored whole-build translation. It removes optional targets from the denominator and neutralizes correct predictions on those targets. All **1,157 false placements** still count, including the 106 at hidden reference positions. Every block with at least one visible vertical pixel stays required by each top-sliver metric, including rims, walls and posts. The facade score includes all visible vertical directions. A wider exemption can lower a score when it removes many correctly placed blocks. The last column is a separate ideal reconstruction bound that adds all missing optional targets while leaving every existing prediction in place.

## What remains

Of **3,240 missed visible positions**, 1,009 had only a top face, 508 only an underside and 1,723 had a vertical face. Of the latter, **1,466 had at least 64 visible vertical pixels** and 1,220 had at least 128. Correctly adding every missed horizontal-only position while keeping existing errors would reach 93.85% on the original strict evaluation. The result rules out tiny roofs as the sole explanation of the current gap to 95%.

- **Open Canopy 00, right.** All 27 blocks with a visible vertical face were recovered, with zero extras. The 23 other visible blocks showed only roof tops and all were missed. Strict F1 was 70.13%. Two misses qualify at the 4 × 4 cutoff and all 23 qualify at 8 × 8. F1 becomes 72.00% and 100.00% respectively. [Photograph and enlarged detail](shape-benchmark-open-26.3--train-canopy-00-right.jpg)
- **Canopy 04, left.** Four of 23 top-only blocks and 32 of 37 blocks with a vertical face were recovered. The 8 × 8 rule makes all 23 top-only blocks optional. Five facade misses and three extras remain, scoring 88.89% against 72.73% strict. [Photograph and enlarged detail](shape-benchmark-26.3--train-canopy-04-left.jpg)
- **Shopfront, plaster, low left on concrete.** None of its 169 targets qualifies for the 4 × 4 rule. Forty-two misses have a vertical face, all with at least 64 native wall pixels. Its 103 correct placements and 21 extras give 70.31% strict F1. [Photograph and enlarged detail](legacy-light_gray_concrete-dev-shopfront-plaster-low-left.jpg)

A separate perfect chooser moved only **existing extras** exactly one cell along a single grid axis into unclaimed missed targets. Bipartite maximum matching prevents one extra from correcting two misses. With the existing translation frozen and all other cells untouched, at most 350 such moves are available, yielding **91.84% strict F1**. Restricting moves to horizontal X/Z gives 299 (91.62%) and vertical Y gives 59 (90.56%). At most 283 one-step moves can recover positions with a visible vertical face. These ceilings use the reference answers. An image-only model must still identify the correct moves, and generating other candidates could raise the ceiling.

## Did the Surface Finder see the missed walls?

Among the 1,466 missed blocks with at least 64 native vertical-face pixels, the Surface Finder covered at least half those pixels on **1,209 blocks (82.5%)**. On **1,052 (71.8%)** a single surface, with a normal within 20 degrees of the correct wall axis, covered half the evidence. Shopfront on concrete had 42 eligible misses. Some mask covered at least half on 35 and a single correctly oriented mask covered half on 28. A trace of grid registration, plane depth and placement filtering on these 1,052 blocks would distinguish failures downstream of detection. Coverage can include a merged or oversized surface and says nothing by itself about depth. [All 104 per-frame coverage records](wall-masks.json)

| Predicted surface evidence | ≥50% of block's visible wall pixels | ≥75% |
| --- | ---: | ---: |
| Any masks combined | 1,209 | 1,172 |
| Any vertical masks combined | 1,197 | 1,150 |
| Masks with the matching wall axis combined | 1,078 | 967 |
| One mask of any orientation | 1,169 | 963 |
| One mask within 20° of the matching wall axis | 1,052 | 891 |

Each support mask was projected to native pixel centers using its stored resolution. At corners, evidence is counted against the actual owning face pixel by pixel. Orientation is graded against the unsigned world-space wall axis through the recorded camera rotation, after the image-only predictions are frozen. The 64-pixel minimum and either coverage threshold apply per block.

## Replay

Run in a Windows reconstruction worktree with the frozen suite restored. The paper trail checks the camera, readback, labels and archive hashes through the standard fixed-view loader. Per-frame records include the RGB hash, pack hash, translation and optional counts. Use a separate output path for a challenger run with the same 104 views.

```powershell
python -m benchmark.facade_visibility --run ../roof-height-round/runs/production --out benchmark/reviews/facade-visibility-01 --workers 4
python -m benchmark.facade_mask_coverage --run ../roof-height-round/runs/production --out benchmark/reviews/facade-visibility-01/wall-masks.json --workers 4
```
