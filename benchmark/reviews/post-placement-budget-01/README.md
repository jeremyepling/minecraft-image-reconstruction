# Geometry remaining after the combined placement changes

The combined foreground chooser, neighboring texture check and dense sampling pass at `eea027b` placed 21,202 blocks across the 112 evaluation images. Of 24,086 visible targets, 19,408 matched and 4,678 were missed. There were 1,794 extra placements. F1 reached **85.71%** after allowing one translation for each complete build. Every best translation on these packs was zero. The evaluator reopened each datapack, checked its SHA-256, decoded its cells and reproduced the published per-image aligned counts. [Case counts and input manifest hashes](cases.json)

## Which missed blocks have placement evidence?

| Miss category, mutually exclusive | Previous default | Combined |
| --- | ---: | ---: |
| Supported camera, strong correct-axis mask samples | 1,986 | 1,670 |
| Supported camera, weak correct-axis mask samples | 756 | 618 |
| Detected samples only on the wrong axis | 482 | 462 |
| Sample hits the block outside all detected masks | 265 | 264 |
| Native-visible block missed by every stride-four sample | 1,510 | 1,280 |
| Camera abstained | 384 | 384 |
| Total | 5,383 | 4,678 |

Strong support requires at least four coarse samples on masks consistent with the true visible face axis and at least half of that block's coarse samples. Weak support has at least one such sample. A visible reference block with zero coarse samples can span a long, thin image region. Among the 1,280 skipped targets, **955** have at least 16 visible native pixels; **494** retain at least 16 native pixels within the corresponding detected axis mask. By dominant visible pixel count, 886 show an upward face, 335 a downward face and 59 a side. A denser pass that still requires a detected face axis can help the 494 with robust same-axis evidence, while a method for other skipped faces also needs to address mask coverage or smaller fragments.

For **865** remaining targets, an extra projected coordinate lies strictly in front of at least half the photographed target pixels. For **590**, extras precede every target pixel. Of the 1,670 misses with strong correct-axis masks, 712 also have a majority of their pixels blocked this way. These counters overlap the mutually exclusive categories above.

## How many depth errors lie on the same rays?

The diagnostic rasterizes the actual decoded build using the scoring camera. It identifies the frontmost predicted face on each native pixel, then looks for overlap with a missing visible reference face. A correspondence requires at least four native pixels *and* at least one quarter of the visible reference face. For the table below those shared pixels must belong to a detected mask consistent with the reference axis, and both faces must point in the same signed direction. Ground truth is used only to diagnose the available depth choices.

| Maximum normal-axis plane distance | Misses with an overlapping parallel face | Misses linked to an extra coordinate | Distinct extra-to-miss relocations | Distinct relocations retaining both in-plane coordinates |
| --- | ---: | ---: | ---: | ---: |
| 1 block | 483 | 455 | 395 | 115 |
| 2 blocks | 716 | 675 | 544 | 129 |
| 4 blocks | 771 | 729 | 578 | 133 |
| 8 blocks | 825 | 783 | 606 | 133 |

The distinct count uses maximum bipartite matching between incorrect output coordinates and missing visible targets. Each coordinate can move once. This is an ideal selection at the reference alignment. Changing the plane depth and intersecting its image rays again may also change in-plane block indices. A normal-only move allows just 133 relocations in the four-block window, worth **86.30%** F1 with perfect decisions. Allowing that in-plane change and perfectly selecting all 578 four-block relocations gives **88.26%**. Expanding to eight blocks raises the ideal to **88.39%**. Relaxing the mask requirement changes the four-block maximum from 578 to 581. The limits apply to these explicit, already-rendered correspondences. Newly proposed faces or layers can extend this pool.

The missing cells' correctly oriented masks overlap *some* frontmost predicted face on 2,037 targets at the same area threshold. Only 834 have that much overlap with a face pointing in the correct signed direction. A candidate generator that considers other face orientations at disagreements is a distinct opportunity from a wider depth search.

| Suite | Misses | Strong axis support | No coarse sample | Majority blocked by extras | Distinct four-block relocations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation, 12 | 257 | 133 | 48 | 32 | 25 |
| Test, 12 | 387 | 150 | 123 | 101 | 68 |
| Fixed exterior, 56 | 2,393 | 819 | 487 | 416 | 248 |
| Native, 12 | 721 | 187 | 317 | 116 | 70 |
| Previously reviewed native, 12 | 600 | 253 | 188 | 127 | 115 |
| Newer confirmation, 8 | 320 | 128 | 117 | 73 | 52 |

## What would 95% require?

Even an oracle that moved **every** one of the 1,794 extras onto a different true miss could only make all 21,202 current placements correct. That would reach **93.63%** F1. At perfect precision, at least 21,793 visible placements are required for 95%, or **591 further correct additions** after the perfect relocation. If the only corrections were the 578 ideal four-block relocations above, another **2,907 correct additions** would be needed at unchanged precision-error count. These are arithmetic scenarios with no further mistakes.

Try a two-block, image-evidence-based peer to the foreground chooser that proposes replacement planes and re-intersects the photographed rays, with different face orientations allowed when the detected mask and rendered face disagree. A separate native-resolution proposal pass is needed for visible top and underside strips. For that pass, count each candidate's visible pixels after joint occlusion and compare grid phase against neighboring texture or atlas alternatives. Score both the added candidates' precision and the missed coordinates they still cannot propose.

## Replay

Use the frozen image, camera and upstream inputs and the seven-cohort combined replay setup. Point `--predictions` to the directory of verified combined prediction packs and manifests. This audit covers its six evaluation cohorts. The earlier budget replay matched every original count and frame record.

```powershell
python -m benchmark.neighbor_depth_headroom --predictions ../combined-visible-recovery/runs/triple --workers 4 --out runs/post-placement-budget/cases.json
```
