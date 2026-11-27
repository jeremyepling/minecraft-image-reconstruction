# Prompted models on the six sky views

Keep unconditional SAM replacements experimental. SAM ViT-B with the original support bounding box improved mean overlap on both splits, but all five model strategies lowered pixel-weighted IoU. In particular, the box-only model added 264,764 incorrectly assigned pixels when summed across the 193 proposals. The sibling selection and reference experiments test whether it helps to keep the original support selectively.

| Source-only arm | Development mean IoU | Other buildings mean IoU | All mean IoU | Pixel-weighted IoU | Targets at ≥0.50 IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original mask | 52.33% | 40.66% | 44.53% | 32.69% | 92 |
| MobileSAM box | 50.92% | 41.40% | 44.55% | 28.77% | 90 |
| MobileSAM box and points | 50.81% | 41.88% | 44.84% | 29.83% | 93 |
| SAM ViT-B box | 54.03% | 42.86% | 46.56% | 29.99% | 99 |
| SAM ViT-B box and points | 48.15% | 40.70% | 43.17% | 28.88% | 81 |
| SAM ViT-B local crop, box and points | 52.30% | 41.90% | 45.35% | 27.91% | 91 |

The comparison uses the 64 source surfaces from the two train buildings and 129 surfaces from four validation/test buildings in the frozen floating-build round, base revision 52de2c9. Each mask keeps the identity of its original proposal and its scored reference target. These are native pixel crop scores. This benchmark does not itself reconstruct final blocks.

The predictor sees the source RGB image and the proposed support. Box prompts use its tight native bounds. The point strategy adds up to three spatially separated interior points and two negatives in unclaimed portions of the box. Each seed maximizes the distance to the remaining support boundary. Each model returns three proposals and uses its own predicted IoU to choose one. The local-crop strategy enlarges the support rectangle by half its longer side, with a 16-pixel minimum margin, and re-encodes that image patch before prompting. Full-frame strategies use one shared image embedding per screenshot. The original supports are the no-op arm.

## Replay

From this worktree, run these commands with the mask-scalelsd CUDA Python environment. The predictor uses the existing MobileSAM checkout under surface-recovery-round. Fetch the SAM ViT-B weights into the output directory first. The generator checks both model digests and the implementation revision. [Official SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

~~~powershell
../mask-scalelsd/runs/env/Scripts/python.exe -m benchmark.sky_crop_models mobile
../mask-scalelsd/runs/env/Scripts/python.exe -m benchmark.sky_crop_models sam-b
../mask-scalelsd/runs/env/Scripts/python.exe -m benchmark.sky_crop_models sam-b --zoom
~~~

The five archived candidate files use native row runs with exclusive right endpoints. The three manifests identify their exact weights, predictor source, image and proposal hashes, and outputs. Scores use the paired evaluator from experiment/sky-crop-reference.
