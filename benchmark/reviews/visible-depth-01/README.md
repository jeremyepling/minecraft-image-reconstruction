# Placing visible Minecraft blocks at the right depth

The new Java 26.3 default placed **17,440 of 22,655 visible block appearances** correctly across 104 evaluation screenshots, up from 14,223. Extra coordinates fell from 7,492 to 2,374. Of these, 107 coincided with hidden structure, down from 1,731. The targets are visible blocks from each individual screenshot. Extra placements are errors even when the source world happens to contain a block there.

## Decoded end-to-end results

Each cell in the two method columns gives correct positions / extra positions / extras on hidden structure. The last column is the harmonic mean of precision and recall for visible positions. These views were excluded from fitting the depth model. The fixed and earlier native suites have informed previous iterations. The fresh transfer suite was selected from separate native builds by a locked camera-support and geometry-diversity rule before scoring their placements.

| Suite | Visible target | Previous default | New default | Position F1, before → after |
|---|---:|---:|---:|---:|
| Cloister, 12 | 2,476 | 1,799 / 713 / 140 | 2,171 / 126 / 0 | 72.1% → 91.0% |
| Twin Spires, 12 | 3,414 | 2,218 / 1,064 / 518 | 2,925 / 154 / 14 | 66.2% → 90.1% |
| Fixed exteriors, 56 | 10,626 | 5,967 / 3,508 / 978 | 7,780 / 979 / 46 | 59.4% → 80.3% |
| Earlier supported native views, 12 | 3,780 | 2,760 / 489 / 27 | 2,926 / 256 / 5 | 78.5% → 84.1% |
| Fresh supported native builds, 12 | 2,359 | 1,479 / 1,718 / 68 | 1,638 / 859 / 42 | 53.2% → 67.5% |

All eight fixed exterior families gained correct coordinates and shed extras. Eight fixed screenshots still lacked sufficient camera scale and produced empty packs. Native transfer scores are conditional on camera support. The additional screen found supported cameras in 40 of 140 eligible screenshots spanning 38 novel shapes. Twelve distinct shapes were selected for placement testing. These suites are deliberately structured and their pooled numbers describe these screenshots.

The exporter uses smooth stone for geometry. Exact position **and** material improved from 73 to 85 on the earlier native views and from 23 to 35 on the fresh native builds. It scored zero on Cloister, Twin Spires and the fixed suite because their visible targets have other block IDs. The trained-on 72 views separately yielded 13,769 correct of 18,322 with 2,458 extras, including 160 on hidden structure.

[Screenshots, predicted masks, visible targets and decoded builds](comparison.jpg) · [Frozen upstream inputs, baseline and split details](../visible-depth-inputs-01/README.md) · [Fresh native release and camera audit](../visible-depth-transfer-01/README.md) · [Decoded builds, paired cases, ablations and perfect-chooser decisions](evidence.zip)

## Which decisions helped

1. **Choose a depth for each observed surface.** A 12-weight model ranks candidate integer planes using evidence from the photographed surface. It checks texture edges, the outline and whether projected block squares match vanilla sprites. Two adjacent squares can each match a different material. Using the six building families in turn as held-out training folds, the ranker recovered 13,563 visible blocks with 3,040 extras. The previous placer recovered 10,056 with 7,330 extras. This change shipped at `53b19c0`. [Model, candidate depths and training comparisons](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/3f822ae/benchmark/reviews/atlas-plane-depth-01/README.md)
2. **Verify partly visible faces and allow a second visible depth.** A partly obscured tile can pass if separate parts of the crop agree on the same sprite and orientation. The visible projected face must align with the detected surface axis. Against the plane ranker, this recovered 126 correct coordinates and lost five across the earlier 92 views, adding four net extras. A second local depth inside a merged mask requires neighboring tiles with strong atlas matches and face coverage. On the fresh builds this last refinement added six correct placements and removed six hidden extras. [Partial-face ablations and per-case results](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/3edd8e0/benchmark/reviews/visible-face-reprojection-01/README.md)
3. **Recheck visibility from the input camera before export.** A proposed cell with no ownership on the coarse pass is dropped only after a full-resolution check also finds no visible pixel. Over the previous two stages combined, this removed 214 extras on Cloister and 120 on Twin Spires without losing a correct block there. On the fixed exteriors it removed 49 extras, including 27 on hidden structure, and lost 22 correct blocks, all on Workshop. An incorrectly placed foreground block can obscure a real face in the prediction. Both native suites produced identical coordinates with and without this check. The verified partial-face, secondary-depth and visibility steps shipped at `659539b`. Missing atlas or model weights preserve the existing placement fallback.

The learned cell-rejection tree remains experimental. On the earlier native suite, adding it to the ranker and partial-face check removed 79 extras but also removed 98 correct blocks. The milder two-clause rule performed well on the fresh canopies while still losing correct cells on other buildings. A floor-calibrated Depth Anything prior improved some offsets, but added 20 errors on hidden structure on the fixed suite relative to the atlas ranker. Direct depth substitution also regressed on Cloister with both Depth Anything and MoGe. [Cell-decision models](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/567be25/benchmark/reviews/visible-cell-decisions-01/README.md) · [Monocular depth trials](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/f73d561/benchmark/reviews/metric-plane-depth-01/README.md)

## Where to focus

On the fresh builds, 11 of 12 improved in position F1. The largest remaining failure is canopy 09 from the right. It went from 26 correct and 264 extras to 22 correct and 301 extras. Its camera's 90th-percentile error at visible corners was only 0.44 pixels. A perfect whole-frame choice between the two defaults would revert that one picture and raise pooled F1 from 67.46% to 68.09%. This oracle reads the answers and measures the scope for a fallback. It cannot recover the other errors within the chosen build. [Canopy photograph, visible target and both decoded roofs](canopy.jpg)

For the next geometry experiment, isolate the thin roof edge, underside and supports. Give a surface-level chooser competing depths or an abstain option, score both its visible faces and openings it incorrectly fills, and measure the perfect choice among those proposals before fitting. Use separate new canopy shapes for confirmation. For material recreation, the next decision is whether each projected face gives enough atlas evidence to name an already-placed block. Grade exact coordinate plus block ID in the exported pack. [Earlier projected-material trial](../../COMBINED_DEFAULT_EVALUATION.md)

## Replay

Restore the frozen inputs and baseline from the linked input release. From a checkout at `659539b` or the report revision, run the following. The scorer calls ordinary placement with the frozen image-derived camera, writes and reads the datapack, and then uses recorded labels and pose to score it.

```powershell
python -m benchmark.visible_depth_score --split all --candidate benchmark.visible_depth_score:integrated --out runs/placement-replay
python -m benchmark.visible_depth_score --split native --candidate benchmark.visible_depth_score:integrated --out runs/placement-replay
python -m benchmark.visible_depth_sheet --baseline runs/visible-depth-rounds/baseline --candidate runs/placement-replay --out runs/placement.jpg
python -m benchmark.visible_depth_oracle --previous runs/visible-depth-rounds/baseline/test/cases.json --current runs/placement-replay/test/cases.json --out runs/test-oracle.json
```

The visibility ablation uses `engine.atlas_plane_depth:place_ranked_without_ray`. Run the ordinary CLI directly with `.\Invoke-WallRecovery.ps1 SCREENSHOT.png -Out runs/my-build`. A fresh Cloister high-right CLI run decoded the same 162 smooth-stone coordinates and IDs as the frozen-input replay, including 159 correct visible positions and three extras.
