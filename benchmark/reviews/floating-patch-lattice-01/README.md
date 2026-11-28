# Can independently measured wall patches detect a bad grid?

Keep this branch experimental. Local repeated edges detected that the chapel sky grid extends through regions with different spacing. Removing these cells also discarded correct geometry from both observatory walls and both previously unseen amphitheater views.

The frozen rule (3630fc5) partitions one source wall into 4×4 chart-coordinate regions. Its 8×8 centered windows may overlap. Each independently measures two line families on RGB pixels within the predicted surface mask. Each family has its own slope, pitch and phase in the perspective-rectified wall coordinates. The Fourier search tests pitches from 0.45 to 2.70 original cells, slopes from −0.4 to +0.4, and continuous phase. Alternating one-cell-wide strips supply independent fit and validation samples. A strong held-out seam signal that exceeds the original chart's signal and disagrees in pitch, slope or phase suppresses expansion in that region. Original accepted seed cells survive. Inconclusive windows retain the whole-wall cells. This test uses source edges and existing atlas seed cells. Repeated edges can come from masonry decoration, texture, or multiblock patterns.

The trim arm passes the remaining chart cells to the ordinary assembler. The rechart arm also offers a separate locally measured chart when both line families verify, limits its footprint to the same mask and region, and reruns the ordinary atlas material matcher. It does not supply a relative 3D origin to force a join. Both arms were also rebuilt with horizontal cells. Final scores use decoded packs after one whole-build translation. All four candidate arms use the same frozen rules for every view. The original 38, first 12, supplemental six and six composites were previously reviewed. The final six hold out three building families and both angles. Each image supplies its own measurements. Reference labels enter only during scoring.

Each table entry gives strict geometry, facade geometry, strict position-plus-ID and facade position-plus-ID F1 (%) in order.

|Panel|Views|Whole wall|Trim|Rechart|Horizontal baseline|Rechart + horizontal|
|---|---:|---|---|---|---|---|
|Original|38|35.34 / 35.55 / 4.51 / 4.53|21.82 / 22.01 / 3.34 / 3.37|21.91 / 22.10 / 3.41 / 3.44|35.64 / 35.82 / 4.63 / 4.65|22.22 / 22.41 / 3.53 / 3.57|
|First additional|12|27.25 / 28.39 / 5.63 / 6.04|20.10 / 21.70 / 4.75 / 5.16|20.10 / 21.70 / 4.75 / 5.16|30.43 / 31.11 / 6.85 / 7.36|23.90 / 24.97 / 6.16 / 6.70|
|Supplemental|6|39.92 / 43.08 / 9.27 / 10.00|32.93 / 35.90 / 7.65 / 8.34|32.93 / 35.90 / 7.65 / 8.34|39.92 / 43.08 / 9.27 / 10.00|32.93 / 35.90 / 7.65 / 8.34|
|Sky composites|6|22.93 / 23.07 / 4.40 / 4.40|22.24 / 22.58 / 4.11 / 4.17|22.24 / 22.58 / 4.11 / 4.17|22.93 / 23.07 / 4.40 / 4.40|22.24 / 22.58 / 4.11 / 4.17|
|New held-out|6|18.85 / 20.13 / 3.39 / 3.71|9.12 / 9.57 / 2.55 / 2.90|9.12 / 9.57 / 2.55 / 2.90|18.83 / 20.12 / 3.39 / 3.70|9.11 / 9.56 / 2.55 / 2.90|

On chapel sky, 53 of 121 regions disagreed. Among 60 vetoing single-axis measurements, 53 had pitch above 1.18 source chart cells. Median pitch was 1.70 cells. Correct/missed/extra went from 118/234/1,431 to 112/240/750. Nine alternate charts totaling 37 cells reached assembly and none entered its selected group. Strict geometry rose from 12.41% to 18.45%. The walled garden sky view gained 16 correct blocks and removed 21 extras. Pillars sky gained F1 while losing 25 correct blocks and removing 635 extras.

Observatory low-left went from 195/559/189 to 124/630/18, and high-right from 184/549/23 to 111/622/8. The high-right main chart's median vetoing pitch was 2.0 original cells. The rival seam signal can validate across strips on a useful grid. The two amphitheater views fell from 132 and 376 correct to 22 and 18. Stable Yard low-left removed four extras with all 81 correct retained. The other Stable Yard view and both Arcbridge views tied. Across the new six, rechart recovered 18 positions, lost 486, added 33 extras and removed 1,904. Its 39 independently offered charts did not change that cohort's exported positions.

Only the original panel benefited from offering local charts after trimming. There the extra charts raised strict F1 by 0.09 points over trim alone, with five improved, 28 regressed and five unchanged views against whole wall. The same cut cannot both control chapel overextension and preserve the observatory walls. This branch changes no default. The historical fixed 104-view cohort, all remaining v3 and Decorative v1 images, and lighting variants were not evaluated in this round.

[Decision regions and selected exports](patches.jpg). Each row shows source, sampled whole-wall cells (red identifies vetoed regions, green retained regions), and the final rechart export projection.

Per-image scores, placement changes, source and mask hashes, asset paths, and selected chart IDs are recorded for the [original 38](original.json.gz), [first 12](fresh.json.gz), [supplemental six](supplemental.json.gz), [sky six](sky.json.gz), and [new six](new.json.gz). Every panel contains separate dataset subtotals. Raw per-axis measurements are archived under measurements/ by panel and frame ID. Local output paths in the reports refer to the registered worktrees on the benchmark PC.

Replay with the benchmark Python from this checkout.

```powershell
python -m benchmark.floating_patch_lattice original fresh supplemental sky new --workers 6
python -m benchmark.floating_patch_report original fresh supplemental sky new
python -m benchmark.floating_patch_sheet
```
