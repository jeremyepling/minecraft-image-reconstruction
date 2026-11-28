# Distant wall seams as a ruler check

On the 86 previously measured image conditions, using distant seams to choose a wall ruler changed strict placement F1 from 31.47% to 31.48%. It removed 258 net extras and lost 37 net correct positions. It accepted 31 of 183 offered wall rulers. The rule helped Sawtooth but missed the two large overexpansions on Chapel and Riverside Mill. Keep the distant-seam and combined rules experimental.

[Projected rulers over three source images](distant-grids.jpg) · [Decoded comparisons by panel and dataset](scores.md) · [Source, reference and pack fingerprints](manifest.json.gz)

## What was measured

The previous experiment found a proposed unit face using local texture, edge and adjacent-tile agreement. This experiment compares the old and proposed projective wall grids **away from their origin tiles**. Both candidates must use the same connected source-mask component. A sample must be at least 2.5 units in maximum-axis distance from each origin center in that ruler's own grid. One native pixel of erosion prevents bilinear samples from seeing beyond the support. The finder had already searched the screenshot globally, so the distant area is held out of the selected face's local verification, not the original proposal search.

At each projected unit seam, compare native RGB differences across the seam with four interior lines offset by 0.19 and 0.31 units on each side. Each test uses local projective spacing. Its two samples lie 0.5–1.25 pixels from the line. Each segment needs four of seven usable sites and a projected pitch of at least five pixels. An edge between different materials can support a boundary, and clipping the contrast limits its influence. Switch when both rulers have at least four measured segments, the proposed mean normalized advantage exceeds 0.08, and it beats the previous grid by over 0.05. The rule keeps the old grid when measurements are sparse. The evaluator never supplies a second face, known integer offsets, camera pose or native ownership. The earlier local patch veto instead used depth or seed-neighborhood evidence. Like the ordinary assembler, this test trusts the connected source mask, which can span different coplanar-looking surfaces. Each ruler samples its own supported grid segments within that common region.

The ruler score and gates were committed in `7d439a0` before opening the new six. The OR combination was frozen in `096c6b7`. It also accepts a chart with a source-verified replacement when the previous chart contains at least 256 cells and the replacement contains at most a quarter as many. The contraction threshold was designed after seeing earlier failures. The 86 are regression conditions. The six additional views are three previously unmeasured layouts from familiar families, two views each.

## Where the choices differed

- On Sawtooth 2 high-right the proposed ruler followed visible masonry joints. The seam advantage rose from −0.069 to +0.973. Its export gained 61 correct positions, lost 60 and removed 223 extras. Strict F1 rose 18.67 points.
- On Watchpost 2 cool low-left the gate switched two charts. The export lost 40 correct positions while removing 25 extras and strict F1 fell 11.58 points. Twinspire 2 cool low-left also fell 11.72 points.
- On Chapel the new ruler had a verified local stone-brick face but the wider chart included calcite, wood, tower and roof regions. The previous and proposed grids scored +0.001 across 2,687 distant segments and −0.029 across 297. The seam gate kept the old ruler. Contraction recovered the useful alternative.
- In the new six the seam gate accepted two of 14 offers, both from Arcbridge 1 high-right. Across all six, the export gained one net correct position, removed three net extras and raised strict position-plus-ID F1 from 11.22% to 11.66%. Applying every source-one proposal reached 18.80% strict geometry versus 18.40% for seams and 18.34% for the unchanged charts.

The reference-only chooser can select the best whole horizontal or source-one export for each view to maximize pooled geometry F1. It reaches 32.69% on the previous 86, 1.22 points above the horizontal baseline, and still misses 70,777 positions. It reaches 18.80% on the extra six. Its options exclude mixed per-chart exports. The source-only OR combination reaches 32.48% on the 86 versus 32.52% with contraction alone. Adding seam switches gains 0.06 points of strict position-plus-ID F1 but costs 38 net correct positions and removes just 35 net extras. The positive seam measurements cannot reliably establish final placement.

The historical 86 cover 26 Visible Builds v2, 36 v3 and 24 Decorative v1 frame conditions, including six separate sky composites. The extra six include four v2 and two v3. The full archives contain 144 v2, 144 v3 and 40 Decorative v1 views. Each panel JSON stores per-image strict and facade geometry and typed counts, assigned and fallback counts, atlas ID accuracy, and paired changed positions. The audit files record per-chart sample counts and margins. Saved source packs were decoded again, including empty exports, and all five method packs have verified SHA-256 fingerprints.

## Replay

Use the saved inputs in the sibling source-proposals and floating-extra-audit worktrees and the benchmark Python from mask-scalelsd.

```powershell
$panels = 'original','fresh','supplemental','sky','holdout','holdout2','holdout3','holdout4','holdout5'
python -m benchmark.floating_distant_seams @panels
python -m benchmark.floating_distant_seams_combined @panels
python -m benchmark.floating_distant_seams_report @panels
python -m benchmark.floating_distant_seams_sheet
```

Only the experiment branch contains these changes. Production fixed 104 carries forward 91.27% strict and 93.42% facade, with 20,773 correct, 2,929 missed and 1,046 extra. The floating integration default, connected atlas at `c0accfd`, carries forward 12.64% strict and 12.90% facade on the six sky composites. The horizontal candidate reaches 22.93% strict there, and the seam rule ties it. Contraction alone and the OR combination both reach 27.50%. The sky improvement over the horizontal candidate comes from contraction.
