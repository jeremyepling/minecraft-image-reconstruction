# Source-only replacement rulers from neighboring wall heights

A direct atlas-window search found a more plausible ruler for Stable Yard 1 low-left. Using it in the ordinary assembler reduced extras from 1,374 to 53 and retained 70 of 78 correct placements. It produced zero correct position-and-ID pairs, down from 26. The local fit failed the distant validation rule. The high-right fit chose a smaller pearlescent froglight pattern and left 234 extras. Keep both methods experimental.

[Projected grids and decoded per-view scores](replacement-rulers.jpg) · [Edge-first decisions and scores](conditions.json.gz) · [Atlas-first decisions and scores](atlas-first/conditions.json.gz)

## Source experiment

Start with the frozen large-contraction wall chart and the earlier source-only contradiction flags on 84 active conditions. Three qualify. On these views two or more independent wall masks project block heights at least twice the disputed wall's estimate. Other masks can be at different depths. Their projected pitches bound a search without fixing the correct target-plane pitch, phase or perspective. Candidate functions consume only source RGB, its predicted mask and chart, and peer ratios. The fixed 84 conditions include six sky composites. All three intervention cases had been reviewed before this experiment.

The edge-first search scans both axes over 0.75 times the smallest through 1.25 times the largest peer ratio, with 24 phases per pitch and four perspective strengths about the mask center. It ranks local RGB contrast at candidate grid lines against nearby quarter-pitch controls in the left 40% of mask pixel centers sorted by source X. The right 40% tests the same phase without adjustment. The middle 20% separates the measurements. This partition is held out from fitting the refit parameters. The pre-existing segmentation, starting chart and trigger already used the whole source. Candidate homographies score independently on both axes. A held-out effect of at least 0.06 and a margin of 0.015 over both half and double pitch are required on each axis. Both partitions must return a whole-sprite match with phase advantage. Sparse line support or failed sprite checks cause abstention. The rule was frozen at `81e6a9e` before scoring.

The atlas-first variant scans actual image windows within the fitting partition. It rectifies using the baseline perspective, searches nine heights across the same peer bracket and permits widths 0.75, 1 or 1.25 times each height in the original chart units. A palette shortlist from nine fitting-region subdivisions limits the texture search. It checks all eight orientations of six chosen opaque full-cube sprites. Native mask coverage, correlation and contrast with quarter-shifted patches test the best candidate windows. The best locally verified fit must survive the same independent right-side seam, harmonic and texture tests. A maximum of 32 geometrically sampled wall cells per partition are tested for independent sprite evidence. The direct search was committed at `fcbe9d0` before its own scores.

For every returned homography, rebuild the original visible mask's cells and materials, rerun joins and component selection, export the ordinary pack, decode it and apply the existing whole-build registration. The bypass reapplies the unchanged chart on all three triggered conditions and matches the decoded baseline exactly. The remaining 81 controls reuse their identical archived packs. The verified arm keeps the old chart when validation fails. The relaxed arm forces the best proposed ruler through export for diagnosis. The two experiments share their source and target fingerprints.

## What the builds contain

| Condition | Whole chart | Edge-first forced | Atlas-first forced | Reference ruler on original mask |
|---|---|---|---|---|
| Stable Yard 1 low-left | 78 / 294 / 1,374 | 30 / 342 / 7 | **70 / 302 / 53** | 76 / 296 / 51 |
| Stable Yard 1 high-right | 63 / 291 / 1,198 | 38 / 316 / 49 | **55 / 299 / 234** | 55 / 299 / 54 |
| Pillars sky low-left | 69 / 314 / 687 | 45 / 338 / 64 | 45 / 338 / 64 | Unmeasured |

Counts are correct / missed / extra. On Pillars the ordinary assembler switched the selected group from mask-1 to mask-4 for both forced methods. Its changed placements come from selecting a different wall. Both Stable Yard exports retained mask-0. The reference comparator comes from the independent exact-ruler diagnostic and was never offered to either search.

- Stable Yard low-left edge-first selected 4.846 × 5.246 old cells per unit face and removed the initial perspective. Its held-out V seam advantage was −0.123. Atlas-first selected a cut-sandstone match at 2.875 × 3.875 old cells with 0.734 correlation. The distant U and V effects were +0.163 and +0.271. The vertical pitch lost its harmonic comparison by 0.113, and no geometrically sampled distant full face passed.
- Stable Yard high-right edge-first selected 3.692 × 3.592 and retained the original perspective. Its distant V effect was +0.603, but U reached only +0.004 and lost its harmonic test. The atlas-first window matched pearlescent froglight with 0.687 correlation at 1.75 × 2.25 old cells. Both views still lack a far full-face match at the selected phase. The high-right baseline contained 16 correct IDs at correct positions, compared with 11 edge-first and four atlas-first.
- Pillars found a locally convincing bone-block window, correlation 0.864 at 5 × 5 cells. Its far region yielded no corroborating full face. The edge-first U axis had only two sufficiently covered distant lines.

| Active 84 | Strict | Facade | Strict + ID | Facade + ID | Correct / missed / extra |
|---|---:|---:|---:|---:|---|
| Frozen baseline, bypass and both verified arms | 33.54% | 34.01% | 5.51% | 5.61% | 18,323 / 61,187 / 11,420 |
| Edge-first forced | 34.38% | 34.91% | 5.62% | 5.74% | 18,226 / 61,284 / 8,281 |
| Atlas-first forced | 34.40% | 34.91% | 5.59% | 5.71% | 18,283 / 61,227 / 8,512 |

The edge-first diagnostic gained 65 correct positions, lost 162, added 60 extras and removed 3,199. Atlas-first gained 35 correct, lost 75, added 142 extras and removed 3,050. A reference-only selector over both forced outputs and the baseline reaches 34.43% strict, with 18,266 correct / 61,244 missed / 8,327 extra. This ceiling covers the offered exports only. No automatic replacement cleared the source checks. Independent tile support across both axes and final material assignment remain unresolved.

The atlas-first forced output on 30/144 Visible Builds v2 conditions scored 42.83% strict, 45.70% facade, 17.02% strict + ID and 18.47% facade + ID. The 30/144 Visible Builds v3 conditions scored 34.31%, 34.47%, 3.93% and 3.94%. Decorative Builds v1 on 24/40 conditions scored 28.53%, 29.58%, 6.48% and 6.96%. [Edge-first cohort counts](summary.json) · [Atlas-first cohort counts](atlas-first/summary.json)

Using the benchmark Python with saved fixtures available, run these commands in order for each method.

```powershell
python -m benchmark.floating_peer_ruler_refit --workers 5
python -m benchmark.floating_peer_ruler_refit --report
python -m benchmark.floating_peer_atlas_search
python -m benchmark.floating_peer_atlas_search --report
python -m benchmark.floating_peer_refit_sheet
python -m unittest tests.test_floating_peer_ruler_refit tests.test_relative_face_assembly
```

The separate fixed-104 production result is carried forward at 91.27% strict and 93.42% facade, with 20,773 correct / 2,929 missed / 1,046 extra.

## Floating branch scorecard

The carried-forward six-sky integration at `c0accfd`, runtime `52de2c9`, scored 12.64% strict, 12.90% facade, 4.68% strict + ID and 4.77% facade + ID, with 273 correct / 3,768 missed / 5 extra. On the same six, the baseline and verified replacements score 27.50%, 27.73%, 5.96% and 5.99%, with 890 / 3,151 / 1,542. The actual refit gate changed zero positions or extras from the expanded baseline. The forced variants score 29.73%, 30.14%, 6.59% and 6.69%, with 866 / 3,175 / 919. They lose 24 net correct and remove 623 net extras against the expanded baseline, or gain 593 correct and add 914 extras against integration. Only the experimental runner and evidence landed on `experiment/floating-peer-ruler-refit`. Integration retains connected-atlas.
