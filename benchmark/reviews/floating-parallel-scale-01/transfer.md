# Parallel brick-scale transfer

The frozen rule at `8a80502` qualified one chart in 84 active conditions across 32 layouts. That chart was the previously reviewed Twinspire 2 earth low-left wall. It gained 19 exact position-and-material matches and lost seven, with zero movement of correct or extra positions. The rule made no change on another layout or any of the six fixed sky views.

The replay tested all 378 charts from the large-contraction receipts at `e97dd44`. One chart had all the required source evidence: a brick-rich perpendicular chart at its seam, a separate parallel brick-rich donor, and a recipient with fewer than 40% strong brick matches. That same chart reached the image texture test and passed. The low-left image's four-column candidate scored 0.504 atlas correlation, up from 0.090. Its 40 source chart cells supplied the same coordinates with new material IDs. At the end of ordinary assembly, its strict and facade placement F1 stayed at 57.44% and 60.54%, and its position-plus-ID F1 rose from 16.92% to 23.08% strict and 17.84% to 24.32% facade.

The 84 conditions use the active flags in the depth-refit receipt at `47d6e7c`. This comparison uses the archived large-contraction charts before independent depth patches so the whole-chart rule consumes the same handoff in both arms. Each changed chart proceeds through ordinary assembly and a fresh export. Conditions without a proposal reuse the archived export. Both arms are decoded, aligned with their recorded whole-build yaw and translation, and rescored on native visible positions, required facade positions, and exact material IDs. Source, chart, surface, pack and reference hashes are recorded per condition. [Condition evidence](transfer-cases.json.gz) · [Pooled scores](transfer-summary.json)

| Active cohort | Conditions tested | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---:|---:|---:|---:|
| All, large-contraction baseline | 84 | 33.542% / 34.015% | 5.512% / 5.612% | 18,323 / 61,187 / 11,420 |
| All, parallel scale | 84 | 33.542% / 34.015% | 5.534% / 5.635% | 18,323 / 61,187 / 11,420 |
| Visible Builds v2 | 30 | 42.831% / 45.697% | 17.020% / 18.469% → 17.244% / 18.713% | 2,300 / 5,441 / 699 |
| Visible Builds v3 | 30 of 144 | 34.100% / 34.243% | 3.897% / 3.903% | 13,974 / 47,475 / 6,535 |
| Decorative Builds v1 | 24 of 40 | 24.754% / 25.407% | 6.040% / 6.371% | 2,049 / 8,271 / 4,186 |

**Reviewed Twinspire inputs.** The earlier high-right and low-left experiments ran on the `5bd7908` charts in a paired replay outside the aggregate. The high-right archived large-contraction packet already has the 143-cell receiver and 31 strong brick matches on its adjoining front wall. Its parallel donor has zero strong brick matches. The reviewed patch packet has 21 on that same donor and qualifies. On the reviewed high-right packet the rule recovered 48 exact IDs, preserved all 110 known correct positions, and removed all 63 extra coordinates. Strict and facade placement rose from 47.11% and 49.22% to 54.46% and 57.29%. Strict and facade position-plus-ID rose from 17.99% and 18.79% to 44.55% and 46.88%. The reviewed low-left packet reproduced the same decoded export and scores as the large-contraction low-left replay. [Reviewed pair](README.md)

Reproduce from this checkout using the benchmark Python environment and archived local handoffs.

```powershell
python -m benchmark.floating_parallel_scale_transfer --workers 6
```

Keep the rule experimental. The observed geometry improvement needs the reviewed high-right packet, and the broad replay found no qualifying transfer to another building. The separate fixed-104 production trend carries forward at 91.27% strict and 93.42% facade placement.

## Floating branch scorecard

Six fixed sky conditions comprise four Visible Builds v3 and two Decorative Builds v1 images. Connected-atlas integration runtime `52de2c9` was replayed at `c0accfd`. Its measurements are carried forward below.

| Method | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---:|---:|---:|
| Integration connected-atlas, carried forward | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Large-contraction baseline | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |
| Frozen parallel scale | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |

On the six fixed sky views the candidate moved zero correct and zero extra blocks and tied every F1 score. The evidence is on `experiment/floating-twinspire-parallel-scale`. No behavior landed on the integration branch.
