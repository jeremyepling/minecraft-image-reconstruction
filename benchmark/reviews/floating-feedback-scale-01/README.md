# Comparing a whole face with the old atlas grid

Keep the atlas comparison experimental. It found the scale Jeremy preferred on Riverside Mill high-right and Sawtooth. On the 81 conditions from layouts outside the feedback set, its only changed export lost 54 correct Shrine positions while removing 64 extras.

The chooser reads the accepted source-image quad and its neighboring sprite matches from the archived face finder. It maps the quad into the old chart, requiring an area of at least two old cells and adjacent full-sprite correlation of at least 0.70. Five points across the proposed quad select old cells for direct native RGB verification. The source face must beat the strongest measured old cell by 0.08 correlation. Only existing cells with at least 95% recorded mask coverage and on-screen corners can vote. A missing proposal keeps the old grid. The rule was frozen against the annotated layouts before scoring the other layouts. Every substitution difference went through the ordinary horizontal assembler and export decoder.

| Annotated wall | Preferred | Rule | Measured proposed/old area | Source evidence |
|---|---|---|---:|---|
| Riverside Mill high-right | B | B | 16.25 | Full face 0.990, adjacent 0.981, strongest old cell 0.740 |
| Riverside Mill low-left | A | A | 1.13 | Little scale difference |
| Sawtooth | B | B | 9.34 | Full face 0.972, adjacent 0.973, strongest old cell 0.693 |
| Twinspire low-left | A | A | 0.93 | Little scale difference |
| Hillside Chapel | B | A | 9.57 | Adjacent 0.697 fell just below 0.70 despite a 0.905 full-face match |
| Amphitheater | Neither | A | — | The finder had no accepted full-face proposal. The sparse alternative displayed as B was a diagnostic. |

Four of the five A/B preferences agree with the chooser. Amphitheater still lacks a usable ruler. The test changes neither the surface masks nor the corner phase.

| Conditions | Method | Correct / missed / extra | Strict / facade F1 | Strict / facade position + ID F1 |
|---|---|---|---|---|
| All 92 | Prior large contraction | 21,240 / 72,356 / 17,374 | 32.13 / 32.67% | 5.10 / 5.20% |
| All 92 | Atlas comparison | 21,210 / 72,386 / 18,556 | 31.81 / 32.33% | 5.07 / 5.17% |
| All 92 | Prior gate or atlas comparison | 21,199 / 72,397 / 17,183 | 32.13 / 32.67% | 5.15 / 5.26% |
| Reviewed 11 | Prior large contraction | 1,109 / 2,858 / 1,875 | 31.91 / 32.74% | 12.46 / 12.78% |
| Reviewed 11 | Atlas comparison | 1,133 / 2,834 / 3,121 | 27.56 / 28.17% | 11.02 / 11.26% |
| Reviewed 11 | Prior gate or atlas comparison | 1,122 / 2,845 / 1,748 | 32.82 / 33.69% | 13.84 / 14.20% |
| Other 81 archived | Prior large contraction | 20,131 / 69,498 / 15,499 | 32.14 / 32.67% | 4.69 / 4.78% |
| Other 81 archived | Either atlas policy | 20,077 / 69,552 / 15,435 | 32.09 / 32.61% | 4.68 / 4.77% |

The standalone chooser gained 24 correct positions and lost 54, adding 1,410 extras and removing 228. At Chapel it restored 1,373 net extras. At Twinspire earth high-right it gained 13 correct and removed 127 net extras. At Shrine, the proposed face covered 15.99 old cells with 0.767 full-face and 0.750 adjacent correlation. None of five old tiles passed full-sprite verification, yet replacing the old 124-cell chart left just six cells. Local evidence for the ruler did not protect the export from losing coverage. Retaining every prior gate decision isolates the added atlas choices and yields 13 gained, 54 lost, 36 extras added, 227 removed. These archives include 30 Visible Builds v2, 38 Visible Builds v3 (37 unique of 144), and 24 Decorative Builds v1 conditions (23 unique of 40). The historical fixed-104 production result carries forward unchanged at 91.27% strict and 93.42% facade.

Run `python -m benchmark.floating_feedback_scale` from this checkout. The runner loads the prior harness from the sibling `floating-extra-isolation` worktree at `e97dd44`. Archive receipts include source, offer, chart and export hashes. [Per-condition decisions and decoded placement](conditions.json.gz) · [Full paired counts](summary.json)

## Floating branch scorecard

Carried-forward connected-atlas integration result for the six sky composites, as reported on the integration branch at `c0accfd`. No integration behavior landed from this challenger.

| Method | Strict / facade F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---|---|---|
| Connected atlas | 12.64 / 12.90% | 4.68 / 4.77% | 273 / 3,768 / 5 |
| Prior large contraction | 27.50 / 27.73% | 5.96 / 5.99% | 890 / 3,151 / 1,542 |
| Atlas comparison | 22.93 / 23.07% | 4.40 / 4.40% | 901 / 3,140 / 2,915 |
| Prior gate or atlas comparison | 27.50 / 27.73% | 5.96 / 5.99% | 890 / 3,151 / 1,542 |

Compared with the running connected-atlas integration export, the standalone chooser gained 628 net correct positions and added 2,910 net extras for +10.29 strict F1 points. The additive control tied the prior challenger with 617 additional correct positions and 1,537 additional extras versus integration.
