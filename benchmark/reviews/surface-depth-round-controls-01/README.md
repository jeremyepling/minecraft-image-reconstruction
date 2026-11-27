# Surface-depth pilot controls

The Java 26.3 pilot uses the promoted ordinary output at `e3bd220` for 13 frozen screenshots. The audit read each actual source image and native visible reference, decoded each hash-matched datapack, and recalculated geometry and exact block-ID scores. All pilot scores matched the promoted release. The other cohort values below are carried forward from the independently graded release. Reaggregating its 288 case rows reproduced every published cohort and split score.

| Cohort | Tested / available | Placement F1 strict / facade | Placement + ID F1 strict / facade | Strict correct / missed / extra |
| --- | ---: | ---: | ---: | ---: |
| Fixed 104, carried forward | 104/104 | 91.27% / 93.42% | 84.68% / 88.51% | 20,773 / 2,929 / 1,046 |
| Visible Builds v3, carried forward | 144/144 | 46.63% / 48.04% | 28.04% / 29.08% | 77,774 / 136,376 / 41,629 |
| Decorative Builds v1, carried forward | 40/40 | 63.46% / 65.23% | 50.51% / 52.56% | 9,240 / 7,686 / 2,957 |
| Pilot fixed-suite regression views | 2/104 | 87.02% / 92.52% | 85.46% / 90.70% | 446 / 101 / 32 |
| Pilot Visible Builds v3 | 9/144 | 51.91% / 55.26% | 31.21% / 34.04% | 5,356 / 7,343 / 2,581 |
| Pilot Decorative Builds v1 | 2/40 | 64.82% / 65.53% | 56.79% / 56.99% | 444 / 324 / 158 |
| Pilot, pooled across its selected views | 13/13 | 54.24% / 57.48% | 35.14% / 37.87% | 6,246 / 7,768 / 2,771 |

All pilot cameras are usable and all pilot packs contain blocks. The full release has seven camera abstentions, all in Visible Builds v3, accounting for 26,122 strict visible misses. The 137 camera-supported Visible Builds v3 views score 50.60% strict placement F1. The machine-readable receipt preserves the per-split scores, camera outcomes, and assigned and fallback material counts. Harbor and Castle are regression families inspected in earlier rounds.

Each expanded pilot row identifies the exact output manifest, surfaces, counts, placements and pack. It also identifies the original source-only field cache and any camera fallback receipt. The four boundary-budget and one horizon-repeat recoveries carry their failed texel-atlas reason from the original ordinary manifest. The fixed-suite rows point to the typed packs and their separately verified geometry pack ancestors. Their recorded placement status comes from the historical geometry run. Their source-only replay starts from the saved pre-inset handoff. Reference pose and visible labels are used for scoring.

Run `python -m benchmark.surface_depth_control_receipt` from the controls worktree with the saved release and fixture checkouts available. The audit runs no model inference and keeps Minecraft closed. [Control receipt](controls.json) · [Promoted model-free release](../camera-native-recovery-controls-01/README.md)
