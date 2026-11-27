# Camera model controls for Java 26.3

The frozen comparison covers 104 fixed, 144 Visible Builds v3 and 40 Decorative Builds v1 views. The ordinary runtime revision is `1ca88221bf2a29d4495b61a783b0c99273ee0634`. The floor-repeat control applies the prior camera fallback on the ordinary camera's 118 abstentions and carries forward the 170 supported exports. It recovered 40 cameras.

For both controls, the verifier hashes the source RGB and recorded reference files. It decodes all 288 ordinary ZIPs and all 40 changed repeat ZIPs, verifies the published scores, and records separate SHA-256 fingerprints for the local block ID inventories, visible truth and facade exemptions. The fixed control verifies that the saved translation achieves maximum overlap, then preserves the archived tie choice for material scoring. Fixed material assignment counts come from the released archive. [Frame IDs and splits](frames.json) · [Fingerprints](manifest.json) · [Decoded inventory checks](inventory.json.gz)

| Cohort | Control | Placement F1 strict / facade | Position + ID F1 strict / facade | Strict correct / missed / extra | Camera failures |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed 104 | Both | 91.27% / 93.42% | 84.68% / 88.51% | 20,773 / 2,929 / 1,046 | 0 |
| Visible Builds v3 | Ordinary | 8.08% / 8.51% | 6.16% / 6.50% | 9,299 / 204,851 / 6,859 | 108 |
| Visible Builds v3 | Floor repeat | 18.99% / 19.88% | 12.49% / 13.12% | 23,798 / 190,352 / 12,644 | 78 |
| Decorative Builds v1 | Ordinary | 54.33% / 56.11% | 43.43% / 45.34% | 7,120 / 9,806 / 2,162 | 10 |
| Decorative Builds v1 | Floor repeat | 63.46% / 65.23% | 50.51% / 52.56% | 9,240 / 7,686 / 2,957 | 0 |

One integer XYZ translation maximizes exact position overlap for the entire build. Material scoring uses that translation. Every visible target counts under strict scoring. Facade priority exempts top-only or underside-only cells lacking a 4×4 square of visibility on a native face. Both metrics charge all extra cells. Camera failures remain in the denominator and camera-supported empty exports have their own tally. [Scores by cohort and split](baseline-summary.json) · [Per-view scores](baseline-cases.json.gz)

Run `python -m benchmark.camera_model_controls freeze` from this checkout with the frozen source and prior run worktrees present. `python -m benchmark.camera_model_entry_check` compares the ordinary image-only CLI against sampled archived exports using verified cached model fields by default. `python -m benchmark.camera_model_grade` scores candidate folders, retains the selected control for unexported views, and records completed, pending and inherited cases separately. Only grading uses the recorded poses and visibility labels.

## Independent camera-model comparison

All four policies were independently rescored on the full 288-frame denominator, with the selected control carried forward for views outside the policy or without a new camera. The fixed 104 stayed at 91.27% strict and 93.42% facade placement for every arm. The three policies that start with floor repeat inherited the same Decorative results. MoGe after ordinary supported 33 of 40 Decorative cameras, with 55.55% strict and 57.20% facade placement and 43.77% strict and 45.64% facade position + ID.

| Visible Builds v3 policy | Scored | Camera failures | Placement F1 strict / facade | Position + ID F1 strict / facade | Strict correct / missed / extra |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoGe after ordinary | 144/144 | 98 | 10.15% / 10.67% | 6.45% / 6.78% | 12,164 / 201,986 / 13,299 |
| MoGe after repeat | 144/144 | 73 | 20.21% / 21.14% | 12.54% / 13.15% | 25,943 / 188,207 / 16,613 |
| GeoCalib after repeat | 144/144 | 68 | 22.36% / 23.25% | 14.59% / 15.24% | 28,857 / 185,293 / 15,069 |
| Matched boundary search after repeat | 144/144 | 70 | 21.84% / 22.75% | 14.13% / 14.79% | 28,086 / 186,064 / 14,949 |

The matched boundary search recovered the same three Harbor validation views as GeoCalib. Its accepted cameras projected all 13,754 native visible corners within two pixels and located 4,255 correct blocks at the registered coordinates, with 4,288 after translation. GeoCalib recovered two additional palettes of one training Shrine view. Both searches left the same Castle views unresolved. The MoGe repeat-first arm projected four of 14,128 corners within two pixels. It placed 668 correct positions at the registered coordinates and 2,145 after translation. On the single newly accepted held-out Castle case, median native reprojection error reached 481.66 pixels and none of the 478 translated correct positions was correct at the registered location. [Cohort and split metrics, exact export IDs and camera validation](experimental-comparison.json) · [Per-view grades](experimental-cases.json.gz)

The production candidate at `dcdee56` reproduced all 118 fallback camera and evidence records. It returned 48 usable cameras. The normal CLI used cached source-only detector fields and ran placement through export without replacing the camera decision. It matched archived projection, decoded block IDs, positions and grade on both sampled new Harbor recoveries, one ordinary-supported Pillars, one repeat-recovered Castle and one unresolved Castle. The earlier ordinary and repeat-port revisions each passed four sampled CLI checks. [Budget CLI check](budget-cli-check.json) · [Ordinary CLI check](ordinary-cli-check.json) · [Repeat CLI check](repeat-cli-check.json)

The candidate default score uses 48 decoded camera-handoff replays and 240 verified retained packs. The revised per-view records preserve the ordinary camera and atlas reasons alongside the latest fallback reason. [Candidate default scores](independent-matched-summary.json) · [Candidate default per-view rows](independent-matched-cases.json.gz)

## Joint camera and grid verification

The joint source gate accepted none of the 70 remaining v3 abstentions and preserved all 288 input-policy pack hashes. Independent grading decoded all eight diagnostic exports on the train Viaduct and test Castle and verified the final placement projection and consumed wall levels. The grid-only replay used the raw MoGe camera and expressed the refined wall levels in that camera's horizontal coordinate system. Whole-build-translated strict placement F1 on corrected grid-only was 31.30% for Viaduct and 15.38% for Castle.

The reference-only chooser offered the existing export on every frame. It selected from existing raw candidates on the 70 residuals and widened that choice on two frames with four joint exports. The raw choices reached 23.18% strict and 24.10% facade v3 placement F1. Adding the coupled exports moved this to 23.32% and 24.25%, with 14.67% strict and 15.32% facade position + ID. Both newly selected joint exports had zero camera-registered correct positions. The camera-only and grid-only ablations received separate grades. [Recomputed source-policy and chooser scores](independent-joint-check.json)
