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
