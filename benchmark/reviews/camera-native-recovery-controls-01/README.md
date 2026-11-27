# Independent native-floor camera scores

This review pairs the previous Java 26.3 default at `62611ee` with two native-floor policies on the same 288 screenshots. The model-free default tests native floor evidence after the existing camera fits abstain. The composed experiment uses GeoCalib to propose an alternative starting point after the model-free fit abstains. Both policies use the same native evidence gate.

The verifier checked every frozen RGB file, visible block ID inventory, facade exemption set and incumbent export. It decoded and scored each new datapack against the same reference as its incumbent and checked the accepted image-only projection at placement. Geometry selects one whole-build integer XYZ translation, which also applies to geometry-plus-block-ID scores. The case files include the camera source, pack fingerprint, decoded block fingerprint and placement scores, with source evidence for every native-floor decision.

The fallback was evaluated on 70 earlier camera failures, all in Visible Builds v3. It recovered 47 of 49 training views, all five validation views and 11 of 16 test views under the model-free policy. The model-free failures retained their empty exports. The composed experiment recovered two additional views, Pillars 03 marble high-right in training and Castle 02 limestone high-right in test. [Model-free cases](independent-default-cases.json.gz), [composed cases](independent-composed-cases.json.gz)

| Cohort and policy | Tested | Geometry F1 strict / facade | Geometry + ID F1 strict / facade | Strict correct / missed / extra |
| --- | ---: | ---: | ---: | ---: |
| Fixed historical cohort, carried forward | 104/104 | 91.27% / 93.42% | 84.68% / 88.51% | 20,773 / 2,929 / 1,046 |
| Visible Builds v3, incumbent | 144/144 | 21.84% / 22.75% | 14.13% / 14.79% | 28,086 / 186,064 / 14,949 |
| Visible Builds v3, model-free | 144/144 | 46.63% / 48.04% | 28.04% / 29.08% | 77,774 / 136,376 / 41,629 |
| Visible Builds v3, composed | 144/144 | 47.33% / 48.75% | 28.47% / 29.52% | 79,572 / 134,578 / 42,550 |
| Decorative Builds v1, carried forward | 40/40 | 63.46% / 65.23% | 50.51% / 52.56% | 9,240 / 7,686 / 2,957 |

The model-free policy added 49,688 correct positions and 26,680 extra positions on Visible Builds v3. Strict geometry F1 increased 22.11 percentage points on training, 12.93 on validation and 37.97 on test. The composed experiment added 1,798 correct positions and 921 extras beyond the model-free result. Promote the model-free policy for its transfer to held-out buildings and keep the GeoCalib composition experimental for the two additional recoveries. [Model-free totals and split deltas](independent-default-summary.json), [composed totals and split deltas](independent-composed-summary.json)

The image evidence gate compares an atlas sprite with four separated floor patches and requires a correlation of at least 0.60 in each patch and a margin of at least 0.26 above the best alternate metric scale. The camera evaluation fits one integer origin on half the distinct visible corners and measures reprojection on the other half. All 77,126 checked model-free corners and all 80,158 composed corners fell within two pixels. The worst frame-level P90 was 0.89 px with that fitted origin and 22.33 px with ordinary camera registration. The largest frame-level median error in relative ground spans was 0.036%.

Castle 01 limestone high-right recovered 1,198 correct positions, with 273 missed and 147 extra. Its strict placement F1 was 85.09% and held-corner P90 was 0.14 px. Harbor 03 seafoam high-right had a held-corner P90 of 0.10 px and strict placement F1 of 17.23%, with 218 correct, 1,704 missed and 391 extra positions. Both views previously had empty exports.

The production check verified all 70 model-free camera decisions against the frozen experiment and checked five cached-field CLI exports and two real CLI exports with live pinned model fields. The former exercise the incumbent atlas floor, both repeat camera policies, a new native recovery and a model-free abstention. The production source controls accepted the original floor example and rejected seven variants covering flat color, noise, stripes, a vertical wall, inconsistent texture phase and a mismatched material. [Production check](independent-production-check.json)

The silhouette ruler used the same 70 residual views and abstained on all of them. Its 288 policy exports match the incumbent. The 16 forced diagnostic exports comprise 12 ground-pose trials and four supplied-reference or unverified-mask trials. The evaluator chose ground-pose trial results with access to placement truth. The measured source policy recovered zero cameras, so keep the silhouette ruler experimental. [Silhouette check](independent-silhouette-check.json)

To reproduce the audit with the saved releases and completed experiment exports, run `python -m benchmark.camera_native_recovery_grade`. The scorer reads the frozen control and experiment checkouts, writes its own cache and review files, and keeps Minecraft closed.
