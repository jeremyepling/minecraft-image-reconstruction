# Native camera recovery and wall-depth experiments

The ordinary default adds a model-free native-pixel atlas fit after the previous camera methods abstain. It recovers 63 of the 70 remaining Visible Builds v3 cameras. Strict placement F1 rises from 21.84% to 46.63%; facade placement rises from 22.75% to 48.04%. The fixed 104 and decorative exports stay unchanged. The camera code was frozen and tested on `experiment/camera-native-floor-inverse` at `aac7da7`, then integrated on `integration/camera-native-defaults` for `main`.

## Placement and materials

| Cohort | Scored / available | Strict / facade placement F1 | Strict / facade position + ID F1 | Correct / missed / extra positions | Camera failures |
|---|---:|---:|---:|---:|---:|
| Fixed trend, unchanged | 104 / 104 | 91.27 / 93.42 | 84.68 / 88.51 | 20,773 / 2,929 / 1,046 | 0 |
| Visible Builds v3, previous default | 144 / 144 | 21.84 / 22.75 | 14.13 / 14.79 | 28,086 / 186,064 / 14,949 | 70 |
| Visible Builds v3, new default | 144 / 144 | 46.63 / 48.04 | 28.04 / 29.08 | 77,774 / 136,376 / 41,629 | 7 |
| Decorative Builds v1, unchanged | 40 / 40 | 63.46 / 65.23 | 50.51 / 52.56 | 9,240 / 7,686 / 2,957 | 0 |

Scores are percentages. Each image permits one integer XYZ translation for the entire build. The material score uses that same translation. Only visible targets count; unsupported added cells are errors. Decorative block-state properties beyond the ID remain outside these metrics.

The new fallback adds **49,688 correct positions and 26,680 extras**. It preserves the packs for all 218 previously supported images. Exact ID accuracy on correctly positioned v3 blocks is 60.14% (46,771 of 77,774). New camera-supported views add many harder surface and material cases; the older exports and their correct IDs are unchanged. Of the seven remaining failures, two are training Pillars views and five are Castle views. Failed cameras remain in the overall denominator.

These totals combine 63 new model-free exports with the verified previous outputs. The independent verifier checked all 288 source images, reference inventories, facade exemptions and baseline packs, and decoded the new exports. The additional GeoCalib composition required two more exported builds. Original building-family splits are preserved; Harbor and Castle have been inspected in earlier rounds and are regression families, not new blind tests. [Independent scores and split results](../camera-native-recovery-controls-01/README.md) · [Interactive scorecard](../pipeline-impact-01/viewer.html#cohorts)

## What changed

The camera fitter predicts how the known floor texture would look at a candidate viewing angle, scale, phase and filtering resolution, then compares that prediction directly with source pixels. This preserved weak texture evidence that previous straightening and matching did not use successfully. The new fallback is currently calibrated to the vanilla Java 26.3 light-gray concrete floor texture. Four separated floor regions must agree, with explicit competing-scale checks. Existing camera successes return before this method runs.

The ordinary path uses the existing atlas and local OpenCV, including the configured recovery interpreter when necessary. It downloads no new model. Missing or broken resources leave an explicit unresolved camera result. [Camera flow and terms](../../../ARCHITECTURE.md#camera-recovery-and-its-fallbacks) · [Frozen method, source negatives and provenance](../camera-native-floor-inverse-01/README.md)

All 77,126 withheld visible reference corners on the 63 new cameras project within two native pixels after fitting one integer origin on the other corners. This is a calibration diagnostic, not a placement score. The new review also corrects the earlier camera diagnosis: some very large raw projection errors mixed arbitrary world-origin displacement with internal shape error. The earlier MoGe and joint camera/grid candidates still fail the stricter relative calibration check. [Origin-invariant audit and known-camera controls](../camera-recovery-next-01/README.md#a-fairer-check-of-relative-camera-geometry)

## What stays experimental

| Branch | Result | Decision |
|---|---|---|
| `experiment/camera-native-floor-inverse`, GeoCalib-seeded composition | Two more accurate cameras, including one additional Castle view. v3 strict / facade placement 47.33 / 48.75; position + ID 28.47 / 29.52. Adds 1,798 correct positions, 921 extras and 1,104 correct IDs beyond the model-free default. | Keep the learned seed experimental. Its real benefit needs a production inference path and broader layout/material validation; the shipped fallback needs no new model setup. |
| `experiment/camera-silhouette-ruler` | No source-accepted recovery among 70 failures. Verified training steps were one block high, while verified Castle steps were two blocks high. Forced candidates were inaccurate. | Keep the branch as evidence; stop treating a visible step as a unit-block ruler without an independent scale cue. |
| `experiment/camera-native-fit-confidence` | A family-grouped, source-feature confidence model adds one training view, with 323 correct positions and 551 extras. It adds no Harbor or Castle recoveries. | Keep experimental; no demonstrated transfer beyond the training family. |
| `experiment/wall-color-partition` | Source-only component depths raise Amphitheater training strict F1 from 52.85 to 69.82, but the ungated rule regresses Shrine and both variants swap many correct and wrong Harbor positions. Four checked fixed views tie. | Keep experimental. Separate genuine depth boundaries from coplanar material changes before enabling per-component depth changes. |

Benchmark code and evidence are retained alongside the default; those experimental inference policies are not enabled in ordinary builds. [Silhouette evidence](../camera-silhouette-ruler-01/README.md) · [Wall partition exports and controls](../wall-color-partition-01/README.md) · [Camera research and confidence decisions](../camera-recovery-next-01/README.md)

## Where to focus next

Camera coverage is **95.14%**, while strict placement is **46.63%**. The seven failures account for 26,122 visible misses. Supported-camera images still have 110,254 misses and 41,629 extras, and their strict placement F1 is 50.60%. This separates unresolved cameras from downstream placement failures.

The main geometry priority is surface ownership, face direction and plane depth on accurately calibrated views. Harbor demonstrates that one detected wall mask can combine several distinct depths. Trace lost or extra cells to their first incorrect mask/axis/depth choice, and use independently visible junctions or depth evidence before relocating a component. The wall-partition diagnostic shows substantial opportunity, but a global ranking-margin gate does not reliably decide where to split.

Continue the smaller camera track with a source-pixel appearance diagnostic. On the five views still rejected by the combined model-free/GeoCalib experiment, the selected floor samples contain no labeled foreground structure. Even the exact floor chart fails the current correlation gate on Castle 03. A simple blend of mip resolutions still fails. Test spatially varying texture footprint and lighting with phase-discontinuity negatives before training another chooser or weakening the acceptance threshold. [Source-image and decoded-export examples, remaining-camera probes](../camera-recovery-next-01/README.md)

## Validation

The integrated runtime passed 471 unit tests. All 70 ordinary camera decisions exactly match the frozen model-free matrices or abstentions. Five cached-field CLI builds cover earlier texel, original repeat, wider repeat, new native and unresolved paths. Two additional Windows launcher builds used live pinned vision inference and reproduced the expected decoded blocks and IDs. Production negatives cover missing/broken resources, absent filtering dependencies, flat/noisy/striped images, a textured vertical wall, inconsistent tile phase, and material mismatches. The independent scorer rechecked the ordinary-entry receipts and source fingerprints. [Validation manifest](validation.json) · [Independent production audit](../camera-native-recovery-controls-01/independent-production-check.json)
