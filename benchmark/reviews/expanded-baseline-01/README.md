# Expanded Java 26.3 baseline

The expanded controls run the ordinary reconstruction at `1ca88221bf2a29d4495b61a783b0c99273ee0634` on the 144 Visible Builds v3 images and 40 Decorative Builds v1 images. Each build receives one frozen source screenshot and Java 26.3. The fixed 104 retain the released geometry and material results. The scorer verifies the released cases and actual datapack payloads and decodes all 104 packs. An independent rerun of the material release verifier also regraded the 204 archived packs, including these 104, against the original reference cache. [Released material controls](../typed-structures-01/README.md)

The scorer verifies the source and reference checksums and opens the capture pose and native visible labels after export. It registers block centers to the captured orientation and finds one integer XYZ translation that maximizes position overlap for the entire build. It grades exact IDs using that same shift. Strict grades every block with visible evidence in the source. Facade priority exempts top-only or underside-only locations without a 4×4 native-pixel square on a visible face and charges all extra placements. Camera-rejected empty builds count as misses for every required target. Camera-registered scores record the result before the relative translation.

Decorative targets retain both full-cube and partial-model block coordinates. The exported identifiers are default-state block IDs. The shape breakdown reports positional recovery and exact IDs per truth category. The separate grader checks all 184 reference cases with perfect packs and verifies the atlas's decorative ID coverage. [Scoring validation](score-validation.json)

The eight previously reported decorative sources reproduce all local coordinates and registry IDs. Their canonical typed pack hashes, placed counts, strict and facade correct positions, and exact ID counts agree with the prior release. The no-op replay on Market Court 2, cool, low-left reproduces 225 decoded placements from the retained model fields. [Prior decorative comparison](../material-seam-confirmation-01/README.md)

## Full-cohort results

| Cohort | Scored / available | Placement F1 strict / facade | Correct / missed / extra positions strict | Position + ID F1 strict / facade | Exact IDs at correct positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 104, released | 104 / 104 | 91.27% / 93.42% | 20,773 / 2,929 / 1,046 | 84.68% / 88.51% | 92.78% |
| Visible Builds v3, ordinary | 144 / 144 | 8.08% / 8.51% | 9,299 / 204,851 / 6,859 | 6.16% / 6.50% | 76.23% |
| Decorative Builds v1, ordinary | 40 / 40 | 54.33% / 56.11% | 7,120 / 9,806 / 2,162 | 43.43% / 45.34% | 79.93% |

The fixed trend line carries forward the verified `1ca8822` placement with a 0.00-point delta. The older 112-view strict placement score was 90.50%. All 184 new builds completed and exported a datapack, with no technical failures. [Pooled metrics and splits](summary.json) · [Per-view scores and camera diagnostics](cases.json) · [Build and dataset fingerprints](manifest.json) · [Eight prior decorative results](decorative-confirmation.json)

| Cohort and split | Scored | Unusable cameras | Placement F1 strict / facade | Position + ID F1 strict / facade |
| --- | ---: | ---: | ---: | ---: |
| Visible Builds v3 train | 108 | 82 | 9.78% / 10.30% | 7.77% / 8.21% |
| Visible Builds v3 validation | 18 | 8 | 16.01% / 19.95% | 9.61% / 11.99% |
| Visible Builds v3 test | 18 | 18 | 0.00% / 0.00% | 0.00% / 0.00% |
| Decorative Builds v1 train | 24 | 6 | 55.45% / 57.42% | 45.05% / 47.14% |
| Decorative Builds v1 validation | 8 | 4 | 46.02% / 47.57% | 41.10% / 42.51% |
| Decorative Builds v1 test | 8 | 0 | 57.46% / 59.10% | 40.95% / 42.80% |

Visible Builds v3 had 108 unusable cameras and 187,656 strict misses on those views. The seam fit reported insufficient repeated seam centers on 107 failing views. The atlas fallback lacked texel-frequency tracks on 85. All 18 Castle test views failed camera fitting. The 36 camera-supported views had 43.60% strict and 45.50% facade placement F1. Across the full cohort, camera registration before relative translation had 6.63% strict and 6.95% facade placement F1.

Decorative Builds v1 had 10 unusable cameras and 4,226 strict misses on those views. The atlas fallback reported incomplete floor texel coverage on six of the ten. The 30 camera-supported views had 64.78% strict and 67.01% facade placement F1. The ordinary exporter located 6,690 of 15,296 visible full cubes and named 5,691 of those correctly. It located 430 of 1,630 visible partial-model coordinates and named zero correctly. All 1,630 partial-model positions were required by the facade scorer. [Atlas coverage and reference target audit](score-validation.json)

## Reproduce

In the checkout, run the local launcher with the ordinary Java 26.3 atlas and pinned model files installed. The launcher uses seven builder workers and limits concurrent GPU model processes to three on this PC. Successful build receipts and model fields from interrupted jobs can be reused after verifying the frozen source manifest and production revision.

```powershell
.\benchmark\Invoke-ExpandedBaseline.ps1
$python = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $python -m benchmark.expanded_scores --workers 4 --output benchmark/reviews/expanded-baseline-01
```

Each build folder under `runs/expanded-baseline-01/<dataset>/<frame-id>/` contains its manifest, measured surfaces, counted grids, placements and decoded-pack input. Its `handoffs` folder also retains the native source detector fields, the actual ranked placement input and output, and the final material input. `benchmark.expanded_baseline.replay_source_fields` supplies verified fields to the ordinary source-only builder. `benchmark.expanded_scores.evaluate_pack` applies the same grader to a new pack on that source. `--partial` writes an interim score snapshot with the missing frame IDs and any technical failures recorded so far.
