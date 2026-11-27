# Expanded Java 26.3 baseline

The expanded controls run the ordinary reconstruction at `1ca88221bf2a29d4495b61a783b0c99273ee0634` on the 144 Visible Builds v3 images and 40 Decorative Builds v1 images. Each build receives one frozen source screenshot and Java 26.3. The fixed 104 retain the released geometry and material results. The scorer verifies the released cases and actual datapack payloads and decodes all 104 packs.

The scorer verifies the source and reference checksums and opens the capture pose and native visible labels after export. It registers block centers to the captured orientation and finds one integer XYZ translation that maximizes position overlap for the entire build. It grades exact IDs using that same shift. Strict grades every block with visible evidence in the source. Facade priority exempts top-only or underside-only locations with a native visible face narrower than a 4×4 patch, and still charges all extra placements. Camera-rejected empty builds count as misses for every required target. Camera-registered scores record the result before the relative translation.

Decorative targets retain both full-cube and partial-model block coordinates. The exported identifiers are default-state block IDs. The shape breakdown reports positional recovery and exact IDs per truth category. The separate grader checks all 184 reference cases with perfect packs and verifies the atlas's decorative ID coverage. [Scoring validation](score-validation.json)

The eight previously reported decorative sources reproduce all local coordinates and registry IDs. Their canonical typed pack hashes, placed counts, strict and facade correct positions, and exact ID counts agree with the prior release. The no-op replay on Market Court 2, cool, low-left reproduces 225 decoded placements from the retained model fields. [Prior decorative comparison](../material-seam-confirmation-01/README.md)

## Reproduce

In the checkout, run the local launcher with the ordinary Java 26.3 atlas and pinned model files installed. The launcher uses seven builder workers and limits concurrent GPU model processes to three on this PC. Successful build receipts and model fields from interrupted jobs can be reused after verifying the frozen source manifest and production revision.

```powershell
.\benchmark\Invoke-ExpandedBaseline.ps1
$python = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $python -m benchmark.expanded_scores --workers 4 --output benchmark/reviews/expanded-baseline-01
```

Each build folder under `runs/expanded-baseline-01/<dataset>/<frame-id>/` contains its manifest, measured surfaces, counted grids, placements and decoded-pack input. Its `handoffs` folder also retains the native source detector fields, the actual ranked placement input and output, and the final material input. `benchmark.expanded_baseline.replay_source_fields` supplies verified fields to the ordinary source-only builder. `benchmark.expanded_scores.evaluate_pack` applies the same grader to a new pack on that source. `--partial` writes an interim score snapshot with the missing frame IDs and any technical failures recorded so far.
