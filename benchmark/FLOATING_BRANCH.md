# Floating-build reconstruction

`experiment/floating-visible-builds` is the long-lived integration branch for reconstructing visible blocks with no ground reference. One translation can place the entire build anywhere. All visible components retain their relative spacing. Production `main` at the fork was `e3bd220`.

Use the [current Floating pipeline map](FLOATING_PIPELINE.md) for stage order, source evidence, and the automated loss report.

## Frozen expanded benchmark

Use the **expanded split-border benchmark** for new floating reconstruction experiments. It pins the selected expanded package at `ac69f22` and the source depth split plus projected border-sky outputs at `4d9aca1`. The selected package uses the guarded replacement ruler and O2 corner repair on top of the large-contraction charts. Compare candidates on the same source images and visible references. Report distinct correct positions gained and lost, exact material matches, and extras separately alongside strict and facade placement F1. Prioritize retaining correct visible placements when judging an apparent improvement from removing extras.

| Benchmark | Strict / facade placement F1 | Strict / facade position + material F1 | Strict correct / missed / extra | Strict precision |
|---|---:|---:|---:|---:|
| Baseline — frozen expanded split-border, 84 conditions | 33.99% / 34.48% | 5.66% / 5.76% | 18,318 / 61,192 / 9,965 | 64.77% |
| Baseline — frozen expanded split-border, six fixed sky views | 27.59% / 27.83% | 5.98% / 6.01% | 890 / 3,151 / 1,521 | 36.91% |

The frozen 84 contain 30 Visible Builds v2, 30 Visible Builds v3, and 24 Decorative Builds v1 conditions. The six fixed sky conditions are included and use four Visible Builds v3 and two Decorative Builds v1 layouts. The full 240-view active cohort remains unevaluated for this benchmark. On the 84 paired inputs, the guarded ruler and O2 package removed 1,426 extras and lost five strict correct positions, four required by the facade score. The subsequent depth and border-sky checks removed a further 29 extras while preserving every correct coordinate and material ID. Twenty-one of those removals were on the six fixed sky views. The pinned exports are the research comparison. The sparse connected-atlas exports are a separate runtime control.

Restore the [exact chart and export archive](reviews/floating-expanded-benchmark-01/exports.zip) with the [frozen manifest](reviews/floating-expanded-benchmark-01/manifest.json). It contains the chart packet, decoded coordinate record, score receipt, and original Minecraft pack bytes for each panel and frame. The manifest gives every archive member's SHA-256, the source image and native reference hashes, and the source revisions used to construct the baseline. Source and reference paths are relative to the reconstruction repository with the recorded captures restored. The archive can be checked and unpacked without the experiment worktrees.

```powershell
$review = Join-Path 'benchmark/reviews' 'floating-expanded-benchmark-01'
$archive = Join-Path $review 'exports.zip'
$manifest = Get-Content (Join-Path $review 'manifest.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $manifest.archive.sha256) {
    throw 'Frozen expanded benchmark archive checksum mismatch'
}
Expand-Archive -LiteralPath $archive -DestinationPath 'runs/frozen-expanded-benchmark-01'
```

## Depth separation before the grid

Find visible surfaces and check which pixels share a plane before estimating a unit block or fitting a grid. Use source normals to identify direction changes. For same-facing surfaces, look for a repeatable signed depth step across a photographed boundary, exposed return, rim or opening. Fit each sufficiently supported plane independently. Preserve those plane boundaries when attaching grids, and leave unsighted regions unknown.

Inspect existing grids that span more than one visible plane first. Grade the source mask and plane assignment against recorded frontmost faces, then grade the decoded export. Keep the frozen split-border benchmark as the comparison. Its pink check groups cells on an inherited grid and can reduce extras, so it does not measure the benefit of giving each plane its own ruler and grid origin.

The local reprojection test at `fb08477` is retired from active Floating development. Its one-block shifts ran after grid fitting. The stored report is historical evidence. Use photographed depth boundaries before grid fitting for future depth work.

Start with the [wrong-plane audit and five source overlays](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/23c9d53/benchmark/reviews/floating-pregrid-plane-audit-01/README.md). The [Stable Yard eave replay](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/edcc59f/benchmark/reviews/floating-pregrid-eave-attachment-01/README.md) shows the downstream value of attaching a separately fitted visible plane at the correct offset. The [visible-return tests](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/e8326b9/benchmark/reviews/floating-pregrid-visible-returns-01/README.md) show where the source line and model depth fields still fail to distinguish parallel planes.

The leading source-only experiment [fits separate rulers and requires corroborated joins](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/experiment/floating-pregrid-depth-refit/benchmark/reviews/floating-pregrid-depth-refit-01/README.md). On the frozen 84 it gained 211 correct placements, lost 28, and held the extra count at 9,965. It kept all six sky exports unchanged. Keep the frozen benchmark as the reproducible comparison and the guarded refit as a candidate pending validation on new qualifying joins.

## Sparse connected-atlas runtime

The connected-atlas baseline transfers shared rows across supported corners, registers native atlas matches to the grid, and proposes grids on additional visible wall patches where neighboring atlas observations agree. It scores **12.64% strict and 12.90% facade placement F1** on six open-sky composites. Strict/facade position-plus-material F1 is **4.68% / 4.77%**. It places 273 correct blocks, misses 3,768, and adds five extras. The preceding registered-seam baseline placed 239 correctly with five extras. The gains came from Walled Garden and Harbor. [Garden experiments and source overlays](reviews/garden-feedback-01/README.md)

The default sky check rejected 23 of 216 proposals across six views, removing 81,896 background pixels from the combined surface masks. All 193 retained supports and their source crops matched their prior versions. The six decoded placement exports matched in coordinates and materials. [Filtered pipeline replay](reviews/sky-mask-feedback-01/filtered-pipeline-replay.json.gz)

### Reproduce the six exports

From this checkout, using the benchmark Python environment and the saved open-sky fixtures, source masks/counts, and dense atlas chart handoffs on this PC:

```powershell
python -m benchmark.floating_build_local_run ../../minecraft-image-reconstruction/runs/floating-build-grid-01/fixtures.json --sky-handoffs-only --include-decorative
python -m benchmark.shared_corner_rows
python -m benchmark.native_surface_counter
python -m benchmark.native_corner_registration
python -m benchmark.garden_anchor_extension --arm connected --workers 3
python -m benchmark.relative_face_round connected-atlas
```

The floating Surface Finder refines a proposal, then tests its native support against a sky mask built from the source image. Each row requires agreeing blue samples at both side borders. Sky and cloud-colored pixels must connect back to a border. A proposal is rejected only when every supported pixel qualifies. Mixed proposals retain their full support. The gate runs before outlines, crops and counts. Each build manifest records its rejected proposal IDs. The sky-handoff command uses the frozen normal and depth fields to reproduce the six inputs. The following stages consume outputs within this checkout. Decoded and scored exports are under `runs/floating-cloud-gate-01/relative-face-assembly/connected-atlas/sky`. This is the benchmark entry point for the floating branch; the ordinary build CLI still runs the production-style pipeline.

## Which work belongs here

| Branch history included | Use |
|---|---|
| Floating local grids, source charts and relative assembly/scoring | Inputs, visible-grid assembly, single-translation evaluation |
| Relative face junction control | Reference-only estimate of how well exact visible grids join |
| Shared corner rows and surface crop first | Previous source-only registered-seam baseline; weaker variants retained for comparisons |
| Sky mask feedback, line leveling and the surface inspector | Experiments and review tools. The whole-mask sky check now runs in the floating Surface Finder. Line leveling helped several surfaces and worsened others. Neither improvement is included in the baseline score as a new placement gain. |

The `experiment/floating-build-floor-bypass` worktree remains a historical diagnostic of the older placer, including supplied-camera interventions. Future work should start here and use sibling worktrees for parallel candidates. Keep the current inspector and annotation exports available at their original paths so existing feedback links keep working.

## Next annotations

Prioritize a second building with a visible corner that the assembler failed to join, a normal-looking crop where straightening breaks the texture rows, and a similar-looking wall where the current counter is already right. Note the surface IDs, the edge or rows you followed, and whether the two surfaces share the full height or only part of it. Mark a block count only when you can actually see it. Cloud examples from real skies, especially beside pale blocks, would extend the synthetic-sky check. Keep development annotations separate from a fresh held-out building used to judge promotion.

The capture-manifest runner selects the [200 Visible Builds views in Floating Combined V1](../training/FLOATING_DATASETS.md) by default. Set `--include-decorative` for an explicit Decorative Builds V1 test. Request the fixed sky fixtures explicitly for their paired sky comparison. Orchid is retired from Floating runs, including when replaying other layouts with `--historical-cohort`.

For saved runs, use the [stage report and Perfect Stage viewer](FLOATING_PIPELINE.md#perfect-from-this-stage) to compare end-to-end results from reference handoffs. Use the [visual review queue](FLOATING_PIPELINE.md#generate-the-visual-review-queue) to find source-image examples of gained and lost placements.
