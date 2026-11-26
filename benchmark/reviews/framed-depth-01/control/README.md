# Whole-structure depth control

The active default at `acfe46b` uses the placement code scored at `36d406a`. The control reopened each archived Minecraft datapack, checked the screenshot and measured camera inputs, registered its decoded coordinates to the recorded camera, and recomputed reference visibility and relative F1. For the dataset splits it used the admitted v2 frames. The other 56 image hashes matched the independently audited exterior list. The eight texture-only frames in the historical archive were excluded at input selection.

| Cohort | Frames | Visible positions | Correct | Extras | Extras on hidden structure | Relative F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 72 | 18,322 | 15,742 | 1,591 | 89 | 88.30% |
| Fixed structure evaluation | 104 | 23,702 | 20,203 | 1,894 | 110 | 88.22% |
| Alternate views of known shapes | 20 | 4,098 | 3,524 | 508 | 57 | 86.69% |
| Previously reserved structures | 4 | 747 | 618 | 128 | 0 | 82.79% |

Canopy 06 left recovered 29 of 84 visible blocks and placed 92 extras. Its export measures 34 × 6 × 62 along X, Y and Z, compared with 6 × 6 × 10 for the visible target. All 92 extras fall outside the horizontal bounding rectangle of the entire source build after scoring alignment. Across the fixed evaluation, 1,084 of 1,894 extras fall outside that rectangle. Perfectly removing just those extras would yield 90.36% pooled F1. The measured rectangle comes from scoring truth and is unavailable at prediction time.

Ten fixed canopy probes recover 294 of 459 visible roof cells at the correct offsets from both front posts. All ten opening widths match and all 137 sampled clear rays remain open. The eight alternate canopy probes recover 232 of 398 anchored roof cells and all eight widths. All 106 sampled rays remain open. [Cohorts and probes](summary.json) · [Per-frame scores, exported-pack hashes, extents and source fingerprints](cases.json)

## Replay

Restore the v2 dataset, the archived placement inputs and the native confirmation fixture on this PC using the [previous round's replay instructions](../../nearby-thin-01/README.md) and [four-structure-view fixture](../../blind-shapes-01/README.md). Run the control from the repository root:

```powershell
python -m benchmark.framed_depth_eval --out runs/framed-depth-control --workers 3
```

Use `--candidate-root <checkout> --candidate module:function` to test a placement refinement. It receives the screenshot, observed surface masks, measured counts, estimated camera and prior local placements. The callable returns local positions or a pair of positions and diagnostic information. Each changed result is exported and decoded before scoring. For an unchanged result, the scorer reuses the exact baseline pack. `--predictions <completed-run>` independently reopens previously exported packs and verifies their input and pack hashes. Select subsets with `--splits train` or `--ids`. The four views in `new` were reserved during the previous round and have already been evaluated.
