# How much does a perfect surface crop improve floating placement?

Paired Java 26.3 evaluation on all 38 published test images from Visible Builds v2 (12), Visible Builds v3 (18), and Decorative Builds v1 (8). They show eight layouts from three families. These are the original rendered source images, including their original backgrounds. The floating solver uses a build-relative grid and scores one global integer translation. The historical six open-sky composites and production fixed 104 remain separate benchmarks.

The source arm runs the floating Surface Finder and its whole-cloud veto. The reference arm replaces the entire list of accepted surfaces before outlines, crops and counting. Each new surface has exactly the frontmost source pixels with its recorded surface ID. The packager groups edge-connected, coplanar unit faces and identifies individual decorative model quads. The runner verifies the binary support against the recorded raster at original resolution and checks total visible pixel area. It estimates each new normal and relative inverse depth from image-predicted fields on that mask. It fits measurement charts, infers block counts, transfers shared rows, assigns atlas materials, selects a connected component, and exports blocks with the same downstream code as the source arm. World positions, known materials, true face normals, correct block sizes, camera pose and hidden geometry are excluded from the reconstruction inputs. Recorded orientation may enter evaluation as in the preexisting floating benchmark.

The paired change measures the outcome of this particular handoff, including the effect of refitting directions and sampling charts to exact native masks. A perfect whole-image chooser is an upper bound only within the two measured exports per image. It maximizes pooled strict placement F1 by fractional optimization, then reports facade and material scores for that same selection. Substituting crops on individual surfaces or changing assembly would define a different experiment.

## Results

| Cohort | Views | Ordinary / exact masks, strict F1 | Exact-mask delta | Whole-image chooser |
|---|---:|---:|---:|---:|
| all | 38 | 3.72% / 8.47% | +4.76 pp | 8.87% |
| decorative-builds-v1 | 8 | 3.79% / 8.03% | +4.24 pp | 8.03% |
| visible-builds-v2 | 12 | 30.90% / 31.24% | +0.34 pp | 35.87% |
| visible-builds-v3 | 18 | 1.57% / 6.80% | +5.23 pp | 6.86% |

The exact masks recovered 1,505 additional correct positions and removed 33 extras across the paired totals. Per-image strict placement improved on 30 views, fell on six and tied on two. Of 4,242 exact native surface masks, 505 received both dimension estimates; the pipeline offered 310 charts to assembly and exported from 59 selected charts across 38 builds. The six-sky result used an earlier full-cube-only intervention and a different background, so its baseline and gain are reported separately. All 76 original-image exports passed hash and decoded-coordinate/material checks.

The reference-only experiment lives on `experiment/crop-ceiling-expanded`. Reproduce from that worktree with the benchmark environment, installed archive checkouts and frozen model assets.

```powershell
python -m benchmark.expanded_crop_ceiling --workers 5
python -m benchmark.expanded_crop_ceiling_score
python -m benchmark.expanded_crop_ceiling_sheet
```

Generating missing v2 fields requires ONNX Runtime compatible with the Python environment. This run reused the original normal and depth fields for v3 and decorative and generated the twelve v2 field pairs using the pinned models. Artifact paths use the historical floating solver's internal sky variant key even for original images; source digests identify the actual RGB.

[Decoded receipts and paired scores](scored-exports.json.gz) · [Six-view sky chooser](six-sky-chooser.json.gz) · [Mask comparison](crop-comparison.jpg)
