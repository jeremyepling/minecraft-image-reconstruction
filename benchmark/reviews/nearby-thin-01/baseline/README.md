# Frozen geometry control

The scorer reopened all 204 datapacks from the two published `cebbffa` archives, checked their SHA-256 hashes, decoded and registered the local block coordinates, and compared them to visible cells in the corresponding screenshot. It verified the RGB and upstream inputs, reference camera and world files and geometry scores on every frame. Each build receives one shared XYZ translation. The ten fixed canopy probes and eight alternate views also measure roof coordinates relative to the two posts.

| Cohort | Visible targets | Correct | Extras | Extras on hidden blocks | Relative F1 |
|---|---:|---:|---:|---:|---:|
| Development, 72 | 18,322 | 15,076 | 1,554 | 81 | 86.27% |
| Fixed evaluation, 112 | 24,086 | 19,408 | 1,794 | 96 | 85.71% |
| Additional views, 20 | 4,098 | 3,173 | 474 | 56 | 81.94% |

In the fixed evaluation, 194 of 459 visible roof blocks align with both posts, all ten measured openings have the correct width, and all 137 sampled clear rays remain open. In the additional views, 131 of 398 roof blocks align, six of eight opening widths are correct, two widths are unresolved, and five of 106 sampled clear rays are blocked in the predicted builds. The additional views depict previously evaluated structures. [Per-cohort measurements](summary.json) · [Per-frame fingerprints, scores and probes](cases.json)

## Reproduce

Restore the fixtures listed in the [fixed-suite evaluation](../../visible-recovery-01/README.md) and the [additional-view evaluation](../../visible-recovery-extra-views-01/README.md). The following commands audit the archived control and score a new set of candidate datapacks. The optional `--splits` argument limits scoring to selected cohorts. Use `--extra-inputs` to point to a restored additional-view fixture. Transfer screenshots use the existing `MCIR_NATIVE_TRANSFER_FIXTURES` setting.

```powershell
python -m benchmark.nearby_thin_comparison --out runs/current-control
python -m benchmark.nearby_thin_comparison --predictions runs/new-placer --extra-predictions runs/new-placer/extra --out runs/new-comparison
```
