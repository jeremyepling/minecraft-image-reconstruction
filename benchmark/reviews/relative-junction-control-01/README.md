# Visible unit-face junction control

Branch `experiment/relative-face-junction-control` builds diagnostic inputs for the six frozen sky composites at base `e3bd220`. Each view has its own arbitrary origin. Each chart has independently rebased U and V coordinates, a unit lattice fitted to recorded image corners, a signed normal, and known materials. Chart IDs and cell ordering are shuffled. The solver receives no global XYZ, common depth, camera pose, ground plane or normal-axis offset.

Each chart contains a four-connected component of coplanar, front-visible full-cube faces. Four separate 16-bit masks per cell identify edge samples supported by native frontmost face pixels. Partial occlusion can hide corners whose theoretical pixel locations the fitted chart still supplies. Coplanarity, cell counts, image corners, orientation, observed coverage, and material identity all use reference answers. These charts measure the assembly opportunity when those answers are supplied together.

The audit independently compares the two physical unit-edge endpoints and requires overlapping owner-supported bins from both adjoining charts. The table counts distinct visible blocks in the largest physically connected chart group. All physical edges in the table meet a minimum projected length of four native pixels. Every disconnected group would need additional evidence to establish its position relative to the selected group.

| Sky composite | Charts | Full-cube targets | Largest group with one shared edge | Largest group with two shared edges |
| --- | ---: | ---: | ---: | ---: |
| Castle | 73 | 1,471 | 594 | 594 |
| Harbor | 80 | 311 | 74 | 74 |
| Viaduct | 113 | 1,091 | 1,071 | 980 |
| Pillars | 43 | 383 | 195 | 195 |
| Hillside chapel | 21 | 317 | 270 | 230 |
| Walled garden | 63 | 392 | 357 | 302 |
| Total | 393 | 3,965 | 2,561 | 2,375 |

The 4,791 observed full-cube faces represent 3,965 blocks. The chapel and garden have 76 additional visible non-full-cube blocks between them. Each source image and all three reference files match the frozen manifest. Per-corner maximum deviations are below 9e-10 pixels against the local chart and 5e-13 pixels against the recorded camera. Native face-ID counts agree with recorded face areas.

Keep this branch experimental. Castle has distinct physical junctions that almost coincide in the image. One incorrect translation of one block along X and one along Z repeats across three unit edges with only 0.90 pixels of mean endpoint error. Harbor also shows limited coverage when local junction geometry is perfect. Its largest supported group accounts for 74 of 311 full-cube blocks. Global assembly needs image evidence for offsets between groups, and the measured near-coincidences make local ambiguity a separate issue.

Run from the branch checkout with the repository's NumPy/Pillow Python environment. The second command accepts a checkout containing the relative face solver and audits its accepted translations against native references.

```powershell
python -m benchmark.relative_oracle_charts --fixtures ../../minecraft-image-reconstruction/runs/floating-build-grid-01/fixtures.json --output runs/relative-face-assembly-01
python -m benchmark.relative_oracle_connectivity --fixtures ../../minecraft-image-reconstruction/runs/floating-build-grid-01/fixtures.json --output runs/relative-face-assembly-01 --solver-checkout ../relative-face-assembly
```

The generated `oracle-manifest.json` records every chart-file checksum, frozen source hash and tolerance. The diagnostic `oracle-connectivity.json` records the solver file digest, per-case component counts, and each incorrectly accepted offset. Only the six `sky/<id>/charts.json` files feed the solver.
