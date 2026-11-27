# Whole-block texture anchors and partial-block occupancy

Keep `experiment/full-block-lattice` experimental. The frozen rule recovered a useful scale on some floor-camera failures. On Decorative Builds it correctly grouped **zero of the 1,630 photographed partial blocks from two distinct visible subfaces**. Applying the rule to already populated builds introduced 4,302 extras on the fixed suite. This branch contains benchmark code and evidence. The ordinary reconstruction on `main` remains at its measured settings.

## What happened to a slab

On the held-out Walled Garden view, the top of the red sandstone slab carried a complete vanilla `red_sandstone_top` sprite. The 16×16 check gave it a 0.956 correlation. It correctly identified a one-block **face width**. That face also belongs to a half-height block. The adjacent vertical fragment was measured at Z = 3.625 and landed in the next occupied cell. The stored source masks preserved both photographed surfaces. The containment variant protected the texture anchor and failed to join the two pieces. [Source image and enlarged fragments](partial-offset.jpg) · [Exact coordinates and state](partial-example.json)

## Paired exported builds

All 288 ordinary control ZIP hashes, geometry, exact-ID scores and geometry-selected translations were verified against the shared baseline at `1ca88221bf2a29d4495b61a783b0c99273ee0634`. Every alternative is an actual Java 26.3 ID-only datapack, decoded with the same strict and facade scorer. Each frame gets one whole-build integer translation. Each percentage below is pooled F1. [Per-frame results](cases.json.gz) · [Scores including all splits](summary.json) · [Frame and atlas hashes](manifest.json)

| Dataset | Policy | Strict position | Facade position | Strict position + ID | Facade position + ID |
|---|---|---:|---:|---:|---:|
| Fixed 104/104 | Ordinary | 91.27 | 93.42 | 84.68 | 88.51 |
| | Frozen lattice | 83.51 | 84.59 | 77.40 | 80.07 |
| | Rescue only | 91.27 | 93.42 | 84.68 | 88.51 |
| | Reference chooser | 91.28 | 93.44 | 84.70 | 88.53 |
| Visible Builds v3 144/144 | Ordinary | 8.08 | 8.51 | 6.16 | 6.50 |
| | Frozen lattice | 8.28 | 8.70 | 6.15 | 6.49 |
| | Rescue only | 8.31 | 8.73 | 6.17 | 6.52 |
| | Reference chooser | 8.33 | 8.75 | 6.17 | 6.52 |
| Decorative Builds v1 40/40 | Ordinary | 54.33 | 56.11 | 43.43 | 45.34 |
| | Frozen lattice | 53.94 | 55.57 | 41.67 | 43.42 |
| | Rescue only | 54.50 | 56.27 | 43.46 | 45.37 |
| | Reference chooser | 54.57 | 56.33 | 42.79 | 44.66 |

The frozen method used the stronger source verification developed on Market Court before the full comparison. The rescue-only policy was evaluated separately after inspecting the full results. It accepts the same selected lattice only if the ordinary camera rejects the image. The reference chooser selects between retaining the original export and the actual ungated proposal per image using facade F1, keeping the original on ties. Strict-first results are under `strict_oracle`. Selecting by geometry can lower exact-ID F1.

| Cohort | Ordinary correct / missed / extra | Frozen lattice correct / missed / extra | Recovered / lost | Extras added / removed |
|---|---|---|---|---|
| Fixed | 20,773 / 2,929 / 1,046 | 20,826 / 2,876 / 5,348 | 53 / 0 | 4,302 / 0 |
| Visible v3 | 9,299 / 204,851 / 6,859 | 9,606 / 204,544 / 8,189 | 307 / 0 | 1,330 / 0 |
| Decorative | 7,120 / 9,806 / 2,162 | 7,287 / 9,639 / 2,806 | 274 / 107 | 745 / 101 |

The source method retains ordinary local coordinates. Whole-build alignment can change when additions change its best translation. This explains the 107 decorative losses. Relative partial placement rose from 430 to 458 of 1,630, with 49 positions gained and 21 lost. Exact partial IDs stayed at zero. The experiment recorded directions and eighth-block plane offsets of observed fragments. The exported ID-only pack uses default states and full-cube substitutes for unresolved partial blocks.

## Camera, extent and visible evidence

When the ordinary floor camera exists, the method verifies full sprite correspondences on estimated planes by comparing native RGB to the calibrated atlas and testing both halves along both texture axes. When the floor fails, the method estimates three orthogonal directions and focal length from saved source line segments, searches full sprites on independently rectified surface masks across scales and phases, and votes for a common fractional XYZ origin. Consensus uses corners of different sprite matches, their optical-center distances and projected directions. Multiple axes are required to modify a populated build. Two matching anchors can support a restricted fallback on one plane. No fitted weights, recorded cameras, partial-shape names, other views or native depth enter source prediction. A full sprite is provisional plane and scale evidence. [Method](../../full_block_lattice.py)

Each checked anchor proposes its surface level and occupied coordinate. Perpendicular neighboring masks can meet the anchor at a measured level on an eighth-block grid. Only supported rays near that photographed junction contribute, with at least four samples per neighboring cell. Original Surface Finder masks remain the input. The grouping output stores its sampled footprints at four-pixel stride and pointers to the original surface IDs. It omits unsupported cells and retains measured subface offsets separately from the ID-only export. Fixed photos use checked historical pre-inset regions along with archived camera and ranked-plane offsets. Expanded photos use the actual retained ordinary placement input.

Of 108 v3 rejected cameras, 53 lacked three line directions, 15 produced no verified sprite, nine had only unconfirmed evidence, 27 passed with repeated same-axis anchors and four passed with multiple axes. On the ten decorative rejects, five lacked directions, one had unconfirmed evidence and four passed with repeated same-axis anchors. One development Market Court case returned ten correct positions, one extra and zero exact IDs after the relative alignment. [Floor-rejected source with green anchors and orange support samples](floor-rejection.jpg) · [Decoded candidate datapack](market-court-source.zip)

| Rescue-only comparison | V3 | Decorative |
|---|---:|---:|
| Newly nonempty / previously rejected | 31 / 108 | 4 / 10 |
| New correct / extra / lost positions | 278 / 52 / 0 | 34 / 10 / 0 |
| Newly correct exact IDs / partial positions | 28 / 0 | 14 / 6 |
| Median worst-axis direction error | 1.57° | 12.71° |
| Median estimated / true focal length | 1.04 | 1.69 |
| Median estimated / true range to a matching visible cube face | 1.08 (115 anchors) | 1.56 (22 anchors) |
| Median absolute error in spans between identified anchors | 0.20 blocks (23 axis spans) | 1.00 blocks (7 axis spans) |

The span test measures orientation-registered differences between observed anchors at least 1.5 blocks apart, independently of any whole-build translation. It measures the extent of identified pieces, not the unseen structure. The relative scorer can slide sparse pieces to another repeated location. None of the 115 v3 or 22 decorative audited fallback anchors landed on its own identified source owner under the geometry-selected translation. Using one reference shift per image fit 111/115 v3 and 15/22 decorative anchor-to-owner correspondences. This limits what the positional gains establish. [Camera and range checks](camera-and-span.json) · [Coordinate-level attribution](coordinates.json.gz)

On held-out v3 test layouts, five exports added 23 correct positions, six extras and three exact IDs. The v3 validation split had no rescue. On decorative validation, Hillside Chapel contributed ten correct, two extra and six exact IDs. Both positions and IDs tied on all eight decorative test views, where the ordinary cameras already worked. All six relative partial-position gains from the decorative fallback were on development views. Two fell at directly matched sprite cells and four at neighboring, independently observed fragments. No rescued partial position had a demonstrated join between distinct observed subfaces. In the broader frozen decorative method, ten newly added cells supported by multiple surfaces were all extras under the final geometry-selected translation. [Location and shape breakdown](attribution.json)

## Counter handoff and genuine partial-block joins

The ordinary counter emitted 147 complete-face observations in decorative views. Their majority native owners covered 57 distinct truth blocks and zero partial blocks. By extracting new sprite and neighboring subface fragments the lattice proposed 2,468 observations, of which 1,963 had at least 65% of samples on a single native owner. Those fragments touched 1,600 distinct target cells, including 128 partial cells. On the eight held-out Walled Garden views, 45 of 352 partial targets had such evidence. These counts measure the available observations before the export gate.

With each fragment separate, there were 363 excess pieces assigned to already represented truth blocks. Grouping the same fragments by inferred XYZ produced 2,443 groups, lowered over-segmentation to 356, correctly joined seven pairs of distinct photographed faces on full cubes, and joined six pairs belonging to different truth blocks. Partial over-segmentation stayed at 90. All eight test views together had five correct full-face joins and one neighbor over-merge. Native face identity distinguishes two faces of the same partial block from two observations of its same face. [Counter before and after](grouping.json)

Only eight partial targets across all 40 images yielded two different frontmost faces from **different source masks** in these proposals. Six paired fragments landed one predicted cell apart, two landed farther apart. Neither the original grouping nor the second source-only containment rule joined any of the eight. The second rule tested whether one projected cube could contain both adjoining, perpendicular subface footprints. It made one additional join between different blocks, removed one extra from the decoded build, and kept all eight held-out builds identical to the first variant. This exploratory replay followed inspection of the main results. [Containment replay](containment.json)

The ordinary Material Matcher then consumed the 35 rescued occupied-cell sets with their source images and estimated cameras. Re-exporting left their coordinates identical and raised rescue-only strict position-plus-ID F1 on v3 from 6.17% to 6.21%. Decorative strict position-plus-ID F1 stayed at 43.46%. The original segmentation was reused throughout. The source experiment does not establish improved continuous masks, openings at native resolution, or complete structure extent.

## Known-lattice positive controls

On the ten rejected decorative views, a reference camera with the image-derived masks and sprite proposals gave seven nonempty packs, 316 correct positions, 52 extras and 184 exact IDs. Geometry strict/facade F1 was 13.76% / 14.30% on these ten. This intervention supplies the true camera, snaps a source-estimated plane to its grid and searches ±2 plane levels. It tests the consumer with improved registration and plane verification together. The frame-scored keep-or-propose oracle above is a separate experiment.

Supplying the correct camera, signed plane of each **visible** subface and native foreground pixel, then calculating the inward cell and decoding the export, mapped all 27,374 decorative subfaces to their correct 16,926 positions. All 1,630 partial positions were recovered, 1,622 from multiple visible subfaces. This reference geometry was never available to prediction. The same calculation correctly assigned 249,650 v3 faces to 214,150 occupied positions. [Positive controls](positive-control.json)

## Reproduction

Use the calibrated Java 26.3 atlas and frozen control worktree beside this checkout on the documented Windows workspace. The runner requires NumPy, Pillow and OpenCV. On this PC use the existing ScaleLSD environment. These scripts use two CPU workers and saved sources. Generated files are private to `runs/full-block-lattice-01`, and report assets are under this review directory.

```powershell
$py = '../mask-scalelsd/runs/env/Scripts/python.exe'
& $py -m benchmark.full_block_lattice_run --workers 2
& $py -m benchmark.full_block_lattice_run --summary
& $py -m benchmark.full_block_lattice_diagnostic --workers 2
& $py -m benchmark.full_block_lattice_diagnostic --summary
& $py -m benchmark.full_block_lattice_attribution --workers 2
& $py -m benchmark.full_block_lattice_groups --workers 2
& $py -m benchmark.full_block_lattice_containment --workers 2
& $py -m benchmark.full_block_lattice_camera
& $py -m benchmark.full_block_lattice_sheet
```
