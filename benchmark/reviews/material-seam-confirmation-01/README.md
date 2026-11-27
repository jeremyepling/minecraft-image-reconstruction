# Directional material seams on new structures

**Keep the seam matcher experimental.** It added 282 correct material IDs on the established 104 images. The four new confirmation buildings showed no net material gain. Three became correct and three previously correct IDs were lost, all on the same Hillside Chapel view. The default remains the visible-material matcher from `80b5d60`.

The extra check measures where neighboring texels get lighter or darker along each atlas axis. It rewards a candidate when independently sampled parts of the source face both match the directions of its texture edges. The rule was frozen at `0dddfe3` before grading the new images. Each pair of exports uses the same source-estimated camera and final visible cube coordinates.

| Views | Geometry strict / facade F1 | Material accuracy, default → challenger | Joint strict F1 | Joint facade F1 | Newly right / lost IDs |
|---|---:|---:|---:|---:|---:|
| Established fixed 104 | 91.27 / 93.42% | 92.78 → 94.14% | 84.68 → 85.92% | 88.51 → 89.82% | 296 / 14 |
| Other 100, including training | 90.89 / 93.17% | 97.24 → 97.74% | 88.39 → 88.84% | 91.43 → 91.86% | 107 / 2 |
| Other 28, excluding training | 89.16 / 89.49% | 97.35 → 97.51% | 86.79 → 86.94% | 88.24 → 88.42% | 9 / 1 |
| New decorative development, 4 | 48.40 / 50.00% | 85.91 → 85.74% | 41.58 → 41.50% | 42.68 → 42.59% | 0 / 1 |
| New decorative confirmation, 4 | 60.36 / 62.42% | 75.49 → 75.49% | 45.56 → 45.56% | 47.52 → 47.52% | 3 / 3 |

The initial fixed-set error review informed the challenger, so the 204-view replay is diagnostic. The new comparison uses eight different layouts from the separately captured decorative dataset. All 40 source hashes in that dataset were checked against the established 204 with no overlaps. The eight selected structure footprints differed from the 58 previously indexed geometry hashes after translation, reflection and 90-degree rotation. Development and confirmation shared no building family.

| New source | Correctly placed positions | Correct IDs, default → challenger | Gain / loss |
|---|---:|---:|---:|
| Market Court 1, cool, high right | 185 | 162 → 162 | 0 / 0 |
| Stable Yard 1, warm, low left | 148 | 114 → 113 | 0 / 1 |
| Riverside Mill 2, cool, low left | 0 | 0 → 0 | 0 / 0 |
| Market Court 2, warm, high right | 256 | 230 → 230 | 0 / 0 |
| Hillside Chapel 1, cool, high right | 303 | 278 → 278 | 3 / 3 |
| Hillside Chapel 2, warm, low left | 0 | 0 → 0 | 0 / 0 |
| Walled Garden 1, warm, low left | 182 | 100 → 100 | 0 / 0 |
| Walled Garden 2, cool, high right | 331 | 238 → 238 | 0 / 0 |

The ordinary entry point exported zero blocks on Riverside Mill 2 and Hillside Chapel 2. Their visible truth stays in the end-to-end denominator. The confirmation set had 1,724 visible target blocks, 980 placements, 816 correct positions and 616 exact IDs at those positions. Material accuracy counts exact default-state registry IDs at correct positions. Both joint scores reuse the whole-structure translation chosen by geometry alone. Facade F1 uses the existing native-pixel horizontal sliver exemption.

## What changed in the errors

Across the fixed 104, polished andesite mislabeled stone bricks fell from 234 to 136 cases. The reverse fell from 103 to 92. Tuff mislabeled light gray wool barely moved, from 147 to 146. White concrete mislabeled iron increased from 98 to 106. White concrete mislabeled polished andesite increased from 93 to 96. The fixed set contained 27 views with a net gain and four with a net loss.

The new Stable Yard loss was packed mud changing to mud bricks. On Hillside Chapel the challenger eliminated three smooth-stone-to-stripped-pale-oak errors, but the net confusion counts show three additional smooth-stone-to-polished-andesite errors. Count newly correct positions explicitly when evaluating a particular confusion pair.

For another test, use new visible placements with polished andesite adjacent to stone bricks, smooth stone and white concrete, under daylight and under an overhang. Measure the same reference block across both lighting conditions. The new decorative cohort contained few of the specific confusions that motivated this test.

## Evidence and replay

[Summed scores and target confusion pairs](summary.json) · [Per-view established gains, losses and hashes](established.json) · [Per-view decorative results](decorative.json) · [Frozen source cohort and fingerprints](cohorts.json)

Run `Invoke-WallRecovery.ps1 -GeometryOnly` on each selected decorative source image. Give each its own output directory named for the capture ID. Run the module below from the experiment checkout. The first command evaluates the eight new exports. The second replays the 204 frozen source-only inputs from the published material evaluation.

```powershell
$ids = (Get-Content benchmark/reviews/material-seam-confirmation-01/cohorts.json -Raw | ConvertFrom-Json).selected.id
python -m benchmark.material_seam_confirmation --source $decorativeDataset --builds $ordinaryBuilds --ids $ids --out runs/decorative.json
python -m benchmark.material_seam_replay --cache $typedInputCache --release $typedReview --out runs/established.json
```

The graders open the reference labels after material assignment. They retain each source hash and geometry-pack hash and verify candidate IDs by decoding a deterministic Minecraft datapack. The new experiments add no geometry or hidden support.
