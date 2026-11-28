# Does one local atlas match support exporting an entire wall?

The four source seeds in each Stable Yard 1 view fit a repeated image pattern. Three low-left and four high-right tiles passed a complete-face-shaped atlas and quarter-phase test as ochre froglight. Neither view had a passing distant probe. Removing unsupported regions also cut useful faces on other buildings. Keep the source guard experimental. Its promising aggregate result comes from rejecting excess placements on three previously reviewed conditions, with fewer correct placements on all three.

[Source images and export grids](wall-expansion.jpg) · [Per-condition source decisions, held-out samples, fingerprints and decoded scores](conditions.json.gz) · [Pooled scores](summary.json) · [Cohort scores](cohorts.json)

## What changed

Starting at distant-seam replay revision `28f9bbe`, use the 84 active flags in the depth-refit archive and its frozen large-contraction wall charts. The three variants operate only on export membership. Joins, horizontal-face proposals, block materials, orientation and the original component ranking use the untouched chart. The report asserts identical selected origins, joins, reference and yaw in all arms and decodes every reported pack. A smaller export may change the scorer's best whole-build translation.

Challenge a vertical chart with at least 256 cells and at least sixteen times its original seed count, provided its homography still agrees with the seed chart. A replaced ruler or an empty seed set retains the baseline export. For each seed, test whether its entire projected interior lies inside the observed source mask, with at least six pixels per side and 36 interior pixels. Rank the complete sprite against the calibrated 26.3 opaque full-cube atlas. Require correlation at least 0.65 and a margin of 0.07 over the median competing quarter-tile shifts, with at least two measured shifts. This verifies the appearance of the rectangle. An internal pattern on a larger physical block can pass.

- **Anchor.** Keep the whole wall if at least one seed passes. Otherwise fall back to the original seed cells.
- **Regional.** Partition the candidate grid into 6×6 cells. In regions away from accepted seeds, test up to three in-mask candidate cells, starting nearest the region center, at least three cells apart from every seed and each other. Require two complete-face matches, allowing the material to vary. From an accepted anchor, traverse four-neighbor cells through supported regions. Preserve the original seed cells. Empty or sparse distant regions contribute no support. When no seed passes, only seeds remain. The mask and grid were already fitted upstream. These locations are held out of the local seed check.
- **Peer guard.** After seeing the failures of regional clipping, gate it on a contradictory projected block height from at least two independent other wall masks. Evaluate the challenged ruler and each peer ruler at the peer's median cell center in source pixels. Both must have a clear vertical axis. Collapse slices of a common source mask to one median ratio, require every peer to project at least twice the unit height, and require zero accepted distant regions. With inconclusive evidence keep the previous whole chart. This threshold has been tested on previously inspected examples only.

The peer ratios were 2.46× and 4.64× on Stable Yard low-left and 2.79× and 3.06× on high-right. These emerged from other surfaces at potentially different depths. They flag scale disagreement but do not determine the correct horizontal pitch, grid phase or wall ownership.

## Decoded placements

| Active 84 | Strict | Facade | Strict + ID | Facade + ID | Correct / missed / extras |
|---|---:|---:|---:|---:|---|
| Large contraction | 33.54% | 34.01% | 5.51% | 5.61% | 18,323 / 61,187 / 11,420 |
| Anchor | 32.73% | 33.23% | 5.51% | 5.62% | 17,741 / 61,769 / 11,165 |
| Regional | 14.18% | 14.36% | 4.21% | 4.33% | 6,263 / 73,247 / 2,562 |
| Peer guard | 34.36% | 34.88% | 5.61% | 5.73% | 18,200 / 61,310 / 8,236 |

Across the 84 conditions, 27 exports had regional edits and two had anchor-only edits. Peer guarding changed three exports. Comparing geometry-selected registrations, it gained 63 correct positions, lost 186, added 42 extras and removed 3,226. The other 81, including both Observatory 2 views, reuse the baseline pack byte for byte. The per-image reference-only chooser across these four complete exports reaches 34.44% strict geometry, with 18,172 correct, 61,338 missed and 7,845 extras. None of its options repairs the ruler.

| Condition | Regional or peer action | Baseline correct / missed / extra | After |
|---|---|---|---|
| Stable Yard 1 low-left | 0 of 148 distant probes pass, trim 1,452 to 69 | 78 / 294 / 1,374 | 41 / 331 / 28 |
| Stable Yard 1 high-right | 0 of 161 pass, trim 1,261 to 36 | 63 / 291 / 1,198 | 22 / 332 / 14 |
| Pillars sky low-left | 1 of 79 passes, trim 756 to 57 | 69 / 314 / 687 | 24 / 359 / 33 |
| Castle 3 ashlar rear | Regional trim, peer guard preserves baseline | 721 / 4,101 / 0 | Regional 54 / 4,768 / 0 |
| Observatory 2 low-left | Regional trim, peer guard preserves baseline | 195 / 559 / 189 | Regional 9 / 745 / 0 |

For the peer guard, Visible Builds v2 30/144 conditions scored 42.83% strict, 45.70% facade, 17.02% strict + ID and 18.47% facade + ID. Visible Builds v3 30/144 scored 34.28%, 34.43%, 3.92% and 3.93%. Decorative Builds v1 24/40 scored 28.26%, 29.40%, 6.67% and 7.17%. The six sky composites are included in the 84 conditions. No new lighting views or unreviewed buildings were tested. The fixed-104 production trend is carried forward at 91.27% strict and 93.42% facade, 20,773 / 2,929 / 1,046.

Use the benchmark Python from the mask-scalelsd worktree with the saved panel archives accessible. Run `python -m benchmark.floating_verified_wall_expansion --workers 5` then `python -m benchmark.floating_verified_wall_expansion --report`. The ordinary assembly regression, including a filtered empty chart and disconnected group, is `python -m unittest tests.test_relative_face_assembly`.

## Floating branch scorecard

The carried-forward six-sky connected-atlas integration at `c0accfd` (runtime `52de2c9`) has 12.64% strict, 12.90% facade, 4.68% strict + ID and 4.77% facade + ID, with 273 correct / 3,768 missed / 5 extra. The frozen large-contraction path on those six scores 27.50%, 27.73%, 5.96% and 5.99%, with 890 / 3,151 / 1,542. On the same six, the peer guard scores 29.27%, 29.59%, 6.51% and 6.62%, with 845 / 3,196 / 888. It loses 45 correct and removes 654 extras against large contraction. Relative to integration it gains 572 correct and adds 883 extras. Only the experiment code and evidence belong on `experiment/floating-verified-wall-expansion`. Connected-atlas remains the floating integration behavior.
