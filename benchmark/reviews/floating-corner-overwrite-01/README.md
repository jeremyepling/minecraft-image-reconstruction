# When does a shared-corner grid overwrite a better wall grid?

Keep the automatic shared-corner rule in place. Restoring every earlier measured grid removed 289 extra blocks across the 84 active floating conditions but also lost 194 correct positions. A reference chooser could gain 12 correct positions and remove 204 extras. Both improvements came from Twinspire. Keep this branch experimental.

| Frozen 84 conditions | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---|---|---|
| Keep current chart | 33.542% / 34.015% | 5.512% / 5.612% | 18,323 / 61,187 / 11,420 |
| Restore all overwritten charts | 33.335% / 33.805% | 5.961% / 6.074% | 18,129 / 61,381 / 11,131 |
| Reference chooser maximizing placement F1 | 33.623% / 34.100% | 5.630% / 5.734% | 18,335 / 61,175 / 11,216 |
| Reference chooser maximizing position + ID F1 | 33.521% / 33.996% | 6.077% / 6.192% | 18,254 / 61,256 / 11,146 |

The 92 recorded conditions include eight excluded by the frozen active split. All 84 active conditions were available. Eighteen had an accepted seam. Its 24 accepted pairs involved 48 charts. Twelve had no earlier measured chart. Each of the remaining 36 had a different surviving transform, across 17 images. Individual restores improved strict placement F1 twice, tied 24 times and reduced it ten times. For strict position plus ID the counts were eleven improvements, nineteen ties and six regressions. Restoring everything gained 103 correct positions and lost 297, adding 76 extras and removing 365. The reference placement chooser gained 13 and lost one, adding 36 extras and removing 240.

**Twinspire distinguishes the original failure from a panel-wide rule.** On the exact reviewed high-right earth image and chart packet at `5bd7908`, restoring mask-1 kept all 110 correct blocks, removed all 63 extras and raised exact IDs from 42 to 88. On the broader panel's earlier packet for this frame, changing mask-1 tied in placement. Changing mask-2 there gained 13 correct and removed 127 net extras. On high-right cool Watchpost, restoring mask-0-wall-1 removed seven extras, lost 45 correct positions, and raised correct IDs from five to 52. Improving a wall's material can cost the join that retained measured blocks.

## Comparison and replay

We froze the panel identities and source hashes from the depth-refit conditions and used each saved large-contraction chart packet from the extra-isolation benchmark. Its chart and exported pack hashes, evaluator input hashes and decoded local coordinates were checked. Sixty-seven of these 84 original pack hashes also match the later source-depth arm. The flat large-contraction input is the paired control for every row here.

For each accepted donor and receiver we verified the saved pre-corner chart digest against the shared packet, matched source and atlas hashes and checked that the corner transform survived in the packet sent to assembly. Each restore takes the earlier chart with its measured atlas cells, resamples its transform on the original visible mask and ranks new cells using the ordinary atlas consumer. Every other chart is kept intact. The no-op does the same with the post-corner seed on every affected frame. No-op chart records, decoded placements, scores and all zip member contents exactly match the archived control. Each alternate is a full assembly and export. The reference chooser can select keep, any individual restore, or the measured all-restore export on each image. It optimizes pooled micro F1 over these complete options.

| Floating conditions represented | Count | Keep placement strict / facade | Restore all placement strict / facade | Keep ID strict / facade | Restore all ID strict / facade |
|---|---:|---|---|---|---|
| Visible Builds v2 | 30 | 42.83% / 45.70% | 41.06% / 43.93% | 17.02% / 18.47% | 22.33% / 24.34% |
| Visible Builds v3 | 30 | 34.10% / 34.24% | 34.10% / 34.24% | 3.90% / 3.90% | 3.90% / 3.90% |
| Decorative Builds v1 | 24 | 24.75% / 25.41% | 24.75% / 25.41% | 6.04% / 6.37% | 6.04% / 6.37% |

Run `python -m benchmark.floating_corner_overwrite --workers 5` and `python -m benchmark.floating_corner_overwrite_report` in this checkout with the archived worktrees available. The benchmark writes private chart packets and exports under `runs/corner-overwrite`. [Per-frame paths, hashes, scores and paired gains/losses](conditions.json.gz) · [Pooled, split and dataset results](summary.json) · [Exact reviewed Twinspire control](exact-t1.json.gz)

The fixed-104 production trend carries forward at 91.27% strict and 93.42% facade. This round measured the sampled floating conditions above. The full 144-view Visible Builds v3 and 40-view Decorative Builds v1 production suites were not rerun.

## Floating branch scorecard

| Six fixed open-sky conditions | Strict / facade placement F1 | Strict / facade position + ID F1 | Correct / missed / extra |
|---|---|---|---|
| Integration default at `9963dee`, carried forward | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Expanded large-contraction control, measured | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |
| Restore all eligible original charts, measured | 27.50% / 27.73% | 5.96% / 5.99% | 890 / 3,151 / 1,542 |

On the same six images the bypass changed strict and facade F1 by zero points, with zero correct positions gained or lost and zero extras added or removed. Only benchmark code and evidence landed on `experiment/floating-corner-overwrite`. Integration behavior is unchanged.
