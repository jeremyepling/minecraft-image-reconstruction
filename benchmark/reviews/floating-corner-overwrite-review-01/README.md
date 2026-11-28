# Which grid should a shared corner keep?

[Zoomable phone PNG](mobile-review.png) · [Interactive comparisons](review.html)

The shared-corner step can replace grids on both sides of an accepted corner. On 84 archived floating conditions, 18 had 24 accepted pairs. Of the participating grids, 36 on 17 views already had an atlas measurement that could be restored. Twelve had no earlier measured cells. By changing one grid at a time, using the same downstream resampling and ordinary export, we tested whether the earlier measurement would help. We also ran a no-change replay and an arm restoring every available earlier grid per view.

| Paired arm on 84 conditions | Correct / missed / extra | Strict / facade placement F1 | Strict / facade position-plus-material F1 |
|---|---:|---:|---:|
| Keep the shared-corner grids | 18,323 / 61,187 / 11,420 | 33.54% / 34.01% | 5.51% / 5.61% |
| Restore every available original grid | 18,129 / 61,381 / 11,131 | 33.33% / 33.81% | 5.96% / 6.07% |
| Reference chooses for strict placement | 18,335 / 61,175 / 11,216 | 33.62% / 34.10% | 5.63% / 5.73% |

Among 36 individual restores, placement F1 improved twice, tied 24 times and regressed ten times. Position plus material improved 11 times, tied 19 and regressed six. The placement chooser selects from keeping, restoring one wall, or restoring all offered walls. It picks a different export on two views. Choosing instead for exact position plus material produces 298 additional exact matches and loses 69 net correct positions. Those choices are diagnostics scored with reference data.

The already reviewed Twinspire right-tower case starts from a different upstream packet. Its own frozen replay preserved 110 correct positions and removed all 63 extras. In the broader archive, Twinspire high-right also shows how a different overwritten wall can change which connected group wins. Keep the separately scored Twinspire arms identified by their inputs.

## Source evidence a chooser can use

A retrospective source-only check asked whether the earlier atlas grid scored better against a full-block texture on both halves of the wall. Texture alone suggested seven switches. Two suggestions lost 30 and 15 correct positions in the decoded exports. Rejecting a switch when its assembly dropped an already-exported chart removed both losses. The remaining five suggestions gained 74 exact position and material matches. The positions and extras were identical to their paired baselines. That check did not choose the known exact Twinspire fix. Treat it as a candidate rule and verify it on other buildings before training or promotion. [Preliminary source audit](source-audit.json)

A shared corner can align rows without replacing the measured size across either wall. Keep both grid candidates available and compare local atlas and silhouette evidence before checking whether their assembled outputs retain the neighboring visible surfaces.

## First-pass review and annotation status

The five comparisons have been reviewed against the gold visible edges and the decoded exports. The viewer gives each assessment before showing optional feedback fields. [Native reference measurements](reference-evidence.json)

| ID | First-pass decision | Status |
|---|---|---|
| O1 | Earlier atlas ruler. It removes 63 extras and keeps all correct placements. | Reviewed example |
| O2 | Earlier grid and its adjoining wall as a pair. Both walls reconnect, keeping all 84 facade-required baseline placements and removing 77 extras. | Resolved by replay |
| O3 | Both grids fit the recorded geometry. B gives 59 more exact material matches at the same positions. | Resolved by replay |
| O4 | Keep A for placement. B's local corner fit improves, but it covers less wall and breaks a join. Carry the material evidence forward as a separate test. | Technical follow-up |
| O5 | Keep A. Its reference alignment is better and the exported build keeps 36 more correct placements. | Resolved by replay |

On O2, across 49 complete visible faces, B increases coverage from 85.4% to 97.2%, reduces median corner error from 0.26 to 0.05 grid cells, and brings the median grid-cell count per real face from 3.17 to 1.01. The one-wall swap projected its matching edge 4.87 pixels from its neighbor, beyond the 2.5-pixel join limit. The revised two-wall replay puts the edges within 2.04 pixels. It raises strict and facade placement F1 from 46.96% / 46.67% to 59.15% / 59.36% and exact position plus material from 28 to 47 blocks. The one net strict position lost is a 30-pixel underside that A placed with the wrong material. [Ordinary export replay and coordinates](o2-placement-replay.json)

The separate Mill M1 and Viaduct V1 annotations remain in the [earlier checklist](../floating-grid-round-04/README.md#annotation-checklist). The six fixed sky views tied in the overwrite replays. The integration baseline carries forward 12.64% strict and 12.90% facade placement F1. The candidate work is on the floating experiment branches.
