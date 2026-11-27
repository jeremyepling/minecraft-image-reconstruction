# Wall-ruler transfer controls

The Java 26.3 ordinary baseline is the promoted `e3bd220` release. The whole-mask comparison uses the native single-tile arm of `experiment/local-atlas-wall-rulers`. Across 13 views, it recovered 136 strict positions, lost 11, added 35 extras and removed 44. The decoded datapacks reproduced the published position and exact ID scores in every cohort and split.

| Cohort | Tested / available | Strict / facade placement F1 | Strict / facade position + ID F1 |
| --- | ---: | ---: | ---: |
| Fixed historical regression | 2 / 104 | 87.02 / 92.52 → 87.02 / 92.52 | 85.46 / 90.70 → 85.46 / 90.70 |
| Visible Builds v3 | 9 / 144 | 51.91 / 55.26 → 52.61 / 56.02 | 31.21 / 34.04 → 32.00 / 34.91 |
| Decorative Builds v1 | 2 / 40 | 64.82 / 65.53 → 68.14 / 69.16 | 56.79 / 56.99 → 59.91 / 60.42 |
| Development | 5 / 5 | 64.36 / 65.10 → 64.43 / 65.21 | 41.18 / 41.60 → 41.15 / 41.61 |
| Pilot transfer | 8 / 8 | 48.92 / 53.02 → 50.14 / 54.39 | 31.97 / 35.69 → 33.33 / 37.21 |
| Pilot pooled | 13 / 13 | 54.24 / 57.48 → 55.05 / 58.37 | 35.14 / 37.87 → 36.01 / 38.83 |

The audit checked 232 file receipts across the 13 views, including the source images, visible labels, saved model handoffs, placement camera receipts and ordinary packs. It verified the versioned atlas and joined each candidate on its corrected ordered surface, axis and level signature and pack hash. Nine views offered carriers and six changed their decoded inventories. Ten archived no-op exports independently matched both the ordinary decoded IDs and geometry-aligned IDs. Scoring maximizes strict geometry after one integer XYZ translation for the whole build and uses that same translation for facade and exact-ID metrics. Prior fixed views retain the released geometry-optimal tie choice. Expanded candidate and no-op assignment counts came from each export's placement map. The fixed receipts' assignment counts are carried from the original score records. [Whole-mask audit](whole-mask.json)

The [frozen extension](extension.json) adds eight verified ordinary packs. Its fixed views are historical regression cases outside the pilot layouts. Harbor 01 and Castle 03 are new layouts in previously inspected families. Decorative coverage adds one view of Hillside Chapel, a family absent from the pilot, and one view of Walled Garden 02. All rows retain their existing split. SHA-256 with salt `wall-ruler-transfer-extension-v1` chooses buildings and views within the declared strata. The Harbor pair uses both palettes at the same selected view angle. Castle explicitly samples one camera-supported and one camera-abstained view. The latter contains 4,822 strict visible targets and remains in the scoring denominator with an empty ordinary pack. The freeze did not use placement or ID scores. The extension receipt measures the ordinary packs only.

## Local chart replay audit

The [local pilot audit](local-pilot.json) verified 13 no-op replays and the revised exports. Seeds alone recovered one position and removed one extra on Walled Garden. The connected option recovered 31, lost 10, added one extra and removed three. The effective changes were Castle 01 and Walled Garden, with strict and facade placement F1 of 54.38% and 57.63% and exact position-plus-ID F1 of 35.18% and 37.90%. Training Viaduct's aligned inventory matched ordinary. On the single exploratory Castle 01 replay that removes an old vote only within the new chart's support, the aligned export recovered 74, lost two, added three extras and removed one.

The [local extension audit](local-extension.json) independently verified eight no-op replays and all candidate outputs. All aligned block inventories from the seed, connected and overlap options matched ordinary on every extension view. The whole-mask option recovered two positions with exact IDs on Hillside Chapel. The whole-mask strict F1 moved from 38.66% to 38.68% on the extension. There were seven camera-supported views with nonempty exports. On those views, whole-mask strict F1 moved from 57.21% to 57.24%. The four saved overlap reuses matched the connected placement handoffs and removed zero old votes. These results support keeping the ruler variants experimental.

From the checkout of `benchmark/wall-ruler-transfer-controls`, run `python -m benchmark.wall_ruler_transfer_controls`, `python -m benchmark.wall_ruler_local_receipt`, and `python -m benchmark.wall_ruler_local_receipt --extension`. Use the workspace Python with OpenCV, the prepared Java 26.3 atlas and the three saved run worktrees. The audit decodes saved packs offline. The branch contains benchmark code and receipts.
