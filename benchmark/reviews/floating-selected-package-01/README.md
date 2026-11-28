# Selected floating reconstruction package

On 84 frozen conditions, starting from large contraction and enabling the guarded edge-aligned ruler with material consensus plus the compatible corner-pair repair removes **1,426 extras**, loses **five correct positions** and gains **37 net exact position-and-ID matches**. All 84 chart packets went through ordinary assembly and Java datapack export. Only Stable Yard 1 low-left and Twinspire 1 earth low-left (O2) changed. The six fixed sky composites retain the expanded baseline exactly.

[Download all 84 chart packets and decoded-verified Java datapacks](selected-charts-and-packs.zip) · [Source fingerprints, selected charts, decoded scores and paired changes](conditions.json.gz) · [Pooled and dataset scorecards](summary.json)

## Package and replay

1. Start from the frozen expanded large-contraction charts in the `floating-extra-isolation` experiment at `e97dd44`.
2. Consume the structural ruler's source-only decision at `a0e67a7`. Both line families must fit and the origin and distant matching atlas faces must survive native half/double-pitch challenges. Use its source material consensus and its **original-fallback** chart packet. An abstention keeps the expanded baseline chart. No forced peer-ruler or recovery fallback enters this package.
3. Consume the corner-pair source decision at `27e86b1`. The receiver's independently measured width and held-out atlas repeats must corroborate its earlier ruler, and the proposed pair must preserve the selected component and all previous joins. The selected pair restores both earlier grids. An abstention keeps the expanded baseline.

These are frozen handoffs for measured images. A novel image requires running the original source predictors. The chart compositor checks that both experiments used the same 84 panel/frame IDs, source RGB and original chart; the edge mask is also fingerprinted. It verifies the exact chart IDs changed by each selected packet. These decisions were disjoint even at the image level. On future overlapping cases the pair needs its source width and connectivity rechecked after the ruler change; this replay stops if that condition arises.

During the run every selected packet was assembled afresh, its saved Java datapack decoded and its positions and IDs compared to the independently measured selected export. The evaluator also checked registered positions, reference fingerprints and complete scores. The 82 original decisions preserved all coordinates and material IDs. No score informed a selection. The two previously reviewed repairs were measured on regression cases; no fresh structure was added.

The portable archive contains a manifest with source, reference, original chart, selected chart and pack SHA-256 values. Each entry under `charts` or `packs` is indexed by panel and frame. For an individual case, the inner `reconstruction.zip` is the ordinary exported Java datapack. The archived chart packets contain all final source cells for the same example.

| Condition | Large contraction correct / missed / extra | Selected package | Exact IDs before → after |
|---|---|---|---|
| Stable Yard 1 warm low-left | 78 / 294 / 1,374 | **74 / 298 / 25** | 26 → 44 |
| Twinspire 1 earth low-left (O2) | 85 / 115 / 77 | **84 / 116 / 0** | 28 → 47 |

Stable Yard gained 18 exact IDs and lost zero exact IDs at their prior world positions. O2 gained 23 exact IDs and lost four. O2's lost geometry target was facade-optional. There were zero newly gained strict positions and zero newly added extras across the package.

| Cohort | Conditions | Strict F1 | Facade F1 | Strict + ID | Facade + ID | Correct / missed / extras |
|---|---:|---:|---:|---:|---:|---|
| Large contraction baseline | 84 | 33.54% | 34.01% | 5.51% | 5.61% | 18,323 / 61,187 / 11,420 |
| Selected package | 84 | **33.98%** | **34.47%** | **5.65%** | **5.76%** | **18,318 / 61,192 / 9,994** |
| Visible Builds v2 | 30/144 | 43.12% | 46.06% | 17.50% | 19.00% | 2,299 / 5,442 / 622 |
| Visible Builds v3 | 30/144 | 34.10% | 34.24% | 3.90% | 3.90% | 13,974 / 47,475 / 6,535 |
| Decorative Builds v1 | 24/40 | 26.90% | 27.77% | 6.82% | 7.20% | 2,045 / 8,275 / 2,837 |

The package assigned 27,723 exported blocks from material evidence and used 589 fallbacks. Exact block-ID matches occurred on 3,048 of 18,318 correct positions (16.64%). The previous baseline assigned 29,132 blocks and used 611 fallbacks, with 3,011 exact IDs on 18,323 correct positions. The six sky composites count toward the 84 conditions.

Use the benchmark Python in the mask-scalelsd checkout with the frozen prior handoff archives available. From this branch, run `python -m benchmark.floating_selected_package all --workers 5`. The stages `charts`, `assembly` and `report` can also be run separately. Materialized charts and scores are under `runs/selected-package/<panel>/`. The fixed-104 production trend remains separately carried forward at 91.27% strict and 93.42% facade, with 20,773 / 2,929 / 1,046.

## Floating branch scorecard

On the six fixed sky composites, the sparse connected-atlas baseline documented at integration `9435934` carries forward **12.64% strict, 12.90% facade, 4.68% strict + ID and 4.77% facade + ID**, with **273 correct / 3,768 missed / 5 extras**. The newly assembled package scores **27.50%, 27.73%, 5.96% and 5.99%**, with **890 / 3,151 / 1,542**. It gains 617 correct and adds 1,537 extras against sparse integration. Both gated repairs abstained on the six, changing zero positions and zero extras compared with large contraction. Only the experiment branch contains the chart compositor, package and results. Integration retains its existing behavior.
