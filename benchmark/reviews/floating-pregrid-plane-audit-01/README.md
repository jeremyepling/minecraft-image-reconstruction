# Trace floating grids to their visible planes

On the frozen 84-condition expanded benchmark, 66 of 166 selected charts span substantial parallel planes, across 59 conditions. Eleven charts in six conditions include substantial support facing another direction. Nine charts in nine conditions contain substantial material changes along one plane with no large second depth or direction. The [source and grid comparison](native-plane-examples.png) shows all three cases and a chart whose source pixels are coherent but whose export lands behind their visible blocks.

## Where the extra blocks originate

| Exported extra, measured against the source center's owner | Expanded 84 | Fixed six sky images |
|---|---:|---:|
| Samples another same-facing source plane and lands at the wrong depth | 5,004 | 237 |
| Samples the dominant source plane and lands at the wrong depth | 1,683 | 321 |
| Lands at the source block's depth and has the wrong tangent coordinate | 2,840 | 904 |
| Samples a face with a different normal | 358 | 52 |
| Samples background or falls outside the image | 80 | 7 |
| **Total extras** | **9,965** | **1,521** |

These numbers describe exported coordinates using the existing whole-build alignment and the native frontmost owner of each grid-cell center. They measure where evidence and placement diverge. A cell may span different owners away from its center, and the count does not estimate how many blocks any proposed split or offset could recover.

- **Stable Yard high-right.** Split `mask-0~p24` already isolates most of the south-facing roof band at z=2. Of 260 emitted cells, 236 center on the band and all 236 export one block behind it. Of those, 212 are extras. Part `p35` still combines two south-facing depths. The low-left corrected ruler still samples both sides of depth steps on the original `mask-0`.
- **Castle 02 ashlar high-right.** `mask-0` contains 118,599 native pixels on south-facing z=27 and 85,946 on z=28, plus smaller setbacks. The chart exports 621 samples at the wrong depth of their native owner. Of 613 extras, 569 are wrong-depth placements from a secondary parallel plane.
- **Viaduct 02 fixed sky.** `mask-0` covers 121,369 native pixels on south-facing z=6 and 59,433 on z=7. Among 173 extras, 36 are at the wrong depth of the visible owner and 124 are already at its depth with a different tangent coordinate.
- **Pillars fixed sky.** Quartz bricks, polished diorite, and spruce planks share the south-facing plane at z=2 in `mask-1~p7`. This chart contributes no wrong-depth centers. Its dense grid still produces misplaced cells along that plane.
- **Walled Garden fixed sky.** All 40 sampled centers on `mask-1-wall-1` land on the same visible birch-trapdoor plane. The exported blocks lie behind the owner block depth for all 40 centers, with 21 extras. Changing source-mask assignment alone would leave the chart's placement unresolved.

## Reproduce from the saved exports

The audit reads the pinned expanded charts, score receipts, and actual pack bytes from the [portable benchmark archive](../floating-expanded-benchmark-01/manifest.json). It decodes the packs, rechecks the single alignment and scores, then reconstructs the emitting chart and UV index of all 28,283 exported blocks using the saved selected origins, confidence precedence, and `export_uv` filters. The reconstructed materials and coordinates match every decoded pack. Source image, reference, chart, and pack hashes are checked before evaluation. All visible label corners reproject against the recorded camera within 0.0001 native pixels, and the pixel counts match the frontmost ID rasters.

The audit records both the parent mask and the selected split part where the frozen export used one. A secondary same-facing plane or another facing direction is substantial when it owns at least 64 native pixels and at least 10% of the visible mask pixels. Material changes use the same thresholds within the dominant plane. The export-depth comparison uses the owner block's cube plane, so an inset visible face on a trapdoor or another partial block does not misclassify an otherwise correctly placed block. Ten selected charts have at least ten visible centers, at least 90% from one physical plane, and at least 90% shifted to the same side of the owner block depth.

Use the archived captures and source handoffs available to the frozen benchmark. The historical `floating-extra-review` checkout provides the handoff lookup and the original horizontal-chart builder. The `floating-package-rechecks` checkout provides the saved split region maps. The scripts write only into `runs/pregrid-plane-audit` in this checkout.

```powershell
python benchmark/reviews/floating-pregrid-plane-audit-01/audit.py
python benchmark/reviews/floating-pregrid-plane-audit-01/render.py
```

The [per-condition audit](conditions.json.gz) records native plane and material counts for each selected chart, the resulting source-center categories, and the exact official geometry and position-plus-material scores. This branch adds the audit and images. Keep it as a diagnostic because the recorded owners and camera are inputs to the evaluation.

## Floating branch scorecard

| Frozen expanded split-border input | Strict / facade placement F1 | Strict / facade position + material F1 | Strict correct / missed / extra |
|---|---:|---:|---:|
| 84 conditions | 33.99% / 34.48% | 5.66% / 5.76% | 18,318 / 61,192 / 9,965 |
| Six fixed sky conditions | 27.59% / 27.83% | 5.98% / 6.01% | 890 / 3,151 / 1,521 |

The measured expanded input comes from the pinned `51bd1e9` archive. The audit verified existing exports and introduced no changes to their scores or positions. No reconstruction behavior landed on integration.
