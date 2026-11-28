# Selected floating package

The selected package combines the expanded large-contraction charts, the edge-aligned ruler with the original-grid fallback, and the source-selected O2 corner pair. A priority rule materializes a single chosen chart packet for each view. The two source checks selected different views in this cohort. Every packet then went through ordinary assembly, export, ZIP decode and reference scoring. The reused source selections were frozen by hash. A future case where both checks activate needs to revalidate the corner on the changed ruler.

[Zoomable comparison of the two changed exports](selected-package.png)

## Compare like inputs

| Cohort and method | Strict / facade placement F1 | Correct / missed / extra | Strict / facade position plus material F1 |
|---|---|---|---|
| Sparse floating integration, six fixed sky views | 12.64% / 12.90% | 273 / 3,768 / 5 | 4.68% / 4.77% |
| Frozen expanded baseline, same six | 27.50% / 27.73% | 890 / 3,151 / 1,542 | 5.96% / 5.99% |
| Selected package, same six | 27.50% / 27.73% | 890 / 3,151 / 1,542 | 5.96% / 5.99% |
| Frozen expanded baseline, 84 | 33.54% / 34.01% | 18,323 / 61,187 / 11,420 | 5.51% / 5.61% |
| Selected package, same 84 | 33.98% / 34.47% | 18,318 / 61,192 / 9,994 | 5.65% / 5.76% |

The archived 84 conditions contain 30 Visible Builds v2, 30 Visible Builds v3, and 24 Decorative Builds v1 conditions. The six fixed sky views are included and match the sparse baseline source images and references. The 84 replay covers a subset of the 240 active views. The sparse default was independently checked from the existing decoded exports. The selected package freshly ran the ordinary assembler on all 84 saved source-derived chart packets.

The package removes 1,426 extras with zero new extras, loses five strict correct positions and gains 37 net exact position and material matches. Stable Yard low left supplies 1,349 fewer extras, four lost facade-required positions, and 18 more exact matches. Twinspire O2 removes 77 extras and keeps all previously correct facade-required blocks, losing one optional strict underside. It gains 23 and loses four exact position and material matches. The other 82 views have identical decoded coordinates and materials.

The two corrections leave the six fixed expanded results identical, so the comparison with sparse integration still shows 617 more correct positions and 1,537 more extras. Use the package as the expanded benchmark for further tests. The sparse integration remains the everyday floating baseline until a separate decision about that trade-off.

[Selected package runner and score archive](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-selected-package)

## Earlier experiments rerun on the selected package

Recompute each switch with the package's actual handoff. The source-only candidates use the existing depth and color partition and the conservative border-connected sky check.

| Paired arm on 84 | Correct / missed / extra | Strict / facade F1 | Exact position plus material |
|---|---|---|---|
| Selected package | 18,318 / 61,192 / 9,994 | 33.978% / 34.470% | 3,048 strict / 3,024 facade |
| Pink split only | 18,318 / 61,192 / 9,974 | 33.985% / 34.476% | 3,048 / 3,024 |
| Projected border sky check | 18,318 / 61,192 / 9,985 | 33.981% / 34.473% | 3,048 / 3,024 |
| Both guarded switches | 18,318 / 61,192 / 9,965 | 33.987% / 34.479% | 3,048 / 3,024 |

Pink split-only removes 12 extras on Pillars and eight on Stable Yard high right. The sky switch removes six from Castle, one from Viaduct and two from Pillars. The older sky run had ten removals on different charts. The new Stable Yard low-left ruler no longer has the sky-majority cell that triggered that retirement. All correct coordinate and material sets survive the combined replay.

On the six fixed sky views, the package with both switches reaches 27.59% strict and 27.83% facade, with 890 correct, 3,151 missed, and 1,521 extra. Position-plus-material F1 is 5.98% / 6.01%. Against the sparse integration this is 617 more net correct positions and 1,516 net additional extras. The sparse integration export remains a separate benchmark.

A deliberate overlap diagnostic reapplied the old oversized-chart split target to the revised Stable Yard low-left chart. It went from 74 correct, 25 extras and 44 exact matches to 56 correct, 22 extras and 39 exact matches. The real switch abstains after checking the corrected chart size. Reusing the old flag would break the ruler improvement.

Use the selected package plus the two guarded switches for the next expanded comparison. Keep comparisons of old chart attachments and horizontal joins separate until they are rerun against that new package.

[Rechecked split and sky runner](https://github.com/jeremyepling/minecraft-image-reconstruction/tree/experiment/floating-package-rechecks)
