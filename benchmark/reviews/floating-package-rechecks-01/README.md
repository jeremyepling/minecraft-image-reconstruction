# Depth split and sky checks on the selected expanded package

Applying both source rules to the actual revised packets removed 29 extras and preserved every retained voxel’s material on the 84-view comparison. Split eligibility must use the current ruler. Reapplying a stale decision to the revised Stable Yard low-left packet lost 18 correct placements and five exact material matches. Keep the two guarded switches available for the expanded research comparison. This result supports a comparison against the package on new views. The sparse runtime has separate inputs and behavior.

The input package at `ac69f22` combines expanded visible charts, the guarded Structural Ruler fallback and O2 paired repair. Rechecks verified all 84 source, chart and emitted pack hashes and baseline scores against the independently published package records. Conditions include 30 Visible Builds v2, 30 Visible Builds v3 and 24 Decorative Builds v1, with the same six fixed sky composites reported separately. These are previously reviewed captures. Whole-build registration, decoded scoring and visible-only targets stayed fixed. [Selected package](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/ac69f22/benchmark/reviews/floating-selected-package-01/README.md)

| Replayed switch | Positions gained / lost | Extras added / removed | Changed exports on the 84 |
|---|---:|---:|---|
| Depth split only | 0 / 0 | 0 / 20 | Pillars 01 low-left removed 12, Stable Yard 1 high-right removed 8 |
| Border-connected projected sky only | 0 / 0 | 0 / 9 | Castle 01 high-right removed 6, Viaduct 02 high-right removed 1, Pillars 01 low-left removed 2 |
| Both switches, in sequence | 0 / 0 | 0 / 29 | The same four images, including 14 distinct extras on Pillars |
| Depth split with pre-package eligibility frozen | 0 / 18 | 10 / 33 | Also alters the revised Stable Yard 1 low-left component |

## Where the earlier measurements transfer

The depth split divides existing cells by native visible-pixel component ID before ordinary assembly. Each cell keeps its ruler and material. An unchanged chart packet reuses its source guards. All changed packets recompute the original peer-height, texture-support and expansion tests. Masks remain valid only with matching screenshot, surface support and immutable mask hash. Splitting again removed the same eight and twelve extras on the two still-eligible charts. It does not establish physical depth or repair a grid’s scale.

The original Stable Yard low-left wall had 1,452 exported cells. Its package replacement has 99 and falls below the 256-cell expansion threshold. Forcing the original target decision split those 99 cells into 13 groups; six rejoined in the retained 78-cell component. The displayed pack changed from 74 correct / 298 missed / 25 extra and 44 correct IDs to 56 / 316 / 22 and 39 correct IDs. The unsplit package and all currently eligible switches preserved the 99-cell export. The O2 package case also recomputed both switches and preserved its source-selected result.

The earlier sky filter removed ten extras over the depth-refit-with-small-steps baseline. Here each current cell uses its actual chart homography and 81 native source sample locations. At least 41 must match row-color sky connected to an image border. Out-of-image samples give zero sky votes. The candidate exports the surviving UVs while preserving the join input and component ranking. On old Stable Yard low-left, the oversized cell at UV (54, 11) had 44 sky votes and was exported. The revised packet has no sky-majority cell. Castle 03 limestone still had one veto in a chart outside the exported result. The broader cloud-color variant previously lost 13 net correct positions. Its earlier evidence does not justify including it in the package.

The 29 combined removals contained five packed ice and four blast furnace IDs. No added voxel or material substitution occurred in that arm. All aligned coordinates and before/after material IDs are in the per-view records. [Per-view source votes, masks and exact changed sites](conditions.json.gz)

The earlier signed part-depth displacement remains a distinct candidate for a new package replay. That work measured 132 net additional correct positions and 127 fewer extras on the flat expanded baseline. It requires the integer normal-offset assembler and, for its later local refit, a per-patch sampling homography. The package uses ordinary chart cells and a single homography per chart. Combining historical totals would miss overlap at Stable Yard. Keep forced atlas refitting and splitting from stale eligibility out of the package. A valid local ruler can still disappear at component selection.

Run `python -m benchmark.floating_package_rechecks` after materializing the 84 selected-package receipts, packets and packs. The runner uses the registered source benchmark checkouts. [Pooled and dataset metrics](summary.json)

## Floating branch scorecard

| Inputs and method | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |
|---|---:|---:|---:|
| Sparse integration at `9435934`, fixed six, carried forward | 12.64% / 12.90% | 4.68% / 4.77% | 273 / 3,768 / 5 |
| Expanded package, six | 27.499% / 27.734% | 5.963% / 5.986% | 890 / 3,151 / 1,542 |
| Expanded package + both, same six | 27.588% / 27.826% | 5.983% / 6.005% | 890 / 3,151 / 1,521 |
| Expanded package, 84 | 33.978% / 34.470% | 5.654% / 5.755% | 18,318 / 61,192 / 9,994 |
| Package + eligible depth split, 84 | 33.985% / 34.476% | 5.655% / 5.757% | 18,318 / 61,192 / 9,974 |
| Package + projected border sky veto, 84 | 33.981% / 34.473% | 5.654% / 5.756% | 18,318 / 61,192 / 9,985 |
| Package + both, 84 | 33.987% / 34.479% | 5.655% / 5.757% | 18,318 / 61,192 / 9,965 |

The combined replay gained 0.009 strict placement F1 points across 84 and 0.090 points on six. Its 29 net removed extras include 21 on the six, with no change to strict correct counts or exact IDs (3,048 and 193). Only comparative benchmark code and evidence landed on the rechecks branch. Floating integration and production retained their previous behavior. The fixed-104 production trend is separate, carried forward at 91.27% strict and 93.42% facade placement F1.
