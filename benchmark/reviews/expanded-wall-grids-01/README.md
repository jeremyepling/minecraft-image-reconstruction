# What happens when the assembler receives every visible wall cell?

On all 38 paired original test views, exact wall grids with ordinary image-predicted directions raised strict placement F1 from 8.47% with exact surface masks alone to 87.76%. Every view improved. The two arms use the same accepted native masks. The grid intervention supplies wall identity, pixel projection, integer scale and phase, and exactly which wall cells have visible evidence. It measures their combined headroom. It also gives the atlas matcher exact per-face samples. A source-only model would have to recover these measurements from the screenshot.

## Decoded exports

| Input and join rule | Correct | Missed | Extra | Strict / facade placement F1 | Strict / facade position + ID F1 |
|---|---:|---:|---:|---:|---:|
| Exact masks, measured grids and directions | 2,632 | 56,806 | 74 | 8.47 / 8.56% | 2.89 / 2.92% |
| Exact wall grids, measured directions, one edge | 47,603 | 11,835 | 1,446 | 87.76 / 88.35% | 50.66 / 51.00% |
| Exact wall grids, measured directions, two edges | 48,657 | 10,781 | 37 | 90.00 / 90.60% | 51.24 / 51.58% |
| Exact wall grids and signed axes, one edge | 47,406 | 12,032 | 7,214 | 83.13 / 83.66% | 47.41 / 47.71% |
| Exact wall grids and signed axes, two edges | 51,380 | 8,058 | 976 | 91.92 / 92.52% | 50.92 / 51.25% |
| Exact wall grids and signed axes, one edge within 0.0001 px | 51,706 | 7,732 | 0 | 93.04 / 93.65% | 51.32 / 51.66% |

| Cohort | Held-out views tested / all available | Exact masks | Exact grids, measured directions, 1 / 2 edges | Exact grids, signed axes, 1 / 2 edges | Signed axes, 0.0001 px |
|---|---:|---:|---:|---:|---:|
| Visible Builds v2, twinspires | 12 / 144¹ | 31.24% | 83.08 / 83.08% | 83.16 / 83.16% | 83.16% |
| Visible Builds v3, castles | 18 / 144 | 6.80% | 90.42 / 92.95% | 84.87 / 94.76% | 95.87% |
| Decorative Builds v1, gardens | 8 / 40 | 8.03% | 34.08 / 32.47% | 44.15 / 39.52% | 44.15% |

¹ All twelve frozen v2 test views were used. Cohort counts include views reserved for training and development. The machine-readable report supplies facade, typed scores and material counts per arm, image and cohort. The fixed 104 production result at the branch fork carries forward from the production scorecard at 91.27% strict and 93.42% facade, with 20,773 correct positions, 2,929 misses and 1,046 extras. This floating diagnostic evaluated 0/104 historical images. The original-image baseline with all ordinary stages is 3.72% on the same 38 views. [Earlier paired exports](../expanded-crop-ceiling-01/README.md)

## Handoff and verification

The starting revision is `d4f2e09` on `experiment/crop-ceiling-expanded`. Its 38 original RGB sources and native-mask packets are identified by SHA-256. All 4,242 surface masks were verified against the native frontmost pixel owners. Per chart the generator selects a visible unit wall face, fits a planar homography to its four native corners and independently rebases local coordinates. U is screen-right and V is world-up. Their cross product is the outward normal. Both image tangent signs were checked for every wall face. All other corners on the chart agreed with the frozen recorded image corners within 3.27 × 10⁻⁹ native pixels. Direct camera reprojection agreed within 5.09 × 10⁻¹³ pixels.

The 1,958 wall charts supply 61,948 frontmost wall-face cells covering 57,650 of the 59,438 visible block positions. Each target wall face has at least one owned source pixel. Native world coordinates and the camera enter the answer generator and validation. The join solver receives independently rebased charts, image corners and cell lists. It estimates the positions of the charts relative to each other and exports only the largest connected component. Projected unit corners may extend behind an occluder on a partially visible face. Like the active chart handoff, this experiment supplies no per-edge visibility flags. The solver may therefore consider the entire projected edge of a partially visible cell.

All three previously proposed nonwall charts and their cells/materials were retained. The measured-direction arm passes through the ordinary orientation estimator. Its anchor gave a fixed scoring yaw on every view. Of the wall charts, 985 received the correct signed axes and 850 received different axes after the common yaw. The orienter filtered out 123. The signed control supplies exact chart tangent and outward axes for all 1,958 full-cube walls. The three preserved charts retain their ordinary orientations expressed in the same compass. Since this input already has world-facing axes the scorer fixes yaw at zero. All scores use one geometry-selected integer translation for the complete export, including material grading.

New chart cells use the ordinary Java 26.3 masked atlas tile ranker, sampling 16 × 16 nearest RGB texels inside the recorded face owner. The ranker chooses a texture and block ID using source colors and patterns. Insufficient visible texels use the ordinary smooth-stone fallback. Confidence comes from the source match and influences chart order on material conflicts. The one-edge measured-direction arm has 48,935 assigned and 114 fallback positions. It identifies 27,479 of 47,603 correctly placed blocks (57.73%). Every arm was exported to an actual datapack and decoded back to its complete coordinate/ID mapping. Each scored row includes the input packet and pack hashes.

## Why exact axes can hurt

The active connected-atlas rule accepts one shared edge within 2.5 px and accepts coplanar alignment within 0.5 px. The unchanged edge-length-dependent gate also applies. The tighter arm sets both pixel tolerances to 0.0001. The two-edge arms change only the number of distinct matching segments required for a perpendicular join.

With exact signed axes the one-edge rule accepted 1,326 physically correct perpendicular joins, 512 correct coplanar joins and 50 perpendicular joins with a wrong integer shift. Castle accounts for 48 of the wrong joins. The two-edge rule reduced the physically wrong count to 22 and lost genuine one-edge garden junctions. Tightening both tolerances on exact grids accepted 1,340 correct perpendicular and 772 correct coplanar joins, with zero incorrect shifts. Competing wrong shifts in a connection cycle can cause the solver to reject a true join. The tighter arm recovers 51,706/57,650 wall-supported positions and leaves 5,944 wall-supported blocks outside the exported component. A further 1,788 visible targets have no full-cube wall face.

On Castle 03 limestone high-right, signed axes admit 102 charts, compared with 100 using measured directions. The selected component expands from 18 charts to 84. One segment pair only 1.08 px apart joins distant walls, shifting 1,104 visible face cells 24 blocks in X and 25 in Z. Another pair 0.67 px apart shifts 247 face cells 8 in X and 12 in Z. Both false connections disappear in the two-edge control. For this frame the signed one-edge export has 434 fewer correct blocks and 773 more extras than the measured-direction version. [Source image and magnified edges](castle-false-joins.jpg) · [Native and estimated shifts](castle-false-joins.json)

Keep `experiment/expanded-wall-grids` experimental. The measured gains show that visible wall lattice coverage can pay off in final placement. The 0.0001 px result depends on exact reference corners. The ordinary game-facing pipeline was unchanged. The one-edge garden result motivates measuring a source-image junction check that can distinguish overlap in projection from a physical corner.

## Reproduction

From this worktree, with the frozen test sources, existing native handoffs and benchmark Python environment installed:

```powershell
python -m benchmark.expanded_wall_grids --workers 5
python -m benchmark.expanded_wall_join_audit --replay source-normal-two-edges signed-two-edges signed-tight-edges --workers 5
python -m benchmark.expanded_wall_join_example
```

The runner stores packets under `runs/wall-grids/{source-normal,signed}/charts/sky` and private exports under `runs/wall-grids/<arm>/assembly/connected-atlas/sky`. The historical folder name `sky` is an internal variant key. Source hashes identify the original screenshots with their original backgrounds. [Scored cases and join audit](scored.json.gz)
