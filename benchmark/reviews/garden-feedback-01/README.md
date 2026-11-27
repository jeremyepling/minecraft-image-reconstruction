# Walled Garden pillar and texture experiments

The export contained 25 new Walled Garden annotations and 21 previously recorded Castle and Harbor notes. The source image, surface and counter fingerprints and per-mask IDs were checked before accepting the new rows.

## Use visible pillars to measure a recessed wall

The source-only ruler scans tall narrow masks for a vertical texture period. It looks for two similarly colored uprights that bracket the wall in the image, overlap its height, and agree on relative spacing. It interpolates between their local pitches, then measures the wall's visible height in those units. It estimates no absolute camera position. The front pillars can look larger than the rear wall; their spacing supplies an estimate, not an exact wall depth.

| Wall | Pillars | Predicted row spacing | Recorded spacing | Measured span | Visible reference |
|---|---|---:|---:|---:|---:|
| S05 | S13, S10 | 37.87 px | 34.48 px | 2.75 blocks | 3 rows |
| S08 | S18, S20 | 32.61 px | 31.47 px | 2.94 blocks | 3 rows |
| S11 | S17, S18 | 30.20 px | 28.62 px | 2.85 blocks | 3 rows |

The initial outline fit interpreted part of an occluding pillar and door as the rear wall's lower rim. A second fit measured the top and bottom on tall columns of visible pixels, then checked each proposed cell against the original wall mask. It preserved the gap behind the door and pillars. [Pillars and visible cell proposals](pillar-grid.png)

The independent-bay export selected S08. All nine cells landed on the correct native terracotta faces in the correct relative 3 × 3 arrangement. They received the smooth-stone fallback. The linked-bay experiment additionally assumed a common plane and rounded horizontal offsets. Its revised outline yielded 13 correct benchmark positions and zero extra positions, but the native face audit needed three different horizontal shifts for the linked group. Keep the pillar ruler experimental for geometry. The next useful constraint is independent evidence for the horizontal block spacing, grid origin and setback between the pillars and the rear wall.

## Sandstone anchor and repeat experiments

| Walled Garden export | Correct / missed / extra positions | Exact position + ID |
|---|---:|---:|
| Previous registered-seam baseline | 5 / 428 / 0 | 0 |
| Independent pillar bays, revised rim | 9 / 424 / 0 | 0 |
| Linked pillar bays, revised rim | 13 / 420 / 0 | 0 |
| Direct texture anchors on the reviewed patches | 19 / 414 / 5 | 13 |
| Relaxed continuation on the reviewed patches | 22 / 411 / 7 | 16 |
| Connected atlas patches across eligible walls | 28 / 405 / 0 | 28 |

The first two anchor trials targeted the three reviewed Garden masks; the other five chart packets were carried forward. On S01 a single projected grid crossed the cap, wall and pillars at three actual depths. Extending it recovered more positions but added extras. The generalized connected-patch rule uses adjacent strong atlas observations and a connected patch of consistent material. It ran across eligible wall masks on all six sources and kept the Castle join. For Garden the winning S02 patch had 28 distinct correct native faces and materials. Harbor independently recovered 11 distinct prismarine-brick faces and correct materials. The other four exports reproduced the baseline. Garden's five previously aligned positions were replaced when the assembler selected the new component.

| Cohort | Views measured | Strict / facade placement F1 | Strict / facade position + ID F1 |
|---|---:|---:|---:|
| Open sky | 6 / 6 | 12.64% / 12.90% | 4.68% / 4.77% |
| Visible Builds v3 | 4 / 144 | 13.87% / 13.98% | 4.17% / 4.20% |
| Decorative Builds v1 | 2 / 40 | 7.36% / 7.94% | 6.87% / 7.41% |

Pooled geometry moved from 239 correct, 3,802 missed and five extras to 273 correct, 3,768 missed and five extras. Exact IDs rose from 62 to 101. Garden is a reviewed test example; these scores establish a benchmark gain on the saved views. The fixed 104 production trend (0 rerun) is 91.27% strict and 93.42% facade. [Decoded comparison](scores.json.gz) · [Export overlays](placement-comparison.jpg)

## Additional annotations

All five user-labeled cloud masks (S09, S15, S19, S24 and S32) contained zero recorded building pixels and were already rejected by the frozen cloud filter. For S25 the source bounding box contains 3,443 birch-fence pixels. Surface Finder accepted 2,247 and the counter quadrilateral retained 2,163 of those; most of the loss occurs before straightening.

The view chooser on `experiment/garden-view-choice` matched six of the 20 Garden votes versus ten for always choosing the existing warp. Its S10 local measurement did work: 40 px per row versus 39.63 px in the reference, explaining five visible rows. Keep the measured row interval as evidence. The automatic preference rule stays experimental.

Run `python -m benchmark.garden_pillar_ruler` after generating the registered-seam chart packets to reproduce the pillar trial. The reference readings enter after predictions and chart exports.
