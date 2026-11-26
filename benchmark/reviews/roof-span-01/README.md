# Can roof texture and the posts recover the visible span?

The strict atlas check raised relative geometry F1 from **90.01% to 90.10%** on 104 fixed whole-structure photographs. It recovered 38 visible blocks and introduced one extra at a hidden source coordinate. Fourteen views improved and 90 tied. This change shipped in the ordinary placer. [Ten lowest-scoring structures, stage by stage](worst/README.md) · [Before and after roof footprints](roof-spans.jpg)

| Placement method | 104 fixed | 20 alternate | 4 loading dock / switchback | 4 shelter / gate | Roof blocks anchored to both posts, fixed / alternate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous default | 90.01% | 88.22% | 83.64% | 82.41% | 293 / 308 |
| Fit roof boundaries to two photographed contacts | 90.03% | 88.22% | 83.64% | 82.41% | 293 / 308 |
| Verify roof strips against the atlas, permissive | 90.32% | 88.45% | 83.76% | 83.40% | 297 / 309 |
| **Verify roof strips against the atlas, strict** | **90.10%** | **88.29%** | **83.64%** | **83.05%** | **296 / 308** |

All methods used the v2 whole-structure admission list and the same verified screenshot, image-derived camera, surface and count inputs. The fixed cohort contains 459 visible roof targets on ten measured canopies. The alternate cohort has 398 on eight. The 72 development images were separate from the table. All displayed evaluations had been used in prior rounds, including the shelter and gate. Scores count only blocks visible in the photograph after a common translation of the entire build. The camera-registered fixed F1 for the shipped method is also 90.10%. For continuity, recombining the eight retired, zero-output texture controls and their 384 old target positions gives **89.35%** over the historical 112 images, versus 89.26% for the preceding default. The ten-layout ranking uses only the 104 framed photos. [Control](control/summary.json) · [Paired results, both coordinate metrics and opening probes](results.json) · [Retired control counts](../nearby-thin-01/structure-scope.json)

## Which evidence helped

The strict check took 0.45–0.83 extra seconds per photo on three timed images. These are median times for the added stage over three warm repetitions. [Timings](timing.json)

A photographed roof fragment or the top of a reconstructed support nominates a short strip in either horizontal direction. The atlas check tests individual block faces on original pixels, compares texture alignment with adjacent depths, checks the image silhouette and renders candidates together. The shipped setting requires at least two adjacent measured cells and an existing roof or support contact. Positions without texture witnesses break the strip. The missing-atlas and missing-camera paths keep the prior placement.

The two-support boundary fit gained 17 true blocks at the cost of 11 extras on the fixed suite. It recovered no additional anchored roof cells. Allowing it to replace a competing layer made no difference to the scores. The permissive texture variant gained 146 true blocks but also 24 extras, including 12 hidden-source positions, and lowered F1 on five fixed images. Strict verification preserved every prior correct block, retained all 18 opening widths and left all 243 sampled clear sightlines open. It gained five visible blocks with zero extras on the alternate angles, and nine with zero extras on the shelter/gate set. The one new fixed-suite error was on Canopy 07 left. [Contact experiment](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/91d22e0/benchmark/reviews/roof-span-contacts-01/README.md) · [Texture settings and development tradeoffs](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/851784c/benchmark/reviews/roof-span-texture-01/README.md)

## Why Canopy 06 is still worst

Its F1 increased from **39.19% to 41.33%**. The build recovers 31 of 84 visible positions, with 35 extras. Only 8 of its 60 visible roof cells are correctly placed against the two front posts, up from six. Its output still extends 45 blocks along a 10-block reference depth. A complete command-line run from the original photo reproduced the 66 exported block positions and material states. [Canopy 06 stage map](worst/structure-01.jpg) · [Production and command-line readback](verification.json)

All 54 previously missed roof cells had enough horizontal pixels inside the detected horizontal mask. A scoring-only diagnostic used the measured roof height to intersect those pixels with the **estimated** camera grid. It recovered the correct horizontal coordinates on all 54. Changing the assumed height by one block in either direction recovered zero. Across the ten fixed canopy views this test succeeded for 84 of 166 previously missed roof cells. Correct height and horizontal segmentation therefore explain only part of the remaining gap. [Per-photo height and mask diagnosis](diagnosis.json)

The permissive strip search nominated all 54 misses on Canopy 06. Thirty-eight produced a preliminary texture match, nine passed its run thresholds and two passed the final visibility check. Accepting the other seven would have obscured 16 earlier placements. Source truth identifies all 16 as misplaced after the post alignment. A separate experiment compared the incoming and old face textures on those contested pixels and required every disappearing block to lose its whole image witness. It produced no gain over the strict method. This replacement experiment and the contact-only and permissive methods remain experimental. The next useful test is to score an entire roof plane from its visible rim, surface texture and the projected tops of the posts **before** placing individual competing blocks. [Candidate and blocker trace](https://github.com/jeremyepling/minecraft-image-reconstruction/blob/851784c/benchmark/reviews/roof-span-texture-01/roof-diagnostic.json)

## Reproduce

Restore the saved v2 inputs, Java 26.3 atlas and portable shelter/gate fixture from the [previous evaluation](../framed-depth-01/README.md). From the review checkout, use a separate output directory for each variant. The compact comparison archive contains every per-image score, input hash and changed export. Identical exports are referenced by their SHA-256 in the [previous default archive](../framed-depth-01/scores.zip). [Comparison archive](comparisons.zip)

```powershell
python -m benchmark.roof_span_eval --candidate engine.roof_span_texture:conservative --out runs/conservative --workers 5
python -m benchmark.roof_span_transfer --inputs ../framed-depth-transfer/runs/transfer-input --candidate engine.roof_span_texture:conservative --out runs/transfer-conservative
python -m benchmark.roof_span_eval --candidate benchmark.framed_geometry_order:production --out runs/production --workers 6
python -m benchmark.roof_span_worst --run runs/conservative --out benchmark/reviews/roof-span-01/worst
python -m benchmark.roof_span_viewer benchmark/reviews/roof-span-01/worst
```

Use `--candidate-root` and the linked experiment commits to replay the other variants. The frozen adapters were `engine.roof_span_contacts:recover`, `engine.roof_span_contacts:exchange`, `engine.roof_span_texture:recover` and `engine.roof_span_occlusion:recover`. The review scripts operate on images offline. [Whole-structure framing contract](../../../training/DATASET.md#whole-structure-admission)
