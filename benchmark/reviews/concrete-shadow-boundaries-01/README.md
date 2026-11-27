# Concrete shading and visible surface boundaries

Compared 204 frozen Minecraft 26.3 source images at geometry revision `983b6b8`. The fixed group contains 104 evaluation views. The other 100 contain 72 training and 28 transfer views. Proposals use source RGB, released support masks and image-estimated plane information. Grading reads the native visible-face raster after freezing proposals. No candidate adds visible area across holes or targets hidden blocks.

## Why concrete looks like multiple surfaces

Shopfront has a broad dark band on its concrete wall. One recorded block face ranged from 112.5 to 161.0 in brightness between its interior 10th and 90th percentiles. Across the fixed views, 644 of 2,281 eligible concrete faces had an interior range of at least 12 levels and 484 reached 24. This is consistent with lighting variation. The captures do not identify the renderer mechanism.

[Shopfront source image beside the same image with the recorded concrete face outlined](shopfront-concrete-light-range.png)

| Fixed 104 views, same concrete material | Two-pixel samples | Brightness step of at least 12 | Share |
| --- | ---: | ---: | ---: |
| Within one block face | 1,077,588 | 398 | 0.037% |
| Across block edges on one recorded surface | 70,200 | 488 | 0.70% |
| Across different face directions | 9,228 | 8,710 | 94.39% |
| Across parallel but separately recorded surfaces | 207 | 34 | 16.43% |

Brightness is the mean of the three source RGB channels. Counts weight long edges more heavily than short edges. A broad change on a single face is common even though an abrupt local step within that face is rare. Color ratios survived the brightness change on 8,708 of the 8,710 direction-change samples. Similar hue on both sides is weak evidence that they share a plane. The directional pattern held on the other 100 views (3,837 of 3,954 samples).

## Joining fragments

A reference surface counts as fragmented if two or more predicted masks each assign at least 70% of their support to that surface. Of 1,462 fixed reference surfaces, 115 met that definition, with 148 excess pieces. All 216 qualifying pairs overlapped by at most 10% of the smaller correctly owned area. On the other 100 views there were 126 such surfaces and 171 excess pieces.

| Method | Fixed surface F1 | Fixed surface matches | Fixed joins, correct / across distinct surfaces / ambiguous | Other 100 surface F1 |
| --- | ---: | ---: | --- | ---: |
| Released masks | 75.56% | 1,093 | — | 70.67% |
| Locally similar colors allowing brightness variation | 63.09% | 789 | 41 / 300 / 51 | 58.09% |
| Also require similar normal, relative depth and a stronger interface | 74.97% | 1,078 | 2 / 13 / 2 | 70.15% |
| Also require a small child surrounded on three sides | 75.56% | 1,093 | 0 / 0 / 0 | 70.69% |

Surface F1 counts one-to-one reference/proposal matches at intersection-over-union of at least 0.50. The enclosure rule merged two fragments on one training Watchpost and recovered 86 matched pixels. The locked rule never fired in the remaining 132 views. [Watchpost comparison](enclosed-train-watchpost3-earth-high-right.png)

Each example below shows a native-pixel crop of the structure in the source image. Orange is the released outline. Cyan in the candidate panel shows a union. The last panel shows actual visible boundaries. Matched pixels are pixels shared by a reference and proposal in a one-to-one surface match.

| Example | Guarded join | Strict outer-edge trim |
| --- | --- | --- |
| [Shopfront](dev-shopfront-plaster-low-left.png) | Fused real faces in the center opening. Lost one match and 952 matched pixels. | No pixels selected. |
| [Courtyard](holdout-courtyard-plaster-low-left.png) | Reunited genuine fragments. Added 151 matched pixels, same number of matches. | Removed 6 real wall pixels and zero background. |
| [Portico](holdout-portico-plaster-low-left.png) | Preserved two excess pieces in this view. | Removed 4 background pixels and zero structure. |
| [Registry](holdout-registry-plaster-high-right.png) | Preserved one excess piece in this view. | No pixels selected. |

## Splitting across a shade step

A whole-mask rule required two smooth shades over at least 90% of the mask and a straight, continuous interface. It fired zero times across 204 views. A local version required a straight source-image step of at least 12 brightness levels with smooth samples on both sides, persisting along at least eight three-pixel scanlines and 75% of the cut inside the mask. It split two fixed masks and gained zero matches (F1 from 75.56% to 75.51%). Four other splits gained one reference match and lost one, with a net loss of 8,612 matched pixels. One [Arcbridge split repaired a crossing](local-step-train-arcbridge3-earth-low-left.png), while a similar [Arcade split broke a good mask](local-step-train-arcade2-cool-low-left.png).

Requiring two different source-estimated normals accepted zero fixed edits and one other edit on Canopy, which added a proposal and recovered zero matches. The pinned dense normal maps were available on 48 regression views. All views could also supply directional evidence from adjacent masks. Most remaining masks did not expose enough of both sides for the tested straight step and normal checks to recover a match.

## Ground and sky

Both outer-edge trims start with the released frame-connected exterior color estimate. They require an unowned exterior sample and a contrasting owned interior sample across a short native-pixel line. Each coordinate is graded once even if multiple masks own it.

| Trim | Fixed background removed | Fixed visible pixels lost | Other background removed | Other visible pixels lost |
| --- | ---: | ---: | ---: | ---: |
| Loose, inside contrast ≥10, outside difference ≤6 | 4,039 | 178 | 3,869 | 433 |
| Tight, inside contrast ≥24, outside difference ≤2 | 3,400 | 104 | 3,450 | 298 |

The tight rule harmed seven fixed views and three others. On one Terrace view it clipped 294 top-face pixels. None of the removed reference-background pixels met the tested blue-sky color criterion. Ground and real platforms can satisfy the same edge test. The files ending in `-trim.png` show which selected pixels were background (green) and visible structure (pink).

The released daylight/blue-cloud detector identified 53.39 million sky pixels on the fixed inputs, and 50.03 million on the other inputs. Zero still occurred inside the final native surface supports. A blanket blue-color rule removed 7,379 reference-background pixels at the cost of 11,155 real fixed structure pixels (18,643 / 34,389 on the other views). Bilateral, exact same-row margin matching also had no eligible support pixels and is sensitive to gradients or clouds in opposite margins.

## Placement and materials

Keep all challengers experimental. The incoming fixed placement baseline is **91.2678% strict relative geometry F1** (20,773 correct, 2,929 missed, 1,046 extra) and **93.4194% facade-priority F1** (18,952 correct, 1,624 missed, 1,046 extra). This surface-only round made no changes to the runtime, counter, exported coordinates or material matching. The placement numbers come from the [incoming decoded-pack audit](../pipeline-consistency-01/README.md).

## Reproduction

Use the registered sibling fixture checkouts documented in the project's offline guide. Each command writes to this review directory. Set `--workers` to control independent frame jobs.

```powershell
python -m benchmark.concrete_shadow_boundaries
python -m benchmark.concrete_boundary_trim
python -m benchmark.concrete_sky_trim
python -m benchmark.concrete_shading_ranges
python -m benchmark.enclosed_shadow_fragment development
python -m benchmark.enclosed_shadow_fragment confirmation
python -m benchmark.concrete_shade_split development
python -m benchmark.concrete_shade_split confirmation
python -m benchmark.concrete_local_shade_step development
python -m benchmark.concrete_local_shade_step confirmation
```

`cases.json` includes the frozen RGB SHA-256, individual join decisions and per-frame native-mask scores. The trim, shading, enclosure and split results use the same frame IDs.
