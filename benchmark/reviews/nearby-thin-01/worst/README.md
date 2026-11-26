# Lowest-scoring reconstruction images

The close-up thumbnails show the entire 1440 × 900 source photographs, scaled to fit. They are texture-only controls. On the 104 fixed structure views, Canopy 06 left is the lowest-scoring image and pooled F1 is 88.22%. The table below also identifies the weakest alternate views. [Structure and full-suite scores](../structure-scope.json)

Eight of the 112 fixed evaluation rows tied for last. All eight depict a close-up wall and returned an empty pack for 48 visible blocks, or 0% F1. They contain four distinct RGB images because both photographed floor variants are fully outside the frame. All 20 alternate views produced nonempty geometry. [All eight source views](camera-abstentions.jpg)

| Evaluation and view | Relative F1 | Correct / visible | Missed | Extra |
|---|---:|---:|---:|---:|
| Fixed · Canopy 06 left | 28.29% | 29 / 84 | 55 | 92 |
| Fixed · Canopy 04 left | 69.39% | 34 / 60 | 26 | 4 |
| Fixed · Shopfront plaster low-left, light gray concrete floor | 70.07% | 103 / 169 | 66 | 22 |
| Alternate · Colonnade 01 oblique-left | 59.49% | 105 / 187 | 82 | 61 |
| Alternate · Canopy 00 left | 65.91% | 29 / 55 | 26 | 4 |

The weakest nonempty fixed build is Canopy 06 left. It placed six of 60 visible roof-level blocks correctly. Its photograph hides most of the error: the exported cells span 62 blocks along Z, while the visible reference spans ten. The top-down diagram shows both sets on a shared grid with the same alignment used for scoring. [Top-down footprint](worst-footprint.png) · [Three weakest nonempty fixed views](fixed.jpg) · [Two weakest alternate views](alternate.jpg)

The source and overlay use the same crop. Green marks the correct block coordinates, red marks extra coordinates and blue marks missed visible coordinates. Only the nearest projected face is drawn at each pixel; the numbers count all scored cells. Each overlay is rendered from the decoded pack after applying its single scoring translation. [Complete rankings and source hashes](ranking.json)

To reproduce the sheets, run `python -m benchmark.nearby_worst_views --predictions runs/release --out benchmark/reviews/nearby-thin-01/worst` using the frozen source fixtures and the final eight cohort folders.
