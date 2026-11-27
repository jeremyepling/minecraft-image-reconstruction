# Can a row of shared cube edges fix the outline?

The final shared-edge check corrected **80 source pixels across eight of 204 images**. It also transferred **13 pixels to the wrong wall** in two images. All eight recounted and rebuilt exports preserved the original geometry. Keep this rule experimental. Agreement between several projected edges can identify a real direction change while still choosing the wrong parallel wall behind it. [Source images and pixel comparison](comparison.jpg) · [All cases](ray-cases.json)

On **Canopy 04 left**, four blocks agree along the visible edge and 16 pixels move to their correct owner. On **Stepped Hall 2**, four blocks also agree, both faces pass atlas checks and five native rays at each disputed pixel hit the proposed row first in the inferred build. Four pixels at source x = 879, y = 704–707 still belong to reference surface 18. Its normal points along +X like the proposed surface 16, but its visible blocks are one block forward along X. A ray cast into an incomplete reconstruction can miss the true occluding face. Nine pixels on **Canopy 06 left** show the other wrong-owner transfer. These 13 pixels were already wrong before refinement. Each moved to a different wrong owner, slightly lowering one surface’s IoU in each source image.

| Fixed 104 source images | Changed images | Pixels corrected | Previously correct pixels harmed | Pixels transferred to another wrong owner | Individual surface IoUs decreased |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two consecutive block edges with source-gradient votes | 86 | 28,960 | 6,870 | 311 | 57 |
| Four strong consecutive votes | 70 | 18,358 | 5,118 | 214 | 35 |
| Four votes, interior color and topology checks | 53 | 1,687 | 0 | 9 | 1 |
| Also preserve both affected surfaces’ integer count tuples | 49 | 1,550 | 0 | 9 | 1 |
| Both-face atlas and native first-hit checks, stable grids and samples | 4 | 26 | 0 | 9 | 1 |

**Where the votes come from.** The estimated camera reprojects existing cube coordinates. A host surface must have both block dimensions measured and cover at least five of six interior samples from the projected face. A second mask with a perpendicular direction must cover at least four samples of the narrower face. Adjacent cubes on one straight edge must independently favor a common source-image contrast step on two disjoint sets of samples. On the fixed set, 1,265 such block observations formed 304 consecutive runs. Source contrast produced 226 candidate corrections before the stricter checks. Only observed, uniquely owned surface pixels can transfer. The outline union, sky, holes in the union and background stay fixed.

The protected rule requires at least four consecutive edges, a small offset, clear contrast, topology-preserving moves and source colors at least four times closer to the receiving side, 1.5 or more pixels inside its proposed border. It fixes substantial mask ownership but sometimes changes a correct integer count. On the masonry Portico high-right view a correct 1 × 5 count on a wall with 98.9% mask purity becomes unresolved. Reverting changes that alter integer counts avoids that particular loss. Re-running placement on the remaining fixed changes gains two correct blocks and loses two others, all optional under facade-priority scoring.

**Independent corroboration.** After four wrong-owner pixels survived the development run, a final variant used the placement experiment’s two-face witnesses. Every block in the run had to pass both texture checks, agree with both counted axes to within 20%, and show at least 60% visibility on both faces. For every disputed pixel and its four native neighbors, a separate exact cube-ray intersection had to see a member of the same run. The affected count tuples, grid origins, sampled tile footprints and material selections had to agree after ordinary and atlas recounts. The allowed fractional count drift is at most 0.01 repeat across the entire crop. When either recount fails the guard, the source masks stay as they were. The 104 fixed images are previously reviewed regression views. [Final source-only witness inputs](two-face-witnesses.zip) · [Rejected recounts and every transfer](ray-cases.json)

| Cohort | New correctly owned pixels | Other wrong-owner transfers | Changed exports |
| --- | ---: | ---: | ---: |
| 72 development | 54 | 4 | 0 |
| 104 fixed | 26 | 9 | 0 |
| 20 alternate, four new-building, four shelter/gate | 0 | 0 | 0 |

**Placement after the final experiment** remains **90.80% strict F1 and 92.90% facade-priority F1**. The fixed result is 20,619 correct, 3,083 missed and 1,097 extra. The historical 112-view F1 is 90.04%. All changed candidate exports were checked by recomputing their control and candidate coordinate sets under the same single-image camera and comparing to the decoded release. The final variant never changed a coordinate, gained or lost a matched surface, or assigned a new background pixel. The broader two- and three-vote variants were evaluated as surface proposals. [Per-cohort measurements](summary.json) · [All initial edge votes](cases.json)

## Replay

Use the Java 26.3 frozen fixtures and atlas in the registered worktrees, with the `abfc0be` default. The two-face witness archive contains only source-derived proposal features plus source and export fingerprints. Original candidate hashes identify the paired placement replay. The evaluator opens source-world labels after producing both the proposed ownership and the witnesses. The 72 development views supplied the guard checks. Repeated floor, palette and camera views keep their original cohorts.

```powershell
python -m benchmark.edge_vote_outline_eval --splits train --out runs/freeze-dev.json --geometry protected
python -m benchmark.edge_vote_outline_eval --splits validation test regression native old fresh extra new shelter --out runs/freeze-held.json --geometry protected
python -m benchmark.edge_vote_outline_corroborated --out runs/ray-final.json --geometry
python -m benchmark.edge_vote_outline_summary runs/freeze-dev.json runs/freeze-held.json --ray runs/ray-final.json
python -m benchmark.edge_vote_outline_sheet 'ray|shape-benchmark-26.3--train-canopy-04-left|Canopy 04' 'ray|train-steppedhall2-earth-high-right|Stepped Hall 2'
```
