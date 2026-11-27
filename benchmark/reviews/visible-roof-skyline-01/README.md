# Photographed roof edges and thin top strips

The roof on the featured Colonnade is visible in a thin run of source pixels above the stone-brick front. Measuring its upper silhouette and requiring nearby measured roof tiles on both sides recovered **two visible blocks on each of its two views**. The right view rose from 75.22% to 75.96% visible-block F1 and the left from 65.92% to 66.67%. Neither view gained an incorrect block. The same rule added three false blocks on other views, including two angles of a recessed structure. Keep these placement rules experimental. [Full photos, skyline and enlarged source-pixel details](colonnade.jpg)

## How the photograph constrains the roof

Use the exterior-connected sky and cloud estimate from the screenshot. For each height with at least three already placed and visible roof tiles below the estimated camera, intersect the native upper silhouette with the horizontal plane and test the projected integer back edges. An edge needs a sequence of at least two adjacent block intervals and three native skyline columns per interval within 1.25 pixels of the projection. Search the two courses nearest that measured edge. Only candidate tiles showing actual image pixels against the current render can enter the pool. Joint frontmost rendering checks each selected tile, rejects newly covered sky, and preserves visible evidence for all prior blocks. The test covers upper silhouettes meeting the sky. Interior top surfaces, undersides, and roofs without a measured initial top course need other proposals.

The choices below were frozen before running each variant. The first rule was frozen before inspecting holdouts. Later variants used previously reviewed failure modes, so the 104 views serve as regressions. The fixed score covers all 104 whole-structure frames and allows one common translation per reconstructed building. Each cell in the cohort columns gives **additional true / additional false positions** relative to the ordinary build at `bec7c94`. No variant removed a prior position. The four additional shelter/gate views tied in every row.

| Additional evidence | Development 72 | Fixed 104 | Alternate angles 20 | New 4 | Fixed F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source tile repeats in a projected neighboring tile | 5 / 15 | 17 / 1 | 9 / 2 | 0 / 0 | 90.453% |
| Repeat plus newly explained pixels, extent bounds and visible-side check | 3 / 0 | 2 / 1 | 1 / 1 | 0 / 0 | 90.417% |
| Missing tile bracketed on one axis by measured roof tiles; fresh top pixels | 0 / 0 | 2 / 1 | 2 / 2 | 0 / 0 | 90.417% |
| Texture pooled from two disjoint sets of measured roof tiles | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 90.417% |
| Repeat on a dominant roof spanning separated columns | 0 / 0 | 7 / 1 | 3 / 2 | 0 / 0 | 90.429% |

The first rule increased the count of roof blocks correctly located relative to both pillars from 346 to 362 of 459 on the ten fixed canopy views; all 137 measured clear rays remained open. It also introduced a hidden position on the alternate Canopy 04 view. The separated-column rule repeated that error. The other rules added zero hidden positions across these cohorts. On the 112-image historical comparison, including the eight retired texture-only controls with 384 additional old targets, the first rule scores 89.69% and the control 89.66%. The fixed control scores 90.414%. Current production keeps its existing geometry.

## Coverage with the answers supplied

The image-only generator proposed 502 new positions on the fixed suite. Of those, 162 coincide with a missed visible position under the control's common translation. Selecting only wholly correct strips and rendering them together recovered **132** clean blocks. Individually selecting the correct cells and applying the same joint frontmost visibility check recovered **150**. The corresponding development figures were 40 and 53. On alternate angles they were 43 and 47. The four new views had no correctly proposed cells.

On the shown Colonnade right view, nine of 13 nominated cells were correct and five survived joint checking when offered individually. On the opposite angle 17 of 18 were correct and twelve survived. The pictured Canopy 04 had five correct candidates, three surviving joint checking, and zero additions from the image-only rules. A barely visible rim gives a substantially better local search region. It cannot establish the entire rear span from this photograph. [Results and cohort totals](results.json) · [Per-image oracle, baseline hashes and changed exports](comparisons.zip)

## Replay

Run these commands in the `experiment/visible-roof-runs` worktree against the saved 26.3 inputs. The canonical control is `../wall-section-round/runs/control`. Frozen code revisions are 803629e (neighbor texture), 527ba59 (strict and bracket), 4df46ff (pooled texture) and 36c9670 (separated supports). The evaluator verifies the source, prior pack and recorded geometry hashes; reference data enters only scoring and the labeled oracle.

```powershell
$methods=@{skyline='recover';strict='strict';bracket='bracket';pooled='pooled';overhang='overhang'}
foreach ($run in $methods.Keys) {
    $adapter='engine.visible_roof_skyline:'+$methods[$run]
    python -m benchmark.wall_section_eval --candidate $adapter --out "runs/$run" --workers 4
    python -m benchmark.wall_section_eval --candidate $adapter --transfer-inputs ../framed-depth-transfer/runs/transfer-input --out "runs/transfer-$run"
}
python -m benchmark.visible_skyline_opportunity --out runs/opportunity.json --workers 4
python -m benchmark.visible_skyline_report
```
