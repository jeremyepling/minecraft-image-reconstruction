# Wall texture and silhouette trials

Keep these methods experimental. On the 104 fixed photos, a selective pillar split and diagonal merge gained four matched visible surfaces. The resulting builds still added four hidden cells. All trials used only the supplied screenshot, estimated camera and existing predicted surfaces and counts. Source-world labels grade predictions after export. The placement score allows one common translation for the entire structure.

[Full photographs, original outlines, trial outlines and recorded visible boundaries](texture-seams.jpg) · [Canopy 04](canopy-04.jpg) · [Shopfront](shopfront.jpg) · [Facade](facade.jpg)

Orange denotes the original or retained surface outlines. Green in the trial denotes a mask edited by the candidate. Cyan in the last row denotes recorded visible boundaries. Each column starts with the entire input photograph. The three rows below use the same crop from that photograph.

## On the three reported images

| Photograph | Image evidence and surface result | Result after counting and placing again |
|---|---|---|
| Canopy 04, left | A silhouette continues into a mask where the rear post peeks out beside the foreground post. Horizontal brightness profiles across 208 source rows correlate at 0.97 within a post and 0.07 across the seam. Splitting there, and removing the old horizontal cut, matches 9 of 11 true surfaces, up from 7. An additional 20,537 pixels agree with their matched visible face. | The wide face counts about 0.95 × 4.87 blocks. The narrow rear face has no reliable count on either axis. Both exports place 36 of 60 visible blocks correctly with three extras, 72.73% F1. |
| Facade, oblique right | The diagonal crosses continuous wall texture over 273 sampled boundary pixels, 25° away from the nearest projected wall axis. The median color step is three intensity levels. Combining the two sibling masks adds 31,234 pixels to the correctly matched wall and removes one unmatched mask. Both versions match 12 of 20 surfaces. | The merged chart measures about 27 × 11 blocks. The final export is identical, with 277 of 361 correct and 32 extras, 82.69% F1. Other holes and wall returns still have incorrect boundaries. |
| Shopfront, plaster, low left | Following the line of shade adds 3,047 correctly paired pixels while also extending 1,432 pixels onto background. The trial still leaves a stepped bottom edge at the left. Both versions match 12 of 19 surfaces. | The shade trial recovers three true positions and loses one. Correct placement goes from 103 to 105 of 169, with 21 extras either way, improving 70.31% to 71.19% F1. Rejecting small or thin masks instead removes a genuine matched surface and three correct positions, scoring 68.97%. |

## Which ideas transferred

The first four trials ran on 72 development and 104 fixed images with identical inputs. A correct match requires a unique prediction/reference pair with mask IoU of at least 0.5. Paired pixels count the intersection in those matches. An unmatched predicted mask can be a duplicate, a fragment, a poor outline or a genuine surface below the matching threshold.

| Method | Development surfaces gained / lost | Fixed surfaces gained / lost | Fixed unmatched masks, change | Fixed matched pixels, change | Fixed background pixels, change |
|---|---:|---:|---:|---:|---:|
| Follow a contour through a texture discontinuity | 0 / 1 | 7 / 2 | +15 | −4,540 | 0 |
| Join neighbors across compatible textures | 0 / 2 | 0 / 4 | −8 | +75,812 | 0 |
| Extend the lower edge toward a strong shade transition | 1 / 0 | 0 / 0 | 0 | +16,804 | +52,002 |
| Discard masks under 512 pixels or eight pixels average thickness | 0 / 51 | 0 / 42 | −225 | −30,136 | −44,256 |

The more selective trial required at least 96 supported native rows, same-side correlation of at least 0.85 and cross-seam correlation no higher than 0.20. Its merge required siblings, at least 96 unique border pixels on a straight seam and an angle at least 18° from both projected wall axes. It uses no structure or palette names. The earlier results informed this revision, so its fixed-set improvement is not independent replication. The 20 alternate photos and four additional structures provide a separate check.

| Cohort | Images with edited masks | Matched visible surfaces before → after | Placed F1 before → after | Correct positions gained / lost | Extras added / removed |
|---|---:|---:|---:|---:|---:|
| Development, 72 | 1 | 873 → 873 | 91.213% → 91.213% | 0 / 0 | 0 / 0 |
| Fixed, 104 | 7 | 1,074 → 1,078 | 90.414% → 90.424% | 9 / 4 | 4 / 3 |
| Other views, 20 | 2 | 186 → 188 | 88.554% → 88.543% | 0 / 0 | 1 / 0 |
| Loading dock and switchback, 4 | 0 | 80 → 80 | 83.580% → 83.580% | 0 / 0 | 0 / 0 |
| Additional shelter and gate, 4 | — | Placement check only | 83.474% → 83.474% | 0 / 0 | 0 / 0 |

Canopy 05 and Canopy 18 gained correct surfaces and some visible block placements, but each also gained two hidden cells. On alternate views, the opposite angle of Canopy 04 stayed unchanged. The opposite angle of Canopy 06 and an open canopy each gained a true surface. The open canopy gained one false cell. Splits on a twin-spire and two masonry Shopfront images reduced correctly paired pixels. The Facade diagonal merge fired only on the reported view. All 243 existing canopy opening sightlines stayed clear. These results support trying the seam as a placement hypothesis alongside the original and scoring both against frontmost visible pixels before export.

See the [first-hit projection selector](projection.md) for the follow-up test against the actual recounted exports.

## Replay and evidence

The baseline is `028819a`. Broad edit code is `56dcd46`, and the selective adapter is `97f9870`. The first command grades the broad surface trials. The second includes the alternate views and new structures. The final command uses the shared placement scorer introduced at `b17caa8` and the usual count and placement engines. Use the released visible-build inputs and the frozen production outputs in the sibling `facade-placement-round` checkout.

```powershell
python -m benchmark.wall_texture_evaluate --out runs/wall-surface-audit --workers 4
python -m benchmark.wall_texture_evaluate --selective --out runs/wall-surface-selective --workers 4
cd ../wall-section-round
python -m benchmark.wall_section_eval --candidate-root ../wall-texture-seams --candidate benchmark.wall_texture_trial:selective --out ../wall-texture-seams/runs/placement-selective --workers 4
python -m benchmark.wall_section_eval --candidate-root ../wall-texture-seams --candidate benchmark.wall_texture_trial:selective --transfer-inputs ../framed-depth-transfer/runs/transfer-input --out ../wall-texture-seams/runs/placement-selective-transfer
```

The source hashes, recorded-surface intersections and per-frame operations are in the [first surface audit](cases.json) and [selective audit](selective/cases.json). See the [first summary](summary.json) and [selective summary](selective/summary.json) for pooled counts. The [placement evidence](placements.zip) includes every per-frame score, the candidate and baseline pack hashes, and actual exports for changed cases. The ZIP SHA-256 is `886f5211bf93953ac0327be3dc1590a71d4768cb39b20c5d0ad88f5568d7af71`.
