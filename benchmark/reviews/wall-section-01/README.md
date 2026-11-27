# Wall placement from photographed seams, returns and depth

The baseline is the ordinary Java 26.3 build at `028819a`. It places 20,510 visible blocks correctly, misses 3,192 and places 1,157 extras across 104 complete-structure photographs. Pooled F1 is **90.414%** after one shared translation for the whole build. The block comparisons below use that same translation policy. The reference face IDs and source occupancy appear only in scoring, the oracle and illustrations. Each candidate receives the saved screenshot and the pipeline's estimates.

## What the local chooser could recover

We ran the perfect chooser before fitting a model. It can keep the prior build or select an alternative for each visible wall section. With nominal depth and origin it could recover 134 clean additional fixed-suite blocks before joint occlusion checks. Varying the origin alone raised that to 180, depth alone to 274, and both to 376. Of the latter, 348 survived rendering the chosen sections together, giving **91.248%**. This measures the attainable clean gain of these local additions on pixels left unclaimed by the current reconstruction. A different proposal set or replacing existing placements has a different ceiling.

| Shown photograph | Additional correct blocks with clean oracle | Correct blocks added by local model | Extras added by local model |
| --- | ---: | ---: | ---: |
| Canopy 04 | 1 | 2 | 1 |
| Colonnade 01 | 0 | 0 | 0 |
| Dev Relief 00 | 13 | 3 | 3 |
| Relief 00 | 14 | 3 | 14 |
| Shopfront | 14 | 0 | 11 |
| Facade | 15 | 5 | 0 |

A model that accepts a mixed section can gain more correct blocks on a view than a chooser requiring every addition to be correct. The trained selector gained 129 correct positions and added 102 extras across the fixed suite, including 66 hidden cells. A stricter existing vertical-face model gained one fixed block on Galleries. The latter tied on the alternate angles, new structures and shelter/gate panel.

On occupied walls the perfect one-to-one exchange test could correct 288 wrong placements after joint visibility checks and remove 288 extras. This gives 91.68% at the control translation. The exchange and addition pools can overlap. The strict image-only exchange selector abstained on every view. On the shown Facade the exchange oracle corrected 23 positions, on Shopfront 13 and on Colonnade one. [Grid alternatives, occupied cells and model transfer](../wall-section-grid-01/README.md)

## What the other stages see

- **Canopy 04.** The narrow strip along the foreground pillar follows a different texture from the pillar itself. Splitting the rear strip improves the Surface Finder from seven to nine of eleven reference surfaces. The Block Counter still cannot measure the rear strip's grid. [Annotated pillar seam](../wall-texture-seams-01/canopy-04.jpg)
- **Colonnade 01.** Reprojecting the build into its estimated camera lets us inspect each ray's frontmost block and compare the shape and texture with the photograph. Even with 93.6% silhouette overlap against the reference, block F1 is 75.2%. Sixty-one of 65 missing positions expose only a top face. A test matching the photographed pavement through openings removes three erroneous positions on the shown view and fifteen on the opposite view. It also erases a true nine-block flush pavement platform in a controlled test. [Projected source comparison](../wall-source-projection-01/source-comparison.jpg)
- **Shopfront.** The dark bottom edge helps recover three correct blocks and costs one, raising this view from 70.31% to 71.19%. Following shade also spills 1,432 pixels onto background. Rejecting tiny surfaces outright removes a genuine surface and three correctly placed blocks. [Bottom outline and visible boundaries](../wall-texture-seams-01/shopfront.jpg)
- **Facade.** A diagonal cuts across 273 pixels of relatively continuous wall texture, around 25° off the nearest projected wall axis. Joining its two masks recovers 31,234 correctly assigned wall pixels. The decoded build is identical on this view. [Diagonal and continuous wall texture](../wall-texture-seams-01/facade.jpg)

The selective split/merge rule gained four fixed-suite surface matches and lost none. Recounting and placing those proposals gained nine true blocks and lost four, and introduced four hidden extras. We replayed both exports through an image-only decision rule comparing the first surfaces hit by camera rays. Fitted texture error fell for both harmful fixed-suite edits, but visible face orientation worsened. Requiring the outline, orientation and withheld texture measurements to support the edit rejected all three changed proposals across the 200 views and reproduced the default. [Surface and placement comparisons](../wall-texture-seams-01/README.md)

## Where the screenshot gives a depth clue

For each missed block with a photographed vertical face, we counted whether its face shares at least four native pixel edges with a perpendicular face of the same block or an immediately adjacent block. Separately, we counted contacts with a visible face of another material. A color or material change can mark neighboring tiles on a flat wall. A return face provides an additional orientation constraint. Counts can overlap. [Annotated source details](contacts/contacts.jpg) · [Per-frame counts and face IDs](contacts/contacts.json)

| Shown photograph | Missed wall blocks | Touch a return | Touch two or more returns | Touch a different material |
| --- | ---: | ---: | ---: | ---: |
| Dev Relief 00 | 96 | 24 | 8 | 26 |
| Relief 00 | 83 | 7 | 3 | 26 |
| Facade | 74 | 0 | 0 | 8 |
| Canopy 04 | 5 | 0 | 0 | 0 |
| Colonnade 01 | 4 | 2 | 0 | 0 |
| Shopfront | 42 | 10 | 4 | 30 |

Across the fixed suite 180 of 1,699 missed wall blocks touch a qualifying return, 89 touch two or more, and 722 touch a material change. The direct-contact count excludes return cues elsewhere on the same wall. The annotated photos include the highest-ranked qualifying examples for the six reported views.

## Which return rules transferred

Each proposed cell must project onto photographed pixels. Placement trials used the native atlas, the estimated camera and a joint visibility pass. We compared four ways of finding a wall from those cues on the same 200 views and on four further shelter/gate photos. Counts below refer to paired changes in the decoded export on the fixed 104. [All measurements](results.json) · [Paired manifests and changed exports](comparisons.zip)

| Visible evidence requested | Correct blocks added | Extra blocks added | Of those, hidden |
| --- | ---: | ---: | ---: |
| Wall and differently oriented return both match the atlas | 3 | 0 | 0 |
| One face matches the atlas and a second has the expected orientation | 7 | 3 | 1 |
| Two placed returns bound a strip with matching front tiles | 0 | 3 | 0 |
| One placed return adjoins a matching front tile | 8 | 2 | 1 |

The strict two-face rule recovered one block on the shown Facade, and none on either shown Relief photograph. It had no effect on the 72 development, 20 alternate, four new or four shelter/gate photographs. The single-return strip added seven correct blocks and eight extras on development. The two-return rule added ten extras there. Bracketing a recess with two edges supplies a depth hypothesis; extending the front still needs independent evidence for each visible block.

An additional variant lets source-image edges nominate walls even where a surface mask misses a small return. Both differently oriented faces must match projected textures, with independently measured phase and neighboring-depth margins. It recovered 16 fixed blocks and added 19 extras, three hidden. Canopy 04 gained one clean block and Colonnade 01 gained three correct and one hidden; Shopfront gained two extras, one hidden. Neither Relief example changed. The development set gained seven correct and eleven extras, alternate angles gained seven correct and two extras, the four new views gained one extra and shelter/gate tied. [Paired cohort totals](texture-edge.json) · [Per-frame evidence and changed exports](texture-edge-comparison.zip)

## Recommendation

Keep the current production geometry at 90.414%. The largest directly demonstrated wall opportunity is choosing among depths for adjacent visible front and return faces that the current build already covers. For the next test, let a visible post or recess compete at several depths while holding the surrounding wall fixed. Score only rays whose first-hit block changes, including the narrow side, the texture alignment on both sides, and any newly exposed background. On Colonnade 01 the 61 top-only misses warrant a separate visible-roof test. [First-hit and depth controls](../wall-source-projection-01/README.md) · [Full-export projection gate](../wall-texture-seams-01/projection.md)

## Replay

Run from a checkout with the saved releases and calibrated atlas restored. The evaluator verifies input hashes and the baseline export before running the candidate. Set `--transfer-inputs ../framed-depth-transfer/runs/transfer-input` for the shelter/gate cohort.

```powershell
python -m benchmark.wall_section_eval --out runs/control --workers 4
python -m benchmark.wall_section_eval --candidate engine.visible_return_faces:recover --out runs/returns-recover --workers 4
python -m benchmark.wall_return_audit --run runs/control --out benchmark/reviews/wall-section-01/contacts --workers 4
```
