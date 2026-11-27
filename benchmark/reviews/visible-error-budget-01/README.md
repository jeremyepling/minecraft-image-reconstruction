# Visible posts, recesses and the remaining placement errors

The ordinary Java 26.3 placer recovered **20,535 of 23,702 visible blocks** across 104 whole-structure views. It gained 25 correct positions, lost none, removed 11 extras including two hidden cells, and introduced no extras. Seven views improved and 97 tied. Relative geometry F1 went from **90.414% to 90.496%**. Scoring allows one shared world translation for all parts. The camera-registered score on these views is also 90.496%. [Explore the six reported issues in the source photos](shipped-regions/viewer.html) · [All cohorts and per-method results](results.json)

## What changed in ordinary builds

**Measure a partly visible pillar from its repeating block rows.** When a photographed vertical seam separates two textures, fit the atlas on alternating block courses and test it on the courses left out. A new row can enter only if the two sides support distinct depths and its own visible face has enough source pixels after joint rendering. This recovered five blocks on Canopy 04, five on Canopy 18 and four on Canopy 05. Two alternate canopy views gained eight more. On Canopy 04, 733 of 757 sampled rays on the narrow side now hit the correct block, up from 190. The foreground pillar remained at 4,638 of 4,650. The pictured build went from 72.73% to 78.85%. [Partial tiles and silhouette tolerances](../narrow-post-rows-01/README.md)

**Move a small connected panel when its texture and visible surfaces support the new depth.** A narrow return or a texture boundary proposes a local patch. Test nearby placements against the original photo and require an independent atlas witness at every destination. A competing texture match on the original geometry triggers an additional per-block comparison of orientation, silhouette and held-out texture. This moved 11 blocks correctly and removed 11 extras. On Dev Relief 00 the two-block change repaired 227 of 228 sampled rays in the first detail. The second detail remained at zero of 196. Other gains were on Twinspire, Portico and a masonry Shopfront. CPU cost for this step ranged from about 0.25 to 2.39 seconds on four measured examples. [Local depth decisions and Dev Relief detail](../local-panel-depth-01/README.md)

**Keep the photographed roof strip as an experiment.** Fitting the upper silhouette and checking each visible tile between measured roof contacts recovered two blocks on each view of Colonnade 01. The representative view went from 75.22% to 75.96% under this experiment. The same rule added three false blocks on other views. The shipped default still scores 75.22% on that view. [Colonnade overlays and roof trials](../visible-roof-skyline-01/README.md)

## How the reported issues guide subsequent tests

The six views and their source regions are named regression cases. For comparable frozen evaluation exports, run the scorecard below. It checks the exact first block along every second native image row and column inside each region and counts correctly clear background where applicable. This local diagnostic keeps the estimated camera fixed. Overall build F1 still grants its common translation. Truth defines the frozen regions and grades the result after export; the placer receives the single photograph and the ordinary upstream estimates.

| Reported symptom | Local check to carry forward |
| --- | --- |
| Canopy 04 pillar overlap | Score the narrow rear strip separately from the wide foreground pillar. A correct depth change should improve the rear while preserving the front. |
| Dev Relief 00 and Relief 00 | Score each photographed protrusion or recess individually. A return suggests where to test alternate depths. A change of material can also occur on a flat wall. |
| Colonnade 01 | Measure the visible top and the pavement in the openings separately. Preserve real flush platforms that share the pavement texture. |
| Shopfront | Separate the real lower edge from the background just outside it. Preserve genuine tiny surfaces. The foreground plaster example still ties. |
| Facade | Check whether removing a false diagonal improves assigned pixels, and separately whether the correct front block appears at that depth. The shown Facade still ties. |

The inspected examples informed design, including the two-pixel post tolerance and the Dev Relief texture ambiguity rule. The counts on these examples measure regression behavior. They do not establish transfer to unseen building families. [Baseline first-hit probes](scorecard/cases.json) · [Challenger probes](scorecard/trials.json) · [Shipped probes and source hashes](shipped-regions/cases.json)

## Biggest remaining problems

Of the **3,167 visible blocks still missed**, the projected build leaves most of the footprint uncovered for **1,838**. For **661** misses the projected block generally has the right face axis at the wrong coordinate; for **652** it has a different face axis. The remaining **16** tie on coverage or face-axis votes. A block is counted once in this breakdown. The new build has 1,146 extras, including 104 in hidden occupied space. [Native-pixel audit of the shipped result](remaining.json) · [Pre-change counts and definitions](baseline.md)

1. **Recover horizontal faces inside already detected regions.** Of those 1,838 mostly uncovered misses, 1,386 already have most of their pixels inside a detected mask. For 632 top-only or underside-only blocks, the correct horizontal orientation fails majority coverage. Test local wall, top and underside interpretations on the same strip, then recover the visible extent from the projected rim and atlas. Score top and underside separately and preserve the openings. This includes thin roofs and the underside of overhangs. The skyline trial only addresses a subset facing clear sky.
2. **Improve local depth selection.** On 590 of the 661 same-axis misses, extras occupy most of the covering pixels. These are candidates for replacing a bad post or panel. Check recesses and protrusions in both directions, and keep any source block that still has its own visible support. The perfect one-panel chooser recovered 92 positions from the pre-change pool. Future depth tests should improve the proposal set as well as its selector.
3. **Recover block counts and extents on correctly oriented regions.** Among the uncovered misses, 481 have majority correct-axis mask coverage without both measured dimensions there; a further 113 have dimensions but lack full grid registration. A partial grid can still lead to a correct building block. Test partial courses and tile transitions at several origins at fixed depth, and score the final relative placement.

Horizontal-only blocks account for 985 top-only and 508 underside-only misses across all categories. Only 427 of the 3,167 misses have fewer than 16 visible source pixels. Dropping tiny surfaces would leave most errors and would discard legitimate evidence. The fixed set has working cameras on all 104 views. Eighteen separate side and elevated views still stop at camera estimation and remain a distinct generalization problem.

## Transfer and verification

Twenty alternate views improved from 88.554% to 88.665%. The 72 development photos, four new loading/switchback views, four independent shelter/gate views, four additional native canopy views and 18 side/elevated camera-challenge views tied. Ten fixed canopy probes retain exact opening widths and all 137 sampled clear rays. The 112-view historical score is 89.737%, counting the eight retired full-frame textures with their frozen abstentions. The build training and ordinary structure evaluation continue to exclude those retired frames.

The combined experiment and ordinary geometry entry point produced equivalent decoded coordinates and materials on all 226 exports. The ordinary screenshot command reproduced both the Canopy 04 correction and the Dev Relief 00 correction. [Verification and capture hashes](verification.json) · [Paired manifests and actual changed exports](comparisons.zip)

```powershell
python -m benchmark.wall_section_eval --candidate benchmark.post_panel_comparison:combined --out runs/primary-combined --workers 6
python -m benchmark.wall_section_eval --candidate benchmark.framed_geometry_order:production --out runs/primary-production --workers 6
python -m benchmark.visible_error_budget --run runs/primary-production --out runs/remaining.json
python -m benchmark.visible_issue_scorecard --run runs/primary-production --out runs/issue-viewer --viewer
```

The release checker reads matching transfer, native, side and elevated runs, the two CLI builds and the scored pillar-tolerance runs. Repeat the respective cohort commands in the linked experiment reports and run `python -m benchmark.visible_issue_release` to verify and collect the release archive.
