# Feeding visible material evidence back to the Block Counter

This Java 26.3 experiment starts at reconstruction revision `5818a37` and uses its exported block IDs, estimated camera and current source masks. The first 72 development views set the rules. The candidate definitions were committed at `414b98c` before grading the other 132 views. The 104 fixed views and the other 28 controls retain their existing building groups.

## What the second pass measures

1. Consider a wall with a sampling chart where the ordinary counter left at least one dimension unresolved. Project the final cubes through the estimated camera. Require a single visible wall plane to cover at least 70% of the wall mask.
2. Back-project that mask onto the plane. Its outer extent on each axis must land within 0.28 blocks of an integer boundary. Preserve every dimension already measured.
3. For each supported frontmost face, use its **exported ID** to look up allowed side sprites. Compare source texels with those sprites at the proposed block scale and at half and double that scale. Allow orientation, one-texel shifts and reduced texture resolution. A cube votes only if the unit-scale correlation reaches 0.46 and exceeds the alternative scales by at least 0.085.
4. Compare accepting the plane alone, requiring votes from at least two cubes and half of informative cubes, requiring both ends with at least 60% of the intermediate offsets on each accepted axis, and requiring a vote at **all four corners** of the proposed extent. The ends rule can report one supported axis. No challenger adds a cube to the observed material map.

The scorer matches detected masks to native reference surfaces one-to-one at intersection over union ≥0.5. For each matched visible wall, the target width and height span the extreme observed Minecraft face coordinates, including visible blocks with partial faces. The native face chart gives the across/down ordering. Unmatched proposals and wrong-orientation proposals are counted separately. A predicted width and height describe a wall's span; an opening within that span can still be empty.

## Measured wall counts

On the 104 fixed views, **878 matched wall charts** were eligible for measurement. The ordinary counter already returned both dimensions for 426 and both were exact for 287. The stricter four-corner rule returned both for 496, including 349 exact pairs. Across the 28 independent controls the exact count rose from 66 to 88 among 262 matched wall charts.

| Feedback required | Fixed added correct pairs | Fixed added wrong pairs | Fixed unpaired or wrong direction | Other 28 added correct / wrong / unpaired |
| --- | ---: | ---: | ---: | --- |
| Coplanar final cubes | 185 | 56 | 24 | 68 / 13 / 7 |
| Material supports unit scale | 137 | 35 | 4 | 48 / 11 / 2 |
| Verified end cells per axis | 90 | 12 | 3 | 36 / 5 / 2 |
| Four verified corners | 62 | 8 | 2 | 22 / 0 / 1 |

On the fixed views, four verified corners added **61 correct widths and 3 wrong widths**, plus **59 correct heights and 5 wrong heights**. The per-axis end check added **117 correct widths and 7 wrong**, and **102 correct heights and 13 wrong**. Fifty-five matched proposals from the end check supplied just one new axis and still lacked the other. On development four corners added 41 correct pairs, 2 incorrect, and 3 unpaired. Every new measurement fills a previously unresolved axis.

Shopfront shows the limitation. On the masonry high-right view the feedback correctly measures a 1 × 5 wall. On the plaster low-left view it completes a 1 × 5 chart whose matched reference spans 3 × 6. Visible texture on the four corners of a fragment can be genuine even when its mask stops short of the full surface. In Crossing's plaster high-right view, the ends confirm the correct height of 3 on `mask-6` while withholding a proposed width of 4. The reference width there is 3. [Shopfront source overlays](shopfront-counts.jpg)

[Development scores](development-summary.json) · [Fixed and independent scores](confirmation-summary.json) · [Development surface records](development-cases.json) · [Fixed and independent surface records](confirmation-cases.json)

## Replay

Run `python -m benchmark.material_feedback_grid --set development --workers 5`, then repeat with `--set confirmation`. The runner uses the published typed sites, the frozen stage replays, and native visible face-ID maps. `python -m benchmark.material_feedback_grid_sheet` recreates the source comparison.

## What happens when the placer gets the correct counts?

Five ordinary builds exercised the complete Java 26.3 entry point on **Cloister 1 high-right, Twinspire 1 high-right, Shopfront plaster low-left, Canopy 04 and Colonnade 01**. These are examples from the fixed suite. Each source was processed five times:

1. Current released build.
2. A no-op replay to test input and export consistency.
3. The source-only four-corner count feedback, using the first pass's actual exported IDs.
4. An oracle giving the exact visible dimensions of **every paired, charted wall** with a missing or incorrect count. The runner computes the chart spacing corresponding to the target visible mask extent. It preserves the ordinary camera, mask, chart corners, grid phase, depth proposals, material ranker and placement refinements. The reference enters only this diagnostic arm and the scorer.
5. A removal test that suppresses raw across/down spacings for every measured chart handed to primary placement. The remainder of the pipeline keeps the ordinary chart corners, phase, recorded dimensions, face footprints, previous surface checks and any later internal placement calls. This isolates the raw spacing handoff.

The runner intercepts only the primary placement call. The ordinary downstream checks can make their own counterfactual placement calls. They keep their own inputs. The five controls match the previously published typed packs at every coordinate and ID. Each arm has the same recorded public mask and count-file hashes up to the intervention. The results also check the estimated camera and decode the Minecraft template outputs.

| Five-view result | Current | Four-corner feedback | True wall dimensions |
| --- | ---: | ---: | ---: |
| Wall count rows changed | 0 | 2 | 28 |
| Strict placement F1 (correct / missed / extra) | 85.52% (644 / 176 / 42) | 85.52% | 85.52% |
| Facade placement F1 | 91.17% | 91.17% | 91.17% |
| Exact IDs at correctly placed coordinates | 627 / 644 | 627 / 644 | 627 / 644 |
| Overall strict F1 | 83.27% | 83.27% | 83.27% |
| Overall facade F1 | 88.74% | 88.74% | 88.74% |

The final cells and block IDs matched at every coordinate in these runs. On Colonnade the seven corrected wall counts still produced zero cells from the strict chart-lifting path; the alternative placement stages supplied the build. A count gives block spacing along the detected wall, while a reliable carrier also needs the chart to lift to an integer plane and agree with the visible support. These five views establish a local count-only comparison. They do not bound the result of also correcting the chart, the boundary or plane depth, or establish a ceiling across the 104 views.

The removal arm suppressed **30 measured chart rows** (25 complete spacing pairs and five partial measurements) across all detected surface directions. Strict chart-derived cells fell from 68 to zero on Twinspire and from 10 to zero on Cloister. The ordinary alternative placement paths recovered the same final positions. Across all five builds the removal produced exactly the original Minecraft IDs and coordinates. Geometry remained 85.52% strict and 91.17% facade, material accuracy 97.36%, and overall F1 83.27% strict and 88.74% facade. This is a direct removal test for one handoff. Earlier counting and later count consumers still ran.

**Decision.** Keep the feedback experimental. The four-corner check supplies useful missing count measurements, but it can confidently measure an undersized fragment. The actual five-view oracle gave zero placement gain. I would test the detected extent and depth together with the material-confirmed grid before feeding back another count. [Measured interventions per surface and decoded scores](count-only-oracle.json) · [Ordinary exported packs for all five arms](count-only-exports.zip)

To reproduce a single source, set `$env:MCIR_COUNT_ORACLE_PYTHON` to the local Python executable and use:

```powershell
.\Invoke-WallRecovery.ps1 -Image $source -Out $privateOutput -Python (Resolve-Path benchmark/CountOraclePython.ps1).Path
python -m benchmark.material_count_oracle_summary $outputCloister $outputTwinspire $outputShopfront $outputCanopy $outputColonnade
```

