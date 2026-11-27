# Using one counted face and one thinner neighbor for placement

The current ordinary geometry exporter at `abfc0be` scores **90.7966% strict** and **92.8961% facade-priority F1** on 104 fixed views. A projected cube that puts two faces on two source masks is common at blocks already placed correctly. Among 998 linked existing positions on fixed, 975 are correct and 23 are extras. Among 135 proposed *new* positions, only **two** are correct, one each on Dev Relief 00 and Relief 00. Our strongest image-only rule recovers two blocks on one development Watchpost but adds nothing on fixed, alternate or new images. Keep this geometry change experimental. [Three source images and their projected faces](corner-examples.jpg)

| Fixed 104 method | Correct gained | Correct lost | Extras added | Extras removed | Strict F1 | Facade-priority F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary build | 0 | 0 | 0 | 0 | 90.7966% | 92.8961% |
| Add every eligible new corner | 2 | 0 | 133 | 0 | 90.5363% | 92.5972% |
| Require agreeing source texture on the side and a moderate match on the host | 0 | 0 | 6 | 0 | 90.7846% | 92.8824% |
| Require strong held-out side texture and aligned host texture phase | 0 | 0 | 0 | 0 | 90.7966% | 92.8961% |
| Also permit exchanging a conflicting foreground block | 0 | 0 | 0 | 0 | 90.7966% | 92.8961% |
| Perfect chooser for the new positions, using visible reference answers | 2 | 0 | 0 | 0 | 90.8014% | 92.9014% |

The two useful fixed additions were nominated with both chart axes counted. Both have an overlapping *extra* at the wrong depth. On Dev Relief 00, the proposed cube explains 71% of that block’s currently visible footprint, and the foreground face has the same axis on every tested ray. On Relief 00 the analogous figures are 47% and 93%. Looking for a wrong orientation misses both opportunities. On the plaster Galleries view, one spurious candidate earned 100% host mask coverage, side mask coverage and shared edge votes. A repeating surface can explain the same two source regions at the wrong integer depth. [Per-frame decoded-pack comparisons](cases.json)

| Cohort | Views | Eligible new positions | Correct available to a perfect chooser | Strong texture result |
| --- | ---: | ---: | ---: | --- |
| Development | 72 | 64 | 5 | Two correct new positions on the same Watchpost, zero extras |
| Fixed evaluation | 104 | 135 | 2 | Tied ordinary placement |
| Alternate source images | 20 | 26 | 5 | Tied ordinary placement |
| Additional layouts/views | 4 | 5 | 0 | Tied ordinary placement |
| Shelter and gate | 4 | 6 | 0 | Tied ordinary placement |

### How a face pair is measured

Use the shipped input masks, texture counts, estimated camera and decoded placement pack from the same source image. Try the existing grid origin and offsets of one and two cells along each axis. A candidate contains a broad counted face and a thinner or partly exposed face against a different observed orientation mask. A single face must have the measured chart scale (within 40% along its measured axes). Of the 135 fixed candidate positions, 127 have a host with both counted axes. A partially measured host can reach a texture-selected rule only after a separate atlas match. Sample the two projected masks on opposing sides of their shared edge, along its length. Score both actual source textures against the atlas and competing texture offsets. Existing geometry supplies the visibility test. Reject an addition that would entirely hide an existing visible cell. The broad diagnostic deliberately omits this protection and the atlas threshold. The oracle may select individual members of the candidate pool with the answer key. It gives no permission to add hidden parts.

The strongest additive selection required shared-edge agreement on at least half the samples, at least 76% support on the main face, and visible side support on at least 24% of its sample positions. A partial side texture must correlate at least .80 on held-out texels with a .08 phase advantage, and the host must also beat a shifted texture by .08. A proposed exchange additionally requires at least four rays, coverage of at least 85% of the previous owner’s visible area and a wrong direction on at least half the sampled owner hits. The broad 135-position pool has 113 distinct nearby owner alternatives on fixed, only two of which would exchange an extra for a correct new point. [Implementation and frozen gates](../../thin_face_placement_score.py) · [All source measurements and subsequent reference grades](proposals.zip)

### Can an already placed cube expose an extra?

A second trial keeps the cube and checks whether removing a nearby foreground block would expose its two source-supported faces. On fixed, 18 corner–blocker observations supplied nine different extras that a reference-informed chooser could remove. A liberal rule removed a single real extra on Twinspire 2. A full rerender showed 58 additional samples with the correct face axis and 36 previously correct-axis samples lost. The change occurred on just this fixed view, with no development or transfer removals. A more conservative full-footprint check, preserving covered source pixels and allowing no loss of correct-axis samples, made no changes across 204. Relaxing its direction requirement to four improved samples per lost sample removed three development extras and made no fixed or transfer changes. Keep the removals experimental. [Whole-block reprojection effects and reference checks](blockers.json)

The experiment uses 204 saved complete structures/views and one common translation for each full build when scoring. The 104 fixed strict counts for the strongly guarded add/exchange rule remain **20,619 correct, 3,083 missed and 1,097 extra**. Its historical 112-view score is 90.04%. Results on these previously inspected geometries measure regression behavior. A perfect chooser of these two-face additions cannot close the placement gap.

### Replay

From the experiment checkout with the existing registered fixture caches, run:

```powershell
python -m benchmark.thin_face_placement_run --splits train validation test regression native old fresh extra new shelter --out runs/thin-final
python -m benchmark.thin_face_placement_score
python -m benchmark.thin_face_placement_blockers --splits train validation test regression native old fresh extra new shelter
python -m benchmark.thin_face_placement_examples
```

The `proposals.zip` archive contains one annotated measurement file per input SHA. Labels enter these files after the source-only candidate pass. The scorer reads only source-derived features for its frozen selection. It reconstructs and checks base coordinates from hash-verified packs, compares visible-only references after a whole-build translation, and retains every extra under both priority and strict scoring. [Cohort scores](summary.json)
