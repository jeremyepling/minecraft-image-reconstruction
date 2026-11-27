# Checking a front cube against the textured plane behind it

The frozen 26.3 default at `5818a37` supplies the visible cubes, projected camera and inferred block IDs. This experiment projects those cubes back into the source image and asks whether removing one would reveal a better-fitting part of an already visible surface. Recommend keeping automatic deletion experimental. The check identifies real extras in Portico and several other fixed views, but it removed genuine blocks in Shopfront and on separate decorative structures.

[Compare Portico and Shopfront in their source images](portico-shopfront.jpg)

## What the challenger sees

1. Find a front cube whose removal would reveal other exported cubes behind at least 95% of its visible projection. Both front and rear material IDs must have source evidence. Require a visible witness for the rear plane and enough samples to compare both interpretations.
2. For each ID, plane and face direction, fit the direction-eligible vanilla atlas sprites to alternating 3 × 3 patches of the subsampled source image, allowing rotation, mip level and local brightness. Test both predictions on the remaining patches. This pixel split checks the texture fit. The frozen block IDs themselves already used the source image.
3. Up to one cube per image can be removed. This avoids combining independent removals that rely on each other. The selected development rule requires at least 16 test samples, a mean RGB error decrease of 10 levels and a rear fit winning at least 65% of the samples. Select the highest qualifying improvement. No new coordinates or hidden support are created.
4. Rerun the default Material Matcher for all retained cubes on changed frames, write and decode their Minecraft templates, then recheck whole-build alignment from geometry. All 29 changed established views kept the original best translation. The recorded camera and world labels enter only in grading.

## How much placement could this address?

Among 1,076 fixed candidates, the visible reference labeled 117 as extras, 889 as required and 70 as optional visible positions. Forty views had an eligible extra. With one edit per view, a perfect chooser that may keep the baseline could remove **40 extras**, lifting strict placement F1 from 91.2678% to 91.3480% and facade-priority F1 from 93.4194% to 93.5116%. This bound holds at the frozen alignment. Deletion cannot recover an unproposed missing position. [Reference-only oracle](oracle.json)

Twenty-four interpretable rules were compared on the 72 development views. The selected rule removed six extras and one required block. Commit `d5e29f1` freezes the candidate and train settings before evaluating the fixed views. [Development settings and tradeoffs](train.json)

| Views | Extras removed / real blocks lost | Strict placement F1 | Facade placement F1 | Strict position + ID F1 | Facade position + ID F1 |
|---|---:|---:|---:|---:|---:|
| Fixed 104 | 16 / 1 | 91.2678 → 91.2975% | 93.4194 → 93.4537% | 84.6818 → 84.7486% | 88.5099 → 88.5766% |
| Other 100, including development | 10 / 2 | 90.8906 → 90.9056% | 93.1713 → 93.1893% | 88.3856 → 88.4042% | 91.4310 → 91.4534% |
| Other 28, excluding development | 4 / 1 | 89.1553 → 89.1775% | 89.4913 → 89.5189% | 86.7935 → 86.8326% | 88.2419 → 88.2911% |
| Separate four decorative development views | 0 / 2 | 48.3977 → 48.2730% | 50.0000 → 49.8676% | 41.5776 → 41.4474% | 42.6808 → 42.5419% |
| Separate four decorative confirmation views | 0 / 1 | 60.3550 → 60.3034% | 62.4224 → 62.3689% | 45.5621 → 45.5050% | 47.5155 → 47.4563% |

The fixed candidate ended with 20,772 correct positions, 2,930 misses and 1,030 extras. Exact material at the correct positions reached 19,282/20,772 (92.83%). Changed geometry also changed material votes for other visible cubes. On the decorative check it removed one real Hillside Chapel position even though the rear interpretation fit the sampled colors much better. Repeated materials, cast shading and imperfect camera registration can all make that photometric decision ambiguous. One empty geometry export in each decorative partition remains in the pooled denominator. The historical 112-view placement score of 90.5043% belongs to the original geometry release and was not rerun. [Established cases and hashes](cases.json) · [Pooled metrics](summary.json) · [Decorative confirmation](decorative.json)

## Which evidence mattered?

With the same candidate support, train-selected thresholds and one-edit allowance, plain plane color removed 19 fixed extras and six required cubes. Grouping colors by the final material ID removed 12 extras and two required cubes. Adding the direction-valid registered atlas texture removed 16 extras and one required cube. These ablations isolate the scoring evidence among the same eligible candidates. Even the plain-color score uses the shared eligibility check. [Ablation counts](ablation.json)

Use the signal as a candidate for inspection or independent depth arbitration. Before automating the change, require a separate cue such as an observed rim or a continuous rear-plane grid extending on both sides. The RGB advantage alone accepted erroneous deletions with high margins. The experiment left ordinary placement, counting and surface masks unchanged.

## Replay

Use the typed-input cache from the baseline release, or rebuild it with `python -m benchmark.typed_structures prepare`. Supply `--cache PATH` to the first runner when using a different cache. Image SHA-256, geometry pack, typed pack and site coordinates are checked before evaluation. Use the same eight saved ordinary decorative geometry-only builds as the separate material-seam confirmation.

```powershell
python -m benchmark.material_feedback_geometry features --workers 6
python -m benchmark.material_feedback_geometry oracle
python -m benchmark.material_feedback_geometry train
python -m benchmark.material_feedback_geometry evaluate --workers 6
python -m benchmark.material_feedback_geometry ablate
python -m benchmark.material_feedback_decorative --workers 4 --data PATH_TO_DECORATIVE_DATASET --builds PATH_TO_EIGHT_BUILDS
python -m benchmark.material_feedback_examples
```

## What if the earlier depth ranker had the correct answers?

We also tested the standard Java 26.3 ranked depth proposals on ten source images, one per split. Within each split we selected the lexicographically lowest source-image SHA-256 before measuring accuracy. This test uses the estimated camera, observed masks and normal directions, and the source-only depth offers from the ordinary algorithm. The reference supplies just the choice among those offers. This is a diagnostic for the earlier ranker; the 104-view material-feedback results above come from a separate image-only experiment.

Of 165 ranked masks, **151** had one visible reference plane covering more than half their source samples. **149** also had the matching estimated axis and facing direction. All 149 correct depths were in the generated set, 148 were in the short list retained for joint foreground arbitration, and **139 were already chosen**. Depth level and direction are measured in the same single frozen alignment as the corresponding geometry release.

We reran primary candidate generation and the subsequent ranked-geometry checks with the other 10 selections forced to that reference plane. A second oracle scored *every* offered level by the exact XYZ of the visible reference cube reached by each sampled source ray, weighting each reference cube equally within a mask. It changed 11 depths, resulting in 1,649 correct, 239 missed and 126 extra positions after ranked placement (90.0355% strict, 93.3028% facade). Neither oracle could change masks, axes, increments or the acceptance rules.

| Measured ten-view stage | Ordinary strict correct / missed / extra | With reference depths | Strict F1 before → after | Facade F1 before → after |
| --- | ---: | ---: | ---: | ---: |
| Union of initial primary plane candidate cells | 1,449 / 439 / 161 | 1,445 / 443 / 147 | 82.85 → 83.05% | 90.91 → 91.18% |
| After strict-grid union, secondary proposals, frontmost support, side ownership and edge checks | 1,436 / 452 / 130 | 1,432 / 456 / 129 | 83.15 → 83.04% | 91.41 → 91.30% |
| After the ordinary ranked-placement refinements | 1,657 / 231 / 127 | 1,650 / 238 / 125 | 90.25 → 90.09% | 93.41 → 93.33% |

The first stage pools the primary proposed cells before strict-grid, secondary-depth and occlusion logic. Between stages the ordinary code both adds and rejects candidates. The last row ends at `atlas_plane_depth.place_ranked`. Later build-level local edits still follow in an ordinary export; the scored release's final positions differ from this stage on two of the ten views. No improvement to exported geometry is claimed by this oracle.

Toggling each of the 10 true-plane substitutions individually gave **nine ties and one strict regression** in the ranked placement counts. A chooser allowed to keep the baseline for each such test recovered zero additional blocks. On Colonnade 01, 269 of 432 mask samples supported the dominant plane; assigning that plane to the whole mask removed seven correct blocks and two extras. On Portico two offered depths were correctable in principle, yet their masks contained only one to three samples on the four-pixel analysis grid. None changed the final counts.

The evaluator rasterizes recorded face ownership at the actual placement-ray centers. See the [coordinate audit and real component bypasses](component-bypasses.md) for independent native-center checks and full ordinary datapack comparisons.

This measures choosing one level for an existing mask. Geometry could still benefit from improved mask coverage, a different face direction, a split into two depths, or competing whole-structure placements. We did not enumerate the combinatorial optimum over joint surface assignments or run a new learned ranker. [Offered levels, shortlist and per-image stages](plane-depth-cases.json) · [Pooled plane-depth counts](plane-depth-summary.json) · [Individual interventions](plane-depth-individual.json) · [Reference ray-to-block chooser](plane-depth-ray-oracle.json)

```powershell
python -m benchmark.material_feedback_depth_oracle --workers 6
python -m benchmark.material_feedback_depth_oracle --individual --workers 6
python -m benchmark.material_feedback_depth_oracle --ray-oracle --workers 6
```
