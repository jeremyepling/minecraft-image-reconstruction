# Can a better chooser improve the Block Counter?

**Keep the current Block Counter.** On 56 saved screenshots, an answer-aware chooser could recover nine additional exact surface sizes. A small trained chooser gained two exact sizes on six regression buildings at the cost of nine more wrong sizes. It changed zero exported blocks on those buildings. [Picture comparison](reviews/grid-chooser-01/comparison.jpg)

## What the perfect chooser could choose

The ordinary counter looks for repeated texture across and down a straightened surface. We kept its exact reading, paired the horizontal and vertical texture peaks it already detects, and allowed a strong peak on one axis to determine the other through the counter's existing surface-aspect rule. This last operation applies the ordinary transfer rule to individual alternative peaks. Each candidate stores the continuous spacing, the visible dimensions after rounding, and the image-derived offset on each axis when a seam can be located. Two grids with the same rounded size remain separate if their spacing differs. The experiment explores 1,144 candidate grids across 714 proposed surfaces.

The dimension oracle asks whether **any** candidate gets the exact width and height of the visible surface. It can abstain when none is correct. A second oracle asks whether one grid per surface actually places visible block corners correctly. Both use recorded answers only in the scorer. The model that could run on a new screenshot gets the candidates and their image evidence. [Candidate lists and independent target file](reviews/grid-chooser-01/pool) · [Independent default replay](reviews/grid-chooser-01/control/manifest.json)

| Visible surface dimensions | Default exact | Forced model exact | Perfect chooser exact | Default wrong | Forced model wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shopfront and Wing, 20 development views, 156 surfaces | 17 | 19 | 21 | 13 | 16 |
| Six other buildings, 36 regression views, 442 surfaces | 78 | 80 | 83 | 37 | 46 |
| All 56 views, 598 surfaces | 95 | 99 | 104 | 50 | 62 |

On the six regression buildings the ordinary method left 327 pairs unresolved. Forced choice left 316 unresolved. Complete guesses on unmatched outlines rose from 10 to 11 and are additional to the table. The same two development buildings supplied the training data. Their row shows the final fitted model on those buildings. The six other layouts are previously reviewed fixtures, with no views used to fit this chooser. The new Visible Builds v1 families remain available for a fresh test. [Per-building counts](reviews/grid-chooser-01/chooser/forced-scores.json) · [Every frame](reviews/grid-chooser-01/chooser/frames.json)

The candidate pool explains the small gain. Of 598 target surfaces, 81 had no matched outline. Another 356 had a matching outline but no complete grid candidate. Of those 356, 61 lacked a usable straightened chart, while 295 had a chart. A further 57 had candidates but every proposed size was wrong. The remaining 104 had an exact candidate, and the ordinary counter already selected 95 of them. On the low-left masonry Wing wall the ordinary counter read 14 × 8 and an alternative read the correct 13 × 8 on both floors. Neither alternative located both grid offsets. [Oracle counts](reviews/grid-chooser-01/pool/scores.json) · [Visible surface examples](reviews/grid-chooser-01/sheet-views.json)

## What a trained chooser did

We fitted a regularized logistic model on CPU using repeat strength in each direction, evidence for the native texture scale, surface aspect, mask coverage, sampling resolution and whether the ordinary method preferred that candidate. Each surface had equal training weight. To choose when it could override the default, we trained on Shopfront and graded Wing, then swapped them. We compared eight confidence rules plus keeping the current reading. The selection score gives each building equal weight, rewards exact sizes and charges for wrong sizes and unmatched guesses.

That test selected the current reading. Forcing the same model to pick its top candidate wherever it could shows what the confidence rule avoided. It picked the correct 3 × 3 Portico face and 3 × 4 Workshop face, both with unknown grid offsets in both directions. On the six regression buildings, forcing it gained two exact sizes and lost zero existing exact sizes. It also produced nine additional wrong pairs and one extra unmatched guess. [Frozen weights and development folds](reviews/grid-chooser-01/chooser/model.json) · [Model and feature provenance](reviews/grid-chooser-01/chooser/manifest.json)

## Did it put visible blocks in better places?

For local placement we paired fully supported predicted face footprints one-to-one with fully visible reference faces. A match needs at least 50% intersection over union, and all four corners must land within a quarter block of their recorded positions. The ordinary counter located the grid offset in both directions on only 15 of 714 proposals. It recovered 100 aligned faces on 14 surfaces. The alignment oracle could recover 118 on 15 surfaces. Requiring **both exact dimensions and at least one aligned block face** gave 12 surfaces for the default and a ceiling of 13. Across the six regression buildings that last measure was 10 for both. Local scoring covered 10,438 eligible fully visible faces. Partial faces can still support a visible placement but are excluded from this four-corner test. [Local placement scores](reviews/grid-chooser-01/selected-placement/scores.json)

We also wrote and decoded geometry-only Minecraft structure packs. The export uses an image-derived camera origin and continuous spans. It supplies smooth stone as a fixed material so it measures coordinates. The scorer registers each build's arbitrary coordinate frame after export and grades it against the visible blocks from that screenshot. The current counter returned 344 correct coordinates and 18 extra across 10,626 visible block appearances. Forced choice produced 352 correct and the same 18 extra. All eight additions and all four extra aligned face crops came from a single Wing development screenshot. The other 55 builds had identical coordinate sets. On regression, both methods returned the same 122 correct coordinates and 18 extras. Sixteen of the 56 views had camera scale and nine produced a nonempty pack. Material accuracy was outside this experiment. [Forced-model export scores](reviews/grid-chooser-01/forced-placement/scores.json) · [Per-frame packs](reviews/grid-chooser-01/forced-placement/geometry-packs.zip)

**Next experiment.** Use the existing labeled face corners to test a spacing-and-offset predictor on the 295 matched surfaces that already had an image crop but lacked a complete candidate. Start with a frozen image encoder and a small trainable output layer on the Visible Builds training families. Measure candidate recall, aligned visible faces and unsupported guesses, then test the frozen choice on a separate building family. The observed ceiling gives a quick check on whether a later chooser is worth fitting.

## Replay

From this checkout with the repository's benchmark Python environment, run the following commands. Each processing command takes a new output directory. The checked-in review contains the original manifests, inputs, weights, per-frame scores and exported packs. [Review files](reviews/grid-chooser-01)

```powershell
python -m benchmark.grid_chooser_round --out runs/chooser-replay
python -m benchmark.grid_ranker --inputs runs/chooser-replay --out runs/ranker-replay
python -m benchmark.grid_placement_check --candidates runs/chooser-replay/candidates.json --selections runs/ranker-replay/selections.json --out runs/placement-replay
python -m benchmark.grid_placement_check --candidates runs/chooser-replay/candidates.json --selections runs/ranker-replay/forced-selections.json --out runs/forced-placement-replay
python -m benchmark.grid_chooser_sheet runs/chooser-replay runs/ranker-replay/forced-selections.json runs/chooser-sheet
```
