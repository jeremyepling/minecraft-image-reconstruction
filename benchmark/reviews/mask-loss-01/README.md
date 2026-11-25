# Where the published B predictor loses surfaces

[Six enlarged examples](stages.jpg) follow a visible surface from the original screenshot through B's processing. The far-right column shows the recorded surface solely for comparison. The first group and fitting-input columns show the candidates with the highest overlap at their respective stages. The last two prediction columns show the **same emitted proposal** before and after fitting its rectangular envelope. A blank overlay means that no candidate touched the reference there.

B produced the same 421 proposals as the archived run on all 56 saved views. This diagnostic verified each emitted mask, rectangle, identifier, confidence, pixel count and source label. Each image was replayed using its own numerical MoGe-2 output before the diagnostic opened the recorded geometry. All eight building layouts had been examined in previous work. The breakdown keeps 20 designated development images separate from the 36 previously reviewed comparison images.

| Evidence available | Reference surfaces matched, out of 598 |
| --- | ---: |
| At least half of the surface pixels point within 30° of the recorded direction | 536 |
| A connected island covers the individual surface, given its true direction | 270 |
| An initial connected group in B covers it | 260 |
| One of B's regions submitted to fitting covers it | 331 |
| Such a region yields a rectangle before filtering | 308 |
| One of the retained masks covers it | 274 |
| One of the retained rectangles covers it | 281 |

536 measures agreement in **direction only**. For the connected-island check, we gave the scorer a recorded surface direction, searched the entire numerical field for matching pixels and grouped connected pixels. This diagnostic demonstrates one way of recovering regions given a known direction. A low score does not establish that the numerical field contains no other information. Adjacent walls and pillars can point in the same direction.

Every row after the first uses one-to-one region overlap of at least 0.50, including false coverage outside the building and across openings. Counts can increase from one stage to the next because B tries subdivisions using image color and rectangles can smooth incomplete masks. Of the 274 mask matches and 281 rectangle matches, 262 refer to the same reference surfaces. The final rectangle loses 12 previously matching masks and gains 19 matches. That net gain does not account for masks that never made it through fitting.

Of the 317 surfaces missed by the rectangles, 59 had weak directional coverage and 175 had majority directional coverage without a sufficiently distinct island or subsequent matching region. Another 15 had a matching diagnostic island but failed grouping, 23 had a matching region reach fitting and fail, 33 passed fitting and failed later selection, and 12 were distorted by their rectangular envelope. These categories locate the furthest stage with a matching proposal, so they describe observations rather than causal experiments.

### Examples

- On the high-right plaster wing, the true-direction island overlapped 98% by intersection over union. B's best initial group scored only 25%. The recorded target is the broad gray wall in row three of the sheet.
- On the low-left plaster shopfront, the irregular recessed surface still scored 62% when submitted to fitting. No emitted B proposal touched it.
- On the high-right masonry crossing, an emitted mask matched the front of the overhang at 65%. Its own four corners reduced overlap to 46%.
- Depth Anything ordered 459 of 464 near/far surface pairs correctly when sampled beside a **known** occlusion boundary. On parallel pairs it ordered 231 of 232 correctly. B uses the relative-depth field after outline detection. These results establish useful depth evidence at known boundaries, while automatic boundary localization remains a separate test.

### What reached block building

Among 281 matched rectangles, B returned block counts along both axes for 12. All 12 were within half a block of both true spans. Only one also registered the interior grid at the archived threshold. Fifteen rectangles returned a metric depth estimate and seven came within one block. The ordinary assembler reads its separate facade model. The B surface command stops at camera-space surfaces, so these counts alone do not establish correct placement in a world.

### Reproduce from saved data

From a checkout containing this experiment, with the numerical cache in the adjacent normal-round checkout, run the following command. Use a fresh output folder. All 56 outputs and the example sheet are regenerated from the saved inputs.

```powershell
python -m benchmark.mask_loss_trace --fields ../normal-round/runs/normal-fields --out runs/replay-mask-loss
```

[Run manifest](manifest.json) · [All scores and split/building breakdown](summary.json) · [Scores for each picture](pictures.json) · [Each surface, candidate index, paired fit result, near/far pair and pixel box](surfaces.json.gz)
