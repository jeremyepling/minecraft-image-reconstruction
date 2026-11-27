# Selecting wall edits from their projected appearance

The projection check rejected all three changed exports across 200 photographs. It avoided all four newly hidden blocks and the alternate-view false placement while also giving up five net correct fixed placements. Fixed F1 stayed at 90.414% and alternate F1 stayed at 88.554%. The four additional shelter and gate exports in the preceding trial had identical geometry before and after. [Development results](projection-development.json) · [All differing pairs and cohort scores](projection-gate.json)

| Changed export | Changed first-hit samples | Silhouette errors removed | Direction errors removed | Held-out RGB error before → after | Decision |
|---|---:|---:|---:|---|---|
| Canopy 18, left | 1,516 | −88 | −22 | 10.12 → 5.19 across 278 samples | Keep original |
| Canopy 05, left | 1,375 | +44 | −55 | 10.17 → 3.53 across 163 samples | Keep original |
| Open Canopy 00, alternate left view | 9 | — | — | Below the 48-sample minimum | Keep original |

The decision rule, frozen at `2bc97b7`, compares only pixels where the nearest visible block or face direction changes between exports. Both alternatives use the same original photo masks, estimated camera and block texture atlas. The rule requires silhouette and face direction errors to each stay level or improve, a combined reduction of at least 1% of changed samples, an RGB error improvement of 0.20 over at least 24 held-out texture samples and four correctly oriented foreground samples per added cell. The renderer samples every second pixel along each image axis. The face texture fit and evaluation use separate patches. The 72 development images had no differing exports, so the thresholds were fixed without a fitted classifier. The geometry-only comparison measures sampled pixels through an estimated camera and cannot establish whether each individual proposed block was actually visible.

Texture alone preferred both edits that incorrectly placed hidden blocks. Face-direction errors identified a reason to reject each. A whole-export choice discarded their useful changes along with the bad ones. With only two changed fixed exports, this provides little evidence to train a decision model. A smaller decision unit, such as one supported post or one changed exposed face, would let a trial preserve useful edits independently.

The half-resolution renderer comes from `review/wall-source-projection` at `3d1a05c`. Replay reads the saved control and selective exports, checks each differing ZIP against its manifest and decodes its local cells before rendering. It opens the geometry score fields only after freezing the image-only selections.

```powershell
python -m benchmark.wall_texture_projected_gate --projection-root ../wall-source-projection --out runs/wall-projection-gate.json
```
