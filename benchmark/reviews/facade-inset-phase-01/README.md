# Testing a recessed wall against the wall's own texture

**The photographic texture test separated all three recessed backs in the reported Facade view, including the visible parts of recesses one, two and three blocks deep.** It skipped the dark flush inlay on the same wall. Repeated texture supplies an independent depth measurement. The scope recommended for ordinary use is the residual of a wall where two visible returns have established a recess elsewhere on that wall. [Actual reconstructed surfaces, dimensions and geometry pack](focus-export.zip) · [Per-surface visible block counts and coordinates](focus.json)

| Visible region | Reference | Measured depth | Mask IoU | Correct backing blocks before → after ordinary recount |
| --- | ---: | ---: | ---: | ---: |
| Lower left | 9 | 1 | 96.70% | 0 → 4 of 12 |
| Upper middle | 14 | 2 | 97.34% | 0 → 8 of 10 |
| Upper right | 19 | 3 | 98.46% | 0 → 6 of 6 |
| Dark panel flush with front | existing wall | No accepted recess | Remains on wall | — |

The reported Facade's host-wall IoU increased from 88.27% to 95.86%. The ordinary placer gained 28 correct blocks across this image, lost zero, removed 15 extras and added zero. Eighteen of the 28 gains are on the three measured backs, and ten lie elsewhere. This is the phase-only trial against c589f1a. The separate signed-depth placement experiment uses additional evidence.

## How the texture test works

The image's own calibrated camera supplies the wall axes. Inside a large predicted wall the method tries integer planes and requires an independently repeating 16×16 texel pattern across alternating wall blocks. A rectangular run of cells that is darker and inconsistent with that texture nominates a possible opening. Matching host texture must surround all four sides. A possible back competes with the flat interpretation and nearer planes on the same native image rays. The winning depth must fit the host texture better than the runner-up, fit both spatial halves separately, and cover at least 65% of the projected visible back inside the existing host mask. The method can abstain on real open-bottom doors, backs with different art, barely visible slivers, insufficient texture or an unsupported camera.

Only pixels already owned by the photographed wall can move to a new child. Each accepted pixel must lie inside both the front rectangle and the projected back rectangle. The original surface pixels and new child pixels remain disjoint and their union reconstructs the exact original mask. The helper uses no reference world, external model or texture-atlas download. [Detector and supplemental entry point](../../../engine/facade_inset_phase.py) · [Regression tests](../../../tests/test_facade_inset_phase.py)

## Fixed evaluation and transfer

The 204 photos consist of 72 development, 104 fixed whole-structure, 20 alternate, four new and four shelter/gate photos. The stand-alone rule accepted ten of 26 photographed aperture candidates. Adding the projected-area guard removed one 13%-IoU duplicate fragment while preserving all ten newly matched true surfaces. Before that guard, the unrelated old rim-view image lost four paired pixels and gained a stray mask. [All front and alternate-depth correlations and post-prediction grading](apertures.json) · [Earlier rule](unrestricted-apertures.json)

| Changed view with guarded stand-alone rule | Newly matched backs | Strict geometry, correct gained / lost | Extras added / removed |
| --- | ---: | ---: | ---: |
| Fixed Facade, right | 3 | 28 / 0 | 0 / 15 |
| Fixed Facade, left | 1 | 11 / 0 | 0 / 6 |
| Fixed test-recess-28, left | 2 | 10 / 1 | 1 / 6 |
| Alternate positive-control Facade, left | 2 | 13 / 1 | 1 / 8 |
| Alternate rim Facade, left | 2 | 18 / 1 | 0 / 14 |

The remaining 199 photos kept their original masks. Some left and right views depict the same layout. The reported Facade was inspected during design, so its three detections demonstrate the intended behavior on a known failure. Test-recess-28 supplies a physically different layout with an additional matched region.

**After the strict visible-jamb-and-sill stage, nine of the ten phase detections duplicate its newly matched backs.** The exception is reference surface 17 on test-recess-28-left. The strict stage already recognized a distinct recess on its host. Applying phase only to that host improves the best mask for surface 17 from 5.06% to **66.49%** IoU. It correctly assigns 627 more pixels and adds one surface match. The 628-pixel child covers 627 of 942 recorded backing pixels. Its winning depth is two blocks, with native texture correlation 0.792 against 0.019 on the next-best depth. Every prior return output and its metadata is preserved. All 204 geometry results tie the strict stage after ordinary recount. Other outputs keep the same masks. The scoped rule returns immediately on the 187 photos without accepted strict return evidence; only 17 need wall texture checks. [Sequential whole-frame comparison](after-returns.json) · [Production-scoped comparison and timings](supplemental.json) · [Actual revised mask output](supplemental-surfaces.zip)

**Additional native captures.** Of 34 archived photos across seven fixture groups, 16 duplicated RGB hashes and 18 were genuinely new views. Thirteen of those 18 lacked a supported image-estimated camera and automatically abstained. All five supported photographs ran through the ordinary c589f1a Surface Finder before the texture rule. The two dark-inlay views and the physical-shade view kept all flat controls intact. The two new dev-recess-15 angles also abstained. These five comprise three normalized occupancy/appearance groups that were already represented in the 204. No additional independent layout or positive detection is claimed from this screen. All 18 new images passed the 2% whole-structure framing screen. [Pixel hashes, camera and per-edge framing](external-screen.json) · [Ordinary surface matches and candidates on the five supported views](external-cases.json)

Recommend using the host-scoped helper as a surface-stage supplement after accepted two-return recesses. Keep unrestricted whole-image searches experimental given their additional runtime and ordinary placement losses on three views. Only the strict signed return observations should feed the separately validated placement constraint. The phase helper maintains independent evidence in its own events.

## Replay

Run in a checkout with access to the shared frozen evaluator and native suites. On this PC the shared evaluator and its baseline stage archive are in the sibling facade-inset-round checkout. The supplemental replay reads its frozen 204-view strict-returns output.

```powershell
python -m benchmark.facade_phase_evidence --workers 5
python -m benchmark.facade_phase_focus
python -m benchmark.facade_phase_after_returns --workers 5
python -m benchmark.facade_phase_after_returns --scoped --workers 5
python -m benchmark.facade_phase_external
python -m unittest discover -s tests -p test_facade_inset_phase.py
```
