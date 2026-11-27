# Recognizing visible recesses in one Minecraft source image

On the reported Facade, Surface Finder assigned 97–99% of three tuff back walls to the surrounding tuff facade. Their visible side faces and sills were already detected. Fitting those two edges to the host block grid measures setbacks of one, two and three blocks. The three new back masks match the recorded visible faces at **96.7%, 97.3% and 98.5% IoU**. Local placement recovers **12/12, 10/10 and 6/6** visible back-wall blocks and removes 15 false foreground blocks. The eight-block dark panel beside the openings is flush and retains all eight correct placements. [Full source image and three proposed surfaces](../facade-insets-01/returns-comparison.jpg) · [Every recess and the flat panel](../facade-local-placement-01/named-openings.json)

## What the normal command uses

The side-and-sill fit tries competing directions and integer setbacks on the two return masks extracted from the source image. The predicted rim must meet the wall's grid, the back must have its own visible pixels, and shading must agree with the proposed opening. The support for the new back is taken from the original mask. Changing materials, an incomplete tile or a rectangular dark patch can nominate an opening; each still needs evidence for a separate plane. The flat inlay on the Facade illustrates the distinction. [Geometric tests](../facade-inset-returns-01/README.md)

The texture check learns a 16×16 repeating pattern from the observed wall. It projects that pattern onto the same image pixels at the front plane and at candidate recessed depths. Both halves of the region must favor the recessed alignment. Running this on the residual of an already confirmed inset wall adds a visible back on **Recess 28**. Its mask reaches 66.49% IoU, with 627 more correctly assigned pixels. The other 203 outputs from this supplemental step tie the geometric masks. Local placement needs independent side-and-sill evidence before committing the depth. [Texture evidence, alternatives and extra controls](../facade-inset-phase-01/README.md)

The geometric and texture checks recover 29 backs in 17 views. A third check measures a proposed opening against the block pattern in the source image. It requires a complete rim, a stable count on the surrounding wall, and agreement between two measurements of the visible width and height. A two-pixel collar leaves the uncertain edge on its original wall. Five additional backs pass on three images. They keep unresolved depth. Across all three checks on **204 saved images**, Surface Finder recovered **34 matched backs in 19 views**, with **134,842** additional correctly paired surface pixels. They preserved every existing surface match, with zero additional unmatched masks or newly covered background pixels. The geometry exporter uses the backs whose setback the visible side and sill establish. The existing wall layout anchors the other parts of the building. On the fixed 104, the default adds 84 correct positions and removes 49 extras. Over all 204 it gains 195 correct positions, removes 111 extras, loses zero correct positions and adds zero extras or hidden blocks. The ordinary build uses smooth stone as its geometry placeholder. [Combined 204-view verification](combined.json) · [Actual placement packs, including alternate-camera results](../facade-local-placement-01/README.md)

| Cohort | Improved builds | Correct positions gained | Extras removed | Strict F1 before → after |
| --- | ---: | ---: | ---: | ---: |
| 104 fixed structure views | 7 | 84 | 49 | 90.50% → 90.80% |
| 72 development views | 6 | 46 | 20 | 91.21% → 91.41% |
| 20 alternate views | 4 | 65 | 42 | 88.67% → 90.02% |
| 4 new-building views | 0 | 0 | 0 | 83.58% → 83.58% |
| 4 shelter and gate views | 0 | 0 | 0 | 83.47% → 83.47% |

Fixed facade-priority F1 rises from **92.56% to 92.90%**. The historical 112-view strict score is 90.04%. The two four-view cohorts supplied negative checks. Successes on alternate angles or palettes of the same layout are counted as views. Fresh ordinary RGB builds on seven images of Facade, Sawtooth 3, Recess 28 and Dev Relief 00 reproduce the evaluated block and material maps. Two contain only the count-verified outline improvement. [Three other buildings in their source images](other-buildings.jpg) · [Ordinary stage and export hashes](ordinary-entry.json)

## What block size reveals

The three visible back walls on the reported Facade span 4 × 3, 5 × 2 and 3 × 2 visible blocks. On the 29 backs with measured depth, the generic rectified texture counter gets both dimensions right on nine. Measuring the span between independently pixel-supported cells on the fitted grid gets both right on **28 of 29**. One irregular Relief wall still measures 3 × 3 against a 5 × 3 visible extent. The count file includes both the measured extent and the number of supported cells. Gaps between those cells retain unknown occupancy. On the 17 affected host walls, complete correct dimensions rose from 9 to 15, preserving all 19 previously correct individual dimensions. The default keeps a newly estimated single host dimension unresolved if its other axis is still missing. [Generic count audit](count-check.json) · [Measured extents](observed-back-spans.json) · [Frozen geometry and texture run used by the count audit](geometry-and-texture.json)

## Can a learned decision help?

With the final safeguards in place, **turning the tree off produced identical masks, anchors and new block counts on all 204 images**. The five extra backs appear in one cool-palette Sawtooth 3 source image and two Facade source images. Every pixel reassigned to them belongs to the recorded back. On Sawtooth 2, a proposed plane spanned a 1 × 4 block grid. The Block Counter independently estimated 1 × 3 from its surface crop. [Full source image and crop](../../../ARCHITECTURE.md#worked-example-of-a-rejected-surface) The tree discarded it early, and the count check discarded it independently. The default checks the rim and block count against the source image. The decision tree remains an experiment. The four new-building and four shelter/gate images supplied no positive examples for this rule. The fixed images also informed the count and collar safeguards, so their scores describe regressions on reviewed images. [Tree on/off comparison](../inset-mask-chooser-01/final-tree-ablation.md) · [Two additional outlines in a second Facade source image, highlighted in gold](learned-outlines.jpg)


A small supervised tree considered additional back candidates from the same pair of return masks. It saw 35 proposals across six development layouts. Inputs describe fit to the side and sill, surrounding wall, competing depth fits and pixel contrast. World positions and visible surface IDs grade each proposal after prediction. At most one setback can be accepted per side-and-sill pair.

A useful *surface-and-depth* decision here requires a previously unmatched, recessed parallel reference surface, at least 80% mask purity, IoU of at least 0.5 and the correct integer setback. On the **104 fixed** views the strict geometric gate accepted **11/11** correct decisions. The perfect chooser, given reference answers for the same loose candidates, could accept **18**, leaving room for seven. This diagnostic counts one candidate family of methods, so it excludes the separate Recess 28 texture gain.

At a threshold selected by leaving out whole layout identities, the learned tree accepted **21** on fixed: **18 correct and three wrong**. Two corresponded to real Cloister backs at six and seven blocks that the tree placed two blocks away. Another reselected a partially detected Shopfront plane. On 20 alternate views it accepted ten correct decisions, three beyond the strict seven. When the development threshold also had to transfer across held-out *families*, the tree retained the strict choices. An independently held-out family differs more than another palette or layout from a training family. [Grouped validation, frozen tree and threshold results](model.json)

Running the permissive mask choices through the ordinary global counter and placer recovered 153 correct blocks on fixed but lost 18 and added 14 extras, including 12 hidden. Global replacement using only the strict masks also disturbed unrelated cells. These interventions explain why the release uses a measured setback and local edits. The reference chooser is a coverage estimate for these proposed surfaces and depths; it does not establish an end-to-end placement ceiling. [Paired mask and real placement tests](challengers.json) · [Per-view evidence and graded local proposals](candidate-scores.zip)

## Replay

Use the registered fixed and whole-structure fixture checkouts and the released Java 26.3 atlas. The two stage inputs changed by the previous default at c589f1a are included in [baseline-stages.zip](baseline-stages.zip) and verified against [ordinary-entry digests](../strip-orientation-01/combined-entry.json). The frozen stage comparison and scorer use the saved source image and estimated camera for prediction. Source-world labels and captured camera calibration enter only during scoring.

```powershell
python -m benchmark.facade_inset_round --out runs/baseline
python -m benchmark.facade_inset_round --candidate-root . --candidate benchmark.inset_return_grid:refine --geometry --out runs/returns
python -m benchmark.inset_chooser_data --candidate-root . --out runs/chooser
python -m benchmark.inset_decision_model --data runs/chooser/rows.json --out runs/chooser/model.json
python -m benchmark.facade_inset_round --candidate-root . --candidate benchmark.inset_tree_trial:refine --geometry --out runs/tree
python -m benchmark.inset_default_verification
python -m benchmark.inset_final_tree_ablation
```
