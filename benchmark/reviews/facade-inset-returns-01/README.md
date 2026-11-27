# Reconstruct visible backs from a photographed jamb and sill

The lower-left, upper-middle and upper-right recessed back walls of Facade all became separate masks. Their native mask IoUs improved from 2.40%, 2.78%, and 2.91% to **96.70%, 97.34%, and 98.46%**. The two existing perpendicular return masks constrain a snapped opening and its depth. The measured recesses were one, two and three blocks deep. The reference labels were opened after prediction. Dark flush panels, ground-level passages and a fourth small recess with insufficient return evidence were left as they were.

The experiment starts from `c589f1a`. Camera orientation and scale come from the photographed floor and the existing camera method. Floor contact along the host wall sets the grid level. For each neighboring jamb and sill inside the projected wall, the candidate tries nearby snapped rim coordinates and differently signed depths. Every ray through the aperture hits either the back, jamb, or sill first. The resulting two return masks must match both detected masks at the **same depth**. Visible host support must surround the opening on all four sides, fill most of the back, and show a photographed color difference from both returns. Only pixels already assigned to the host can move to the new back. Faces can have irregular visible footprints. No geometry is fabricated to complete the rectangular hypothesis.

| Cohort | Active / total views | New matched backs | Correctly reassigned pixels | Correct blocks gained / lost | Extras added / removed |
|---|---:|---:|---:|---:|---:|
| Development | 6 / 72 | 10 | 35,725 | 48 / 3 | 9 / 36 |
| Fixed evaluation | 7 / 104 | 11 | 43,296 | 87 / 15 | 11 / 60 |
| Alternate views | 4 / 20 | 7 | 33,056 | 62 / 15 | 18 / 47 |
| New layouts and independent shelter/gate | 0 / 8 | 0 | 0 | 0 / 0 | 0 / 0 |

The 28 splits produced 28 additional one-to-one matched reference surfaces. No existing match was lost, and the number of unmatched masks and background pixels remained constant throughout. The named Facade gained 28 correct blocks, lost none, and removed 15 extras. Its host remained 27 × 11. These are improvements in identifying previously mixed regions. Total covered pixels are unchanged.

Some downstream placements still deteriorated. On alternate `train-recess-02-left`, one new true back-wall mask gained 3,144 correctly assigned pixels, yet the ordinary placer gained one correct block, lost five and added eight extras. Across the fixed suite the number of occupied-but-hidden erroneous blocks increased by ten, even as total extras fell by 49. Use the image-measured depth handoff as a separate experiment. A Surface Finder gain does not establish that every placement change is correct.

Run `python -m unittest tests.test_inset_return_grid` in the candidate checkout. The coordinated runner `benchmark.facade_inset_round` accepts `--candidate-root ../facade-inset-returns --candidate benchmark.inset_return_grid:refine --geometry --out runs/returns`. Then run `python -m benchmark.inset_return_summary` here to reproduce the [hash-checked per-frame results](result.json). The frozen candidate is `15202d5`, with a separate looser decision-model proposal interface at `b7ec543`.
