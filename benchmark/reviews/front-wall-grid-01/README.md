# Recovering a close-up plane from the vanilla atlas

Six of eight unsupported exterior screenshots yielded usable local grids. These are four camera/material combinations from two previously reviewed buildings, repeated on two floors. Together the decoded packs recovered 262 of 384 visible target coordinates with zero extras after one whole-build shift. Masonry recovered 46/48 on Wing and 45/48 on Courtyard. Pale concrete recovered 40/48 on Wing and abstained on Courtyard. All 58 previously unsupported photos in the separate native-camera screen still abstained.

The predictor measures the horizontal and vertical pixel spacing of texel edges and matches their phase to the calibrated 16×16 vanilla atlas in two disjoint halves of the image. Each proposed cell, including a clipped border cell, must also match the visible part of the sprite. Flat or insufficiently observed patches abstain. Only the RGB and atlas enter prediction. The benchmark resolves compass from the recorded camera for scoring.

**Keep `experiment/front-wall-atlas-grid` experimental.** On independently generated front-facing wall controls with four materials, varied texel sizes and an occluded cell, the method recovered all 506 visible positions without extras. The exact same pixels can depict a floor seen from directly above. Treating those images as upright walls placed 434 extras under the fixed-orientation metric. A photographed reference for gravity or a supplied face orientation could make the local grid usable in placement. Another close-up wall alone cannot resolve that ambiguity.

Replay using `python -m benchmark.front_wall_trial --out runs/front-wall-audit` and `python -m benchmark.front_wall_controls --out runs/front-wall-audit`. Both use the saved inputs and installed calibrated Java 26.3 atlas. [Measured grids and native-screen results](manifest.json) · [Paired orientation and occlusion controls](controls.json)

