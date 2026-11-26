# Review the crops from split-only zoom

[Open the full-size portrait sheet](reviews/crop-review-02/comparison.jpg) · [Look up a picture, surface and source rectangle](reviews/crop-review-02/index.json)

Each Block Counter panel shows **Counter** for the estimated dimensions and **Actual** for the recorded surface dimensions, both in blocks across × blocks down the face. A question mark marks an unresolved estimate on that axis.

Pick a building column and follow one lettered surface down the stage rows. Zoom in on the first row where it looks wrong. A reply like **“Workshop B, straightened crop — the top row of blocks disappears”** identifies the example, stage, and edge. A good crop is worth marking too, especially if it has a hole or a steep viewing angle.

**Start with Wing A, Shopfront B and Workshop C.** Wing A shows a wall where the window includes almost all of the outline and both dimensions come out right at 13 × 8 blocks. In Shopfront B, the violet window contains only 41% of the cyan mask. Workshop C shows a thin angled face where the window contains 68%. In each case the percentage measures how much of the *predicted* surface fits inside the window. The separate gold reference lets you check whether that prediction starts on the right face.

**Courtyard B and Wing C show two other cases.** Courtyard B’s window encloses the entire predicted mask, but much of the rectangular image is dark because the surface is irregular. Wing C has a matched roof surface and no usable straightened crop. There are 24 selected examples from ten existing screenshots. Shopfront B and Workshop C use alternate saved views labeled on the sheet. The other 22 use the usual eight building views.

Across the whole suite, **505 of 516** matched surfaces produced a usable straightened sample. For **104 of those 505**, the sampling window left more than 10% of the finder’s own pixels outside. For 15, it left more than 20% outside. This measures coverage at original screenshot resolution before edge masking. It makes window size a good first experiment. To judge success, compare recovered surface pixels, neighboring pixels accidentally admitted, and block-size accuracy with the finder held fixed. [Coverage for every matched surface](reviews/crop-coverage-01/coverage.json)

| What you see | What to compare | What we could test |
| --- | --- | --- |
| A strip of a neighboring wall or sky appears | Check whether the finder already included it inside the outline. Neighboring pixels in the padded source view are intentional. | If those pixels also enter the face-on sample, compare the source mask and the pixels accepted for sampling. |
| The outline follows the correct edge, but part of the wall disappears | Compare the outline with the four corners used for straightening. An irregular surface can extend past that window. | Enlarge the window in the plane of the wall while preserving the original pixel mask and its holes. |
| Brick courses still slope, or a pillar leans | Follow the same long mortar seam or pillar edge from the source to the straightened crop. | Compare the current direction estimate with one fitted to several parallel edges of the same plane. |
| Seams are straight but the blocks look unusually thin or wide | Compare the horizontal and vertical spacing of a repeating pattern. | Test the proportions of the output image separately from its tilt. |

The standard context crop includes up to eight extra source pixels on each side. The straightener samples from the original screenshot. Its mask restricts all four pixels used to blend each output color, and it trims a narrow rim along the sampling window. Increasing the display margin by itself would give the reviewer more context and produce the same face-on sample. [Source crop](../engine/surface_perimeter.py) · [Surface Straightener](../engine/surface_sample.py)

Review comments belong to this sheet’s run, image hash, proposal, stage and native source rectangle. The index maps each lettered example to those fields. Keep reviewer observations alongside the measured scores. Use them to choose a general test.

### Reproduce the sheet

```powershell
.\Invoke-SurfaceRound.ps1 -Round crop-review -Out runs/my-crop-review
```

Use a new output directory. The review reads the frozen screenshots and the selected finder’s archived predictions from `crop-split-01`. Minecraft stays closed.

### Crop expansion

Run `crop-fit-01` kept the finder, perspective axes, mask and Block Counter fixed on 56 saved views of eight previously reviewed buildings. Both challengers enlarged the window in the surface plane. Full expansion included the outermost mask edges. Guarded expansion skipped windows already covering 90% of the mask, ignored the extreme 1% at either end of each axis and limited the distance each side could grow.

| On the same 56 views | Current crop | Full expansion | Guarded expansion |
|---|---:|---:|---:|
| Usable crops missing more than 10% of their detected surface | 67 | 0 | 2 |
| Correct width and height in blocks | 94 | 90 | 93 |
| Incorrect width or height | 51 | 51 | 51 |
| Eligible surfaces without a complete size | 453 | 457 | 454 |

All three methods started with 517 matched surfaces out of 598 and produced 456 usable crops. In the development buildings, exact/incorrect counts were 17/13, 17/14 and 17/13. On the previously reviewed regression buildings they were 77/38, 73/37 and 76/38. In the high-right plaster view of Portico on concrete, the current crop counted R12 correctly at 1 × 5. Both expansions still found five blocks vertically but lost the width. Full expansion added 168,654 pixels belonging to the correct recorded surfaces and 56,382 pixels from elsewhere already inside the predicted masks. Window coverage measures what the chart encloses before the sampler rejects boundary pixels. Neither alternative added an exact size. The ordinary crop remains the default.

[Eight-building comparison](reviews/crop-fit-01/comparison.jpg) · [Per-picture measurements](reviews/crop-fit-01/cases.json) · [Building and split scores](reviews/crop-fit-01/scores.json) · [Manifest](reviews/crop-fit-01/manifest.json)

Reproduce offline with a fresh output directory:

```powershell
.\Invoke-SurfaceRound.ps1 -Round crop-fit -Out runs/my-crop-fit
```
