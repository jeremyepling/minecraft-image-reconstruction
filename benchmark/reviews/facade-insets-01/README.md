# Where the Facade photograph shows recessed walls

Three recesses stand out because the Surface Finder assigns most of each complete opening to the flat front wall. Their backs, side returns, sills and surrounding wall are all tuff. On the original full-resolution image, the depth change is visible where the texture turns the corner and a strip of horizontal sill meets the vertical return. The green patches below are photographed back walls, the orange faces look toward world +X, and the blue faces point up. Letters on the full photograph let you find the same locations at either scale. [Marked full photograph](facade-marked.jpg) · [Three photographed details and their recorded directions](inset-detail.jpg)

| Recess | Back depth behind facade | Share of back assigned to facade | Share of entire recess assigned to facade | Separate return / sill mask IoU | Correct back blocks |
| --- | ---: | ---: | ---: | --- | ---: |
| D · middle left | 1 block | 97.1% | 72.4% | 0.90 / 0.68 | 0 / 12 |
| B · upper middle | 2 blocks | 98.1% | 69.3% | 0.92 / 0.75 | 0 / 10 |
| C · upper right | 3 blocks | 99.0% | 70.1% | 0.90 / 0.82 | 0 / 6 |

The finder already has two useful source masks at each recess, but its Block Counter gives neither dimension for any of the six side and sill masks. It measures the combined front mask as 27 × 11. The photographer can see 4 × 3 back blocks in D, 5 × 2 in B, and 3 × 2 in C. Every approved geometry claim here uses the visible pixels of this particular photograph. Grading counts 28 missing back blocks, 13/13 placed visible side-return blocks, and 13/23 placed visible sill blocks across these three recesses. The latter two counts include blocks which have more than one photographed face.

When the reconstructed picture is rendered from its estimated camera, the estimated geometry is in front of the recorded back at 4,896 / 5,069 pixels in D, 5,822 / 5,822 in B and 4,161 / 6,022 in C. All other pixels on those backs have a clear sightline through the estimated geometry. Removing the front-wall assignment and placing the back at the right setback are separately measurable steps.

The image-only return-and-grid challenger separates these three backs with mask IoU of 0.97, 0.97 and 0.98 for D, B and C. Its front-wall mask IoU rises from 0.8827 to 0.9586. The middle two columns show actual masks produced from the same photograph. The rightmost overlay uses recorded visible faces for grading. [Before and after on the original pixels](returns-comparison.jpg)

## Other outlines in the same image

| Mark | Visible back or panel | Finder result | Back or panel positions placed |
| --- | --- | --- | ---: |
| A · small upper left | Smooth stone, 1 block back | Separate return. Front mask covers 89.6% of the back. | 0 / 2 |
| E · ground left | Tuff, 2 blocks back | Separate return. Front mask covers 86.5% of the back. | 0 / 5 |
| F · ground middle | Andesite, 3 blocks back | The back occupies only 179 pixels. Most of the visible opening is its large return. | 0 / 4 |
| G · ground right | Cobblestone, 2 blocks back | The back has its own mask with 0.969 IoU. | 24 / 24 |
| H · center-right panel | Eight polished deepslate blocks at facade depth | This material change is flush with the surrounding tuff. | 8 / 8 |

The seven recess boxes trace only the observed side, sill when visible, and back surfaces. Floor outside the build, hidden cavity volume, and unseen faces are excluded. Letter H is a control for methods using color change to select estimated depth. Its recovered positions are a geometry score. Material classification is a separate measure.

## Regrade the same locations

From the repository root, run the following command with the archived production comparison and the ordinary Facade run from the previous round available beside this checkout. The grader verifies the original RGB and reference hashes, the ordinary finder output and the equivalence of the decoded geometry to the frozen production pack. All measures use the 1440 × 900 source pixels. The single whole-build translation is fitted to all 361 visible blocks. Production scores 277 correct, 84 missed and 32 extras on this view. [Machine-readable measured areas, visible coordinates, source face IDs and count/placement results](openings.json)

```powershell
python -m benchmark.facade_inset_evidence
python -m benchmark.facade_inset_evidence --surfaces runs/candidate/surfaces.json --counts runs/candidate/counts.json --pack runs/candidate/reconstruction.zip --out runs/candidate-inset-scores
```

Add `--comparison` to draw the four-column source, original, proposed and reference figure. For a surface-only probe, supply just `--surfaces` and `--out`. Use the same RGB and estimated camera for all candidates. Review the loss of correct host-wall pixels and the seven visible openings alongside mask IoU and recovered coordinates. Keep the eight flat panel cells as a negative control. Use the fixed 104 and the separate transfer buildings for claims beyond this reviewed example. [Frozen suite and single-shift scoring](../strip-orientation-01/README.md)
