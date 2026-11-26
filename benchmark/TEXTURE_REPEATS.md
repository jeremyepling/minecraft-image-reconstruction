# Reading block counts from repeating textures

On Portico the cobblestone pillar and spruce beam have measurable whole-block repeats. The ordinary Surface Straightener distorted their spacing enough to fool the counter. Changing just the four source corners, while keeping the predicted mask and normal, let the existing counter get **1 × 5** for cobblestone and **18 × 1** for spruce. Using just the recorded mask fixed neither. The **6 × 3** white concrete wall remained unresolved with both corrections. These are checks with known geometry, separate from image-only predictions.

[Original, vanilla tile, predicted crop, measured grids and recorded-corner control](reviews/texture-count-01/portico.jpg) · [Four-way corners and mask control](reviews/portico-atlas-01/controls.json)

## What repeats in these crops?

* **Spruce beam.** The experimental crop-only matcher found the same patch again 32.46 pixels to the right. The average width of a recorded block in that chart was 32.56 pixels. This supports about 18 blocks across. The strong vertical repeat at 9.42 pixels is about one-quarter of a block: several plank courses occur within one vanilla sprite. A chart that admits these conflicting scales should record competing hypotheses. Reporting independent axes without a confidence check would have added 12 right and 33 wrong measurements across the suite. [Pitch evidence and independent-axis trial](reviews/portico-repeat-01/README.md)
* **Cobblestone pillar.** Its predicted face-on crop was 20 × 148 pixels, with 67% usable. On the recorded chart, the ordinary counter measured a vertical repeat at 32.94 pixels versus a recorded pitch of 32.80, returning 4.98 repeats over the five-block height. The atlas also confirmed the correct spacing on the recorded crop. On a real 14 × 5 Workshop cobblestone wall, the ordinary counter already counted correctly from its predicted crop. [Real and synthetic controls](reviews/portico-atlas-01/README.md)
* **White concrete.** A correct material name is insufficient here. The image provides too little stable contrast to measure a period, even with the recorded corners. A separate future test could borrow scale from an adjoining textured surface **only when their shared edge and reconstructed perspective support the same scale**. The white-concrete count should stay unresolved if that link cannot be established.

## Testing the two ways to find a repeat

Every method received the same single-image surface mask and face-on crop. The crop-only challenger checked whether small patches repeat across separate bands. The atlas challenger compared candidate spacings, including half and double pitches, with repeated 16 × 16 vanilla sprites under shifts and rotations. It sought agreement on a separate set of visible patches. These measurements feed the same visible-surface dimension calculation.

| All 56 saved views | Exact width and height | Wrong complete size | Eligible surfaces without a complete size |
|---|---:|---:|---:|
| Shipped counter | 95 | 50 | 453 |
| Additional crop-only repeats | 99 | 57 | 442 |
| Atlas check on unresolved counts | 95 | 51 | 452 |

**Keep the ordinary counter.** The extra crop-only matches supplied four new correct answers and seven new wrong ones. The atlas fallback supplied one extra wrong answer. It recognized stone bricks on Crossing and estimated 2 × 2 for a 2 × 3 wall. Finding a tile can establish a pitch without establishing where the whole surface starts or stops. [All image-only scores](reviews/texture-count-01/scores.json) · [Per-picture evidence and manifest](reviews/texture-count-01/manifest.json)

## How much could better straightening help?

The following diagnostic kept the counter fixed and gave it recorded plane corners, then recorded corners plus the true visible mask. Each cell lists exact / incorrect / unresolved sizes. Only surfaces already matched by the finder are included. The known-corner chart uses its vanishing directions for aspect; the Portico control above separately holds the predicted normal fixed.

| Surface appearances | Current crops | Recorded corners, original mask | Recorded corners and mask |
|---|---:|---:|---:|
| All 517 matches | 95 / 50 / 372 | 226 / 126 / 165 | 295 / 75 / 147 |
| 82 pure cobblestone matches | 19 / 7 / 56 | 56 / 22 / 4 | 71 / 8 / 3 |

This measures what the counter can read when it gets better input. Ordinary prediction has no access to the recorded corners or mask. [Per-surface diagnostic](reviews/texture-count-01/recorded-crop-diagnostic.json)

Reproduce the image-only comparison and Portico sheet using a fresh output directory:

```powershell
python -m benchmark.texture_count_round --method repeat --method atlas --sheet --out runs/my-texture-count
python -m benchmark.texture_crop_diagnostic --out runs/my-texture-count/recorded-crop-diagnostic.json
```
