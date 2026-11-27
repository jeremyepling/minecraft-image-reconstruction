# Where the Workshop plank errors came from

This audit uses the ordinary Java 26.3 placements at `53252df`, before the Workshop foreground change. The four selected views were ranked 2, 3, 4 and 9 in that viewer. “Extra over planks” means an extra exported cube covered more spruce reference pixels than any other reference material or background when projected back onto the source image. That identifies where an error is visible. The placement trace identifies which surface supplied the coordinate.

| Source image | Extras | Extra over spruce | Required spruce misses | Placement source of the extras over spruce |
| --- | ---: | ---: | ---: | --- |
| Workshop, plaster, high right | 19 | 7 | 26 | Four from a false raised strip above the canopy, two from the measured canopy top, one from nearby-face recovery |
| Shopfront, plaster, low left | 21 | 3 | 9 | Two from the underside plane, one from nearby-face recovery |
| Galleries, plaster, low left | 39 | 7 | 6 | Three from the underside, two from an upper horizontal face, one from a vertical proposal, one from horizontal-face recovery |
| Portico, plaster, low left | 30 | 4 | 0 | All four from the underside plane |

None of these 21 plank-associated extras came from the texture-crop seeds that caused the Courtyard corner error. Shopfront and Portico generated zero such seeds. Galleries generated 14 and all belonged at their exported coordinates. Workshop generated 45 and one was extra, over mostly background pixels.

## Workshop horizon and canopy

The distant horizon strips `mask-6` and `mask-13` covered 2,691 background pixels and four cobblestone pixels in total. Their 14 provisional coordinates all disappeared before export. The more consequential strip, `planarity-new-11-0`, sits just above the spruce canopy. It contains 2,360 background pixels and one spruce pixel. The placer interpreted it as a horizontal plane at local height 6, one block above the real canopy. Twelve of its proposals survived into the export as extras, and a later pass supplied a thirteenth adjoining extra.

| Workshop placement replay | Correct | Missed | Extras |
| --- | ---: | ---: | ---: |
| Ordinary baseline | 211 | 54 | 19 |
| Omit the two distant horizon masks | 211 | 54 | 20 |
| Omit the false strip above the canopy | 234 | 31 | 7 |
| Omit all three | 234 | 31 | 7 |

Omitting the near-canopy strip recovered 22 spruce positions and one cobblestone position, removed 13 extras and introduced one new extra. All previously correct positions survived. The recovered spruce positions bring Workshop from 34 to 56 of its 60 visible plank blocks. This is a diagnostic using a known mask ID. A production rule needs source-image evidence to distinguish background from a real narrow overhang or platform.

## Other plank structures

The dominant reference underside in Shopfront and Portico is at local height 5. Galleries' is at height 4. Placement chose the correct height in all three. The following table gives strict correct / missed / extra after rerunning placement with each primary underside omitted or its provisional plane moved one block. Other surfaces retain their normal inputs, and downstream placement can react to the change.

| Source image | Baseline | Omit underside | One block higher | One block lower |
| --- | --- | --- | --- | --- |
| Shopfront | 105 / 64 / 21 | 102 / 67 / 18 | 102 / 67 / 24 | 104 / 65 / 18 |
| Galleries | 188 / 86 / 39 | 176 / 98 / 40 | 182 / 92 / 44 | 185 / 89 / 40 |
| Portico | 148 / 61 / 30 | 101 / 108 / 27 | 113 / 96 / 48 | 79 / 130 / 34 |

The false placements are mixed among real blocks on the same plane. As a separate diagnostic, trimming its exported candidate cells whenever less than 75% of the projected face fell inside the underside mask would remove two extras and nine correct blocks across those three views. The next question is which outer boundary or adjoining visible face constrains each overextended row. A universal height change or mask-coverage threshold failed this comparison.

## Evidence and replay

The four layouts contribute 24 saved views at this baseline. Workshop and Galleries each contribute four. Shopfront and Portico each contribute eight. The machine-readable audit records pooled counts across these views, source and pack hashes for the four selected images, each test's correct positions gained and lost, and extra placements added and removed. [Scored audit](plank-audit.json)

From the reconstruction checkout with the frozen geometry fixtures, run `python benchmark/reviews/workshop-horizon-01/replay-plank-audit.py`. Add `--layout workshop`, `shopfront`, `galleries` or `portico` for a single source image. The replay verifies its ordinary controls against archived block coordinates, repeats the known-mask tests and checks the saved scores. To repeat the pre-foreground behavior after later algorithm changes, run the script from a checkout at `53252df` with the script and its adjacent JSON supplied together. [Replay script](replay-plank-audit.py)
