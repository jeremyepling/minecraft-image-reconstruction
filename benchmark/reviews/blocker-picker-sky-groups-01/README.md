# Match only the selected face of each block

On the frozen Java 26.3 suite with the shipped sky exclusion, each matcher ran once on the same selected face of every eligible group. Selection uses effective width in the original screenshot, then projected area as a tiebreaker. A group whose widest face is under five pixels is listed separately.

| Matcher | Automatic grid (101 blocks) | Recorded corners (562 blocks) |
| --- | ---: | ---: |
| Atlas | 16 | 439 |
| Rendered exposure | 74 | 425 |
| Texture | 101 | 433 |
| Texture and local shade | 101 | 431 |

The automatic grid produced complete faces in 12 of 56 views. These were 94 polished-andesite and seven stone-brick observations, all singleton groups. The recorded-corner diagnostic supplied nearby faces around visible corners. It contained 177 two-face and seven three-face blocks. The grouper recovered all 198 true pairs with zero false joins. Of the 672 block observations, 110 had only thin faces and were excluded from the material score.

Texture is the first choice for another automatic-crop trial. Atlas led the recorded-corner comparison, and the narrow automatic material mix supports keeping the current matcher choices available. [Scoring breakdown](scores.json) · [Predictions and input fingerprints](predictions.json.gz)
