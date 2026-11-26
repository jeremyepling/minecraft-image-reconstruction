# Counting blocks with texture-guided crop corners

The ordinary counter got 95 exact visible spans and 50 wrong spans on the 56 previously reviewed clear-noon views. We kept each complete baseline estimate, searched crop corners using the spacing of repeated texture in two outer strips, and checked that spacing on a center strip reserved across all tested corners. Training and checking used single-image RGB, surface normal, mask and camera-lens assumption. Each chart retained only pixels from the given surface. The recorded corners, visible extents and one-to-one surface matches were used for scoring.

| Setting | Correct | Wrong | Eligible without complete answer | Unmatched complete proposals |
|---|---:|---:|---:|---:|
| Ordinary counter | 95 | 50 | 453 | 10 |
| Texture-guided corners with a reserved strip | 121 | 62 | 415 | 13 |
| Same method plus the atlas confirmation rule | 99 | 52 | 447 | 10 |

Settings were committed before running the 20 development pictures (Shopfront and Wing) and remained frozen for the other 36. Development counts were 17/13, 22/14 and 19/13 respectively. Remaining regression counts were 78/37, 99/48 and 80/39. These are familiar buildings. Material IDs were available only to the scorer.

**Portico spruce beam.** On plaster low-left with a smooth-stone floor, the fitted crop showed a repeat every 32 and 33 pixels in two separate outer bands. The central band confirmed it with a correlation of 0.709. The resulting 18 by 1 estimate matched the recorded size. The atlas chose the wrong texture, and the middle strip rejected that match. The default and atlas-gated rules both left the width unresolved and returned five for the height. The one-by-five cobblestone pillar and white concrete wall in that same image remained unresolved. Across all Portico views, the ordinary counter got 20 correct / 10 wrong, the repeat-only method 29 / 15, and the atlas-gated method 22 / 12.

**What the atlas accepted.** Six proposals passed the whole-tile check against half and double spacings, texture rotation, translation and an independent central strip. Four were exact, including two 1 by 5 Portico surfaces in different images. Two were wrong: a masonry Portico beam measured 15 by 1 instead of 18 by 1, and a plaster Portico post measured 2 by 5 instead of 1 by 5. A convincing local texture pitch cannot fix a biased visible extent.

**Recommendation.** Keep the atlas check experimental. It retained four of 26 additional correct repeat-only estimates and two of twelve newly wrong ones. Test repeat-only acceptance on the additional families before choosing a default. Neither method adds world positions or fills hidden faces. Separate per-layout scores, the chosen charts, atlas margins, held-strip scores and per-surface estimates are in [scores](scores.json), [cases](cases.json), and the [run manifest](manifest.json).

```powershell
python -m benchmark.phase_corner_round --out runs/my-phase-corners
```
