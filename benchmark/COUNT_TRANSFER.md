# Using repeated textures and neighboring surfaces to count blocks

**Recommendation. Promote texture-guided straightening as a fallback for incomplete block counts.** Across 80 saved views it supplied 40 new exact widths and heights and 18 new wrong sizes. Every previously exact size survived. It also returned five additional complete guesses on unmatched surface proposals. The production counter still runs on `main`. The two challengers, atlas check and review sheet are available as experiments.

[Zoomable comparison, one column per building](reviews/count-transfer-comparison-01/comparison.jpg) · [Every surface and picture in the 56-view suite](reviews/count-transfer-comparison-01/cases.json) · [Results on the 24 separate views](reviews/count-transfer-check-01/README.md)

| Method | 56 familiar views, correct / wrong | 24 additional views, correct / wrong | No complete size across 896 eligible appearances | Complete guesses on unmatched proposals |
|---|---:|---:|---:|---:|
| Ordinary counter | 95 / 50 | 59 / 34 | 658 | 25 |
| Previous outline-based crop adjustment | 118 / 62 | 70 / 41 | 605 | 29 |
| Texture-guided crop, verify the repeat on unused pixels | 121 / 62 | 73 / 40 | 600 | 30 |
| Same crop, also require a vanilla atlas match | 99 / 52 | 62 / 34 | 649 | 25 |
| Borrow another surface's grid | 99 / 54 | 60 / 38 | 645 | 27 |

A correct size requires both dimensions at the matched visible surface. An incomplete answer includes one unknown axis, two unknown axes or an eligible surface that the finder missed. The cross-building test kept all angles and palettes together. Cloister and Twin Spires had previously tested the Surface Finder outlines and were kept out of tuning for these counting methods. Cloister gained three correct sizes and three wrong ones with texture-guided crops; Twin Spires gained eleven correct and three wrong. [Frozen inputs, production masks and atlas](reviews/phase-corners-01/manifest.json) · [Independent-family manifest](reviews/count-transfer-check-01/manifest.json)

## Portico

On the low-left plaster view, the image-only fallback picked a crop with a spruce repeat every 32 to 33 pixels in two outside strips. The center strip confirmed the spacing. It counted the beam as **18 by 1**. Requiring a matching atlas tile discarded that correct result. The cobblestone pillar and white concrete wall in this particular view still lacked a reliable repeat. Atlas verification was selective on the extra families (three new correct, zero new wrong), but across the 80 views it retained only seven of the 40 additional correct sizes. [Crop trials](reviews/phase-corners-01/README.md)

On a different Portico view, a tuff pillar supplied the scale for a second tuff pillar. The transfer correctly measured **1 by 5** and placed the second pillar **four blocks to the right** of the first on their shared plane. On Galleries, borrowing from cobblestone made a white concrete wall **9 by 3** instead of **2 by 3**. Four of five transfers on Twin Spires crossed an actual two-block setback. A matching direction and plausible spacing did not establish the depth needed for placement. These runs measured offsets on the proposed plane. They did not produce new world-coordinate builds. [Transfers, positions and negative controls](reviews/coplanar-transfer-01/README.md)

## Where borrowing scale could still help

The visible reference joins touching blocks on the same plane, including blocks with different materials. In the original 56 views, 374 matched surface appearances had incomplete sizes. Of those, 112 shared a recorded plane with another matched surface, 34 had a neighbor with a complete count, and 28 had a neighbor with a correct count. Only three of those 28 had a correctly counted neighbor whose materials were entirely different. This scoring-only inventory counts separate visible regions; it excludes textured patches inside one continuous wall. A useful next trial would measure grid spacing on a textured patch inside a single detected wall, then extend it across a lower-contrast part while checking alignment at the far edge. [Opportunity by building and frame](reviews/count-transfer-comparison-01/opportunities.json)

## Replay

Run from the repository root with the fixed exterior fixtures and released Visible Builds v1 validation and test ZIPs available. Supply `--dataset` if the ZIPs are outside `runs/visible-builds-v1`. Use a fresh output directory.

```powershell
python -m benchmark.count_transfer_round --out runs/my-count-transfer
```

The predictor uses one screenshot, the existing image-derived surface masks, normals and relative depth, and the versioned atlas for the atlas check. Original model inference is reused. The scoring pass opens the saved camera, block labels and geometry after predictions. The single command replays both frozen trials, the independent-family check, the donor opportunity audit and the portrait sheet.
