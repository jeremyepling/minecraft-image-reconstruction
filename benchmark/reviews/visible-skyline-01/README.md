# Recovering a wall up to the photographed sky edge

The guarded rule recovered 225 previously unclaimed native pixels on the Watchpost 3 low-left development view. Every added pixel belongs to the same recorded wall. The other 203 screenshots were unaffected, including all 104 fixed cases and the 28 alternate and transfer photos. This verifies one example of the correction. The tested rule has yet to fire on an independent building. [Watchpost screenshot and recovered pixels](watchpost.jpg) · [204-view grade](guarded/summary.json) · [Per-view results](guarded/cases.json)

The color-only trial followed a short run of matching pixels up to clear sky and gained 369 correctly assigned pixels on seven of 72 development photos. It also assigned 86 visible pixels to the wrong neighboring face. A roof above the eye can expose a vertical rim, while a roof below the eye can put a top face between the wall and the sky. The guarded method requires a front-facing wall normal and an upward-looking view, with the proposed edge above the estimated horizon throughout a 50–90° field-of-view interval. It requires at least twelve contiguous columns with an unambiguous owner, one to four unclaimed pixels, near-identical wall color in every proposed pixel, and a clear color step at the sky. The entire column above the edge must be sky. [Development ablation](development/summary.json)

The early Surface Finder applies the check before exporting its outlines and crops. The same Watchpost screenshot was run from RGB through placement using both revisions of the normal command. The saved counts and placement JSON were byte-identical. Decoding the datapacks returned the same 384 block positions and materials. Checks cover competing owners, openings, sky absence, view ambiguity and contradictory colors. [Ordinary-entry hashes](cli.json) · [Timing with the existing sky classification](timing.json)

```powershell
python -m benchmark.visible_skyline_extent --splits train --out runs/skyline-dev
python -m benchmark.visible_skyline_extent --below-rim --splits train validation test regression native old fresh extra new shelter_gate --out runs/skyline-guard
python -m unittest discover -s tests -p test_skyline_extent.py
```
