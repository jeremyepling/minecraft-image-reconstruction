# Smooth source pixels along a cut-off wall

The fixed benchmark has 104 complete structure views. The other 100 are a mixture of previously inspected and transfer controls. Each candidate reads a single source image and its ordinary predicted masks at revision `a95816d`. The runs use identical input hashes. Recorded plane IDs and visible block coordinates grade the result after prediction. Registry and Shopfront informed development. Checks that use their outcomes a second time are exploratory.

The gray Shopfront plaster wall has a visible patch at x=790–849, y=490–544. The existing mask covers 624 of 3,300 pixels there. Joining neighboring masks cannot cover the other 2,676. On the Registry high-right image, the corresponding question is the east-facing upper wall. A lower, parallel wall is recessed one block.

The first trial tested whether an unassigned horizontal or vertical span between two runs of the same mask had continuous RGB values. It increased the number of exact matched surfaces by two on the fixed set, but crossed onto 2,047 background pixels and did not reach the named Shopfront patch. Requiring two axes of support avoided these background errors and improved paired area by only 363 pixels.

The flat-color trial samples frequently repeated colors inside the source masks. It searches for connected regions with all RGB channels within three intensity levels of a seed. A candidate must terminate within the visible frame, contain at least 320 native pixels of its sponsoring mask, and have at least 98.5% agreement among its previously assigned pixels. Only unassigned, source-visible pixels can be added. The broad test recovered the whole 2,676-pixel Shopfront gap and 15,788 of Registry's 15,833 missing upper-wall pixels.

| Background handling | Fixed pixels on sponsoring plane / other plane / background | Other 100 views |
| --- | --- | --- |
| Closed region alone | 84,401 / 20 / 312 | 580 / 0 / 2,449 |
| Veto any similar margin chromaticity in the image | 0 / 0 / 0 | 569 / 0 / 0 |
| Compare frequent margin chromaticities over the entire region's image rows | 15,542 / 0 / 0 | 569 / 0 / 0 |
| Compare absolute color and shadow-darkened pavement only at the new pixels' rows | 45,879 / 18 / 312 | 569 / 0 / 507 |

These count pixels newly assigned to the original mask's dominant recorded plane. The one-to-one surface metric can change by more when a mask crosses the 0.50 intersection-over-union threshold.

## Placement

Every changed view was recounted and placed from the frozen upstream stages. Each control reproduced its archived exported coordinates. Facade-priority ignores only the specified thin horizontal-only targets, and charges every extra.

| How the flat-color trial changes placement | Correct gained / lost | Required gained / lost | Extras added / removed |
| --- | ---: | ---: | ---: |
| Rebuild globally, fixed 104 | 106 / 54 | 105 / 45 | 8 / 19 |
| Rebuild globally, other 100 | 1 / 7 | 1 / 7 | 2 / 0 |
| Add only locally supported cubes, fixed 104 | 77 / 0 | 77 / 0 | 0 / 0 |
| Add only locally supported cubes, other 100 | 0 / 0 | 0 / 0 | 0 / 0 |

The local rule projects the union of original and proposed cubes into the source image. A new cube needs at least 16 visible samples and 60% of its visible samples on newly supported pixels with the same face axis. The unchanged build continues to occupy its original locations during this visibility test. Registry high-right gained 43, three Shopfront blocks were recovered across two views, and Portico gained five. Courtyard received no new placements. The unrestricted method lost 41 required blocks on Registry low-left.

[Shopfront comparison](plaster-gap.jpg) · [Registry comparison](registry-right.jpg) · [Source IDs and surface scores](cases.json) · [Per-view placement](flat-placement.json)

## Reproduce

Run on the frozen inputs registered in the workspace. The benchmark uses its private outputs under the review directory.

```powershell
python -m benchmark.shopfront_outline_gaps
python -m benchmark.flat_extent_placement
python -m benchmark.shopfront_outline_gaps --transfer --out benchmark/reviews/shopfront-outline-gaps-01/transfer
python -m benchmark.flat_extent_placement --out benchmark/reviews/shopfront-outline-gaps-01/transfer
```

Add `--exterior-veto`, `--row-veto` or `--patch-veto` to the surface command with a separate output folder to replay each background ablation. The atlas and ordinary-entry follow-ups have independent reports.
