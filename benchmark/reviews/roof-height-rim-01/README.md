# Measuring a roof course from the photographed perimeter

This experiment tests three image-only ways to choose one integer roof height before placing a patch of visible blocks. All models were frozen on 72 development screenshots in the six training families. The starting geometry is the exported, strict texture-span default shipped at `54c9fee`. Scoring uses the same translation for the whole structure. [Per-frame decisions and visible-only development scores](development.json)

| Rule | Main height follows visible patch | No main height | New visible cells | Extras | Extras at hidden positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Horizontal perimeter and grounded support tops | 62 / 66 | 6 | 32 | 11 | 1 |
| Require both independent grounded tops and existing support columns | 36 / 36 | 36 | 12 | 1 | 1 |
| One-course vertical roof band and adjacent face | 39 / 42 | 30 | 14 | 3 | 2 |

The first method snaps a detected horizontal perimeter to nearby RGB transitions. It tests different heights for the same complete patch using integer-grid alignment over the perimeter, estimated tops of vertical masks that touch the floor, and existing vertical stacks. Each added block needs a photographed horizontal-face witness after all proposed cells compete in the same rendering. The stricter method additionally tries replacing an old cell only if every one of its previously visible rays is explained by the incoming roof patch. No replacement passed that test during development.

The independent band test fits both long edges and the lateral extent of a photographed vertical face. A fit spanning exactly one vertical course can identify the roof height even if a horizontal face was missed. Nearby horizontal masks can use this vote when the same 3D edge projects to them. Coverage of an actual source-facing surface is measured independently for each new block.

A reported main height refers to the candidate's selected patch. For development grading, a single real course must occupy more than half of that detected mask in the dense visibility labels, among faces of the proposed orientation. The first rule selected three ambiguous main patches and one wrong unambiguous height. The band test selected three wrong main heights. The stricter rule gained cells on one family only. The 18 authored canopy probes in the coordinator's evaluation test the separate question of whether a selected patch belongs to the main canopy roof.

The three adapters live on `experiment/roof-height-rim`. Reproduce from a checkout containing the previous comparison outputs:

```powershell
python -m benchmark.roof_height_rim_dev --out runs/development
python -m benchmark.roof_height_rim_dev --exchange --out runs/exchange
python -m benchmark.roof_height_rim_dev --band --out runs/band
python -m benchmark.roof_height_rim_labels --runs runs/development runs/exchange runs/band --out benchmark/reviews/roof-height-rim-01
python -m unittest discover -s tests -p test_roof_height_rim.py
```

`--base` can supply another location for the hashed `roof-span-round/runs/conservative` predictions. All three adapters accept only the image, detected surface and count data, estimated camera and exported local coordinates. An absent camera or unsupported height leaves the initial build intact.
