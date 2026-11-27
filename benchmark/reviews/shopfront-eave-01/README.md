# Tiny gaps inside Shopfront’s eave

In the low-left plaster source image the thin dark band is a visible underside. The small orange rings in its outline enclose holes in the underside mask. On the light gray concrete floor, all nine holes examined between x=690 and 1040, y=374 and 400 belonged to that same underside, totaling 220 source pixels. The conservative repair recovers three of them, totaling 100 pixels. On smooth stone it recovers another 28 pixels. [Zoom the source images, both masks, and the recorded underside](eave-closeups.png)

The same source image has a different problem at ground level. In the 100 × 40 patch starting at (550,575), 2,154 pixels belong to the structure. The old masks cover 506 of them and include zero pavement pixels. Recovering the remaining 1,648 requires finding the wall’s outward boundary.

## Tested rules

All candidates read the full source image and its existing estimated masks. Recorded visible geometry was read only after selecting patches. Counts below cover the 104 fixed structure views. “Paired pixels” counts agreement between a predicted surface and a recorded surface with an overlap match of at least 0.50 IoU. Changes in covered background count individual masks.

| Candidate | Patches acted on | Matched surfaces gained / lost | Paired pixel change | Covered background change |
|---|---:|---:|---:|---:|
| Delete small suspect pieces | 33 | 0 / 9 | −4,662 | −611 |
| Delete small pieces resembling the same-row margins | 2 | 0 / 0 | 0 | −254 |
| Assign enclosed pieces or holes to a neighboring mask | 415 | 0 / 9 | +2,252 | +1,239 |
| Require a similar texture color distribution and 80% surrounding support, or reject matching background | 174 | 0 / 0 | +4,485 | +466 |
| Limit texture agreement to holes within an elongated downward-facing surface | 47 | 0 / 0 | +376 | 0 |
| Require continuous surrounding support three pixels deep | 9 | 0 / 0 | **+148** | **0** |

Deleting small suspect pieces removed 8,565 actual visible pixels, including 1,338 on Shopfront. Same-row background color removed 51 pavement pixels in masonry Shopfront, but on the second affected view it also dropped 15 genuine pixels. The 80% eave-only rule assigned eight pixels to the neighboring plane. Each of the 148 pixels accepted by the final rule belonged to its host, including on Portico and Twin Spire. [Per-image evidence, candidate actions, source fingerprints and native reference hashes](cases.json)

The final rule tests a downward-facing mask at least four times wider than tall. It considers an enclosed hole of 4–1,024 native pixels, with no occupied pixels inside. The entire three-pixel ring must already belong to the host. It compares the lower quartile, median and upper quartile of each color channel inside the hole with the ring, with a maximum difference of eight on the 0–255 scale. Both lateral image margins at the same height supply background samples. If 15% or more of the hole resembles both margins, it abstains. Other protected cases include a cut reaching the outline, an independently detected face inside the hole, and two surfaces claiming the same ring. The ring requirement was tightened after reviewing Workshop and Portico. The additional 100 source images supplied a subsequent cross-check.

| Affected source view | Holes | New correctly assigned visible pixels |
|---|---:|---:|
| Fixed · gray concrete Shopfront, plaster low-left | 3 | 100 |
| Fixed · smooth stone Shopfront, plaster low-left | 3 | 28 |
| Fixed · smooth stone Portico, plaster low-left | 1 | 8 |
| Fixed · Twin Spire 3, cool low-left | 1 | 4 |
| Fixed · Twin Spire 3, earth low-left | 1 | 8 |
| Additional · Arcbridge 3, cool low-left | 2 | 124 |
| Additional · Watchpost 2, cool high-right | 2 | 20 |
| Additional · Platform Shelter, copper low-left | 1 | 24 |

## Placement

The repaired visible outline drives the finder overlay and source crop. Each changed surface saves its previous support in `measured_eave_support` for the Block Counter and placement. Testing the same repairs as direct depth evidence changed Arcbridge from 200 to 257 placed blocks, losing four correct blocks and adding 61 extras including eight hidden blocks. On Watchpost it gained eleven correct blocks, lost four and added one extra. Neither input’s measured dimensions changed. The original grid/depth support keeps those layouts intact. [Paired placement and unrestricted-mask ablation](placement.json) · [Ordinary command checks, source hashes and verified exports](ordinary.json)

The fixed suite retains 20,619 correct positions, 3,083 misses and 1,089 extras, or 90.8126% strict F1. Facade-priority retains 18,798 correct, 1,778 required misses and 1,089 extras, or 92.9145%. Both deltas are zero. Each changed view is checked against its prior exported block-state map. Inputs and controls for other views come from the pavement release. [Previous placement release](../pavement-local-01/README.md)

## Replay

From this checkout, with the registered frozen fixtures and prepared Java 26.3 atlas installed, run the following in PowerShell. Set `$atlasPath` to the atlas directory in the stable checkout.

```powershell
python -m benchmark.eave_patch_audit --all-cohorts
python -m benchmark.eave_patch_release inputs
python -m benchmark.eave_patch_release placements
$cases = Get-Content runs/eave-final/tasks.json -Raw | ConvertFrom-Json
foreach ($case in $cases) {
    .\Invoke-WallRecovery.ps1 -Image $case.path -Through placement -Atlas $atlasPath -Out (Join-Path runs/eave-final $case.id)
}
python -m benchmark.eave_patch_release verify
python -m benchmark.eave_patch_sheet
python -m unittest tests.test_eave_pinholes
```
