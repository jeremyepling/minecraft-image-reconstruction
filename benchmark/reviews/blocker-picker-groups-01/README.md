# One material pick per visible block

The counted-face material path groups neighboring faces across different surfaces when projected cube geometry supports a convex shared edge. A three-face group requires support on all three edges. Ambiguous contacts stay separate. Each block takes its material from the eligible face with the greatest effective width in original pixels, using projected area to break ties. A linked thin face inherits the ID. When every member falls below five pixels the group remains unresolved. The counts report lists face indices, selection, geometric evidence, inherited IDs and block totals. This adds no unobserved faces or placements.

## Fixed 56-view comparison

| Matcher | Automatic, independent or grouped (106) | Recorded corners, independent (562) | Recorded corners, grouped (562) |
|---|---:|---:|---:|
| Atlas | 15 | 413 | 439 |
| Rendered exposure | 70 | 397 | 425 |
| Texture | 106 | 411 | 433 |
| Texture with local shade | 106 | 408 | 431 |

Counts are per eligible physical block appearance in a single view. The independent method succeeds if all eligible faces of that block pick the correct ID. A block must have at least one face five source pixels wide to enter either denominator.

The ordinary counter gave 106 complete faces across 12 of 56 pictures, all south-facing, spanning 99 polished-andesite and seven stone-brick observations. It never supplied a joinable pair. To test joins on the same pictures, the diagnostic supplies recorded two-dimensional face outlines in neighborhoods centered on visible corners. Coordinates and material IDs are held separately for scoring. It contains 177 two-face blocks and seven three-face blocks. The grouper matched all 198 true pairs and added zero incorrect pairs. Of the 672 block appearances, 110 had only thin faces. The texture-with-shade matcher additionally supplied the correct inherited ID on 59 thin faces. These scores describe this fixed, reviewed set and the deliberately corner-rich diagnostic.

[Examples](examples.png) · [Scores and input fingerprints](scores.json) · [Predictions](predictions.json.gz)

## Reproduce

From this checkout, run with the benchmark Python environment:

~~~powershell
$cal = 'C:\code\minecraft\workspace\builds\mcir-port-validation\final-cli-26.3'
python -m benchmark.group_fixture runs/group-proof/input
python -m benchmark.group_matchers --faces runs/group-proof/input/faces.json runs/group-proof/input/diagnostic-faces.json --calibration $cal --out runs/group-proof/predictions.json.gz --workers 4
python -m benchmark.group_metrics runs/group-proof/input runs/group-proof/predictions.json.gz runs/group-proof/scores.json
~~~

An ordinary single-screenshot counts build can use -MaterialMatch atlas, exposure, shape or shape-shadow. For example:

~~~powershell
.\Invoke-WallRecovery.ps1 picture.png -Out runs/my-blocks -MaterialMatch exposure -Atlas "$cal/atlas" -Library "$cal/face_textures/softened-library.npz"
~~~
