# Same-facing votes on smooth visible walls

In Courtyard’s high-right plaster source image on smooth stone, a pale connected patch contains 46,667 pixels. Of the 15,244 already claimed, one wall mask owns 14,004 and a narrow mask facing the same way owns 1,228. A third mask owns 12. The largest mask alone supplies 91.87% of the known pixels, below the existing 98.5% requirement. Allowing same-facing wall masks to vote together gives the sponsor 99.92% directional support. A small right-angle contact stays a separate face.

The two original masks keep their own outlines, sampling charts, counts and depths. Only unassigned visible pixels can extend the winning mask. Color must continue from a source swatch on the sponsor; the atlas must support a smooth material and the same-row image margins must reject background. A two-pixel interior margin protects the silhouette. Candidate 3D blocks are accepted individually by projecting their visible faces back onto that patch. The earlier single-owner rule runs first and every position it places is preserved.

| Changed source image | New correctly assigned surface pixels | Newly correct required positions | New extras |
|---|---:|---:|---:|
| Smooth stone · Courtyard plaster high-right | 30,402 | 27 | 0 |
| Smooth stone · Courtyard plaster low-left | 5,706 | 5 | 0 |
| Light gray concrete · Courtyard plaster high-right | 1,228 | 0 | 0 |
| Light gray concrete · Portico plaster high-right | 81 | 0 | 0 |

One previously missed reference surface now meets the 0.50 IoU threshold on high-right Courtyard. The other three extend already detected walls. Across the 104 fixed views, 37,417 newly assigned native pixels belong to their host; zero land on a neighboring surface or background. Camera and visible-world labels only grade finished outputs. The input to the rule is one source image and its predicted surfaces. [Per-view surface IDs, coordinates, hashes and previous/updated placement counts](cases.json)

The 100 additional views include training, alternate material/floor and separately captured layouts. None supplied an eligible new upright-wall patch. An earlier diagnostic that also allowed horizontal faces would have drawn 101 pixels onto background on Watchpost at the final atlas stage. Only upright sponsors whose camera-space normals satisfy |y| ≤ 0.5 can pool votes. Each supporting normal must have a dot product of at least 0.985 with the sponsor. Tests cover a horizontal slab and a pair of differently facing walls. [Fallback and ownership checks](../../../tests/test_common_wall_votes.py)

## Placement

The previous release added 77 correct blocks on eight fixed views; all remain present. This incremental release adds 32 more, across two different views, with zero lost positions, extras or hidden additions. High-right Courtyard goes from 159 correct, 33 required misses and eight extras to 186 correct, six required misses and eight extras. Low-left Courtyard goes from 159 correct, 16 required misses and eleven extras to 164 correct, eleven required misses and eleven extras.

| Fixed 104-view placement | Previous F1 | Current F1 | Correct | Missed | Extra |
|---|---:|---:|---:|---:|---:|
| Strict | 90.9974% | 91.0741% | 20,728 | 2,974 | 1,089 |
| Facade-priority | 93.1179% | 93.2022% | 18,907 | 1,669 | 1,089 |

The eight retired wall close-ups are excluded from these numbers. Including them for historical comparability gives 90.3122% strict F1. [Pooled and registered scores](summary.json) · [Two incremental saved exports](default-exports.zip) · [Previous eight exports](../smooth-local-placement-01/summary.json) · [Ordinary CLI verification on all eleven affected or preserved builds](ordinary.json)

## Replay

Use the fixed local fixtures, the Java 26.3 atlas, and a Python environment with the project dependencies. Set `$atlasPath` to the prepared atlas in the stable checkout.

```powershell
python -m benchmark.smooth_joint_placement
python -m benchmark.smooth_joint_placement --inputs
$cases = Get-Content runs/smooth-joint-placement-01/tasks.json -Raw | ConvertFrom-Json
foreach ($case in $cases) {
    $out = Join-Path runs/smooth-joint-placement-01/public $case.id
    .\Invoke-WallRecovery.ps1 -Image $case.path -Through placement -Atlas $atlasPath -Out $out
}
python -m benchmark.smooth_joint_placement --public
python -m unittest tests.test_common_wall_votes tests.test_smooth_atlas_outline tests.test_smooth_extent_placement
```
