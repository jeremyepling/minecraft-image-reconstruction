# Can MobileSAM separate walls?

**MobileSAM offers useful boundaries for vertical walls.** On the same 56 screenshots, Wall Separator found 423 of 598 eligible surfaces. Letting two or more MobileSAM masks divide a wall raised that to 444, with all 423 previous matches retained. Applying the same rule after junction and underside recovery raised that combination from 448 to 471. Each score counts distinct visible surfaces in the individual pictures.

| Surface Finder | Development, 20 pictures | Reviewed regression, 36 pictures | Total | Guesses without an eligible match |
| --- | ---: | ---: | ---: | ---: |
| Wall Separator | 114 / 156 | 309 / 442 | 423 / 598 | 203 |
| Wall Separator with MobileSAM splits | 117 / 156 | 327 / 442 | 444 / 598 | 211 |
| Junctions followed by underside recovery | 120 / 156 | 328 / 442 | 448 / 598 | 197 |
| Junctions, underside recovery, then MobileSAM splits | 122 / 156 | 349 / 442 | 471 / 598 | 210 |

The last column includes guesses that overlap a real surface smaller than the eligibility cutoff. The two comparisons each retain all previously matched surfaces and preserve the original foreground exactly. Both still cover 601 of the 16,630 sampled opening pixels. Twelve of 98 openings remain at least half covered.

## What the new model saw

The probe used one screenshot at a time. A foreground point every 32 original image pixels and a point near each existing wall mask's center went into MobileSAM. Each point produced up to three shapes. The program retained the numerical mask at each four-pixel scoring sample. Identical screenshot pixels and prompting masks shared one inference.

The complete candidate pool contains a matching mask for **105 of the original 175 misses**. This is a reference-assisted upper bound. It takes the best shapes from 42,503 proposals across the suite. Filtering by the model's confidence, stability, building coverage and overlap reduced the pool to 686 suggestions. Of these, 261 matched eligible surfaces, including 75 that Wall Separator had missed. The remaining 425 show the cost of accepting the whole list. The filtered additions were all vertical. In 71 cases, the Wall Separator had merged the missing surface with a parallel surface at a different depth. No raw candidate recovered a missed underside.

## How an actual split works

The original wall must have a near-horizontal predicted normal. Two or more high-confidence MobileSAM masks must each cover at least 10% of that wall and 32 scoring samples. Each mask must fall at least 60% inside the wall. They can overlap at most 5% of the smaller mask and must together explain 80% of the wall. Their seeds expand within the original mask. They cannot spread across a hole or an opening. An unreachable area stays unresolved. Each connected child keeps its own perimeter and receives a fresh sampling chart when the existing chart fitter succeeds.

On development data, requiring 90% or 70% individual containment proposed zero splits. The 60% gate recovered three surfaces, lost zero and added one unmatched guess. We froze this setting before scoring the 36 regression views. All eight buildings had been reviewed in preceding experiments. They measure regression on the fixed layouts.

With Wall Separator, 20 of the 21 new matches came from a mask that had merged parallel walls at different depths. The addition raised correctly assigned samples from 992,737 to 1,024,123 out of 1,132,348. The junction and underside combination went from 1,015,101 to 1,042,178 correct samples and from 348 to 377 correct connections out of 606. Each run inherited the original relative-depth information. These comparisons grade visible surface boundaries.

The Galleries crop shows the front pillars and beam in the same view before and after splitting the wall. All colored lines come from predictions. [Prediction crop](../mobile-walls-01/prediction-crop.jpg)

## Reproduce

The archive has the compressed masks and prompt metadata for all 52 unique images. A duplicate image in another fixture slot retains its own score. Use the archived masks to replay either layout without loading model weights or opening Minecraft. From the reconstruction checkout on this machine, run the following PowerShell commands. Each output directory should be new.

```powershell
$py = '../mask-scalelsd/runs/env/Scripts/python.exe'
$probe = 'benchmark/reviews/mobile-regions-01'
foreach ($split in 'development','regression') {
    & $py -B -m benchmark.mobile_wall_split --probe $probe --out runs/mobile-walls-replay --split $split
    & $py -B -m benchmark.mobile_wall_split --probe $probe --junction-stack benchmark/reviews/junction-underside-01 --out runs/mobile-combined-replay --split $split
}
```

The official TinyViT model SHA, source revision and dependency versions are in the manifest. For a fresh inference and scoring run with the pinned official checkout and weights, run the following command. On the RTX 4080, the timed replay spent 18.1 seconds in MobileSAM and 17.8 seconds placing prompts and processing masks across all 52 unique images. Scoring took another 5.2 seconds. The wall-only split consumed 0.60 seconds of cached processing across 56 pictures.

```powershell
& '../mask-scalelsd/runs/env/Scripts/python.exe' -B -m benchmark.mobile_region_probe --source ../surface-recovery-round/runs/mobile-sam-source --out runs/mobile-probe-replay
```

Per-picture comparison records contain exact gained and lost reference IDs and a link to each image-only prediction. The proposal IDs in each model-screen match index the packed binary masks in its corresponding screenshot archive. [Model screen and input hashes](manifest.json) · [Raw and filtered matches by picture](cases.json) · [Wall split results](../mobile-walls-01/regression/cases.json) · [Full combination results](../mobile-combined-01/regression/cases.json) · [Official MobileSAM source](https://github.com/ChaoningZhang/MobileSAM)
