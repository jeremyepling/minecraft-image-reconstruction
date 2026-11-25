# Can one outline separate a wall?

**Yes. The default Surface Finder now finds 501 of 598 scored surfaces on the 56 saved pictures.** The previous five-model Wall Recovery pipeline found 471. The new rule keeps all five models and lets one MobileSAM outline divide a wall when Depth Anything V2 sees a consistent step across its edge. Allowing that single cut helped on all eight buildings. [Zoomable comparison](reviews/single-wall-depth-01/comparison.jpg) · [Every picture](reviews/single-wall-depth-01/pictures.md)

## What changed in the pictures?

Look down the **Courtyard** column. The previous finder merged a ledge with the wall behind it. Both new methods separated it, taking the overlap with its recorded surface from 15% to 88%. The corresponding Portico face went from 16% to 83%. In the Galleries high-right plaster picture, both methods gained one wall and lost a different one. Its total alone hides that trade. The bottom of the sheet enlarges an actual proposed cut, marks where depth was sampled, and shows what each model decided.

| Method | Development, 20 pictures | Reviewed regression, 36 | Found / 598 | Extra guesses | Both block dimensions right | Wrong size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous Wall Recovery | 122 / 156 | 349 / 442 | 471 | 189 | 73 | 36 |
| One outline + Depth Anything V2 | **132** | **369** | **501** | **201** | 78 | 41 |
| Same outline + Apple Depth Pro | 130 | **369** | 499 | 205 | **83** | **37** |

The Depth Anything version gained 31 distinct matches and lost one. Average border error on matched surfaces fell from 5.71 to 4.32 source pixels. It assigned 93.3% of scored surface pixels correctly, up from 92.0%, and recovered 406 of 606 true connections, up from 377. Depth Pro gained 29 matches and lost the same one. It recovered 408 connections. All three methods matched 14 of 18 eligible undersides and covered the same 601 of 16,630 opening samples. The extra-guess count excludes guesses that pair with a real below-cutoff surface. [All geometry and dimension scores](reviews/single-wall-depth-01/scores.json)

**For surface accuracy, use Depth Anything V2. For block dimensions, Depth Pro scored highest.** Depth Pro kept all 73 previously correct sizes and added ten. Depth Anything added six and lost one. The ordinary Surface Finder uses the same outlines for its surface and count stages. Select `-SurfaceBoundary wall-recovery-single-pro` to use the Depth Pro outlines for a build. World-coordinate placement and in-game resemblance were outside this test.

| Building | Existing finder | One outline + V2 | One outline + Pro | Scored |
| --- | ---: | ---: | ---: | ---: |
| Courtyard | 51 | 53 | 53 | 56 |
| Crossing | 35 | 38 | 38 | 46 |
| Galleries | 80 | 82 | 81 | 98 |
| Portico | 119 | 130 | 130 | 148 |
| Registry | 26 | 27 | 28 | 34 |
| Shopfront | 90 | 98 | 98 | 120 |
| Wing | 32 | 34 | 32 | 36 |
| Workshop | 38 | 39 | 39 | 60 |

## What the models actually did

1. Each prediction began with the previous Wall Recovery masks. MoGe-2 supplied surface directions, Depth Anything V2 supplied relative depth, ScaleLSD v2 supplied wall junctions, M-LSD Tiny supplied underside lines, and MobileSAM suggested shapes. Both challengers saw the exact same high-confidence MobileSAM shapes. All 56 frame IDs, including repeated pixels in different fixture slots, retained their prior identity.
2. The new code considered one candidate shape on a vertical parent wall. It sampled depth immediately inside and outside the proposed edge and farther along each side to estimate the slope of a continuous plane. It required at least six sample groups and consistent evidence along at least 70% of the boundary samples. The proposed cut kept the original parent mask, including every gap. Each accepted piece and its remainder went through the ordinary Block Counter.
3. The second test substituted Apple’s original Depth Pro checkpoint for the **last depth vote only**. Depth Anything V2 still supplied all earlier stages. We fed Depth Pro the source RGB with unknown focal length. The checkpoint restored the native image dimensions before sampling. Both depth fields used the exact same pixel positions and the same rule, with relative depth normalized per picture. [Apple implementation](https://github.com/apple/ml-depth-pro) · [Checkpoint, numerical field hashes and timings](reviews/depth-pro-fields-01/manifest.json)
4. Settings were chosen on Shopfront and Wing and frozen before scoring the six other previously reviewed buildings. The reference geometry and camera were used for grading and labeled overlays. Runtime and model count were excluded from selection. Generating Depth Pro fields took 14.55 seconds across the 52 distinct screenshots on an RTX 4080, plus 12.23 seconds to verify and load the checkpoint. Replaying the boundary votes using cached fields took less than one second per model across the 56 frame slots on the measured run. [Frozen settings](reviews/single-wall-selection-01/manifest.json) · [Development choice](reviews/single-wall-depth-01/development-selection.json)

## Where the remaining misses came from

The new pass considered 66 possible cuts. At 31 of them, a majority of the paired samples crossed parallel recorded surfaces at different depths. Depth Anything identified the nearer side at 30 of those and Depth Pro at all 31. At another 31 proposed cuts, a majority of the samples landed on **one flat wall**. Depth Anything accepted five of those false cuts and Depth Pro accepted six. On the ten where a majority also crossed a change in block material, the models accepted three and five respectively. Four candidates had mixed or insufficient reference agreement. Good depth ordering alone did not ensure the correct surface partition.

Of the 41 previously missed surfaces with a matching MobileSAM suggestion, Depth Anything recovered 23 and Depth Pro recovered 22. The other gains included neighboring surfaces exposed as the remainder of a cut. Sixteen of the 41 had no matching outline reach the depth vote. Changing the depth model alone cannot move those outlines through the earlier filtering step. [Per-picture diagnostic](reviews/single-wall-depth-01/opportunities.json)

## Run the comparison or build one screenshot

```powershell
.\Invoke-SurfaceRound.ps1 -Round single-wall
```

This command regenerates predictions from the frozen numerical evidence, scores block counts, writes the per-picture grades and produces the portrait sheet in a new run directory. All 56 images are Java 26.3 screenshots captured in clear weather at noon. The replay runs with Minecraft closed and needs no checkpoint downloads. `-Round detection-sheet` redraws just the saved visual. [Run manifest](reviews/single-wall-depth-01/manifest.json)

```powershell
.\Invoke-WallRecovery.ps1 .\screenshot.png -Out .\runs\my-build
```

The ordinary build uses the selected single-outline method. To run the frozen control, pass `-SurfaceBoundary wall-recovery`. Fresh builds reproduced all saved masks and IDs. On the same smooth-floor Portico picture, the previous method found 14 of 17 surfaces, the new default found 15, and Depth Pro found all 17. [Build instructions](WALL_RECOVERY_BUILD.md) · [Fresh-image verification](reviews/single-wall-entry-01.json)
