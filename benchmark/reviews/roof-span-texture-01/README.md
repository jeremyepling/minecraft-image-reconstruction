# Visible roof extents from texture and contacts

Starting from the exported coordinates on `main` at `99655be`, each photographed roof fragment or post top nominates horizontal strips along the two grid directions. Candidate positions can extend nine blocks from an observed fragment. Native image rays and the observed outline bound each strip. The candidate tests the texture on its horizontal face and vertical rim against calibrated atlas sprites and adjacent integer depths. A visible material change can select a new sprite on the same span. Every recovered cell must earn its own texture match against the pixels it will own in the combined build. A gap without evidence breaks the strip.

Both variants keep preexisting blocks. The guarded setting also accepts a single measured cell at a strong roof contact or between observed cells. The conservative variant requires two adjacent measured cells, a contact, and stronger correlations. Both check the final silhouette and protect the last visible pixel of each prior cell. Frozen entries are `engine.roof_span_texture:recover` at `b389083` and `engine.roof_span_texture:conservative` at `02623ed`. Commit `5140180` converts generated coordinates to the native integer type required by Minecraft export.

## Development

Only the 72 training views informed these settings. Each count compares additions to the baseline after its single whole-building translation. Training cases contain 2,370 missed visible cells. Across the six families, 825 were in the initial texture candidate pool along with 3,851 extras. The pool included 436 hidden positions with a coincidental match. Of the 154 true and 60 extra candidates selected by the guarded run rule, the final checks retained 124 true and 21 extras. [Per-photo counts](development.json)

| Development choice | Added correct | Added extras | Extras on hidden structure |
| --- | ---: | ---: | ---: |
| Strict texture runs | 39 | 0 | 0 |
| Less strict texture runs | 83 | 7 | 2 |
| Guarded runs and strong single-cell contacts | 124 | 21 | 5 |
| Conservative runs with silhouette guard | 39 | 0 | 0 |

## Roof failure after freezing

For this diagnostic, every new candidate is scored against the same two-post alignment as the baseline. Correct counts exclude already recovered cells. The last two rows describe absolute numbers remaining after each image check. Re-registering an entire output can give a different credited count if a new post changes the best alignment. [Per-view measurements and offending prior coordinates](roof-diagnostic.json)

| Stage | Ten fixed canopy views | Eight alternate canopy views | Canopy 06 left |
| --- | ---: | ---: | ---: |
| Missing visible roof cells | 166 | 90 | 54 |
| Nominated by roof strips | 166 | 90 | 54 |
| At least four sampled rays in front of existing cells | 127 | 59 | 54 |
| Preliminary atlas match | 92 | 42 | 38 |
| Selected by guarded contact and texture rule | 19 | 6 | 9 |
| Retain a texture match after joint rendering | 11 | 1 | 9 |
| Pass native silhouette guard | 11 | 1 | 9 |
| Retain all previously visible blocks | 4 | 1 | 2 |

On Canopy 06 left, seven correctly located roof cells failed only because they would obscure the last pixel of 16 distinct earlier placements. All 16 earlier placements were wrong under the common post alignment. Source labels establish this retrospective result. A subsequent experiment could compare the two texture hypotheses on the exact contested pixels before authorizing a replacement.

## Paired replacement experiment

The diagnostic above prompted a separate experimental entry, `engine.roof_span_occlusion:recover`, frozen at `18ac4ac`. Its input is the conservative set of proposed roof strips. A contested ray compares the incoming roof's sampled sprite against a fresh atlas fit using the previous face's projected texture coordinates. The new block must win on each disappearing cell's original pixels. An earlier cell can only be removed once it has no visible pixel left. These decisions are made entirely from the camera estimated from the photograph, the calibrated atlas, the exported positions and the photograph.

Across the 72 development images the conservative guard had rejected eight correct and five extra positions. The same-ray comparison recovered none of them with either tested cutoff. Both reproduced the conservative result exactly. The stricter setting requires a correlation of at least 0.78 and an advantage of at least 0.18 over the previous geometry. The looser cutoff used 0.65 and 0.12. Canopy 06 had already been inspected when proposing this experiment and counts as a regression fixture for it. [Per-image comparison](paired-development.json)

## Replay

Use the Visible Builds v2 data, frozen camera, surface and count measurements, and the calibrated Java 26.3 atlas described in the [baseline](../framed-depth-01/README.md).

```powershell
python -m benchmark.roof_span_texture_dev --out runs/roof-texture-development --workers 4
python -m benchmark.roof_span_texture_limits --out runs/roof-texture-canopy.json --workers 4
python -m benchmark.roof_span_pair_dev --out runs/roof-texture-pair --workers 4
python -m unittest tests.test_roof_span_texture
```
