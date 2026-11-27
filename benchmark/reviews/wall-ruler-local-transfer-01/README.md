# How far can a verified wall ruler travel?

**Keep `experiment/wall-ruler-local-transfer` experimental.** On 13 original views, requiring a fresh texture measurement on each neighboring block eliminated eight extra Viaduct placements and kept 21 net correct placements across Castle 01 and Walled Garden. Applying the older whole-wall rule gained 125 net correct placements on the same views. On [eight additional frozen views](../wall-ruler-local-transfer-extension-01/README.md), local verification changed zero exports and the whole-wall rule gained two correctly typed blocks. Ordinary reconstruction at `e3bd220` remains the default.

## Results on the original 13 views

Each entry gives pooled placement F1 (strict / facade), followed by pooled position-plus-exact-ID F1 (strict / facade). Each exported build receives one geometry-selected integer translation, reused for both ID scores. New local arms were run through ordinary downstream placement edits and material export. Ordinary packs and previous native-single whole-wall results came from the [frozen ruler experiment](../local-wall-rulers-01/README.md). A fresh empty-hook replay matched all 13 released inventories before and after registration, and their translations and scores. All cameras were supported and every camera-supported image exported blocks.

| Views tested / available | Ordinary | Whole wall | Measured tiles only | Connected verification |
|---|---|---|---|---|
| Fixed 2 / 104 | 87.02/92.52 · 85.46/90.70 | same | same | same |
| Visible Builds v3 9 / 144 | 51.91/55.26 · 31.21/34.04 | 52.61/56.02 · 32.00/34.91 | same as ordinary | 52.05/55.41 · 31.23/34.06 |
| Decorative Builds v1 2 / 40 | 64.82/65.53 · 56.79/56.99 | 68.14/69.16 · 59.91/60.42 | 64.96/65.68 · 56.93/57.14 | 65.11/65.84 · 57.08/57.30 |
| Development 5 / 5 | 64.36/65.10 · 41.18/41.60 | 64.43/65.21 · 41.15/41.61 | same as ordinary | same as ordinary |
| Transfer 8 / 8 | 48.92/53.02 · 31.97/35.69 | 50.14/54.39 · 33.33/37.21 | 48.93/53.04 · 31.98/35.70 | 49.13/53.26 · 32.02/35.74 |
| All 13 | 54.24/57.48 · 35.14/37.87 | 55.05/58.37 · 36.01/38.83 | 54.25/57.49 · 35.15/37.88 | 54.38/57.63 · 35.18/37.90 |

Ordinary strict counts were 6,246 correct, 7,768 missed and 2,771 extra. Connected verification produced 6,267 correct, 7,747 missed and 2,769 extra. It assigned 9,035 IDs with one fallback. Of the 6,267 correct positions, 4,054 had the exact required ID (64.69%). The 33 measured first/held tile footprints grew to 1,063 proposed footprints across nine frames. Whole-wall reached the same nine frames. Local changes reached two final exports. Multiple certified tiles can vote for blocks already supplied by ordinary placement.

Each tuple below is correct positions recovered / lost, extras added / removed, after alignment against ordinary. A dash means all four counts were zero. Lighting/palette variants are separate screenshots, not pooled as new structures.

| Original source view | Whole wall | Measured tiles | Connected |
|---|---:|---:|---:|
| Amphitheater 02 ashlar high right | 1/0/1/0 | — | — |
| Viaduct 02 riverstone high right | 0/0/8/0 | — | — |
| Shrine 01 ochre low left | 3/0/0/0 | — | — |
| Observatory 02 seafoam low left | — | — | — |
| Harbor 02 kiln high right | — | — | — |
| Harbor 02 seafoam high right | — | — | — |
| Harbor 03 seafoam high right | — | — | — |
| Castle 01 ashlar high right | 102/1/12/2 | — | 29/10/1/1 |
| Castle 02 ashlar rear oblique | — | — | — |
| Facade oblique left | — | — | — |
| Colonnade 01 oblique right | — | — | — |
| Market Court 1 cool high right | 5/3/5/10 | — | — |
| Walled Garden 1 warm high right | 25/7/9/32 | 1/0/0/1 | 2/0/0/2 |

On Castle 01 an older strict carrier for the same surface sits at Z −17 while the verified ruler sits at Z 9. The whole-wall arm replaces that carrier. Both local arms above supplement ordinary strict votes, preserving the existing carrier even inside the measured footprint. A **targeted diagnostic** replaced only old strict votes whose entire sampled source footprint lay inside the new certified cells on that surface. Four old votes qualified. Votes for the same coordinate from other surfaces were preserved. The downstream export recovered 74 correct positions, lost two, added three extras and removed one. This yielded 78.56/78.59 placement and 48.56/48.58 position-plus-ID, versus 76.29/76.31 and 42.64/42.66 when supplementing. One old vote can change frontmost ownership for multiple provisional cells. This arm was chosen after the original Castle result and was subsequently frozen for the eight-view extension.

## What evidence admits a block?

The original branch supplied the source-only native single-sprite fit and independently held third-tile score. This experiment retains their exact candidate planes, scale, phase, thresholds and source pixels. Only accepted unit-pitch, zero-phase integer-plane candidates proceed. For a measured first or held tile, at least 80% of its sampled rays must land in the same proposed cell. All native pixel centers inside that complete projected cell must lie in the observed surface mask and at least 130 native samples must be available. Clipped tiles abstain. The projected unit must also pass the ordinary strict-cell mapping on the current placement mask.

The connected arm advances to four-neighbor cells one full cell at a time. Each new cell repeats the native-footprint support check. A central chart stripe selects an oriented side sprite from the source-color shortlist plus the ruler's initial sprite. At most 384 native pixels per tile are used. Reprojection samples four atlas subpixels per native pixel. Center-stripe grayscale correlation must reach .62 and untouched seam-side stripes .52. On those same held stripes, the integer phase must beat each of four quarter-block shifts by .03. Correlation tolerates constant brightness and offset; a different legal sprite can be selected per cell. This rule can abstain on untextured, heavily shaded, narrow or weakly resolved surfaces. Acceptance is source evidence of a grid, not a native face-ID proof for every neighbor.

Pixel-complete support and the existing strict map veto unsupported cells. The original carriers stay available outside the measured area. Chart cells reach the actual strict-cell handoff directly, avoiding the 50% overlap threshold that would drop small masks. All later ownership, ray, refinement and material stages run as before. [Viaduct](train-viaduct-02-riverstone-high-right.jpg), [Castle 01](test-castle-01-ashlar-high-right.jpg) and [Walled Garden](test-walled-garden-1-warm-high-right.jpg) show the source screenshots with measured footprints in amber and independently verified neighbors in cyan.

## Reproduce

Use the workspace OpenCV Python, verified 26.3 atlas, original control checkout and raw native-single candidate receipts from `experiment/local-atlas-wall-rulers`. The benchmark branch starts from `4942e2e` and shares ordinary runtime code with `e3bd220`. [Receipts with source, candidate, reference and pack hashes, strict-cell handoffs and decoded scores](exports.json)

```powershell
foreach ($split in 'development','transfer') {
    python -m benchmark.wall_ruler_local_transfer regions --split $split
    python -m benchmark.wall_ruler_local_transfer export --split $split
    python -m benchmark.wall_ruler_local_transfer noop --split $split
}
python -m benchmark.wall_ruler_local_transfer castle-overlap --split transfer
python -m benchmark.wall_ruler_local_summary
```

Full-cohort ordinary results carried forward from the previous release: fixed 104 placement 91.27/93.42, position+ID 84.68/88.51, strict 20,773 correct / 2,929 missed / 1,046 extras. Visible Builds v3 144 placement 46.63/48.04 and position+ID 28.04/29.08. Decorative Builds v1 40 placement 63.46/65.23 and position+ID 50.51/52.56. No complete-cohort challenger was run. The experiment contains benchmark code and receipts, with no change to ordinary builds.
