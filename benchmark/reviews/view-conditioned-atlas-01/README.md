# Camera-conditioned vanilla atlas

Keep `experiment/view-conditioned-atlas` experimental. Integrating each source pixel's projected texture footprint reduced material accuracy on the fixed suite and decorative buildings. The small increase on Visible Builds v3 came mostly from giving the existing material decision a second vote. Surface boundary evidence helped some buildings and damaged others when fed back into the complete pipeline.

## Material decisions on the ordinary exported geometry

These Java 26.3 experiments use the frozen ordinary placements and the estimated camera for the same screenshot. Across 288 views every starting decoded pack reproduced the archived strict, facade, assignment and block-ID scores exactly. The experimental source-only reranker was frozen at `83a8589` before scoring any labels. No model was trained and no thresholds were selected on test structures. Held-out layout and split names are retained per row. The 104 historical views are existing regression evidence.

The ordinary matcher already includes perspective UV coordinates, isotropic mip levels, exposure, orientation, atlas phase offsets and image-palette votes. Both new arms start from its emitted block ID. They compare the same face-legal ID mapping, 18 exposure/color-shortlisted tiles plus the incumbent's legal sprites, eight sprite orientations and native source pixels. The point arm uses a projected center texel. The footprint arm samples a 3×3 array of subpixel rays through the camera into the face. Both fit directions on the same two interleaved sets of samples, accept a second opinion only above the frozen correlation and margin thresholds, and use the dominant photographed face. This tests the incremental effect of camera-conditioned spatial averaging inside the same second-opinion policy. The direct comparison with the ordinary matcher also includes the added vote.

The basis is the Java 26.3 atlas with 448 opaque, untinted sprites mapped to 384 full-cube block IDs. The atlas SHA-256 is `49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950`. The prepared catalog is reused within each scoring process. Source camera position, height, projected scale, viewing angle and subpixel phase determine the taps on demand. This permits continuous views from the same versioned tiles. All 1,630 observed partial-block targets across the 40 decorative views have IDs outside the catalog, representing 42 distinct stairs, slabs, walls, gates, fences and trapdoors. They remain in the denominators.

| Cohort | Views tested/available | Strict F1 ordinary / point / footprint | Facade F1 ordinary / point / footprint | Correct ID at matched positions, ordinary / point / footprint |
|---|---:|---|---|---|
| Fixed | 104/104 | .846818 / .832846 / .820105 | .885099 / .876029 / .873121 | 19,274 / 18,956 / 18,666 of 20,773 |
| Visible Builds v3 | 144/144 | .061561 / .062108 / .062126 | .065006 / .065585 / .065622 | 7,089 / 7,152 / 7,154 of 9,299 |
| Decorative Builds v1 | 40/40 | .434295 / .417964 / .415064 | .453351 / .436454 / .433842 | 5,691 / 5,477 / 5,439 of 7,120 |

All candidate IDs were serialized to an ordinary structure datapack and decoded before grading. The ordinary geometry chose one whole-build integer translation per screenshot. The material-only arms retained every coordinate. Fixed geometry strict F1 is .912678, with 20,773 correct positions, 2,929 missed and 1,046 extra. Fixed facade geometry F1 is .934194. Geometry F1 change is zero in all three cohorts. Assigned/fallback export counts are 21,809/10 fixed, 16,157/1 v3 and 9,282/0 decorative. Empty camera/placement results remain in the totals, including 108 v3 and 10 decorative images.

The footprint arm changed 1,436 ordinary fixed IDs, 1,606 v3 IDs and 794 decorative IDs. Point and footprint differ on 1,057, 1,044 and 403 respectively. A **reference-only per-coordinate chooser** allowed to keep the incumbent or take the footprint ID reaches strict F1 .855517, .063680 and .439789. That recovers 198, 244 and 72 additional exact IDs over ordinary placement if supplied the answers. This is headroom for this pair of IDs only. It leaves missing coordinates, absent catalog IDs and shared wrong predictions unresolved.

## Viewing conditions

Bins use the estimated camera distance to the chosen face center and the angle between its normal and the eye direction. The respective change in exact ID count from point to footprint is shown below. Only eligible faces on correctly located cells enter the bin, so these denominators exclude camera misses and unsupported or unscored faces.

| Estimated condition | Fixed | v3 | Decorative |
|---|---:|---:|---:|
| Near, under 20 blocks | −62 / 2,120 | +2 / 96 | 0 / 29 |
| Midrange, 20 to under 40 blocks | −230 / 15,949 | +11 / 4,915 | −38 / 6,411 |
| Far, at least 40 blocks | +2 / 341 | −11 / 2,349 | 0 eligible |
| Frontal, under 30° | −3 / 3,910 | +3 / 2,285 | +5 / 2,022 |
| Moderate, 30 to under 60° | −4 / 8,525 | 0 / 3,098 | −3 / 2,075 |
| Oblique, at least 60° | −283 / 5,975 | −1 / 1,977 | −40 / 2,343 |

The [source and rendered sprites](source-and-rendered-faces.png) show an oblique polished deepslate face recovered by footprint integration and a more frontal polished andesite face lost to dispenser. The gallery shows RGB fits to the selected ID on the observed face. [Source IDs, cells and hashes](example-sources.json) and [all viewing bins](view-bins.json) accompany it.

## Local block counts and surface ownership

On the first pack-producing capture of each represented expanded layout, the secondary probe used 11 v3 views and 9 decorative views. The probe counted visible extents on estimated axis-aligned planes supporting a common material ID. It offered one-block changes to each eligible dimension and a quarter-block shift of either origin, keeping the existing grid as an option. The native pixel scorer checked whether neighboring projected samples fell across an actual native face boundary within the same recorded surface.

Of 399 eligible groups, point sampling retained every ordinary grid. Footprint sampling retained 398, selected zero width or height changes, and shifted one 24×9 plane by −.25 block. The shrine gained one extra incorrect exported cell. The other 19 views retained ordinary positions. Across eligible grid samples, true detected / missed / extra source seams changed from 32,110 / 34,173 / 27,473 to 32,080 / 34,203 / 27,533. On the 11 v3 probe views, strict position-plus-ID F1 changed from .210565 to .210552. The nine decorative probe views tied at .453108. These are conditional post-placement grid proposals. Their sampling uses observed pixels and never closes unobserved faces.

A separate decision compared photo fit to each of two neighboring first-hit planes on the same source screenshot. Of 933,727 modeled foreground samples (two-pixel stride), 75,704 had a neighboring plane offered at a seam. The footprint arm switched 20,287 samples, correcting 6,897 native assignments and damaging 4,093. A further 2,227 remained correct and 7,070 remained wrong. Each predicted plane's reference name was determined from the common untouched interior after prediction. The point-only arm achieved +2,715 net corrected sites, versus +2,804 with footprints. The incremental footprint effect was +89 sites (853 improved, 764 damaged relative to point). Footprint versus ordinary improved 13 views, regressed six and tied one. On validation, the three harbors changed by +53, +155 and zero, while chapel lost 87. The two test gardens gained 292 and 445. This experiment assigned existing foreground samples, with no new surface discoveries or background claims. [Per-view decisions and counts](geometry-probes.json)

## Feeding surface changes back into an ordinary build

The first pass provided projected plane ownership. A second pass transferred a switch only where the old owner agreed with an existing pre-inset Surface Finder mask, the projected plane had a majority correspondence and both masks shared a resolution. The ordinary topology check preserved connectivity and holes. All inference inputs were screenshot-derived. Cached image-model fields reproduced their original provenance. Both a no-op and an edited run executed the ordinary counter, placer, terminal checks, material assignment and pack writer on each image. All three no-op decoded inventories matched their ordinary controls.

| Decorative screenshot | Support sites changed | Count/phase rows changed | Export coordinates added / removed | Strict geometry F1 | Strict position-plus-ID F1 |
|---|---:|---:|---:|---|---|
| Market court 1, train | 262 | 0 | 81 / 90 | .64559 → .54730 | .53910 → .26689 |
| Hillside chapel 1, validation | 309 | 2 | 0 / 0 | .73806 → .73806 | .62603 → .62603 |
| Walled garden 1, test | 285 | 2 | 48 / 4 | .56698 → .63557 | .31153 → .58601 |

These three traces show why more correct boundary pixels do not establish an improvement to downstream reconstruction. Market court lost placements even though its public count and phase values tied. Its inputs to placement still include the changed surface supports. [Feedback traces and pack hashes](mask-feedback.json)

## Reproduce

Run from this experimental checkout at ordinary-equivalent base `6c22345`, with the Java 26.3 fixtures installed.

```
python -m benchmark.view_conditioned_atlas_run --workers 7
python -m benchmark.view_conditioned_atlas_geometry
python -m benchmark.view_conditioned_atlas_feedback
python -m benchmark.view_conditioned_atlas_evidence
```

The material runner verifies the RGB and baseline pack hashes and checks that every decoded candidate retains the ordinary cells. [Frame manifest and leaderboard](cases.json) contain per-frame cohort, source and pack hashes, unchanged geometry fingerprint, counts and all arms. [Pooled results](summary.json) include assigned and fallback counts and the per-site oracle. Private runs under `runs/view-conditioned-atlas` retain the candidate packs, cell-level evidence and full feedback replays. The runner expects the saved expanded baseline under the sibling `full-expanded-controls/runs/expanded-baseline-01`, archived fixed source and reference handoffs under `typed-structure-scores/runs/typed-structures`, the checked-in typed export archive, and the stable checkout's released RGB inputs and Java 26.3 atlas. Normal evaluation reads reference labels only after the frozen source decision. No ordinary implementation or configuration was changed.
