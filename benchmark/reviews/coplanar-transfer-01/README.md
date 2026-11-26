# Borrowing a block grid from another visible surface

The target keeps its own visible mask. A textured donor supplies the number of blocks per unit length on its estimated plane. The rule projects the target mask through that plane, accounting for perspective, and checks its four extents against block lines measured from the donor's visible outer corner. It can report an offset across and down the visible plane. A down-axis on a roof can represent horizontal world depth. Plane-to-plane setback and thickness are unresolved. The method uses the ordinary crop, MoGe surface direction, relative inverse depth and the donor's measured texture repeats. It uses no material IDs in prediction.

The earlier neighbor-scale trial at c0ec44a transferred depth at a shared corner between differently facing surfaces. This trial tests same-facing surfaces which can be separated by a gap and have different textures.

| 56 fixed views | Correct complete dimensions | Wrong complete dimensions | Unresolved or missed | Complete guesses on unmatched proposals |
|---|---:|---:|---:|---:|
| Ordinary counter on photo-refined outlines | 95 | 50 | 453 | 10 |
| Strict donor grid | 99 | 54 | 445 | 12 |

On the 20 development views the strict rule added two correct and one wrong pair. Looser texture, depth and boundary thresholds added three correct and six wrong pairs plus one unpaired guess. We froze the strict rule before running the remaining 36 reviewed views. There it added two correct and three wrong pairs, plus two guesses on unpaired proposals. All previous complete counts survived.

**Portico gives a genuine positive example.** On the smooth-stone floor, masonry palette, low-left angle, tuff pillar R16 supplies a grid to tuff pillar R17. R17 measured 1 x 5. Its predicted origin was four blocks right and zero down from R16. Both agree exactly with the saved geometry. A synthetic front-facing wall also successfully transfers a measured random RGB repeat to a flat recipient of a different color and rejects the same transfer when the target depth is moved.

**Parallel walls remain ambiguous.** Seven of ten selected transfers had matched donor and recipient IDs. Six of these pairs were on parallel planes at different depths and all six had different materials. The Wing spruce-to-white transfer correctly measured 13 x 8 despite its incorrect plane assignment. On Galleries, a cobblestone donor predicted 9 x 3 for a 2 x 3 white concrete wall. A paired donor and recipient were available for exact offset grading on only the genuine Portico pair. Three additional transfers had at least one unpaired proposal. For color/palette interpretation and frame IDs see cases.json.

Before final boundary and integer checks, the depth veto admitted 11 of 14 candidate pairs actually on a common plane, and admitted 49 of 121 parallel pairs at a different depth. The projected block corners alone could not resolve those false alignments. A connected reference surface already joins edge-adjacent coplanar block faces, even if they use different materials; separated pillars and disconnected visible patches are the main opportunity for this specific test. The 56 regression images provide no verified cross-material common-plane transfer with both proposals matched. The synthetic control verifies that the implementation permits it.

Keep this experimental. Before placing blocks from borrowed scale, improve the evidence for a common plane and test estimates of setback separately. For touching surfaces one useful next control would compare the depth discontinuity on both sides of their boundary in the native depth maps.

[Settings, checksums and per-frame index](manifest.json) and [all selected donors, offsets and scoring truth](cases.json). The script replays immutable image-derived predictions from texture-photo-count-01. The 56 image-only transfers took 0.28 seconds of cumulative processing on this run. Source model inference, file reading and scoring are excluded. All eight buildings are previously reviewed regression fixtures.

```powershell
python -m benchmark.coplanar_count_round --out runs/my-coplanar
python -m unittest discover -s tests -p test_coplanar_count.py
```
