# A learned gate for additional visible inset masks

The final rule found five previously unmatched back-wall surfaces in the locked 204 images, beyond the existing paired-return finder and its scoped texture pass. Two are in one development Sawtooth view and three in two previously reviewed fixed Facade views. Every pixel transferred by the final rule belonged to the corresponding photographed back. All five new back counts matched the extent of its visibly represented unit faces, and each host kept its original count tuple, including unresolved entries. No existing matched surface lost IoU. The other 201 images produced the same surfaces. This is a surface and count result; no learned aperture or camera depth is passed to block placement.

| Photographed structure | New backs | Back mask IoU | Independently recounted dimensions |
|---|---:|---|---|
| Development Sawtooth 3, cool, high right | 2 | 76.45%, 63.69% | 1 × 5, 1 × 5 |
| Fixed entrance confirmation, Facade oblique right | 2 | 92.70%, 76.06% | 3 × 4, 3 × 3 |
| Fixed mixed corner, Facade oblique left | 1 | 78.94% | 2 × 3 |

The fixed cohort gained 18,618 correctly paired native pixels, and development gained 3,520. Across the 20 alternate views, four new layouts/views, and four independent shelter and gate views, the guarded rule abstained throughout. The original heldout Facade right that motivated the return finder already had its three back walls recovered by the strict method. The additional Facade results above are different images.

## Training and controls

Only 35 image-derived candidate hypotheses from six development layouts participated in training. Nineteen had a new, parallel, genuinely recessed physical surface at native IoU ≥0.5 and purity ≥0.8. Surface correctness did not require the hypothetical integer depth to be correct. A depth-two classification tree trained on these labels produced one accepted leaf at probability threshold 0.90. The learned feature is side-return IoU >0.684603; the leaf contained 18 positive samples out of 18 and its smoothed training fraction was 0.95. This fraction is not a calibrated confidence on a new material or layout. Leaving out one layout accepted six correct supplemental proposals; leaving out an entire building family accepted none. The screened 104 fixed views are regression evidence, and the final count/collar variants were investigated after examining earlier fixed outcomes. These views are not an untouched estimate of future model quality. [Training and grouped cross-validation](model.json)

Four photographed sides must each show at least 55% host support around the rim. At least 90% of the back proposal must remain assigned to the same host. Existing foreground overlap, prior strict return assignments and repeated anchor groups veto duplicate ownership. The frozen rule keeps a two-native-pixel collar on the host, requires a fully resolved ordinary RGB recount to agree with the span of visible cells on the candidate’s projected grid, and requires that span to agree using one- and four-pixel cell support. The candidate grid can veto a count; it cannot fill one in. Recounting the residual host must leave all its reported dimensions identical. The new outline records unresolved depth and keeps the original host ID as its parent. Its companion event is diagnostic and never authorizes local placement.

| Variant after the strict returns and scoped texture | Development gained / new unmatched | Fixed gained / new unmatched | Alternate gained / new unmatched | Count and ownership outcome |
|---|---:|---:|---:|---|
| Complete-rim and overlap checks, highest depth margin, no tree | 6 / 1 | 3 / 0 | 2 / 0 | An additional unmatched Sawtooth fragment |
| Tree, complete rim and overlap | 6 / 0 | 3 / 0 | 2 / 0 | Counted one 1 × 3 back as 4 × 12; two alternate back widths undershot by one |
| Also verify the proposed back cell span | 2 / 0 | 3 / 0 | 0 / 0 | All five new back counts correct, but 60 pixels previously owned by their correct host transferred and one unresolved host height became an incorrect 6 |
| Final two-pixel collar, count check and identical host tuple | 2 / 0 | 3 / 0 | 0 / 0 | All new back counts correct, zero misplaced transferred pixels, unchanged host counts |

The tree prevents one development error that the other early geometry gates alone admit. The count check, collar and stable host are responsible for the stronger ownership and count result. The full rule is a candidate for the Surface Finder default. Keep broader learned admissions experimental; five successful new masks across three already examined views do not establish transfer to unseen families. [No-tree control](no-tree.json) · [Surface-only chooser](results.json) · [Count guard](count-guard.json) · [Final outcomes and frame IDs](collar.json)

## Reproduction

Run `python -m benchmark.inset_mask_chooser_train` to fit the model on the 72-view development split. Use `python -m benchmark.inset_mask_chooser_eval` followed by the separate `--no-tree`, `--count-guard`, and `--collar` runs to reproduce the comparisons. The comparison uses the coordinator's frozen raw inputs and the per-image strict-return and scoped-texture baseline counts. The shared checkout is `../facade-inset-round`.

`python -m benchmark.inset_mask_engine_parity` checks frame IDs across all 204 views and compares accepted masks pixel-for-pixel with the engine port. Five workers were active in the timing pass. The native viability check took median 0.9 ms, and 134 images qualified for camera-based consideration. Re-estimating those cameras from RGB took median 400 ms and all matched the archived estimates. The chooser took median 144 ms and p95 254 ms across all 204 frames excluding camera fitting. Reusing the pipeline camera avoids repeating that fit. All accepted masks preserved the research footprint. [Per-frame engine outputs and timings](engine-parity.json)
