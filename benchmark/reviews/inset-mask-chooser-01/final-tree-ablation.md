# Does the tree improve the final surface result?

Switching off the learned gate produced exactly the same binary footprints for every surface in all 204 images. Both variants selected the same five visible back walls, assigned the same host and two return anchors, and reported the same independently recounted dimensions. All other inputs and guards were held fixed, including proposal order within equal-ranked groups, the complete rim, duplicate/ownership checks, the two-pixel collar, independent grid span validation and stable host count.

| Cohort | Views | Before final count, tree on / off | Final masks added, tree on / off | Identical final outputs |
|---|---:|---:|---:|---:|
| Development | 72 | 6 / 7 | 2 / 2 | 72 |
| Reviewed fixed structures | 104 | 3 / 3 | 3 / 3 | 104 |
| Alternate photos | 20 | 2 / 2 | 0 / 0 | 20 |
| New layouts/views | 4 | 0 / 0 | 0 / 0 | 4 |
| Shelter/gate | 4 | 0 / 0 | 0 / 0 | 4 |

The tree screened out one incorrect preliminary fragment on `train-sawtooth2-earth-high-right`. With the tree disabled, the independent RGB recount on that fragment returned 1 × 3 while the visible projected grid spanned 1 × 4. The physical check rejected it. The only observed effect of the tree was skipping this candidate's recount. No end-to-end timing advantage was measured.

The evidence supports the photographic and count safeguards alone as a default. Retain the decision model as a research comparison until an independent data set shows it improves the *final* output. This ablation changed no runtime files. Replay with `python -m benchmark.inset_final_tree_ablation`. The report lists both final native mask hashes on every active frame and records paired footprint and count equality everywhere. [Frame-level paired results](final-tree-ablation.json) · [Earlier comparisons](README.md)
