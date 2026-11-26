# Count-stable photographic boundary repair

This follow-up used the same 56 previously reviewed pictures **after** seeing four new incorrect complete counts in the frozen photo comparison. It is a retrospective regression check. The same 4-pixel photographic change, then 2-pixel fallback, now also declines a new two-axis estimate where the incoming proposal had an unresolved axis. If neither collar preserves the complete estimates for a frame, all original masks stay. This protects existing decisions and can also prevent a genuinely new correct count.

| Across 56 pictures | Shipped | Photo with original count fallback | Photo with stricter fallback |
|---|---:|---:|---:|
| Surface matches out of 598 | 517 | 519 | 519 |
| Correct / wrong complete sizes | 95 / 50 | 95 / 54 | 95 / 50 |
| Native paired boundary error, px | 3.8244 | 3.7125 | 3.7197 |
| Paired native mask overlap | 95.142% | 95.323% | 95.310% |
| Opening samples occupied | 511 | 481 | 482 |
| Correct connections | 426 | 428 | 428 |

Both alternatives kept all old matches and added Shopfront plaster low-left R3 and Workshop masonry high-right R19 on concrete. The stricter fallback added 17,427 target pixels and 2,131 other pixels on the originally matched surfaces, and removed 17,319 incorrectly assigned pixels and 1,485 real ones. Exact-size IDs remained identical to the shipped method. The softer fallback made four new complete estimates on Portico instead of abstaining, and all four were wrong. [Changed estimates and recorded widths and heights](count-transitions.json)

The extra frozen plane stage was tested separately. Against the original photo fallback it shaved another 0.0071 px off the mean edge distance and added 0.038 percentage points of mask overlap, retaining the same 519 matches, 95 correct / 54 wrong sizes and 481 opening samples. It did not remove the reason to test a stricter count rule. [Frozen composition](../edge-detail-stack-01/scores.json) · [Paired per-picture results for the stricter fallback](cases.json)

Use the reproduction command in the [manifest](manifest.json) with a fresh output directory. The 56 prediction archives, frame-level matched identities, gridded scores and native-resolution paired metrics are retained for independent review. The supplemental count transitions are a scoring-side diagnosis of pictures where the original fallback changed the count statistics.
