# Post-projection sky veto

Run python -m benchmark.floating_projected_sky with the registered floating experiment checkouts. Each case verifies its source image and archived pack hash. The filter tests a 9 by 9 sample of each exported face with its recorded source grid and a strict majority requirement. It preserves the ordinary assembly path and join inputs. Out-of-frame samples give no sky vote.

Across 84 paired conditions the border-connected sky cue removed 10 extra positions and lost zero correct positions. Nine removals came from the six fixed floating sky views. Strict F1 rose from 33.8054% to 33.8085% across 84. Adding the exterior clear/cloud cue lost 13 net correct positions. Both arms leave the annotated Mill roof cloud in place. Keep the conservative filter as an experimental comparator and diagnose neutral gray cloud continuity.
