# Early wall visibility vote

This placement experiment gives original upright-wall candidates a second vote on rays obscured by elevated cubes proposed exclusively by a source-image-confirmed pavement opening. It checks the same face direction and minimum of four samples as the ordinary vote. Every original vote still counts. The elevated candidates remain available to the downstream pipeline.

All 204 frozen Java 26.3 inputs were screened. Only the right and left Colonnade 01 views triggered the second vote. Replaying the rest of placement and decoding both new exports produced identical block coordinates to their baseline exports. The other 202 source images had no qualifying opening.

On the right view, the second vote initially recovers both missing wall candidates at local `(0,1,6)` and `(1,1,6)`. Both survive seam ownership. The subsequent call to `fully_occluded_cells` removes them because the elevated cubes still exist when that check runs. Final strict F1 on that view is 77.06% and facade-priority F1 is 92.63%.

| Fixed 104 views | Before | After | Correct | Missed | Extra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict F1 | 91.0741% | 91.0741% | 20,728 | 2,974 | 1,089 |
| Facade-priority F1 | 93.2022% | 93.2022% | 18,907 | 1,669 | 1,089 |

The pools outside the fixed suite also tied on every source image. The historical 112-view strict F1 stays 90.3122%. Keep the early-vote rule experimental. A useful recovery must still be valid after cleanup. [Per-frame source hashes, triggered views, and pooled counts](summary.json)

Run `python -m benchmark.window_wall_vote_only --workers 5` in this checkout to rescore the two outputs and check the trigger across the whole suite.
