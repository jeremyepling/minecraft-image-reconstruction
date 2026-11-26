# Resolving a repeated wall edge

This placement experiment starts at `eeb6e13`. Adjacency 26 right is the development image. The 56 original exterior screenshots and the other 11 additional supported views give 67 comparison views.

For a grounded, contiguous candidate column, project the exposed side faces of the next column onto the disputed front faces. Keep the strict placements locked. A proposed removal requires a link at ground level and at least three linked rows. At each row, at least 40% of the front face and four stride-four samples must overlap the retained side and the input surface masks.

Project the floor under the disputed column into the original RGB image and compare it to the same texture positions in the neighboring exposed tile. Filter the rays using the predicted blocks. Require at least twelve in-frame samples, 60% of samples within 14 RGB levels in every channel, and a repeated brightness pattern (5th–95th percentile spread of 12 and clipped correlation of 0.20). A flat floor-color match is ambiguous. Re-render the remaining cells to check that at most one previously supported silhouette sample disappears per removed cell. Camera poses and source-world blocks enter the scorer after the export.

| Method | Adjacency right known duplicates removed (26) | Other incorrect coordinates removed | Correct lost, 68 views |
| --- | ---: | ---: | ---: |
| Floor color and three linked rows (`6d598f4`) | 16 | 78 | 0 |
| Add repeated texture and silhouette check (`01e4748`) | 8 | 7 | 0 |

The stricter rule left all 56 original exterior-suite exports unchanged. Two additional views changed. Adjacency right kept 257 correct cells and went from 67 extras to 59. Its source-camera silhouette overlap went from 94.603% to 94.621%. Corner right retained 105 correct positions and went from 137 extras to 130. It lost three source-camera building samples and silhouette overlap went from 88.495% to 88.468%. The broader rule had reduced that Corner overlap to roughly 84.75%.

Five synthetic controls cover simulated floor and wall textures, a separately supported neighboring front, an elevated strip, a flat-color ambiguity and a deletion that would uncover supported pixels. Run them with `python -m unittest discover -s tests -p test_seam_runs.py`. The shared evaluation harness on the benchmark PC is `workspace/builds/seam-review/replay.py`, with `--checkout` pointing at this worktree and `--out` pointing at a private run directory. Use `workspace/builds/seam-review/shape.py <control-run> <candidate-run>` to reproduce the silhouette comparison.
