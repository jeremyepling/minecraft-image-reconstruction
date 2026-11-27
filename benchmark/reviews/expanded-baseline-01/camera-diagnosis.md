# Why the floor camera rejected these images

The ordinary camera stage rejected **108 of 144 Visible Builds v3 views** and **10 of 40 Decorative Builds v1 views** on revision `1ca8822`. The camera uses the floor to recover two perpendicular directions and the physical size of a Minecraft block. If both floor methods abstain, placement has no metric camera. The same layout at the same angle gave the same acceptance decision in both material palettes for all 92 pairs. All 184 captures use 1440 × 900 pixels and a 55° vertical field of view. [Per-image measurements and full-build verification](camera-diagnosis.json)

| Cohort | Low-left failures | High-right failures | Rear-oblique failures |
| --- | ---: | ---: | ---: |
| Visible Builds v3 | 30 / 48 | 38 / 48 | 40 / 48 |
| Decorative Builds v1 | 8 / 20 | 2 / 20 | No such view |

All 18 Factory views and all 18 Castle views failed. Harbor failed on 8 of 18. In the decorative cohort, all eight Stable Yard views and all eight Walled Garden views succeeded.

Each standalone camera replay matched the completed ordinary build's matrix or abstention and its recorded reason on all 184 views. Surface Finder still found at least five candidate surfaces on every failed view. On v3 its median was 23 surfaces on failures and 23.5 on successes.

## What the photographs have in common

In v3 the camera was typically farther from the subject in failed views. The median eye-to-target distance was **58.6 blocks on failures and 31.7 on successes**. The floor texture became finer in those views. At 90% of image height, the finer of its two horizontal periods spanned a median of **1.99 pixels per texel on failures and 4.19 on successes**. The search accepts periods from approximately 2.13 to 40 pixels. In 56 failed v3 views, at least one true floor direction fell outside that window in more than 45% of the final band's rows. All 36 supported v3 views kept both directions in range in at least 55% of those rows. [Track search](../../../engine/camera_texel_floor.py)

We also checked what the program could see in the eight strongest candidate frequencies on each sampled row of that band. In **92 of 108 failed v3 views**, at least one real floor direction appeared too infrequently to meet the 55% row requirement. The same was true in two of 36 supported views, which could also be accepted from an earlier band. The final reason was missing tracks on 85 views, disagreement on the floor horizon on nine, incomplete texel coverage on ten, ambiguous atlas alignment on two, and a flat rectified floor on two. These are the recorded reasons on the final attempted band. [Ordinary atlas camera and retry order](../../../engine/camera_atlas_floor.py)

The decorative buildings had a median longest edge of 22 blocks in both the failed and supported groups. All ten failed photos passed the ordinary floor-texture matcher when supplied with the recorded camera's floor coordinates. In six, the estimator had followed **the same direction twice**, once at its normal frequency and once at twice that frequency. This leaves a line where the matcher needs a two-dimensional patch. The four remaining failures were both angles of Hillside Chapel 2 in both palettes. On low-left the estimator retained just one direction. On high-right the two fitted tracks put the horizon 53 pixels apart, beyond the 13.5-pixel tolerance.

## Source examples

- **Market Court 1, cool palette.** Low-left had about 6.0 and 14.8 image pixels per floor texel along a horizontal scan line. The estimator picked one axis and its double. Using the true floor grid, both halves matched light-gray concrete with correlations above 0.99. High-right recovered the two independent axes. Market Court 2 also succeeded from low-left in both palettes.
- **Pillars 03, marble palette.** In the high-right view the finer true period was about 1.2 pixels. Even supplying the true floor grid produced correlations of only 0.34 and 0.36 against the ordinary minimum of 0.75. This source loses useful fine texture at capture resolution.
- **Viaduct 01, riverstone palette.** Low-left had the same doubled track as Market Court. High-right and rear-oblique succeeded. The building was recognized well enough to produce surface candidates in all three views.

Across both cohorts, all 16 cases reporting incomplete floor texel coverage selected two directions within one degree of parallel in true floor coordinates. Twelve of those used a frequency and its double. With the true grid, all 184 source images had at least three eligible observations of each of the 256 texture positions in each half of the image. Exact coordinates also passed the subsequent sprite and phase test in **70 of the 108 failed v3 views**. They still failed that test in 38 v3 photos. The reference-camera intervention changes only the atlas sampler and measures image evidence for calibration. It does not measure a gain in final placement.

## Reproduce

Run `python -m benchmark.expanded_camera_diagnosis --camera-replay` in the baseline worktree with its Java 26.3 atlas and frozen fixtures available. The camera replay receives only the source RGB and ordinary assets. The separate diagnostics use the recorded projection to calculate the true floor-plane derivative and sampler coordinates. The JSON records native margins, visible building coverage in the floor strip, paired decisions, actual replay-to-build agreement, and the projection measurements per image. [Diagnostic implementation](../../expanded_camera_diagnosis.py)
