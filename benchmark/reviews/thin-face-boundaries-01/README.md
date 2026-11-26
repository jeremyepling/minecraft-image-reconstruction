# Checking texture just outside a detected surface

Frozen candidate `0a53b11` extends the photographic witness by at most two native pixels at a mask boundary when adjacent RGB values differ by at most six per channel. Both runs used the frozen single-pass atlas classifier at .85 on identical inputs. Each table entry lists additional correct positions / extras relative to the pre-round default.

| Cohort | Original mask | Extended mask |
| --- | ---: | ---: |
| 72 training views | 314 / 15 | 305 / 13 |
| 112 fixed evaluation views | 475 / 11 | 476 / 10 |
| 20 alternate views | 246 / 9 | 243 / 6 |

The fixed-set difference is one correct block and one extra. The alternate cameras and training views both lost some correct additions. Keep the original mask in the default and retire this extension. [Pooled counts](counts.json)
