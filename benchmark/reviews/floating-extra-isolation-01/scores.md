# Decoded placement by panel

Counts are TP / FP / FN. Percentages are micro precision / recall / F1. Each export gets one whole-build registration shared by geometry and IDs.

| Panel | Views | Arm | Metric | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| original | 38 | floating-default | strict | 1127 | 107 | 58311 | 91.33 | 1.90 | 3.72 |
| original | 38 | floating-default | facade | 1127 | 107 | 57587 | 91.33 | 1.92 | 3.76 |
| original | 38 | floating-default | strict + ID | 466 | 768 | 58972 | 37.76 | 0.78 | 1.54 |
| original | 38 | floating-default | facade + ID | 466 | 768 | 58248 | 37.76 | 0.79 | 1.55 |
| original | 38 | baseline | strict | 14065 | 5422 | 45373 | 72.18 | 23.66 | 35.64 |
| original | 38 | baseline | facade | 13992 | 5422 | 44722 | 72.07 | 23.83 | 35.82 |
| original | 38 | baseline | strict + ID | 1828 | 17659 | 57610 | 9.38 | 3.08 | 4.63 |
| original | 38 | baseline | facade + ID | 1817 | 17597 | 56897 | 9.36 | 3.09 | 4.65 |
| original | 38 | source-one | strict | 13824 | 5101 | 45614 | 73.05 | 23.26 | 35.28 |
| original | 38 | source-one | facade | 13753 | 5101 | 44961 | 72.94 | 23.42 | 35.46 |
| original | 38 | source-one | strict + ID | 1792 | 17133 | 57646 | 9.47 | 3.01 | 4.57 |
| original | 38 | source-one | facade + ID | 1787 | 17067 | 56927 | 9.48 | 3.04 | 4.61 |
| original | 38 | large-contraction | strict | 14065 | 5422 | 45373 | 72.18 | 23.66 | 35.64 |
| original | 38 | large-contraction | facade | 13992 | 5422 | 44722 | 72.07 | 23.83 | 35.82 |
| original | 38 | large-contraction | strict + ID | 1828 | 17659 | 57610 | 9.38 | 3.08 | 4.63 |
| original | 38 | large-contraction | facade + ID | 1817 | 17597 | 56897 | 9.36 | 3.09 | 4.65 |
| fresh | 12 | floating-default | strict | 263 | 12 | 7251 | 95.64 | 3.50 | 6.75 |
| fresh | 12 | floating-default | facade | 261 | 12 | 6531 | 95.60 | 3.84 | 7.39 |
| fresh | 12 | floating-default | strict + ID | 168 | 107 | 7346 | 61.09 | 2.24 | 4.31 |
| fresh | 12 | floating-default | facade + ID | 168 | 105 | 6624 | 61.54 | 2.47 | 4.76 |
| fresh | 12 | baseline | strict | 1600 | 1401 | 5914 | 53.32 | 21.29 | 30.43 |
| fresh | 12 | baseline | facade | 1509 | 1401 | 5283 | 51.86 | 22.22 | 31.11 |
| fresh | 12 | baseline | strict + ID | 360 | 2641 | 7154 | 12.00 | 4.79 | 6.85 |
| fresh | 12 | baseline | facade + ID | 357 | 2553 | 6435 | 12.27 | 5.26 | 7.36 |
| fresh | 12 | source-one | strict | 1575 | 1349 | 5939 | 53.86 | 20.96 | 30.18 |
| fresh | 12 | source-one | facade | 1480 | 1349 | 5312 | 52.32 | 21.79 | 30.77 |
| fresh | 12 | source-one | strict + ID | 342 | 2582 | 7172 | 11.70 | 4.55 | 6.55 |
| fresh | 12 | source-one | facade + ID | 330 | 2499 | 6462 | 11.66 | 4.86 | 6.86 |
| fresh | 12 | large-contraction | strict | 1603 | 1341 | 5911 | 54.45 | 21.33 | 30.66 |
| fresh | 12 | large-contraction | facade | 1512 | 1341 | 5280 | 53.00 | 22.26 | 31.35 |
| fresh | 12 | large-contraction | strict + ID | 360 | 2584 | 7154 | 12.23 | 4.79 | 6.88 |
| fresh | 12 | large-contraction | facade + ID | 357 | 2496 | 6435 | 12.51 | 5.26 | 7.40 |
| supplemental | 6 | floating-default | strict | 249 | 0 | 3004 | 100.00 | 7.65 | 14.22 |
| supplemental | 6 | floating-default | facade | 249 | 0 | 2673 | 100.00 | 8.52 | 15.70 |
| supplemental | 6 | floating-default | strict + ID | 69 | 180 | 3184 | 27.71 | 2.12 | 3.94 |
| supplemental | 6 | floating-default | facade + ID | 69 | 180 | 2853 | 27.71 | 2.36 | 4.35 |
| supplemental | 6 | baseline | strict | 900 | 356 | 2353 | 71.66 | 27.67 | 39.92 |
| supplemental | 6 | baseline | facade | 900 | 356 | 2022 | 71.66 | 30.80 | 43.08 |
| supplemental | 6 | baseline | strict + ID | 209 | 1047 | 3044 | 16.64 | 6.42 | 9.27 |
| supplemental | 6 | baseline | facade + ID | 209 | 1047 | 2713 | 16.64 | 7.15 | 10.00 |
| supplemental | 6 | source-one | strict | 854 | 324 | 2399 | 72.50 | 26.25 | 38.55 |
| supplemental | 6 | source-one | facade | 854 | 324 | 2068 | 72.50 | 29.23 | 41.66 |
| supplemental | 6 | source-one | strict + ID | 246 | 932 | 3007 | 20.88 | 7.56 | 11.10 |
| supplemental | 6 | source-one | facade + ID | 246 | 932 | 2676 | 20.88 | 8.42 | 12.00 |
| supplemental | 6 | large-contraction | strict | 900 | 356 | 2353 | 71.66 | 27.67 | 39.92 |
| supplemental | 6 | large-contraction | facade | 900 | 356 | 2022 | 71.66 | 30.80 | 43.08 |
| supplemental | 6 | large-contraction | strict + ID | 209 | 1047 | 3044 | 16.64 | 6.42 | 9.27 |
| supplemental | 6 | large-contraction | facade + ID | 209 | 1047 | 2713 | 16.64 | 7.15 | 10.00 |
| sky | 6 | floating-default | strict | 273 | 5 | 3768 | 98.20 | 6.76 | 12.64 |
| sky | 6 | floating-default | facade | 273 | 5 | 3682 | 98.20 | 6.90 | 12.90 |
| sky | 6 | floating-default | strict + ID | 101 | 177 | 3940 | 36.33 | 2.50 | 4.68 |
| sky | 6 | floating-default | facade + ID | 101 | 177 | 3854 | 36.33 | 2.55 | 4.77 |
| sky | 6 | baseline | strict | 901 | 2915 | 3140 | 23.61 | 22.30 | 22.93 |
| sky | 6 | baseline | facade | 896 | 2915 | 3059 | 23.51 | 22.65 | 23.07 |
| sky | 6 | baseline | strict + ID | 173 | 3643 | 3868 | 4.53 | 4.28 | 4.40 |
| sky | 6 | baseline | facade + ID | 171 | 3640 | 3784 | 4.49 | 4.32 | 4.40 |
| sky | 6 | source-one | strict | 888 | 1542 | 3153 | 36.54 | 21.97 | 27.45 |
| sky | 6 | source-one | facade | 883 | 1542 | 3072 | 36.41 | 22.33 | 27.68 |
| sky | 6 | source-one | strict + ID | 193 | 2237 | 3848 | 7.94 | 4.78 | 5.97 |
| sky | 6 | source-one | facade + ID | 191 | 2234 | 3764 | 7.88 | 4.83 | 5.99 |
| sky | 6 | large-contraction | strict | 890 | 1542 | 3151 | 36.60 | 22.02 | 27.50 |
| sky | 6 | large-contraction | facade | 885 | 1542 | 3070 | 36.46 | 22.38 | 27.73 |
| sky | 6 | large-contraction | strict + ID | 193 | 2239 | 3848 | 7.94 | 4.78 | 5.96 |
| sky | 6 | large-contraction | facade + ID | 191 | 2236 | 3764 | 7.87 | 4.83 | 5.99 |
| holdout | 6 | floating-default | strict | 91 | 0 | 4522 | 100.00 | 1.97 | 3.87 |
| holdout | 6 | floating-default | facade | 90 | 0 | 3862 | 100.00 | 2.28 | 4.45 |
| holdout | 6 | floating-default | strict + ID | 82 | 9 | 4531 | 90.11 | 1.78 | 3.49 |
| holdout | 6 | floating-default | facade + ID | 82 | 8 | 3870 | 91.11 | 2.07 | 4.06 |
| holdout | 6 | baseline | strict | 700 | 2120 | 3913 | 24.82 | 15.17 | 18.83 |
| holdout | 6 | baseline | facade | 679 | 2120 | 3273 | 24.26 | 17.18 | 20.12 |
| holdout | 6 | baseline | strict + ID | 126 | 2694 | 4487 | 4.47 | 2.73 | 3.39 |
| holdout | 6 | baseline | facade + ID | 125 | 2674 | 3827 | 4.47 | 3.16 | 3.70 |
| holdout | 6 | source-one | strict | 685 | 2119 | 3928 | 24.43 | 14.85 | 18.47 |
| holdout | 6 | source-one | facade | 664 | 2119 | 3288 | 23.86 | 16.80 | 19.72 |
| holdout | 6 | source-one | strict + ID | 121 | 2683 | 4492 | 4.32 | 2.62 | 3.26 |
| holdout | 6 | source-one | facade + ID | 120 | 2663 | 3832 | 4.31 | 3.04 | 3.56 |
| holdout | 6 | large-contraction | strict | 700 | 2120 | 3913 | 24.82 | 15.17 | 18.83 |
| holdout | 6 | large-contraction | facade | 679 | 2120 | 3273 | 24.26 | 17.18 | 20.12 |
| holdout | 6 | large-contraction | strict + ID | 126 | 2694 | 4487 | 4.47 | 2.73 | 3.39 |
| holdout | 6 | large-contraction | facade + ID | 125 | 2674 | 3827 | 4.47 | 3.16 | 3.70 |
| holdout2 | 6 | floating-default | strict | 134 | 0 | 2972 | 100.00 | 4.31 | 8.27 |
| holdout2 | 6 | floating-default | facade | 129 | 0 | 2696 | 100.00 | 4.57 | 8.73 |
| holdout2 | 6 | floating-default | strict + ID | 35 | 99 | 3071 | 26.12 | 1.13 | 2.16 |
| holdout2 | 6 | floating-default | facade + ID | 35 | 94 | 2790 | 27.13 | 1.24 | 2.37 |
| holdout2 | 6 | baseline | strict | 499 | 901 | 2607 | 35.64 | 16.07 | 22.15 |
| holdout2 | 6 | baseline | facade | 488 | 901 | 2337 | 35.13 | 17.27 | 23.16 |
| holdout2 | 6 | baseline | strict + ID | 111 | 1289 | 2995 | 7.93 | 3.57 | 4.93 |
| holdout2 | 6 | baseline | facade + ID | 111 | 1278 | 2714 | 7.99 | 3.93 | 5.27 |
| holdout2 | 6 | source-one | strict | 419 | 380 | 2687 | 52.44 | 13.49 | 21.46 |
| holdout2 | 6 | source-one | facade | 408 | 380 | 2417 | 51.78 | 14.44 | 22.59 |
| holdout2 | 6 | source-one | strict + ID | 106 | 693 | 3000 | 13.27 | 3.41 | 5.43 |
| holdout2 | 6 | source-one | facade + ID | 106 | 682 | 2719 | 13.45 | 3.75 | 5.87 |
| holdout2 | 6 | large-contraction | strict | 473 | 444 | 2633 | 51.58 | 15.23 | 23.51 |
| holdout2 | 6 | large-contraction | facade | 462 | 444 | 2363 | 50.99 | 16.35 | 24.77 |
| holdout2 | 6 | large-contraction | strict + ID | 116 | 801 | 2990 | 12.65 | 3.73 | 5.77 |
| holdout2 | 6 | large-contraction | facade + ID | 116 | 790 | 2709 | 12.80 | 4.11 | 6.22 |
| holdout3 | 6 | floating-default | strict | 142 | 3 | 7646 | 97.93 | 1.82 | 3.58 |
| holdout3 | 6 | floating-default | facade | 142 | 3 | 7405 | 97.93 | 1.88 | 3.69 |
| holdout3 | 6 | floating-default | strict + ID | 44 | 101 | 7744 | 30.34 | 0.56 | 1.11 |
| holdout3 | 6 | floating-default | facade + ID | 44 | 101 | 7503 | 30.34 | 0.58 | 1.14 |
| holdout3 | 6 | baseline | strict | 1815 | 4573 | 5973 | 28.41 | 23.31 | 25.61 |
| holdout3 | 6 | baseline | facade | 1809 | 4573 | 5738 | 28.35 | 23.97 | 25.97 |
| holdout3 | 6 | baseline | strict + ID | 145 | 6243 | 7643 | 2.27 | 1.86 | 2.05 |
| holdout3 | 6 | baseline | facade + ID | 141 | 6241 | 7406 | 2.21 | 1.87 | 2.02 |
| holdout3 | 6 | source-one | strict | 1817 | 4350 | 5971 | 29.46 | 23.33 | 26.04 |
| holdout3 | 6 | source-one | facade | 1811 | 4350 | 5736 | 29.39 | 24.00 | 26.42 |
| holdout3 | 6 | source-one | strict + ID | 168 | 5999 | 7620 | 2.72 | 2.16 | 2.41 |
| holdout3 | 6 | source-one | facade + ID | 164 | 5997 | 7383 | 2.66 | 2.17 | 2.39 |
| holdout3 | 6 | large-contraction | strict | 1816 | 4350 | 5972 | 29.45 | 23.32 | 26.03 |
| holdout3 | 6 | large-contraction | facade | 1810 | 4350 | 5737 | 29.38 | 23.98 | 26.41 |
| holdout3 | 6 | large-contraction | strict + ID | 168 | 5998 | 7620 | 2.72 | 2.16 | 2.41 |
| holdout3 | 6 | large-contraction | facade + ID | 164 | 5996 | 7383 | 2.66 | 2.17 | 2.39 |
| holdout4 | 6 | floating-default | strict | 95 | 0 | 1919 | 100.00 | 4.72 | 9.01 |
| holdout4 | 6 | floating-default | facade | 95 | 0 | 1837 | 100.00 | 4.92 | 9.37 |
| holdout4 | 6 | floating-default | strict + ID | 46 | 49 | 1968 | 48.42 | 2.28 | 4.36 |
| holdout4 | 6 | floating-default | facade + ID | 46 | 49 | 1886 | 48.42 | 2.38 | 4.54 |
| holdout4 | 6 | baseline | strict | 513 | 2979 | 1501 | 14.69 | 25.47 | 18.63 |
| holdout4 | 6 | baseline | facade | 513 | 2979 | 1419 | 14.69 | 26.55 | 18.92 |
| holdout4 | 6 | baseline | strict + ID | 164 | 3328 | 1850 | 4.70 | 8.14 | 5.96 |
| holdout4 | 6 | baseline | facade + ID | 164 | 3328 | 1768 | 4.70 | 8.49 | 6.05 |
| holdout4 | 6 | source-one | strict | 445 | 290 | 1569 | 60.54 | 22.10 | 32.38 |
| holdout4 | 6 | source-one | facade | 445 | 290 | 1487 | 60.54 | 23.03 | 33.37 |
| holdout4 | 6 | source-one | strict + ID | 112 | 623 | 1902 | 15.24 | 5.56 | 8.15 |
| holdout4 | 6 | source-one | facade + ID | 112 | 623 | 1820 | 15.24 | 5.80 | 8.40 |
| holdout4 | 6 | large-contraction | strict | 458 | 309 | 1556 | 59.71 | 22.74 | 32.94 |
| holdout4 | 6 | large-contraction | facade | 458 | 309 | 1474 | 59.71 | 23.71 | 33.94 |
| holdout4 | 6 | large-contraction | strict + ID | 165 | 602 | 1849 | 21.51 | 8.19 | 11.87 |
| holdout4 | 6 | large-contraction | facade + ID | 165 | 602 | 1767 | 21.51 | 8.54 | 12.23 |
| pooled-86 | 86 | floating-default | strict | 2374 | 127 | 89393 | 94.92 | 2.59 | 5.04 |
| pooled-86 | 86 | floating-default | facade | 2366 | 127 | 86273 | 94.91 | 2.67 | 5.19 |
| pooled-86 | 86 | floating-default | strict + ID | 1011 | 1490 | 90756 | 40.42 | 1.10 | 2.14 |
| pooled-86 | 86 | floating-default | facade + ID | 1011 | 1482 | 87628 | 40.55 | 1.14 | 2.22 |
| pooled-86 | 86 | baseline | strict | 20993 | 20667 | 70774 | 50.39 | 22.88 | 31.47 |
| pooled-86 | 86 | baseline | facade | 20786 | 20667 | 67853 | 50.14 | 23.45 | 31.96 |
| pooled-86 | 86 | baseline | strict + ID | 3116 | 38544 | 88651 | 7.48 | 3.40 | 4.67 |
| pooled-86 | 86 | baseline | facade + ID | 3095 | 38358 | 85544 | 7.47 | 3.49 | 4.76 |
| pooled-86 | 86 | source-one | strict | 20507 | 15455 | 71260 | 57.02 | 22.35 | 32.11 |
| pooled-86 | 86 | source-one | facade | 20298 | 15455 | 68341 | 56.77 | 22.90 | 32.64 |
| pooled-86 | 86 | source-one | strict + ID | 3080 | 32882 | 88687 | 8.56 | 3.36 | 4.82 |
| pooled-86 | 86 | source-one | facade + ID | 3056 | 32697 | 85583 | 8.55 | 3.45 | 4.91 |
| pooled-86 | 86 | large-contraction | strict | 20905 | 15884 | 70862 | 56.82 | 22.78 | 32.52 |
| pooled-86 | 86 | large-contraction | facade | 20698 | 15884 | 67941 | 56.58 | 23.35 | 33.06 |
| pooled-86 | 86 | large-contraction | strict + ID | 3165 | 33624 | 88602 | 8.60 | 3.45 | 4.92 |
| pooled-86 | 86 | large-contraction | facade + ID | 3144 | 33438 | 85495 | 8.59 | 3.55 | 5.02 |
| holdout5 | 6 | floating-default | strict | 181 | 0 | 1648 | 100.00 | 9.90 | 18.01 |
| holdout5 | 6 | floating-default | facade | 181 | 0 | 1526 | 100.00 | 10.60 | 19.17 |
| holdout5 | 6 | floating-default | strict + ID | 179 | 2 | 1650 | 98.90 | 9.79 | 17.81 |
| holdout5 | 6 | floating-default | facade + ID | 179 | 2 | 1528 | 98.90 | 10.49 | 18.96 |
| holdout5 | 6 | baseline | strict | 335 | 1490 | 1494 | 18.36 | 18.32 | 18.34 |
| holdout5 | 6 | baseline | facade | 334 | 1490 | 1373 | 18.31 | 19.57 | 18.92 |
| holdout5 | 6 | baseline | strict + ID | 205 | 1620 | 1624 | 11.23 | 11.21 | 11.22 |
| holdout5 | 6 | baseline | facade + ID | 204 | 1620 | 1503 | 11.18 | 11.95 | 11.55 |
| holdout5 | 6 | source-one | strict | 344 | 1487 | 1485 | 18.79 | 18.81 | 18.80 |
| holdout5 | 6 | source-one | facade | 343 | 1487 | 1364 | 18.74 | 20.09 | 19.39 |
| holdout5 | 6 | source-one | strict + ID | 204 | 1627 | 1625 | 11.14 | 11.15 | 11.15 |
| holdout5 | 6 | source-one | facade + ID | 204 | 1626 | 1503 | 11.15 | 11.95 | 11.54 |
| holdout5 | 6 | large-contraction | strict | 335 | 1490 | 1494 | 18.36 | 18.32 | 18.34 |
| holdout5 | 6 | large-contraction | facade | 334 | 1490 | 1373 | 18.31 | 19.57 | 18.92 |
| holdout5 | 6 | large-contraction | strict + ID | 205 | 1620 | 1624 | 11.23 | 11.21 | 11.22 |
| holdout5 | 6 | large-contraction | facade + ID | 204 | 1620 | 1503 | 11.18 | 11.95 | 11.55 |
