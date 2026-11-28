# Source scores

The four F1 columns are percentages. Each whole structure is registered once before scoring.

| Panel | Dataset | Method | Strict placement | Facade placement | Strict position + ID | Facade position + ID | Correct / missed / extra |
|---|---|---|---:|---:|---:|---:|---|
| original | all | Current floating candidate | 35.64 | 35.82 | 4.63 | 4.65 | 14065 / 45373 / 5422 |
| original | all | Source-discovered face | 35.28 | 35.46 | 4.57 | 4.61 | 13824 / 45614 / 5101 |
| original | all | Replace a non-square seed | 35.64 | 35.82 | 4.63 | 4.65 | 14065 / 45373 / 5422 |
| original | all | Require two discovered faces | 35.83 | 36.01 | 5.04 | 5.06 | 14131 / 45307 / 5303 |
| original | all | Refine with a second face if available | 35.32 | 35.50 | 4.79 | 4.83 | 13842 / 45596 / 5102 |
| original | all | Verify existing seeds | 34.37 | 34.53 | 4.55 | 4.57 | 13424 / 46014 / 5263 |
| original | all | Learned seed verifier | 35.41 | 35.58 | 4.53 | 4.55 | 13951 / 45487 / 5419 |
| original | all | Verify before substituting | 35.61 | 35.79 | 4.59 | 4.61 | 14053 / 45385 / 5431 |
| original | all | Verify or find, fall back to ordinary seed extent | 34.38 | 34.55 | 4.52 | 4.54 | 13432 / 46006 / 5266 |
| original | all | Perfect chooser among source exports | 36.02 | 36.20 | 4.95 | 4.97 | 14194 / 45244 / 5180 |
| original | decorative-builds-v1 | Current floating candidate | 24.46 | 25.63 | 3.09 | 3.34 | 633 / 2989 / 921 |
| original | decorative-builds-v1 | Source-discovered face | 23.22 | 24.28 | 3.47 | 3.76 | 589 / 3033 / 862 |
| original | decorative-builds-v1 | Replace a non-square seed | 24.46 | 25.63 | 3.09 | 3.34 | 633 / 2989 / 921 |
| original | decorative-builds-v1 | Require two discovered faces | 25.22 | 26.42 | 4.09 | 4.42 | 654 / 2968 / 910 |
| original | decorative-builds-v1 | Refine with a second face if available | 23.22 | 24.28 | 3.43 | 3.71 | 589 / 3033 / 862 |
| original | decorative-builds-v1 | Verify existing seeds | 23.95 | 25.08 | 2.87 | 3.10 | 618 / 3004 / 921 |
| original | decorative-builds-v1 | Learned seed verifier | 23.95 | 25.08 | 2.87 | 3.10 | 618 / 3004 / 921 |
| original | decorative-builds-v1 | Verify before substituting | 24.46 | 25.63 | 3.09 | 3.34 | 633 / 2989 / 921 |
| original | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 23.95 | 25.08 | 2.87 | 3.10 | 618 / 3004 / 921 |
| original | decorative-builds-v1 | Perfect chooser among source exports | 25.23 | 26.42 | 4.09 | 4.42 | 654 / 2968 / 909 |
| original | visible-builds-v2 | Current floating candidate | 43.88 | 46.14 | 20.31 | 21.37 | 1087 / 2327 / 453 |
| original | visible-builds-v2 | Source-discovered face | 42.23 | 44.42 | 19.09 | 20.20 | 991 / 2423 / 288 |
| original | visible-builds-v2 | Replace a non-square seed | 43.88 | 46.14 | 20.31 | 21.37 | 1087 / 2327 / 453 |
| original | visible-builds-v2 | Require two discovered faces | 46.29 | 48.71 | 26.01 | 27.39 | 1132 / 2282 / 345 |
| original | visible-builds-v2 | Refine with a second face if available | 42.83 | 45.04 | 22.71 | 24.02 | 1009 / 2405 / 289 |
| original | visible-builds-v2 | Verify existing seeds | 40.52 | 42.77 | 19.78 | 20.88 | 942 / 2472 / 294 |
| original | visible-builds-v2 | Learned seed verifier | 40.73 | 42.85 | 19.21 | 20.23 | 988 / 2426 / 450 |
| original | visible-builds-v2 | Verify before substituting | 43.43 | 45.66 | 19.63 | 20.66 | 1075 / 2339 / 462 |
| original | visible-builds-v2 | Verify or find, fall back to ordinary seed extent | 40.76 | 43.02 | 19.14 | 20.20 | 950 / 2464 / 297 |
| original | visible-builds-v2 | Perfect chooser among source exports | 49.46 | 52.12 | 24.75 | 26.12 | 1195 / 2219 / 223 |
| original | visible-builds-v3 | Current floating candidate | 35.89 | 35.82 | 3.62 | 3.60 | 12345 / 40057 / 4048 |
| original | visible-builds-v3 | Source-discovered face | 35.70 | 35.65 | 3.66 | 3.66 | 12244 / 40158 / 3951 |
| original | visible-builds-v3 | Replace a non-square seed | 35.89 | 35.82 | 3.62 | 3.60 | 12345 / 40057 / 4048 |
| original | visible-builds-v3 | Require two discovered faces | 35.89 | 35.82 | 3.62 | 3.60 | 12345 / 40057 / 4048 |
| original | visible-builds-v3 | Refine with a second face if available | 35.70 | 35.65 | 3.66 | 3.66 | 12244 / 40158 / 3951 |
| original | visible-builds-v3 | Verify existing seeds | 34.73 | 34.66 | 3.64 | 3.62 | 11864 / 40538 / 4048 |
| original | visible-builds-v3 | Learned seed verifier | 35.89 | 35.82 | 3.62 | 3.60 | 12345 / 40057 / 4048 |
| original | visible-builds-v3 | Verify before substituting | 35.89 | 35.82 | 3.62 | 3.60 | 12345 / 40057 / 4048 |
| original | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 34.73 | 34.66 | 3.64 | 3.62 | 11864 / 40538 / 4048 |
| original | visible-builds-v3 | Perfect chooser among source exports | 35.89 | 35.82 | 3.62 | 3.60 | 12345 / 40057 / 4048 |
| fresh | all | Current floating candidate | 30.43 | 31.11 | 6.85 | 7.36 | 1600 / 5914 / 1401 |
| fresh | all | Source-discovered face | 30.18 | 30.77 | 6.55 | 6.86 | 1575 / 5939 / 1349 |
| fresh | all | Replace a non-square seed | 30.43 | 31.11 | 6.85 | 7.36 | 1600 / 5914 / 1401 |
| fresh | all | Require two discovered faces | 27.41 | 28.55 | 6.32 | 6.79 | 1418 / 6096 / 1416 |
| fresh | all | Refine with a second face if available | 27.19 | 28.35 | 6.43 | 6.78 | 1396 / 6118 / 1357 |
| fresh | all | Verify existing seeds | 30.84 | 31.55 | 6.90 | 7.42 | 1626 / 5888 / 1404 |
| fresh | all | Learned seed verifier | 30.43 | 31.11 | 6.85 | 7.36 | 1600 / 5914 / 1401 |
| fresh | all | Verify before substituting | 31.44 | 32.19 | 7.18 | 7.72 | 1663 / 5851 / 1402 |
| fresh | all | Verify or find, fall back to ordinary seed extent | 31.44 | 32.19 | 7.18 | 7.72 | 1663 / 5851 / 1402 |
| fresh | all | Perfect chooser among source exports | 31.82 | 32.61 | 8.15 | 8.76 | 1676 / 5838 / 1344 |
| fresh | decorative-builds-v1 | Current floating candidate | 33.10 | 34.66 | 8.19 | 9.06 | 372 / 1348 / 156 |
| fresh | decorative-builds-v1 | Source-discovered face | 31.18 | 32.64 | 9.95 | 10.42 | 348 / 1372 / 164 |
| fresh | decorative-builds-v1 | Replace a non-square seed | 33.10 | 34.66 | 8.19 | 9.06 | 372 / 1348 / 156 |
| fresh | decorative-builds-v1 | Require two discovered faces | 33.14 | 34.71 | 11.55 | 12.78 | 373 / 1347 / 158 |
| fresh | decorative-builds-v1 | Refine with a second face if available | 31.25 | 32.72 | 11.94 | 12.63 | 348 / 1372 / 159 |
| fresh | decorative-builds-v1 | Verify existing seeds | 33.10 | 34.66 | 8.19 | 9.06 | 372 / 1348 / 156 |
| fresh | decorative-builds-v1 | Learned seed verifier | 33.10 | 34.66 | 8.19 | 9.06 | 372 / 1348 / 156 |
| fresh | decorative-builds-v1 | Verify before substituting | 33.10 | 34.66 | 8.19 | 9.06 | 372 / 1348 / 156 |
| fresh | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 33.10 | 34.66 | 8.19 | 9.06 | 372 / 1348 / 156 |
| fresh | decorative-builds-v1 | Perfect chooser among source exports | 33.21 | 34.79 | 11.55 | 12.78 | 374 / 1346 / 158 |
| fresh | visible-builds-v2 | Current floating candidate | 58.27 | 57.51 | 24.64 | 26.72 | 421 / 545 / 58 |
| fresh | visible-builds-v2 | Source-discovered face | 57.88 | 56.53 | 19.57 | 20.82 | 417 / 549 / 58 |
| fresh | visible-builds-v2 | Replace a non-square seed | 58.27 | 57.51 | 24.64 | 26.72 | 421 / 545 / 58 |
| fresh | visible-builds-v2 | Require two discovered faces | 37.33 | 39.83 | 16.78 | 17.91 | 238 / 728 / 71 |
| fresh | visible-builds-v2 | Refine with a second face if available | 37.33 | 39.83 | 16.78 | 17.91 | 238 / 728 / 71 |
| fresh | visible-builds-v2 | Verify existing seeds | 60.65 | 60.12 | 24.69 | 26.74 | 447 / 519 / 61 |
| fresh | visible-builds-v2 | Learned seed verifier | 58.27 | 57.51 | 24.64 | 26.72 | 421 / 545 / 58 |
| fresh | visible-builds-v2 | Verify before substituting | 64.15 | 63.93 | 26.24 | 28.37 | 484 / 482 / 59 |
| fresh | visible-builds-v2 | Verify or find, fall back to ordinary seed extent | 64.15 | 63.93 | 26.24 | 28.37 | 484 / 482 / 59 |
| fresh | visible-builds-v2 | Perfect chooser among source exports | 64.86 | 64.71 | 27.55 | 29.78 | 492 / 474 / 59 |
| fresh | visible-builds-v3 | Current floating candidate | 23.66 | 24.46 | 2.64 | 2.77 | 807 / 4021 / 1187 |
| fresh | visible-builds-v3 | Source-discovered face | 23.95 | 24.77 | 2.66 | 2.80 | 810 / 4018 / 1127 |
| fresh | visible-builds-v3 | Replace a non-square seed | 23.66 | 24.46 | 2.64 | 2.77 | 807 / 4021 / 1187 |
| fresh | visible-builds-v3 | Require two discovered faces | 23.66 | 24.46 | 2.64 | 2.77 | 807 / 4021 / 1187 |
| fresh | visible-builds-v3 | Refine with a second face if available | 23.95 | 24.77 | 2.66 | 2.80 | 810 / 4018 / 1127 |
| fresh | visible-builds-v3 | Verify existing seeds | 23.66 | 24.46 | 2.64 | 2.77 | 807 / 4021 / 1187 |
| fresh | visible-builds-v3 | Learned seed verifier | 23.66 | 24.46 | 2.64 | 2.77 | 807 / 4021 / 1187 |
| fresh | visible-builds-v3 | Verify before substituting | 23.66 | 24.46 | 2.64 | 2.77 | 807 / 4021 / 1187 |
| fresh | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 23.66 | 24.46 | 2.64 | 2.77 | 807 / 4021 / 1187 |
| fresh | visible-builds-v3 | Perfect chooser among source exports | 23.95 | 24.77 | 2.66 | 2.80 | 810 / 4018 / 1127 |
| supplemental | all | Current floating candidate | 39.92 | 43.08 | 9.27 | 10.00 | 900 / 2353 / 356 |
| supplemental | all | Source-discovered face | 38.55 | 41.66 | 11.10 | 12.00 | 854 / 2399 / 324 |
| supplemental | all | Replace a non-square seed | 39.92 | 43.08 | 9.27 | 10.00 | 900 / 2353 / 356 |
| supplemental | all | Require two discovered faces | 39.57 | 42.72 | 13.28 | 14.34 | 888 / 2365 / 347 |
| supplemental | all | Refine with a second face if available | 38.35 | 41.45 | 13.29 | 14.37 | 848 / 2405 / 322 |
| supplemental | all | Verify existing seeds | 38.98 | 42.10 | 9.25 | 9.99 | 872 / 2381 / 349 |
| supplemental | all | Learned seed verifier | 39.92 | 43.08 | 9.27 | 10.00 | 900 / 2353 / 356 |
| supplemental | all | Verify before substituting | 38.73 | 41.83 | 9.27 | 10.01 | 865 / 2388 / 349 |
| supplemental | all | Verify or find, fall back to ordinary seed extent | 38.73 | 41.83 | 9.27 | 10.01 | 865 / 2388 / 349 |
| supplemental | all | Perfect chooser among source exports | 40.20 | 43.38 | 9.74 | 10.51 | 908 / 2345 / 356 |
| supplemental | decorative-builds-v1 | Current floating candidate | 40.83 | 41.60 | 17.99 | 18.33 | 286 / 717 / 112 |
| supplemental | decorative-builds-v1 | Source-discovered face | 40.37 | 41.14 | 14.03 | 14.30 | 282 / 721 / 112 |
| supplemental | decorative-builds-v1 | Replace a non-square seed | 40.83 | 41.60 | 17.99 | 18.33 | 286 / 717 / 112 |
| supplemental | decorative-builds-v1 | Require two discovered faces | 39.63 | 40.38 | 13.83 | 14.10 | 275 / 728 / 110 |
| supplemental | decorative-builds-v1 | Refine with a second face if available | 39.63 | 40.38 | 13.83 | 14.10 | 275 / 728 / 110 |
| supplemental | decorative-builds-v1 | Verify existing seeds | 40.83 | 41.60 | 17.99 | 18.33 | 286 / 717 / 112 |
| supplemental | decorative-builds-v1 | Learned seed verifier | 40.83 | 41.60 | 17.99 | 18.33 | 286 / 717 / 112 |
| supplemental | decorative-builds-v1 | Verify before substituting | 40.83 | 41.60 | 17.99 | 18.33 | 286 / 717 / 112 |
| supplemental | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 40.83 | 41.60 | 17.99 | 18.33 | 286 / 717 / 112 |
| supplemental | decorative-builds-v1 | Perfect chooser among source exports | 40.83 | 41.60 | 17.99 | 18.33 | 286 / 717 / 112 |
| supplemental | visible-builds-v2 | Current floating candidate | 45.63 | 62.58 | 5.05 | 6.92 | 235 / 528 / 32 |
| supplemental | visible-builds-v2 | Source-discovered face | 39.03 | 55.31 | 16.88 | 23.92 | 185 / 578 / 0 |
| supplemental | visible-builds-v2 | Replace a non-square seed | 45.63 | 62.58 | 5.05 | 6.92 | 235 / 528 / 32 |
| supplemental | visible-builds-v2 | Require two discovered faces | 44.58 | 61.50 | 18.93 | 26.12 | 226 / 537 / 25 |
| supplemental | visible-builds-v2 | Refine with a second face if available | 39.20 | 55.52 | 19.39 | 27.46 | 186 / 577 / 0 |
| supplemental | visible-builds-v2 | Verify existing seeds | 41.61 | 57.82 | 4.82 | 6.70 | 207 / 556 / 25 |
| supplemental | visible-builds-v2 | Learned seed verifier | 45.63 | 62.58 | 5.05 | 6.92 | 235 / 528 / 32 |
| supplemental | visible-builds-v2 | Verify before substituting | 40.49 | 56.42 | 4.86 | 6.77 | 200 / 563 / 25 |
| supplemental | visible-builds-v2 | Verify or find, fall back to ordinary seed extent | 40.49 | 56.42 | 4.86 | 6.77 | 200 / 563 / 25 |
| supplemental | visible-builds-v2 | Perfect chooser among source exports | 45.63 | 62.58 | 5.05 | 6.92 | 235 / 528 / 32 |
| supplemental | visible-builds-v3 | Current floating candidate | 36.48 | 36.94 | 5.49 | 5.56 | 379 / 1108 / 212 |
| supplemental | visible-builds-v3 | Source-discovered face | 37.10 | 37.57 | 6.52 | 6.60 | 387 / 1100 / 212 |
| supplemental | visible-builds-v3 | Replace a non-square seed | 36.48 | 36.94 | 5.49 | 5.56 | 379 / 1108 / 212 |
| supplemental | visible-builds-v3 | Require two discovered faces | 37.10 | 37.57 | 10.16 | 10.29 | 387 / 1100 / 212 |
| supplemental | visible-builds-v3 | Refine with a second face if available | 37.10 | 37.57 | 10.16 | 10.29 | 387 / 1100 / 212 |
| supplemental | visible-builds-v3 | Verify existing seeds | 36.48 | 36.94 | 5.49 | 5.56 | 379 / 1108 / 212 |
| supplemental | visible-builds-v3 | Learned seed verifier | 36.48 | 36.94 | 5.49 | 5.56 | 379 / 1108 / 212 |
| supplemental | visible-builds-v3 | Verify before substituting | 36.48 | 36.94 | 5.49 | 5.56 | 379 / 1108 / 212 |
| supplemental | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 36.48 | 36.94 | 5.49 | 5.56 | 379 / 1108 / 212 |
| supplemental | visible-builds-v3 | Perfect chooser among source exports | 37.10 | 37.57 | 6.52 | 6.60 | 387 / 1100 / 212 |
| sky | all | Current floating candidate | 22.93 | 23.07 | 4.40 | 4.40 | 901 / 3140 / 2915 |
| sky | all | Source-discovered face | 27.45 | 27.68 | 5.97 | 5.99 | 888 / 3153 / 1542 |
| sky | all | Replace a non-square seed | 27.50 | 27.73 | 5.96 | 5.99 | 890 / 3151 / 1542 |
| sky | all | Require two discovered faces | 22.93 | 23.07 | 4.40 | 4.40 | 901 / 3140 / 2915 |
| sky | all | Refine with a second face if available | 27.45 | 27.68 | 5.97 | 5.99 | 888 / 3153 / 1542 |
| sky | all | Verify existing seeds | 23.36 | 23.50 | 5.07 | 5.08 | 917 / 3124 / 2894 |
| sky | all | Learned seed verifier | 22.93 | 23.07 | 4.40 | 4.40 | 901 / 3140 / 2915 |
| sky | all | Verify before substituting | 27.50 | 27.73 | 5.96 | 5.99 | 890 / 3151 / 1542 |
| sky | all | Verify or find, fall back to ordinary seed extent | 28.01 | 28.26 | 6.77 | 6.81 | 906 / 3135 / 1521 |
| sky | all | Perfect chooser among source exports | 28.01 | 28.26 | 6.77 | 6.81 | 906 / 3135 / 1521 |
| sky | decorative-builds-v1 | Current floating candidate | 11.54 | 11.84 | 1.60 | 1.64 | 137 / 648 / 1452 |
| sky | decorative-builds-v1 | Source-discovered face | 25.45 | 27.07 | 7.88 | 8.38 | 126 / 659 / 79 |
| sky | decorative-builds-v1 | Replace a non-square seed | 25.45 | 27.07 | 7.88 | 8.38 | 126 / 659 / 79 |
| sky | decorative-builds-v1 | Require two discovered faces | 11.54 | 11.84 | 1.60 | 1.64 | 137 / 648 / 1452 |
| sky | decorative-builds-v1 | Refine with a second face if available | 25.45 | 27.07 | 7.88 | 8.38 | 126 / 659 / 79 |
| sky | decorative-builds-v1 | Verify existing seeds | 12.92 | 13.25 | 3.80 | 3.90 | 153 / 632 / 1431 |
| sky | decorative-builds-v1 | Learned seed verifier | 11.54 | 11.84 | 1.60 | 1.64 | 137 / 648 / 1452 |
| sky | decorative-builds-v1 | Verify before substituting | 25.45 | 27.07 | 7.88 | 8.38 | 126 / 659 / 79 |
| sky | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 28.83 | 30.67 | 13.20 | 14.04 | 142 / 643 / 58 |
| sky | decorative-builds-v1 | Perfect chooser among source exports | 28.83 | 30.67 | 13.20 | 14.04 | 142 / 643 / 58 |
| sky | visible-builds-v3 | Current floating candidate | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| sky | visible-builds-v3 | Source-discovered face | 27.81 | 27.78 | 5.62 | 5.58 | 762 / 2494 / 1463 |
| sky | visible-builds-v3 | Replace a non-square seed | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| sky | visible-builds-v3 | Require two discovered faces | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| sky | visible-builds-v3 | Refine with a second face if available | 27.81 | 27.78 | 5.62 | 5.58 | 762 / 2494 / 1463 |
| sky | visible-builds-v3 | Verify existing seeds | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| sky | visible-builds-v3 | Learned seed verifier | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| sky | visible-builds-v3 | Verify before substituting | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| sky | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| sky | visible-builds-v3 | Perfect chooser among source exports | 27.87 | 27.85 | 5.62 | 5.58 | 764 / 2492 / 1463 |
| holdout | all | Current floating candidate | 18.83 | 20.12 | 3.39 | 3.70 | 700 / 3913 / 2120 |
| holdout | all | Source-discovered face | 18.47 | 19.72 | 3.26 | 3.56 | 685 / 3928 / 2119 |
| holdout | all | Replace a non-square seed | 18.83 | 20.12 | 3.39 | 3.70 | 700 / 3913 / 2120 |
| holdout | all | Require two discovered faces | 18.83 | 20.12 | 3.39 | 3.70 | 700 / 3913 / 2120 |
| holdout | all | Refine with a second face if available | 18.47 | 19.72 | 3.26 | 3.56 | 685 / 3928 / 2119 |
| holdout | all | Verify existing seeds | 20.71 | 22.66 | 5.21 | 5.87 | 600 / 4013 / 580 |
| holdout | all | Learned seed verifier | 18.83 | 20.12 | 3.39 | 3.70 | 700 / 3913 / 2120 |
| holdout | all | Verify before substituting | 18.83 | 20.12 | 3.39 | 3.70 | 700 / 3913 / 2120 |
| holdout | all | Verify or find, fall back to ordinary seed extent | 20.71 | 22.66 | 5.21 | 5.87 | 600 / 4013 / 580 |
| holdout | all | Perfect chooser among source exports | 20.71 | 22.66 | 5.21 | 5.87 | 600 / 4013 / 580 |
| holdout | decorative-builds-v1 | Current floating candidate | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | decorative-builds-v1 | Source-discovered face | 22.92 | 23.27 | 6.77 | 7.83 | 132 / 764 / 124 |
| holdout | decorative-builds-v1 | Replace a non-square seed | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | decorative-builds-v1 | Require two discovered faces | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | decorative-builds-v1 | Refine with a second face if available | 22.92 | 23.27 | 6.77 | 7.83 | 132 / 764 / 124 |
| holdout | decorative-builds-v1 | Verify existing seeds | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | decorative-builds-v1 | Learned seed verifier | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | decorative-builds-v1 | Verify before substituting | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | decorative-builds-v1 | Perfect chooser among source exports | 25.15 | 25.91 | 7.19 | 8.30 | 147 / 749 / 126 |
| holdout | visible-builds-v2 | Current floating candidate | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v2 | Source-discovered face | 18.00 | 20.98 | 6.80 | 8.29 | 45 / 390 / 20 |
| holdout | visible-builds-v2 | Replace a non-square seed | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v2 | Require two discovered faces | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v2 | Refine with a second face if available | 18.00 | 20.98 | 6.80 | 8.29 | 45 / 390 / 20 |
| holdout | visible-builds-v2 | Verify existing seeds | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v2 | Learned seed verifier | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v2 | Verify before substituting | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v2 | Verify or find, fall back to ordinary seed extent | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v2 | Perfect chooser among source exports | 18.04 | 21.03 | 7.62 | 9.29 | 45 / 390 / 19 |
| holdout | visible-builds-v3 | Current floating candidate | 17.62 | 18.98 | 2.25 | 2.43 | 508 / 2774 / 1975 |
| holdout | visible-builds-v3 | Source-discovered face | 17.62 | 18.98 | 2.25 | 2.43 | 508 / 2774 / 1975 |
| holdout | visible-builds-v3 | Replace a non-square seed | 17.62 | 18.98 | 2.25 | 2.43 | 508 / 2774 / 1975 |
| holdout | visible-builds-v3 | Require two discovered faces | 17.62 | 18.98 | 2.25 | 2.43 | 508 / 2774 / 1975 |
| holdout | visible-builds-v3 | Refine with a second face if available | 17.62 | 18.98 | 2.25 | 2.43 | 508 / 2774 / 1975 |
| holdout | visible-builds-v3 | Verify existing seeds | 19.78 | 21.97 | 4.36 | 4.85 | 408 / 2874 / 435 |
| holdout | visible-builds-v3 | Learned seed verifier | 17.62 | 18.98 | 2.25 | 2.43 | 508 / 2774 / 1975 |
| holdout | visible-builds-v3 | Verify before substituting | 17.62 | 18.98 | 2.25 | 2.43 | 508 / 2774 / 1975 |
| holdout | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 19.78 | 21.97 | 4.36 | 4.85 | 408 / 2874 / 435 |
| holdout | visible-builds-v3 | Perfect chooser among source exports | 19.78 | 21.97 | 4.36 | 4.85 | 408 / 2874 / 435 |
| holdout2 | all | Current floating candidate | 22.15 | 23.16 | 4.93 | 5.27 | 499 / 2607 / 901 |
| holdout2 | all | Source-discovered face | 21.46 | 22.59 | 5.43 | 5.87 | 419 / 2687 / 380 |
| holdout2 | all | Replace a non-square seed | 22.15 | 23.16 | 4.93 | 5.27 | 499 / 2607 / 901 |
| holdout2 | all | Require two discovered faces | 22.15 | 23.16 | 4.93 | 5.27 | 499 / 2607 / 901 |
| holdout2 | all | Refine with a second face if available | 21.46 | 22.59 | 5.43 | 5.87 | 419 / 2687 / 380 |
| holdout2 | all | Verify existing seeds | 20.52 | 21.44 | 3.90 | 4.17 | 453 / 2653 / 856 |
| holdout2 | all | Learned seed verifier | 22.15 | 23.16 | 4.93 | 5.27 | 499 / 2607 / 901 |
| holdout2 | all | Verify before substituting | 22.15 | 23.16 | 4.93 | 5.27 | 499 / 2607 / 901 |
| holdout2 | all | Verify or find, fall back to ordinary seed extent | 20.52 | 21.44 | 3.90 | 4.17 | 453 / 2653 / 856 |
| holdout2 | all | Perfect chooser among source exports | 23.51 | 24.77 | 5.77 | 6.22 | 473 / 2633 / 444 |
| holdout2 | decorative-builds-v1 | Current floating candidate | 22.47 | 21.86 | 2.37 | 2.48 | 152 / 507 / 542 |
| holdout2 | decorative-builds-v1 | Source-discovered face | 28.97 | 28.50 | 4.83 | 5.20 | 126 / 533 / 85 |
| holdout2 | decorative-builds-v1 | Replace a non-square seed | 22.47 | 21.86 | 2.37 | 2.48 | 152 / 507 / 542 |
| holdout2 | decorative-builds-v1 | Require two discovered faces | 22.47 | 21.86 | 2.37 | 2.48 | 152 / 507 / 542 |
| holdout2 | decorative-builds-v1 | Refine with a second face if available | 28.97 | 28.50 | 4.83 | 5.20 | 126 / 533 / 85 |
| holdout2 | decorative-builds-v1 | Verify existing seeds | 22.47 | 21.86 | 2.37 | 2.48 | 152 / 507 / 542 |
| holdout2 | decorative-builds-v1 | Learned seed verifier | 22.47 | 21.86 | 2.37 | 2.48 | 152 / 507 / 542 |
| holdout2 | decorative-builds-v1 | Verify before substituting | 22.47 | 21.86 | 2.37 | 2.48 | 152 / 507 / 542 |
| holdout2 | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 22.47 | 21.86 | 2.37 | 2.48 | 152 / 507 / 542 |
| holdout2 | decorative-builds-v1 | Perfect chooser among source exports | 28.97 | 28.50 | 4.83 | 5.20 | 126 / 533 / 85 |
| holdout2 | visible-builds-v2 | Current floating candidate | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v2 | Source-discovered face | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v2 | Replace a non-square seed | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v2 | Require two discovered faces | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v2 | Refine with a second face if available | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v2 | Verify existing seeds | 46.10 | 47.28 | 10.61 | 10.88 | 139 / 319 / 6 |
| holdout2 | visible-builds-v2 | Learned seed verifier | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v2 | Verify before substituting | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v2 | Verify or find, fall back to ordinary seed extent | 46.10 | 47.28 | 10.61 | 10.88 | 139 / 319 / 6 |
| holdout2 | visible-builds-v2 | Perfect chooser among source exports | 53.31 | 54.49 | 16.43 | 16.79 | 185 / 273 / 51 |
| holdout2 | visible-builds-v3 | Current floating candidate | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout2 | visible-builds-v3 | Source-discovered face | 9.23 | 10.16 | 2.39 | 2.63 | 108 / 1881 / 244 |
| holdout2 | visible-builds-v3 | Replace a non-square seed | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout2 | visible-builds-v3 | Require two discovered faces | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout2 | visible-builds-v3 | Refine with a second face if available | 9.23 | 10.16 | 2.39 | 2.63 | 108 / 1881 / 244 |
| holdout2 | visible-builds-v3 | Verify existing seeds | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout2 | visible-builds-v3 | Learned seed verifier | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout2 | visible-builds-v3 | Verify before substituting | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout2 | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout2 | visible-builds-v3 | Perfect chooser among source exports | 13.18 | 14.43 | 3.09 | 3.39 | 162 / 1827 / 308 |
| holdout3 | all | Current floating candidate | 25.61 | 25.97 | 2.05 | 2.02 | 1815 / 5973 / 4573 |
| holdout3 | all | Source-discovered face | 26.04 | 26.42 | 2.41 | 2.39 | 1817 / 5971 / 4350 |
| holdout3 | all | Replace a non-square seed | 25.61 | 25.97 | 2.05 | 2.02 | 1815 / 5973 / 4573 |
| holdout3 | all | Require two discovered faces | 26.08 | 26.46 | 2.65 | 2.64 | 1820 / 5968 / 4350 |
| holdout3 | all | Refine with a second face if available | 26.08 | 26.46 | 2.65 | 2.64 | 1820 / 5968 / 4350 |
| holdout3 | all | Verify existing seeds | 27.37 | 27.81 | 1.99 | 1.97 | 1770 / 6018 / 3375 |
| holdout3 | all | Learned seed verifier | 25.61 | 25.97 | 2.05 | 2.02 | 1815 / 5973 / 4573 |
| holdout3 | all | Verify before substituting | 25.61 | 25.97 | 2.05 | 2.02 | 1815 / 5973 / 4573 |
| holdout3 | all | Verify or find, fall back to ordinary seed extent | 27.37 | 27.81 | 1.99 | 1.97 | 1770 / 6018 / 3375 |
| holdout3 | all | Perfect chooser among source exports | 27.92 | 28.38 | 2.66 | 2.65 | 1775 / 6013 / 3152 |
| holdout3 | decorative-builds-v1 | Current floating candidate | 8.20 | 8.02 | 2.44 | 2.26 | 141 / 585 / 2572 |
| holdout3 | decorative-builds-v1 | Source-discovered face | 8.20 | 8.02 | 2.44 | 2.26 | 141 / 585 / 2572 |
| holdout3 | decorative-builds-v1 | Replace a non-square seed | 8.20 | 8.02 | 2.44 | 2.26 | 141 / 585 / 2572 |
| holdout3 | decorative-builds-v1 | Require two discovered faces | 8.20 | 8.02 | 2.44 | 2.26 | 141 / 585 / 2572 |
| holdout3 | decorative-builds-v1 | Refine with a second face if available | 8.20 | 8.02 | 2.44 | 2.26 | 141 / 585 / 2572 |
| holdout3 | decorative-builds-v1 | Verify existing seeds | 8.74 | 8.47 | 2.37 | 2.07 | 96 / 630 / 1374 |
| holdout3 | decorative-builds-v1 | Learned seed verifier | 8.20 | 8.02 | 2.44 | 2.26 | 141 / 585 / 2572 |
| holdout3 | decorative-builds-v1 | Verify before substituting | 8.20 | 8.02 | 2.44 | 2.26 | 141 / 585 / 2572 |
| holdout3 | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 8.74 | 8.47 | 2.37 | 2.07 | 96 / 630 / 1374 |
| holdout3 | decorative-builds-v1 | Perfect chooser among source exports | 8.74 | 8.47 | 2.37 | 2.07 | 96 / 630 / 1374 |
| holdout3 | visible-builds-v2 | Current floating candidate | 30.99 | 31.11 | 4.12 | 4.13 | 128 / 335 / 235 |
| holdout3 | visible-builds-v2 | Source-discovered face | 42.98 | 43.19 | 13.22 | 13.29 | 130 / 333 / 12 |
| holdout3 | visible-builds-v2 | Replace a non-square seed | 30.99 | 31.11 | 4.12 | 4.13 | 128 / 335 / 235 |
| holdout3 | visible-builds-v2 | Require two discovered faces | 43.75 | 43.97 | 18.75 | 18.84 | 133 / 330 / 12 |
| holdout3 | visible-builds-v2 | Refine with a second face if available | 43.75 | 43.97 | 18.75 | 18.84 | 133 / 330 / 12 |
| holdout3 | visible-builds-v2 | Verify existing seeds | 30.99 | 31.11 | 4.12 | 4.13 | 128 / 335 / 235 |
| holdout3 | visible-builds-v2 | Learned seed verifier | 30.99 | 31.11 | 4.12 | 4.13 | 128 / 335 / 235 |
| holdout3 | visible-builds-v2 | Verify before substituting | 30.99 | 31.11 | 4.12 | 4.13 | 128 / 335 / 235 |
| holdout3 | visible-builds-v2 | Verify or find, fall back to ordinary seed extent | 30.99 | 31.11 | 4.12 | 4.13 | 128 / 335 / 235 |
| holdout3 | visible-builds-v2 | Perfect chooser among source exports | 43.75 | 43.97 | 18.75 | 18.84 | 133 / 330 / 12 |
| holdout3 | visible-builds-v3 | Current floating candidate | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Source-discovered face | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Replace a non-square seed | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Require two discovered faces | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Refine with a second face if available | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Verify existing seeds | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Learned seed verifier | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Verify before substituting | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout3 | visible-builds-v3 | Perfect chooser among source exports | 31.20 | 31.75 | 1.74 | 1.77 | 1546 / 5053 / 1766 |
| holdout4 | all | Current floating candidate | 18.63 | 18.92 | 5.96 | 6.05 | 513 / 1501 / 2979 |
| holdout4 | all | Source-discovered face | 32.38 | 33.37 | 8.15 | 8.40 | 445 / 1569 / 290 |
| holdout4 | all | Replace a non-square seed | 18.63 | 18.92 | 5.96 | 6.05 | 513 / 1501 / 2979 |
| holdout4 | all | Require two discovered faces | 18.25 | 18.53 | 4.61 | 4.68 | 499 / 1515 / 2956 |
| holdout4 | all | Refine with a second face if available | 32.48 | 33.48 | 8.23 | 8.48 | 446 / 1568 / 286 |
| holdout4 | all | Verify existing seeds | 18.63 | 18.92 | 5.96 | 6.05 | 513 / 1501 / 2979 |
| holdout4 | all | Learned seed verifier | 18.63 | 18.92 | 5.96 | 6.05 | 513 / 1501 / 2979 |
| holdout4 | all | Verify before substituting | 18.63 | 18.92 | 5.96 | 6.05 | 513 / 1501 / 2979 |
| holdout4 | all | Verify or find, fall back to ordinary seed extent | 18.63 | 18.92 | 5.96 | 6.05 | 513 / 1501 / 2979 |
| holdout4 | all | Perfect chooser among source exports | 33.08 | 34.09 | 11.51 | 11.86 | 460 / 1554 / 307 |
| holdout4 | decorative-builds-v1 | Current floating candidate | 13.69 | 13.74 | 2.86 | 2.87 | 273 / 636 / 2805 |
| holdout4 | decorative-builds-v1 | Source-discovered face | 33.31 | 33.66 | 2.25 | 2.28 | 207 / 702 / 127 |
| holdout4 | decorative-builds-v1 | Replace a non-square seed | 13.69 | 13.74 | 2.86 | 2.87 | 273 / 636 / 2805 |
| holdout4 | decorative-builds-v1 | Require two discovered faces | 13.15 | 13.20 | 0.61 | 0.61 | 260 / 649 / 2784 |
| holdout4 | decorative-builds-v1 | Refine with a second face if available | 33.39 | 33.74 | 2.12 | 2.14 | 205 / 704 / 114 |
| holdout4 | decorative-builds-v1 | Verify existing seeds | 13.69 | 13.74 | 2.86 | 2.87 | 273 / 636 / 2805 |
| holdout4 | decorative-builds-v1 | Learned seed verifier | 13.69 | 13.74 | 2.86 | 2.87 | 273 / 636 / 2805 |
| holdout4 | decorative-builds-v1 | Verify before substituting | 13.69 | 13.74 | 2.86 | 2.87 | 273 / 636 / 2805 |
| holdout4 | decorative-builds-v1 | Verify or find, fall back to ordinary seed extent | 13.69 | 13.74 | 2.86 | 2.87 | 273 / 636 / 2805 |
| holdout4 | decorative-builds-v1 | Perfect chooser among source exports | 34.55 | 34.91 | 9.19 | 9.29 | 218 / 691 / 135 |
| holdout4 | visible-builds-v2 | Current floating candidate | 26.28 | 28.95 | 8.16 | 8.99 | 87 / 425 / 63 |
| holdout4 | visible-builds-v2 | Source-discovered face | 26.89 | 29.62 | 6.65 | 7.32 | 89 / 423 / 61 |
| holdout4 | visible-builds-v2 | Replace a non-square seed | 26.28 | 28.95 | 8.16 | 8.99 | 87 / 425 / 63 |
| holdout4 | visible-builds-v2 | Require two discovered faces | 26.36 | 29.05 | 9.39 | 10.35 | 87 / 425 / 61 |
| holdout4 | visible-builds-v2 | Refine with a second face if available | 26.89 | 29.62 | 5.14 | 5.66 | 89 / 423 / 61 |
| holdout4 | visible-builds-v2 | Verify existing seeds | 26.28 | 28.95 | 8.16 | 8.99 | 87 / 425 / 63 |
| holdout4 | visible-builds-v2 | Learned seed verifier | 26.28 | 28.95 | 8.16 | 8.99 | 87 / 425 / 63 |
| holdout4 | visible-builds-v2 | Verify before substituting | 26.28 | 28.95 | 8.16 | 8.99 | 87 / 425 / 63 |
| holdout4 | visible-builds-v2 | Verify or find, fall back to ordinary seed extent | 26.28 | 28.95 | 8.16 | 8.99 | 87 / 425 / 63 |
| holdout4 | visible-builds-v2 | Perfect chooser among source exports | 26.89 | 29.62 | 6.65 | 7.32 | 89 / 423 / 61 |
| holdout4 | visible-builds-v3 | Current floating candidate | 35.71 | 36.04 | 18.67 | 18.85 | 153 / 440 / 111 |
| holdout4 | visible-builds-v3 | Source-discovered face | 35.31 | 35.65 | 18.01 | 18.18 | 149 / 444 / 102 |
| holdout4 | visible-builds-v3 | Replace a non-square seed | 35.71 | 36.04 | 18.67 | 18.85 | 153 / 440 / 111 |
| holdout4 | visible-builds-v3 | Require two discovered faces | 35.51 | 35.85 | 19.39 | 19.58 | 152 / 441 / 111 |
| holdout4 | visible-builds-v3 | Refine with a second face if available | 35.51 | 35.85 | 19.39 | 19.58 | 152 / 441 / 111 |
| holdout4 | visible-builds-v3 | Verify existing seeds | 35.71 | 36.04 | 18.67 | 18.85 | 153 / 440 / 111 |
| holdout4 | visible-builds-v3 | Learned seed verifier | 35.71 | 36.04 | 18.67 | 18.85 | 153 / 440 / 111 |
| holdout4 | visible-builds-v3 | Verify before substituting | 35.71 | 36.04 | 18.67 | 18.85 | 153 / 440 / 111 |
| holdout4 | visible-builds-v3 | Verify or find, fall back to ordinary seed extent | 35.71 | 36.04 | 18.67 | 18.85 | 153 / 440 / 111 |
| holdout4 | visible-builds-v3 | Perfect chooser among source exports | 35.71 | 36.04 | 18.67 | 18.85 | 153 / 440 / 111 |
