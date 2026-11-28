# Source-only comparisons

Every row uses the floating assembler with horizontal faces enabled. Scores pool correct, missed and extra blocks across each subset before calculating F1. A single global translation is used for each export.

| Panel | Cohort | Method | Placement | Facade placement | Placement + ID | Facade placement + ID |
|---|---|---|---:|---:|---:|---:|
| original | all | Current floating candidate | 35.64 | 35.82 | 4.63 | 4.65 |
| original | all | Local lattice veto | 22.13 | 22.32 | 3.47 | 3.50 |
| original | all | Local lattice refit | 22.22 | 22.41 | 3.53 | 3.57 |
| original | all | Local plane veto | 30.76 | 30.96 | 4.55 | 4.57 |
| original | all | Both vetoes required | 32.52 | 32.72 | 4.44 | 4.45 |
| original | all | Contradiction required | 35.64 | 35.82 | 4.63 | 4.65 |
| original | visible-builds-v2 | Current floating candidate | 43.88 | 46.14 | 20.31 | 21.37 |
| original | visible-builds-v2 | Local lattice veto | 41.69 | 43.91 | 19.28 | 20.32 |
| original | visible-builds-v2 | Local lattice refit | 41.79 | 44.02 | 19.51 | 20.57 |
| original | visible-builds-v2 | Local plane veto | 43.94 | 46.20 | 20.33 | 21.39 |
| original | visible-builds-v2 | Both vetoes required | 43.91 | 46.17 | 20.32 | 21.38 |
| original | visible-builds-v2 | Contradiction required | 43.88 | 46.14 | 20.31 | 21.37 |
| original | visible-builds-v3 | Current floating candidate | 35.89 | 35.82 | 3.62 | 3.60 |
| original | visible-builds-v3 | Local lattice veto | 20.93 | 20.92 | 2.26 | 2.27 |
| original | visible-builds-v3 | Local lattice refit | 21.02 | 21.02 | 2.32 | 2.32 |
| original | visible-builds-v3 | Local plane veto | 30.67 | 30.61 | 3.44 | 3.41 |
| original | visible-builds-v3 | Both vetoes required | 32.47 | 32.41 | 3.36 | 3.34 |
| original | visible-builds-v3 | Contradiction required | 35.89 | 35.82 | 3.62 | 3.60 |
| original | decorative-builds-v1 | Current floating candidate | 24.46 | 25.63 | 3.09 | 3.34 |
| original | decorative-builds-v1 | Local lattice veto | 16.90 | 18.53 | 2.55 | 2.80 |
| original | decorative-builds-v1 | Local lattice refit | 16.90 | 18.53 | 2.55 | 2.80 |
| original | decorative-builds-v1 | Local plane veto | 16.85 | 18.26 | 3.20 | 3.51 |
| original | decorative-builds-v1 | Both vetoes required | 20.48 | 22.19 | 2.80 | 3.06 |
| original | decorative-builds-v1 | Contradiction required | 24.46 | 25.63 | 3.09 | 3.34 |
| fresh | all | Current floating candidate | 30.43 | 31.11 | 6.85 | 7.36 |
| fresh | all | Local lattice veto | 23.90 | 24.97 | 6.16 | 6.70 |
| fresh | all | Local lattice refit | 23.90 | 24.97 | 6.16 | 6.70 |
| fresh | all | Local plane veto | 28.51 | 29.33 | 7.67 | 8.27 |
| fresh | all | Both vetoes required | 29.37 | 30.19 | 6.90 | 7.42 |
| fresh | all | Contradiction required | 30.43 | 31.11 | 6.85 | 7.36 |
| fresh | visible-builds-v2 | Current floating candidate | 58.27 | 57.51 | 24.64 | 26.72 |
| fresh | visible-builds-v2 | Local lattice veto | 58.51 | 57.77 | 24.74 | 26.84 |
| fresh | visible-builds-v2 | Local lattice refit | 58.51 | 57.77 | 24.74 | 26.84 |
| fresh | visible-builds-v2 | Local plane veto | 58.27 | 57.51 | 24.64 | 26.72 |
| fresh | visible-builds-v2 | Both vetoes required | 58.27 | 57.51 | 24.64 | 26.72 |
| fresh | visible-builds-v2 | Contradiction required | 58.27 | 57.51 | 24.64 | 26.72 |
| fresh | visible-builds-v3 | Current floating candidate | 23.66 | 24.46 | 2.64 | 2.77 |
| fresh | visible-builds-v3 | Local lattice veto | 16.96 | 18.34 | 2.50 | 2.71 |
| fresh | visible-builds-v3 | Local lattice refit | 16.96 | 18.34 | 2.50 | 2.71 |
| fresh | visible-builds-v3 | Local plane veto | 21.55 | 22.23 | 3.68 | 3.91 |
| fresh | visible-builds-v3 | Both vetoes required | 22.54 | 23.28 | 2.58 | 2.71 |
| fresh | visible-builds-v3 | Contradiction required | 23.66 | 24.46 | 2.64 | 2.77 |
| fresh | decorative-builds-v1 | Current floating candidate | 33.10 | 34.66 | 8.19 | 9.06 |
| fresh | decorative-builds-v1 | Local lattice veto | 18.92 | 20.42 | 3.27 | 3.65 |
| fresh | decorative-builds-v1 | Local lattice refit | 18.92 | 20.42 | 3.27 | 3.65 |
| fresh | decorative-builds-v1 | Local plane veto | 29.46 | 32.02 | 8.26 | 9.12 |
| fresh | decorative-builds-v1 | Both vetoes required | 30.88 | 33.30 | 8.23 | 9.09 |
| fresh | decorative-builds-v1 | Contradiction required | 33.10 | 34.66 | 8.19 | 9.06 |
| supplemental | all | Current floating candidate | 39.92 | 43.08 | 9.27 | 10.00 |
| supplemental | all | Local lattice veto | 32.93 | 35.90 | 7.65 | 8.34 |
| supplemental | all | Local lattice refit | 32.93 | 35.90 | 7.65 | 8.34 |
| supplemental | all | Local plane veto | 40.55 | 43.84 | 9.47 | 10.24 |
| supplemental | all | Both vetoes required | 40.54 | 43.81 | 9.41 | 10.17 |
| supplemental | all | Contradiction required | 39.92 | 43.08 | 9.27 | 10.00 |
| supplemental | visible-builds-v2 | Current floating candidate | 45.63 | 62.58 | 5.05 | 6.92 |
| supplemental | visible-builds-v2 | Local lattice veto | 38.31 | 54.35 | 4.23 | 6.01 |
| supplemental | visible-builds-v2 | Local lattice refit | 38.31 | 54.35 | 4.23 | 6.01 |
| supplemental | visible-builds-v2 | Local plane veto | 45.63 | 62.58 | 5.05 | 6.92 |
| supplemental | visible-builds-v2 | Both vetoes required | 45.63 | 62.58 | 5.05 | 6.92 |
| supplemental | visible-builds-v2 | Contradiction required | 45.63 | 62.58 | 5.05 | 6.92 |
| supplemental | visible-builds-v3 | Current floating candidate | 36.48 | 36.94 | 5.49 | 5.56 |
| supplemental | visible-builds-v3 | Local lattice veto | 26.89 | 27.29 | 4.23 | 4.30 |
| supplemental | visible-builds-v3 | Local lattice refit | 26.89 | 27.29 | 4.23 | 4.30 |
| supplemental | visible-builds-v3 | Local plane veto | 37.64 | 38.13 | 5.66 | 5.73 |
| supplemental | visible-builds-v3 | Both vetoes required | 37.32 | 37.81 | 5.61 | 5.69 |
| supplemental | visible-builds-v3 | Contradiction required | 36.48 | 36.94 | 5.49 | 5.56 |
| supplemental | decorative-builds-v1 | Current floating candidate | 40.83 | 41.60 | 17.99 | 18.33 |
| supplemental | decorative-builds-v1 | Local lattice veto | 37.13 | 37.88 | 14.67 | 14.96 |
| supplemental | decorative-builds-v1 | Local lattice refit | 37.13 | 37.88 | 14.67 | 14.96 |
| supplemental | decorative-builds-v1 | Local plane veto | 41.02 | 41.82 | 18.39 | 18.75 |
| supplemental | decorative-builds-v1 | Both vetoes required | 41.48 | 42.28 | 18.27 | 18.63 |
| supplemental | decorative-builds-v1 | Contradiction required | 40.83 | 41.60 | 17.99 | 18.33 |
| sky | all | Current floating candidate | 22.93 | 23.07 | 4.40 | 4.40 |
| sky | all | Local lattice veto | 22.24 | 22.58 | 4.11 | 4.17 |
| sky | all | Local lattice refit | 22.24 | 22.58 | 4.11 | 4.17 |
| sky | all | Local plane veto | 24.13 | 24.35 | 4.58 | 4.65 |
| sky | all | Both vetoes required | 24.79 | 24.99 | 4.65 | 4.71 |
| sky | all | Contradiction required | 25.62 | 25.82 | 4.81 | 4.81 |
| sky | visible-builds-v3 | Current floating candidate | 27.87 | 27.85 | 5.62 | 5.58 |
| sky | visible-builds-v3 | Local lattice veto | 24.19 | 24.35 | 3.99 | 4.02 |
| sky | visible-builds-v3 | Local lattice refit | 24.19 | 24.35 | 3.99 | 4.02 |
| sky | visible-builds-v3 | Local plane veto | 26.24 | 26.25 | 5.41 | 5.44 |
| sky | visible-builds-v3 | Both vetoes required | 28.73 | 28.75 | 5.91 | 5.94 |
| sky | visible-builds-v3 | Contradiction required | 27.87 | 27.85 | 5.62 | 5.58 |
| sky | decorative-builds-v1 | Current floating candidate | 11.54 | 11.84 | 1.60 | 1.64 |
| sky | decorative-builds-v1 | Local lattice veto | 17.48 | 18.11 | 4.40 | 4.56 |
| sky | decorative-builds-v1 | Local lattice refit | 17.48 | 18.11 | 4.40 | 4.56 |
| sky | decorative-builds-v1 | Local plane veto | 14.30 | 15.12 | 0.73 | 0.78 |
| sky | decorative-builds-v1 | Both vetoes required | 13.81 | 14.25 | 1.16 | 1.20 |
| sky | decorative-builds-v1 | Contradiction required | 14.30 | 15.12 | 0.73 | 0.78 |
| holdout | all | Current floating candidate | 18.83 | 20.12 | 3.39 | 3.70 |
| holdout | all | Local lattice veto | 9.11 | 9.56 | 2.55 | 2.90 |
| holdout | all | Local lattice refit | 9.11 | 9.56 | 2.55 | 2.90 |
| holdout | all | Local plane veto | 18.90 | 20.58 | 3.38 | 3.79 |
| holdout | all | Both vetoes required | 19.16 | 20.54 | 3.43 | 3.80 |
| holdout | all | Contradiction required | 18.83 | 20.12 | 3.39 | 3.70 |
| holdout | visible-builds-v2 | Current floating candidate | 18.04 | 21.03 | 7.62 | 9.29 |
| holdout | visible-builds-v2 | Local lattice veto | 18.04 | 21.03 | 7.62 | 9.29 |
| holdout | visible-builds-v2 | Local lattice refit | 18.04 | 21.03 | 7.62 | 9.29 |
| holdout | visible-builds-v2 | Local plane veto | 17.41 | 20.30 | 7.29 | 8.91 |
| holdout | visible-builds-v2 | Both vetoes required | 18.04 | 21.03 | 7.62 | 9.29 |
| holdout | visible-builds-v2 | Contradiction required | 18.04 | 21.03 | 7.62 | 9.29 |
| holdout | visible-builds-v3 | Current floating candidate | 17.62 | 18.98 | 2.25 | 2.43 |
| holdout | visible-builds-v3 | Local lattice veto | 2.33 | 2.65 | 0.23 | 0.26 |
| holdout | visible-builds-v3 | Local lattice refit | 2.33 | 2.65 | 0.23 | 0.26 |
| holdout | visible-builds-v3 | Local plane veto | 17.89 | 19.82 | 1.85 | 2.05 |
| holdout | visible-builds-v3 | Both vetoes required | 17.66 | 19.11 | 1.91 | 2.07 |
| holdout | visible-builds-v3 | Contradiction required | 17.62 | 18.98 | 2.25 | 2.43 |
| holdout | decorative-builds-v1 | Current floating candidate | 25.15 | 25.91 | 7.19 | 8.30 |
| holdout | decorative-builds-v1 | Local lattice veto | 25.24 | 26.02 | 7.21 | 8.33 |
| holdout | decorative-builds-v1 | Local lattice refit | 25.24 | 26.02 | 7.21 | 8.33 |
| holdout | decorative-builds-v1 | Local plane veto | 23.27 | 23.70 | 7.35 | 8.52 |
| holdout | decorative-builds-v1 | Both vetoes required | 25.15 | 25.91 | 7.19 | 8.30 |
| holdout | decorative-builds-v1 | Contradiction required | 25.15 | 25.91 | 7.19 | 8.30 |
| holdout2 | all | Current floating candidate | 22.15 | 23.16 | 4.93 | 5.27 |
| holdout2 | all | Local lattice veto | 18.11 | 19.01 | 4.59 | 4.99 |
| holdout2 | all | Local lattice refit | 18.11 | 19.01 | 4.59 | 4.99 |
| holdout2 | all | Local plane veto | 21.48 | 22.52 | 5.54 | 5.95 |
| holdout2 | all | Both vetoes required | 21.86 | 22.90 | 6.21 | 6.66 |
| holdout2 | all | Contradiction required | 22.15 | 23.16 | 4.93 | 5.27 |
| holdout2 | visible-builds-v2 | Current floating candidate | 53.31 | 54.49 | 16.43 | 16.79 |
| holdout2 | visible-builds-v2 | Local lattice veto | 46.69 | 47.88 | 19.87 | 20.37 |
| holdout2 | visible-builds-v2 | Local lattice refit | 46.69 | 47.88 | 19.87 | 20.37 |
| holdout2 | visible-builds-v2 | Local plane veto | 54.20 | 55.41 | 24.64 | 25.19 |
| holdout2 | visible-builds-v2 | Both vetoes required | 54.20 | 55.41 | 24.64 | 25.19 |
| holdout2 | visible-builds-v2 | Contradiction required | 53.31 | 54.49 | 16.43 | 16.79 |
| holdout2 | visible-builds-v3 | Current floating candidate | 13.18 | 14.43 | 3.09 | 3.39 |
| holdout2 | visible-builds-v3 | Local lattice veto | 5.96 | 6.64 | 1.63 | 1.82 |
| holdout2 | visible-builds-v3 | Local lattice refit | 5.96 | 6.64 | 1.63 | 1.82 |
| holdout2 | visible-builds-v3 | Local plane veto | 10.41 | 11.51 | 1.44 | 1.59 |
| holdout2 | visible-builds-v3 | Both vetoes required | 11.61 | 12.80 | 2.88 | 3.18 |
| holdout2 | visible-builds-v3 | Contradiction required | 13.18 | 14.43 | 3.09 | 3.39 |
| holdout2 | decorative-builds-v1 | Current floating candidate | 22.47 | 21.86 | 2.37 | 2.48 |
| holdout2 | decorative-builds-v1 | Local lattice veto | 26.04 | 25.45 | 1.58 | 1.68 |
| holdout2 | decorative-builds-v1 | Local lattice refit | 26.04 | 25.45 | 1.58 | 1.68 |
| holdout2 | decorative-builds-v1 | Local plane veto | 23.09 | 22.49 | 2.45 | 2.57 |
| holdout2 | decorative-builds-v1 | Both vetoes required | 22.74 | 22.14 | 2.39 | 2.51 |
| holdout2 | decorative-builds-v1 | Contradiction required | 22.47 | 21.86 | 2.37 | 2.48 |
