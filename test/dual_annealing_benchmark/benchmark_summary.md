# YouZhiSan Dual Annealing Optimization Benchmark (Phase 2 Matrix)

- **Total Problem Instances**: 76 (19 stochastic + 57 Basin-Hopping starting points)
- **Configurations Tested**: 15 Synergistic Parameter Sets
- **Total Optimization Runs**: 1140
- **Parallel Workers**: 16 CPU cores
- **Total Benchmark Wall Time**: 5650.65 s (94.18 min)

## Overall Results Table (All 76 Instances)

| Configuration | Mean SSD | Median SSD | Paired Diff vs Base | Win / Loss / Tie | p-value (t-test / Wilcoxon) | Avg Time (s) | Avg nfev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `C01_Baseline_Default` | 68.380 | 1.633 | +0.000 ± 0.000 | 0 / 0 / 76 | 1.0000 / 1.0000 | 67.40 | 11167 |
| `C02_Restart_1e-4` | 50.400 | 1.452 | -17.981 ± 19.037 | 8 / 3 / 65 | 0.3479 / 0.0044 | 67.67 | 11319 |
| `C03_Restart_5e-4` | 37.475 | 1.204 | -30.905 ± 19.697 | 17 / 6 / 53 | 0.1208 / 0.0014 | 67.06 | 11591 |
| `C04_Restart_1e-3_Standard` | 43.293 | 1.058 | -25.088 ± 18.025 | 11 / 7 / 58 | 0.1681 / 0.0484 | 69.26 | 11845 |
| `C05_Restart_2e-3_Frequent` | 38.336 | 1.218 | -30.045 ± 19.768 | 9 / 8 / 59 | 0.1328 / 0.2449 | 71.09 | 12093 |
| `C06_Gentle_Visit_2.50_Restart_1e-3` | 40.758 | 1.452 | -27.622 ± 19.793 | 13 / 10 / 53 | 0.1670 / 0.1883 | 68.62 | 11712 |
| `C07_Moderate_Visit_2.70_Restart_1e-3` | 39.623 | 1.237 | -28.757 ± 19.801 | 10 / 7 / 59 | 0.1506 / 0.0676 | 69.44 | 11956 |
| `C08_Extended_Visit_2.75_Restart_1e-3` | 42.027 | 1.223 | -26.353 ± 20.058 | 11 / 9 / 56 | 0.1929 / 0.1736 | 68.87 | 12051 |
| `C09_Levy_Visit_2.90_Restart_1e-3` | 40.367 | 1.500 | -28.013 ± 19.759 | 10 / 8 / 58 | 0.1604 / 0.1012 | 70.32 | 12264 |
| `C10_Warm_Temp_10000_Restart_1e-3` | 37.603 | 1.251 | -30.778 ± 19.705 | 14 / 4 / 58 | 0.1225 / 0.0004 | 67.83 | 11897 |
| `C11_Hot_Temp_20000_Restart_1e-3` | 38.152 | 1.008 | -30.229 ± 18.578 | 15 / 5 / 56 | 0.1079 / 0.0010 | 67.83 | 11941 |
| `C12_Selective_Accept_m15_Restart_1e-3` | 39.760 | 1.204 | -28.620 ± 17.694 | 15 / 3 / 58 | 0.1100 / 0.0001 | 66.48 | 11793 |
| `C13_Strict_Accept_m30_Restart_1e-3` | 45.906 | 1.237 | -22.475 ± 18.694 | 13 / 5 / 58 | 0.2330 / 0.0082 | 66.60 | 11837 |
| `C14_Fast_Budget_500_Restart_1e-3` | 45.304 | 1.319 | -23.077 ± 18.063 | 7 / 8 / 61 | 0.2053 / 0.3387 | 33.92 | 5984 |
| `C15_Deep_Budget_1500_Restart_1e-3` | 41.838 | 0.637 | -26.543 ± 18.430 | 25 / 3 / 48 | 0.1540 / 0.0000 | 248.07 | 43403 |

## Subgroup: 19 Stochastic Initial Guesses (seed=42)

| Configuration | Mean SSD | Median SSD | Paired Diff vs Base | Win / Loss / Tie | Avg Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `C01_Baseline_Default` | 115.695 | 1.368 | +0.000 ± 0.000 | 0 / 0 / 19 | 74.53 |
| `C02_Restart_1e-4` | 57.334 | 0.617 | -58.361 ± 76.017 | 3 / 3 / 13 | 68.55 |
| `C03_Restart_5e-4` | 39.174 | 1.633 | -76.521 ± 73.563 | 7 / 4 / 8 | 69.45 |
| `C04_Restart_1e-3_Standard` | 48.731 | 0.912 | -66.964 ± 64.901 | 4 / 4 / 11 | 71.74 |
| `C05_Restart_2e-3_Frequent` | 41.685 | 1.368 | -74.010 ± 73.953 | 2 / 6 / 11 | 73.63 |
| `C06_Gentle_Visit_2.50_Restart_1e-3` | 49.570 | 3.290 | -66.125 ± 74.328 | 6 / 8 / 5 | 71.88 |
| `C07_Moderate_Visit_2.70_Restart_1e-3` | 47.956 | 1.368 | -67.739 ± 74.313 | 2 / 5 / 12 | 72.72 |
| `C08_Extended_Visit_2.75_Restart_1e-3` | 40.492 | 1.368 | -75.202 ± 73.894 | 4 / 5 / 10 | 71.99 |
| `C09_Levy_Visit_2.90_Restart_1e-3` | 47.775 | 1.633 | -67.920 ± 74.128 | 2 / 6 / 11 | 72.33 |
| `C10_Warm_Temp_10000_Restart_1e-3` | 38.225 | 1.368 | -77.469 ± 73.563 | 6 / 3 / 10 | 69.17 |
| `C11_Hot_Temp_20000_Restart_1e-3` | 43.352 | 1.633 | -72.343 ± 68.683 | 6 / 3 / 10 | 71.32 |
| `C12_Selective_Accept_m15_Restart_1e-3` | 47.714 | 1.368 | -67.981 ± 64.843 | 6 / 1 / 12 | 67.74 |
| `C13_Strict_Accept_m30_Restart_1e-3` | 48.634 | 1.633 | -67.061 ± 64.897 | 5 / 2 / 12 | 68.34 |
| `C14_Fast_Budget_500_Restart_1e-3` | 54.142 | 1.633 | -61.553 ± 65.263 | 2 / 5 / 12 | 36.16 |
| `C15_Deep_Budget_1500_Restart_1e-3` | 44.346 | 0.912 | -71.349 ± 66.640 | 8 / 2 / 9 | 215.90 |

## Subgroup: 57 Basin-Hopping Refined Initial Guesses

| Configuration | Mean SSD | Median SSD | Paired Diff vs Base | Win / Loss / Tie | Avg Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `C01_Baseline_Default` | 52.609 | 1.633 | +0.000 ± 0.000 | 0 / 0 / 57 | 65.03 |
| `C02_Restart_1e-4` | 48.088 | 1.633 | -4.520 ± 3.882 | 5 / 0 / 52 | 67.37 |
| `C03_Restart_5e-4` | 36.909 | 1.204 | -15.700 ± 9.822 | 10 / 2 / 45 | 66.27 |
| `C04_Restart_1e-3_Standard` | 41.480 | 1.204 | -11.129 ± 10.721 | 7 / 3 / 47 | 68.44 |
| `C05_Restart_2e-3_Frequent` | 37.220 | 1.204 | -15.389 ± 9.823 | 7 / 2 / 48 | 70.25 |
| `C06_Gentle_Visit_2.50_Restart_1e-3` | 37.821 | 1.204 | -14.788 ± 9.791 | 7 / 2 / 48 | 67.53 |
| `C07_Moderate_Visit_2.70_Restart_1e-3` | 36.846 | 1.204 | -15.763 ± 9.817 | 8 / 2 / 47 | 68.35 |
| `C08_Extended_Visit_2.75_Restart_1e-3` | 42.539 | 1.204 | -10.070 ± 10.698 | 7 / 4 / 46 | 67.83 |
| `C09_Levy_Visit_2.90_Restart_1e-3` | 37.898 | 1.270 | -14.711 ± 9.786 | 8 / 2 / 47 | 69.65 |
| `C10_Warm_Temp_10000_Restart_1e-3` | 37.395 | 1.232 | -15.214 ± 9.811 | 8 / 1 / 48 | 67.38 |
| `C11_Hot_Temp_20000_Restart_1e-3` | 36.418 | 0.912 | -16.191 ± 9.840 | 9 / 2 / 46 | 66.67 |
| `C12_Selective_Accept_m15_Restart_1e-3` | 37.109 | 1.204 | -15.500 ± 9.811 | 9 / 2 / 46 | 66.06 |
| `C13_Strict_Accept_m30_Restart_1e-3` | 44.996 | 1.204 | -7.613 ± 12.530 | 8 / 3 / 46 | 66.03 |
| `C14_Fast_Budget_500_Restart_1e-3` | 42.358 | 1.204 | -10.251 ± 10.701 | 5 / 3 / 49 | 33.18 |
| `C15_Deep_Budget_1500_Restart_1e-3` | 41.002 | 0.541 | -11.607 ± 10.715 | 17 / 1 / 39 | 258.80 |
