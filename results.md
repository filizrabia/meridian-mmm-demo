## Results

Model fit: R² = 0.981 on 208 weeks.
Holdout (last 26 weeks, not used for fitting): MAPE 1.07% vs 0.97% on training weeks.

### ROI vs ground truth (model only, 90% bootstrap interval)

|         |   true_roi |   estimated_roi |   p05 |   p95 |
|:--------|-----------:|----------------:|------:|------:|
| tv      |       0.96 |            1.12 |  0.95 |  1.2  |
| search  |       1.75 |            1.56 |  1.14 |  1.95 |
| social  |       1.43 |            1.46 |  1.06 |  1.72 |
| display |       0.73 |            1.05 |  0.97 |  1.58 |

### Calibrated with a lift test on display (measured ROI 0.63 ± 0.12)

|         |   true |   uncalibrated |   calibrated |
|:--------|-------:|---------------:|-------------:|
| tv      |   0.96 |           1.12 |         1.13 |
| search  |   1.75 |           1.56 |         1.56 |
| social  |   1.43 |           1.46 |         1.44 |
| display |   0.73 |           1.05 |         0.63 |

### Budget reallocation (calibrated model, same weekly budget, ±50% per channel)

|         |   current_spend |   optimized_spend |   current_revenue |   optimized_revenue |
|:--------|----------------:|------------------:|------------------:|--------------------:|
| tv      |           25802 |             24666 |             30800 |               29801 |
| search  |           12993 |             15184 |             21053 |               23186 |
| social  |            9684 |             11975 |             14842 |               17108 |
| display |            6692 |              3346 |              4414 |                1577 |

Media-driven revenue uplift at the same spend: **+0.8%**
