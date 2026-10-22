## Results

Model fit: R² = 0.981 on 208 weeks.

### ROI vs ground truth

|         |   true_roi |   estimated_roi |   p05 |   p95 |
|:--------|-----------:|----------------:|------:|------:|
| tv      |       0.96 |            1.12 |  0.95 |  1.2  |
| search  |       1.75 |            1.56 |  1.14 |  1.95 |
| social  |       1.43 |            1.46 |  1.06 |  1.72 |
| display |       0.73 |            1.05 |  0.97 |  1.58 |

### Budget reallocation (same weekly budget, ±50% per channel)

|         |   current_spend |   optimized_spend |   current_revenue |   optimized_revenue |
|:--------|----------------:|------------------:|------------------:|--------------------:|
| tv      |           25802 |             23033 |             30616 |               28119 |
| search  |           12993 |             14688 |             21057 |               22760 |
| social  |            9684 |             11541 |             15046 |               16932 |
| display |            6692 |              5910 |              7366 |                6669 |

Media-driven revenue uplift at the same spend: **+0.5%**
