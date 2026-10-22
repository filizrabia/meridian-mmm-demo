# Marketing mix modeling, end to end

A compact marketing mix model (MMM) on synthetic weekly data where the true channel effects are known, plus a script that fits the same data with [Google Meridian](https://github.com/google/meridian).

Because the ground truth is built into the data, the interesting question is not only "what does the model say?" but "does it recover what actually happened?"

![Channel ROI vs ground truth](images/roi.png)

## What's inside

| | |
|---|---|
| `mmm/transforms.py` | Geometric adstock (carryover) and Hill saturation (diminishing returns) |
| `mmm/data.py` | Four years of weekly revenue driven by baseline, trend, seasonality, promos and four media channels |
| `mmm/model.py` | MMM fitted by penalized least squares (a MAP estimate with weak priors), block-bootstrap ROI intervals, response curves and a budget optimizer |
| `meridian/fit_meridian.py` | The same dataset fitted with Meridian's Bayesian model |
| `run_demo.py` | Runs everything and writes the charts and `results.md` |

## Run it

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -t .
```

The numpy model runs in a few seconds. For the Meridian version, `pip install google-meridian` and run `python meridian/fit_meridian.py` (several minutes of MCMC on a CPU).

## Results

The model explains the data well (R² ≈ 0.98), and the decomposition separates baseline demand from media:

![Revenue decomposition](images/decomposition.png)

ROI recovery is the harder test. Ranking and magnitude are right for TV, search and social, and the true value falls inside the 90% interval for those three. **Display is overestimated** (0.73 true vs ~1.05 estimated): it is the smallest channel, and its spend moves together with the others, so the model cannot cleanly separate its effect. That is the typical MMM failure mode, and the reason real projects calibrate with lift tests.

Response curves show where each channel sits on its saturation curve:

![Response curves](images/response_curves.png)

Full numbers, including the budget reallocation, are in [results.md](results.md).

## Modeling notes

- **Adstock before saturation.** Spend is carried over first, then saturated, which matches how Meridian orders the two transforms by default.
- **Priors as penalties.** Weak penalties on decay, half-saturation (relative to average spend) and Hill slope keep the curves identifiable. Without them, saturation and scale trade off against each other.
- **Linear parts solved exactly.** For any guess of the media parameters, the baseline, trend, seasonality and promo coefficients are solved by least squares, so the optimizer only searches over the nonlinear media parameters.
- **Bounded budget moves.** The optimizer keeps each channel within ±50% of current spend, so recommendations stay inside the range the model has seen.

## License

MIT
