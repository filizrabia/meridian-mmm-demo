"""Fit the same synthetic dataset with Google Meridian.

    pip install google-meridian
    python run_demo.py              # writes data/synthetic_weekly.csv
    python meridian/fit_meridian.py

This runs the full Bayesian model (MCMC), so it takes several minutes on a CPU.
The data is national (no geo column), with spend used as the media input.
"""

from __future__ import annotations

from pathlib import Path

from meridian.analysis import analyzer, optimizer
from meridian.data import load
from meridian.model import model, prior_distribution, spec

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["tv", "search", "social", "display"]


def main():
    coord_to_columns = load.CoordToColumns(
        time="week",
        kpi="revenue",
        controls=["promo"],
        media=CHANNELS,
        media_spend=CHANNELS,
    )
    loader = load.CsvDataLoader(
        csv_path=str(ROOT / "data" / "synthetic_weekly.csv"),
        kpi_type="revenue",
        coord_to_columns=coord_to_columns,
        media_to_channel={c: c for c in CHANNELS},
        media_spend_to_channel={c: c for c in CHANNELS},
    )
    data = loader.load()

    # A weakly informative ROI prior centred near 1, the same assumption the
    # numpy model makes through its penalty terms.
    prior = prior_distribution.PriorDistribution()
    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(prior=prior))
    mmm.sample_prior(500)
    mmm.sample_posterior(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)

    summary = analyzer.Analyzer(mmm).summary_metrics(confidence_level=0.9)
    print(summary["roi"].to_dataframe())

    opt = optimizer.BudgetOptimizer(mmm).optimize()
    opt.output_optimization_summary("meridian_optimization.html", str(ROOT))


if __name__ == "__main__":
    main()
