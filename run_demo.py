"""End-to-end demo: generate data, fit the MMM, check it against the truth, plan budget.

    python run_demo.py

Writes charts to ``images/`` and a results summary to ``results.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm.data import make_dataset
from mmm.model import MarketingMixModel

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

PALETTE = {"baseline": "#C9CED6", "tv": "#3B5BDB", "search": "#2F9E44", "social": "#E8590C", "display": "#9C36B5"}
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def euros(x, _=None):
    return f"€{x / 1000:,.0f}k"


def main():
    ds = make_dataset()
    df = ds.df
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "synthetic_weekly.csv", index=False)
    model = MarketingMixModel(ds.channels).fit(df)
    contrib = model.contributions(df)

    # 1. Fit
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(df["week"], df["revenue"], color="#868E96", lw=1.2, label="Actual")
    ax.plot(df["week"], model.predict(df), color="#3B5BDB", lw=1.6, label="Model")
    ax.yaxis.set_major_formatter(euros)
    ax.set_title(f"Weekly revenue: actual vs model (R² = {model.r2(df):.2f})", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fit.png")
    plt.close(fig)

    # 2. Decomposition
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cols = ["baseline"] + ds.channels
    ax.stackplot(df["week"], *[contrib[c] for c in cols], labels=cols, colors=[PALETTE[c] for c in cols], alpha=0.9)
    ax.yaxis.set_major_formatter(euros)
    ax.set_title("Revenue decomposition by driver", loc="left")
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "decomposition.png")
    plt.close(fig)

    # 3. ROI vs truth, with bootstrap intervals
    boot = model.bootstrap_roi(df, n_boot=30)
    roi = pd.concat([ds.true_roi(), model.roi(df), boot], axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(ds.channels))
    ax.errorbar(
        roi["estimated_roi"], y,
        xerr=[roi["estimated_roi"] - roi["p05"], roi["p95"] - roi["estimated_roi"]],
        fmt="o", color="#3B5BDB", capsize=3, label="Estimated (90% interval)",
    )
    ax.scatter(roi["true_roi"], y, marker="D", color="#E8590C", zorder=3, label="True")
    ax.set_yticks(y, ds.channels)
    ax.invert_yaxis()
    ax.set_xlabel("Revenue per €1 spent")
    ax.set_title("Channel ROI: estimated vs ground truth", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "roi.png")
    plt.close(fig)

    # 4. Response curves
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for c in ds.channels:
        grid = np.linspace(0, df[c].max() * 1.5, 200)
        ax.plot(grid, model.response_curve(c, grid), color=PALETTE[c], lw=2, label=c)
        ax.axvline(df[c].mean(), color=PALETTE[c], ls=":", lw=1)
    ax.xaxis.set_major_formatter(euros)
    ax.yaxis.set_major_formatter(euros)
    ax.set_xlabel("Weekly spend (dotted = current average)")
    ax.set_ylabel("Weekly revenue")
    ax.set_title("Response curves", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "response_curves.png")
    plt.close(fig)

    # 5. Budget reallocation at the same total spend
    current = {c: float(df[c].mean()) for c in ds.channels}
    plan = model.optimize_budget(sum(current.values()), current, max_change=0.5)
    uplift = plan["optimized_revenue"].sum() / plan["current_revenue"].sum() - 1

    summary = [
        "## Results",
        "",
        f"Model fit: R² = {model.r2(df):.3f} on {len(df)} weeks.",
        "",
        "### ROI vs ground truth",
        "",
        roi[["true_roi", "estimated_roi", "p05", "p95"]].round(2).to_markdown(),
        "",
        "### Budget reallocation (same weekly budget, ±50% per channel)",
        "",
        plan.astype(int).to_markdown(),
        "",
        f"Media-driven revenue uplift at the same spend: **{uplift:+.1%}**",
    ]
    (Path(__file__).parent / "results.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
