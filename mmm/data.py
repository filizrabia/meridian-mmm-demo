"""Synthetic weekly marketing data with a known ground truth.

Because the true channel effects are known, we can check whether a model
recovers them, which is impossible with real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .transforms import media_response


@dataclass(frozen=True)
class ChannelTruth:
    beta: float  # maximum weekly revenue the channel can drive
    decay: float  # adstock carryover
    half_saturation: float  # adstocked weekly spend at half of max response
    slope: float  # Hill curve shape
    mean_spend: float  # average weekly spend
    flighting: float  # how bursty the spend pattern is (0 = flat)


DEFAULT_CHANNELS: dict[str, ChannelTruth] = {
    "tv": ChannelTruth(beta=60_000, decay=0.7, half_saturation=30_000, slope=1.6, mean_spend=25_000, flighting=0.9),
    "search": ChannelTruth(beta=45_000, decay=0.1, half_saturation=12_000, slope=1.2, mean_spend=12_000, flighting=0.3),
    "social": ChannelTruth(beta=30_000, decay=0.4, half_saturation=10_000, slope=1.4, mean_spend=9_000, flighting=0.6),
    "display": ChannelTruth(beta=12_000, decay=0.3, half_saturation=9_000, slope=1.1, mean_spend=6_000, flighting=0.4),
}


@dataclass
class SyntheticDataset:
    df: pd.DataFrame
    channels: list[str]
    truth: dict[str, ChannelTruth] = field(default_factory=dict)
    true_contributions: pd.DataFrame | None = None

    def true_roi(self) -> pd.Series:
        """Revenue driven per euro spent, over the whole period."""
        contrib = self.true_contributions[self.channels].sum()
        spend = self.df[self.channels].sum()
        return (contrib / spend).rename("true_roi")


def make_dataset(
    n_weeks: int = 208,
    channels: dict[str, ChannelTruth] | None = None,
    noise_sd: float = 4_000,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate weekly revenue driven by baseline, trend, seasonality, promos and media."""
    rng = np.random.default_rng(seed)
    channels = channels or DEFAULT_CHANNELS
    t = np.arange(n_weeks)
    week = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")

    baseline = 200_000 + 250 * t
    seasonality = 25_000 * np.sin(2 * np.pi * t / 52) + 10_000 * np.cos(4 * np.pi * t / 52)
    promo = (rng.random(n_weeks) < 0.12).astype(float)
    promo_effect = 18_000 * promo

    df = pd.DataFrame({"week": week, "promo": promo})
    contributions = {}
    for name, ch in channels.items():
        # Bursty spend: a smooth base pattern plus on/off flights.
        flights = rng.random(n_weeks) < 0.55
        pattern = 1 + ch.flighting * (flights.astype(float) * 2 - 1)
        spend = np.clip(ch.mean_spend * pattern * rng.lognormal(0, 0.25, n_weeks), 0, None)
        df[name] = spend.round(2)
        contributions[name] = media_response(spend, ch.beta, ch.decay, ch.half_saturation, ch.slope)

    media_total = np.sum(list(contributions.values()), axis=0)
    noise = rng.normal(0, noise_sd, n_weeks)
    df["revenue"] = (baseline + seasonality + promo_effect + media_total + noise).round(2)

    true_contrib = pd.DataFrame(contributions)
    true_contrib.insert(0, "week", week)
    return SyntheticDataset(df=df, channels=list(channels), truth=dict(channels), true_contributions=true_contrib)
