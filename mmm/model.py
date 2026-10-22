"""A compact marketing mix model fitted by penalized least squares.

Revenue is modeled as

    revenue_t = intercept + trend * t + seasonality_t + promo_effect * promo_t
                + sum_c beta_c * hill(adstock(spend_c), half_sat_c, slope_c) + noise

The weak penalties on the media parameters act like priors (the fit is a MAP
estimate), which keeps saturation and decay identifiable on a few years of
weekly data. This is the same structure Meridian uses, without the full
Bayesian sampler, so it runs in seconds with numpy and scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from .transforms import hill, media_response

_PARAMS_PER_CHANNEL = 4  # log_beta, logit_decay, log_half_sat_ratio, log_slope


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1 - p))


@dataclass
class ChannelParams:
    beta: float
    decay: float
    half_saturation: float
    slope: float


class MarketingMixModel:
    def __init__(self, channels: list[str], max_lag: int = 8, prior_strength: float = 3.0, seed: int = 0):
        self.channels = list(channels)
        self.max_lag = max_lag
        self.prior_strength = prior_strength
        self.seed = seed
        self.params_: dict[str, ChannelParams] | None = None
        self.base_coef_: np.ndarray | None = None

    # ---- design ---------------------------------------------------------
    @staticmethod
    def _base_matrix(n: int, promo: np.ndarray) -> np.ndarray:
        t = np.arange(n)
        return np.column_stack(
            [
                np.ones(n),
                t / n,
                np.sin(2 * np.pi * t / 52),
                np.cos(2 * np.pi * t / 52),
                np.sin(4 * np.pi * t / 52),
                np.cos(4 * np.pi * t / 52),
                promo,
            ]
        )

    def _unpack(self, theta: np.ndarray) -> dict[str, ChannelParams]:
        out = {}
        for i, ch in enumerate(self.channels):
            lb, ld, lh, ls = theta[i * _PARAMS_PER_CHANNEL : (i + 1) * _PARAMS_PER_CHANNEL]
            out[ch] = ChannelParams(
                beta=float(np.exp(lb)),
                decay=float(_sigmoid(ld) * 0.95),
                half_saturation=float(np.exp(lh) * self._spend_scale[ch]),
                slope=float(np.exp(ls)),
            )
        return out

    def _media_matrix(self, spend: pd.DataFrame, params: dict[str, ChannelParams]) -> np.ndarray:
        return np.column_stack(
            [
                media_response(spend[ch].to_numpy(), p.beta, p.decay, p.half_saturation, p.slope, self.max_lag)
                for ch, p in params.items()
            ]
        )

    # ---- fitting --------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        target: str = "revenue",
        promo: str = "promo",
        n_starts: int = 6,
        lift_tests: dict[str, tuple[float, float]] | None = None,
        train_mask: np.ndarray | None = None,
    ):
        """Fit the model.

        lift_tests: optional experiment results to calibrate against, as
            ``{channel: (measured_roi, standard_error)}``. Each one adds a penalty
            pulling the model's ROI for that channel towards the experiment,
            weighted by how precise the experiment was.
        train_mask: optional boolean array; weeks marked False are left out of
            the fit (for holdout validation) but still carry adstock forward.
        """
        y = df[target].to_numpy(float)
        spend = df[self.channels]
        mask = np.ones(len(df), bool) if train_mask is None else np.asarray(train_mask, bool)
        self._spend_scale = {ch: float(spend.loc[mask, ch].mean()) for ch in self.channels}
        X_base = self._base_matrix(len(df), df[promo].to_numpy(float))
        y_scale = float(np.std(y[mask]))
        y_mean = float(np.mean(y[mask]))
        lift_tests = lift_tests or {}
        lift_idx = [self.channels.index(c) for c in lift_tests]
        lift_target = np.array([v[0] for v in lift_tests.values()])
        lift_se = np.array([v[1] for v in lift_tests.values()])
        spend_tot = spend.loc[mask].sum().to_numpy()

        prior_mu = np.array([np.log(y_mean * 0.1), _logit(0.4 / 0.95), 0.0, np.log(1.3)])
        prior_sd = np.array([2.0, 1.5, 0.6, 0.4])
        n_train = int(mask.sum())

        def residuals(theta):
            params = self._unpack(theta)
            media_mat = self._media_matrix(spend, params)
            media = media_mat.sum(axis=1)
            # Base coefficients are linear given the media fit: solve them exactly.
            coef, *_ = np.linalg.lstsq(X_base[mask], (y - media)[mask], rcond=None)
            fit_res = ((y - media - X_base @ coef) / y_scale)[mask]
            theta_mat = theta.reshape(len(self.channels), _PARAMS_PER_CHANNEL)
            prior_res = self.prior_strength * ((theta_mat - prior_mu) / prior_sd).ravel() / np.sqrt(n_train)
            parts = [fit_res, prior_res]
            if lift_idx:
                roi = media_mat[mask][:, lift_idx].sum(axis=0) / spend_tot[lift_idx]
                parts.append((roi - lift_target) / lift_se)
            return np.concatenate(parts)

        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(n_starts):
            x0 = np.tile(prior_mu, len(self.channels)) + rng.normal(0, 0.5, len(self.channels) * _PARAMS_PER_CHANNEL)
            res = least_squares(residuals, x0, method="trf", max_nfev=4000)
            if best is None or res.cost < best.cost:
                best = res

        self.theta_ = best.x
        self.params_ = self._unpack(best.x)
        media = self._media_matrix(spend, self.params_).sum(axis=1)
        self.base_coef_, *_ = np.linalg.lstsq(X_base[mask], (y - media)[mask], rcond=None)
        self._n_weeks = len(df)
        self.lift_tests_ = lift_tests
        return self

    # ---- outputs --------------------------------------------------------
    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weekly revenue attributed to each channel and to the baseline."""
        media = self._media_matrix(df[self.channels], self.params_)
        X_base = self._base_matrix(len(df), df["promo"].to_numpy(float))
        out = pd.DataFrame(media, columns=self.channels, index=df.index)
        out.insert(0, "baseline", X_base @ self.base_coef_)
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.contributions(df).sum(axis=1).to_numpy()

    def roi(self, df: pd.DataFrame) -> pd.Series:
        contrib = self.contributions(df)[self.channels].sum()
        return (contrib / df[self.channels].sum()).rename("estimated_roi")

    def r2(self, df: pd.DataFrame) -> float:
        y = df["revenue"].to_numpy()
        resid = y - self.predict(df)
        return float(1 - resid.var() / y.var())

    def bootstrap_roi(self, df: pd.DataFrame, n_boot: int = 30, block: int = 8, seed: int = 1) -> pd.DataFrame:
        """ROI uncertainty from a moving-block residual bootstrap.

        Residuals are resampled in blocks of ``block`` weeks to keep their
        autocorrelation, added back to the fitted values, and the model is
        refitted. Returns the 5th, 50th and 95th percentile ROI per channel.
        """
        rng = np.random.default_rng(seed)
        fitted = self.predict(df)
        resid = df["revenue"].to_numpy() - fitted
        n = len(resid)
        rois = []
        for _ in range(n_boot):
            starts = rng.integers(0, n - block, size=n // block + 1)
            boot_resid = np.concatenate([resid[s : s + block] for s in starts])[:n]
            boot_df = df.copy()
            boot_df["revenue"] = fitted + boot_resid
            m = MarketingMixModel(self.channels, self.max_lag, self.prior_strength, seed=int(rng.integers(1e9)))
            m.fit(boot_df, n_starts=2, lift_tests=self.lift_tests_)
            rois.append(m.roi(boot_df))
        rois = pd.DataFrame(rois)
        return rois.quantile([0.05, 0.5, 0.95]).T.rename(columns={0.05: "p05", 0.5: "p50", 0.95: "p95"})

    def response_curve(self, channel: str, weekly_spend: np.ndarray) -> np.ndarray:
        """Steady-state weekly revenue for a constant weekly spend level.

        With normalized adstock, a constant spend s settles at adstock s, so the
        long-run response is simply beta * hill(s).
        """
        p = self.params_[channel]
        return p.beta * hill(weekly_spend, p.half_saturation, p.slope)

    def optimize_budget(self, total_weekly_budget: float, current: dict[str, float], max_change: float = 0.5) -> pd.DataFrame:
        """Reallocate a weekly budget across channels to maximize steady-state revenue.

        Each channel may move at most ``max_change`` (e.g. 50%) away from its
        current level, which keeps recommendations within the range the model
        has actually seen.
        """
        chans = self.channels
        cur = np.array([current[c] for c in chans], float)
        bounds = [(c * (1 - max_change), c * (1 + max_change)) for c in cur]

        def neg_revenue(s):
            return -sum(self.response_curve(c, np.array([v]))[0] for c, v in zip(chans, s))

        cons = {"type": "eq", "fun": lambda s: s.sum() - total_weekly_budget}
        x0 = cur * total_weekly_budget / cur.sum()
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(neg_revenue, x0, bounds=bounds, constraints=[cons], method="SLSQP")

        cur_rev = np.array([self.response_curve(c, np.array([v]))[0] for c, v in zip(chans, cur)])
        opt_rev = np.array([self.response_curve(c, np.array([v]))[0] for c, v in zip(chans, res.x)])
        return pd.DataFrame(
            {
                "current_spend": cur,
                "optimized_spend": res.x,
                "current_revenue": cur_rev,
                "optimized_revenue": opt_rev,
            },
            index=chans,
        ).round(0)
