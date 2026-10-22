"""Media transformations used in marketing mix models.

Two effects are modeled for every channel:

* **Adstock (carryover):** spend this week keeps working in the following weeks,
  decaying geometrically.
* **Saturation (diminishing returns):** each extra euro buys less response than
  the one before it, following a Hill curve.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock(spend: np.ndarray, decay: float, max_lag: int = 8) -> np.ndarray:
    """Apply normalized geometric adstock to a 1-D spend series.

    The weights ``decay**0, decay**1, ..., decay**max_lag`` are normalized to sum
    to 1, so the transformed series keeps the same total as the input and the
    decay only changes *when* the spend has its effect.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    spend = np.asarray(spend, dtype=float)
    weights = decay ** np.arange(max_lag + 1)
    weights /= weights.sum()
    out = np.zeros_like(spend)
    for lag, w in enumerate(weights):
        out[lag:] += w * spend[: len(spend) - lag]
    return out


def hill(x: np.ndarray, half_saturation: float, slope: float) -> np.ndarray:
    """Hill saturation curve scaled to [0, 1).

    ``half_saturation`` is the input level that produces half of the maximum
    response; ``slope`` controls how sharply the curve bends.
    """
    if half_saturation <= 0 or slope <= 0:
        raise ValueError("half_saturation and slope must be positive")
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    return x**slope / (x**slope + half_saturation**slope)


def media_response(
    spend: np.ndarray,
    beta: float,
    decay: float,
    half_saturation: float,
    slope: float,
    max_lag: int = 8,
) -> np.ndarray:
    """Full channel response: adstock, then saturation, scaled by ``beta``."""
    return beta * hill(geometric_adstock(spend, decay, max_lag), half_saturation, slope)
