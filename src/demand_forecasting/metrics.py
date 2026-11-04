"""Forecast accuracy measures used throughout the project."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y, f) -> float:
    """Weighted absolute percentage error: total absolute error / total actual units."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(f - y).sum() / max(y.sum(), 1e-9))


def bias(y, f) -> float:
    """Total over (+) or under (-) forecast as a share of actual units."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f - y).sum() / max(y.sum(), 1e-9))


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(f, float) - np.asarray(y, float))))


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(f, float) - np.asarray(y, float)) ** 2)))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def summarise(df: pd.DataFrame, models: list[str], by: list[str] | None = None) -> pd.DataFrame:
    def one(g):
        return pd.Series({**{f"wape_{m}": round(100 * wape(g["units"], g[m]), 2) for m in models},
                          **{f"bias_{m}": round(100 * bias(g["units"], g[m]), 2) for m in models},
                          "units": int(g["units"].sum()), "rows": len(g)})
    if not by:
        return one(df).to_frame().T
    return df.groupby(by, observed=True).apply(one, include_groups=False).reset_index()
