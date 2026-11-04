"""Forecasting models: three baselines that a retailer might already use, and LightGBM.

All models produce forecasts for the HORIZON days after an origin date, using only data before it.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import HORIZON, LGB_PARAMS, QUANTILES
from .features import FEATURES


def _history(feat: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    return feat[feat["date"] < origin]


def seasonal_naive(feat: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """Same weekday in the last full week before the origin ("what we sold last Tuesday")."""
    hist = _history(feat, origin)
    last = hist[hist["date"] >= origin - pd.Timedelta(days=7)]
    key = last.set_index(["store_id", "sku", last["date"].dt.day_of_week])["units"]
    idx = pd.MultiIndex.from_arrays([test["store_id"], test["sku"], test["date"].dt.day_of_week])
    return key.reindex(idx).fillna(0).to_numpy(float)


def moving_average(feat: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp, weeks: int = 4) -> np.ndarray:
    """Average of the same weekday over the last 4 weeks, ignoring stockout and closed days."""
    hist = _history(feat, origin)
    last = hist[(hist["date"] >= origin - pd.Timedelta(days=7 * weeks)) & (hist["stockout"] == 0)
                & (hist["store_closed"] == 0)]
    key = last.groupby(["store_id", "sku", last["date"].dt.day_of_week], observed=True)["units"].mean()
    idx = pd.MultiIndex.from_arrays([test["store_id"], test["sku"], test["date"].dt.day_of_week])
    return key.reindex(idx).fillna(0).to_numpy(float)


def holt_winters(feat: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp, history_days: int = 364) -> np.ndarray:
    """Classical exponential smoothing per store-product series (additive trend-free, weekly seasonality),
    fitted on log(1 + units) over the last year."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    hist = feat[(feat["date"] < origin) & (feat["date"] >= origin - pd.Timedelta(days=history_days))]
    out = pd.Series(0.0, index=test.index)
    for (st, sk), h in hist.groupby(["store_id", "sku"], observed=True, sort=False):
        y = h["units"].astype(float).where((h["stockout"] == 0) & (h["store_closed"] == 0))
        y = np.log1p(y.interpolate(limit_direction="both").fillna(0).to_numpy())
        rows = test.index[(test["store_id"] == st) & (test["sku"] == sk)]
        if len(rows) == 0:
            continue
        if len(y) < 28 or y.std() == 0:
            out[rows] = np.expm1(y[-28:].mean()) if len(y) else 0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y, seasonal="add", seasonal_periods=7, initialization_method="estimated").fit()
            fc = fit.forecast(HORIZON)
        steps = (test.loc[rows, "date"] - origin).dt.days.to_numpy()
        out[rows] = np.clip(np.expm1(fc[steps]), 0, None)
    return out.to_numpy()


def _train_rows(feat: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    hist = _history(feat, origin)
    # learn demand: drop stockout days (sales understate demand) and closed days (always zero)
    return hist[(hist["stockout"] == 0) & (hist["store_closed"] == 0) & hist["mean_28"].notna()]


def fit_lightgbm(feat: pd.DataFrame, origin: pd.Timestamp, quantiles=QUANTILES, trees: int | None = None) -> dict:
    tr = _train_rows(feat, origin)
    X, y = tr[FEATURES], tr["units"].astype(float)
    params = dict(LGB_PARAMS)
    if trees:
        params["n_estimators"] = trees
    models = {"mean": lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **params).fit(X, y)}
    for q in quantiles:
        qp = dict(params, n_estimators=max(150, params["n_estimators"] // 2))
        models[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, **qp).fit(X, y)
    return models


def predict_lightgbm(models: dict, test: pd.DataFrame) -> pd.DataFrame:
    X = test[FEATURES]
    out = pd.DataFrame({"lgbm": np.clip(models["mean"].predict(X), 0, None)}, index=test.index)
    qs = sorted(k for k in models if k != "mean")
    if qs:
        Q = np.column_stack([np.clip(models[q].predict(X), 0, None) for q in qs])
        Q = np.sort(Q, axis=1)                       # no quantile crossing
        for i, q in enumerate(qs):
            out[f"q{int(q * 100)}"] = Q[:, i]
    closed = test["store_closed"].to_numpy() == 1
    out.loc[closed, :] = 0.0
    return out
