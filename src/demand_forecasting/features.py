"""Feature engineering for a direct 1-28 day forecast.

Every history-based feature uses data at least 28 days old, so one model can forecast any day in the
next four weeks without seeing the future. Plans that are known in advance (calendar, promotions,
prices, store closures) and the weather forecast are used for the target day itself.
Stockout and closed days are blanked out of the history so the model learns demand, not lost sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

from .config import HORIZON

LAGS = [28, 35, 42, 49, 56, 364]
CATEGORICAL = ["store_id", "sku", "category"]


def _days_to(dates: pd.Series, targets: dict[int, pd.Timestamp], clip: int) -> np.ndarray:
    tgt = dates.dt.year.map(targets)
    nxt = dates.dt.year.add(1).map(lambda y: targets.get(y, pd.Timestamp(f"{y}-12-25")))
    d = (tgt - dates).dt.days
    d = np.where(d < 0, (nxt - dates).dt.days, d)
    return np.clip(d, 0, clip)


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    years = range(dates.dt.year.min(), dates.dt.year.max() + 2)
    xmas = {y: pd.Timestamp(f"{y}-12-25") for y in years}
    east = {y: pd.Timestamp(easter(y)) for y in years}
    last_fri = (dates.dt.day_of_week == 4) & ((dates + pd.Timedelta(days=7)).dt.month != dates.dt.month)
    return pd.DataFrame({
        "dow": dates.dt.day_of_week, "month": dates.dt.month, "doy": dates.dt.dayofyear,
        "dom": dates.dt.day, "week": dates.dt.isocalendar().week.astype(int).values,
        "days_to_christmas": _days_to(dates, xmas, 60), "days_to_easter": _days_to(dates, east, 60),
        "payday": last_fri.astype(int).values,
    }, index=dates.index)


def build_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    s = data["sales"].merge(data["products"][["sku", "category", "regular_price"]], on="sku")
    s = s.merge(data["stores"][["store_id", "size_index"]], on="store_id")
    s = s.merge(data["weather"], on=["date", "store_id"], how="left")
    cal = data["calendar"][["date", "holiday"]].copy()
    cal["is_holiday"] = (cal["holiday"].fillna("") != "").astype(int)
    cal["day_before_holiday"] = cal["is_holiday"].shift(-1, fill_value=0)
    s = s.merge(cal[["date", "is_holiday", "day_before_holiday"]], on="date", how="left")
    s = s.sort_values(["store_id", "sku", "date"]).reset_index(drop=True)

    # Clean history: lost sales (stockouts) and closures are not real demand
    clean = s["units"].astype(float).where((s["stockout"] == 0) & (s["store_closed"] == 0))
    g = clean.groupby([s["store_id"], s["sku"]], sort=False)
    for k in LAGS:
        s[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(HORIZON)
    gs = shifted.groupby([s["store_id"], s["sku"]], sort=False)
    for w in (7, 28, 91):
        s[f"mean_{w}"] = gs.transform(lambda x, w=w: x.rolling(w, min_periods=max(3, w // 3)).mean())
    s["std_28"] = gs.transform(lambda x: x.rolling(28, min_periods=10).std())
    s["same_dow_mean"] = s[[f"lag_{k}" for k in (28, 35, 42, 49)]].mean(axis=1)
    s["dow_ratio"] = s["same_dow_mean"] / (s["mean_28"] + 0.5)
    # Chain-level demand for the product (helps new or sparse store-product series)
    chain = s.groupby(["sku", "date"])["mean_28"].transform("mean")
    s["sku_chain_mean_28"] = chain

    s = pd.concat([s, calendar_features(s["date"])], axis=1)
    s["discount"] = (1 - s["price"] / s["regular_price"]).round(3)
    promo_g = s.groupby(["store_id", "sku"], sort=False)["on_promo"]
    starts = (s["on_promo"].diff().fillna(0) > 0) | ((s["on_promo"] == 1) & (promo_g.shift(1).fillna(0) == 0))
    run_id = starts.groupby([s["store_id"], s["sku"]]).cumsum()
    s["promo_day"] = np.where(s["on_promo"] == 1, s.groupby([s["store_id"], s["sku"], run_id]).cumcount() + 1, 0)
    ended = (promo_g.shift(1).fillna(0) == 1) & (s["on_promo"] == 0)
    end_id = ended.groupby([s["store_id"], s["sku"]]).cumsum()
    since = s.groupby([s["store_id"], s["sku"], end_id]).cumcount() + 1
    s["post_promo"] = ((end_id > 0) & (since <= 7) & (s["on_promo"] == 0)).astype(int)
    for c in CATEGORICAL:
        s[c] = s[c].astype("category")
    return s


FEATURES = (CATEGORICAL + ["size_index", "regular_price", "price", "discount", "on_promo", "promo_day", "post_promo",
                           "temp_c", "is_holiday", "day_before_holiday", "store_closed",
                           "dow", "month", "doy", "dom", "week", "days_to_christmas", "days_to_easter", "payday"]
            + [f"lag_{k}" for k in LAGS] + ["mean_7", "mean_28", "mean_91", "std_28", "same_dow_mean", "dow_ratio",
                                           "sku_chain_mean_28"])
