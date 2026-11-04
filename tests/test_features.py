import numpy as np
import pandas as pd

from demand_forecasting.features import FEATURES, build_features, calendar_features


def test_no_future_leakage(data, feat):
    """Changing sales on or after a date must not change any history feature before date + 28 days."""
    cut = pd.Timestamp("2025-06-01")
    changed = {k: v.copy() for k, v in data.items()}
    s = changed["sales"]
    s.loc[s["date"] >= cut, "units"] = s.loc[s["date"] >= cut, "units"] * 3 + 50
    f2 = build_features(changed)
    hist_cols = [c for c in FEATURES if c.startswith(("lag_", "mean_", "std_", "same_dow", "dow_ratio", "sku_chain"))]
    a = feat.loc[feat["date"] < cut + pd.Timedelta(days=28), hist_cols].to_numpy(float)
    b = f2.loc[f2["date"] < cut + pd.Timedelta(days=28), hist_cols].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)
    later = f2["date"] == cut + pd.Timedelta(days=28)
    assert not np.allclose(feat.loc[later, "lag_28"], f2.loc[later, "lag_28"], equal_nan=True)


def test_stockout_days_are_blanked_from_history(feat):
    s = feat[(feat["store_id"] == "S01") & (feat["sku"] == "P01")].reset_index(drop=True)
    idx = s.index[(s["stockout"] == 1) & (s.index + 28 < len(s))]
    assert len(idx) > 0
    assert s.loc[idx + 28, "lag_28"].isna().all()


def test_calendar_features():
    c = calendar_features(pd.Series(pd.to_datetime(["2025-12-20", "2025-12-26", "2025-04-18", "2025-10-31"])))
    assert list(c["days_to_christmas"]) == [5, 60, 60, 55]
    assert c.loc[2, "days_to_easter"] == 2                      # Easter Sunday 20 April 2025
    assert c.loc[3, "payday"] == 1                               # last Friday of October 2025


def test_promo_day_counter(feat):
    s = feat[(feat["store_id"] == "S02") & (feat["sku"] == "P20")]
    runs = s.loc[s["on_promo"] == 1, "promo_day"]
    assert runs.min() == 1 and runs.max() in (7, 14)
    assert (s.loc[s["on_promo"] == 0, "promo_day"] == 0).all()
    assert s["post_promo"].sum() > 0
