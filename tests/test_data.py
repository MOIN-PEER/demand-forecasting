import numpy as np
import pandas as pd


def test_shape_and_keys(data):
    s = data["sales"]
    assert s["store_id"].nunique() == 10 and s["sku"].nunique() == 30
    assert not s.duplicated(["date", "store_id", "sku"]).any()
    assert (s["units"] >= 0).all()


def test_closed_days_have_no_sales(data):
    s = data["sales"]
    closed = s[s["store_closed"] == 1]
    assert len(closed) > 0 and (closed["units"] == 0).all()
    assert set(closed["date"].dt.strftime("%m-%d")) >= {"12-25"}
    assert pd.Timestamp("2024-03-31") in set(closed["date"])          # Easter Sunday 2024


def test_promo_prices_are_discounted(data):
    s = data["sales"].merge(data["products"][["sku", "regular_price"]], on="sku")
    promo = s[s["on_promo"] == 1]
    assert (promo["price"] < promo["regular_price"]).all()
    assert (s.loc[s["on_promo"] == 0, "price"] == s.loc[s["on_promo"] == 0, "regular_price"]).all()


def test_built_in_effects_are_visible(data):
    s = data["sales"].merge(data["weather"], on=["date", "store_id"])
    open_ = s[(s["store_closed"] == 0) & (s["on_promo"] == 0)]
    by_dow = open_.groupby(open_["date"].dt.day_of_week)["units"].mean()
    assert by_dow.idxmax() == 5 and by_dow.idxmin() == 6              # Saturday peak, Sunday low
    ic = open_[open_["sku"] == "P13"]
    assert ic.loc[ic["temp_c"] > 15, "units"].mean() > 2 * ic.loc[ic["temp_c"] < 5, "units"].mean()
    pies = s[s["sku"] == "P27"]
    assert pies.loc[pies["date"].dt.month < 11, "units"].sum() == 0
    assert s.loc[(s["sku"] == "P12"), "date"].min() == pd.Timestamp("2024-03-01")


def test_stockouts_reduce_sales(data):
    s = data["sales"]
    e = data["expected_demand"]
    m = s.merge(e, on=["date", "store_id", "sku"])
    assert 0.005 < s["stockout"].mean() < 0.05
    assert m.loc[m["stockout"] == 1, "units"].mean() < m.loc[m["stockout"] == 1, "expected_units"].mean()
    assert np.isfinite(m["expected_units"]).all()
