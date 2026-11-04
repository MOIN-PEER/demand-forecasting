import numpy as np
import pandas as pd
import pytest

from demand_forecasting import metrics as m
from demand_forecasting import models as M
from demand_forecasting.inventory import critical_ratios, quantile_at, simulate

ORIGIN = pd.Timestamp("2025-12-01")


@pytest.fixture(scope="module")
def test_rows(feat):
    return feat[(feat["date"] >= ORIGIN) & (feat["date"] < ORIGIN + pd.Timedelta(days=28))]


def test_metrics():
    y, f = np.array([10, 0, 5]), np.array([8, 1, 5])
    assert m.wape(y, f) == pytest.approx(3 / 15)
    assert m.bias(y, f) == pytest.approx(-1 / 15)
    assert m.coverage(y, f - 2, f + 2) == 1.0
    assert m.pinball([10], [8], 0.9) == pytest.approx(1.8)


def test_seasonal_naive_uses_last_week(feat, test_rows):
    f = M.seasonal_naive(feat, test_rows, ORIGIN)
    r = test_rows.iloc[0]
    week_before = feat[(feat["store_id"] == r["store_id"]) & (feat["sku"] == r["sku"])
                       & (feat["date"] == r["date"] - pd.Timedelta(days=7))]["units"].iloc[0]
    assert f[0] == week_before


def test_baselines_shapes(feat, test_rows):
    for fn in (M.seasonal_naive, M.moving_average):
        out = fn(feat, test_rows, ORIGIN)
        assert len(out) == len(test_rows) and np.isfinite(out).all() and (out >= 0).all()


def test_lightgbm_beats_naive_and_quantiles_ordered(feat, test_rows):
    mods = M.fit_lightgbm(feat, ORIGIN, quantiles=[0.1, 0.9], trees=120)
    p = M.predict_lightgbm(mods, test_rows)
    ok = (test_rows["stockout"] == 0) & (test_rows["store_closed"] == 0)
    naive = M.seasonal_naive(feat, test_rows, ORIGIN)
    assert m.wape(test_rows.loc[ok, "units"], p.loc[ok, "lgbm"]) < m.wape(test_rows.loc[ok, "units"], naive[ok.to_numpy()])
    assert (p["q10"] <= p["q90"]).all()
    assert (p.loc[test_rows["store_closed"] == 1].to_numpy() == 0).all()


def test_training_excludes_future_and_stockouts(feat):
    tr = M._train_rows(feat, ORIGIN)
    assert tr["date"].max() < ORIGIN
    assert tr["stockout"].sum() == 0 and tr["store_closed"].sum() == 0


def test_critical_ratio_and_quantile_interpolation(data):
    cr = critical_ratios(data["products"])
    assert cr["shelf_life_days"].max() <= 7 and cr["critical_ratio"].between(0.5, 0.8).all()
    q = np.array([[2, 10, 14, 20]])
    assert quantile_at(q, [0.1, 0.5, 0.75, 0.9], np.array([0.625]))[0] == pytest.approx(12)


def test_inventory_simulation_accounts_balance(data):
    pred = pd.DataFrame({"sku": ["P01"] * 3, "units": [10, 20, 30], "stockout": 0, "store_closed": 0,
                         "naive": [12, 12, 12], "moving_avg": [20, 20, 20], "lgbm": [10, 20, 30],
                         "q10": 5, "q50": [10, 20, 30], "q75": [12, 22, 32], "q90": [15, 25, 35]})
    res, _ = simulate(pred, data["products"])
    for _, r in res.iterrows():
        assert r["sold"] + r["lost_units"] == r["demand"] and r["sold"] + r["waste_units"] == r["ordered"]
    perfect = res.set_index("policy").loc["LightGBM average forecast"]
    assert perfect["fill_rate_pct"] == 100 and perfect["waste_units"] == 0
