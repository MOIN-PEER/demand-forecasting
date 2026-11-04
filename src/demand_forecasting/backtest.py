"""Rolling-origin back-test: six 28-day folds, each trained only on data before its origin."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import models as M
from .config import FOLD_ORIGINS, HORIZON, QUANTILES, SEED
from .features import build_features
from .generate_data import load

MODELS = ["naive", "moving_avg", "holt_winters", "lgbm"]
LABELS = {"naive": "Seasonal naive", "moving_avg": "4-week average", "holt_winters": "Holt-Winters (ETS)",
          "lgbm": "LightGBM"}


def _weather_forecast(test: pd.DataFrame, origin: pd.Timestamp, rng) -> pd.Series:
    """Replace observed temperature by a forecast whose error grows with the horizon."""
    h = (test["date"] - origin).dt.days + 1
    return test["temp_c"] + rng.normal(0, 1, len(test)) * (0.6 + 0.12 * h)


def counterfactuals(model, test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """What the model believes promotions and warm weather do, measured on out-of-sample rows."""
    from .features import FEATURES
    base = test[(test["store_closed"] == 0)].copy()
    promo = base[base["on_promo"] == 1]
    out = {}
    if len(promo):
        no = promo.copy()
        no["on_promo"], no["discount"], no["promo_day"], no["price"] = 0, 0.0, 0, no["regular_price"]
        p1, p0 = model.predict(promo[FEATURES]), model.predict(no[FEATURES])
        out["promo"] = pd.DataFrame({"sku": promo["sku"].astype(str).values, "discount": promo["discount"].values,
                                     "pred_promo": p1, "pred_no_promo": p0})
    warm = base.copy()
    warm["temp_c"] = warm["temp_c"] + 5
    out["temp"] = pd.DataFrame({"sku": base["sku"].astype(str).values, "pred": model.predict(base[FEATURES]),
                                "pred_plus5": model.predict(warm[FEATURES])})
    return out


def run(origins=None, trees: int | None = None, verbose: bool = True):
    origins = origins or FOLD_ORIGINS
    n = int(os.getenv("DF_FOLDS", len(origins)))
    origins = origins[-n:]
    data = load()
    feat = build_features(data)
    rng = np.random.default_rng(SEED)
    preds, cf_promo, cf_temp, imps, timings = [], [], [], [], []
    for k, o in enumerate(origins, 1):
        o = pd.Timestamp(o)
        test = feat[(feat["date"] >= o) & (feat["date"] < o + pd.Timedelta(days=HORIZON))].copy()
        test["temp_c"] = _weather_forecast(test, o, rng)
        t0 = time.perf_counter()
        out = test[["date", "store_id", "sku", "category", "units", "stockout", "store_closed", "on_promo"]].copy()
        out["store_id"], out["sku"], out["category"] = (out[c].astype(str) for c in ("store_id", "sku", "category"))
        out["fold"], out["origin"] = k, o
        out["horizon"] = (test["date"] - o).dt.days + 1
        out["naive"] = M.seasonal_naive(feat, test, o)
        out["moving_avg"] = M.moving_average(feat, test, o)
        t1 = time.perf_counter()
        out["holt_winters"] = M.holt_winters(feat, test, o)
        t2 = time.perf_counter()
        mods = M.fit_lightgbm(feat, o, QUANTILES, trees)
        lg = M.predict_lightgbm(mods, test)
        t3 = time.perf_counter()
        for c in lg:
            out[c] = lg[c].values
        closed = out["store_closed"] == 1
        out.loc[closed, MODELS] = 0.0
        preds.append(out)
        cf = counterfactuals(mods["mean"], test)
        cf_promo.append(cf.get("promo"))
        cf_temp.append(cf["temp"])
        imps.append(pd.DataFrame({"feature": mods["mean"].booster_.feature_name(),
                                  "gain": mods["mean"].booster_.feature_importance("gain"), "fold": k}))
        timings.append({"fold": k, "baselines_s": round(t1 - t0, 1), "holt_winters_s": round(t2 - t1, 1),
                        "lightgbm_s": round(t3 - t2, 1)})
        if verbose:
            ok = (out["stockout"] == 0) & (out["store_closed"] == 0)
            w = {m: round(100 * np.abs(out.loc[ok, m] - out.loc[ok, "units"]).sum() / out.loc[ok, "units"].sum(), 1)
                 for m in MODELS}
            print(f"fold {k} origin {o.date()}  WAPE %: {w}  ({t3 - t0:.0f}s)")
    return {"predictions": pd.concat(preds, ignore_index=True),
            "cf_promo": pd.concat([c for c in cf_promo if c is not None], ignore_index=True),
            "cf_temp": pd.concat(cf_temp, ignore_index=True),
            "importance": pd.concat(imps, ignore_index=True), "timings": pd.DataFrame(timings), "data": data}
