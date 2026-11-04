"""Turn forecasts into daily order quantities for short-life products and count the cost of each policy.

For a product that must sell on the day (or be reduced to clear), the best order is a quantile of the
demand forecast, not the average: the "newsvendor" critical ratio

    critical ratio = underage cost / (underage cost + overage cost)

"Mismatch cost" = waste cost + lost margin + lost goodwill: the total the order policy is trying to minimise.

underage (each unit short):  lost margin (price - cost) + lost goodwill (20% of price)
overage  (each unit unsold): cost - salvage, where reduced-to-clear recovers 20% of cost

Simplification: every short-life product is treated as a one-day decision with demand on non-stockout
days as the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOODWILL, SALVAGE = 0.20, 0.20
MAX_SHELF_LIFE = 7


def critical_ratios(products: pd.DataFrame) -> pd.DataFrame:
    p = products[products["shelf_life_days"] <= MAX_SHELF_LIFE].copy()
    cu = p["regular_price"] - p["unit_cost"] + GOODWILL * p["regular_price"]
    co = p["unit_cost"] * (1 - SALVAGE)
    p["critical_ratio"] = (cu / (cu + co)).round(3)
    return p


def quantile_at(row_q: np.ndarray, levels: list[float], target: np.ndarray) -> np.ndarray:
    """Linear interpolation of each row's forecast quantiles at its own target level."""
    out = np.empty(len(target))
    for i in range(len(target)):
        out[i] = np.interp(target[i], levels, row_q[i])
    return out


def simulate(pred: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cr = critical_ratios(products)
    d = pred[(pred["stockout"] == 0) & (pred["store_closed"] == 0) & pred["sku"].isin(cr["sku"])].copy()
    d = d.drop(columns=[c for c in ("product", "regular_price") if c in d]).merge(
        cr[["sku", "product", "regular_price", "unit_cost", "critical_ratio"]], on="sku")
    levels = [0.1, 0.5, 0.75, 0.9]
    qcols = ["q10", "q50", "q75", "q90"]
    d["service_quantile"] = quantile_at(d[qcols].to_numpy(), levels, d["critical_ratio"].to_numpy())
    policies = {"Last week's sales (naive)": "naive", "4-week average": "moving_avg",
                "LightGBM average forecast": "lgbm", "LightGBM + target service level": "service_quantile"}
    rows, per_prod = [], []
    price = d["regular_price"].to_numpy()
    cost = d["unit_cost"].to_numpy()
    demand = d["units"].to_numpy(float)
    for name, col in policies.items():
        order = np.round(d[col].to_numpy())
        sold = np.minimum(order, demand)
        waste, lost = order - sold, demand - sold
        margin = sold * (price - cost)
        waste_cost = waste * cost * (1 - SALVAGE)
        lost_margin = lost * (price - cost)
        goodwill = lost * GOODWILL * price
        rows.append({"policy": name, "ordered": int(order.sum()), "demand": int(demand.sum()),
                     "sold": int(sold.sum()), "waste_units": int(waste.sum()), "lost_units": int(lost.sum()),
                     "fill_rate_pct": round(100 * sold.sum() / demand.sum(), 1),
                     "waste_pct_of_ordered": round(100 * waste.sum() / order.sum(), 1),
                     "waste_cost_gbp": round(waste_cost.sum(), 0), "lost_margin_gbp": round(lost_margin.sum(), 0),
                     "gross_margin_gbp": round(margin.sum(), 0),
                     "net_profit_gbp": round((margin - waste_cost).sum(), 0),
                     "mismatch_cost_gbp": round((waste_cost + lost_margin + goodwill).sum(), 0)})
        g = pd.DataFrame({"product": d["product"], "waste_cost": waste_cost, "lost_margin": lost_margin,
                          "profit": margin - waste_cost}).groupby("product").sum()
        g["policy"] = name
        per_prod.append(g.reset_index())
    res = pd.DataFrame(rows)
    base = res.loc[0, "net_profit_gbp"]
    res["profit_vs_naive_gbp"] = (res["net_profit_gbp"] - base).round(0)
    res["mismatch_vs_naive_pct"] = (100 * (res["mismatch_cost_gbp"] / res.loc[0, "mismatch_cost_gbp"] - 1)).round(1)
    return res, pd.concat(per_prod, ignore_index=True)
