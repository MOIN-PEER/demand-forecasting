"""Turn back-test predictions into result tables, figures, an interactive explorer and an insights file."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import LABELS, MODELS  # noqa: E402
from .config import FIGURES, INTERACTIVE, RESULTS  # noqa: E402
from .inventory import critical_ratios, simulate  # noqa: E402
from .metrics import bias, coverage, mae, pinball, rmse, summarise, wape  # noqa: E402

COL = {"naive": "#9AA5B1", "moving_avg": "#F0A830", "holt_winters": "#8E7CC3", "lgbm": "#0F766E",
       "oracle": "#C8453B"}
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left"})


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate(bt: dict) -> dict:
    data, p = bt["data"], bt["predictions"].copy()
    p = p.merge(data["expected_demand"].rename(columns={"expected_units": "oracle"}), on=["date", "store_id", "sku"],
                how="left")
    p = p.merge(data["products"][["sku", "product"]], on="sku")
    ev = p[(p["stockout"] == 0) & (p["store_closed"] == 0)].copy()     # true demand is only known on these days
    allm = MODELS + ["oracle"]
    overall = pd.DataFrame([{"model": LABELS.get(m, "Best possible (true mean)"), "key": m,
                             "wape_pct": round(100 * wape(ev["units"], ev[m]), 1),
                             "bias_pct": round(100 * bias(ev["units"], ev[m]), 1),
                             "mae": round(mae(ev["units"], ev[m]), 2), "rmse": round(rmse(ev["units"], ev[m]), 2)}
                            for m in allm])
    by_fold = summarise(ev, allm, ["fold"])
    by_fold["origin"] = by_fold["fold"].map(p.groupby("fold")["origin"].first().dt.date.astype(str))
    by_cat = summarise(ev, allm, ["category"]).sort_values("wape_lgbm")
    ev["week"] = (ev["horizon"] - 1) // 7 + 1
    by_week = summarise(ev, allm, ["week"])
    by_promo = summarise(ev, allm, ["on_promo"])
    by_product = summarise(ev, allm, ["product"]).sort_values("wape_lgbm")

    # Hierarchy: errors partly cancel when forecasts are added up
    levels = {"Store × product × day": ["store_id", "sku", "date"], "Product × day (chain)": ["sku", "date"],
              "Store × day": ["store_id", "date"], "Chain × day": ["date"], "Store × product × week": ["store_id", "sku", "fold", "week"]}
    hier = []
    for name, keys in levels.items():
        g = ev.groupby(keys, observed=True)[["units"] + allm].sum()
        hier.append({"level": name, **{m: round(100 * wape(g["units"], g[m]), 1) for m in allm}})
    hier = pd.DataFrame(hier)

    cov = {"q10_q90_coverage_pct": round(100 * coverage(ev["units"], ev["q10"], ev["q90"]), 1),
           "share_below": {q: round(100 * float((ev["units"] <= ev[f"q{q}"]).mean()), 1) for q in (10, 50, 75, 90)},
           "pinball": {q: round(pinball(ev["units"], ev[f"q{q}"], q / 100), 3) for q in (10, 50, 75, 90)}}

    imp = (bt["importance"].groupby("feature")["gain"].mean().sort_values(ascending=False) / 1)
    imp = (100 * imp / imp.sum()).round(2).rename("gain_pct").reset_index()

    truth = data["known_truth"]
    cf = bt["cf_promo"].merge(truth, on="sku").merge(data["products"][["sku", "product", "category"]], on="sku")
    cf["true_uplift"] = (1 - cf["discount"]) ** cf["promo_elasticity"]
    cf["model_uplift"] = cf["pred_promo"] / cf["pred_no_promo"].clip(lower=0.05)
    promo_rec = cf.groupby(["sku", "product", "category"])[["true_uplift", "model_uplift"]].mean().round(3).reset_index()
    t = bt["cf_temp"].merge(truth, on="sku").merge(data["products"][["sku", "product"]], on="sku")
    t = t[t["temp_effect_per_degree"].abs() > 0.015]
    t["true"] = np.exp(5 * t["temp_effect_per_degree"])
    t["model"] = t["pred_plus5"] / t["pred"].clip(lower=0.05)
    temp_rec = t.groupby(["sku", "product"])[["true", "model"]].mean().round(3).reset_index()

    inv, inv_prod = simulate(p, data["products"])
    return {"p": p, "ev": ev, "overall": overall, "by_fold": by_fold, "by_cat": by_cat, "by_week": by_week,
            "by_promo": by_promo, "by_product": by_product, "hier": hier, "coverage": cov, "importance": imp,
            "promo_rec": promo_rec, "temp_rec": temp_rec, "inventory": inv, "inventory_by_product": inv_prod,
            "critical_ratios": critical_ratios(data["products"])}


def figures(r: dict, data: dict) -> None:
    s = data["sales"].merge(data["products"][["sku", "category", "regular_price"]], on="sku")
    s["revenue"] = s["units"] * s["price"]
    # 01 history
    daily = s.groupby("date")["units"].sum()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(daily.index, daily.values, lw=0.6, color="#5B6B7B")
    ax.plot(daily.rolling(28, center=True).mean(), lw=2, color=COL["lgbm"], label="28-day average")
    for y in (2023, 2024, 2025):
        ax.annotate("Christmas", (pd.Timestamp(f"{y}-12-23"), daily.max() * 0.97), ha="right", fontsize=8, color="#C8453B")
    ax.set_title("Harbour Fresh: units sold per day, all 10 stores (synthetic data)")
    ax.set_ylabel("units"); ax.legend(frameon=False, loc="upper left")
    _save(fig, "01_daily_sales_history.png")

    # 02 drivers: day of week, category seasonality, temperature
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    op = s[s["store_closed"] == 0]
    dw = op.groupby(op["date"].dt.day_of_week)["units"].sum() / op.groupby(op["date"].dt.day_of_week)["date"].nunique()
    axs[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dw / dw.mean(), color=COL["lgbm"])
    axs[0].axhline(1, color="#333", lw=0.8, ls="--"); axs[0].set_title("Day of week (index)")
    m = op.assign(m=op["date"].dt.month).groupby(["category", "m"])["units"].sum().unstack(0)
    m = m / m.mean()
    for c, colr in zip(["Frozen", "Drinks", "Ambient", "Bakery"], ["#2F6DB5", "#36C2A1", "#C8453B", "#F0A830"]):
        axs[1].plot(m.index, m[c], marker="o", ms=3, label=c, color=colr)
    axs[1].set_xticks(range(1, 13)); axs[1].set_title("Monthly pattern by category (index)"); axs[1].legend(frameon=False, fontsize=8)
    w = s.merge(data["weather"], on=["date", "store_id"])
    ic = w[(w["sku"] == "P13") & (w["on_promo"] == 0) & (w["store_closed"] == 0)]
    ic = ic.groupby(ic["temp_c"].round()).agg(units=("units", "mean"), n=("units", "size"))
    ic = ic[ic["n"] > 30]
    so = w[(w["sku"] == "P19") & (w["on_promo"] == 0) & (w["store_closed"] == 0)]
    so = so.groupby(so["temp_c"].round())["units"].mean().reindex(ic.index)
    axs[2].plot(ic.index, ic["units"], marker="o", ms=3, color="#2F6DB5", label="Ice cream 2L")
    axs[2].plot(so.index, so.values, marker="o", ms=3, color="#C8453B", label="Tomato soup")
    axs[2].set_xlabel("temperature (°C)"); axs[2].set_ylabel("units / store / day")
    axs[2].set_title("Weather effect"); axs[2].legend(frameon=False, fontsize=8)
    _save(fig, "02_demand_drivers.png")

    ov = r["overall"]
    # 03 accuracy by model
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    keys = MODELS + ["oracle"]
    vals = [ov.set_index("key").loc[k, "wape_pct"] for k in keys]
    bars = axs[0].bar([LABELS.get(k, "Best possible").replace(" (", "\n(").replace("4-week ", "4-week\n").replace("Seasonal ", "Seasonal\n").replace("Best ", "Best\n") for k in keys], vals, color=[COL[k] for k in keys])
    axs[0].bar_label(bars, fmt="%.1f%%", fontsize=9)
    axs[0].set_ylabel("WAPE (lower is better)"); axs[0].set_title("Forecast error, 6 back-test folds")
    axs[0].tick_params(axis="x", labelsize=8.5)
    bf = r["by_fold"]
    for k in keys:
        axs[1].plot(bf["origin"], bf[f"wape_{k}"], marker="o", color=COL[k], label=LABELS.get(k, "Best possible"),
                    ls="--" if k == "oracle" else "-")
    axs[1].set_ylabel("WAPE %"); axs[1].set_title("By fold (forecast origin)"); axs[1].tick_params(axis="x", labelsize=8, rotation=20)
    axs[1].legend(frameon=False, fontsize=8, ncol=2); axs[1].set_ylim(0, None)
    _save(fig, "03_accuracy_by_model.png")

    # 04 example forecasts in the Christmas fold
    p = r["p"]
    last = p[p["fold"] == p["fold"].max()]
    ex = [("S01", "P01", "White sliced bread, Belfast City Centre"), ("S05", "P27", "Mince pies, Derry~Londonderry"),
          ("S03", "P16", "Beef burgers, Lisburn")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (st, sk, title) in zip(axs, ex):
        h = s[(s["store_id"] == st) & (s["sku"] == sk) & (s["date"] >= last["origin"].iloc[0] - pd.Timedelta(days=35))
              & (s["date"] < last["origin"].iloc[0])]
        f = last[(last["store_id"] == st) & (last["sku"] == sk)]
        ax.plot(h["date"], h["units"], color="#5B6B7B", lw=1.2, label="history")
        ax.fill_between(f["date"], f["q10"], f["q90"], color=COL["lgbm"], alpha=0.18, label="LightGBM 10–90% range")
        ax.plot(f["date"], f["units"], color="#222", lw=1.4, marker="o", ms=2.5, label="actual")
        ax.plot(f["date"], f["lgbm"], color=COL["lgbm"], lw=2, label="LightGBM")
        ax.plot(f["date"], f["naive"], color=COL["naive"], lw=1.2, ls="--", label="Seasonal naive")
        ax.axvline(last["origin"].iloc[0], color="#999", lw=0.8)
        ax.set_title(title, fontsize=10.5); ax.tick_params(axis="x", labelsize=7.5, rotation=30)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    _save(fig, "04_example_forecasts_december.png")

    # 05 by category
    bc = r["by_cat"]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(bc)); wdt = 0.2
    for j, k in enumerate(MODELS):
        ax.bar(x + (j - 1.5) * wdt, bc[f"wape_{k}"], wdt, color=COL[k], label=LABELS[k])
    ax.scatter(x, bc["wape_oracle"], marker="_", s=400, color=COL["oracle"], zorder=3, label="Best possible")
    ax.set_xticks(x); ax.set_xticklabels(bc["category"], fontsize=9); ax.set_ylabel("WAPE %")
    ax.set_title("Error by category"); ax.legend(frameon=False, fontsize=8, ncol=5)
    _save(fig, "05_accuracy_by_category.png")

    # 06 horizon and hierarchy
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    bw = r["by_week"]
    for k in keys:
        axs[0].plot(bw["week"], bw[f"wape_{k}"], marker="o", color=COL[k], label=LABELS.get(k, "Best possible"),
                    ls="--" if k == "oracle" else "-")
    axs[0].set_xticks([1, 2, 3, 4]); axs[0].set_xlabel("weeks ahead"); axs[0].set_ylabel("WAPE %")
    axs[0].set_title("Error by forecast horizon"); axs[0].set_ylim(0, None); axs[0].legend(frameon=False, fontsize=8, ncol=2)
    hz = r["hier"]
    y = np.arange(len(hz))
    axs[1].barh(y + 0.2, hz["naive"], 0.4, color=COL["naive"], label="Seasonal naive")
    axs[1].barh(y - 0.2, hz["lgbm"], 0.4, color=COL["lgbm"], label="LightGBM")
    for i, v in enumerate(hz["lgbm"]):
        axs[1].text(v + 0.5, i - 0.2, f"{v}%", va="center", fontsize=8)
    axs[1].set_yticks(y); axs[1].set_yticklabels(hz["level"], fontsize=8.5); axs[1].invert_yaxis()
    axs[1].set_xlabel("WAPE %"); axs[1].set_title("Error at different levels of aggregation"); axs[1].legend(frameon=False, fontsize=8)
    _save(fig, "06_horizon_and_hierarchy.png")

    # 07 feature importance
    imp = r["importance"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["gain_pct"], color=COL["lgbm"])
    ax.set_xlabel("share of total gain (%)"); ax.set_title("What the LightGBM model relies on (top 15)")
    _save(fig, "07_feature_importance.png")

    # 08 recovered effects
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    pr = r["promo_rec"]
    axs[0].scatter(pr["true_uplift"], pr["model_uplift"], color=COL["lgbm"])
    lim = [1, max(pr[["true_uplift", "model_uplift"]].max()) + 0.15]
    axs[0].plot(lim, lim, ls="--", color="#888"); axs[0].set_xlim(lim); axs[0].set_ylim(lim)
    axs[0].set_xlabel("true uplift built into the data"); axs[0].set_ylabel("uplift the model predicts")
    axs[0].set_title("Promotion uplift by product")
    tr = r["temp_rec"]
    yy = np.arange(len(tr))
    axs[1].barh(yy + 0.2, tr["true"] - 1, 0.4, color="#9AA5B1", label="true")
    axs[1].barh(yy - 0.2, tr["model"] - 1, 0.4, color=COL["lgbm"], label="model")
    axs[1].set_yticks(yy); axs[1].set_yticklabels(tr["product"], fontsize=8.5); axs[1].axvline(0, color="#333", lw=0.8)
    axs[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axs[1].set_title("Effect of a day 5 °C warmer"); axs[1].legend(frameon=False, fontsize=8)
    _save(fig, "08_recovered_effects.png")

    # 09 inventory
    inv = r["inventory"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    names = ["Last week's\nsales", "4-week\naverage", "LightGBM\naverage", "LightGBM +\nservice level"]
    xx = np.arange(len(inv))
    axs[0].bar(xx - 0.2, inv["waste_cost_gbp"] / 1000, 0.4, color="#C8453B", label="waste cost")
    axs[0].bar(xx + 0.2, inv["lost_margin_gbp"] / 1000, 0.4, color="#F0A830", label="lost margin (stockouts)")
    axs[0].set_xticks(xx); axs[0].set_xticklabels(names, fontsize=8.5); axs[0].set_ylabel("£ thousand")
    axs[0].set_title("Cost of ordering from each forecast"); axs[0].legend(frameon=False, fontsize=8)
    axs[1].bar(xx, inv["fill_rate_pct"], color=[COL["naive"], COL["moving_avg"], COL["lgbm"], "#134E4A"])
    for i, v in enumerate(inv["fill_rate_pct"]):
        axs[1].text(i, v + 0.4, f"{v}%", ha="center", fontsize=9)
    axs[1].set_ylim(70, 95); axs[1].set_xticks(xx); axs[1].set_xticklabels(names, fontsize=8.5)
    axs[1].set_ylabel("fill rate (% of demand met)"); axs[1].set_title("Customers who found the product")
    _save(fig, "09_inventory_policies.png")

    # 10 interval calibration
    cov = r["coverage"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    qs = [10, 50, 75, 90]
    ax.plot([0, 100], [0, 100], ls="--", color="#888", label="perfect calibration")
    ax.plot(qs, [cov["share_below"][q] for q in qs], marker="o", color=COL["lgbm"], label="LightGBM quantiles")
    for q in qs:
        ax.annotate(f"{cov['share_below'][q]}%", (q, cov["share_below"][q]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("quantile level (%)"); ax.set_ylabel("actual share of demand below it (%)")
    ax.set_title("Are the forecast ranges honest?"); ax.legend(frameon=False, fontsize=8)
    _save(fig, "10_interval_calibration.png")


def interactive(r: dict, data: dict) -> None:
    import plotly.graph_objects as go
    p, s = r["p"], data["sales"]
    prods = data["products"].set_index("sku")["product"]
    series = [(st, sk) for st in ("S01", "S05") for sk in prods.index if sk not in ("P28",)]
    fig = go.Figure()
    traces_per = 0
    for i, (st, sk) in enumerate(series):
        h = s[(s["store_id"] == st) & (s["sku"] == sk) & (s["date"] >= "2025-06-01")]
        f = p[(p["store_id"] == st) & (p["sku"] == sk)].sort_values("date")
        vis = i == 0
        tr = [go.Scatter(x=h["date"], y=h["units"], name="actual", line=dict(color="#222", width=1.2), visible=vis),
              go.Scatter(x=f["date"], y=f["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip", visible=vis),
              go.Scatter(x=f["date"], y=f["q10"], fill="tonexty", fillcolor="rgba(15,118,110,0.18)", line=dict(width=0),
                         name="LightGBM 10–90%", visible=vis),
              go.Scatter(x=f["date"], y=f["lgbm"].round(1), name="LightGBM", line=dict(color=COL["lgbm"], width=2), visible=vis),
              go.Scatter(x=f["date"], y=f["holt_winters"].round(1), name="Holt-Winters", line=dict(color=COL["holt_winters"], width=1.2, dash="dot"), visible=vis),
              go.Scatter(x=f["date"], y=f["naive"], name="Seasonal naive", line=dict(color=COL["naive"], width=1.2, dash="dash"), visible=vis)]
        traces_per = len(tr)
        for t in tr:
            fig.add_trace(t)
    buttons = []
    for i, (st, sk) in enumerate(series):
        v = [False] * (len(series) * traces_per)
        v[i * traces_per:(i + 1) * traces_per] = [True] * traces_per
        name = f"{'Belfast City Centre' if st == 'S01' else 'Derry~Londonderry'} · {prods[sk]}"
        buttons.append(dict(label=name, method="update", args=[{"visible": v}, {"title.text": name}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, y=1.16, xanchor="left", showactive=True)],
                      title=dict(text=buttons[0]["label"], x=0.5), template="plotly_white", height=560,
                      yaxis_title="units per day", legend=dict(orientation="h", y=-0.15),
                      margin=dict(t=110, l=60, r=20, b=60))
    INTERACTIVE.mkdir(parents=True, exist_ok=True)
    fig.write_html(INTERACTIVE / "forecast_explorer.html", include_plotlyjs="cdn")


def write_results(r: dict, bt: dict) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ("overall", "by_fold", "by_cat", "by_week", "by_promo", "by_product", "hier", "importance",
                 "promo_rec", "temp_rec", "inventory", "inventory_by_product"):
        r[name].to_csv(RESULTS / f"{name}.csv", index=False)
    r["critical_ratios"][["sku", "product", "regular_price", "unit_cost", "critical_ratio"]].to_csv(
        RESULTS / "critical_ratios.csv", index=False)
    bt["timings"].to_csv(RESULTS / "timings.csv", index=False)
    cols = ["date", "store_id", "sku", "fold", "horizon", "units", "stockout", "store_closed", "naive", "moving_avg",
            "holt_winters", "lgbm", "q10", "q50", "q75", "q90"]
    r["p"][cols].round({c: 2 for c in cols[8:]}).to_csv(RESULTS / "backtest_predictions.csv.gz", index=False, compression="gzip")
    ov = r["overall"].set_index("key")
    inv = r["inventory"].set_index("policy")
    d = bt["data"]
    s = d["sales"]
    summary = {
        "data": {"stores": int(s["store_id"].nunique()), "products": int(s["sku"].nunique()), "rows": len(s),
                 "days": int(s["date"].nunique()), "units": int(s["units"].sum()),
                 "revenue_gbp": round(float((s["units"] * s["price"]).sum()), 0),
                 "stockout_rate_pct": round(100 * s["stockout"].mean(), 2),
                 "promo_rate_pct": round(100 * s["on_promo"].mean(), 1)},
        "backtest": {"folds": int(r["p"]["fold"].nunique()), "horizon_days": 28,
                     "evaluated_rows": len(r["ev"])},
        "wape_pct": ov["wape_pct"].to_dict(), "bias_pct": ov["bias_pct"].to_dict(),
        "lgbm_improvement_vs_naive_pct": round(100 * (1 - ov.loc["lgbm", "wape_pct"] / ov.loc["naive", "wape_pct"]), 1),
        "lgbm_improvement_vs_best_baseline_pct": round(100 * (1 - ov.loc["lgbm", "wape_pct"] / ov.loc[["naive", "moving_avg", "holt_winters"], "wape_pct"].min()), 1),
        "hierarchy_lgbm_wape_pct": r["hier"].set_index("level")["lgbm"].to_dict(),
        "intervals": r["coverage"],
        "promo_wape_pct": r["by_promo"].set_index("on_promo")[["wape_lgbm", "wape_naive"]].to_dict(),
        "inventory": {k: inv.loc[k, ["fill_rate_pct", "waste_pct_of_ordered", "waste_cost_gbp", "lost_margin_gbp",
                                     "net_profit_gbp", "mismatch_cost_gbp", "mismatch_vs_naive_pct"]].to_dict() for k in inv.index},
        "top_features": r["importance"].head(8).set_index("feature")["gain_pct"].to_dict(),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def insights(summary: dict, r: dict) -> None:
    w, inv = summary["wape_pct"], summary["inventory"]
    naive, lg, sl = (inv["Last week's sales (naive)"], inv["LightGBM average forecast"],
                     inv["LightGBM + target service level"])
    bc = r["by_cat"]
    lines = [
        "# Insights (generated from the back-test)", "",
        "*All data is synthetic (fictional chain \"Harbour Fresh\").*", "",
        f"1. **LightGBM cut forecast error from {w['naive']}% (seasonal naive) to {w['lgbm']}% WAPE**, "
        f"{summary['lgbm_improvement_vs_best_baseline_pct']}% lower than the best baseline (Holt-Winters, {w['holt_winters']}%).",
        f"2. **It is close to the limit.** A forecast that knew the true average demand would still score {w['oracle']}%, "
        "because daily sales of single products are noisy. The remaining gap is small.",
        f"3. **Errors cancel when added up:** {summary['hierarchy_lgbm_wape_pct']['Store × day']}% at store-day level and "
        f"{summary['hierarchy_lgbm_wape_pct']['Chain × day']}% for the whole chain per day. Plan staff on store totals; plan stock per product.",
        f"4. **Hardest categories:** {bc.iloc[-1]['category']} ({bc.iloc[-1]['wape_lgbm']:.1f}%) and {bc.iloc[-2]['category']} "
        f"({bc.iloc[-2]['wape_lgbm']:.1f}%), which are small, weather-driven or short-season lines. Easiest: {bc.iloc[0]['category']} ({bc.iloc[0]['wape_lgbm']:.1f}%).",
        f"5. **Stock decisions:** ordering short-life products from LightGBM instead of last week's sales raised fill rate from "
        f"{naive['fill_rate_pct']}% to {lg['fill_rate_pct']}%, cut waste from {naive['waste_pct_of_ordered']}% to "
        f"{lg['waste_pct_of_ordered']}% of stock ordered and added £{lg['net_profit_gbp'] - naive['net_profit_gbp']:,.0f} "
        "of profit over the 24 test weeks (11 products, 10 stores).",
        f"6. **Service-level ordering** (a forecast quantile at each product's critical ratio) lifted fill rate further to "
        f"{sl['fill_rate_pct']}% and gave the lowest total mismatch cost (waste + lost margin + goodwill), at the price of "
        "slightly more waste.",
        f"7. **Ranges are a little too narrow:** {summary['intervals']['q10_q90_coverage_pct']}% of actual demand fell inside "
        "the 10–90% range (target 80%). Calibrating the quantiles is the next improvement.",
    ]
    (RESULTS / "insights.md").write_text("\n".join(lines) + "\n")
