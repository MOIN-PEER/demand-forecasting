"""Command line:

    python -m demand_forecasting data       # generate the synthetic dataset -> data/raw
    python -m demand_forecasting backtest   # 6-fold back-test, results, figures, explorer -> outputs/
    python -m demand_forecasting all        # both (about 5 minutes on a laptop)

Set DF_FOLDS=2 for a quicker run (last two folds only).
"""

import sys
import time


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd not in ("data", "backtest", "all"):
        sys.exit(__doc__)
    t0 = time.perf_counter()
    if cmd in ("data", "all"):
        from .generate_data import write
        d = write()
        print(f"Data: {len(d['sales']):,} store-product-days, {d['sales']['store_id'].nunique()} stores, "
              f"{d['sales']['sku'].nunique()} products -> data/raw")
    if cmd in ("backtest", "all"):
        from . import reporting as R
        from .backtest import run
        bt = run()
        r = R.evaluate(bt)
        R.figures(r, bt["data"])
        R.interactive(r, bt["data"])
        s = R.write_results(r, bt)
        R.insights(s, r)
        w = s["wape_pct"]
        print(f"\nWAPE: naive {w['naive']}% | 4-week avg {w['moving_avg']}% | Holt-Winters {w['holt_winters']}% | "
              f"LightGBM {w['lgbm']}% | best possible {w['oracle']}%")
        print("Results: outputs/results  ·  Figures: outputs/figures  ·  Explorer: outputs/interactive/forecast_explorer.html")
    print(f"Done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
