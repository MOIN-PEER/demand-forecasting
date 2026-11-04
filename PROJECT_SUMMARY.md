# Project summary: Retail Demand Forecasting

**Author:** Mohamed Moin Peer  |  **Type:** Time-series forecasting and inventory optimisation  |  **Stack:** Python, pandas, LightGBM, statsmodels, matplotlib, Plotly, pytest

## Problem
A grocery chain orders stock for every product in every store. If it orders too much, short-life food is wasted; if it orders too little, shelves are empty and customers leave. Orders are often based on last week's sales. The question is how much better forecasts reduce waste and stockouts, and what that is worth.

## What I built
* **Synthetic dataset.** "Harbour Fresh", a fictional chain of 10 Northern Ireland stores with 30 products: 324,550 store-product-days over 2023–2025. It includes prices, promotions, weather, NI holidays and store closures, seasonal lines and stockouts. The effects are known, so the models can be checked against the truth.
* **Leak-free features** for a direct 1–28-day forecast: lags of at least 28 days, rolling means, a same-weekday average, the promotion plan, the calendar and a weather forecast. Stockout days are removed from the history; a test proves no future data leaks in.
* **Four forecasting methods:** seasonal naive, a 4-week average, Holt-Winters per series, and one global **LightGBM** model with quantile models for ranges.
* **Rolling-origin back-test.** Six 28-day folds, each trained only on the past, with the error measured against the best possible score.
* **Inventory simulation.** Daily newsvendor orders for 11 short-life products, comparing four order policies on fill rate, waste and profit.
* Ten figures, an interactive forecast explorer, 16 tests, a presentation and a final report.

## Results (6 folds, 49,296 store-product-days)
* **WAPE: LightGBM 27.4%** against 45.3% (naive), 37.2% (4-week average) and 34.6% (Holt-Winters). That is 39% less error than naive and 21% less than the best baseline. The best possible score is 26.7%.
* **Promotion days:** the baselines under-forecast by 41–44%; LightGBM's bias is −3%.
* **Store-day error is 6.4% and chain-day error 2.4%**, because errors cancel when aggregated.
* **Stock:** ordering from LightGBM instead of last week's sales raised the fill rate from 81% to 88%, cut waste from 20% to 12% of stock and added **£156k of profit** over 24 weeks. Service-level ordering reached **91% availability** with the lowest total cost.
* **Honest limits:**
  * The data is synthetic.
  * The 10–90% ranges cover 76% of demand rather than 80%.
  * Weather sensitivity is underestimated by about a third.

## Skills demonstrated
Time-series forecasting · LightGBM (Tweedie and quantile) · exponential smoothing · back-testing · forecast evaluation and calibration · inventory optimisation · business impact analysis · testing · visualisation.

## How to run
`pip install -r requirements.txt && pip install -e . && python -m demand_forecasting all && pytest`
