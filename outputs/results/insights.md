# Insights (generated from the back-test)

*All data is synthetic (fictional chain "Harbour Fresh").*

1. **LightGBM cut forecast error from 45.3% (seasonal naive) to 27.4% WAPE**, 20.8% lower than the best baseline (Holt-Winters, 34.6%).
2. **It is close to the limit.** A forecast that knew the true average demand would still score 26.7%, because daily sales of single products are noisy. The remaining gap is small.
3. **Errors cancel when added up:** 6.4% at store-day level and 2.4% for the whole chain per day. Plan staff on store totals; plan stock per product.
4. **Hardest categories:** Seasonal (41.8%) and Frozen (37.6%), which are small, weather-driven or short-season lines. Easiest: Dairy (22.8%).
5. **Stock decisions:** ordering short-life products from LightGBM instead of last week's sales raised fill rate from 81.2% to 87.7%, cut waste from 20.0% to 11.6% of stock ordered and added £156,160 of profit over the 24 test weeks (11 products, 10 stores).
6. **Service-level ordering** (a forecast quantile at each product's critical ratio) lifted fill rate further to 90.8% and gave the lowest total mismatch cost (waste + lost margin + goodwill), at the price of slightly more waste.
7. **Ranges are a little too narrow:** 76.0% of actual demand fell inside the 10–90% range (target 80%). Calibrating the quantiles is the next improvement.
