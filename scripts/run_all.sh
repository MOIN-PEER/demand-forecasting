#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m demand_forecasting all
echo "Open outputs/interactive/forecast_explorer.html in your browser"
