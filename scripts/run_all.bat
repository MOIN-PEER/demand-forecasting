@echo off
REM Generate the data, run the back-test and open the interactive explorer
cd /d "%~dp0\.."
python -m demand_forecasting all
start outputs\interactive\forecast_explorer.html
