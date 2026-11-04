"""Paths and settings. Override paths with environment variables if needed."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.getenv("DF_DATA_DIR", ROOT / "data" / "raw"))
SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
INTERACTIVE = ROOT / "outputs" / "interactive"

SEED = 42
START, END = "2023-01-01", "2025-12-31"

# Forecasting setup: every forecast is made on an "origin" date for the next 28 days.
HORIZON = 28
# Six back-test folds; the last one covers the run-up to Christmas 2025.
FOLD_ORIGINS = ["2025-07-14", "2025-08-11", "2025-09-08", "2025-10-06", "2025-11-03", "2025-12-01"]
QUANTILES = [0.1, 0.5, 0.75, 0.9]

LGB_PARAMS = dict(n_estimators=int(os.getenv("DF_TREES", 500)), learning_rate=0.05, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, verbose=-1, n_jobs=-1)
