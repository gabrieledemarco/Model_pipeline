#!/usr/bin/env python3
"""
Fit (or refresh) the ML RandomForest 8h strategy's production model —
docs/ML_RF_8H_STRATEGY_SPEC.md section 9.4 ("Refresh del modello").

Fetches ~7.5 months of live 1H/5M/15M OHLCV from Binance FAPI production
(the same data source used by every other live strategy) plus the existing
~10-month 1D fetch, builds the 23 causal features (src/live/ml_rf_features.py)
over the full fetched range for correct rolling/pivot warm-up, then fits
HMM(3-state) + StandardScaler + RandomForestClassifier on the trailing
WF_TRAIN_M=6 months, and persists all three (+ metadata) to
logs/strategies/{strategy_id}/model/ via joblib.

Usage:
    .venv/bin/python3 scripts/train_ml_rf_8h.py

Re-run every ~2 months (WF_STEP_M) per the spec's refresh cadence.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.live.data_live import (
    fetch_tf_data, fetch_15m_bars, fetch_1h_bars_extended, fetch_5m_bars_extended,
)
from src.strategy.indicators import add_indicators
from src.strategy.hmm_regime import fit_hmm
from src.live.ml_rf_features import build_features, FEATURE_NAMES
from src.live.strategy_id import slugify_strategy_id

SCENARIO = "ML RandomForest 8h"
EXCHANGE = "bybit"
STRATEGY_ID = slugify_strategy_id(SCENARIO, EXCHANGE)
MODEL_DIR = ROOT / "logs" / "strategies" / STRATEGY_ID / "model"

WF_TRAIN_M = 6          # months actually used to fit the production model
HORIZON = 8             # hours — validated best of 2h/4h/8h
FETCH_1H_BARS = 5500    # ≈ 229 days ≈ 7.5 months (6mo train + ~6wk warm-up)
FETCH_15M_BARS = FETCH_1H_BARS * 4
FETCH_5M_BARS = FETCH_1H_BARS * 12

RF_PARAMS = dict(n_estimators=200, max_depth=5, min_samples_leaf=50,
                  random_state=42, n_jobs=-1)


def fit_and_persist_model() -> None:
    """Fetch fresh data, fit HMM+scaler+RandomForest on the trailing
    WF_TRAIN_M-month window, and persist all artifacts to MODEL_DIR.

    Callable both from this script's CLI entry point and from
    src.live.ml_rf_live's auto-refresh-on-staleness check (mirrors
    src/live/hmm_regime.py's pattern for S07's HMM gate).
    """
    t0 = time.time()
    print(f"[1/5] Fetching ~{FETCH_1H_BARS} 1H bars, {FETCH_15M_BARS} 15M, "
          f"{FETCH_5M_BARS} 5M, 1D …")
    df_1h = add_indicators(fetch_1h_bars_extended(limit=FETCH_1H_BARS))
    df_15m = fetch_15m_bars(limit=FETCH_15M_BARS)
    df_5m = fetch_5m_bars_extended(limit=FETCH_5M_BARS)
    df_1d = fetch_tf_data()["1D"]
    print(f"      1H: {len(df_1h):,}  15M: {len(df_15m):,}  5M: {len(df_5m):,}  "
          f"1D: {len(df_1d):,}   ({time.time()-t0:.0f}s)")

    train_start = df_1h.index[-1] - pd.DateOffset(months=WF_TRAIN_M)
    train_mask = df_1h.index >= train_start
    print(f"[2/5] Training window: {df_1h.index[train_mask][0]} → {df_1h.index[-1]} "
          f"({train_mask.sum():,} bars)")

    print("[3/5] Fitting HMM(3-state) on the training window …")
    hmm_model, hmm_sorted_idx = fit_hmm(df_1h.loc[train_mask], n_states=3, random_state=42)

    print("[4/5] Building 23 causal features over the full fetched range …")
    feats_full = build_features(df_1h, df_5m, df_15m, df_1d, hmm_model, hmm_sorted_idx)
    feats_train = feats_full.loc[train_mask].copy()

    close_train = df_1h.loc[train_mask, "close"].values.astype(float)
    log_close = np.log(close_train)
    fwd_ret = np.concatenate([log_close[HORIZON:] - log_close[:-HORIZON],
                               np.full(HORIZON, np.nan)])
    y = (fwd_ret > 0).astype(float)
    y[np.isnan(fwd_ret)] = np.nan

    X = feats_train[FEATURE_NAMES].values
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    valid = ~np.isnan(y)
    X_valid, y_valid = X[valid], y[valid]
    print(f"      {len(X_valid):,} labeled training rows "
          f"(dropped {(~valid).sum()} with no forward target yet)")

    print("[5/5] Fitting StandardScaler + RandomForestClassifier …")
    scaler = StandardScaler().fit(X_valid)
    X_scaled = scaler.transform(X_valid)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_scaled, y_valid)

    train_acc = rf.score(X_scaled, y_valid)
    class_balance = float(y_valid.mean())
    print(f"      in-sample accuracy={train_acc:.3f}  class_balance(P(up))={class_balance:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(hmm_model, MODEL_DIR / "hmm_model.joblib")
    joblib.dump(hmm_sorted_idx, MODEL_DIR / "hmm_sorted_idx.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(rf, MODEL_DIR / "rf.joblib")

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_window_start": str(df_1h.index[train_mask][0]),
        "train_window_end": str(df_1h.index[-1]),
        "n_train_rows": int(len(X_valid)),
        "horizon_hours": HORIZON,
        "feature_names": FEATURE_NAMES,
        "rf_params": RF_PARAMS,
        "train_accuracy": train_acc,
        "class_balance": class_balance,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model artifacts to {MODEL_DIR}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    fit_and_persist_model()
