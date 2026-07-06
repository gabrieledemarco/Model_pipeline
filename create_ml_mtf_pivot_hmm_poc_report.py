#!/usr/bin/env python3
"""
create_ml_mtf_pivot_hmm_poc_report.py
=======================================
POC ML: driver = OHLCV (return/ATR%/volume), regime HMM causale 3-stati
(src/strategy/hmm_regime.py), pivot causali su 4 timeframe — M5, M15, H1,
Daily (src/strategy/mtf_swing.py) — allineati sulla base H1.

Target: genuinamente forward-shifted (direzione del return realizzato a
h ore), MAI lo stesso principio di leakage del Pine Script SatohK.

Protocollo di validazione (lo stesso di tutti i POC di questa sessione):
  - Walk-forward causale 6m IS / 2m OOS / 2m step
  - Breakdown per anno FIN DA SUBITO (non aggiunto dopo)
  - Holdout 2025-2026 genuino: fit solo su finestre pre-2025, eval su
    2025-2026 mai toccato

Modelli: LogisticRegression (semplice, difficile da overfittare) e
HistGradientBoostingClassifier (cattura interazioni non lineari).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features, HMM_FEATURE_NAMES

SEP = "═" * 78
START_YEAR = 2020
HORIZONS = [4, 16, 24, 48]
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
CUTOFF = pd.Timestamp("2025-01-01")

# Pivot lookback (left=right) per timeframe — moderate defaults, not tuned
PIVOT_LR = {"5m": 12, "15m": 8, "1h": 6, "1d": 3}

print(SEP)
print("ML POC — driver: OHLCV + regime HMM (3-stati) + pivot M5/M15/H1/D")
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# DATA — base timeframe H1, + M5/M15/Daily for pivot features
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[DATA] Loading H1 (base) …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H = len(df1h)
print(f"  H1: {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

print("[DATA] Loading M5, M15, Daily for pivot features …")
df_5m = fetch_binance_vision_klines("5m", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
df_15m = fetch_binance_vision_klines("15m", start_year=START_YEAR, start_month=1,
                                      workers=6, verbose=False)
df_1d = fetch_binance_vision_klines("1d", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
print(f"  M5: {len(df_5m):,}   M15: {len(df_15m):,}   Daily: {len(df_1d):,}   "
      f"(loaded in {time.time()-t0:.0f}s)")

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
LOGCL = np.log(CL)

# ══════════════════════════════════════════════════════════════════════════════
# PIVOT FEATURES — causal, per timeframe, aligned onto H1
# ══════════════════════════════════════════════════════════════════════════════
print("\n[PIVOTS] Computing causal pivot structure per timeframe …")

def pivot_features_on_h1(df_tf, left_right, tf_label):
    close = df_tf["close"].values.astype(float)
    high = df_tf["high"].values.astype(float)
    low = df_tf["low"].values.astype(float)
    state = causal_trend_state(close, high, low, left_right, left_right)
    atr_tf = (df_tf["high"] - df_tf["low"]).rolling(14).mean().bfill().values
    state["atr"] = atr_tf
    state["close"] = close
    aligned = align_htf_to_ltf(df_tf.index, state, IDX1H)

    atr_on_h1 = np.where(aligned["atr"].values > 0, aligned["atr"].values, 1.0)
    close_on_h1 = aligned["close"].values

    dist_to_high = (aligned["target_high"].values - close_on_h1) / atr_on_h1
    dist_to_low = (close_on_h1 - aligned["target_low"].values) / atr_on_h1
    # sentinel for "no valid unbroken pivot in that direction"
    dist_to_high = np.where(np.isnan(dist_to_high), 10.0, np.clip(dist_to_high, 0, 10))
    dist_to_low = np.where(np.isnan(dist_to_low), 10.0, np.clip(dist_to_low, 0, 10))

    return pd.DataFrame({
        f"{tf_label}_trend": aligned["trend_state"].values,
        f"{tf_label}_dist_high": dist_to_high,
        f"{tf_label}_dist_low": dist_to_low,
    }, index=IDX1H)

pivot_feats = pd.concat([
    pivot_features_on_h1(df_5m, PIVOT_LR["5m"], "m5"),
    pivot_features_on_h1(df_15m, PIVOT_LR["15m"], "m15"),
    pivot_features_on_h1(df1h, PIVOT_LR["1h"], "h1"),
    pivot_features_on_h1(df_1d, PIVOT_LR["1d"], "d1"),
], axis=1)
print(f"  Pivot feature columns: {list(pivot_feats.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# OHLCV-DERIVED FEATURES (H1 native, causal)
# ══════════════════════════════════════════════════════════════════════════════
close_s = pd.Series(CL, index=IDX1H)
vol_s = pd.Series(VOL, index=IDX1H)
ohlcv_feats = pd.DataFrame({
    "ret_1h": np.log(close_s / close_s.shift(1)),
    "ret_4h": np.log(close_s / close_s.shift(4)),
    "ret_24h": np.log(close_s / close_s.shift(24)),
    "atr_pct": df1h["atr_pct"].values,
    "vol_ratio": (vol_s / vol_s.rolling(20).mean()).values,
    "range_pct": ((HI - LO) / CL),
}, index=IDX1H)

# ══════════════════════════════════════════════════════════════════════════════
# TARGETS — genuinely forward-shifted
# ══════════════════════════════════════════════════════════════════════════════
targets = {}
for h in HORIZONS:
    fwd_ret = np.concatenate([LOGCL[h:] - LOGCL[:-h], np.full(h, np.nan)])
    t = (fwd_ret > 0).astype(float)
    t[np.isnan(fwd_ret)] = np.nan
    targets[h] = t

# ══════════════════════════════════════════════════════════════════════════════
# WFO WINDOWS
# ══════════════════════════════════════════════════════════════════════════════
def wf_dates(idx):
    t0_ = idx[0]; windows = []
    while True:
        tr_s = t0_; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > idx[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0_ += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
print(f"\n[WFO] {len(WF_WINDOWS)} windows")

# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD: fit HMM (causal) + ML classifier per window, evaluate OOS
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS_STATIC = list(ohlcv_feats.columns) + list(pivot_feats.columns)
MODELS = {
    "LogReg": lambda: LogisticRegression(C=1.0, max_iter=500),
    "GBoost": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100,
                                                       learning_rate=0.05, random_state=42),
}

def run_wfo(windows):
    rows = []
    for wi, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        idx_is = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 500 or len(idx_oos) < 50:
            continue

        # HMM: fit causal on IS, predict on IS (for training features) and OOS
        model, sorted_idx = fit_hmm(df1h.iloc[idx_is], n_states=3, random_state=42)
        hmm_is = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_is])
        hmm_oos = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_oos])

        X_is_static = np.column_stack([ohlcv_feats.iloc[idx_is].values,
                                        pivot_feats.iloc[idx_is].values])
        X_oos_static = np.column_stack([ohlcv_feats.iloc[idx_oos].values,
                                         pivot_feats.iloc[idx_oos].values])
        X_is = np.column_stack([X_is_static, hmm_is.values])
        X_oos = np.column_stack([X_oos_static, hmm_oos.values])
        X_is = np.nan_to_num(X_is, nan=0.0, posinf=10.0, neginf=-10.0)
        X_oos = np.nan_to_num(X_oos, nan=0.0, posinf=10.0, neginf=-10.0)

        for h in HORIZONS:
            y_all = targets[h]
            valid_is = ~np.isnan(y_all[idx_is])
            valid_oos = ~np.isnan(y_all[idx_oos])
            if valid_is.sum() < 200 or valid_oos.sum() < 50:
                continue
            yis = y_all[idx_is][valid_is]
            yoos = y_all[idx_oos][valid_oos]
            Xis_h = X_is[valid_is]
            Xoos_h = X_oos[valid_oos]

            scaler = StandardScaler().fit(Xis_h)
            Xis_s = scaler.transform(Xis_h)
            Xoos_s = scaler.transform(Xoos_h)

            for name, make_model in MODELS.items():
                model_ml = make_model()
                model_ml.fit(Xis_s, yis)
                proba = model_ml.predict_proba(Xoos_s)[:, 1]
                pred = (proba > 0.5).astype(float)
                hit = float(np.mean(pred == yoos))
                corr = float(np.corrcoef(proba - 0.5, yoos - 0.5)[0, 1]) if len(yoos) > 2 else np.nan
                rows.append(dict(horizon=h, model=name, oo_s=oo_s, year=oo_s.year,
                                  n=len(yoos), hit_rate=hit, corr=corr))
        print(".", end="", flush=True)
    return pd.DataFrame(rows)

print("\n[RUN] Walk-forward (full sample) …")
t1 = time.time()
wfo_results = run_wfo(WF_WINDOWS)
print(f"\n  {len(wfo_results)} window-level results  ({time.time()-t1:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(f"\n{SEP}")
w("RISULTATI AGGREGATI (full-sample walk-forward)")
w(SEP)
for model in MODELS:
    w(f"\n  Modello: {model}")
    w(f"    {'Horizon':>8}  {'n tot':>8}  {'Hit rate':>9}  {'Corr':>8}")
    for h in HORIZONS:
        sub = wfo_results[(wfo_results.model == model) & (wfo_results.horizon == h)]
        n_tot = sub["n"].sum()
        if n_tot == 0: continue
        hit_w = float((sub["hit_rate"] * sub["n"]).sum() / n_tot)
        corr_w = float((sub["corr"] * sub["n"]).sum() / n_tot)
        w(f"    {h:>6}h  {n_tot:>8}  {hit_w:>8.1%}  {corr_w:>+7.3f}")

w(f"\n{SEP}")
w("RISULTATI PER ANNO (verifica di stabilità)")
w(SEP)
for model in MODELS:
    w(f"\n  Modello: {model}")
    for h in HORIZONS:
        w(f"\n    Horizon {h}h:")
        w(f"      {'Year':>6}  {'n':>7}  {'Hit rate':>9}  {'Corr':>8}")
        sub_h = wfo_results[(wfo_results.model == model) & (wfo_results.horizon == h)]
        for yr in sorted(sub_h["year"].unique()):
            sub = sub_h[sub_h.year == yr]
            n_tot = sub["n"].sum()
            if n_tot < 50: continue
            hit_w = float((sub["hit_rate"] * sub["n"]).sum() / n_tot)
            corr_w = float((sub["corr"] * sub["n"]).sum() / n_tot)
            w(f"      {yr:>6}  {n_tot:>7}  {hit_w:>8.1%}  {corr_w:>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# GENUINE HOLDOUT: fit only on pre-2025 windows, evaluate on 2025-2026 only
# ══════════════════════════════════════════════════════════════════════════════
WF_HOLDOUT = [wd for wd in WF_WINDOWS if wd[2] >= CUTOFF]
w(f"\n{SEP}")
w(f"HOLDOUT GENUINO 2025-2026 (N={len(WF_HOLDOUT)} finestre, mai usate prima)")
w(SEP)
holdout_results = run_wfo(WF_HOLDOUT)
for model in MODELS:
    w(f"\n  Modello: {model}")
    w(f"    {'Horizon':>8}  {'n':>7}  {'Hit rate':>9}  {'Corr':>8}")
    for h in HORIZONS:
        sub = holdout_results[(holdout_results.model == model) & (holdout_results.horizon == h)]
        n_tot = sub["n"].sum()
        if n_tot == 0: continue
        hit_w = float((sub["hit_rate"] * sub["n"]).sum() / n_tot)
        corr_w = float((sub["corr"] * sub["n"]).sum() / n_tot)
        w(f"    {h:>6}h  {n_tot:>7}  {hit_w:>8.1%}  {corr_w:>+7.3f}")

w(f"\n{SEP}")
w("INTERPRETAZIONE")
w(SEP)
w("  Hit rate 50% = nessun potere predittivo. Edge genuino richiede hit rate")
w("  >52-53% CONSISTENTE su tutti gli anni e sull'holdout 2025-2026.")

out_path = Path("reports/ml_mtf_pivot_hmm_poc.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# ML POC — OHLCV + HMM regime + pivot M5/M15/H1/D\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
wfo_results.to_csv("reports/ml_mtf_pivot_hmm_poc_raw.csv", index=False)
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
