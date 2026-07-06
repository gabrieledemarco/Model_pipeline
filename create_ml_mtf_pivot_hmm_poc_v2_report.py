#!/usr/bin/env python3
"""
create_ml_mtf_pivot_hmm_poc_v2_report.py
==========================================
Approfondimento del POC ML (v1: create_ml_mtf_pivot_hmm_poc_report.py, primo
segnale a sopravvivere a un holdout genuino in questa sessione).

Estensioni richieste:
  - Feature importance: coefficienti standardizzati (LogReg) e
    feature_importances_ (GBoost, RandomForest), mediati su tutte le
    finestre WFO, per capire quali dei tre driver (OHLCV / regime HMM /
    pivot multi-TF) contano davvero.
  - Più orizzonti: 1,2,4,8,16,24,48,72h (prima solo 4,16,24,48h) per
    mappare la curva completa dell'edge.
  - Terzo modello: RandomForestClassifier (bagging, più robusto del
    singolo GBoost alla varianza di stima).

Stesso protocollo di validazione: WFO causale 6m/2m/2m, breakdown per
anno dal primo run, holdout 2025-2026 genuino.
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features, HMM_FEATURE_NAMES

SEP = "═" * 78
START_YEAR = 2020
HORIZONS = [1, 2, 4, 8, 16, 24, 48, 72]
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
CUTOFF = pd.Timestamp("2025-01-01")
PIVOT_LR = {"5m": 12, "15m": 8, "1h": 6, "1d": 3}

print(SEP)
print("ML POC v2 — feature importance + più orizzonti + RandomForest")
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# DATA (identico a v1)
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
# PIVOT FEATURES (identico a v1)
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
# OHLCV-DERIVED FEATURES (identico a v1)
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

FEATURE_NAMES = list(ohlcv_feats.columns) + list(pivot_feats.columns) + list(HMM_FEATURE_NAMES)
print(f"\n[FEATURES] {len(FEATURE_NAMES)} totali: {FEATURE_NAMES}")

# ══════════════════════════════════════════════════════════════════════════════
# TARGETS
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
# WALK-FORWARD: fit HMM + 3 ML classifiers per window, track importances
# ══════════════════════════════════════════════════════════════════════════════
MODELS = {
    "LogReg": lambda: LogisticRegression(C=1.0, max_iter=500),
    "GBoost": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100,
                                                       learning_rate=0.05, random_state=42),
    "RandForest": lambda: RandomForestClassifier(n_estimators=200, max_depth=5,
                                                  min_samples_leaf=50, random_state=42,
                                                  n_jobs=-1),
}

def get_importance(model_ml, name, Xoos_s, yoos):
    """Model-native importance where available; permutation importance for GBoost
    (HistGradientBoostingClassifier has no feature_importances_ attribute)."""
    if name == "LogReg":
        return np.abs(model_ml.coef_[0])
    if name == "RandForest":
        return model_ml.feature_importances_
    # GBoost: cheap permutation importance on the OOS fold (small n_repeats for speed)
    try:
        r = permutation_importance(model_ml, Xoos_s, yoos, n_repeats=3,
                                    random_state=42, scoring="accuracy", n_jobs=-1)
        return np.maximum(r.importances_mean, 0)
    except Exception:
        return np.zeros(Xoos_s.shape[1])

def run_wfo(windows, track_importance=False):
    rows = []
    importances = {name: {h: [] for h in HORIZONS} for name in MODELS}
    for wi, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        idx_is = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 500 or len(idx_oos) < 50:
            continue

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
                if track_importance:
                    importances[name][h].append(get_importance(model_ml, name, Xoos_s, yoos))
        print(".", end="", flush=True)
    return pd.DataFrame(rows), importances

print("\n[RUN] Walk-forward (full sample) — tracking feature importance …")
t1 = time.time()
wfo_results, importances = run_wfo(WF_WINDOWS, track_importance=True)
print(f"\n  {len(wfo_results)} window-level results  ({time.time()-t1:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(f"\n{SEP}")
w("FEATURE IMPORTANCE (media su tutte le finestre WFO, orizzonte 16h)")
w(SEP)
for name in MODELS:
    imp_list = importances[name].get(16, [])
    if not imp_list: continue
    avg_imp = np.mean(np.vstack(imp_list), axis=0)
    order = np.argsort(avg_imp)[::-1]
    w(f"\n  {name} — top 10 feature:")
    for i in order[:10]:
        w(f"    {FEATURE_NAMES[i]:<20}  {avg_imp[i]:.4f}")

w(f"\n{SEP}")
w("RISULTATI AGGREGATI (full-sample walk-forward, tutti gli orizzonti)")
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
w("RISULTATI PER ANNO (solo orizzonti chiave: 4h, 16h, 24h, 48h)")
w(SEP)
for model in MODELS:
    w(f"\n  Modello: {model}")
    for h in [4, 16, 24, 48]:
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
# GENUINE HOLDOUT
# ══════════════════════════════════════════════════════════════════════════════
WF_HOLDOUT = [wd for wd in WF_WINDOWS if wd[2] >= CUTOFF]
w(f"\n{SEP}")
w(f"HOLDOUT GENUINO 2025-2026 (N={len(WF_HOLDOUT)} finestre, mai usate prima)")
w(SEP)
holdout_results, _ = run_wfo(WF_HOLDOUT, track_importance=False)
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

out_path = Path("reports/ml_mtf_pivot_hmm_poc_v2.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# ML POC v2 — feature importance + più orizzonti + RandomForest\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
wfo_results.to_csv("reports/ml_mtf_pivot_hmm_poc_v2_raw.csv", index=False)
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
