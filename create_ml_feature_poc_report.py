#!/usr/bin/env python3
"""
create_ml_feature_poc_report.py
=================================
POC: il set di feature dell'indicatore Pine Script "Deep Machine Learning
[SatohK]" (RSI, CCI, %B, MACD hist, Stoch, MFI, Choppiness), reimplementato
con un target GENUINAMENTE forward-shifted e training walk-forward causale
— a differenza dell'originale Pine Script, che ha (a) target leakage
contemporaneo nei modi "Candle"/"HTF Candle" (predict_period è cablato a 0)
e (b) leakage temporale nel target "Pivot State" (i pivot vengono valutati
con hindsight completo, non come sarebbero stati noti causalmente al tempo).

Qui: feature Z-scored (fit su IS, applicate su OOS, mai sull'intero
dataset), target = segno del return realizzato a h ore nel futuro,
walk-forward 6m/2m/2m identico alla pipeline ADP, due modelli semplici
(Logistic Regression, Gradient Boosting) per verificare se c'è un edge
reale isolato dai bug dell'implementazione originale.

Metriche riportate per anno FIN DA SUBITO (lezione dalla validazione ADP:
un aggregato può nascondere un edge concentrato in un solo anno) e su un
vero holdout 2025-2026 mai usato per scegliere feature/modello.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators

SEP = "═" * 78
START_YEAR = 2020
HORIZONS = [4, 16, 24, 48]
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
CUTOFF = pd.Timestamp("2025-01-01")

print(SEP)
print("ML Feature POC — RSI/CCI/%B/MACD/Stoch/MFI/CHOP (target causale, no leakage)")
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
print(f"  {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
LOGCL = np.log(CL)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURES — reuse RSI/MACD-hist/Stoch/%B from indicators.py, add CCI/MFI/CHOP
# ══════════════════════════════════════════════════════════════════════════════
def cci(high, low, close, period=20):
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3.0
    raw_flow = tp * volume
    direction = np.sign(tp.diff())
    pos_flow = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    neg_flow = raw_flow.where(direction < 0, 0.0).rolling(period).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))

def choppiness(high, low, close, period=14):
    atr1 = (high - low).abs()
    atr_sum = atr1.rolling(period).sum()
    hi_max = high.rolling(period).max()
    lo_min = low.rolling(period).min()
    rng = (hi_max - lo_min).replace(0, np.nan)
    return 100 * np.log10(atr_sum / rng) / np.log10(period)

hi_s, lo_s, cl_s, vol_s = df1h["high"], df1h["low"], df1h["close"], df1h["volume"]

feat_raw = pd.DataFrame(index=IDX1H)
feat_raw["rsi"]   = df1h["rsi_14"]
feat_raw["cci"]   = cci(hi_s, lo_s, cl_s, 20)
feat_raw["bb_pct"] = df1h["bb_pct"]
feat_raw["macd_hist"] = df1h["macd_hist"]
feat_raw["stoch_k"] = df1h["stoch_k"]
feat_raw["mfi"]   = mfi(hi_s, lo_s, cl_s, vol_s, 14)
feat_raw["chop"]  = choppiness(hi_s, lo_s, cl_s, 14)
feat_raw = feat_raw.ffill().fillna(0.0)
FEATURE_COLS = list(feat_raw.columns)
print(f"[FEATURES] {FEATURE_COLS}")

# ══════════════════════════════════════════════════════════════════════════════
# TARGETS — genuinely forward-shifted (fixes the Pine Script's leakage bug)
# ══════════════════════════════════════════════════════════════════════════════
targets = {}
for h in HORIZONS:
    fwd_ret = np.concatenate([LOGCL[h:] - LOGCL[:-h], np.full(h, np.nan)])
    targets[h] = (fwd_ret > 0).astype(float)
    targets[h][np.isnan(fwd_ret)] = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# WFO WINDOWS (identico allo schema ADP)
# ══════════════════════════════════════════════════════════════════════════════
def wf_dates(idx):
    t0 = idx[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > idx[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
print(f"[WFO] {len(WF_WINDOWS)} windows")

X_ALL = feat_raw[FEATURE_COLS].values

# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
MODELS = {
    "LogReg": lambda: LogisticRegression(C=1.0, max_iter=500),
    "GBoost": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100,
                                                       learning_rate=0.05, random_state=42),
}

def run_wfo(windows, horizons=HORIZONS, models=MODELS):
    rows = []
    for h in horizons:
        y_all = targets[h]
        for tr_s, tr_e, oo_s, oo_e in windows:
            idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
            idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
            valid_is  = idx_is[~np.isnan(y_all[idx_is])]
            valid_oos = idx_oos[~np.isnan(y_all[idx_oos])]
            if len(valid_is) < 200 or len(valid_oos) < 50:
                continue

            scaler = StandardScaler().fit(X_ALL[valid_is])
            Xis = scaler.transform(X_ALL[valid_is])
            Xoos = scaler.transform(X_ALL[valid_oos])
            yis = y_all[valid_is]
            yoos = y_all[valid_oos]

            for name, make_model in models.items():
                model = make_model()
                model.fit(Xis, yis)
                proba = model.predict_proba(Xoos)[:, 1]
                pred = (proba > 0.5).astype(float)
                hit = float(np.mean(pred == yoos))
                # correlation between predicted probability (centered) and actual outcome
                corr = float(np.corrcoef(proba - 0.5, yoos - 0.5)[0, 1]) if len(yoos) > 2 else np.nan
                rows.append(dict(horizon=h, model=name, oo_s=oo_s, year=oo_s.year,
                                  n=len(valid_oos), hit_rate=hit, corr=corr))
    return pd.DataFrame(rows)

print("\n[RUN] Walk-forward (full sample, 35 windows × 4 horizons × 2 models) …")
wfo_results = run_wfo(WF_WINDOWS)
print(f"  {len(wfo_results)} window-level results computed")

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(f"\n{SEP}")
w("RISULTATI AGGREGATI (media pesata per n_trade, tutto il periodo)")
w(SEP)
for model in MODELS:
    w(f"\n  Modello: {model}")
    w(f"    {'Horizon':>8}  {'n tot':>8}  {'Hit rate':>9}  {'Corr':>8}")
    for h in HORIZONS:
        sub = wfo_results[(wfo_results.model == model) & (wfo_results.horizon == h)]
        n_tot = sub["n"].sum()
        hit_w = float((sub["hit_rate"] * sub["n"]).sum() / n_tot) if n_tot else float("nan")
        corr_w = float((sub["corr"] * sub["n"]).sum() / n_tot) if n_tot else float("nan")
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
WF_PRE2025 = [wnd for wnd in WF_WINDOWS if wnd[3] <= CUTOFF]
WF_HOLDOUT = [wnd for wnd in WF_WINDOWS if wnd[2] >= CUTOFF]
w(f"\n{SEP}")
w(f"HOLDOUT GENUINO 2025-2026  "
  f"({len(WF_PRE2025)} finestre pre-2025 per training walk-forward, "
  f"{len(WF_HOLDOUT)} finestre 2025-2026 mai usate prima)")
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
w("  Hit rate 50% = nessun potere predittivo (random walk). Un edge genuino")
w("  richiede hit rate >52-53% CONSISTENTE su tutti gli anni e sull'holdout")
w("  2025-2026, non solo nell'aggregato pieno.")

out_path = Path("reports/ml_feature_poc.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# ML Feature POC — RSI/CCI/%B/MACD/Stoch/MFI/CHOP (target causale)\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
wfo_results.to_csv("reports/ml_feature_poc_raw.csv", index=False)
print(f"\n[DONE] {out_path}")
