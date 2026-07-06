#!/usr/bin/env python3
"""
create_ml_trading_strategy_backtest_report.py
================================================
Traduce il segnale ML (POC v2: OHLCV + regime HMM + pivot H1/Daily,
RandomForest — il modello più robusto su holdout genuino) in una vera
strategia di trading con entry/exit/fee, e la sottopone allo stesso
protocollo di validazione rigoroso di tutta la sessione.

Semplificazione rispetto al POC v2: feature M5/M15 rimosse (non
comparivano nella top-10 di nessun modello nella feature importance) —
si mantengono OHLCV + HMM (3 stati) + pivot H1 + pivot Daily.

Regola di trading (v2, dopo diagnosi):
  - RandomForest stima P(return_h > 0) ad ogni bar H1 (causale, walk-forward)
  - Long se P > soglia_long, Short se P < soglia_short, altrimenti flat
    (filtra i trade a bassa confidenza)
  - USCITA A TEMPO FISSO all'orizzonte h (fedele a quanto realmente
    validato: il modello predice il SEGNO del return a h ore, non una
    soglia di magnitudo). Uno stop di sicurezza largo (4×ATR) protegge
    solo da eventi estremi, non è pensato per attivarsi spesso.
  - PRIMA VERSIONE testata (TP=2×ATR / SL=1×ATR, stesso max_hold=h) dava
    ret fortemente negativo nonostante win rate >33% teoricamente
    sufficiente per un R:R 2:1. Diagnosi: entro una finestra così breve,
    lo stop a 1×ATR viene toccato dal puro rumore MOLTO più spesso del
    target a 2×ATR (verificato anche su segnali simulati senza alcuno
    skill direzionale: sl-hit 3× più frequente di tp-hit) — un'asimmetria
    strutturale nella probabilità di first-passage che nessuno skill
    direzionale modesto (52-54%) può superare. Da qui la scelta di
    un'uscita a tempo fisso, che è anche l'unica realmente coerente con
    il target di classificazione validato.
  - Cooldown per evitare posizioni sovrapposte sullo stesso segnale

Sizing: rischio fisso in dollari (non % del capitale corrente) per
isolare l'edge grezzo dagli effetti di compounding — stessa lezione
imparata nel sistema MTF trend-scan.

Validazione: WFO causale 6m/2m/2m, breakdown per anno dal primo run,
holdout 2025-2026 genuino, Monte Carlo i.i.d. + block-bootstrap.
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features, HMM_FEATURE_NAMES
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0004
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
PIVOT_LR = {"5m": 12, "15m": 8, "1h": 6, "1d": 3}

HORIZONS_TO_TEST = [2, 4, 8]
THRESHOLD = 0.55          # confidence filter: |P-0.5| > 0.05
SAFETY_SL_ATR_MULT = 4.0  # wide crash-protection stop, not meant to bind often
COOLDOWN_HOURS = 4

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("ML Trading Strategy Backtest — RandomForest signal -> real trade rule")
w(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# DATA — H1 base + M5/M15/Daily (full feature set as actually validated in the
# POC — importance ranking showed M5/M15 outside the top-10, but that doesn't
# mean they're safe to drop: removing them changes the model and needs its own
# validation, which we have NOT done. Test the signal that was actually shown
# to survive holdout, not a modified one.)
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[DATA] Loading H1 (base) + M5/M15/Daily …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H = len(df1h)
df_5m = fetch_binance_vision_klines("5m", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
df_15m = fetch_binance_vision_klines("15m", start_year=START_YEAR, start_month=1,
                                      workers=6, verbose=False)
df_1d = fetch_binance_vision_klines("1d", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
print(f"  H1: {N1H:,}   M5: {len(df_5m):,}   M15: {len(df_15m):,}   Daily: {len(df_1d):,}  "
      f"(loaded in {time.time()-t0:.0f}s)")

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
LOGCL = np.log(CL)
ATR = np.where(df1h["atr_14"].shift(1).values > 0, df1h["atr_14"].shift(1).values, 1.0)

# ══════════════════════════════════════════════════════════════════════════════
# PIVOT FEATURES (H1, Daily only)
# ══════════════════════════════════════════════════════════════════════════════
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
    dist_high = (aligned["target_high"].values - close_on_h1) / atr_on_h1
    dist_low = (close_on_h1 - aligned["target_low"].values) / atr_on_h1
    dist_high = np.where(np.isnan(dist_high), 10.0, np.clip(dist_high, 0, 10))
    dist_low = np.where(np.isnan(dist_low), 10.0, np.clip(dist_low, 0, 10))
    return pd.DataFrame({
        f"{tf_label}_trend": aligned["trend_state"].values,
        f"{tf_label}_dist_high": dist_high,
        f"{tf_label}_dist_low": dist_low,
    }, index=IDX1H)

pivot_feats = pd.concat([
    pivot_features_on_h1(df_5m, PIVOT_LR["5m"], "m5"),
    pivot_features_on_h1(df_15m, PIVOT_LR["15m"], "m15"),
    pivot_features_on_h1(df1h, PIVOT_LR["1h"], "h1"),
    pivot_features_on_h1(df_1d, PIVOT_LR["1d"], "d1"),
], axis=1)

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
print(f"[FEATURES] {len(FEATURE_NAMES)}: {FEATURE_NAMES}")

# ══════════════════════════════════════════════════════════════════════════════
# TARGETS
# ══════════════════════════════════════════════════════════════════════════════
targets = {}
for h in HORIZONS_TO_TEST:
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
WF_HOLDOUT = [wd for wd in WF_WINDOWS if wd[2] >= CUTOFF]
print(f"[WFO] {len(WF_WINDOWS)} windows  |  {len(WF_HOLDOUT)} holdout")

# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD: fit HMM + RandomForest per window -> proba array over full OOS span
# ══════════════════════════════════════════════════════════════════════════════
def compute_oos_probas(windows, horizon):
    """Returns proba array (len N1H, NaN outside any OOS window) causal walk-forward."""
    proba_full = np.full(N1H, np.nan)
    y_all = targets[horizon]
    for tr_s, tr_e, oo_s, oo_e in windows:
        idx_is = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 500 or len(idx_oos) < 50:
            continue
        valid_is = ~np.isnan(y_all[idx_is])
        if valid_is.sum() < 200:
            continue

        model, sorted_idx = fit_hmm(df1h.iloc[idx_is], n_states=3, random_state=42)
        hmm_is = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_is])
        hmm_oos = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_oos])

        X_is = np.column_stack([ohlcv_feats.iloc[idx_is].values,
                                 pivot_feats.iloc[idx_is].values, hmm_is.values])
        X_oos = np.column_stack([ohlcv_feats.iloc[idx_oos].values,
                                  pivot_feats.iloc[idx_oos].values, hmm_oos.values])
        X_is = np.nan_to_num(X_is, nan=0.0, posinf=10.0, neginf=-10.0)
        X_oos = np.nan_to_num(X_oos, nan=0.0, posinf=10.0, neginf=-10.0)

        Xis_valid = X_is[valid_is]
        yis_valid = y_all[idx_is][valid_is]
        scaler = StandardScaler().fit(Xis_valid)
        Xis_s = scaler.transform(Xis_valid)
        Xoos_s = scaler.transform(X_oos)

        clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=50,
                                      random_state=42, n_jobs=-1)
        clf.fit(Xis_s, yis_valid)
        proba_oos = clf.predict_proba(Xoos_s)[:, 1]
        proba_full[idx_oos] = proba_oos
        print(".", end="", flush=True)
    print()
    return proba_full

# ══════════════════════════════════════════════════════════════════════════════
# TRADING RULE -> EVENTS -> BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def make_events(proba, horizon, idx_arr, cooldown_bars):
    evs = []
    last_s = -cooldown_bars
    for k in idx_arr:
        if k >= N1H or np.isnan(proba[k]) or ATR[k] <= 0: continue
        if k - last_s < cooldown_bars: continue
        p = proba[k]
        if p > THRESHOLD:
            d = 1
        elif p < (1 - THRESHOLD):
            d = -1
        else:
            continue
        ep = CL[k]
        sl = ep - d * SAFETY_SL_ATR_MULT * ATR[k]   # wide, crash-protection only
        evs.append(dict(i=k, d=d, ep=ep, sl=sl, max_hold=horizon))
        last_s = k
    return evs

def run_bt(events):
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_time=0, n_sl=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_time = 0; n_sl = 0
    for ev in events:
        i, d, ep, sl, max_hold = ev["i"], ev["d"], ev["ep"], ev["sl"], ev["max_hold"]
        out = "time"
        for k in range(1, max_hold + 1):
            if i + k >= N1H: break
            hk, lk = HI[i + k], LO[i + k]
            if d == 1 and lk <= sl: out = "sl"; break
            if d == -1 and hk >= sl: out = "sl"; break
        if out == "sl":
            exit_price = sl
            n_sl += 1
        else:
            j = min(i + max_hold, N1H - 1)
            exit_price = CL[j]
            n_time += 1
        stop_dist = abs(ep - sl)
        if stop_dist <= 0: continue
        risk = INIT_CAP * RISK_PCT
        units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
        notional = units * ep
        pnl_dollar = units * (exit_price - ep) * d - FEE * 2 * notional
        cap += pnl_dollar
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl_dollar > 0)
        net_pnls.append(pnl_dollar)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_time=n_time, n_sl=n_sl)

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    df_mc = pd.DataFrame({"net_pnl": pnls})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=20):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    df_mc = pd.DataFrame({"net_pnl": pnls})
    mc = run_monte_carlo_block(df_mc, INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

# ══════════════════════════════════════════════════════════════════════════════
# RUN for each horizon
# ══════════════════════════════════════════════════════════════════════════════
for horizon in HORIZONS_TO_TEST:
    w(f"\n{SEP}")
    w(f"HORIZON = {horizon}h   (threshold={THRESHOLD}, exit=time-based@{horizon}h, "
      f"safety-SL={SAFETY_SL_ATR_MULT}xATR, cooldown={COOLDOWN_HOURS}h)")
    w(SEP)
    cooldown_bars = COOLDOWN_HOURS

    print(f"\n[RUN] Computing OOS probas (full-sample walk-forward) …")
    t1 = time.time()
    proba_full = compute_oos_probas(WF_WINDOWS, horizon)
    print(f"  done in {time.time()-t1:.0f}s")

    all_evs = []
    year_evs: dict[int, list] = {}
    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_oos) < 50: continue
        evs = make_events(proba_full, horizon, idx_oos, cooldown_bars)
        all_evs.extend(evs)
        year_evs.setdefault(oo_s.year, []).extend(evs)

    res = run_bt(all_evs)
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL-SAMPLE: n={res['n']}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  "
      f"mdd={res['mdd']:.1f}%  (exits: time={res['n_time']} safety-sl={res['n_sl']})")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    for yr in sorted(year_evs):
        evs = year_evs[yr]
        if len(evs) < 5: continue
        yres = run_bt(evs)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    # Genuine holdout: each WFO window already fits its own model causally
    # (IS strictly precedes OOS, no cross-window state) — the "full-sample"
    # proba_full array's values for 2025-2026 OOS bars are therefore already
    # exactly what a from-scratch "fit only on pre-2025" pass would produce.
    # Just re-select those indices instead of recomputing (same result).
    holdout_evs = []
    for tr_s, tr_e, oo_s, oo_e in WF_HOLDOUT:
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_oos) < 50: continue
        holdout_evs.extend(make_events(proba_full, horizon, idx_oos, cooldown_bars))
    hres = run_bt(holdout_evs)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  "
      f"ret={hres['ret']:+.1f}%  mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/ml_trading_strategy_backtest.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# ML Trading Strategy Backtest\n\n```\n" + "\n".join(report_lines) +
                     "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
