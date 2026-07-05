#!/usr/bin/env python3
"""
create_momentum_intraday_report.py
====================================
New intraday BTC/USDT strategies targeting higher returns.

Root cause of low MR return: IC=0.019, WR=30%, pure oscillation edge.
BTC is a trending asset — missing the directional move is the main gap.

Groups in this scan:
  TF  — EMA crossover trend-following (let winners run, wide TP)
  PB  — Pullback-in-trend (MR entry timing WITH trend direction)
  DN  — Donchian channel breakout
  MC  — MACD signal line crossover
  ADP — Adaptive: MR in ranging + TF in trending (HMM state)
  CMB — Combo: multi-signal confirmation

TP_GRID for TF/breakout: [2, 3, 5, 7, 10] × SL_GRID: [0.5, 1.0, 1.5, 2.0]
MAX_HOLD extended to 48H for trend-following groups.
"""
from __future__ import annotations

import base64, io, sys, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmmlearn import hmm as hmmlib

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import run_monte_carlo

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP   = 100_000.0
RISK_PCT   = 0.01
FEE        = 0.0004
FEE_RT_PCT = FEE * 2 * 100
MAX_LEV    = 5.0
IC_HORIZON = 16        # 16H forward return — balances MR and TF signals
START_YEAR = 2020
N_SIMS     = 5_000
COOLDOWN   = 4
MAX_HOLD   = 48        # 48H for trend-following (2 days max)
WARMUP     = 200

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

# Wider grid for trend-following: let winners run
TP_GRID_TF = [2.0, 3.0, 5.0, 7.0, 10.0]
SL_GRID_TF = [0.5, 1.0, 1.5, 2.0]
# Standard MR grid
TP_GRID_MR = [1.0, 2.0, 3.0, 5.0]
SL_GRID_MR = [0.25, 0.5, 0.75, 1.0]

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("Intraday BTC — Momentum & Trend-Following Strategies")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h  = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
print(f"  1H: {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

CL   = df1h["close"].values.astype(float)
HI   = df1h["high"].values.astype(float)
LO   = df1h["low"].values.astype(float)
VOL  = df1h["volume"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0,
                df1h["atr_14"].shift(1).values, 1.0)

CL_s  = pd.Series(CL, index=IDX1H)
VOL_s = pd.Series(VOL, index=IDX1H)

IDX_DT   = pd.to_datetime(IDX1H)
HOUR     = IDX_DT.hour.values
DATE_STR = IDX_DT.strftime('%Y-%m-%d')

ACTIVE_MASK = (HOUR >= 7) & (HOUR < 22)
NY_MASK     = (HOUR >= 13) & (HOUR < 22)

# ── EMA helper ────────────────────────────────────────────────────────────────
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE INDICATORS (all shift(1) → causal)
# ══════════════════════════════════════════════════════════════════════════════

# Z-score 24H (base MR signal, reused in ADP)
_zm = CL_s.rolling(24).mean()
_zs = CL_s.rolling(24).std().replace(0, np.nan)
Z24  = ((CL_s - _zm) / _zs).fillna(0).shift(1).values

# EMA series (shifted → causal)
EMA = {}
for span in [5, 8, 12, 20, 24, 48, 100, 200]:
    EMA[span] = ema(CL_s, span).shift(1).values

# EMA slopes (sign of change over last N bars)
def ema_slope(span: int, lookback: int = 4) -> np.ndarray:
    e = pd.Series(EMA[span], index=IDX1H)
    return np.sign(e - e.shift(lookback)).fillna(0).values

# ADX-like: directional momentum (simplified, causal)
# Using rolling N-bar high/low range vs ATR as trend strength proxy
def adx_proxy(period: int = 14) -> np.ndarray:
    """Proxy for trend strength: rolling range / (period * ATR1). Shift(1)."""
    _hi_roll = pd.Series(HI, index=IDX1H).rolling(period).max()
    _lo_roll = pd.Series(LO, index=IDX1H).rolling(period).min()
    rng = (_hi_roll - _lo_roll).fillna(0)
    avg_atr = pd.Series(ATR1, index=IDX1H).rolling(period).mean().replace(0, 1)
    return (rng / (avg_atr * period)).shift(1).fillna(0).values

ADX_PROXY14 = adx_proxy(14)
ADX_PROXY20 = adx_proxy(20)

# MACD: EMA(12) - EMA(26), signal = EMA(9) of MACD, all shift(1)
_macd_line   = (ema(CL_s, 12) - ema(CL_s, 26)).shift(1)
_macd_signal = ema(_macd_line, 9)               # already shift(1) because input is
MACD_LINE    = _macd_line.values
MACD_SIG     = _macd_signal.values
MACD_HIST    = (MACD_LINE - MACD_SIG)           # positive = bullish

# Donchian channels: N-bar rolling high/low (shift(1) → causal)
def donchian(n: int):
    hi = pd.Series(HI, index=IDX1H).rolling(n).max().shift(1).values
    lo = pd.Series(LO, index=IDX1H).rolling(n).min().shift(1).values
    return hi, lo

DON = {}
for n in [12, 24, 48]:
    DON[n] = donchian(n)

# Volume ratio
VOL_RATIO = (VOL_s / VOL_s.rolling(24).mean().replace(0, np.nan)
             ).fillna(1.0).shift(1).values

# HMM features (causal)
LOG_RET = np.concatenate([[0.0], np.log(CL[1:] / np.where(CL[:-1] > 0, CL[:-1], 1.0))])
ATR_PCT = np.where(CL > 0, ATR1 / CL, 0.001)

# ── WFO windows ───────────────────────────────────────────────────────────────
def wf_dates(IDX):
    t0 = IDX[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > IDX[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
print(f"[WFO] {len(WF_WINDOWS)} windows\n")

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ev(i, direction, tp_f, sl_f):
    d = 1 if direction == "long" else -1
    a = ATR1[i]; ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep+d*tp_f*a, sl=ep-d*sl_f*a, a=a)

def run_bt(events, max_hold=MAX_HOLD):
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, exppnl=0.0, net_pnls=[], cap=INIT_CAP)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, max_hold + 1):
            if i+k >= N1H: break
            hk, lk = HI[i+k], LO[i+k]
            if d == 1:
                if hk >= tp: out = "tp"; break
                if lk <= sl: out = "sl"; break
            else:
                if lk <= tp: out = "tp"; break
                if hk >= sl: out = "sl"; break
        if out == "none": continue
        pnl_r     = (abs(tp-ep)/a) if out == "tp" else -(abs(sl-ep)/a)
        fee_atr   = FEE * 2 * ep / a
        pnl_r_net = pnl_r - fee_atr
        risk      = cap * RISK_PCT
        lev       = min(max(abs(tp-ep)/ep, abs(sl-ep)/ep), MAX_LEV)
        dpnl      = pnl_r_net * risk * lev
        cap += dpnl; peak = max(peak, cap)
        mdd  = min(mdd, (cap-peak)/peak)
        wins += int(out == "tp"); net_pnls.append(dpnl)
    n = len(net_pnls); wr = wins/n if n else 0.0
    ret = (cap/INIT_CAP - 1) * 100
    avg_tp = np.mean([abs(ev["tp"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    avg_sl = np.mean([abs(ev["sl"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    sln = avg_sl + FEE_RT_PCT; tpn = avg_tp - FEE_RT_PCT
    return dict(n=n, wr=wr, ret=ret, mdd=mdd*100, exppnl=wr*tpn-(1-wr)*sln,
                net_pnls=net_pnls, cap=cap)

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    arr   = np.array(pnls, dtype=float)
    df_mc = pd.DataFrame({"net_pnl": arr, "gross_pnl": arr,
                           "total_fees": np.zeros(len(arr))})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)),
                p_ruin=float(mc.get("p_ruin", 1.0)))

def ic_test(sig, horizon=IC_HORIZON):
    fwd  = np.log(np.roll(CL, -horizon) / CL)
    mask = sig != 0; mask[-horizon:] = False
    x, y = sig[mask], fwd[mask]
    if len(x) < 30: return 0.0, 1.0, 0
    r, p = st.spearmanr(x, y)
    return float(r), float(p), int(mask.sum())

def make_events(sig, idx_arr, tp_f, sl_f):
    evs = []; last_s = -COOLDOWN
    for k in idx_arr:
        if k >= N1H or sig[k] == 0 or ATR1[k] <= 0: continue
        if k - last_s < COOLDOWN: continue
        evs.append(_ev(k, "long" if sig[k] > 0 else "short", tp_f, sl_f))
        last_s = k
    return evs

def is_scan(sig, idx_is, tp_grid, sl_grid):
    best = (tp_grid[0], sl_grid[0]); best_xp = -np.inf
    for tf, sf in product(tp_grid, sl_grid):
        res = run_bt(make_events(sig, idx_is, tf, sf))
        if res["exppnl"] > best_xp: best_xp = res["exppnl"]; best = (tf, sf)
    return best

# ══════════════════════════════════════════════════════════════════════════════
# HMM HELPERS (causal, no lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def build_hmm_feats(bar_indices):
    lr  = LOG_RET[bar_indices]
    atr = ATR_PCT[bar_indices]
    return np.column_stack([lr, atr]).astype(float)

def hmm_forward_predict(model, obs):
    frame_lp = model._compute_log_likelihood(obs)
    T, K = frame_lp.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = np.log(model.startprob_ + 1e-300) + frame_lp[0]
    log_transmat = np.log(model.transmat_ + 1e-300)
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_transmat[:, j]) + frame_lp[t, j]
    return np.argmax(log_alpha, axis=1)

def fit_hmm(idx_is, n_states=2):
    X = build_hmm_feats(idx_is)
    mu = X.mean(0); std = X.std(0); std[std < 1e-8] = 1.0
    Xn = (X - mu) / std
    m = hmmlib.GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=150, tol=1e-4, random_state=42)
    try:
        m.fit(Xn); m._mu = mu; m._std = std; return m
    except Exception:
        return None

def hmm_oos_states(model, idx_oos):
    X = build_hmm_feats(idx_oos)
    Xn = (X - model._mu) / model._std
    return hmm_forward_predict(model, Xn)

def ranging_state(model):
    return int(np.argmin(model.covars_[:, 0, 0]))   # min log_return variance

def trending_state(model):
    return int(np.argmax(model.covars_[:, 0, 0]))   # max log_return variance

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(variant_id: str, name: str, sig: np.ndarray,
                 tp_grid=None, sl_grid=None, max_hold=MAX_HOLD,
                 grp: str = "") -> dict:
    if tp_grid is None: tp_grid = TP_GRID_TF
    if sl_grid is None: sl_grid = SL_GRID_TF

    n_long = int((sig > 0).sum()); n_short = int((sig < 0).sum())
    n_sig  = n_long + n_short
    print(f"  [{variant_id}] {name}")
    print(f"    Signals: {n_sig:,}  (L={n_long:,}/S={n_short:,})", end="")

    ic, p_ic, n_ic = ic_test(sig)
    ic_pass = (abs(ic) > 0) and (p_ic < 0.05)
    print(f"  IC={ic:+.4f} p={p_ic:.4f} → {'PASS ✓' if ic_pass else 'FAIL ✗'}")

    rec = dict(
        id=variant_id, name=name, grp=grp,
        n_sig=n_sig, n_long=n_long, n_short=n_short,
        ic=ic, p_ic=p_ic, n_ic=n_ic, ic_pass=ic_pass,
        wfo_run=False, oos_n=0, oos_wr=0.0, oos_ret=0.0, oos_mdd=0.0,
        mc_p_profit=0.0, mc_p_ruin=1.0, validated=False,
        oos_pnls=[], oos_equity=[INIT_CAP],
        best_tp=None, best_sl=None,
    )
    if not ic_pass: return rec

    sig_used = sig if ic > 0 else -sig
    print("    WFO …", end="", flush=True)
    rec["wfo_run"] = True; all_evs = []
    agg_tp = []; agg_sl = []
    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue
        tp_f, sl_f = is_scan(sig_used, idx_is, tp_grid, sl_grid)
        agg_tp.append(tp_f); agg_sl.append(sl_f)
        all_evs.extend(make_events(sig_used, idx_oos, tp_f, sl_f))
        print(".", end="", flush=True)
    print()

    if not all_evs:
        print("    No OOS events"); return rec

    res  = run_bt(all_evs, max_hold=max_hold)
    mc   = mc_summary(res["net_pnls"])
    valid = (res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05)

    eq = [INIT_CAP]; cap_eq = INIT_CAP
    for p in res["net_pnls"]: cap_eq += p; eq.append(cap_eq)

    print(f"    OOS: n={res['n']}  wr={res['wr']:.1%}  "
          f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    print(f"    MC:  p_profit={mc['p_profit']:.3f}  p_ruin={mc['p_ruin']:.3f}")
    print(f"    {'VALIDATED ✅' if valid else 'NOT VALIDATED ❌'}")

    rec.update(oos_n=res["n"], oos_wr=res["wr"], oos_ret=res["ret"],
               oos_mdd=res["mdd"], mc_p_profit=mc["p_profit"],
               mc_p_ruin=mc["p_ruin"], validated=valid,
               oos_pnls=res["net_pnls"], oos_equity=eq,
               best_tp=round(np.mean(agg_tp),1) if agg_tp else None,
               best_sl=round(np.mean(agg_sl),2) if agg_sl else None)
    return rec


def run_pipeline_adp(variant_id: str, name: str,
                     sig_mr: np.ndarray, sig_tf: np.ndarray,
                     n_states: int = 2) -> dict:
    """
    Adaptive pipeline: MR signal in ranging state, TF signal in trending state.
    HMM fit on IS, forward-causal prediction on OOS. Zero lookahead.
    """
    n_sig = int((sig_mr != 0).sum()) + int((sig_tf != 0).sum())
    print(f"  [{variant_id}] {name}")
    print(f"    MR signals: {int((sig_mr != 0).sum()):,}  "
          f"TF signals: {int((sig_tf != 0).sum()):,}", end="")

    # IC of combined signal (just for reference)
    combo = np.where(sig_mr != 0, sig_mr, sig_tf)
    ic, p_ic, n_ic = ic_test(combo)
    ic_pass = True   # we run ADP regardless of overall IC (each arm has its own logic)
    print(f"  combo IC={ic:+.4f} p={p_ic:.4f}")

    rec = dict(
        id=variant_id, name=name, grp="ADP",
        n_sig=n_sig, n_long=0, n_short=0,
        ic=ic, p_ic=p_ic, n_ic=n_ic, ic_pass=True,
        wfo_run=True, oos_n=0, oos_wr=0.0, oos_ret=0.0, oos_mdd=0.0,
        mc_p_profit=0.0, mc_p_ruin=1.0, validated=False,
        oos_pnls=[], oos_equity=[INIT_CAP],
    )

    print("    WFO …", end="", flush=True)
    all_evs = []

    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue

        # HMM fit on IS
        model = fit_hmm(idx_is, n_states=n_states)
        if model is None:
            # Fallback: MR only
            tp_f, sl_f = is_scan(sig_mr, idx_is, TP_GRID_MR, SL_GRID_MR)
            all_evs.extend(make_events(sig_mr, idx_oos, tp_f, sl_f))
            print("e", end="", flush=True)
            continue

        r_st = ranging_state(model)
        t_st = trending_state(model)
        oos_states = hmm_oos_states(model, idx_oos)

        # IS scan: separate for MR and TF arms
        tp_mr, sl_mr = is_scan(sig_mr, idx_is, TP_GRID_MR, SL_GRID_MR)
        tp_tf, sl_tf = is_scan(sig_tf, idx_is, TP_GRID_TF, SL_GRID_TF)

        # Build combined signal for OOS: MR when ranging, TF when trending
        sig_adp = np.zeros(N1H)
        for pos, k in enumerate(idx_oos):
            if oos_states[pos] == r_st and sig_mr[k] != 0:
                sig_adp[k] = sig_mr[k]
            elif oos_states[pos] == t_st and sig_tf[k] != 0:
                sig_adp[k] = sig_tf[k]

        # Separate events per arm (different TP/SL)
        for pos, k in enumerate(idx_oos):
            if sig_adp[k] == 0 or ATR1[k] <= 0: continue
            if oos_states[pos] == r_st:
                all_evs.append(_ev(k, "long" if sig_adp[k] > 0 else "short", tp_mr, sl_mr))
            else:
                all_evs.append(_ev(k, "long" if sig_adp[k] > 0 else "short", tp_tf, sl_tf))

        print(".", end="", flush=True)

    print()
    if not all_evs:
        print("    No OOS events"); return rec

    res  = run_bt(all_evs, max_hold=MAX_HOLD)
    mc   = mc_summary(res["net_pnls"])
    valid = (res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05)

    eq = [INIT_CAP]; cap_eq = INIT_CAP
    for p in res["net_pnls"]: cap_eq += p; eq.append(cap_eq)

    print(f"    OOS: n={res['n']}  wr={res['wr']:.1%}  "
          f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    print(f"    MC:  p_profit={mc['p_profit']:.3f}  p_ruin={mc['p_ruin']:.3f}")
    print(f"    {'VALIDATED ✅' if valid else 'NOT VALIDATED ❌'}")

    rec.update(oos_n=res["n"], oos_wr=res["wr"], oos_ret=res["ret"],
               oos_mdd=res["mdd"], mc_p_profit=mc["p_profit"],
               mc_p_ruin=mc["p_ruin"], validated=valid,
               oos_pnls=res["net_pnls"], oos_equity=eq)
    return rec

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def sig_ema_cross(fast: int, slow: int, mask=None) -> np.ndarray:
    """EMA(fast) > EMA(slow) → long; EMA(fast) < EMA(slow) → short."""
    s = np.sign(EMA[fast] - EMA[slow]).astype(float)
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def sig_ema_cross_adx(fast: int, slow: int, adx_thr: float,
                      adx_prx: np.ndarray, mask=None) -> np.ndarray:
    """EMA crossover only when ADX proxy > threshold (strong trend)."""
    cross = np.sign(EMA[fast] - EMA[slow])
    s = np.where(adx_prx > adx_thr, cross, 0.0)
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def sig_pullback_in_trend(trend_ema: int, pb_ema: int,
                           dev_thr: float, mask=None) -> np.ndarray:
    """
    Pullback-in-trend: enter in EMA(trend) direction on EMA(pb) deviation.
    Long  when price > EMA(trend)  AND  price < EMA(pb) * (1-dev_thr)
    Short when price < EMA(trend)  AND  price > EMA(pb) * (1+dev_thr)
    Entry signal: previous close used (shift already applied to EMA arrays).
    """
    prev_cl = CL_s.shift(1).values
    uptrend  = prev_cl > EMA[trend_ema]
    dntrend  = prev_cl < EMA[trend_ema]
    pullback_up = prev_cl < EMA[pb_ema] * (1 - dev_thr)   # price dipped below short EMA
    pullback_dn = prev_cl > EMA[pb_ema] * (1 + dev_thr)   # price spiked above short EMA
    s = np.where(uptrend & pullback_up, 1.0,
        np.where(dntrend & pullback_dn, -1.0, 0.0))
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def sig_donchian_break(n: int, mask=None) -> np.ndarray:
    """Price breaks above N-bar high → long; below N-bar low → short."""
    prev_cl = CL_s.shift(1).values
    dh, dl = DON[n]
    # Breakout: previous close > previous high (new high territory)
    s = np.where(prev_cl > dh, 1.0, np.where(prev_cl < dl, -1.0, 0.0))
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def sig_macd(zero_cross: bool = True, mask=None) -> np.ndarray:
    """
    MACD signal:
    - zero_cross=True:  MACD histogram crosses zero (momentum shift)
    - zero_cross=False: MACD line crosses signal line
    """
    if zero_cross:
        hist_prev = pd.Series(MACD_HIST, index=IDX1H).shift(1).values
        s = np.where((MACD_HIST > 0) & (hist_prev <= 0),  1.0,
            np.where((MACD_HIST < 0) & (hist_prev >= 0), -1.0, 0.0))
    else:
        s = np.sign(MACD_HIST).astype(float)
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def sig_supertrend_proxy(atr_mult: float, ema_span: int, mask=None) -> np.ndarray:
    """
    Supertrend-like signal: price above EMA ± ATR band → directional bias.
    Long  when price > EMA(span) + atr_mult * ATR
    Short when price < EMA(span) - atr_mult * ATR
    Using previous bar values (causal).
    """
    prev_cl  = CL_s.shift(1).values
    e = EMA[ema_span]
    atr = ATR1     # already shift(1)
    s = np.where(prev_cl > e + atr_mult * atr,  1.0,
        np.where(prev_cl < e - atr_mult * atr, -1.0, 0.0))
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def sig_zscore_mr(win: int, thr: float, mask=None) -> np.ndarray:
    zm = CL_s.rolling(win).mean()
    zs = CL_s.rolling(win).std().replace(0, np.nan)
    z = ((CL_s - zm) / zs).fillna(0).shift(1).values
    s = np.where(z < -thr, 1.0, np.where(z > thr, -1.0, 0.0))
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY GROUPS
# ══════════════════════════════════════════════════════════════════════════════
ALL_RESULTS: list[dict] = []
GRP: dict[str, list[dict]] = {}
GRP_META: dict[str, tuple[str,str]] = {}

# ── Group TF: EMA Crossover Trend-Following ───────────────────────────────────
print(f"\n{SEP2}\nGroup TF — EMA Crossover Trend-Following\n{SEP2}")
TF = []
for fast, slow, adx_thr, adx_arr, tag in [
    (5,  20, None, None,       "EMA5x20"),
    (8,  24, None, None,       "EMA8x24"),
    (5,  48, None, None,       "EMA5x48"),
    (12, 48, None, None,       "EMA12x48"),
    (5,  20, 1.2,  ADX_PROXY14, "EMA5x20-ADX14"),
    (8,  24, 1.2,  ADX_PROXY14, "EMA8x24-ADX14"),
    (5,  48, 1.3,  ADX_PROXY20, "EMA5x48-ADX20"),
    (12, 48, 1.3,  ADX_PROXY20, "EMA12x48-ADX20"),
]:
    vid = f"TF-{tag}"
    if adx_thr is None:
        s = sig_ema_cross(fast, slow)
    else:
        s = sig_ema_cross_adx(fast, slow, adx_thr, adx_arr)
    r = run_pipeline(vid, f"EMA({fast},{slow})"
                          + (f" + ADX>{adx_thr}" if adx_thr else ""),
                     s, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                     max_hold=MAX_HOLD, grp="TF")
    ALL_RESULTS.append(r); TF.append(r)

GRP["TF"] = TF
GRP_META["TF"] = ("EMA Crossover Trend-Following",
    "EMA veloce > EMA lenta → long; < lenta → short. "
    "Varianti con filtro ADX-proxy per richiedere trend forte. "
    "TP grid 2–10 ATR per catturare movimenti estesi.")

# ── Group PB: Pullback in Trend ───────────────────────────────────────────────
print(f"\n{SEP2}\nGroup PB — Pullback in Trend\n{SEP2}")
PB = []
for t_ema, pb_ema, dev, tag in [
    (100,  5, 0.002, "T100-PB5-d02"),
    (100,  8, 0.003, "T100-PB8-d03"),
    (200,  5, 0.002, "T200-PB5-d02"),
    (200,  8, 0.003, "T200-PB8-d03"),
    (200, 12, 0.005, "T200-PB12-d05"),
    (100, 20, 0.005, "T100-PB20-d05"),
    (200, 24, 0.008, "T200-PB24-d08"),
]:
    vid = f"PB-{tag}"
    s = sig_pullback_in_trend(t_ema, pb_ema, dev)
    r = run_pipeline(vid, f"Pullback  trend=EMA{t_ema}  pb=EMA{pb_ema}  dev={dev*100:.1f}%",
                     s, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                     max_hold=MAX_HOLD, grp="PB")
    ALL_RESULTS.append(r); PB.append(r)

GRP["PB"] = PB
GRP_META["PB"] = ("Pullback in Trend",
    "Trend filter (EMA lenta) + entry su pullback (prezzo dips sotto EMA veloce). "
    "Combina il bias direzionale con un timing di mean-reversion: "
    "WR atteso più alto del puro TF.")

# ── Group DN: Donchian Breakout ────────────────────────────────────────────────
print(f"\n{SEP2}\nGroup DN — Donchian Channel Breakout\n{SEP2}")
DN = []
for n, vol_filt, tag in [
    (12, False, "DN12"),
    (24, False, "DN24"),
    (48, False, "DN48"),
    (12, True,  "DN12-VSPK"),
    (24, True,  "DN24-VSPK"),
]:
    vid = f"DN-{tag}"
    s = sig_donchian_break(n)
    if vol_filt:
        s = np.where((VOL_RATIO > 1.5) & (s != 0), s, 0.0).astype(float)
        s[:WARMUP] = 0
    r = run_pipeline(vid, f"Donchian({n}H) breakout"
                          + (" + vol>1.5×" if vol_filt else ""),
                     s, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                     max_hold=MAX_HOLD, grp="DN")
    ALL_RESULTS.append(r); DN.append(r)

GRP["DN"] = DN
GRP_META["DN"] = ("Donchian Channel Breakout",
    "Chiusura precedente sopra/sotto il massimo/minimo delle ultime N ore "
    "→ segnale di breakout direzionale. "
    "Varianti con filtro volume spike per ridurre falsi breakout.")

# ── Group MC: MACD Momentum ────────────────────────────────────────────────────
print(f"\n{SEP2}\nGroup MC — MACD Momentum\n{SEP2}")
MC = []
for zc, mask_name, mask_arr, tag in [
    (True,  "ALL",    None,        "MACD-ZX-ALL"),
    (True,  "ACTIVE", ACTIVE_MASK, "MACD-ZX-ACT"),
    (False, "ALL",    None,        "MACD-BIAS-ALL"),
    (False, "ACTIVE", ACTIVE_MASK, "MACD-BIAS-ACT"),
]:
    vid = f"MC-{tag}"
    s = sig_macd(zero_cross=zc, mask=mask_arr)
    r = run_pipeline(vid, f"MACD {'zero-cross' if zc else 'bias'} sess={mask_name}",
                     s, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                     max_hold=MAX_HOLD, grp="MC")
    ALL_RESULTS.append(r); MC.append(r)

GRP["MC"] = MC
GRP_META["MC"] = ("MACD Momentum",
    "MACD(12,26,9): zero-cross dell'istogramma per catturare cambi di momentum, "
    "o bias continuo (segno dell'istogramma). Classico indicatore di momentum trend.")

# ── Group ST: Supertrend-proxy ────────────────────────────────────────────────
print(f"\n{SEP2}\nGroup ST — Supertrend Proxy\n{SEP2}")
ST = []
for mult, span, tag in [
    (1.5, 48, "ST15-E48"),
    (2.0, 48, "ST20-E48"),
    (1.5, 100,"ST15-E100"),
    (2.0, 100,"ST20-E100"),
    (3.0, 200,"ST30-E200"),
]:
    vid = f"ST-{tag}"
    s = sig_supertrend_proxy(mult, span)
    r = run_pipeline(vid, f"Supertrend proxy  atr×{mult}  EMA{span}",
                     s, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                     max_hold=MAX_HOLD, grp="ST")
    ALL_RESULTS.append(r); ST.append(r)

GRP["ST"] = ST
GRP_META["ST"] = ("Supertrend Proxy",
    "Segnale direzionale quando il prezzo supera EMA(N) ± k×ATR. "
    "Cattura breakout rispetto alla fascia di volatilità attesa.")

# ── Group ADP: Adaptive MR + TF ───────────────────────────────────────────────
print(f"\n{SEP2}\nGroup ADP — Adaptive: MR in ranging + TF in trending\n{SEP2}")
ADP = []

_sig_mr = sig_zscore_mr(24, 1.5)          # base MR signal (validated)
_sig_tf5x20 = sig_ema_cross(5, 20)        # best pure TF candidate
_sig_tf8x24 = sig_ema_cross(8, 24)
_sig_pb     = sig_pullback_in_trend(100, 8, 0.003)

for n_st, sig_tf, tf_tag in [
    (2, _sig_tf5x20, "MR24-TF5x20-2s"),
    (2, _sig_tf8x24, "MR24-TF8x24-2s"),
    (2, _sig_pb,     "MR24-PB100x8-2s"),
    (3, _sig_tf5x20, "MR24-TF5x20-3s"),
    (3, _sig_pb,     "MR24-PB100x8-3s"),
]:
    r = run_pipeline_adp(f"ADP-{tf_tag}", f"Adaptive(HMM-{n_st}s) MR+TF",
                         _sig_mr, sig_tf, n_states=n_st)
    ALL_RESULTS.append(r); ADP.append(r)

GRP["ADP"] = ADP
GRP_META["ADP"] = ("Adaptive MR + TF",
    "HMM 2–3 stati causal (fit su IS, forward predict su OOS): "
    "Z24 MR in regime ranging; EMA/Pullback TF in regime trending. "
    "TP/SL ottimizzati separatamente per ogni braccio.")

# ── Group CMB: Combo Confirmation ────────────────────────────────────────────
print(f"\n{SEP2}\nGroup CMB — Combo Signals\n{SEP2}")
CMB = []

# CMB1: EMA5x20 + MACD same direction
_e5x20  = sig_ema_cross(5, 20)
_macd_z = sig_macd(zero_cross=False)
_c1 = np.where((_e5x20 > 0) & (_macd_z > 0),  1.0,
      np.where((_e5x20 < 0) & (_macd_z < 0), -1.0, 0.0))
_c1[:WARMUP] = 0
r = run_pipeline("CMB-EMA5x20-MACD", "EMA5x20 + MACD bias agree",
                 _c1, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                 max_hold=MAX_HOLD, grp="CMB")
ALL_RESULTS.append(r); CMB.append(r)

# CMB2: EMA8x24 + Vol spike confirmation
_e8x24 = sig_ema_cross(8, 24)
_c2 = np.where((_e8x24 != 0) & (VOL_RATIO > 1.5), _e8x24, 0.0).astype(float)
_c2[:WARMUP] = 0
r = run_pipeline("CMB-EMA8x24-VSPK", "EMA8x24 + vol spike >1.5×",
                 _c2, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                 max_hold=MAX_HOLD, grp="CMB")
ALL_RESULTS.append(r); CMB.append(r)

# CMB3: Donchian24 + EMA trend filter
_dn24 = sig_donchian_break(24)
_c3 = np.where((_dn24 > 0) & (EMA[5] > EMA[20]),  1.0,
      np.where((_dn24 < 0) & (EMA[5] < EMA[20]), -1.0, 0.0))
_c3[:WARMUP] = 0
r = run_pipeline("CMB-DN24-EMA5x20", "Donchian(24H) + EMA5x20 confirm",
                 _c3, tp_grid=TP_GRID_TF, sl_grid=SL_GRID_TF,
                 max_hold=MAX_HOLD, grp="CMB")
ALL_RESULTS.append(r); CMB.append(r)

GRP["CMB"] = CMB
GRP_META["CMB"] = ("Combo Confirmation",
    "Due segnali TF devono concordare per generare il trade. "
    "Riduce i falsi segnali in mercati laterali a scapito della frequenza.")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
n_total = len(ALL_RESULTS)
n_ic    = sum(1 for r in ALL_RESULTS if r["ic_pass"])
n_wfo   = sum(1 for r in ALL_RESULTS if r["wfo_run"])
n_val   = sum(1 for r in ALL_RESULTS if r["validated"])

print(f"\n{SEP2}\nSUMMARY ({n_total} varianti)\n{SEP2}")
print(f"  IC pass: {n_ic}/{n_total}   WFO run: {n_wfo}/{n_total}   "
      f"Validated: {n_val}/{n_total}\n")

ranked = sorted([r for r in ALL_RESULTS if r["wfo_run"]],
                key=lambda r: r["oos_ret"], reverse=True)
print(f"  {'ID':<26}  {'n':>5}  {'WR':>6}  {'Ret':>7}  {'MDD':>7}  "
      f"{'P(pr)':>6}  {'P(ru)':>6}  Status")
print(f"  {'-'*26}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  "
      f"{'-'*6}  {'-'*6}  {'-'*10}")
for r in ranked:
    flag = "✅ VALID" if r["validated"] else ("⚠  OOS+" if r["oos_ret"] > 0 else "✗  FAIL")
    print(f"  {r['id']:<26}  {r['oos_n']:>5}  {r['oos_wr']:>6.1%}  "
          f"{r['oos_ret']:>+6.1f}%  {r['oos_mdd']:>6.1f}%  "
          f"{r['mc_p_profit']:>6.3f}  {r['mc_p_ruin']:>6.3f}  {flag}")
for r in ALL_RESULTS:
    if not r["wfo_run"]:
        print(f"  {r['id']:<26}  IC FAIL  ic={r['ic']:+.4f}  p={r['p_ic']:.4f}")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _equity_chart(rec: dict) -> str:
    eq = rec["oos_equity"]
    if len(eq) < 2: return ""
    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=_BG)
    ax.set_facecolor(_BG)
    color = _GRN if eq[-1] >= INIT_CAP else _RED
    ax.plot(range(len(eq)), eq, color=color, lw=1.5)
    ax.axhline(INIT_CAP, color=_GRID, ls="--", lw=0.8)
    ax.set_ylabel("Equity ($)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.15, color=_GRID)
    ax.set_title(f"{rec['id']} — OOS Equity", color=_TEXT, fontsize=9)
    b64 = _fig_b64(fig); plt.close(fig); return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;max-width:820px;">')

def _overview_chart() -> str:
    ran = sorted([r for r in ALL_RESULTS if r["wfo_run"]],
                 key=lambda r: r["oos_ret"], reverse=True)
    if not ran: return ""
    ids  = [r["id"]  for r in ran]
    rets = [r["oos_ret"] for r in ran]
    pps  = [r["mc_p_profit"] for r in ran]
    grps = [r["grp"] for r in ran]

    grp_colors = {"TF": _ACC, "PB": _GRN, "DN": _YEL, "MC": _ORG,
                  "ST": "#ab47bc", "ADP": "#26c6da", "CMB": "#ff7043", "": _TEXT}
    bar_colors = [grp_colors.get(g, _TEXT) for g in grps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4.5), facecolor=_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG); ax.tick_params(colors=_TEXT, labelsize=5.5)
        for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
        ax.grid(axis="y", alpha=0.2, color=_GRID)

    ax1.bar(range(len(ids)), rets,
            color=[_GRN if v > 0 else _RED for v in rets],
            edgecolor=_GRID, lw=0.3)
    ax1.axhline(0, color=_TEXT, lw=0.7)
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=60, ha="right", fontsize=4.5)
    ax1.set_ylabel("OOS Return (%)", color=_TEXT, fontsize=9)
    ax1.set_title("OOS Return per variante (verde=profit, rosso=loss)",
                  color=_TEXT, fontsize=10)

    ax2.bar(range(len(ids)), pps, color=bar_colors, edgecolor=_GRID, lw=0.3)
    ax2.axhline(0.90, color=_YEL, ls="--", lw=0.9, label="soglia 0.90")
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids, rotation=60, ha="right", fontsize=4.5)
    ax2.set_ylabel("MC P(profit)", color=_TEXT, fontsize=9)
    ax2.set_title("MC P(profit) — colore per gruppo", color=_TEXT, fontsize=10)
    ax2.set_ylim(0, 1.05)
    from matplotlib.patches import Patch
    legend_el = [Patch(color=v, label=k) for k, v in grp_colors.items() if k]
    ax2.legend(handles=legend_el, labelcolor=_TEXT, facecolor=_BG,
               fontsize=7, loc="lower right")

    fig.tight_layout(pad=1.5)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1300px;">'

def badge(r):
    if r["validated"]:              return '<span class="badge green">VALIDATED ✅</span>'
    if r["wfo_run"] and r["oos_ret"] > 0:
                                     return '<span class="badge yellow">OOS+ ⚠</span>'
    if r["wfo_run"]:                 return '<span class="badge orange">WFO FAIL ✗</span>'
    return                                  '<span class="badge red">IC FAIL ✗</span>'

def build_section(gid, title, desc, variants):
    rows = ""
    for r in variants:
        if r["wfo_run"]:
            rc = "green" if r["oos_ret"] > 0 else "red"
            pc = "green" if r["mc_p_profit"] > 0.90 else ("yellow" if r["mc_p_profit"] > 0.60 else "red")
            qc = "green" if r["mc_p_ruin"] < 0.05 else "red"
            tp_sl = (f"{r['best_tp']}ATR / {r['best_sl']}ATR"
                     if r.get("best_tp") else "—")
            wc = (f"<td>{r['oos_n']}</td><td>{r['oos_wr']:.1%}</td>"
                  f"<td class='{rc}'>{r['oos_ret']:+.1f}%</td>"
                  f"<td class='red'>{r['oos_mdd']:.1f}%</td>"
                  f"<td class='{pc}'>{r['mc_p_profit']:.3f}</td>"
                  f"<td class='{qc}'>{r['mc_p_ruin']:.3f}</td>"
                  f"<td>{tp_sl}</td>")
        else:
            wc = "<td colspan='7' style='color:#555'>—</td>"
        ic_c = "green" if r["ic_pass"] else "red"
        rows += (f"<tr>"
                 f"<td><code>{r['id']}</code></td>"
                 f"<td>{r['name']}</td>"
                 f"<td class='{ic_c}'>{r['ic']:+.4f}</td>"
                 f"<td class=\"{'green' if r['p_ic']<0.05 else 'red'}\">{r['p_ic']:.4f}</td>"
                 f"{wc}"
                 f"<td>{badge(r)}</td></tr>")
    eq_imgs = "".join(_equity_chart(r) for r in variants
                      if r["wfo_run"] and r["oos_n"] > 0)
    return f"""
    <div class="card section-card">
      <h2 style="margin-top:0">Group {gid} — {title}</h2>
      <p class="desc">{desc}</p>
      <div class="tbl-wrap"><table class="tbl">
        <thead><tr>
          <th>ID</th><th>Variante</th><th>IC</th><th>p-val</th>
          <th>OOS n</th><th>WR</th><th>OOS Ret</th><th>MDD</th>
          <th>P(profit)</th><th>P(ruin)</th><th>Best TP/SL</th><th>Status</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
      {eq_imgs}
    </div>"""

# assemble
overview_img  = _overview_chart()
sections_html = ""
for gid in ["TF", "PB", "DN", "MC", "ST", "ADP", "CMB"]:
    if GRP.get(gid):
        t, d = GRP_META.get(gid, ("", ""))
        sections_html += build_section(gid, t, d, GRP[gid])

best_val = [r for r in ALL_RESULTS if r["validated"]]
best_ret = max((r["oos_ret"] for r in ALL_RESULTS if r["wfo_run"]), default=0.0)
best_pp  = max((r["mc_p_profit"] for r in ALL_RESULTS if r["wfo_run"]), default=0.0)

CSS = f"""
  :root{{--bg:{_BG};--card:{_CARD};--grid:{_GRID};--text:{_TEXT};
         --acc:{_ACC};--grn:{_GRN};--red:{_RED};--yel:{_YEL};--org:{_ORG};}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);
        font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;
        line-height:1.5;padding:24px;}}
  h1{{color:var(--acc);font-size:22px;margin-bottom:4px;}}
  h2{{color:var(--acc);font-size:15px;margin:0 0 8px;}}
  .sub{{color:#777;font-size:12px;margin-bottom:24px;}}
  .kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px;}}
  .kpi{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
         padding:12px 18px;min-width:130px;}}
  .kpi .val{{font-size:26px;font-weight:700;color:var(--acc);}}
  .kpi .lbl{{font-size:11px;color:#777;}}
  .card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
          padding:18px;margin-bottom:20px;}}
  .section-card{{border-left:3px solid var(--acc);}}
  .desc{{color:#888;font-size:12px;margin-bottom:14px;}}
  .tbl-wrap{{overflow-x:auto;}}
  .tbl{{border-collapse:collapse;width:100%;min-width:950px;}}
  .tbl th{{background:var(--grid);color:var(--acc);text-align:left;
            padding:6px 10px;font-size:12px;white-space:nowrap;}}
  .tbl td{{padding:5px 10px;border-bottom:1px solid var(--grid);
            font-size:12px;white-space:nowrap;}}
  .tbl tr:hover td{{background:var(--grid);}}
  .green{{color:var(--grn)!important;}} .red{{color:var(--red)!important;}}
  .yellow{{color:var(--yel)!important;}} .dim{{color:#555!important;}}
  .badge{{display:inline-block;border-radius:4px;padding:2px 7px;
           font-size:11px;font-weight:600;}}
  .badge.green{{background:#1b3a1e;color:var(--grn);}}
  .badge.red{{background:#3a1a1a;color:var(--red);}}
  .badge.orange{{background:#3a2a00;color:var(--org);}}
  .badge.yellow{{background:#2d2a00;color:var(--yel);}}
  img{{display:block;margin-bottom:12px;border-radius:6px;max-width:100%;}}
  code{{font-size:11px;background:var(--grid);padding:1px 4px;border-radius:3px;}}
"""

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum & TF Intraday Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>BTC/USDT Intraday — Momentum & Trend-Following Strategies</h1>
<p class="sub">
  {n_total} varianti · 7 gruppi · IC {IC_HORIZON}H forward ·
  WFO 6m/2m/2m · MC N={N_SIMS:,} · Max hold {MAX_HOLD}H ·
  TP grid TF: [2,3,5,7,10]×ATR &nbsp;|&nbsp;
  {IDX1H[0].date()} – {IDX1H[-1].date()} · {N1H:,} bar 1H
</p>

<div class="kpi-row">
  <div class="kpi"><div class="val">{n_total}</div>
    <div class="lbl">Varianti testate</div></div>
  <div class="kpi"><div class="val">{n_ic}</div>
    <div class="lbl">IC pass (p&lt;0.05)</div></div>
  <div class="kpi"><div class="val">{n_wfo}</div>
    <div class="lbl">WFO eseguito</div></div>
  <div class="kpi">
    <div class="val" style="color:{'var(--grn)' if n_val>0 else 'var(--red)'}">{n_val}</div>
    <div class="lbl">Validate</div></div>
  <div class="kpi"><div class="val" style="font-size:18px">{best_ret:+.1f}%</div>
    <div class="lbl">Miglior OOS ret</div></div>
  <div class="kpi"><div class="val" style="font-size:18px">{best_pp:.3f}</div>
    <div class="lbl">Miglior MC p_profit</div></div>
</div>

<div class="card">
  <h2>Overview — OOS Return e MC P(profit) per variante</h2>
  {overview_img}
</div>

{sections_html}

<hr style="border-color:var(--grid);margin:32px 0 16px">
<p style="color:#444;font-size:11px;text-align:center;">
  BTCUSDT perpetual futures · Binance Vision 2020–2026 · 1H bars ·
  IC {IC_HORIZON}H Spearman · WFO 6m IS / 2m OOS / 2m step ·
  MC N={N_SIMS:,} · Risk 1%/trade · Fee 0.08% RT ·
  Max hold {MAX_HOLD}H · Cooldown {COOLDOWN}H
</p>
</body>
</html>"""

out = Path("reports/report_momentum_intraday.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n[DONE] Report → {out}  ({out.stat().st_size//1024} KB)")
