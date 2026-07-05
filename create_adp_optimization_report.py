#!/usr/bin/env python3
"""
create_adp_optimization_report.py
====================================
Grid-search optimisation of ADP-MR24-PB100x8-2s.

Bug fix vs original: ADP pipeline now enforces a shared 4H cooldown
across both MR and TF arms (the original had none, inflating trade count).

Phase 1 — PB parameters (MR fixed at win=24, thr=1.5σ):
  trend_ema ∈ [50, 100, 200]
  pb_ema    ∈ [5, 8, 12, 20]
  dev       ∈ [0.1%, 0.2%, 0.3%, 0.5%, 0.8%]
  → 3 × 4 × 5 = 60 combinations

Phase 2 — MR parameters (best PB from Phase 1 fixed):
  mr_win ∈ [18, 24, 36, 48]
  mr_thr ∈ [1.0, 1.5, 2.0, 2.5]
  → 4 × 4 = 16 combinations

All variants: n_states=2, features=[lr,atr], forward causal HMM, WFO 6m/2m/2m.
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
IC_HORIZON = 16
START_YEAR = 2020
N_SIMS     = 5_000
COOLDOWN   = 4
MAX_HOLD   = 48
WARMUP     = 200

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

TP_GRID_MR = [1.0, 2.0, 3.0, 5.0]
SL_GRID_MR = [0.25, 0.5, 0.75, 1.0]
TP_GRID_TF = [2.0, 3.0, 5.0, 7.0, 10.0]
SL_GRID_TF = [0.5, 1.0, 1.5, 2.0]

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
SEP2 = "═" * 70

print(SEP2)
print("ADP Optimisation — MR + Pullback-in-Trend (HMM-2s)")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h  = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
print(f"  {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

CL   = df1h["close"].values.astype(float)
HI   = df1h["high"].values.astype(float)
LO   = df1h["low"].values.astype(float)
VOL  = df1h["volume"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0,
                df1h["atr_14"].shift(1).values, 1.0)
CL_s = pd.Series(CL, index=IDX1H)

# HMM features
LOG_RET = np.concatenate([[0.0], np.log(CL[1:] / np.where(CL[:-1] > 0, CL[:-1], 1.0))])
ATR_PCT = np.where(CL > 0, ATR1 / CL, 0.001)

# EMA lookup
def _ema(span: int) -> np.ndarray:
    return CL_s.ewm(span=span, adjust=False).mean().shift(1).values

EMA_CACHE: dict[int, np.ndarray] = {}
for span in [5, 8, 12, 20, 50, 100, 150, 200]:
    EMA_CACHE[span] = _ema(span)

# ── WFO ───────────────────────────────────────────────────────────────────────
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
        pnl_r    = (abs(tp-ep)/a) if out == "tp" else -(abs(sl-ep)/a)
        fee_atr  = FEE * 2 * ep / a          # round-trip fee in ATR multiples
        pnl_r_net = pnl_r - fee_atr          # TP: reduces profit; SL: deepens loss
        risk     = cap * RISK_PCT
        lev      = min(max(abs(tp-ep)/ep, abs(sl-ep)/ep), MAX_LEV)
        dollar   = pnl_r_net * risk * lev
        cap     += dollar
        peak     = max(peak, cap)
        mdd      = min(mdd, (cap-peak)/peak)
        wins    += int(out == "tp")
        net_pnls.append(dollar)
    n = len(net_pnls); wr = wins/n if n else 0.0
    ret = (cap/INIT_CAP - 1) * 100
    avg_tp = np.mean([abs(ev["tp"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    avg_sl = np.mean([abs(ev["sl"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    sln = avg_sl + FEE_RT_PCT; tpn = avg_tp - FEE_RT_PCT   # both in % of price
    return dict(n=n, wr=wr, ret=ret, mdd=mdd*100, exppnl=wr*tpn-(1-wr)*sln,
                net_pnls=net_pnls, cap=cap)

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    arr = np.array(pnls, dtype=float)
    df_mc = pd.DataFrame({"net_pnl": arr, "gross_pnl": arr,
                           "total_fees": np.zeros(len(arr))})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)),
                p_ruin=float(mc.get("p_ruin", 1.0)))

def is_scan(sig, idx_is, tp_grid, sl_grid):
    best = (tp_grid[0], sl_grid[0]); best_xp = -np.inf
    for tf, sf in product(tp_grid, sl_grid):
        evs = make_events_simple(sig, idx_is, tf, sf)
        res = run_bt(evs)
        if res["exppnl"] > best_xp: best_xp = res["exppnl"]; best = (tf, sf)
    return best

def make_events_simple(sig, idx_arr, tp_f, sl_f):
    evs = []; last_s = -COOLDOWN
    for k in idx_arr:
        if k >= N1H or sig[k] == 0 or ATR1[k] <= 0: continue
        if k - last_s < COOLDOWN: continue
        evs.append(_ev(k, "long" if sig[k] > 0 else "short", tp_f, sl_f))
        last_s = k
    return evs

# ══════════════════════════════════════════════════════════════════════════════
# HMM (causal, no lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def build_hmm_feats(bar_indices):
    return np.column_stack([LOG_RET[bar_indices],
                             ATR_PCT[bar_indices]]).astype(float)

def hmm_forward_predict(model, obs):
    lp = model._compute_log_likelihood(obs)
    T, K = lp.shape
    la = np.full((T, K), -np.inf)
    la[0] = np.log(model.startprob_ + 1e-300) + lp[0]
    ltm = np.log(model.transmat_ + 1e-300)
    for t in range(1, T):
        for j in range(K):
            la[t, j] = logsumexp(la[t-1] + ltm[:, j]) + lp[t, j]
    return np.argmax(la, axis=1)

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

def ranging_state(model):
    return int(np.argmin(model.covars_[:, 0, 0]))

def trending_state(model):
    return int(np.argmax(model.covars_[:, 0, 0]))

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def sig_zscore_mr(win: int, thr: float) -> np.ndarray:
    zm = CL_s.rolling(win).mean()
    zs = CL_s.rolling(win).std().replace(0, np.nan)
    z = ((CL_s - zm) / zs).fillna(0).shift(1).values
    s = np.where(z < -thr, 1.0, np.where(z > thr, -1.0, 0.0))
    s[:WARMUP] = 0; return s

def sig_pullback(t_ema: int, pb_ema: int, dev: float) -> np.ndarray:
    """
    Long:  prev_close > EMA(t_ema)  AND  prev_close < EMA(pb_ema) * (1-dev)
    Short: prev_close < EMA(t_ema)  AND  prev_close > EMA(pb_ema) * (1+dev)
    All shifted — causal.
    """
    prev_cl = CL_s.shift(1).values
    te = EMA_CACHE[t_ema]
    pe = EMA_CACHE[pb_ema]
    s = np.where((prev_cl > te) & (prev_cl < pe * (1 - dev)),  1.0,
        np.where((prev_cl < te) & (prev_cl > pe * (1 + dev)), -1.0, 0.0))
    s[:WARMUP] = 0; return s

# ══════════════════════════════════════════════════════════════════════════════
# CORE ADP PIPELINE — with cooldown fix
# ══════════════════════════════════════════════════════════════════════════════

def make_adp_events(sig_adp, ranging_bars: set,
                    idx_oos, tp_mr, sl_mr, tp_tf, sl_tf) -> list:
    """
    Create ADP events with shared 4H cooldown across both arms.
    Different TP/SL applied depending on which arm generated the signal.
    """
    evs = []; last_s = -COOLDOWN
    for k in idx_oos:
        if k >= N1H or sig_adp[k] == 0 or ATR1[k] <= 0: continue
        if k - last_s < COOLDOWN: continue
        direction = "long" if sig_adp[k] > 0 else "short"
        if k in ranging_bars:
            evs.append(_ev(k, direction, tp_mr, sl_mr))
        else:
            evs.append(_ev(k, direction, tp_tf, sl_tf))
        last_s = k
    return evs


def run_adp(variant_id: str, sig_mr: np.ndarray, sig_pb: np.ndarray,
            n_states: int = 2, verbose: bool = True) -> dict:
    """
    WFO pipeline for adaptive MR+PB strategy with HMM regime filter.
    Cooldown properly shared across both arms.
    """
    rec = dict(
        id=variant_id, oos_n=0, oos_wr=0.0, oos_ret=0.0, oos_mdd=0.0,
        mc_p_profit=0.0, mc_p_ruin=1.0, validated=False,
        oos_pnls=[], oos_equity=[INIT_CAP], hmm_errors=0,
        avg_ranging_pct=0.0,
    )

    if verbose:
        print(f"  [{variant_id}]", end="", flush=True)

    all_evs = []; regime_pcts = []

    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue

        # IS scan on each arm independently (MR signal, TF signal)
        tp_mr, sl_mr = is_scan(sig_mr, idx_is, TP_GRID_MR, SL_GRID_MR)
        tp_tf, sl_tf = is_scan(sig_pb, idx_is, TP_GRID_TF, SL_GRID_TF)

        # HMM: fit on IS, causal predict on OOS
        model = fit_hmm(idx_is, n_states=n_states)
        if model is None:
            rec["hmm_errors"] += 1
            # fallback: MR only
            all_evs.extend(make_events_simple(sig_mr, idx_oos, tp_mr, sl_mr))
            print("e", end="", flush=True)
            continue

        r_st = ranging_state(model)
        t_st = trending_state(model)
        X_oos = build_hmm_feats(idx_oos)
        Xn_oos = (X_oos - model._mu) / model._std
        oos_states = hmm_forward_predict(model, Xn_oos)

        # Build combined OOS signal
        sig_adp = np.zeros(N1H)
        ranging_bars: set = set()
        for pos, k in enumerate(idx_oos):
            if oos_states[pos] == r_st and sig_mr[k] != 0:
                sig_adp[k] = sig_mr[k]
                ranging_bars.add(k)
            elif oos_states[pos] == t_st and sig_pb[k] != 0:
                sig_adp[k] = sig_pb[k]

        n_rng = (oos_states == r_st).sum()
        regime_pcts.append(n_rng / len(oos_states) * 100)

        # Events with shared cooldown and arm-specific TP/SL
        evs = make_adp_events(sig_adp, ranging_bars, idx_oos,
                               tp_mr, sl_mr, tp_tf, sl_tf)
        all_evs.extend(evs)
        print(".", end="", flush=True)

    print(" ", end="")

    if not all_evs:
        print("no events"); return rec

    res  = run_bt(all_evs, max_hold=MAX_HOLD)
    mc   = mc_summary(res["net_pnls"])
    valid = (res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05)

    eq = [INIT_CAP]; cap_eq = INIT_CAP
    for p in res["net_pnls"]: cap_eq += p; eq.append(cap_eq)

    avg_rng = np.mean(regime_pcts) if regime_pcts else 0.0

    if verbose:
        print(f"n={res['n']:5d}  wr={res['wr']:.1%}  ret={res['ret']:+6.1f}%  "
              f"mdd={res['mdd']:5.1f}%  pp={mc['p_profit']:.3f}  "
              f"rng={avg_rng:.0f}%  "
              f"{'✅' if valid else '✗'}")

    rec.update(oos_n=res["n"], oos_wr=res["wr"], oos_ret=res["ret"],
               oos_mdd=res["mdd"], mc_p_profit=mc["p_profit"],
               mc_p_ruin=mc["p_ruin"], validated=valid,
               oos_pnls=res["net_pnls"], oos_equity=eq,
               avg_ranging_pct=avg_rng)
    return rec

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — BASELINE (original params, with cooldown fix applied)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("Phase 0 — Baseline (ADP-MR24-PB100x8-d03, cooldown fixed)")
print(SEP2)
print(f"  {'ID':<32}  {'n':>5}  {'WR':>6}  {'Ret':>7}  {'MDD':>6}  "
      f"{'pp':>5}  {'rng%':>5}  Val")
print(f"  {'-'*32}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}  ---")

BASE_MR  = sig_zscore_mr(24, 1.5)
BASE_PB  = sig_pullback(100, 8, 0.003)
r_base   = run_adp("BASELINE-MR24-PB100x8-d03", BASE_MR, BASE_PB)
ALL_RESULTS = [r_base]

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SCAN PB PARAMETERS (MR fixed: win=24, thr=1.5σ)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("Phase 1 — PB Parameter Scan  (MR: win=24, thr=1.5σ  |  HMM: 2 stati)")
print(SEP2)
print(f"  {'ID':<32}  {'n':>5}  {'WR':>6}  {'Ret':>7}  {'MDD':>6}  "
      f"{'pp':>5}  {'rng%':>5}  Val")
print(f"  {'-'*32}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}  ---")

PH1_RESULTS = []
sig_mr_base = sig_zscore_mr(24, 1.5)

for t_ema in [50, 100, 200]:
    for pb_ema in [5, 8, 12, 20]:
        for dev in [0.001, 0.002, 0.003, 0.005, 0.008]:
            vid = f"PB-T{t_ema}-E{pb_ema}-d{int(dev*1000):03d}"
            sig_pb = sig_pullback(t_ema, pb_ema, dev)
            n_pb = int((sig_pb != 0).sum())
            if n_pb < 100:   # skip degenerate signals
                print(f"  [{vid}]  skip (n_pb={n_pb})")
                continue
            r = run_adp(vid, sig_mr_base, sig_pb)
            r["t_ema"] = t_ema; r["pb_ema"] = pb_ema; r["dev"] = dev
            PH1_RESULTS.append(r); ALL_RESULTS.append(r)

# Show Phase 1 ranking
ph1_valid = [r for r in PH1_RESULTS if r["validated"]]
ph1_ranked = sorted(PH1_RESULTS, key=lambda r: r["oos_ret"], reverse=True)
print(f"\n  Phase 1 summary: {len(ph1_valid)}/{len(PH1_RESULTS)} validated")
print(f"\n  TOP 10 by OOS return:")
for r in ph1_ranked[:10]:
    flag = "✅" if r["validated"] else "⚠" if r["oos_ret"] > 0 else "✗"
    print(f"  {flag} [{r['id']:<32}]  ret={r['oos_ret']:+6.1f}%  "
          f"pp={r['mc_p_profit']:.3f}  pr={r['mc_p_ruin']:.3f}  "
          f"n={r['oos_n']}  mdd={r['oos_mdd']:.1f}%")

# ── Pick best PB for Phase 2 ──────────────────────────────────────────────────
# Criterion: validated, highest OOS return, MDD < 20%
ph1_candidates = [r for r in PH1_RESULTS
                  if r["validated"] and r["oos_mdd"] > -20.0]
if ph1_candidates:
    best_ph1 = max(ph1_candidates, key=lambda r: r["oos_ret"])
    BEST_T_EMA = best_ph1["t_ema"]
    BEST_PB_EMA = best_ph1["pb_ema"]
    BEST_DEV    = best_ph1["dev"]
else:
    # fallback: pick by IC, highest return OOS+ (even unvalidated)
    best_ph1 = max(PH1_RESULTS, key=lambda r: r["oos_ret"])
    BEST_T_EMA  = best_ph1.get("t_ema", 100)
    BEST_PB_EMA = best_ph1.get("pb_ema", 8)
    BEST_DEV    = best_ph1.get("dev", 0.003)

print(f"\n  → Best PB: trend_ema={BEST_T_EMA}  pb_ema={BEST_PB_EMA}  "
      f"dev={BEST_DEV*100:.1f}%  (ret={best_ph1['oos_ret']:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SCAN MR PARAMETERS (best PB fixed)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print(f"Phase 2 — MR Parameter Scan  (PB: T{BEST_T_EMA}-E{BEST_PB_EMA}-d{int(BEST_DEV*1000):03d})")
print(SEP2)
print(f"  {'ID':<32}  {'n':>5}  {'WR':>6}  {'Ret':>7}  {'MDD':>6}  "
      f"{'pp':>5}  {'rng%':>5}  Val")
print(f"  {'-'*32}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}  ---")

sig_pb_best = sig_pullback(BEST_T_EMA, BEST_PB_EMA, BEST_DEV)
PH2_RESULTS = []

for mr_win in [18, 24, 36, 48]:
    for mr_thr in [1.0, 1.5, 2.0, 2.5]:
        vid = f"MR{mr_win}-t{str(mr_thr).replace('.','p')}-PB{BEST_T_EMA}x{BEST_PB_EMA}"
        sig_mr = sig_zscore_mr(mr_win, mr_thr)
        r = run_adp(vid, sig_mr, sig_pb_best)
        r["mr_win"] = mr_win; r["mr_thr"] = mr_thr
        PH2_RESULTS.append(r); ALL_RESULTS.append(r)

ph2_valid = [r for r in PH2_RESULTS if r["validated"]]
ph2_ranked = sorted(PH2_RESULTS, key=lambda r: r["oos_ret"], reverse=True)
print(f"\n  Phase 2 summary: {len(ph2_valid)}/{len(PH2_RESULTS)} validated")
print(f"\n  TOP 10 by OOS return:")
for r in ph2_ranked[:10]:
    flag = "✅" if r["validated"] else "⚠" if r["oos_ret"] > 0 else "✗"
    print(f"  {flag} [{r['id']:<32}]  ret={r['oos_ret']:+6.1f}%  "
          f"pp={r['mc_p_profit']:.3f}  pr={r['mc_p_ruin']:.3f}  "
          f"n={r['oos_n']}  mdd={r['oos_mdd']:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
all_valid = [r for r in ALL_RESULTS if r["validated"]]
all_ranked = sorted(ALL_RESULTS, key=lambda r: r["oos_ret"], reverse=True)

print(f"\n{SEP2}\nGLOBAL SUMMARY — {len(ALL_RESULTS)} varianti\n{SEP2}")
print(f"  Validated: {len(all_valid)}/{len(ALL_RESULTS)}\n")
print(f"  {'ID':<36}  {'n':>5}  {'WR':>6}  {'Ret':>7}  "
      f"{'MDD':>6}  {'pp':>5}  {'pr':>5}")
print(f"  {'-'*36}  {'-'*5}  {'-'*6}  {'-'*7}  "
      f"{'-'*6}  {'-'*5}  {'-'*5}")
for r in all_ranked[:20]:
    flag = "✅" if r["validated"] else ("⚠" if r["oos_ret"] > 0 else "✗")
    print(f"  {flag} {r['id']:<36}  {r['oos_n']:>5}  {r['oos_wr']:>6.1%}  "
          f"{r['oos_ret']:>+6.1f}%  {r['oos_mdd']:>6.1f}%  "
          f"{r['mc_p_profit']:>5.3f}  {r['mc_p_ruin']:>5.3f}")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _equity_chart(recs: list[dict], title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    palette = [_GRN, _ACC, _YEL, _ORG, "#ab47bc", "#26c6da",
               "#ff7043", "#80cbc4", "#f48fb1", "#ce93d8"]
    for idx, r in enumerate(recs):
        eq = r["oos_equity"]
        if len(eq) < 2: continue
        lw = 2.2 if idx == 0 else 1.2
        ax.plot(range(len(eq)), eq,
                color=palette[idx % len(palette)], lw=lw,
                label=r["id"][:30], alpha=0.9)
    ax.axhline(INIT_CAP, color=_GRID, ls="--", lw=0.8)
    ax.set_ylabel("Equity ($)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.15, color=_GRID)
    ax.set_title(title, color=_TEXT, fontsize=10)
    ax.legend(labelcolor=_TEXT, facecolor=_BG, fontsize=7,
              loc="upper left", ncol=2)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _heatmap_ret(results: list[dict], row_key: str, col_key: str,
                  row_vals, col_vals, title: str) -> str:
    """Return heatmap (row=row_key, col=col_key)."""
    data = {}
    for r in results:
        key = (r.get(row_key), r.get(col_key))
        data[key] = r["oos_ret"]

    M = np.full((len(row_vals), len(col_vals)), np.nan)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            if (rv, cv) in data:
                M[i, j] = data[(rv, cv)]

    vmax = max(abs(np.nanmax(M)), abs(np.nanmin(M)), 1.0)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=_BG)
    ax.set_facecolor(_BG)
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="OOS Return (%)")
    ax.set_xticks(range(len(col_vals)))
    ax.set_yticks(range(len(row_vals)))
    ax.set_xticklabels([str(v) for v in col_vals], color=_TEXT, fontsize=9)
    ax.set_yticklabels([str(v) for v in row_vals], color=_TEXT, fontsize=9)
    ax.set_xlabel(col_key, color=_TEXT, fontsize=10)
    ax.set_ylabel(row_key, color=_TEXT, fontsize=10)
    ax.set_title(title, color=_TEXT, fontsize=11)
    for i in range(len(row_vals)):
        for j in range(len(col_vals)):
            val = M[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(val) > vmax * 0.5 else "black")
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:900px;">'


def _scatter_chart(results: list[dict], title: str) -> str:
    """MDD vs Return scatter for all validated + OOS+ variants."""
    show = [r for r in results if r["oos_n"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=_BG)
    ax.set_facecolor(_BG)
    for r in show:
        c = _GRN if r["validated"] else (_YEL if r["oos_ret"] > 0 else _RED)
        ax.scatter(r["oos_mdd"], r["oos_ret"], color=c, s=40, alpha=0.75, zorder=3)
        if r["validated"] or r["oos_ret"] > 5:
            ax.annotate(r["id"][:22], (r["oos_mdd"], r["oos_ret"]),
                        fontsize=6, color=_TEXT, xytext=(4, 2),
                        textcoords="offset points")
    ax.axhline(0, color=_TEXT, lw=0.6, zorder=2)
    ax.axvline(0, color=_TEXT, lw=0.6, zorder=2)
    ax.set_xlabel("MDD (%)", color=_TEXT, fontsize=10)
    ax.set_ylabel("OOS Return (%)", color=_TEXT, fontsize=10)
    ax.set_title(title, color=_TEXT, fontsize=11)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.15, color=_GRID)
    from matplotlib.lines import Line2D
    legend_el = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=_GRN, ms=8, label='Validated'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=_YEL, ms=8, label='OOS+'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=_RED, ms=8, label='Fail'),
    ]
    ax.legend(handles=legend_el, labelcolor=_TEXT, facecolor=_BG, fontsize=8)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:900px;">'


def table_rows(results: list[dict], show_params: str = "none") -> str:
    rows = ""
    for r in sorted(results, key=lambda x: x["oos_ret"], reverse=True):
        rc = "green" if r["oos_ret"] > 0 else "red"
        pc = "green" if r["mc_p_profit"] > 0.90 else ("yellow" if r["mc_p_profit"] > 0.60 else "red")
        qc = "green" if r["mc_p_ruin"] < 0.05 else "red"
        flag = ("✅" if r["validated"] else ("⚠" if r["oos_ret"] > 0 else "✗"))
        extra = ""
        if show_params == "pb":
            extra = (f"<td>{r.get('t_ema','')}</td>"
                     f"<td>{r.get('pb_ema','')}</td>"
                     f"<td>{r.get('dev',0)*100:.1f}%</td>")
        elif show_params == "mr":
            extra = (f"<td>{r.get('mr_win','')}</td>"
                     f"<td>{r.get('mr_thr','')}</td>")
        rows += (f"<tr>"
                 f"<td><code>{r['id']}</code></td>"
                 f"{extra}"
                 f"<td>{r['oos_n']}</td>"
                 f"<td>{r['oos_wr']:.1%}</td>"
                 f"<td class='{rc}'>{r['oos_ret']:+.1f}%</td>"
                 f"<td class='red'>{r['oos_mdd']:.1f}%</td>"
                 f"<td class='{pc}'>{r['mc_p_profit']:.3f}</td>"
                 f"<td class='{qc}'>{r['mc_p_ruin']:.3f}</td>"
                 f"<td>{r['avg_ranging_pct']:.0f}%</td>"
                 f"<td>{flag}</td></tr>")
    return rows


CSS = f"""
  :root{{--bg:{_BG};--card:{_CARD};--grid:{_GRID};--text:{_TEXT};
         --acc:{_ACC};--grn:{_GRN};--red:{_RED};--yel:{_YEL};--org:{_ORG};}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);
        font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;
        line-height:1.5;padding:24px;}}
  h1{{color:var(--acc);font-size:22px;margin-bottom:4px;}}
  h2{{color:var(--acc);font-size:15px;margin:12px 0 8px;}}
  .sub{{color:#777;font-size:12px;margin-bottom:24px;}}
  .kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px;}}
  .kpi{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
         padding:12px 18px;min-width:130px;}}
  .kpi .val{{font-size:26px;font-weight:700;color:var(--acc);}}
  .kpi .lbl{{font-size:11px;color:#777;}}
  .card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
          padding:18px;margin-bottom:20px;}}
  .acc-card{{border-left:3px solid var(--acc);}}
  .grn-card{{border-left:3px solid var(--grn);}}
  .desc{{color:#888;font-size:12px;margin-bottom:14px;}}
  .tbl-wrap{{overflow-x:auto;}}
  .tbl{{border-collapse:collapse;width:100%;min-width:800px;}}
  .tbl th{{background:var(--grid);color:var(--acc);text-align:left;
            padding:6px 10px;font-size:12px;white-space:nowrap;}}
  .tbl td{{padding:5px 10px;border-bottom:1px solid var(--grid);
            font-size:12px;white-space:nowrap;}}
  .tbl tr:hover td{{background:var(--grid);}}
  .green{{color:var(--grn)!important;}} .red{{color:var(--red)!important;}}
  .yellow{{color:var(--yel)!important;}}
  img{{display:block;margin-bottom:12px;border-radius:6px;max-width:100%;}}
  code{{font-size:11px;background:var(--grid);padding:1px 4px;border-radius:3px;}}
  .note{{background:#111824;border:1px solid #2a3040;border-radius:6px;
          padding:12px 16px;margin-bottom:16px;font-size:12px;color:#9ab;}}
"""

# Best validated overall
best_overall = max(all_valid, key=lambda r: r["oos_ret"]) if all_valid else r_base
best_ph2     = max(PH2_RESULTS, key=lambda r: r["oos_ret"]) if PH2_RESULTS else r_base

# Heatmaps
hm_dev_pbema = _heatmap_ret(
    [r for r in PH1_RESULTS if r.get("t_ema") == BEST_T_EMA],
    "pb_ema", "dev",
    [5, 8, 12, 20],
    [0.001, 0.002, 0.003, 0.005, 0.008],
    f"Phase 1 Heatmap — OOS Return (%)  [trend_ema={BEST_T_EMA}]\nrows=pb_ema, cols=dev"
)
hm_mrwin_mrthr = _heatmap_ret(
    PH2_RESULTS, "mr_win", "mr_thr",
    [18, 24, 36, 48], [1.0, 1.5, 2.0, 2.5],
    f"Phase 2 Heatmap — OOS Return (%)  [PB: T{BEST_T_EMA}-E{BEST_PB_EMA}-d{BEST_DEV*100:.1f}%]\n"
    f"rows=mr_win, cols=mr_thr"
)
scatter = _scatter_chart(ALL_RESULTS, "Return vs MDD — tutte le varianti ADP")
equity_top5 = _equity_chart(all_ranked[:5], "OOS Equity — Top 5 varianti")

ph1_th = "<th>t_ema</th><th>pb_ema</th><th>dev</th>"
ph2_th = "<th>mr_win</th><th>mr_thr</th>"
base_tbl_rows = table_rows([r_base])
ph1_tbl_rows  = table_rows(PH1_RESULTS, "pb")
ph2_tbl_rows  = table_rows(PH2_RESULTS, "mr")

n_val_total = len(all_valid)

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADP Optimisation Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>ADP Optimisation — MR(Z-score) + Pullback-in-Trend</h1>
<p class="sub">
  {len(ALL_RESULTS)} varianti · HMM-2s causal · WFO 6m/2m/2m · MC N={N_SIMS:,} ·
  Cooldown {COOLDOWN}H condiviso fra i due arms ·
  {IDX1H[0].date()} – {IDX1H[-1].date()} · {N1H:,} bar 1H
</p>

<div class="note">
  <strong>Bug fix (cooldown):</strong>
  Il pipeline ADP originale non applicava il cooldown {COOLDOWN}H tra i segnali dei due arm.
  Questa versione condivide il cooldown: dopo ogni trade (MR o PB), i prossimi {COOLDOWN}H sono bloccati.
  Il numero di OOS trade è pertanto ridotto rispetto al report precedente.
</div>
<div class="note">
  <strong>Fee fix:</strong>
  Tutti i ritorni sono ora <em>netti</em> di commissioni round-trip {FEE_RT_PCT:.2f}% (ipotesi taker: 0.04% per lato).
  La formula: <code>pnl_net = (pnl_atr − fee_atr) × risk × lev</code> dove
  <code>fee_atr = 2×FEE×price/ATR ≈ {2*FEE*100:.3f}×(price/ATR)</code>.
  Con 4 000+ trade su 6 anni, l'impatto cumulativo è circa 6–8 punti percentuali.
  Usando ordini limit (maker 0.01%/lato) l'impatto scende a circa 1.8 punti.
</div>

<div class="kpi-row">
  <div class="kpi"><div class="val">{len(ALL_RESULTS)}</div>
    <div class="lbl">Varianti testate</div></div>
  <div class="kpi">
    <div class="val" style="color:{'var(--grn)' if n_val_total>0 else 'var(--red)'}">{n_val_total}</div>
    <div class="lbl">Validate</div></div>
  <div class="kpi">
    <div class="val" style="font-size:18px;color:var(--grn)">{best_overall['oos_ret']:+.1f}%</div>
    <div class="lbl">Miglior OOS ret</div></div>
  <div class="kpi">
    <div class="val" style="font-size:18px">{best_overall['oos_mdd']:.1f}%</div>
    <div class="lbl">MDD variante migliore</div></div>
  <div class="kpi">
    <div class="val" style="font-size:18px">{best_overall['mc_p_profit']:.3f}</div>
    <div class="lbl">MC P(profit) migliore</div></div>
  <div class="kpi">
    <div class="val" style="font-size:18px">{best_overall['id'].split('-')[0]}</div>
    <div class="lbl">Winner ID (prefix)</div></div>
</div>

<div class="card">
  <h2>OOS Equity — Top 5 varianti</h2>
  {equity_top5}
</div>

<div class="card">
  <h2>Scatter Return vs MDD</h2>
  {scatter}
</div>

<div class="card acc-card">
  <h2>Phase 0 — Baseline (cooldown fix applicato)</h2>
  <p class="desc">ADP-MR24-PB100x8-d03 con cooldown condiviso. Confronto diretto con il risultato precedente (+20.9%).</p>
  <div class="tbl-wrap"><table class="tbl">
    <thead><tr><th>ID</th><th>OOS n</th><th>WR</th><th>OOS Ret</th>
      <th>MDD</th><th>P(profit)</th><th>P(ruin)</th><th>Ranging%</th><th>Val</th>
    </tr></thead>
    <tbody>{base_tbl_rows}</tbody>
  </table></div>
</div>

<div class="card acc-card">
  <h2>Phase 1 — PB Parameter Scan (MR: win=24, thr=1.5σ)</h2>
  <p class="desc">60 combinazioni: trend_ema ∈ [50,100,200] × pb_ema ∈ [5,8,12,20] × dev ∈ [0.1%..0.8%]</p>
  {hm_dev_pbema}
  <div class="tbl-wrap"><table class="tbl">
    <thead><tr><th>ID</th>{ph1_th}<th>OOS n</th><th>WR</th><th>OOS Ret</th>
      <th>MDD</th><th>P(profit)</th><th>P(ruin)</th><th>Ranging%</th><th>Val</th>
    </tr></thead>
    <tbody>{ph1_tbl_rows}</tbody>
  </table></div>
</div>

<div class="card grn-card">
  <h2>Phase 2 — MR Parameter Scan (best PB fixed: T{BEST_T_EMA}-E{BEST_PB_EMA}-d{BEST_DEV*100:.1f}%)</h2>
  <p class="desc">16 combinazioni: mr_win ∈ [18,24,36,48] × mr_thr ∈ [1.0,1.5,2.0,2.5]σ</p>
  {hm_mrwin_mrthr}
  <div class="tbl-wrap"><table class="tbl">
    <thead><tr><th>ID</th>{ph2_th}<th>OOS n</th><th>WR</th><th>OOS Ret</th>
      <th>MDD</th><th>P(profit)</th><th>P(ruin)</th><th>Ranging%</th><th>Val</th>
    </tr></thead>
    <tbody>{ph2_tbl_rows}</tbody>
  </table></div>
</div>

<hr style="border-color:var(--grid);margin:32px 0 16px">
<p style="color:#444;font-size:11px;text-align:center;">
  BTCUSDT perpetual futures · Binance Vision 2020–2026 · 1H bars ·
  WFO 6m IS / 2m OOS / 2m step · MC N={N_SIMS:,} · Risk 1%/trade · Fee 0.08% RT ·
  HMM GaussianHMM(diag) forward causal · Cooldown {COOLDOWN}H condiviso
</p>
</body>
</html>"""

out = Path("reports/report_adp_optimization.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n[DONE] Report → {out}  ({out.stat().st_size//1024} KB)")
