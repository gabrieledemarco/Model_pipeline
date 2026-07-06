#!/usr/bin/env python3
"""
create_holdout_validation_report.py
=====================================
Pre-paper-trading validation checklist for the 5 DSR-validated ADP variants
(MR{18,24,36,48}-t{2.0,2.5}-PB100x5), primary pick: MR24-t2p0-PB100x5.

Sections:
  A. Per-window time stability (all 5 configs) — is the edge concentrated
     in one lucky stretch, or consistent year over year?
  B. Genuine temporal holdout — re-select best MR params using ONLY windows
     with oo_e<=2025-01-01, then evaluate the frozen winner on 2025-2026
     windows never used in selection.
  C. Slippage sensitivity (winning config) — 0 / 2 / 5 / 10 bps additional
     adverse fill on both TP and SL.
  D. No-touch trade handling (winning config) — forced close-at-market at
     max_hold instead of silently dropping unresolved trades.
  E. Position sizing reality check (winning config) — implied BTC notional
     per trade as % of equity, distribution.
  F. Block-bootstrap Monte Carlo (winning config) — contiguous-block
     resampling vs i.i.d., to partially capture regime-persistence risk.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from hmmlearn import hmm as hmmlib
from scipy.special import logsumexp

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78

# ── Config (identico a create_adp_optimization_report.py) ─────────────────────
INIT_CAP   = 100_000.0
RISK_PCT   = 0.01
FEE        = 0.0004
MAX_LEV    = 5.0
START_YEAR = 2020
COOLDOWN   = 4
MAX_HOLD   = 48
WARMUP     = 200
N_SIMS     = 5_000
DSR_THRESHOLD = 0.95

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

TP_GRID_MR = [1.0, 2.0, 3.0, 5.0]
SL_GRID_MR = [0.25, 0.5, 0.75, 1.0]
TP_GRID_TF = [2.0, 3.0, 5.0, 7.0, 10.0]
SL_GRID_TF = [0.5, 1.0, 1.5, 2.0]

# Best PB from the original full-sample Phase 1 (frozen; see note in Section B)
T_EMA, PB_EMA, DEV = 100, 5, 0.003

# The 5 DSR-validated MR configs
VALIDATED_CONFIGS = [
    ("MR24-t2p0-PB100x5", 24, 2.0),
    ("MR18-t2p5-PB100x5", 18, 2.5),
    ("MR48-t2p5-PB100x5", 48, 2.5),
    ("MR24-t2p5-PB100x5", 24, 2.5),
    ("MR36-t2p5-PB100x5", 36, 2.5),
]
PRIMARY = ("MR24-t2p0-PB100x5", 24, 2.0)

print(SEP)
print("Holdout & Pre-Paper-Trading Validation Checklist")
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw   = fetch_extended_data(start_year=START_YEAR, start_month=1,
                             fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h  = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
print(f"  {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

CL   = df1h["close"].values.astype(float)
HI   = df1h["high"].values.astype(float)
LO   = df1h["low"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0,
                df1h["atr_14"].shift(1).values, 1.0)
CL_s = pd.Series(CL, index=IDX1H)

LOG_RET = np.concatenate([[0.0], np.log(CL[1:] / np.where(CL[:-1] > 0, CL[:-1], 1.0))])
ATR_PCT = np.where(CL > 0, ATR1 / CL, 0.001)

def _ema(span: int) -> np.ndarray:
    return CL_s.ewm(span=span, adjust=False).mean().shift(1).values

EMA_CACHE = {span: _ema(span) for span in [5, 8, 12, 20, 50, 100, 150, 200]}

def wf_dates(idx, start=None, end=None):
    t0 = idx[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > idx[-1]: break
        if (start is None or oo_e > start) and (end is None or oo_s < end):
            windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
CUTOFF = pd.Timestamp("2025-01-01")
WF_PRE2025 = [w for w in WF_WINDOWS if w[3] <= CUTOFF]
WF_HOLDOUT = [w for w in WF_WINDOWS if w[2] >= CUTOFF]
print(f"[WFO] {len(WF_WINDOWS)} total windows  |  "
      f"{len(WF_PRE2025)} pre-2025  |  {len(WF_HOLDOUT)} holdout (2025-2026)")

# ══════════════════════════════════════════════════════════════════════════════
# HMM (causal, no lookahead) — identico
# ══════════════════════════════════════════════════════════════════════════════
def build_hmm_feats(bar_indices):
    return np.column_stack([LOG_RET[bar_indices], ATR_PCT[bar_indices]]).astype(float)

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
# SIGNAL BUILDERS — identico
# ══════════════════════════════════════════════════════════════════════════════
def sig_zscore_mr(win: int, thr: float) -> np.ndarray:
    zm = CL_s.rolling(win).mean()
    zs = CL_s.rolling(win).std().replace(0, np.nan)
    z = ((CL_s - zm) / zs).fillna(0).shift(1).values
    s = np.where(z < -thr, 1.0, np.where(z > thr, -1.0, 0.0))
    s[:WARMUP] = 0; return s

def sig_pullback(t_ema: int, pb_ema: int, dev: float) -> np.ndarray:
    prev_cl = CL_s.shift(1).values
    te = EMA_CACHE[t_ema]; pe = EMA_CACHE[pb_ema]
    s = np.where((prev_cl > te) & (prev_cl < pe * (1 - dev)),  1.0,
        np.where((prev_cl < te) & (prev_cl > pe * (1 + dev)), -1.0, 0.0))
    s[:WARMUP] = 0; return s

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _ev(i, direction, tp_f, sl_f):
    d = 1 if direction == "long" else -1
    a = ATR1[i]; ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep+d*tp_f*a, sl=ep-d*sl_f*a, a=a,
                tp_f=tp_f, sl_f=sl_f)

def is_scan(sig, idx_is, tp_grid, sl_grid):
    from itertools import product
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

def make_adp_events(sig_adp, ranging_bars, idx_oos, tp_mr, sl_mr, tp_tf, sl_tf):
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

FEE_RT_PCT = FEE * 2 * 100

def run_bt(events, max_hold=MAX_HOLD, slippage_pct=0.0, force_close_notouch=False):
    """
    Fee-corrected backtest, extended with:
      slippage_pct        : extra adverse fill (fraction of price) applied to
                             BOTH tp and sl execution (worsens both).
      force_close_notouch : if True, trades that touch neither tp nor sl
                             within max_hold are force-closed at the close
                             of bar (i+max_hold) instead of being dropped.
    """
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, exppnl=0.0, net_pnls=[], cap=INIT_CAP,
                     n_dropped_none=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_dropped_none = 0
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"; exit_price = None
        for k in range(1, max_hold + 1):
            if i+k >= N1H: break
            hk, lk = HI[i+k], LO[i+k]
            if d == 1:
                if hk >= tp: out = "tp"; break
                if lk <= sl: out = "sl"; break
            else:
                if lk <= tp: out = "tp"; break
                if hk >= sl: out = "sl"; break
        if out == "none":
            if not force_close_notouch:
                n_dropped_none += 1
                continue
            j = min(i + max_hold, N1H - 1)
            exit_price = CL[j]
            pnl_r = d * (exit_price - ep) / a
        else:
            fill_tp = tp - d * slippage_pct * ep   # adverse: reduces the favorable move
            fill_sl = sl - d * slippage_pct * ep   # adverse: deepens the loss
            if out == "tp":
                pnl_r = d * (fill_tp - ep) / a
            else:
                pnl_r = d * (fill_sl - ep) / a
        fee_atr  = FEE * 2 * ep / a
        pnl_r_net = pnl_r - fee_atr
        risk     = cap * RISK_PCT
        lev      = min(max(abs(tp-ep)/ep, abs(sl-ep)/ep), MAX_LEV)
        dollar   = pnl_r_net * risk * lev
        cap     += dollar
        peak     = max(peak, cap)
        mdd      = min(mdd, (cap-peak)/peak)
        wins    += int(pnl_r > 0)
        net_pnls.append(dollar)
    n = len(net_pnls); wr = wins/n if n else 0.0
    ret = (cap/INIT_CAP - 1) * 100
    avg_tp = np.mean([abs(ev["tp"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    avg_sl = np.mean([abs(ev["sl"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    sln = avg_sl + FEE_RT_PCT; tpn = avg_tp - FEE_RT_PCT
    return dict(n=n, wr=wr, ret=ret, mdd=mdd*100, exppnl=wr*tpn-(1-wr)*sln,
                net_pnls=net_pnls, cap=cap, n_dropped_none=n_dropped_none)

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    arr = np.array(pnls, dtype=float)
    df_mc = pd.DataFrame({"net_pnl": arr})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def gen_events_for_config(mr_win, mr_thr, windows, per_window=False):
    """
    Run the ADP WFO event-generation loop for a given MR config (PB fixed),
    restricted to `windows`. If per_window=True, also returns a list of
    (oo_s, events_in_window) tuples for time-stability analysis.
    """
    sig_mr = sig_zscore_mr(mr_win, mr_thr)
    sig_pb = sig_pullback(T_EMA, PB_EMA, DEV)
    all_evs = []
    window_evs = []
    for tr_s, tr_e, oo_s, oo_e in windows:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50:
            if per_window: window_evs.append((oo_s, []))
            continue

        tp_mr, sl_mr = is_scan(sig_mr, idx_is, TP_GRID_MR, SL_GRID_MR)
        tp_tf, sl_tf = is_scan(sig_pb, idx_is, TP_GRID_TF, SL_GRID_TF)

        model = fit_hmm(idx_is, n_states=2)
        if model is None:
            evs = make_events_simple(sig_mr, idx_oos, tp_mr, sl_mr)
            all_evs.extend(evs)
            if per_window: window_evs.append((oo_s, evs))
            print("e", end="", flush=True)
            continue

        r_st = ranging_state(model); t_st = trending_state(model)
        X_oos = build_hmm_feats(idx_oos)
        Xn_oos = (X_oos - model._mu) / model._std
        oos_states = hmm_forward_predict(model, Xn_oos)

        sig_adp = np.zeros(N1H)
        ranging_bars: set = set()
        for pos, k in enumerate(idx_oos):
            if oos_states[pos] == r_st and sig_mr[k] != 0:
                sig_adp[k] = sig_mr[k]; ranging_bars.add(k)
            elif oos_states[pos] == t_st and sig_pb[k] != 0:
                sig_adp[k] = sig_pb[k]

        evs = make_adp_events(sig_adp, ranging_bars, idx_oos, tp_mr, sl_mr, tp_tf, sl_tf)
        all_evs.extend(evs)
        if per_window: window_evs.append((oo_s, evs))
        print(".", end="", flush=True)
    return (all_evs, window_evs) if per_window else all_evs

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Per-window time stability (all 5 validated configs)
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("SECTION A — Time stability per finestra WFO (5 config validati)")
w(SEP)

all_events_cache = {}   # name -> all_evs (full-sample), reused in Section C-F for PRIMARY
year_stability = {}      # name -> {year: (n, ret%, wr)}

for name, mr_win, mr_thr in VALIDATED_CONFIGS:
    print(f"\n  [{name}] generating events across {len(WF_WINDOWS)} windows ", end="")
    all_evs, window_evs = gen_events_for_config(mr_win, mr_thr, WF_WINDOWS, per_window=True)
    all_events_cache[name] = all_evs
    print()

    by_year: dict[int, list] = {}
    for oo_s, evs in window_evs:
        by_year.setdefault(oo_s.year, []).extend(evs)

    row = {}
    for year in sorted(by_year):
        res = run_bt(by_year[year])
        row[year] = (res["n"], res["ret"], res["wr"])
    year_stability[name] = row

w(f"\n  {'Config':<22}" + "".join(f"{y:>14}" for y in sorted({yy for r in year_stability.values() for yy in r})))
all_years = sorted({yy for r in year_stability.values() for yy in r})
for name, _, _ in VALIDATED_CONFIGS:
    row = year_stability[name]
    cells = "".join(
        f"{(f'{row[y][1]:+.1f}%(n={row[y][0]})' if y in row else '—'):>14}"
        for y in all_years
    )
    w(f"  {name:<22}{cells}")

w("\n  Nota: ogni cella è il return% OOS aggregato per quell'anno solare "
  "(somma dei trade le cui finestre OOS iniziano in quell'anno), non ricapitalizzato "
  "in modo indipendente — utile per vedere in QUALI anni si concentra l'edge.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Genuine temporal holdout (2025-2026 never used in selection)
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w(f"SECTION B — Holdout temporale genuino  "
  f"(selezione su {len(WF_PRE2025)} finestre pre-2025, eval su {len(WF_HOLDOUT)} finestre 2025-2026)")
w(SEP)
w("  NOTA METODOLOGICA: la config Pullback-in-Trend (T100-E5-d003) resta quella scelta "
  "nella Phase 1 originale (full-sample) — rifare da zero anche quella selezione "
  "sotto vincolo temporale rigoroso richiederebbe rieseguire l'intera Phase 1 (60 "
  "varianti) ristretta a pre-2025, un altro ordine di grandezza di calcolo. Qui "
  "isoliamo la domanda più rilevante: la scelta dei parametri MR (soglia/finestra) "
  "regge se selezionata SOLO su dati fino al 2024?")

MR_GRID = [(win, thr) for win in [18, 24, 36, 48] for thr in [1.0, 1.5, 2.0, 2.5]]
ph2_pre2025 = []
for win, thr in MR_GRID:
    name = f"MR{win}-t{str(thr).replace('.','p')}-PB{T_EMA}x{PB_EMA}"
    evs = gen_events_for_config(win, thr, WF_PRE2025)
    res = run_bt(evs)
    mc = mc_summary(res["net_pnls"])
    valid = res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05
    ph2_pre2025.append(dict(id=name, mr_win=win, mr_thr=thr, oos_n=res["n"], oos_ret=res["ret"],
                             oos_mdd=res["mdd"], mc_p_profit=mc["p_profit"], mc_p_ruin=mc["p_ruin"],
                             validated=valid, net_pnls=res["net_pnls"]))
    print(f"\n  [{name}] n={res['n']} ret={res['ret']:+.1f}% pp={mc['p_profit']:.3f} "
          f"{'✅' if valid else '✗'}", end="")

print()
deflated_sharpe_ratio_family(ph2_pre2025, sharpe_key="sharpe_hat_ph2", dsr_key="dsr_ph2",
                              pnls_key="net_pnls")
for r in ph2_pre2025:
    r["validated_dsr"] = r["validated"] and r["dsr_ph2"] >= DSR_THRESHOLD

w(f"\n  Selezione ristretta a pre-2025 (N={len(WF_PRE2025)} finestre):")
w(f"  {'ID':<24}  {'n':>5}  {'Ret':>7}  {'pp':>5}  {'DSR':>5}  Val")
ranked_pre = sorted(ph2_pre2025, key=lambda r: r["oos_ret"], reverse=True)
for r in ranked_pre[:6]:
    flag = "✅" if r["validated_dsr"] else ("⚠" if r["validated"] else "✗")
    w(f"  {flag} {r['id']:<22}  {r['oos_n']:>5}  {r['oos_ret']:>+6.1f}%  "
      f"{r['mc_p_profit']:>5.3f}  {r['dsr_ph2']:>5.3f}")

pre2025_valid = [r for r in ph2_pre2025 if r["validated_dsr"]]
winner_pre2025 = max(pre2025_valid, key=lambda r: r["oos_ret"]) if pre2025_valid \
    else max(ph2_pre2025, key=lambda r: r["oos_ret"])
w(f"\n  → Winner selezionato SOLO su pre-2025: {winner_pre2025['id']}  "
  f"(ret={winner_pre2025['oos_ret']:+.1f}%, pp={winner_pre2025['mc_p_profit']:.3f})")
w(f"  Coincide con il winner full-sample (MR24-t2p0)? "
  f"{'SÌ' if winner_pre2025['id']==PRIMARY[0] else 'NO — vedi sotto'}")

# Evaluate the FROZEN pre-2025 winner on the true 2025-2026 holdout
holdout_evs = gen_events_for_config(winner_pre2025["mr_win"], winner_pre2025["mr_thr"], WF_HOLDOUT)
holdout_res = run_bt(holdout_evs)
holdout_mc = mc_summary(holdout_res["net_pnls"])
w(f"\n  {winner_pre2025['id']} valutato SUL VERO HOLDOUT 2025-2026 "
  f"(N={len(WF_HOLDOUT)} finestre, MAI viste in selezione):")
w(f"    n={holdout_res['n']}  ret={holdout_res['ret']:+.2f}%  mdd={holdout_res['mdd']:.2f}%  "
  f"pp={holdout_mc['p_profit']:.3f}  pr={holdout_mc['p_ruin']:.3f}")

# Also show PRIMARY (MR24-t2p0) performance restricted to the same holdout window,
# for direct comparison even if it wasn't the pre-2025-selected winner.
if winner_pre2025["id"] != PRIMARY[0]:
    primary_holdout_evs = gen_events_for_config(PRIMARY[1], PRIMARY[2], WF_HOLDOUT)
    primary_holdout_res = run_bt(primary_holdout_evs)
    primary_holdout_mc = mc_summary(primary_holdout_res["net_pnls"])
    w(f"\n  Per confronto, {PRIMARY[0]} (winner full-sample) sullo stesso holdout 2025-2026:")
    w(f"    n={primary_holdout_res['n']}  ret={primary_holdout_res['ret']:+.2f}%  "
      f"mdd={primary_holdout_res['mdd']:.2f}%  pp={primary_holdout_mc['p_profit']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Reuse full-sample events for PRIMARY (already generated in Section A) for
# Sections C-F — no need to regenerate.
# ══════════════════════════════════════════════════════════════════════════════
primary_evs = all_events_cache[PRIMARY[0]]
baseline_res = run_bt(primary_evs)
w(f"\n{SEP}")
w(f"Baseline {PRIMARY[0]} (full-sample, per riferimento nelle sezioni C-F):")
w(f"  n={baseline_res['n']}  ret={baseline_res['ret']:+.2f}%  mdd={baseline_res['mdd']:.2f}%  "
  f"n_dropped_none={baseline_res['n_dropped_none']}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Slippage sensitivity
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("SECTION C — Slippage sensitivity (bps aggiuntivi su fill TP/SL)")
w(SEP)
w(f"  {'Slippage':>10}  {'n':>6}  {'Ret%':>8}  {'MDD%':>7}  {'pp':>6}")
for slip_bps in [0, 2, 5, 10]:
    res = run_bt(primary_evs, slippage_pct=slip_bps/10000.0)
    mc = mc_summary(res["net_pnls"])
    w(f"  {slip_bps:>7}bps  {res['n']:>6}  {res['ret']:>+7.2f}%  {res['mdd']:>6.2f}%  "
      f"{mc['p_profit']:>6.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — No-touch trade handling
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("SECTION D — Gestione trade 'no-touch' (né TP né SL entro max_hold)")
w(SEP)
res_drop  = run_bt(primary_evs, force_close_notouch=False)
res_force = run_bt(primary_evs, force_close_notouch=True)
mc_drop  = mc_summary(res_drop["net_pnls"])
mc_force = mc_summary(res_force["net_pnls"])
w(f"  {'Modalità':<28}  {'n':>6}  {'Ret%':>8}  {'MDD%':>7}  {'pp':>6}")
w(f"  {'Scarta (comportamento attuale)':<28}  {res_drop['n']:>6}  {res_drop['ret']:>+7.2f}%  "
  f"{res_drop['mdd']:>6.2f}%  {mc_drop['p_profit']:>6.3f}")
w(f"  {'Forza chiusura a mercato':<28}  {res_force['n']:>6}  {res_force['ret']:>+7.2f}%  "
  f"{res_force['mdd']:>6.2f}%  {mc_force['p_profit']:>6.3f}")
w(f"  Trade coinvolti: {res_drop['n_dropped_none']} "
  f"({res_drop['n_dropped_none']/len(primary_evs):.1%} del totale)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Position sizing reality check
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("SECTION E — Verifica realistica del position sizing")
w(SEP)
cap_track = INIT_CAP
notional_pcts = []
btc_qty = []
for ev in primary_evs:
    ep, a = ev["ep"], ev["a"]
    lev = min(max(abs(ev["tp"]-ep)/ep, abs(ev["sl"]-ep)/ep), MAX_LEV)
    risk = cap_track * RISK_PCT
    shares = (risk * lev) / a
    notional = shares * ep
    notional_pcts.append(notional / cap_track * 100)
    btc_qty.append(shares)
notional_pcts = np.array(notional_pcts)
w(f"  Notional implicito come % del capitale corrente (N={len(notional_pcts)} trade):")
for pct in [50, 75, 90, 95, 99, 100]:
    label = "max" if pct == 100 else f"p{pct}"
    w(f"    {label:>5}: {np.percentile(notional_pcts, pct) if pct<100 else notional_pcts.max():.1f}%")
w(f"\n  Interpretazione: un notional-implicito mediano di "
  f"{np.percentile(notional_pcts,50):.0f}% del capitale corrente equivale a leva reale "
  f"~{np.percentile(notional_pcts,50)/100:.2f}x; il {np.percentile(notional_pcts,95):.0f}° "
  f"percentile (p95) implica leva ~{np.percentile(notional_pcts,95)/100:.2f}x sui trade "
  f"più aggressivi (tp_f/sl_f più ampi). Verificare che questi livelli siano compatibili "
  f"col margine disponibile sull'exchange target prima di tradare live.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Block-bootstrap Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("SECTION F — Monte Carlo: i.i.d. bootstrap vs block-bootstrap")
w(SEP)
pnls_df = pd.DataFrame({"net_pnl": baseline_res["net_pnls"]})
mc_iid = run_monte_carlo(pnls_df, INIT_CAP, N_SIMS, seed=42)
w(f"  {'Metodo':<20}  {'p_profit':>9}  {'p_ruin':>8}  {'p5 MDD%':>9}  {'p50 MDD%':>9}")
w(f"  {'i.i.d. bootstrap':<20}  {mc_iid['p_profit']:>9.3f}  {mc_iid['p_ruin']:>8.4f}  "
  f"{np.percentile(mc_iid['max_drawdown']*100,5):>8.2f}%  {np.percentile(mc_iid['max_drawdown']*100,50):>8.2f}%")
for block_size in [10, 25, 50]:
    mc_blk = run_monte_carlo_block(pnls_df, INIT_CAP, N_SIMS, block_size=block_size, seed=42)
    w(f"  {'block (n='+str(block_size)+')':<20}  {mc_blk['p_profit']:>9.3f}  {mc_blk['p_ruin']:>8.4f}  "
      f"{np.percentile(mc_blk['max_drawdown']*100,5):>8.2f}%  {np.percentile(mc_blk['max_drawdown']*100,50):>8.2f}%")

w(f"\n{SEP}")
w("[DONE]")

out_path = Path("reports/holdout_validation_checklist.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Pre-Paper-Trading Validation Checklist\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}")

