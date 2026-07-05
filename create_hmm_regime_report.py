#!/usr/bin/env python3
"""
create_hmm_regime_report.py
============================
HMM-based regime filter applied to A-Z24s15-ALL (validated 24H Z-score MR).

Lookahead-free design:
  · HMM fitted on IS bars only (never sees OOS data during fit)
  · OOS states decoded with causal forward algorithm (not Viterbi smoothing)
    → at bar t, P(state_t | obs_0..obs_t) — no future observations used
  · IS-scan (TP/SL optimisation) uses unfiltered base signal

Variants tested
  n_states ∈ {2, 3}  ×  feature set ∈ {v1=[lr,atr], v2=[lr,atr,vr]}
  → 4 HMM variants + 1 base (no filter)

Validation criterion: OOS ret > 0 AND MC p_profit > 0.90 AND MC p_ruin < 0.05
"""
from __future__ import annotations

import base64, io, sys, traceback, warnings
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

# ── Config (identical to create_intraday_strategy_report.py) ──────────────────
INIT_CAP   = 100_000.0
RISK_PCT   = 0.01
FEE        = 0.0004
FEE_RT_PCT = FEE * 2 * 100
MAX_LEV    = 5.0
IC_HORIZON = 8
START_YEAR = 2020
N_SIMS     = 5_000
COOLDOWN   = 4
MAX_HOLD   = 24
WARMUP     = 200

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

TP_GRID = [1.0, 2.0, 3.0, 5.0]
SL_GRID = [0.25, 0.5, 0.75, 1.0]

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("HMM Regime Filter — A-Z24s15-ALL (24H Z-score MR)")
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

# ── Precompute HMM features (causal — bar k uses data known at close of bar k) ─
# log_return[k]  = log(CL[k] / CL[k-1])  — return of bar k, known at bar k close
# atr_pct[k]     = ATR1[k] / CL[k]       — ATR1 is already shift(1), causal
# vol_ratio[k]   = VOL[k] / rolling24_mean[k-1], already shift(1) applied
LOG_RET  = np.concatenate([[0.0], np.log(CL[1:] / np.where(CL[:-1] > 0, CL[:-1], 1.0))])
ATR_PCT  = np.where(CL > 0, ATR1 / CL, 0.001)
VOL_RATIO = (VOL_s / VOL_s.rolling(24).mean().replace(0, np.nan)
             ).fillna(1.0).shift(1).values

# ── Base Z-score 24H signal (A-Z24s15-ALL) ────────────────────────────────────
_zm = CL_s.rolling(24).mean()
_zs = CL_s.rolling(24).std().replace(0, np.nan)
Z24  = ((CL_s - _zm) / _zs).fillna(0).shift(1).values
BASE_SIG = np.where(Z24 < -1.5, 1.0, np.where(Z24 > 1.5, -1.0, 0.0))
BASE_SIG[:WARMUP] = 0

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
# BACKTEST & PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ev(i, direction, tp_f, sl_f):
    d = 1 if direction == "long" else -1
    a = ATR1[i]; ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep+d*tp_f*a, sl=ep-d*sl_f*a, a=a)

def run_bt(events):
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, exppnl=0.0, net_pnls=[], cap=INIT_CAP)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, MAX_HOLD + 1):
            if i+k >= N1H: break
            hk, lk = HI[i+k], LO[i+k]
            if d == 1:
                if hk >= tp: out = "tp"; break
                if lk <= sl: out = "sl"; break
            else:
                if lk <= tp: out = "tp"; break
                if hk >= sl: out = "sl"; break
        if out == "none": continue
        pnl_r = (abs(tp-ep)/a) if out == "tp" else -(abs(sl-ep)/a)
        risk  = cap * RISK_PCT
        lev   = min(max(abs(tp-ep)/ep, abs(sl-ep)/ep), MAX_LEV)
        dpnl  = pnl_r * risk * lev
        cap += dpnl; peak = max(peak, cap)
        mdd  = min(mdd, (cap-peak)/peak)
        wins += int(out == "tp"); net_pnls.append(dpnl)
    n = len(net_pnls); wr = wins/n if n else 0.0
    ret = (cap/INIT_CAP - 1) * 100
    avg_tp = np.mean([abs(ev["tp"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    avg_sl = np.mean([abs(ev["sl"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    sln = avg_sl + FEE_RT_PCT/100; tpn = avg_tp - FEE_RT_PCT/100
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

def ic_test(sig):
    fwd  = np.log(np.roll(CL, -IC_HORIZON) / CL)
    mask = sig != 0; mask[-IC_HORIZON:] = False
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

def is_scan(sig, idx_is):
    best = (TP_GRID[0], SL_GRID[0]); best_xp = -np.inf
    for tf, sf in product(TP_GRID, SL_GRID):
        res = run_bt(make_events(sig, idx_is, tf, sf))
        if res["exppnl"] > best_xp: best_xp = res["exppnl"]; best = (tf, sf)
    return best

# ══════════════════════════════════════════════════════════════════════════════
# HMM HELPERS — strictly causal, no lookahead
# ══════════════════════════════════════════════════════════════════════════════

def build_feats(bar_indices: np.ndarray, feat_cols: list[str]) -> np.ndarray:
    """
    Assemble HMM feature matrix for given bar indices.
    All features are causal at bar k (use data ≤ k).

      log_ret[k]   = log(CL[k] / CL[k-1])       — known at close of bar k
      atr_pct[k]   = ATR1[k] / CL[k]            — ATR1 is shift(1), causal
      vol_ratio[k] = VOL[k] / rol24mean[k-1]    — shift(1) applied, causal
    """
    cols = []
    if "lr" in feat_cols:
        cols.append(LOG_RET[bar_indices])
    if "atr" in feat_cols:
        cols.append(ATR_PCT[bar_indices])
    if "vr" in feat_cols:
        cols.append(VOL_RATIO[bar_indices])
    return np.column_stack(cols).astype(float)


def hmm_causal_predict(model: hmmlib.GaussianHMM, obs: np.ndarray) -> np.ndarray:
    """
    Causal state sequence: argmax_s P(s_t | o_0, …, o_t) for each t.

    Uses the forward algorithm (log-space) — no backward pass, no smoothing.
    At each bar t, only past and current observations are used.
    This is strictly causal and has zero lookahead.
    """
    frame_lp = model._compute_log_likelihood(obs)   # (T, K)
    T, K = frame_lp.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = np.log(model.startprob_ + 1e-300) + frame_lp[0]
    log_transmat = np.log(model.transmat_ + 1e-300)
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = (logsumexp(log_alpha[t-1] + log_transmat[:, j])
                               + frame_lp[t, j])
    return np.argmax(log_alpha, axis=1)


def ranging_state(model: hmmlib.GaussianHMM) -> int:
    """
    Identify the 'ranging' (low-volatility) state from IS-fitted HMM.
    The state with the smallest variance of log_return (feature index 0)
    corresponds to a calm, range-bound market — favorable for MR trading.
    """
    # covars_ shape: (n_states, n_features, n_features) in hmmlearn 0.3.x
    return_vars = model.covars_[:, 0, 0]
    return int(np.argmin(return_vars))


def fit_hmm_on_is(idx_is: np.ndarray, feat_cols: list[str],
                  n_states: int) -> hmmlib.GaussianHMM | None:
    """
    Fit GaussianHMM on IS bars. Returns None if fitting fails.
    HMM sees ONLY IS data — no future (OOS) observations.
    """
    X = build_feats(idx_is, feat_cols)
    # Standardise features to improve convergence (fit stats from IS only)
    mu = X.mean(axis=0); std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X - mu) / std
    model = hmmlib.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=150,
        tol=1e-4,
        random_state=42,
    )
    try:
        model.fit(X_norm)
        model._hmm_mu = mu   # store IS normalisation params on model
        model._hmm_std = std
        return model
    except Exception:
        return None


def predict_oos_regime(model: hmmlib.GaussianHMM,
                       idx_oos: np.ndarray,
                       feat_cols: list[str]) -> np.ndarray:
    """
    Predict OOS regimes using IS-fitted model and causal forward algorithm.
    Normalisation uses IS statistics (stored on model) — no OOS data leaks.
    """
    X = build_feats(idx_oos, feat_cols)
    X_norm = (X - model._hmm_mu) / model._hmm_std
    return hmm_causal_predict(model, X_norm)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(variant_id: str, name: str, feat_cols: list[str] | None,
                 n_states: int | None) -> dict:
    """
    Run WFO pipeline for A-Z24s15-ALL with optional HMM regime filter.

    feat_cols=None, n_states=None → baseline (no regime filter)
    """
    use_hmm = feat_cols is not None
    sig_used = BASE_SIG.copy()   # already verified IC > 0 in prior run

    # IC on full series (informational only)
    ic, p_ic, n_ic = ic_test(sig_used)
    print(f"\n  [{variant_id}] {name}")
    print(f"    IC: {ic:+.4f}  p={p_ic:.4f}  n={n_ic:,}")

    rec = dict(
        id=variant_id, name=name,
        feat_cols=feat_cols, n_states=n_states, use_hmm=use_hmm,
        ic=ic, p_ic=p_ic, n_ic=n_ic,
        oos_n=0, oos_wr=0.0, oos_ret=0.0, oos_mdd=0.0,
        mc_p_profit=0.0, mc_p_ruin=1.0, validated=False,
        oos_pnls=[], oos_equity=[INIT_CAP],
        regime_stats=[],   # per-window regime proportions
        hmm_errors=0,
    )

    print("    WFO …", end="", flush=True)
    all_evs = []

    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue

        # IS scan always uses unfiltered base signal
        tp_f, sl_f = is_scan(sig_used, idx_is)

        if not use_hmm:
            all_evs.extend(make_events(sig_used, idx_oos, tp_f, sl_f))
            print(".", end="", flush=True)
            continue

        # ── HMM: fit on IS, predict on OOS (causal) ──────────────────────────
        model = fit_hmm_on_is(idx_is, feat_cols, n_states)
        if model is None:
            rec["hmm_errors"] += 1
            # Fallback: no filter for this window
            all_evs.extend(make_events(sig_used, idx_oos, tp_f, sl_f))
            print("e", end="", flush=True)
            continue

        r_state = ranging_state(model)
        oos_states = predict_oos_regime(model, idx_oos, feat_cols)

        # Ranging % in this OOS window
        n_ranging = int((oos_states == r_state).sum())
        rec["regime_stats"].append({
            "oos_start": oo_s,
            "pct_ranging": n_ranging / len(oos_states) * 100,
            "ranging_state": r_state,
        })

        # Build regime-filtered signal: zero out bars in trending regime
        sig_filtered = sig_used.copy()
        for pos, k in enumerate(idx_oos):
            if oos_states[pos] != r_state:
                sig_filtered[k] = 0.0

        all_evs.extend(make_events(sig_filtered, idx_oos, tp_f, sl_f))
        print(".", end="", flush=True)

    print()

    if not all_evs:
        print("    No OOS events — skip"); return rec

    res   = run_bt(all_evs)
    mc    = mc_summary(res["net_pnls"])
    valid = (res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05)

    eq = [INIT_CAP]; cap_eq = INIT_CAP
    for p in res["net_pnls"]: cap_eq += p; eq.append(cap_eq)

    # Average ranging % across windows
    avg_rng = (np.mean([s["pct_ranging"] for s in rec["regime_stats"]])
               if rec["regime_stats"] else 100.0)

    print(f"    OOS: n={res['n']}  wr={res['wr']:.1%}  "
          f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    print(f"    MC:  p_profit={mc['p_profit']:.3f}  p_ruin={mc['p_ruin']:.3f}")
    if use_hmm:
        print(f"    Regime: {avg_rng:.1f}% bars in ranging state"
              f"  | HMM errors: {rec['hmm_errors']}")
    print(f"    {'VALIDATED ✅' if valid else 'NOT VALIDATED ❌'}")

    rec.update(oos_n=res["n"], oos_wr=res["wr"], oos_ret=res["ret"],
               oos_mdd=res["mdd"], mc_p_profit=mc["p_profit"],
               mc_p_ruin=mc["p_ruin"], validated=valid,
               oos_pnls=res["net_pnls"], oos_equity=eq)
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# RUN VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    ("BASE",      "A-Z24s15-ALL — No filter (base)",               None,           None),
    ("HMM-2s-v1", "HMM 2 stati · feat=[lr,atr] · forward causal",  ["lr","atr"],   2),
    ("HMM-3s-v1", "HMM 3 stati · feat=[lr,atr] · forward causal",  ["lr","atr"],   3),
    ("HMM-2s-v2", "HMM 2 stati · feat=[lr,atr,vr] · forward causal",["lr","atr","vr"], 2),
    ("HMM-3s-v2", "HMM 3 stati · feat=[lr,atr,vr] · forward causal",["lr","atr","vr"], 3),
]

print(f"\n{SEP2}")
print("Running pipeline variants …")
print(SEP2)

ALL_RESULTS = []
for vid, name, feat_cols, n_states in VARIANTS:
    rec = run_pipeline(vid, name, feat_cols, n_states)
    ALL_RESULTS.append(rec)

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\nSUMMARY\n{SEP2}")
print(f"  {'ID':<14}  {'OOS n':>6}  {'WR':>6}  {'Ret':>7}  {'MDD':>7}  "
      f"{'P(profit)':>9}  {'P(ruin)':>8}  Status")
print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  "
      f"{'-'*9}  {'-'*8}  {'-'*12}")
for r in ALL_RESULTS:
    flag = "✅ VALID" if r["validated"] else ("⚠  OOS+" if r["oos_ret"] > 0 else "✗  FAIL")
    print(f"  {r['id']:<14}  {r['oos_n']:>6}  {r['oos_wr']:>6.1%}  "
          f"{r['oos_ret']:>+6.1f}%  {r['oos_mdd']:>6.1f}%  "
          f"{r['mc_p_profit']:>9.3f}  {r['mc_p_ruin']:>8.3f}  {flag}")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _equity_chart(recs: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    colors = [_TEXT, _ACC, _GRN, _YEL, _ORG]
    lws    = [1.0, 1.8, 1.8, 1.8, 1.8]
    for idx, r in enumerate(recs):
        eq = r["oos_equity"]
        if len(eq) < 2: continue
        ax.plot(range(len(eq)), eq,
                color=colors[idx % len(colors)],
                lw=lws[idx],
                label=r["id"],
                alpha=0.9)
    ax.axhline(INIT_CAP, color=_GRID, ls="--", lw=0.8)
    ax.set_ylabel("Equity ($)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.15, color=_GRID)
    ax.set_title("OOS Equity — base vs HMM filtri", color=_TEXT, fontsize=10)
    ax.legend(labelcolor=_TEXT, facecolor=_BG, fontsize=8, loc="upper left")
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _bar_chart() -> str:
    ids  = [r["id"] for r in ALL_RESULTS]
    rets = [r["oos_ret"] for r in ALL_RESULTS]
    pps  = [r["mc_p_profit"] for r in ALL_RESULTS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.8), facecolor=_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG); ax.tick_params(colors=_TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
        ax.grid(axis="y", alpha=0.2, color=_GRID)

    bar_c1 = [_GRN if v > 0 else _RED for v in rets]
    ax1.bar(range(len(ids)), rets, color=bar_c1, edgecolor=_GRID, lw=0.5, zorder=3)
    ax1.axhline(0, color=_TEXT, lw=0.7, zorder=2)
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("OOS Return (%)", color=_TEXT, fontsize=9)
    ax1.set_title("OOS Return", color=_TEXT, fontsize=10)

    bar_c2 = [_GRN if v > 0.90 else (_YEL if v > 0.60 else _RED) for v in pps]
    ax2.bar(range(len(ids)), pps, color=bar_c2, edgecolor=_GRID, lw=0.5, zorder=3)
    ax2.axhline(0.90, color=_YEL, ls="--", lw=0.9, label="soglia 0.90", zorder=2)
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("MC P(profit)", color=_TEXT, fontsize=9)
    ax2.set_title("MC P(profit)", color=_TEXT, fontsize=10)
    ax2.legend(labelcolor=_TEXT, facecolor=_BG, fontsize=8)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout(pad=1.5)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _regime_timeline_chart(r: dict) -> str:
    """Ranging% per OOS window for an HMM variant."""
    stats = r.get("regime_stats", [])
    if not stats: return ""
    dates = [str(s["oos_start"])[:7] for s in stats]
    pcts  = [s["pct_ranging"] for s in stats]

    fig, ax = plt.subplots(figsize=(11, 2.4), facecolor=_BG)
    ax.set_facecolor(_BG)
    colors = [_GRN if p >= 50 else _RED for p in pcts]
    ax.bar(range(len(dates)), pcts, color=colors, edgecolor=_GRID, lw=0.4)
    ax.axhline(50, color=_YEL, ls="--", lw=0.8, label="50%")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% bar in ranging", color=_TEXT, fontsize=8)
    ax.set_title(f"{r['id']} — % OOS bars in ranging state (per finestra)", color=_TEXT, fontsize=9)
    ax.set_ylim(0, 100)
    ax.tick_params(colors=_TEXT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(axis="y", alpha=0.15, color=_GRID)
    ax.legend(labelcolor=_TEXT, facecolor=_BG, fontsize=7)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def badge(r):
    if r["validated"]:         return '<span class="badge green">VALIDATED ✅</span>'
    if r["oos_ret"] > 0:       return '<span class="badge yellow">OOS+ ⚠</span>'
    return                            '<span class="badge red">FAIL ✗</span>'


def _hmm_note(r: dict) -> str:
    if not r["use_hmm"]: return ""
    stats = r.get("regime_stats", [])
    avg   = np.mean([s["pct_ranging"] for s in stats]) if stats else 0.0
    return (f"<p class='desc'>"
            f"Feature: [{', '.join(r['feat_cols'])}] &nbsp;|&nbsp; "
            f"Stati: {r['n_states']} &nbsp;|&nbsp; "
            f"Ranging medio OOS: <strong>{avg:.1f}%</strong> delle barre &nbsp;|&nbsp; "
            f"Errori HMM (finestre fallite): {r['hmm_errors']}"
            f"</p>")


CSS = f"""
  :root{{--bg:{_BG};--card:{_CARD};--grid:{_GRID};--text:{_TEXT};
         --acc:{_ACC};--grn:{_GRN};--red:{_RED};--yel:{_YEL};--org:{_ORG};}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);
        font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;
        line-height:1.5;padding:24px;}}
  h1{{color:var(--acc);font-size:22px;margin-bottom:4px;}}
  h2{{color:var(--acc);font-size:15px;margin:0 0 8px;}}
  h3{{color:var(--yel);font-size:13px;margin:12px 0 6px;}}
  .sub{{color:#777;font-size:12px;margin-bottom:24px;}}
  .kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px;}}
  .kpi{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
         padding:12px 18px;min-width:130px;}}
  .kpi .val{{font-size:26px;font-weight:700;color:var(--acc);}}
  .kpi .lbl{{font-size:11px;color:#777;}}
  .card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
          padding:18px;margin-bottom:20px;}}
  .hmm-card{{border-left:3px solid var(--acc);}}
  .base-card{{border-left:3px solid #555;}}
  .desc{{color:#888;font-size:12px;margin-bottom:10px;}}
  .tbl-wrap{{overflow-x:auto;}}
  .tbl{{border-collapse:collapse;width:100%;min-width:700px;}}
  .tbl th{{background:var(--grid);color:var(--acc);text-align:left;
            padding:6px 10px;font-size:12px;white-space:nowrap;}}
  .tbl td{{padding:5px 10px;border-bottom:1px solid var(--grid);
            font-size:12px;white-space:nowrap;}}
  .tbl tr:hover td{{background:var(--grid);}}
  .tbl tr.base td{{background:#18191f;}}
  .green{{color:var(--grn)!important;}} .red{{color:var(--red)!important;}}
  .yellow{{color:var(--yel)!important;}} .dim{{color:#555!important;}}
  .badge{{display:inline-block;border-radius:4px;padding:2px 7px;
           font-size:11px;font-weight:600;}}
  .badge.green{{background:#1b3a1e;color:var(--grn);}}
  .badge.red{{background:#3a1a1a;color:var(--red);}}
  .badge.yellow{{background:#2d2a00;color:var(--yel);}}
  .note-box{{background:#111824;border:1px solid #2a3040;border-radius:6px;
              padding:12px 16px;margin-bottom:16px;font-size:12px;color:#9ab;}}
  .note-box code{{background:var(--grid);padding:1px 4px;border-radius:3px;}}
  img{{display:block;margin-bottom:12px;border-radius:6px;max-width:100%;}}
  code{{font-size:11px;background:var(--grid);padding:1px 4px;border-radius:3px;}}
  .section-sep{{border:none;border-top:1px solid var(--grid);margin:24px 0;}}
"""

# Summary table rows
tbl_rows = ""
for r in ALL_RESULTS:
    rc  = "green"  if r["oos_ret"] > 0 else "red"
    pc  = "green"  if r["mc_p_profit"] > 0.90 else ("yellow" if r["mc_p_profit"] > 0.60 else "red")
    qc  = "green"  if r["mc_p_ruin"] < 0.05 else "red"
    cls = "base"   if not r["use_hmm"] else ""
    avg_rng = (np.mean([s["pct_ranging"] for s in r["regime_stats"]])
               if r["regime_stats"] else "—")
    avg_rng_s = f"{avg_rng:.1f}%" if isinstance(avg_rng, float) else avg_rng
    tbl_rows += (
        f"<tr class='{cls}'>"
        f"<td><code>{r['id']}</code></td>"
        f"<td>{r['name']}</td>"
        f"<td>{r['oos_n']}</td>"
        f"<td class='{'green' if r['oos_wr'] > 0.5 else ''}'>{r['oos_wr']:.1%}</td>"
        f"<td class='{rc}'>{r['oos_ret']:+.1f}%</td>"
        f"<td class='red'>{r['oos_mdd']:.1f}%</td>"
        f"<td class='{pc}'>{r['mc_p_profit']:.3f}</td>"
        f"<td class='{qc}'>{r['mc_p_ruin']:.3f}</td>"
        f"<td>{avg_rng_s}</td>"
        f"<td>{badge(r)}</td>"
        f"</tr>"
    )

# Per-variant cards
variant_cards = ""
for r in ALL_RESULTS:
    cls = "base-card" if not r["use_hmm"] else "hmm-card"
    eq_img    = ""
    rng_img   = ""
    if r["oos_n"] > 0:
        eq_img  = _equity_chart([r])
        rng_img = _regime_timeline_chart(r)
    variant_cards += f"""
    <div class="card {cls}">
      <h2>{r['id']} — {r['name']}</h2>
      {_hmm_note(r)}
      {eq_img}
      {rng_img}
    </div>"""

# Build HTML
equity_overlay = _equity_chart(ALL_RESULTS)
bar_img        = _bar_chart()

n_val = sum(1 for r in ALL_RESULTS if r["validated"])
best  = max(ALL_RESULTS, key=lambda r: r["oos_ret"])

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HMM Regime Filter — A-Z24s15-ALL</title>
<style>{CSS}</style>
</head>
<body>
<h1>HMM Regime Filter — A-Z24s15-ALL (24H Z-score MR)</h1>
<p class="sub">
  {len(ALL_RESULTS)} varianti · IC {IC_HORIZON}H · WFO 6m/2m/2m ·
  MC N={N_SIMS:,} · Cooldown {COOLDOWN}H · Max hold {MAX_HOLD}H &nbsp;|&nbsp;
  {IDX1H[0].date()} – {IDX1H[-1].date()} · {N1H:,} bar 1H
</p>

<div class="note-box">
  <strong>Lookahead-free design</strong><br>
  • HMM fittato <em>esclusivamente</em> su barre IS (mai vede OOS durante il training).<br>
  • Predizione OOS con <strong>forward algorithm</strong> (causal):
    P(stato_t | obs_0…obs_t) — nessuna osservazione futura usata.<br>
  • IS-scan (TP/SL) eseguito sul segnale base non filtrato.<br>
  • Normalizzazione feature: media/std calcolati su IS, applicati a OOS — nessun data leak.<br>
  • Feature: <code>log_return[k] = log(CL[k]/CL[k-1])</code> (ritorno barra corrente),
    <code>atr_pct[k] = ATR14[k-1]/CL[k]</code> (shift(1)),
    <code>vol_ratio[k] = VOL[k]/rol24[k-1]</code> (shift(1)).<br>
  • Stato "ranging" identificato su IS: stato con minima varianza del log_return.
</div>

<div class="kpi-row">
  <div class="kpi"><div class="val">{len(ALL_RESULTS)}</div>
    <div class="lbl">Varianti testate</div></div>
  <div class="kpi">
    <div class="val" style="color:{'var(--grn)' if n_val>0 else 'var(--red)'}">{n_val}</div>
    <div class="lbl">Validate</div></div>
  <div class="kpi"><div class="val" style="font-size:18px">{best['oos_ret']:+.1f}%</div>
    <div class="lbl">Miglior OOS ret ({best['id']})</div></div>
  <div class="kpi"><div class="val" style="font-size:18px">{best['mc_p_profit']:.3f}</div>
    <div class="lbl">MC P(profit) miglior var.</div></div>
</div>

<div class="card">
  <h2>Confronto OOS — Return e P(profit)</h2>
  {bar_img}
</div>

<div class="card">
  <h2>OOS Equity Overlay — tutte le varianti</h2>
  {equity_overlay}
</div>

<div class="card">
  <h2>Tabella Riassuntiva</h2>
  <p class="desc">Riga grigia = baseline senza filtro HMM. % Ranging = % barre OOS classificate come ranging.</p>
  <div class="tbl-wrap"><table class="tbl">
    <thead><tr>
      <th>ID</th><th>Variante</th><th>OOS n</th><th>WR</th>
      <th>OOS Ret</th><th>MDD</th><th>P(profit)</th><th>P(ruin)</th>
      <th>% Ranging</th><th>Status</th>
    </tr></thead>
    <tbody>{tbl_rows}</tbody>
  </table></div>
</div>

{variant_cards}

<hr class="section-sep">
<p style="color:#444;font-size:11px;text-align:center;">
  BTCUSDT perpetual futures · Binance Vision 2020–2026 · 1H bars ·
  IC {IC_HORIZON}H Spearman · WFO 6m IS / 2m OOS / 2m step ·
  MC N={N_SIMS:,} · Risk 1%/trade · Fee 0.08% RT ·
  HMM: GaussianHMM (hmmlearn 0.3) · covariance=diag · forward algorithm (causal)
</p>
</body>
</html>"""

out = Path("reports/report_hmm_regime.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n[DONE] Report → {out}  ({out.stat().st_size//1024} KB)")
