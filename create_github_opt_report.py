"""
create_github_opt_report.py
============================
Ottimizzazioni di H01 (RSI MR) e H14 (Z-score MR) — le 2 strategie con OOS>0.

Varianti testate:
  OPT-A : RSI threshold scan {20/80, 25/75, 30/70}
  OPT-B : Z-score window × threshold scan  {30,50,100,200} × {1.5,2.0,2.5}σ
  OPT-C : Filtro ADX<20 / ADX<25 applicato a H01 e H14
  OPT-D : Filtro volatilità ATR% [0.3%–3.0%] applicato a H01 e H14
  OPT-E : Segnale combo H01+H14 (entrambi concordi sulla direzione)

Pipeline: IC (Spearman 16H) → WFO (6m IS / 2m OOS / 2m step) → MC (N=5000)
Criterio: ret_oos > 0  AND  MC p_profit > 0.90  AND  MC p_ruin < 0.05
"""
from __future__ import annotations

import base64, io, sys, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategy.data_fetcher import (
    fetch_extended_data,
    fetch_binance_vision_taker_flow,
    fetch_binance_vision_funding,
)
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import run_monte_carlo

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP    = 100_000.0
RISK_PCT    = 0.01
FEE         = 0.0004
FEE_RT_PCT  = FEE * 2 * 100
MAX_LEV     = 5.0
IC_HORIZON  = 16
START_YEAR  = 2020
N_SIMS      = 5_000
COOLDOWN    = 8
MAX_HOLD    = 96
WARMUP      = 200

WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2

TP_GRID  = [1.0, 2.0, 3.0, 5.0]
SL_GRID  = [0.25, 0.5, 0.75, 1.0]

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("GitHub Strategies — Optimization Report")
print("  H01 (RSI MR) e H14 (Z-score MR) — 22 varianti")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw   = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h  = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
print(f"  1H: {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

CL   = df1h["close"].values.astype(float)
HI   = df1h["high"].values.astype(float)
LO   = df1h["low"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0,
                df1h["atr_14"].shift(1).values, 1.0)

# ── Indicators (causal, shift(1)) ─────────────────────────────────────────────
CL_s = pd.Series(CL, index=IDX1H)

# RSI(14)
_d   = CL_s.diff()
_ag  = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
_al  = (-_d).clip(lower=0).ewm(com=13, adjust=False).mean()
_rsi = (100 - 100 / (_ag / _al.replace(0, np.nan))).fillna(50.0)
RSI  = _rsi.shift(1).values

# ADX (from add_indicators)
ADX  = df1h["adx"].shift(1).values

# ATR% (annualized)
ATR_PCT = df1h["atr_pct"].shift(1).values   # atr_14 / close * 100

# Pre-compute Z-scores for all windows at once
ZSCORE = {}
for win in [30, 50, 100, 200]:
    zm = CL_s.rolling(win).mean()
    zs = CL_s.rolling(win).std().replace(0, np.nan)
    ZSCORE[win] = ((CL_s - zm) / zs).fillna(0).shift(1).values

# ── WFO windows ───────────────────────────────────────────────────────────────
def wf_dates(IDX):
    t0 = IDX[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > IDX[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 = t0 + pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
print(f"[WFO] {len(WF_WINDOWS)} windows\n")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ev(i, direction, tp_f, sl_f):
    d = 1 if direction == "long" else -1
    a = ATR1[i]; ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep + d*tp_f*a, sl=ep - d*sl_f*a, a=a)

def run_bt(events):
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, exppnl=0.0,
                    net_pnls=[], cap=INIT_CAP)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0
    net_pnls = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"],ev["d"],ev["ep"],ev["tp"],ev["sl"],ev["a"]
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
        pnl_r = (abs(tp-ep)/a) if out=="tp" else -(abs(sl-ep)/a)
        risk  = cap * RISK_PCT
        lev   = min(max(abs(tp-ep)/ep, abs(sl-ep)/ep), MAX_LEV)
        dpnl  = pnl_r * risk * lev
        cap  += dpnl; peak = max(peak, cap); mdd = min(mdd,(cap-peak)/peak)
        wins += int(out=="tp"); net_pnls.append(dpnl)
    n  = len(net_pnls)
    wr = wins/n if n else 0.0
    ret = (cap/INIT_CAP - 1) * 100
    avg_tp = np.mean([abs(ev["tp"]-ev["ep"])/ev["ep"] for ev in events])*100
    avg_sl = np.mean([abs(ev["sl"]-ev["ep"])/ev["ep"] for ev in events])*100
    sln = avg_sl + FEE_RT_PCT/100; tpn = avg_tp - FEE_RT_PCT/100
    ev_adj = wr*tpn - (1-wr)*sln
    return dict(n=n, wr=wr, ret=ret, mdd=mdd*100, exppnl=ev_adj,
                net_pnls=net_pnls, cap=cap)

def mc_summary(pnls):
    if len(pnls) < 5:
        return dict(p_profit=0.0, p_ruin=1.0)
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
    best_xp = -np.inf; best = (TP_GRID[0], SL_GRID[0])
    for tf, sf in product(TP_GRID, SL_GRID):
        evs = make_events(sig, idx_is, tf, sf)
        res = run_bt(evs)
        if res["exppnl"] > best_xp:
            best_xp = res["exppnl"]; best = (tf, sf)
    return best

def run_pipeline(variant_id: str, name: str, sig: np.ndarray) -> dict:
    """Full IC → WFO → MC pipeline for one variant. Returns result dict."""
    print(f"  {SEP}")
    n_long  = int((sig > 0).sum())
    n_short = int((sig < 0).sum())
    n_sig   = n_long + n_short
    print(f"  [{variant_id}] {name}")
    print(f"    Signals: {n_sig:,}  (L={n_long:,} / S={n_short:,})")

    ic, p_ic, n_ic = ic_test(sig)
    ic_pass = (abs(ic) > 0) and (p_ic < 0.05)
    print(f"    IC: {ic:+.4f}  p={p_ic:.4f}  n={n_ic:,}  "
          f"→ {'PASS ✓' if ic_pass else 'FAIL ✗'}")

    rec = dict(
        id=variant_id, name=name,
        n_sig=n_sig, n_long=n_long, n_short=n_short,
        ic=ic, p_ic=p_ic, n_ic=n_ic, ic_pass=ic_pass,
        wfo_run=False, oos_n=0, oos_wr=0.0, oos_ret=0.0, oos_mdd=0.0,
        mc_p_profit=0.0, mc_p_ruin=1.0, validated=False,
        oos_pnls=[], oos_equity=[INIT_CAP],
    )

    if not ic_pass:
        return rec

    sig_used = sig if ic > 0 else -sig

    # WFO
    print(f"    WFO …", end="", flush=True)
    rec["wfo_run"] = True
    all_evs: list = []

    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue
        tp_f, sl_f = is_scan(sig_used, idx_is)
        all_evs.extend(make_events(sig_used, idx_oos, tp_f, sl_f))
        print(".", end="", flush=True)
    print()

    if not all_evs:
        print("    No OOS events — skip")
        return rec

    res = run_bt(all_evs)
    mc  = mc_summary(res["net_pnls"])
    valid = (res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05)

    # Rebuild equity curve
    eq = [INIT_CAP]
    cap = INIT_CAP
    for p in res["net_pnls"]:
        cap += p; eq.append(cap)

    print(f"    OOS: n={res['n']}  wr={res['wr']:.1%}  "
          f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    print(f"    MC:  p_profit={mc['p_profit']:.3f}  p_ruin={mc['p_ruin']:.3f}")
    print(f"    {'VALIDATED ✅' if valid else 'NOT VALIDATED ❌'}")

    rec.update(
        oos_n=res["n"], oos_wr=res["wr"], oos_ret=res["ret"], oos_mdd=res["mdd"],
        mc_p_profit=mc["p_profit"], mc_p_ruin=mc["p_ruin"],
        validated=valid, oos_pnls=res["net_pnls"], oos_equity=eq,
    )
    return rec

# ══════════════════════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_rsi_sig(lo_thr: float, hi_thr: float) -> np.ndarray:
    """MR: long when RSI < lo_thr, short when RSI > hi_thr."""
    s = np.where(RSI < lo_thr, 1.0, np.where(RSI > hi_thr, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def make_zscore_sig(win: int, thr: float) -> np.ndarray:
    """MR: long when Z < -thr, short when Z > +thr."""
    z = ZSCORE[win]
    s = np.where(z < -thr, 1.0, np.where(z > thr, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def apply_adx_filter(sig: np.ndarray, adx_thr: float) -> np.ndarray:
    """Zero out signals when ADX >= adx_thr (trending market)."""
    out = sig.copy()
    out[ADX >= adx_thr] = 0
    return out

def apply_vol_filter(sig: np.ndarray,
                     lo_pct: float = 0.3, hi_pct: float = 3.0) -> np.ndarray:
    """Keep signals only when ATR% is in [lo_pct, hi_pct] (moderate vol)."""
    out = sig.copy()
    out[(ATR_PCT < lo_pct) | (ATR_PCT > hi_pct)] = 0
    return out

# ── OPT-A: RSI threshold scan ─────────────────────────────────────────────────
print(f"\n{SEP2}\nOPT-A — RSI Threshold Scan\n{SEP2}")
RSI_THRESHOLDS = [(20, 80), (25, 75), (30, 70)]
VARIANTS_A = []
for lo, hi in RSI_THRESHOLDS:
    sig = make_rsi_sig(lo, hi)
    rec = run_pipeline(f"A-RSI{lo}/{hi}", f"RSI MR ({lo}/{hi})", sig)
    VARIANTS_A.append(rec)

# ── OPT-B: Z-score window × threshold scan ───────────────────────────────────
print(f"\n{SEP2}\nOPT-B — Z-score Window × Threshold Scan\n{SEP2}")
ZSCORE_WINDOWS = [30, 50, 100, 200]
ZSCORE_THRS    = [1.5, 2.0, 2.5]
VARIANTS_B = []
for win in ZSCORE_WINDOWS:
    for thr in ZSCORE_THRS:
        sig = make_zscore_sig(win, thr)
        rec = run_pipeline(f"B-Z{win}s{thr}", f"Z-score MR (win={win}, thr={thr}σ)", sig)
        VARIANTS_B.append(rec)

# ── OPT-C: ADX filter on H01 and H14 ─────────────────────────────────────────
print(f"\n{SEP2}\nOPT-C — ADX Regime Filter\n{SEP2}")
base_rsi = make_rsi_sig(30, 70)
base_z14  = make_zscore_sig(50, 2.0)    # original H14 params
VARIANTS_C = []
for base_sig, base_name, tag in [
    (base_rsi, "RSI MR (30/70)", "H01"),
    (base_z14, "Z-score MR (50,2σ)", "H14"),
]:
    for adx_thr in [20, 25]:
        sig = apply_adx_filter(base_sig, adx_thr)
        rec = run_pipeline(f"C-{tag}+ADX{adx_thr}",
                           f"{base_name} + ADX<{adx_thr}", sig)
        VARIANTS_C.append(rec)

# ── OPT-D: Volatility filter on H01 and H14 ──────────────────────────────────
print(f"\n{SEP2}\nOPT-D — Volatility (ATR%) Filter\n{SEP2}")
# Narrow (0.5–2.0%) and wider (0.3–3.0%)
VOL_RANGES = [(0.5, 2.0), (0.3, 3.0)]
VARIANTS_D = []
for base_sig, base_name, tag in [
    (base_rsi, "RSI MR (30/70)", "H01"),
    (base_z14, "Z-score MR (50,2σ)", "H14"),
]:
    for lo_v, hi_v in VOL_RANGES:
        sig = apply_vol_filter(base_sig, lo_v, hi_v)
        rec = run_pipeline(f"D-{tag}+VOL{lo_v}-{hi_v}",
                           f"{base_name} + ATR%∈[{lo_v},{hi_v}]", sig)
        VARIANTS_D.append(rec)

# ── OPT-E: Combo H01 + H14 (entrambi concordi) ───────────────────────────────
print(f"\n{SEP2}\nOPT-E — Combo H01+H14\n{SEP2}")

def make_combo_sig(rsi_lo, rsi_hi, z_win, z_thr) -> np.ndarray:
    rsi_long  = RSI < rsi_lo
    rsi_short = RSI > rsi_hi
    z = ZSCORE[z_win]
    z_long  = z < -z_thr
    z_short = z > z_thr
    s = np.where(rsi_long  & z_long,  1.0,
         np.where(rsi_short & z_short, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

VARIANTS_E = []
# Strict combo: RSI 30/70 + Z50 2.0σ
sig_combo1 = make_combo_sig(30, 70, 50, 2.0)
rec = run_pipeline("E-COMBO-strict",
                   "Combo RSI(30/70) + Z(50bar,2σ)", sig_combo1)
VARIANTS_E.append(rec)

# Relaxed combo: RSI 35/65 + Z50 1.5σ (more signals)
sig_combo2 = make_combo_sig(35, 65, 50, 1.5)
rec = run_pipeline("E-COMBO-relax",
                   "Combo RSI(35/65) + Z(50bar,1.5σ)", sig_combo2)
VARIANTS_E.append(rec)

# Combo + ADX<25 (all three filters)
sig_combo3 = apply_adx_filter(make_combo_sig(30, 70, 50, 2.0), 25)
rec = run_pipeline("E-COMBO+ADX25",
                   "Combo RSI+Z + ADX<25", sig_combo3)
VARIANTS_E.append(rec)

# ══════════════════════════════════════════════════════════════════════════════
# ALL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
ALL_RESULTS = VARIANTS_A + VARIANTS_B + VARIANTS_C + VARIANTS_D + VARIANTS_E

n_ic   = sum(1 for r in ALL_RESULTS if r["ic_pass"])
n_wfo  = sum(1 for r in ALL_RESULTS if r["wfo_run"])
n_val  = sum(1 for r in ALL_RESULTS if r["validated"])

print(f"\n{SEP2}\nSUMMARY ({len(ALL_RESULTS)} varianti)\n{SEP2}")
print(f"  IC pass   : {n_ic}/{len(ALL_RESULTS)}")
print(f"  WFO run   : {n_wfo}/{len(ALL_RESULTS)}")
print(f"  Validated : {n_val}/{len(ALL_RESULTS)}")
print()

# Sort by OOS ret descending (WFO variants first)
wfo_ran = [r for r in ALL_RESULTS if r["wfo_run"]]
wfo_ran.sort(key=lambda r: r["oos_ret"], reverse=True)

for r in wfo_ran:
    flag = "✅" if r["validated"] else ("⚠" if r["oos_ret"] > 0 else "✗")
    print(f"  {flag} [{r['id']:<18}] {r['name']:<38}  "
          f"ret={r['oos_ret']:+5.1f}%  "
          f"pp={r['mc_p_profit']:.2f}  "
          f"pr={r['mc_p_ruin']:.3f}")
for r in ALL_RESULTS:
    if not r["wfo_run"]:
        print(f"  ✗ [{r['id']:<18}] {r['name']:<38}  IC FAIL (p={r['p_ic']:.3f})")

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
    if len(eq) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=_BG)
    ax.set_facecolor(_BG)
    color = _GRN if eq[-1] >= INIT_CAP else _RED
    ax.plot(range(len(eq)), eq, color=color, lw=1.5)
    ax.axhline(INIT_CAP, color=_GRID, ls="--", lw=0.8)
    ax.set_ylabel("Equity ($)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.15, color=_GRID)
    ax.set_title(f"{rec['id']} — OOS equity", color=_TEXT, fontsize=9)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:820px;">'


def _overview_chart() -> str:
    ran = [r for r in ALL_RESULTS if r["wfo_run"]]
    if not ran:
        return ""
    ran_s = sorted(ran, key=lambda r: r["oos_ret"], reverse=True)
    ids   = [r["id"] for r in ran_s]
    rets  = [r["oos_ret"] for r in ran_s]
    pps   = [r["mc_p_profit"] for r in ran_s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor=_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
        ax.grid(axis="y", alpha=0.2, color=_GRID)

    colors_r = [_GRN if v > 0 else _RED for v in rets]
    ax1.bar(range(len(ids)), rets, color=colors_r, edgecolor=_GRID, lw=0.4)
    ax1.axhline(0, color=_TEXT, lw=0.7)
    ax1.set_xticks(range(len(ids))); ax1.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("OOS Return (%)", color=_TEXT, fontsize=9)
    ax1.set_title("OOS Return per variante", color=_TEXT, fontsize=10)

    colors_p = [_GRN if v > 0.90 else (_YEL if v > 0.60 else _RED) for v in pps]
    ax2.bar(range(len(ids)), pps, color=colors_p, edgecolor=_GRID, lw=0.4)
    ax2.axhline(0.90, color=_YEL, ls="--", lw=0.9, label="soglia 0.90")
    ax2.set_xticks(range(len(ids))); ax2.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("MC P(profit)", color=_TEXT, fontsize=9)
    ax2.set_title("MC P(profit) per variante", color=_TEXT, fontsize=10)
    ax2.legend(labelcolor=_TEXT, facecolor=_BG, fontsize=8)

    fig.tight_layout(pad=1.5)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _heatmap_b() -> str:
    """Z-score parameter heatmap (OOS ret)."""
    wins = ZSCORE_WINDOWS
    thrs = ZSCORE_THRS
    data = np.zeros((len(wins), len(thrs)))
    for r in VARIANTS_B:
        if not r["wfo_run"]: continue
        # parse win and thr from id: B-Z{win}s{thr}
        tag = r["id"].replace("B-Z", "")
        for j, thr in enumerate(thrs):
            stag = f"s{thr}"
            if stag in tag:
                win = int(tag.split("s")[0])
                if win in wins:
                    i = wins.index(win); data[i, j] = r["oos_ret"]

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=_BG)
    ax.set_facecolor(_BG)
    vmax = max(abs(data.max()), abs(data.min()), 0.1)
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(thrs))); ax.set_xticklabels([f"{t}σ" for t in thrs],
                                                          color=_TEXT, fontsize=9)
    ax.set_yticks(range(len(wins))); ax.set_yticklabels([f"{w}bar" for w in wins],
                                                          color=_TEXT, fontsize=9)
    ax.set_xlabel("Soglia z-score", color=_TEXT, fontsize=9)
    ax.set_ylabel("Finestra rolling", color=_TEXT, fontsize=9)
    ax.set_title("OPT-B: OOS Return (%) — Z-score parameter grid", color=_TEXT, fontsize=9)
    for i in range(len(wins)):
        for j in range(len(thrs)):
            val = data[i, j]
            ax.text(j, i, f"{val:+.1f}%" if val != 0 else "FAIL",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(val) < vmax * 0.6 else _TEXT)
    cb = fig.colorbar(im, ax=ax); cb.ax.tick_params(labelcolor=_TEXT)
    fig.tight_layout()
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:680px;">'


def badge(r):
    if r["validated"]:
        return '<span class="badge green">VALIDATED</span>'
    if r["wfo_run"] and r["oos_ret"] > 0:
        return '<span class="badge yellow">OOS POSITIVO</span>'
    if r["wfo_run"]:
        return '<span class="badge orange">WFO FAIL</span>'
    return '<span class="badge red">IC FAIL</span>'


def section(group_id, title, desc, variants):
    rows = ""
    for r in variants:
        wfo_cols = ""
        if r["wfo_run"]:
            ret_c = "green" if r["oos_ret"] > 0 else "red"
            pp_c  = "green" if r["mc_p_profit"] > 0.90 else ("yellow" if r["mc_p_profit"] > 0.60 else "red")
            pr_c  = "green" if r["mc_p_ruin"] < 0.05 else "red"
            wfo_cols = f"""
              <td>{r['oos_n']}</td>
              <td>{r['oos_wr']:.1%}</td>
              <td class="{ret_c}">{r['oos_ret']:+.1f}%</td>
              <td class="red">{r['oos_mdd']:.1f}%</td>
              <td class="{pp_c}">{r['mc_p_profit']:.3f}</td>
              <td class="{pr_c}">{r['mc_p_ruin']:.3f}</td>
            """
        else:
            wfo_cols = "<td colspan='6' style='color:#555'>—</td>"
        ic_c = "green" if r["ic"] > 0 else "red" if r["ic_pass"] else "dim"
        rows += f"""<tr>
          <td><code>{r['id']}</code></td>
          <td>{r['name']}</td>
          <td>{r['n_sig']:,}</td>
          <td class="{ic_c}">{r['ic']:+.4f}</td>
          <td class="{'green' if r['p_ic']<0.05 else 'red'}">{r['p_ic']:.4f}</td>
          {wfo_cols}
          <td>{badge(r)}</td>
        </tr>"""

    eq_imgs = "".join(_equity_chart(r)
                      for r in variants if r["wfo_run"] and r["oos_n"] > 0)

    return f"""
    <div class="card section-card">
      <h2 style="margin-top:0">{group_id} — {title}</h2>
      <p class="desc">{desc}</p>
      <div class="tbl-wrap">
      <table class="tbl">
        <thead>
          <tr>
            <th>ID</th><th>Variante</th><th>N segnali</th>
            <th>IC</th><th>p-val</th>
            <th>OOS n</th><th>OOS WR</th><th>OOS Ret</th><th>OOS MDD</th>
            <th>P(profit)</th><th>P(ruin)</th><th>Status</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      </div>
      {eq_imgs}
    </div>"""


overview_img = _overview_chart()
heatmap_img  = _heatmap_b()

sec_a = section("OPT-A", "RSI Threshold Scan",
    "Tre livelli di sensibilità: soglie {20/80, 25/75, 30/70}. "
    "Soglie più estreme → meno segnali ma più affidabili.", VARIANTS_A)

sec_b = section("OPT-B", "Z-score Window × Threshold",
    "Grid 4×3: finestra rolling {30,50,100,200 bar} × soglia {1.5,2.0,2.5}σ. "
    "Finestre più lunghe → stima media più stabile. Soglie più alte → segnali più rari.",
    VARIANTS_B)

sec_c = section("OPT-C", "ADX Regime Filter",
    "Filtro ranging: segnali MR attivi solo quando ADX < soglia. "
    "ADX alto = mercato in trend = MR controproducente.", VARIANTS_C)

sec_d = section("OPT-D", "Volatility (ATR%) Filter",
    "Filtro volatilità moderata. Volatilità troppo bassa = nessun movimento; "
    "troppo alta = breakout/trend che distrugge MR.", VARIANTS_D)

sec_e = section("OPT-E", "Combo H01 + H14",
    "Segnale solo quando RSI MR e Z-score MR concordano entrambi. "
    "Segnali molto più rari ma con doppia conferma.", VARIANTS_E)

n_valid = sum(1 for r in ALL_RESULTS if r["validated"])

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GitHub Strategies — Optimization Report</title>
<style>
  :root{{--bg:{_BG};--card:{_CARD};--grid:{_GRID};--text:{_TEXT};
         --acc:{_ACC};--grn:{_GRN};--red:{_RED};--yel:{_YEL};--org:{_ORG};}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
        font-size:14px;line-height:1.5;padding:24px;}}
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
  .tbl{{border-collapse:collapse;width:100%;min-width:900px;}}
  .tbl th{{background:var(--grid);color:var(--acc);text-align:left;
            padding:6px 10px;font-size:12px;white-space:nowrap;}}
  .tbl td{{padding:5px 10px;border-bottom:1px solid var(--grid);
            font-size:12px;white-space:nowrap;}}
  .tbl tr:hover td{{background:var(--grid);}}
  .green{{color:var(--grn)!important;}}
  .red{{color:var(--red)!important;}}
  .yellow{{color:var(--yel)!important;}}
  .dim{{color:#555!important;}}
  .badge{{display:inline-block;border-radius:4px;padding:2px 7px;
           font-size:11px;font-weight:600;}}
  .badge.green{{background:#1b3a1e;color:var(--grn);}}
  .badge.red{{background:#3a1a1a;color:var(--red);}}
  .badge.orange{{background:#3a2a00;color:var(--org);}}
  .badge.yellow{{background:#2d2a00;color:var(--yel);}}
  img{{display:block;margin-bottom:12px;border-radius:6px;max-width:100%;}}
  code{{font-size:11px;background:var(--grid);padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>

<h1>GitHub Strategies — Optimization Report</h1>
<p class="sub">
  Ottimizzazioni sistematiche di H01 (RSI MR) e H14 (Z-score MR) &nbsp;|&nbsp;
  {len(ALL_RESULTS)} varianti &nbsp;|&nbsp;
  Data: {IDX1H[0].date()} – {IDX1H[-1].date()} · {N1H:,} bar 1H
</p>

<div class="kpi-row">
  <div class="kpi"><div class="val">{len(ALL_RESULTS)}</div><div class="lbl">Varianti testate</div></div>
  <div class="kpi"><div class="val">{n_ic}</div><div class="lbl">IC pass (p&lt;0.05)</div></div>
  <div class="kpi"><div class="val">{n_wfo}</div><div class="lbl">WFO eseguito</div></div>
  <div class="kpi"><div class="val" style="color:{'var(--grn)' if n_valid>0 else 'var(--red)'}">{n_valid}</div><div class="lbl">Validate</div></div>
  <div class="kpi">
    <div class="val" style="font-size:18px">
      {max((r['oos_ret'] for r in ALL_RESULTS if r['wfo_run']), default=0):+.1f}%
    </div>
    <div class="lbl">Miglior OOS ret</div>
  </div>
  <div class="kpi">
    <div class="val" style="font-size:18px">
      {max((r['mc_p_profit'] for r in ALL_RESULTS if r['wfo_run']), default=0):.3f}
    </div>
    <div class="lbl">Miglior MC p_profit</div>
  </div>
</div>

<div class="card">
  <h2>Overview — OOS Return e MC P(profit) per variante</h2>
  {overview_img}
</div>

{sec_a}

<div class="card">
  <h2>OPT-B Heatmap — OOS Return per (window, threshold)</h2>
  {heatmap_img}
</div>

{sec_b}
{sec_c}
{sec_d}
{sec_e}

<hr style="border-color:var(--grid);margin:32px 0 16px">
<p style="color:#444;font-size:11px;text-align:center;">
  BTCUSDT perpetual futures · Binance Vision 2020–2026 · 1H · WFO 6m/2m/2m ·
  MC N=5,000 · Risk 1% · Fee 0.08% RT
</p>
</body>
</html>"""

out = Path("reports/report_github_opt.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n[DONE] Report → {out}  ({out.stat().st_size//1024} KB)")
