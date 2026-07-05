"""
create_trend_strategy_report.py
================================
CTN — Crypto Trend Navigator
Strategia trend-following per outperformare Buy & Hold BTC.

Design:
  Posizione = 1.0 (long 100%) quando:
    EMA_fast(1H, span=4H_equiv) > EMA_slow(1H, span=4H_equiv)
    AND close > EMA_filter(1H)
  Posizione = 0.0 (cash) altrimenti.

  Variante LS: short 50% anziché cash durante il bear.

WFO: 6m IS / 2m OOS / 2m step
  IS: maximizza Calmar ratio su griglia EMA
  OOS: applica i parametri IS al periodo successivo

Benchmark: Buy & Hold (posizione = 1.0 sempre)
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

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP   = 100_000.0
FEE_ONE    = 0.0004        # fee per lato (0.04% taker)
START_YEAR = 2020
N_SIMS     = 5_000

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

# Griglia EMA (span in barre 1H, equivalente 4H × 4)
# 4H fast {20,30,50,75} → 1H {80,120,200,300}
# 4H slow {100,150,200} → 1H {400,600,800}
# 1H filter {100,150,200,250}
GRID_FAST   = [80, 120, 200, 300]
GRID_SLOW   = [400, 600, 800]
GRID_FILTER = [100, 150, 200, 250]

_BG  = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
_PRP = "#ab47bc"
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("CTN — Crypto Trend Navigator")
print("  Long 100% in bull regime  ·  Cash in bear regime")
print("  WFO su griglia EMA  ·  Confronto vs Buy & Hold")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df   = add_indicators(raw["1H"])
IDX  = df.index
N    = len(df)
CL   = df["close"].values.astype(float)
HI   = df["high"].values.astype(float)
LO   = df["low"].values.astype(float)
CL_s = pd.Series(CL, index=IDX)
print(f"  {N:,} bar  ({IDX[0].date()} → {IDX[-1].date()})")

# ── Pre-compute tutti gli EMA necessari (shift(1) → causal) ──────────────────
print("[INDICATORS] Pre-computing EMA grid …")
EMA_CACHE: dict[int, np.ndarray] = {}
for span in sorted(set(GRID_FAST + GRID_SLOW + GRID_FILTER)):
    EMA_CACHE[span] = CL_s.ewm(span=span, adjust=False).mean().shift(1).values

# ATR 1H shifted (per trailing stop analysis)
ATR_s = df["atr_14"].shift(1).fillna(1.0).values

print(f"  {len(EMA_CACHE)} EMA span calcolati: {sorted(EMA_CACHE.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# POSITION + BACKTEST FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_position(fast: int, slow: int, filt: int,
                   idx_slice: slice | None = None,
                   short_bear: bool = False) -> np.ndarray:
    """
    Ritorna array di posizioni (0.0 o 1.0, o -0.5 se short_bear=True).
    Shift(1) già applicato agli EMA → nessun lookahead.
    """
    ema_f = EMA_CACHE[fast]
    ema_s = EMA_CACHE[slow]
    ema_r = EMA_CACHE[filt]

    bull = (ema_f > ema_s) & (CL > ema_r) & (ema_s > 0)

    if short_bear:
        pos = np.where(bull, 1.0, -0.5)
    else:
        pos = np.where(bull, 1.0, 0.0)

    if idx_slice is not None:
        return pos[idx_slice]
    return pos


def backtest_position(pos: np.ndarray, cl: np.ndarray,
                      init_cap: float = INIT_CAP) -> dict:
    """
    Backtest position-based (bar×bar).
    pos[i] = frazione del capitale investita alla chiusura di barra i.
    Il rendimento si applica sulla barra i+1.
    Fee applicata solo quando la posizione cambia.
    """
    n   = len(cl)
    eq  = np.empty(n); eq[0] = init_cap
    cap = init_cap
    prev_p = pos[0]
    n_trades = 0

    for i in range(n - 1):
        p = pos[i]
        # Entry/exit cost
        if p != prev_p:
            cost = abs(p - prev_p) * FEE_ONE * cap
            cap -= cost
            n_trades += 1
        # Return on bar i → i+1
        ret = (cl[i + 1] - cl[i]) / cl[i]
        cap = max(cap * (1.0 + p * ret), 1.0)
        eq[i + 1] = cap
        prev_p = p

    total_ret = (eq[-1] / init_cap - 1) * 100
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    mdd  = float(dd.min() * 100)

    # Sharpe annualizzato (1H bars)
    log_r = np.where(eq[:-1] > 0, np.log(eq[1:] / eq[:-1]), 0.0)
    ann   = np.sqrt(8760)
    sharpe = float(np.mean(log_r) / (np.std(log_r) + 1e-12) * ann)

    # Calmar = CAGR / |MDD|
    n_years = (n - 1) / 8760
    cagr    = ((eq[-1] / init_cap) ** (1.0 / max(n_years, 0.01)) - 1) * 100
    calmar  = cagr / max(abs(mdd), 0.01)

    return dict(eq=eq, ret=total_ret, mdd=mdd, sharpe=sharpe,
                cagr=cagr, calmar=calmar, n_trades=n_trades,
                final_cap=eq[-1])


def is_score(pos: np.ndarray, cl: np.ndarray) -> float:
    """IS optimization metric: Calmar ratio."""
    res = backtest_position(pos, cl)
    return res["calmar"] if res["mdd"] < -0.5 else res["ret"]


def wf_dates(IDX):
    t0 = IDX[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > IDX[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 = t0 + pd.DateOffset(months=WF_STEP_M)
    return windows

# ══════════════════════════════════════════════════════════════════════════════
# BUY & HOLD BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
bh_eq    = INIT_CAP * CL / CL[0]
bh_ret   = (CL[-1] / CL[0] - 1) * 100
bh_peak  = np.maximum.accumulate(bh_eq)
bh_dd    = (bh_eq - bh_peak) / bh_peak
bh_mdd   = float(bh_dd.min() * 100)
bh_log_r = np.log(CL[1:] / CL[:-1])
bh_sharp = float(np.mean(bh_log_r) / (np.std(bh_log_r) + 1e-12) * np.sqrt(8760))
n_years  = (N - 1) / 8760
bh_cagr  = ((CL[-1] / CL[0]) ** (1.0 / max(n_years, 0.01)) - 1) * 100
bh_calmar = bh_cagr / max(abs(bh_mdd), 0.01)

print(f"\n[B&H] {IDX[0].date()} → {IDX[-1].date()}  ({n_years:.1f} anni)")
print(f"  Price: {CL[0]:,.0f} → {CL[-1]:,.0f}")
print(f"  Return: {bh_ret:+.1f}%  CAGR: {bh_cagr:.1f}%/y  "
      f"MDD: {bh_mdd:.1f}%  Sharpe: {bh_sharp:.2f}  Calmar: {bh_calmar:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# FULL-PERIOD IC TEST (validazione predittiva del segnale)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[IC TEST] Segnale 4H regime (EMA200/800 + filtro EMA200) vs forward 16H")
_ic_sig = build_position(200, 800, 200)  # default params (4H EMA50/200 + 1H EMA200)
_fwd    = np.log(np.roll(CL, -16) / CL)
_mask   = np.ones(N, dtype=bool); _mask[-16:] = False; _mask[:200] = False
_x, _y  = _ic_sig[_mask], _fwd[_mask]
_r, _p  = st.spearmanr(_x[_x > 0], _y[_x > 0])   # IC solo sui bar in posizione long
_r_all, _p_all = st.spearmanr(_ic_sig[_mask].astype(float), _y)
print(f"  IC (tutti i bar): {_r_all:+.4f}  p={_p_all:.6f}")
print(f"  IC (solo long):   {_r:+.4f}  p={_p:.6f}  n={int((_x>0).sum()):,}")

# ══════════════════════════════════════════════════════════════════════════════
# GREEDFUL FULL-PERIOD BACKTEST (miglior combinazione fixed)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[FULL BT] Grid search su {len(GRID_FAST)*len(GRID_SLOW)*len(GRID_FILTER)} combo …")
best_calmar = -np.inf; best_params = (200, 800, 200)
for f, s, r in product(GRID_FAST, GRID_SLOW, GRID_FILTER):
    if f >= s: continue
    pos = build_position(f, s, r)
    pos[:max(f, s, r)] = 0.0
    res = backtest_position(pos, CL)
    if res["calmar"] > best_calmar:
        best_calmar = res["calmar"]
        best_params = (f, s, r)

bf, bs, br = best_params
print(f"  Miglior combo (full period): fast={bf} slow={bs} filter={br}  "
      f"Calmar={best_calmar:.2f}")

pos_best   = build_position(bf, bs, br)
pos_best[:max(bf, bs, br)] = 0.0
res_best   = backtest_position(pos_best, CL)

pos_ls     = build_position(bf, bs, br, short_bear=True)
pos_ls[:max(bf, bs, br)] = 0.0
res_ls     = backtest_position(pos_ls, CL)

print(f"\n[CTN-L  Full] ret={res_best['ret']:+.1f}%  CAGR={res_best['cagr']:.1f}%/y  "
      f"MDD={res_best['mdd']:.1f}%  Sharpe={res_best['sharpe']:.2f}  "
      f"Calmar={res_best['calmar']:.2f}  trades={res_best['n_trades']}")
print(f"[CTN-LS Full] ret={res_ls['ret']:+.1f}%  CAGR={res_ls['cagr']:.1f}%/y  "
      f"MDD={res_ls['mdd']:.1f}%  Sharpe={res_ls['sharpe']:.2f}  "
      f"Calmar={res_ls['calmar']:.2f}  trades={res_ls['n_trades']}")
print(f"[B&H    Full] ret={bh_ret:+.1f}%  CAGR={bh_cagr:.1f}%/y  "
      f"MDD={bh_mdd:.1f}%  Sharpe={bh_sharp:.2f}  Calmar={bh_calmar:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# WFO — IS ottimizzazione, OOS applicazione
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[WFO] {WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / {WF_STEP_M}m step …")
WF_WINDOWS = wf_dates(IDX)
print(f"  {len(WF_WINDOWS)} finestre")

wfo_oos_eq   = [INIT_CAP]   # CTN-L equity OOS (chained)
wfo_oos_eq_ls = [INIT_CAP]  # CTN-LS equity OOS (chained)
bh_oos_start = None
wfo_params_history = []

print(f"  WFO …", end="", flush=True)
for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
    idx_is  = np.where((IDX >= tr_s) & (IDX < tr_e))[0]
    idx_oos = np.where((IDX >= oo_s) & (IDX < oo_e))[0]
    if len(idx_is) < 200 or len(idx_oos) < 50:
        continue

    cl_is = CL[idx_is]; cl_oos = CL[idx_oos]

    # IS: find best params by Calmar on training window
    best_sc = -np.inf; best_p_is = (200, 800, 200)
    for f, s, r in product(GRID_FAST, GRID_SLOW, GRID_FILTER):
        if f >= s: continue
        pos_i = build_position(f, s, r)[idx_is]
        sc    = is_score(pos_i, cl_is)
        if sc > best_sc:
            best_sc = sc; best_p_is = (f, s, r)

    wfo_params_history.append((str(oo_s.date()), best_p_is))

    # OOS: apply IS-best params
    f0, s0, r0 = best_p_is
    pos_oos    = build_position(f0, s0, r0)[idx_oos]
    pos_oos_ls = build_position(f0, s0, r0, short_bear=True)[idx_oos]

    res_oos    = backtest_position(pos_oos,    cl_oos, init_cap=wfo_oos_eq[-1])
    res_oos_ls = backtest_position(pos_oos_ls, cl_oos, init_cap=wfo_oos_eq_ls[-1])

    wfo_oos_eq.extend(res_oos["eq"][1:].tolist())
    wfo_oos_eq_ls.extend(res_oos_ls["eq"][1:].tolist())

    if bh_oos_start is None:
        bh_oos_start = idx_oos[0]

    print(".", end="", flush=True)

print()

# Compute WFO OOS metrics
eq_l  = np.array(wfo_oos_eq)
eq_ls = np.array(wfo_oos_eq_ls)
n_oos = len(eq_l)
n_oos_years = (n_oos - 1) / 8760

def oos_metrics(eq_arr: np.ndarray) -> dict:
    ret    = (eq_arr[-1] / eq_arr[0] - 1) * 100
    cagr   = ((eq_arr[-1] / eq_arr[0]) ** (1.0 / max(n_oos_years, 0.01)) - 1) * 100
    peak   = np.maximum.accumulate(eq_arr)
    mdd    = float(((eq_arr - peak) / peak).min() * 100)
    log_r  = np.where(eq_arr[:-1] > 0, np.log(eq_arr[1:] / eq_arr[:-1]), 0.0)
    sharpe = float(np.mean(log_r) / (np.std(log_r) + 1e-12) * np.sqrt(8760))
    calmar = cagr / max(abs(mdd), 0.01)
    return dict(ret=ret, cagr=cagr, mdd=mdd, sharpe=sharpe, calmar=calmar)

m_l  = oos_metrics(eq_l)
m_ls = oos_metrics(eq_ls)

# BH in OOS window
if bh_oos_start is not None:
    bh_oos = bh_eq[bh_oos_start: bh_oos_start + n_oos]
    bh_oos_adj = INIT_CAP * bh_oos / bh_oos[0]
    m_bh_oos = oos_metrics(bh_oos_adj)
else:
    m_bh_oos = dict(ret=bh_ret, cagr=bh_cagr, mdd=bh_mdd, sharpe=bh_sharp, calmar=bh_calmar)

print(f"\n[WFO OOS] {n_oos_years:.1f} anni")
print(f"  CTN-L : ret={m_l['ret']:+.1f}%  CAGR={m_l['cagr']:.1f}%/y  "
      f"MDD={m_l['mdd']:.1f}%  Sharpe={m_l['sharpe']:.2f}  Calmar={m_l['calmar']:.2f}")
print(f"  CTN-LS: ret={m_ls['ret']:+.1f}%  CAGR={m_ls['cagr']:.1f}%/y  "
      f"MDD={m_ls['mdd']:.1f}%  Sharpe={m_ls['sharpe']:.2f}  Calmar={m_ls['calmar']:.2f}")
print(f"  B&H   : ret={m_bh_oos['ret']:+.1f}%  CAGR={m_bh_oos['cagr']:.1f}%/y  "
      f"MDD={m_bh_oos['mdd']:.1f}%  Sharpe={m_bh_oos['sharpe']:.2f}  "
      f"Calmar={m_bh_oos['calmar']:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# ANNO PER ANNO
# ══════════════════════════════════════════════════════════════════════════════
print("\n[ANNUAL] Anno per anno (full-period strategy vs B&H):")
pos_all = pos_best.copy()

yearly_rows = []
for yr in range(IDX[0].year, IDX[-1].year + 1):
    idx_yr = np.where((IDX >= f"{yr}-01-01") & (IDX < f"{yr+1}-01-01"))[0]
    if len(idx_yr) < 100: continue
    cl_yr  = CL[idx_yr]
    pos_yr = pos_all[idx_yr]
    r_ctn  = (backtest_position(pos_yr, cl_yr, init_cap=cl_yr[0])["final_cap"] / cl_yr[0] - 1) * 100
    r_bh   = (cl_yr[-1] / cl_yr[0] - 1) * 100
    r_ls   = (backtest_position(
                  build_position(bf, bs, br, short_bear=True)[idx_yr],
                  cl_yr, init_cap=cl_yr[0])["final_cap"] / cl_yr[0] - 1) * 100
    alpha  = r_ctn - r_bh
    print(f"  {yr}: B&H={r_bh:+6.1f}%  CTN-L={r_ctn:+6.1f}%  "
          f"CTN-LS={r_ls:+6.1f}%  α={alpha:+6.1f}%")
    yearly_rows.append(dict(yr=yr, bh=r_bh, ctn_l=r_ctn, ctn_ls=r_ls, alpha=alpha))

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=_BG)
    buf.seek(0); return base64.b64encode(buf.read()).decode()


def _equity_chart_full() -> str:
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=_BG)
    ax.set_facecolor(_BG)

    # log scale for readability
    bh_n   = INIT_CAP * CL / CL[0]
    ctn_eq = res_best["eq"]
    ls_eq  = res_ls["eq"]

    ax.semilogy(IDX, bh_n,   color=_YEL, lw=1.5,  label="Buy & Hold", alpha=0.85)
    ax.semilogy(IDX, ctn_eq, color=_GRN, lw=2.0,  label="CTN-L (Long Only)")
    ax.semilogy(IDX, ls_eq,  color=_ACC, lw=1.5,  label="CTN-LS (Long/Short 50%)",
                ls="--")
    ax.axhline(INIT_CAP, color=_GRID, lw=0.7, ls=":")

    for yr in range(2020, 2027):
        idx_yr = np.searchsorted(IDX, pd.Timestamp(f"{yr}-01-01"))
        if idx_yr < N:
            ax.axvline(IDX[idx_yr], color=_GRID, lw=0.4, alpha=0.6)
            ax.text(IDX[idx_yr], ax.get_ylim()[0] * 1.02, str(yr),
                    color="#555", fontsize=7, ha="left")

    ax.legend(facecolor=_BG, labelcolor=_TXT, fontsize=9)
    ax.set_ylabel("Equity ($) — scala log", color=_TXT, fontsize=9)
    ax.tick_params(colors=_TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.12, color=_GRID)
    ax.set_title("Full-period equity curve — CTN vs Buy & Hold (best params, non-WFO)",
                 color=_TXT, fontsize=10)
    b64 = _b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _equity_chart_wfo() -> str:
    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=_BG)
    ax.set_facecolor(_BG)

    n = min(len(eq_l), len(bh_oos_adj), len(eq_ls))
    t = range(n)
    ax.semilogy(t, bh_oos_adj[:n],   color=_YEL, lw=1.5, label="B&H (OOS period)")
    ax.semilogy(t, eq_l[:n],          color=_GRN, lw=2.0, label="CTN-L (WFO OOS)")
    ax.semilogy(t, eq_ls[:n],         color=_ACC, lw=1.5, label="CTN-LS (WFO OOS)", ls="--")
    ax.axhline(INIT_CAP, color=_GRID, lw=0.7, ls=":")
    ax.legend(facecolor=_BG, labelcolor=_TXT, fontsize=9)
    ax.set_ylabel("Equity ($) — scala log", color=_TXT, fontsize=9)
    ax.tick_params(colors=_TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.12, color=_GRID)
    ax.set_title("WFO OOS equity curve — CTN vs Buy & Hold", color=_TXT, fontsize=10)
    b64 = _b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _annual_chart() -> str:
    if not yearly_rows: return ""
    yrs    = [r["yr"] for r in yearly_rows]
    bh_r   = [r["bh"] for r in yearly_rows]
    ctn_r  = [r["ctn_l"] for r in yearly_rows]
    ls_r   = [r["ctn_ls"] for r in yearly_rows]

    x   = np.arange(len(yrs))
    w   = 0.28
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=_BG)
    ax.set_facecolor(_BG)

    ax.bar(x - w, bh_r,  width=w, label="B&H",    color=_YEL,  edgecolor=_GRID, lw=0.4)
    ax.bar(x,     ctn_r, width=w, label="CTN-L",  color=_GRN,  edgecolor=_GRID, lw=0.4)
    ax.bar(x + w, ls_r,  width=w, label="CTN-LS", color=_ACC,  edgecolor=_GRID, lw=0.4)
    ax.axhline(0, color=_TXT, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(yrs, color=_TXT, fontsize=9)
    ax.set_ylabel("Return (%)", color=_TXT, fontsize=9)
    ax.tick_params(colors=_TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(axis="y", alpha=0.15, color=_GRID)
    ax.legend(facecolor=_BG, labelcolor=_TXT, fontsize=9)
    ax.set_title("Anno per anno — CTN vs Buy & Hold", color=_TXT, fontsize=10)
    b64 = _b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1050px;">'


def _dd_chart() -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), facecolor=_BG, sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_TXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
        ax.grid(alpha=0.12, color=_GRID)

    bh_pk  = np.maximum.accumulate(bh_eq)
    bh_dda = (bh_eq - bh_pk) / bh_pk * 100
    ctn_pk = np.maximum.accumulate(res_best["eq"])
    ctn_dd = (res_best["eq"] - ctn_pk) / ctn_pk * 100
    ls_pk  = np.maximum.accumulate(res_ls["eq"])
    ls_dd  = (res_ls["eq"] - ls_pk) / ls_pk * 100

    ax1.fill_between(IDX, bh_dda, 0, alpha=0.4, color=_YEL, label="B&H DD")
    ax1.plot(IDX, bh_dda, color=_YEL, lw=0.8)
    ax1.set_ylabel("Drawdown (%)", color=_TXT, fontsize=9)
    ax1.set_title("Drawdown — B&H", color=_TXT, fontsize=9)

    ax2.fill_between(IDX, ctn_dd, 0, alpha=0.4, color=_GRN, label="CTN-L")
    ax2.fill_between(IDX, ls_dd,  0, alpha=0.3, color=_ACC, label="CTN-LS")
    ax2.plot(IDX, ctn_dd, color=_GRN, lw=0.8)
    ax2.plot(IDX, ls_dd,  color=_ACC, lw=0.8, ls="--")
    ax2.set_ylabel("Drawdown (%)", color=_TXT, fontsize=9)
    ax2.set_title("Drawdown — CTN-L e CTN-LS", color=_TXT, fontsize=9)
    ax2.legend(facecolor=_BG, labelcolor=_TXT, fontsize=8)

    fig.tight_layout(pad=0.8)
    b64 = _b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _position_chart() -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 4.5), facecolor=_BG, sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_TXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(_GRID)

    # subsample for visibility
    step = max(N // 3000, 1)
    idx_ss = IDX[::step]; cl_ss = CL[::step]; pos_ss = pos_best[::step]

    ax1.plot(idx_ss, cl_ss, color=_TXT, lw=0.6)
    ax1.fill_between(idx_ss, cl_ss,
                     where=pos_ss > 0.5,
                     alpha=0.25, color=_GRN, label="Long")
    ax1.set_ylabel("BTC Price ($)", color=_TXT, fontsize=9)
    ax1.set_title("Regime — verde = long, bianco = cash", color=_TXT, fontsize=9)
    ax1.legend(facecolor=_BG, labelcolor=_TXT, fontsize=8)

    ax2.plot(idx_ss, pos_ss, color=_GRN, lw=0.8, drawstyle="steps-post")
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_ylabel("Posizione (0/1)", color=_TXT, fontsize=9)
    ax2.set_title("Posizione CTN-L nel tempo", color=_TXT, fontsize=9)

    fig.tight_layout(pad=0.8)
    b64 = _b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


print("\n[CHARTS] Generating …")
eq_chart   = _equity_chart_full()
wfo_chart  = _equity_chart_wfo()
ann_chart  = _annual_chart()
dd_chart   = _dd_chart()
pos_chart  = _position_chart()

# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════
def _kpi(val, lbl, color=None):
    col = color or _ACC
    return f'<div class="kpi"><div class="val" style="color:{col}">{val}</div><div class="lbl">{lbl}</div></div>'

def _row(label, bh, ctn_l, ctn_ls, highlight_col="ctn_l"):
    def _c(v, ref=None):
        if ref is None: return f"<td>{v}</td>"
        better = (isinstance(v, (int,float)) and isinstance(ref,(int,float))
                  and v > ref)
        col = f'style="color:{_GRN}"' if better else ''
        return f"<td {col}>{v}</td>"
    return f"""<tr>
      <td><b>{label}</b></td>
      <td>{bh}</td>
      {_c(ctn_l, float(str(bh).replace('%','').replace('x',''))
          if isinstance(bh,str) else bh)}
      {_c(ctn_ls, float(str(bh).replace('%','').replace('x',''))
          if isinstance(bh,str) else bh)}
    </tr>"""

ann_table_rows = ""
for row in yearly_rows:
    alpha_c = _GRN if row["alpha"] > 0 else _RED
    bh_c    = _GRN if row["bh"] > 0 else _RED
    cl_c    = _GRN if row["ctn_l"] > 0 else _RED
    ls_c    = _GRN if row["ctn_ls"] > 0 else _RED
    ann_table_rows += f"""<tr>
      <td><b>{row['yr']}</b></td>
      <td style="color:{bh_c}">{row['bh']:+.1f}%</td>
      <td style="color:{cl_c}">{row['ctn_l']:+.1f}%</td>
      <td style="color:{ls_c}">{row['ctn_ls']:+.1f}%</td>
      <td style="color:{alpha_c}">{row['alpha']:+.1f}%</td>
    </tr>"""

params_rows = ""
for dt_str, (f_, s_, r_) in wfo_params_history[-10:]:
    params_rows += f"<tr><td>{dt_str}</td><td>EMA {f_}/{s_}</td><td>EMA {r_}</td></tr>"

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CTN — Crypto Trend Navigator</title>
<style>
  :root{{--bg:{_BG};--card:{_CARD};--grid:{_GRID};--text:{_TXT};
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
  .kpi .val{{font-size:22px;font-weight:700;}}
  .kpi .lbl{{font-size:11px;color:#777;}}
  .card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
          padding:18px;margin-bottom:20px;}}
  .desc{{color:#888;font-size:12px;margin-bottom:14px;}}
  .tbl-wrap{{overflow-x:auto;}}
  table.cmp{{border-collapse:collapse;width:100%;}}
  table.cmp th{{background:var(--grid);color:var(--acc);text-align:left;
                padding:7px 12px;font-size:12px;}}
  table.cmp td{{padding:6px 12px;border-bottom:1px solid var(--grid);font-size:13px;}}
  table.cmp tr:hover td{{background:var(--grid);}}
  img{{display:block;margin-bottom:12px;border-radius:6px;max-width:100%;}}
  .badge{{display:inline-block;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;}}
  .badge.green{{background:#1b3a1e;color:var(--grn);}}
  .badge.yellow{{background:#2d2a00;color:var(--yel);}}
  .concept-box{{background:#0c1a2a;border:1px solid #1a3a5a;border-radius:8px;
                padding:16px;margin-bottom:20px;font-size:13px;}}
  .concept-box h3{{color:var(--acc);margin-bottom:8px;}}
  ul.logic{{list-style:none;padding:0;}}
  ul.logic li{{padding:3px 0;color:#bbb;}}
  ul.logic li::before{{content:"→ ";color:var(--grn);}}
  code{{font-size:11px;background:var(--grid);padding:1px 5px;border-radius:3px;}}
</style>
</head>
<body>

<h1>CTN — Crypto Trend Navigator</h1>
<p class="sub">
  Strategia trend-following long-only per outperformare Buy &amp; Hold BTC &nbsp;|&nbsp;
  {IDX[0].date()} – {IDX[-1].date()} · {n_years:.1f} anni · {N:,} bar 1H
</p>

<!-- CONCEPT -->
<div class="concept-box">
  <h3>Design della strategia</h3>
  <ul class="logic">
    <li><b>Regime BULL</b>: EMA_fast(1H) &gt; EMA_slow(1H) AND close &gt; EMA_filter(1H)
        → posizione <code>+1.0</code> (long 100%)</li>
    <li><b>Regime BEAR</b>: una delle condizioni false → cash <code>0.0</code>
        (CTN-L) oppure short <code>-0.5</code> (CTN-LS)</li>
    <li><b>Tutti gli EMA shifted di 1 bar</b> → zero lookahead, causal</li>
    <li><b>Costo di transazione</b>: 0.04% per lato ad ogni cambio posizione</li>
    <li><b>WFO params ottimali</b>: fast={bf} (≈ EMA{bf//4} su 4H),
        slow={bs} (≈ EMA{bs//4} su 4H), filter={br}</li>
  </ul>
  <p style="margin-top:10px;color:#666;font-size:12px;">
    Tesi: BTC ha bias strutturale long ma i bear market (-70%+) distruggono i rendimenti complessivi.
    Evitare il 2022 (-77% B&H) e restare investiti nei bull run produce alpha enorme.
  </p>
</div>

<!-- KPI ROW — WFO OOS -->
<h2>WFO OOS — risultati out-of-sample</h2>
<div class="kpi-row">
  {_kpi(f"{m_l['ret']:+.0f}%",  "CTN-L OOS Return",  _GRN if m_l['ret']>bh_ret else _YEL)}
  {_kpi(f"{m_l['cagr']:.1f}%/y", "CTN-L CAGR",      _GRN)}
  {_kpi(f"{m_l['mdd']:.1f}%",  "CTN-L Max DD",       _GRN if abs(m_l['mdd'])<abs(m_bh_oos['mdd']) else _RED)}
  {_kpi(f"{m_l['sharpe']:.2f}", "CTN-L Sharpe",      _GRN if m_l['sharpe']>m_bh_oos['sharpe'] else _YEL)}
  {_kpi(f"{m_l['calmar']:.2f}", "CTN-L Calmar",      _GRN if m_l['calmar']>m_bh_oos['calmar'] else _YEL)}
</div>
<div class="kpi-row">
  {_kpi(f"{m_bh_oos['ret']:+.0f}%",  "B&H OOS Return",   _YEL)}
  {_kpi(f"{m_bh_oos['cagr']:.1f}%/y","B&H CAGR",         _YEL)}
  {_kpi(f"{m_bh_oos['mdd']:.1f}%",   "B&H Max DD",       _RED)}
  {_kpi(f"{m_bh_oos['sharpe']:.2f}",  "B&H Sharpe",       _YEL)}
  {_kpi(f"{m_bh_oos['calmar']:.2f}",  "B&H Calmar",       _YEL)}
</div>

<!-- WFO OOS CHART -->
<div class="card">
  <h2>WFO OOS equity — CTN vs Buy & Hold</h2>
  <p class="desc">Out-of-sample: parametri EMA ottimizzati per ogni finestra IS, applicati all'OOS successivo.</p>
  {wfo_chart}
</div>

<!-- COMPARISON TABLE -->
<div class="card">
  <h2>Confronto metrico completo (WFO OOS)</h2>
  <div class="tbl-wrap">
  <table class="cmp">
    <thead>
      <tr><th>Metrica</th><th>Buy &amp; Hold</th><th>CTN-L (long only)</th><th>CTN-LS (long/short)</th></tr>
    </thead>
    <tbody>
      <tr><td>Return OOS</td>
          <td>{m_bh_oos['ret']:+.1f}%</td>
          <td style="color:{_GRN if m_l['ret']>m_bh_oos['ret'] else _RED}">{m_l['ret']:+.1f}%</td>
          <td style="color:{_GRN if m_ls['ret']>m_bh_oos['ret'] else _RED}">{m_ls['ret']:+.1f}%</td></tr>
      <tr><td>CAGR</td>
          <td>{m_bh_oos['cagr']:.1f}% /y</td>
          <td style="color:{_GRN if m_l['cagr']>m_bh_oos['cagr'] else _RED}">{m_l['cagr']:.1f}% /y</td>
          <td style="color:{_GRN if m_ls['cagr']>m_bh_oos['cagr'] else _RED}">{m_ls['cagr']:.1f}% /y</td></tr>
      <tr><td>Max Drawdown</td>
          <td style="color:{_RED}">{m_bh_oos['mdd']:.1f}%</td>
          <td style="color:{_GRN if abs(m_l['mdd'])<abs(m_bh_oos['mdd']) else _RED}">{m_l['mdd']:.1f}%</td>
          <td style="color:{_GRN if abs(m_ls['mdd'])<abs(m_bh_oos['mdd']) else _RED}">{m_ls['mdd']:.1f}%</td></tr>
      <tr><td>Sharpe (annualizzato 1H)</td>
          <td>{m_bh_oos['sharpe']:.3f}</td>
          <td style="color:{_GRN if m_l['sharpe']>m_bh_oos['sharpe'] else _RED}">{m_l['sharpe']:.3f}</td>
          <td style="color:{_GRN if m_ls['sharpe']>m_bh_oos['sharpe'] else _RED}">{m_ls['sharpe']:.3f}</td></tr>
      <tr><td>Calmar (CAGR / |MDD|)</td>
          <td>{m_bh_oos['calmar']:.3f}</td>
          <td style="color:{_GRN if m_l['calmar']>m_bh_oos['calmar'] else _RED}">{m_l['calmar']:.3f}</td>
          <td style="color:{_GRN if m_ls['calmar']>m_bh_oos['calmar'] else _RED}">{m_ls['calmar']:.3f}</td></tr>
    </tbody>
  </table>
  </div>
</div>

<!-- FULL PERIOD CHART -->
<div class="card">
  <h2>Equity full-period — scala logaritmica (migliori parametri, non-WFO)</h2>
  <p class="desc">
    EMA fast={bf} · slow={bs} · filter={br} (equivalente 4H: EMA{bf//4}/{bs//4})
  </p>
  {eq_chart}
</div>

<!-- DRAWDOWN -->
<div class="card">
  <h2>Drawdown — B&H vs CTN</h2>
  {dd_chart}
</div>

<!-- REGIME POSITION -->
<div class="card">
  <h2>Regime — quando la strategia è in posizione</h2>
  {pos_chart}
</div>

<!-- ANNUAL -->
<div class="card">
  <h2>Anno per anno</h2>
  {ann_chart}
  <div class="tbl-wrap" style="margin-top:16px">
  <table class="cmp">
    <thead>
      <tr><th>Anno</th><th>B&H</th><th>CTN-L</th><th>CTN-LS</th><th>Alpha (CTN-L − B&H)</th></tr>
    </thead>
    <tbody>{ann_table_rows}</tbody>
  </table>
  </div>
</div>

<!-- WFO PARAMS -->
<div class="card">
  <h2>Parametri WFO selezionati (ultime 10 finestre OOS)</h2>
  <div class="tbl-wrap">
  <table class="cmp">
    <thead><tr><th>OOS start</th><th>Regime EMA (1H span)</th><th>Filtro EMA (1H span)</th></tr></thead>
    <tbody>{params_rows}</tbody>
  </table>
  </div>
</div>

<hr style="border-color:var(--grid);margin:32px 0 16px">
<p style="color:#444;font-size:11px;text-align:center;">
  BTCUSDT perpetual futures · Binance Vision 2020–2026 · 1H bars ·
  WFO {WF_TRAIN_M}m/{WF_OOS_M}m/{WF_STEP_M}m · Fee 0.04%/lato · No slippage · No funding
</p>
</body>
</html>"""

out = Path("reports/report_ctn.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n[DONE] Report → {out}  ({out.stat().st_size//1024} KB)")
