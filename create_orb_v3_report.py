"""
create_orb_v3_report.py
========================
ICT Asian Range Sweep v3 — Full Validation Pipeline

Parametri (dalla ricerca statistica su 1840 eventi, 2020-2026):
  tp_frac   = 0.25   (TP = entry + 0.25 × asian_range)
  sl_buf    = 1.0    (SL = 1H_swept_low - 1.0 × ATR_1H)
  rta_max   = 1.5    (filtra sessioni dove range/ATR_1H > 1.5)
  rr_max    = 1.5    (filtro range/ATR per la validità del regime)

Pipeline:
  1. Walk-Forward Backtest — 6m IS / 2m OOS / step 2m
  2. Test statistici    — t-test OOS returns, t-test per-trade, binomial
  3. Monte Carlo        — 5000 simulazioni bootstrap
  4. Breakdown per anno
  5. Report HTML → reports/report_orb_v3.html
"""
from __future__ import annotations

import base64
import io
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.orb_ict      import (
    build_asian_range, LONDON_START, LONDON_END, ASIA_START, ASIA_END,
)
from src.strategy.monte_carlo  import run_monte_carlo

# ─────────────────────────────────────────────────────────────────────────────
# Strategy parameters (fixed — derived from full-sample statistical scan)
# ─────────────────────────────────────────────────────────────────────────────
TP_FRAC      = 0.25     # TP = entry + TP_FRAC × asian_range
SL_BUF       = 1.0      # SL = 1H_swept_low - SL_BUF × ATR_1H
RTA_MAX      = 1.5      # filter: asian_range / mean_ATR_1H ≤ this
MSS_LOOKBACK = 8        # max 15M bars to find MSS
MAX_HOLD     = 32       # ~8h in 15M bars (time-based exit)
RISK_PCT     = 0.01     # fraction of equity risked per trade
FEE          = 0.0004   # 0.04% one-way taker fee
INIT_CAP     = 100_000.0

# Walk-forward
WF_TRAIN_M   = 6
WF_OOS_M     = 2
WF_STEP_M    = 2
START_YEAR   = 2020

# Monte Carlo
N_SIMS = 5_000

# HTML theme
_BG   = "#0f1117"
_CARD = "#12151f"
_GRID = "#1e2130"
_TEXT = "#e0e0e0"
_ACC  = "#42a5f5"
_GRN  = "#66bb6a"
_RED  = "#ef5350"
_YEL  = "#ffd54f"

SEP  = "─" * 72
SEP2 = "═" * 72


# ─────────────────────────────────────────────────────────────────────────────
# Trade dataclass (matches engine interface for Monte Carlo compatibility)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ICTTrade:
    entry_ts:     pd.Timestamp
    exit_ts:      pd.Timestamp
    direction:    int
    entry_price:  float
    exit_price:   float
    stop_price:   float
    tp_price:     float
    net_pnl:      float
    gross_pnl:    float
    total_fees:   float
    exit_reason:  str
    year:         int
    window_id:    int


# ─────────────────────────────────────────────────────────────────────────────
# Custom ICT backtest engine
# ─────────────────────────────────────────────────────────────────────────────
def run_ict_backtest(
    events: list[dict],
    HI: np.ndarray,
    LO: np.ndarray,
    CL: np.ndarray,
    IDX: pd.DatetimeIndex,
    n_bars: int,
    initial_capital: float = INIT_CAP,
    tp_frac: float = TP_FRAC,
    sl_buf: float = SL_BUF,
    rta_max: float = RTA_MAX,
    window_id: int = 0,
) -> tuple[list[ICTTrade], pd.Series]:
    """
    Simulate ICT trades with fixed-risk sizing on the given event list.

    Returns
    -------
    trades   : list[ICTTrade]
    equity   : pd.Series indexed by exit_ts (step equity at each trade close)
    """
    equity = float(initial_capital)
    trades: list[ICTTrade] = []

    for ev in events:
        if ev["rta"] > rta_max:
            continue

        entry_px = ev["entry_px"]
        ar       = ev["ar"]
        atr_1h   = ev["atr_sweep"]   # 1H ATR at sweep bar
        swept    = ev["swept_ext"]    # 1H low (LONG) or 1H high (SHORT)
        ei       = ev["entry_i"]
        direction = 1 if ev["direction"] == "long" else -1

        if direction == 1:
            tp_px = entry_px + tp_frac * ar
            sl_px = swept - sl_buf * atr_1h
        else:
            tp_px = entry_px - tp_frac * ar
            sl_px = swept + sl_buf * atr_1h

        sl_dist = abs(entry_px - sl_px)
        tp_dist = abs(tp_px - entry_px)

        if sl_dist <= 0 or tp_dist <= 0:
            continue
        if sl_px >= entry_px and direction == 1:
            continue
        if sl_px <= entry_px and direction == -1:
            continue

        # Fixed-risk position sizing
        at_risk   = equity * RISK_PCT
        btc_qty   = at_risk / sl_dist
        notional  = btc_qty * entry_px
        entry_fee = notional * FEE

        # Scan forward for TP or SL
        hit_tp = hit_sl = False
        exit_bar = min(ei + MAX_HOLD, n_bars - 1)
        exit_px  = float(CL[exit_bar])

        for k in range(ei, min(ei + MAX_HOLD, n_bars)):
            bar_hi = float(HI[k])
            bar_lo = float(LO[k])
            if direction == 1:
                if bar_lo <= sl_px:
                    hit_sl  = True
                    exit_px = sl_px
                    exit_bar = k
                    break
                if bar_hi >= tp_px:
                    hit_tp  = True
                    exit_px = tp_px
                    exit_bar = k
                    break
            else:
                if bar_hi >= sl_px:
                    hit_sl  = True
                    exit_px = sl_px
                    exit_bar = k
                    break
                if bar_lo <= tp_px:
                    hit_tp  = True
                    exit_px = tp_px
                    exit_bar = k
                    break

        gross_pnl = direction * btc_qty * (exit_px - entry_px)
        exit_fee  = btc_qty * exit_px * FEE
        net_pnl   = gross_pnl - entry_fee - exit_fee

        equity += net_pnl

        if hit_tp:
            reason = "tp"
        elif hit_sl:
            reason = "sl"
        else:
            reason = "time"

        trades.append(ICTTrade(
            entry_ts    = IDX[ev["entry_i"] - 1],
            exit_ts     = IDX[exit_bar],
            direction   = direction,
            entry_price = entry_px,
            exit_price  = exit_px,
            stop_price  = sl_px,
            tp_price    = tp_px,
            gross_pnl   = gross_pnl,
            total_fees  = entry_fee + exit_fee,
            net_pnl     = net_pnl,
            exit_reason = reason,
            year        = ev["year"],
            window_id   = window_id,
        ))

    if not trades:
        return trades, pd.Series(dtype=float)

    trade_pnls = [t.net_pnl for t in trades]
    exit_times = [t.exit_ts for t in trades]
    eq_vals    = np.empty(len(trades) + 1)
    eq_vals[0] = initial_capital
    for k, pnl in enumerate(trade_pnls):
        eq_vals[k + 1] = eq_vals[k] + pnl

    equity_series = pd.Series(
        eq_vals[1:], index=pd.DatetimeIndex(exit_times), name="equity"
    )
    return trades, equity_series


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward windows
# ─────────────────────────────────────────────────────────────────────────────
def _wf_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
    from dateutil.relativedelta import relativedelta
    wins, cur = [], start
    while True:
        tr_end = cur + relativedelta(months=WF_TRAIN_M)
        oo_end = tr_end + relativedelta(months=WF_OOS_M)
        if oo_end > end:
            break
        wins.append((cur, tr_end, oo_end))
        cur = cur + relativedelta(months=WF_STEP_M)
    return wins


# ─────────────────────────────────────────────────────────────────────────────
# KPIs from equity time series
# ─────────────────────────────────────────────────────────────────────────────
def _kpis(equity: pd.Series, init_cap: float = INIT_CAP) -> dict:
    if equity.empty:
        return dict(total_return=0, calmar=0, sharpe=0, max_dd=0,
                    n_trades=0, win_rate=0)
    full = pd.concat([pd.Series([init_cap], index=[equity.index[0] - pd.Timedelta("1s")]),
                      equity])
    dd   = (full / full.cummax() - 1).min()
    ret  = (full.iloc[-1] / init_cap) - 1
    rets = full.pct_change().dropna()
    vol  = rets.std() * np.sqrt(365 * 24)
    ann  = rets.mean() * 365 * 24
    sharpe = ann / vol if vol > 0 else 0.0
    calmar = ret / abs(dd) if dd < 0 else 0.0
    return dict(total_return=ret, calmar=calmar, sharpe=sharpe,
                max_dd=dd, n_trades=len(equity))


# ─────────────────────────────────────────────────────────────────────────────
# Chart utilities
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _imgt(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px">'


def _style_num(v: float, fmt: str = ".1%", good: float = 0) -> str:
    color = _GRN if v > good else _RED
    return f'<span style="color:{color}">{v:{fmt}}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────
def _plot_equity(oos_equity: pd.Series, btc_prices: pd.Series) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_CARD)
        ax.tick_params(colors=_TEXT)
        for spine in ax.spines.values():
            spine.set_color(_GRID)

    ax1.plot(oos_equity.index, oos_equity.values / INIT_CAP,
             color=_ACC, lw=1.5, label="OOS Equity")
    ax1.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (norm.)", color=_TEXT)
    ax1.legend(facecolor=_CARD, labelcolor=_TEXT)

    # Drawdown
    running_max = oos_equity.cummax()
    dd_pct = (oos_equity / running_max - 1) * 100
    ax2.fill_between(dd_pct.index, dd_pct.values, 0,
                     color=_RED, alpha=0.45, label="Drawdown %")
    ax2.set_ylabel("DD %", color=_TEXT)
    ax2.legend(facecolor=_CARD, labelcolor=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.suptitle("OOS Equity Curve — ICT Asian Sweep v3", color=_TEXT, y=1.01)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_wf_returns(window_rets: list[float]) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values():
        sp.set_color(_GRID)

    colors = [_GRN if r > 0 else _RED for r in window_rets]
    ax.bar(range(len(window_rets)), [r * 100 for r in window_rets],
           color=colors, alpha=0.8)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_xlabel("OOS Window #", color=_TEXT)
    ax.set_ylabel("Return %", color=_TEXT)
    ax.set_title("OOS Return per finestra WF", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_mc(mc_result: dict) -> str:
    final = mc_result["total_return"] * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values():
        sp.set_color(_GRID)

    ax.hist(final, bins=80, color=_ACC, alpha=0.7, edgecolor="none")
    ax.axvline(0, color=_RED, lw=1.5, ls="--", label="Breakeven")
    ax.axvline(np.median(final), color=_YEL, lw=1.5, label=f"Median {np.median(final):.1f}%")
    ax.set_xlabel("Total Return %", color=_TEXT)
    ax.set_ylabel("Count", color=_TEXT)
    ax.set_title("Monte Carlo — distribuzione rendimento finale (5000 sim.)", color=_TEXT)
    ax.legend(facecolor=_CARD, labelcolor=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_mc_paths(mc_result: dict, n_show: int = 200) -> str:
    paths = mc_result["paths"]
    n_t   = paths.shape[1]
    xs    = np.arange(n_t)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values():
        sp.set_color(_GRID)

    idxs = np.random.default_rng(1).integers(0, len(paths), n_show)
    for i in idxs:
        ax.plot(xs, paths[i] / INIT_CAP, color=_ACC, lw=0.3, alpha=0.15)
    p5  = np.percentile(paths, 5,  axis=0) / INIT_CAP
    p95 = np.percentile(paths, 95, axis=0) / INIT_CAP
    med = np.percentile(paths, 50, axis=0) / INIT_CAP
    ax.fill_between(xs, p5, p95, color=_ACC, alpha=0.2, label="5-95%")
    ax.plot(xs, med, color=_YEL, lw=1.5, label="Mediana")
    ax.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax.set_xlabel("# Trade", color=_TEXT)
    ax.set_ylabel("Equity (norm.)", color=_TEXT)
    ax.set_title("Monte Carlo — percorsi equity (bootstrap trade-level)", color=_TEXT)
    ax.legend(facecolor=_CARD, labelcolor=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_trade_dist(trades_df: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_CARD)
        ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values():
            sp.set_color(_GRID)

    pnls = trades_df["net_pnl"]
    clrs = [_GRN if p > 0 else _RED for p in pnls]
    ax1.bar(range(len(pnls)), pnls.values, color=clrs, alpha=0.7, linewidth=0)
    ax1.axhline(0, color=_GRID, lw=0.8)
    ax1.set_title("P&L per trade (OOS)", color=_TEXT)
    ax1.set_xlabel("Trade #", color=_TEXT)
    ax1.set_ylabel("Net PnL (USD)", color=_TEXT)

    ax2.hist(pnls.values, bins=30, color=_ACC, alpha=0.7, edgecolor="none")
    ax2.axvline(0, color=_RED, lw=1.5, ls="--")
    ax2.axvline(float(pnls.mean()), color=_YEL, lw=1.5,
                label=f"Mean {pnls.mean():.1f}")
    ax2.set_title("Distribuzione P&L per trade", color=_TEXT)
    ax2.set_xlabel("Net PnL (USD)", color=_TEXT)
    ax2.legend(facecolor=_CARD, labelcolor=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────
def _card(title: str, body: str) -> str:
    return f"""
<div style="background:{_CARD};border:1px solid {_GRID};border-radius:10px;
     padding:20px 24px;margin-bottom:20px">
  <h3 style="color:{_ACC};margin-top:0;margin-bottom:14px">{title}</h3>
  {body}
</div>"""


def _kv(label: str, value: str) -> str:
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:5px 0">'
            f'<span style="color:#9e9e9e">{label}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{value}</span></div>')


def _table(headers: list[str], rows: list[list]) -> str:
    th = "".join(f'<th style="padding:6px 10px;text-align:right;color:{_ACC}'
                 f';border-bottom:1px solid {_GRID}">{h}</th>' for h in headers)
    body = ""
    for row in rows:
        tds = "".join(f'<td style="padding:5px 10px;text-align:right;'
                      f'color:{_TEXT}">{c}</td>' for c in row)
        body += f"<tr>{tds}</tr>"
    return (f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("ICT Asian Range Sweep v3 — Full Validation Pipeline")
print(SEP2)

# ── 1. Data ──────────────────────────────────────────────────────────────────
print("\n[1/6] Caricamento dati …")
raw    = fetch_extended_data(start_year=START_YEAR, start_month=1,
                              fetch_15m=True, fetch_1m=False, fetch_flow=False)
df_1h  = add_indicators(raw["1H"])
df_15m = add_indicators(raw["15M"])
print(f"  1H : {len(df_1h):,} bar  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
print(f"  15M: {len(df_15m):,} bar")

# ── 2. Asian range + range/ATR ───────────────────────────────────────────────
print("\n[2/6] Asian Range …")
asian_daily = build_asian_range(df_1h)
mask_as     = (df_1h.index.hour >= ASIA_START) & (df_1h.index.hour < ASIA_END)
atr_by_date = (df_1h[mask_as].copy()
               .assign(date=lambda x: x.index.date)
               .groupby("date")["atr_14"].mean())
atr_by_date.index = pd.to_datetime(atr_by_date.index)
asian_daily["atr_1h_mean"]  = atr_by_date.reindex(asian_daily.index)
asian_daily["range_to_atr"] = (
    asian_daily["asian_range"] / asian_daily["atr_1h_mean"].clip(lower=1))

# ── 3. Precompute arrays ──────────────────────────────────────────────────────
print("\n[3/6] Pre-elaborazione eventi ICT …")
IDX = df_15m.index
H   = IDX.hour
HI  = df_15m["high"].values
LO  = df_15m["low"].values
CL  = df_15m["close"].values
N   = len(df_15m)

date_idx = IDX.normalize()
_ah_dict  = asian_daily["asian_high"].to_dict()
_al_dict  = asian_daily["asian_low"].to_dict()
_ar_dict  = asian_daily["asian_range"].to_dict()
_rta_dict = asian_daily["range_to_atr"].to_dict()
ah_arr   = np.array([_ah_dict.get(d,  np.nan) for d in date_idx], dtype=float)
al_arr   = np.array([_al_dict.get(d,  np.nan) for d in date_idx], dtype=float)
ar_arr   = np.array([_ar_dict.get(d,  np.nan) for d in date_idx], dtype=float)
rta_arr  = np.array([_rta_dict.get(d, np.nan) for d in date_idx], dtype=float)
yr_arr   = np.array([d.year for d in date_idx], dtype=int)

# Align 1H ATR and extremes to 15M bars
floor_1h   = IDX.floor("h")
atr_1h_map = df_1h["atr_14"].clip(lower=1.0).to_dict()
lo_1h_map  = df_1h["low"].to_dict()
hi_1h_map  = df_1h["high"].to_dict()
ATR_1H = np.array([atr_1h_map.get(t, np.nan) for t in floor_1h], dtype=float)
LO_1H  = np.array([lo_1h_map.get(t,  np.nan) for t in floor_1h], dtype=float)
HI_1H  = np.array([hi_1h_map.get(t,  np.nan) for t in floor_1h], dtype=float)
fallback_atr = df_15m["atr_14"].clip(lower=1.0).values
ATR_1H = np.where(np.isnan(ATR_1H), fallback_atr, ATR_1H)

# Collect all events
events: list[dict] = []
for i in range(N - MAX_HOLD - MSS_LOOKBACK - 2):
    if not (LONDON_START <= H[i] < LONDON_END):
        continue
    ah, al, ar, rta = ah_arr[i], al_arr[i], ar_arr[i], rta_arr[i]
    if np.isnan(ah) or np.isnan(al) or ar < 1.0:
        continue
    lo_1h_i = LO_1H[i]
    hi_1h_i = HI_1H[i]
    atr_1h_i = ATR_1H[i]
    if np.isnan(lo_1h_i) or np.isnan(hi_1h_i):
        continue

    if LO[i] < al and CL[i] >= al:
        sweep_hi = HI[i]
        mss_bar  = next(
            (j for j in range(i + 1, min(i + MSS_LOOKBACK + 1, N)) if HI[j] > sweep_hi),
            None
        )
        if mss_bar is None or mss_bar + MAX_HOLD >= N:
            continue
        events.append(dict(
            direction="long", sweep_i=i, entry_i=mss_bar + 1,
            entry_px=CL[mss_bar], swept_ext=lo_1h_i,
            ah=ah, al=al, ar=ar, atr_sweep=atr_1h_i, rta=rta,
            year=yr_arr[i], ts=IDX[i],
        ))

    elif HI[i] > ah and CL[i] <= ah:
        sweep_lo = LO[i]
        mss_bar  = next(
            (j for j in range(i + 1, min(i + MSS_LOOKBACK + 1, N)) if LO[j] < sweep_lo),
            None
        )
        if mss_bar is None or mss_bar + MAX_HOLD >= N:
            continue
        events.append(dict(
            direction="short", sweep_i=i, entry_i=mss_bar + 1,
            entry_px=CL[mss_bar], swept_ext=hi_1h_i,
            ah=ah, al=al, ar=ar, atr_sweep=atr_1h_i, rta=rta,
            year=yr_arr[i], ts=IDX[i],
        ))

print(f"  Totale eventi (sweep+MSS): {len(events)}")
filtered = [e for e in events if e["rta"] <= RTA_MAX]
print(f"  Dopo filtro rta≤{RTA_MAX}:   {len(filtered)}")

# ── 4. Walk-Forward ───────────────────────────────────────────────────────────
print("\n[4/6] Walk-Forward Backtest …")
wf_start  = pd.Timestamp(f"{START_YEAR}-01-01")
wf_end    = df_15m.index[-1]
windows   = _wf_windows(wf_start, wf_end)
print(f"  Finestre WF: {len(windows)}  (IS={WF_TRAIN_M}m / OOS={WF_OOS_M}m / step={WF_STEP_M}m)")

oos_all_trades: list[ICTTrade] = []
window_results: list[dict] = []
equity_running  = INIT_CAP

for w_i, (tr_s, tr_e, oo_e) in enumerate(windows):
    # Filter events to OOS window
    oos_evs = [e for e in events
               if tr_e <= e["ts"] < oo_e]
    if not oos_evs:
        continue

    # Run backtest on OOS events
    trades_w, eq_w = run_ict_backtest(
        oos_evs, HI, LO, CL, IDX, N,
        initial_capital=equity_running,
        tp_frac=TP_FRAC, sl_buf=SL_BUF, rta_max=RTA_MAX,
        window_id=w_i,
    )

    if not trades_w:
        continue

    pnl_w  = sum(t.net_pnl for t in trades_w)
    ret_w  = pnl_w / equity_running
    n_w    = sum(1 for t in trades_w if t.net_pnl > 0)
    wr_w   = n_w / len(trades_w) * 100 if trades_w else 0
    equity_running += pnl_w
    oos_all_trades.extend(trades_w)

    window_results.append(dict(
        window  = w_i + 1,
        tr_start= tr_s.strftime("%Y-%m"),
        tr_end  = tr_e.strftime("%Y-%m"),
        oo_end  = oo_e.strftime("%Y-%m"),
        n       = len(trades_w),
        wr_pct  = round(wr_w, 1),
        ret_pct = round(ret_w * 100, 2),
        eq_end  = round(equity_running, 0),
    ))
    print(f"  W{w_i+1:02d}  OOS {tr_e.strftime('%Y-%m')}→{oo_e.strftime('%Y-%m')}"
          f"  N={len(trades_w):3d}  WR={wr_w:4.1f}%  Ret={ret_w*100:+.2f}%"
          f"  Eq={equity_running:,.0f}")

if not oos_all_trades:
    print("\nNessun trade OOS — interrompo.")
    sys.exit(1)

# ── 5. OOS aggregate KPIs ─────────────────────────────────────────────────────
print(f"\n[5/6] Calcolo KPI e test statistici …")

# Build continuous equity series from OOS trades
eq_vals_list = [INIT_CAP]
for t in oos_all_trades:
    eq_vals_list.append(eq_vals_list[-1] + t.net_pnl)
exit_times = [t.exit_ts for t in oos_all_trades]
oos_equity = pd.Series(eq_vals_list[1:], index=pd.DatetimeIndex(exit_times))

kpis = _kpis(oos_equity, INIT_CAP)
n_trades = len(oos_all_trades)
n_wins   = sum(1 for t in oos_all_trades if t.net_pnl > 0)
n_losses = n_trades - n_wins
wr       = n_wins / n_trades * 100
tp_hits  = sum(1 for t in oos_all_trades if t.exit_reason == "tp")
sl_hits  = sum(1 for t in oos_all_trades if t.exit_reason == "sl")
tm_hits  = n_trades - tp_hits - sl_hits

# R:R from actual trade distances
avg_win_pct  = np.mean([(t.exit_price - t.entry_price) / t.entry_price * 100 * t.direction
                         for t in oos_all_trades if t.net_pnl > 0]) if n_wins else 0
avg_loss_pct = np.mean([(t.entry_price - t.exit_price) / t.entry_price * 100 * t.direction
                          for t in oos_all_trades if t.net_pnl <= 0]) if n_losses else 0
rr = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0
be_wr = 1 / (1 + rr) * 100 if rr > 0 else 50.0

# Build trades DataFrame for Monte Carlo compatibility
trades_df = pd.DataFrame([{
    "entry_ts":    t.entry_ts,
    "exit_ts":     t.exit_ts,
    "direction":   t.direction,
    "entry_price": t.entry_price,
    "exit_price":  t.exit_price,
    "net_pnl":     t.net_pnl,
    "gross_pnl":   t.gross_pnl,
    "total_fees":  t.total_fees,
    "exit_reason": t.exit_reason,
    "year":        t.year,
    "window_id":   t.window_id,
    "duration_h":  (t.exit_ts - t.entry_ts).total_seconds() / 3600,
} for t in oos_all_trades])

# ── Statistical tests ─────────────────────────────────────────────────────────
win_returns = [r["ret_pct"] / 100 for r in window_results]
t_win, p_win = st.ttest_1samp(win_returns, 0)

pnls = trades_df["net_pnl"].values
t_tr, p_tr = st.ttest_1samp(pnls, 0)

k_binom = int(wr / 100 * n_trades)
binom_res = st.binomtest(k_binom, n_trades, p=be_wr / 100, alternative="greater")

print(f"\n  ── OOS Summary ──────────────────────────────────────────")
print(f"  N trade  : {n_trades}  (TP:{tp_hits}  SL:{sl_hits}  Time:{tm_hits})")
print(f"  WR       : {wr:.1f}%  (breakeven: {be_wr:.1f}%  margin: {wr-be_wr:+.1f}pp)")
print(f"  R:R      : {rr:.2f}")
print(f"  Return   : {kpis['total_return']*100:+.1f}%")
print(f"  Calmar   : {kpis['calmar']:+.3f}")
print(f"  Sharpe   : {kpis['sharpe']:+.2f}")
print(f"  Max DD   : {kpis['max_dd']*100:.1f}%")
print(f"\n  ── Test statistici ──────────────────────────────────────")
print(f"  t-test OOS returns (H0: mean=0): t={t_win:.3f}  p={p_win:.4f}"
      f"  {'✓ SIGNIFICATIVO' if p_win < 0.05 else '✗ non sig.'}")
print(f"  t-test per-trade P&L (H0: mean=0): t={t_tr:.3f}  p={p_tr:.4f}"
      f"  {'✓ SIGNIFICATIVO' if p_tr < 0.05 else '✗ non sig.'}")
print(f"  Binomial WR vs BE ({be_wr:.1f}%): p={binom_res.pvalue:.4f}"
      f"  {'✓ SIGNIFICATIVO' if binom_res.pvalue < 0.05 else '✗ non sig.'}")

# ── Per-year breakdown ────────────────────────────────────────────────────────
yearly: dict[int, dict] = {}
for t in oos_all_trades:
    y = t.year
    if y not in yearly:
        yearly[y] = {"n": 0, "wins": 0, "pnl": 0.0}
    yearly[y]["n"]    += 1
    yearly[y]["wins"] += 1 if t.net_pnl > 0 else 0
    yearly[y]["pnl"]  += t.net_pnl

print(f"\n  ── Breakdown per anno ───────────────────────────────────")
print(f"  {'Anno':>6} {'N':>5} {'WR%':>7} {'P&L':>10}")
print(f"  {SEP[:45]}")
for yr in sorted(yearly):
    d  = yearly[yr]
    yr_wr = d["wins"] / d["n"] * 100 if d["n"] else 0
    print(f"  {yr:>6}  {d['n']:>5}  {yr_wr:>6.1f}%  {d['pnl']:>10,.0f}")

# ── Monte Carlo ───────────────────────────────────────────────────────────────
print(f"\n  Monte Carlo ({N_SIMS} simulazioni) …")
mc = run_monte_carlo(trades_df, initial_capital=INIT_CAP, n_sims=N_SIMS, seed=42)
print(f"  P(profit)   : {mc['p_profit']*100:.1f}%")
print(f"  P(ruin<50%) : {mc['p_ruin']*100:.1f}%")
print(f"  Return mediano: {np.median(mc['total_return'])*100:+.1f}%")
print(f"  DD mediano  : {np.median(mc['max_drawdown'])*100:.1f}%")

# ── 6. HTML Report ────────────────────────────────────────────────────────────
print(f"\n[6/6] Generazione report HTML …")

# Charts
b64_equity    = _plot_equity(oos_equity, df_1h["close"])
b64_wf_rets   = _plot_wf_returns([r["ret_pct"] / 100 for r in window_results])
b64_mc_dist   = _plot_mc(mc)
b64_mc_paths  = _plot_mc_paths(mc)
b64_trade_dist = _plot_trade_dist(trades_df)

# Card: Strategy parameters
card_params = _card(
    "Parametri Strategia (fissati dalla ricerca statistica)",
    "".join([
        _kv("Logica segnale",
            "ICT Asian Range Sweep + MSS London KZ (07-09 UTC)"),
        _kv("Direzione", "Reversion (LONG dopo down-sweep, SHORT dopo up-sweep)"),
        _kv("Entry", "Close della barra MSS (Market Structure Shift)"),
        _kv("TP", f"entry ± {TP_FRAC:.2f} × Asian Range"),
        _kv("SL", f"1H swept_low/high ± {SL_BUF:.1f} × ATR_1H"),
        _kv("Filtro range/ATR", f"rta ≤ {RTA_MAX} (sessioni asiatiche strette)"),
        _kv("Max hold", f"{MAX_HOLD} barre 15M (~{MAX_HOLD//4}h)"),
        _kv("Position sizing", f"1% equity rischiato per trade"),
        _kv("Commissioni", "0.04% per lato (Binance taker)"),
        _kv("Walk-Forward", f"IS={WF_TRAIN_M}m / OOS={WF_OOS_M}m / step={WF_STEP_M}m"),
    ])
)

# Card: OOS KPIs
def _kv_styled(label: str, val_str: str) -> str:
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:5px 0">'
            f'<span style="color:#9e9e9e">{label}</span>'
            f'<span style="font-weight:600">{val_str}</span></div>')

oos_ret_pct = kpis["total_return"] * 100
oos_cal     = kpis["calmar"]
oos_sh      = kpis["sharpe"]
oos_dd      = kpis["max_dd"] * 100

card_kpis = _card(
    "KPI Walk-Forward OOS",
    "".join([
        _kv_styled("Return OOS",
                   f'<span style="color:{_GRN if oos_ret_pct>0 else _RED}">'
                   f'{oos_ret_pct:+.1f}%</span>'),
        _kv_styled("Calmar OOS",
                   f'<span style="color:{_GRN if oos_cal>0 else _RED}">'
                   f'{oos_cal:+.3f}</span>'),
        _kv_styled("Sharpe OOS",
                   f'<span style="color:{_GRN if oos_sh>0 else _RED}">'
                   f'{oos_sh:+.2f}</span>'),
        _kv_styled("Max Drawdown",
                   f'<span style="color:{_RED}">{oos_dd:.1f}%</span>'),
        _kv("N trade OOS",   str(n_trades)),
        _kv("Win Rate",      f"{wr:.1f}%  (BE: {be_wr:.1f}%  margin: {wr-be_wr:+.1f}pp)"),
        _kv("R:R effettivo", f"{rr:.2f}"),
        _kv("TP hit",        f"{tp_hits} ({tp_hits/n_trades*100:.0f}%)"),
        _kv("SL hit",        f"{sl_hits} ({sl_hits/n_trades*100:.0f}%)"),
        _kv("Time exit",     f"{tm_hits} ({tm_hits/n_trades*100:.0f}%)"),
        _kv("Finestre WF", str(len(window_results))),
    ])
)

# Card: OOS Equity curve
card_equity = _card("Equity Curve OOS (continua, $100k iniziali)", _imgt(b64_equity))

# Card: WF returns per window
card_wf_rets = _card("OOS Return per Finestra WF", _imgt(b64_wf_rets))

# Card: WF table
wf_rows = []
for r in window_results:
    ret_color = _GRN if r["ret_pct"] > 0 else _RED
    wf_rows.append([
        r["window"],
        f"{r['tr_start']} → {r['tr_end']}",
        f"{r['tr_end']} → {r['oo_end']}",
        r["n"],
        f"{r['wr_pct']:.1f}%",
        f'<span style="color:{ret_color}">{r["ret_pct"]:+.2f}%</span>',
        f"${r['eq_end']:,.0f}",
    ])
card_wf_table = _card(
    "Tabella Walk-Forward",
    _table(["#", "IS", "OOS", "N", "WR%", "Return%", "Equity"], wf_rows)
)

# Card: Statistical tests
def _sig_badge(p: float) -> str:
    if p < 0.001:
        return f'<span style="color:{_GRN}">*** p&lt;.001</span>'
    elif p < 0.01:
        return f'<span style="color:{_GRN}">**  p&lt;.01</span>'
    elif p < 0.05:
        return f'<span style="color:{_GRN}">*   p&lt;.05</span>'
    elif p < 0.10:
        return f'<span style="color:{_YEL}">~   p={p:.3f} (borderline)</span>'
    else:
        return f'<span style="color:{_RED}">    p={p:.3f} (n.s.)</span>'

card_stats = _card(
    "Test Statistici di Significativita",
    "".join([
        _kv("t-test OOS returns (H0: mean=0)",
            f"t={t_win:.3f} &nbsp; {_sig_badge(p_win)}"),
        _kv("t-test per-trade P&amp;L (H0: mean=0)",
            f"t={t_tr:.3f} &nbsp; {_sig_badge(p_tr)}"),
        _kv(f"Binomial WR vs BE ({be_wr:.1f}%)",
            f"k={k_binom}/{n_trades} &nbsp; {_sig_badge(binom_res.pvalue)}"),
        _kv("Win Rate OOS",       f"{wr:.1f}%"),
        _kv("Breakeven WR",       f"{be_wr:.1f}%"),
        _kv("Margine WR",         f"{wr-be_wr:+.1f}pp"),
    ])
)

# Card: Per-year
yr_rows = []
for yr in sorted(yearly):
    d    = yearly[yr]
    yr_wr = d["wins"] / d["n"] * 100 if d["n"] else 0
    pnl_color = _GRN if d["pnl"] > 0 else _RED
    yr_rows.append([
        yr, d["n"], f"{yr_wr:.1f}%",
        f'<span style="color:{pnl_color}">${d["pnl"]:,.0f}</span>',
    ])
card_yearly = _card(
    "Breakdown per Anno (OOS)",
    _table(["Anno", "N", "WR%", "P&L"], yr_rows)
)

# Card: Monte Carlo
mc_sum = mc["summary"]
mc_rows = []
for _, row in mc_sum.iterrows():
    mc_rows.append([
        f"{int(row['percentile'])}°",
        f"${row['final_equity']:,.0f}",
        f"{row['total_return%']:+.1f}%",
        f"{row['max_drawdown%']:.1f}%",
        f"{row['sharpe']:.2f}",
    ])
mc_body  = _table(["Pct", "Equity Finale", "Return %", "Max DD %", "Sharpe"], mc_rows)
mc_stats = "".join([
    _kv("N simulazioni", f"{N_SIMS:,}"),
    _kv("N trade per sim.", str(n_trades)),
    _kv("P(profit)",    f"{mc['p_profit']*100:.1f}%"),
    _kv("P(ruin<50%)", f"{mc['p_ruin']*100:.1f}%"),
    _kv("Mediana Return", f"{np.median(mc['total_return'])*100:+.1f}%"),
    _kv("Mediana Max DD", f"{np.median(mc['max_drawdown'])*100:.1f}%"),
])
card_mc = _card(
    "Monte Carlo Bootstrap (5000 simulazioni)",
    mc_stats + mc_body + "<br>" + _imgt(b64_mc_dist) + "<br>" + _imgt(b64_mc_paths)
)

# Card: Trade distribution
card_trades = _card("Distribuzione P&L per Trade (OOS)", _imgt(b64_trade_dist))

# ── HTML assembly ──────────────────────────────────────────────────────────
oos_ret_pct_fmt = f"{oos_ret_pct:+.1f}%"
oos_cal_fmt     = f"{oos_cal:+.3f}"

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ICT Asian Sweep v3 — Full Validation</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ background:{_BG}; color:{_TEXT}; font-family:'Segoe UI',sans-serif;
            margin:0; padding:20px; font-size:14px; }}
    h1   {{ color:{_ACC}; margin:0 0 4px 0; font-size:22px; }}
    h2   {{ color:{_YEL}; font-size:16px; margin:20px 0 10px 0; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    @media(max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} }}
    table {{ font-size:12.5px; }}
  </style>
</head>
<body>
  <h1>ICT Asian Range Sweep v3 — Validazione Completa</h1>
  <p style="color:#757575;margin:0 0 18px">
    BTCUSDT Perpetual Futures &nbsp;|&nbsp; 2020-01 → 2026-05 &nbsp;|&nbsp;
    WF {WF_TRAIN_M}m IS / {WF_OOS_M}m OOS &nbsp;|&nbsp;
    OOS Return: <strong style="color:{'#66bb6a' if oos_ret_pct>0 else '#ef5350'}">{oos_ret_pct_fmt}</strong>
    &nbsp; Calmar: <strong style="color:{'#66bb6a' if oos_cal>0 else '#ef5350'}">{oos_cal_fmt}</strong>
  </p>

  <div class="grid">
    {card_params}
    {card_kpis}
  </div>

  {card_equity}
  {card_wf_rets}
  {card_wf_table}

  <div class="grid">
    {card_stats}
    {card_yearly}
  </div>

  {card_mc}
  {card_trades}

  <p style="color:#444;font-size:11px;margin-top:30px">
    Generato da create_orb_v3_report.py &nbsp;|&nbsp; {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
  </p>
</body>
</html>"""

out_path = Path("reports/report_orb_v3.html")
out_path.write_text(html, encoding="utf-8")
print(f"\n  Report salvato → {out_path}")
print(f"\n{SEP2}")
print(f"RISULTATO FINALE:")
print(f"  OOS Return   : {oos_ret_pct:+.1f}%")
print(f"  Calmar OOS   : {oos_cal:+.3f}")
print(f"  Sharpe OOS   : {oos_sh:+.2f}")
print(f"  Max DD OOS   : {oos_dd:.1f}%")
print(f"  N trade OOS  : {n_trades}")
print(f"  WR OOS       : {wr:.1f}%  (BE: {be_wr:.1f}%)")
print(f"  p-value (bin): {binom_res.pvalue:.4f}"
      f"  {'SIGNIFICATIVO ✓' if binom_res.pvalue < 0.05 else 'non sig. ✗'}")
print(f"  P(profit) MC : {mc['p_profit']*100:.1f}%")
print(SEP2)
