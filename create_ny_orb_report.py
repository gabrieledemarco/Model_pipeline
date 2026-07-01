"""
create_ny_orb_report.py
========================
NY Opening Range Breakout — Full Validation Pipeline

Strategia:
  1. NY open = 14:30 UTC (09:30 AM ET)
  2. Prima candela 15M: body_hi = max(open,close), body_lo = min(open,close)
  3. Breakout: prima barra 15M successiva che chiude sopra body_hi (LONG)
               o sotto body_lo (SHORT)
  4. Entry: close della breakout bar
  5. SL: body_lo - SL_BUF × ATR_1H  (LONG)
         body_hi + SL_BUF × ATR_1H  (SHORT)
  6. TP: entry ± TP_FRAC × body_range

Parametri (ottimizzati IS su 2080 eventi 2020-2026):
  TP_FRAC = 2.0
  SL_BUF  = 0.25
  RBA_MAX = 999  (nessun filtro body/ATR)

Pipeline:
  1. Walk-Forward — 6m IS / 2m OOS / step 2m
  2. Test statistici — t-test, binomial
  3. Monte Carlo — 5000 simulazioni bootstrap
  4. Breakdown per anno
  5. Report HTML → reports/report_ny_orb.html
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
from src.strategy.monte_carlo  import run_monte_carlo

# ─────────────────────────────────────────────────────────────────────────────
# Parametri strategia (da IS scan)
# ─────────────────────────────────────────────────────────────────────────────
TP_FRAC     = 2.0
SL_BUF      = 0.25
RBA_MAX     = 999.0    # no filter
NY_HOUR     = 14
NY_MIN      = 30
NY_END_HOUR = 20
MAX_HOLD    = 32
MIN_BODY    = 10.0

RISK_PCT    = 0.01
FEE         = 0.0004
INIT_CAP    = 100_000.0

# Walk-forward
WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2
START_YEAR  = 2020

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
# Trade dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NYORBTrade:
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    direction:   int
    entry_price: float
    exit_price:  float
    stop_price:  float
    tp_price:    float
    net_pnl:     float
    gross_pnl:   float
    total_fees:  float
    exit_reason: str
    year:        int
    window_id:   int


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────
def run_ny_orb_backtest(
    events: list[dict],
    HI: np.ndarray,
    LO: np.ndarray,
    CL: np.ndarray,
    IDX: pd.DatetimeIndex,
    n_bars: int,
    initial_capital: float = INIT_CAP,
    tp_frac: float = TP_FRAC,
    sl_buf: float = SL_BUF,
    rba_max: float = RBA_MAX,
    window_id: int = 0,
) -> tuple[list[NYORBTrade], pd.Series]:
    equity = float(initial_capital)
    trades: list[NYORBTrade] = []

    for ev in events:
        if ev["rba"] > rba_max:
            continue

        entry_px  = ev["entry_px"]
        rw        = ev["rw"]
        atr_1h    = ev["atr_1h"]
        ei        = ev["entry_i"]
        direction = 1 if ev["direction"] == "long" else -1

        if direction == 1:
            tp_px = entry_px + tp_frac * rw
            sl_px = ev["body_lo"] - sl_buf * atr_1h
        else:
            tp_px = entry_px - tp_frac * rw
            sl_px = ev["body_hi"] + sl_buf * atr_1h

        sl_dist = abs(entry_px - sl_px)
        tp_dist = abs(tp_px - entry_px)

        if sl_dist <= 0 or tp_dist <= 0:
            continue
        if direction == 1 and sl_px >= entry_px:
            continue
        if direction == -1 and sl_px <= entry_px:
            continue

        at_risk   = equity * RISK_PCT
        btc_qty   = at_risk / sl_dist
        notional  = btc_qty * entry_px
        entry_fee = notional * FEE

        start = ei + 1      # scan path starting from bar after entry
        hit_tp = hit_sl = False
        exit_bar = min(start + MAX_HOLD, n_bars - 1)
        exit_px  = float(CL[exit_bar])

        for k in range(start, min(start + MAX_HOLD, n_bars)):
            bh = float(HI[k])
            bl = float(LO[k])
            if direction == 1:
                if bl <= sl_px:
                    hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bh >= tp_px:
                    hit_tp = True; exit_px = tp_px; exit_bar = k; break
            else:
                if bh >= sl_px:
                    hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bl <= tp_px:
                    hit_tp = True; exit_px = tp_px; exit_bar = k; break

        gross_pnl = direction * btc_qty * (exit_px - entry_px)
        exit_fee  = btc_qty * exit_px * FEE
        net_pnl   = gross_pnl - entry_fee - exit_fee

        equity += net_pnl
        reason = "tp" if hit_tp else ("sl" if hit_sl else "time")

        trades.append(NYORBTrade(
            entry_ts    = IDX[ei],
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

    pnls  = [t.net_pnl  for t in trades]
    times = [t.exit_ts  for t in trades]
    eq    = np.empty(len(trades) + 1)
    eq[0] = initial_capital
    for k, p in enumerate(pnls):
        eq[k + 1] = eq[k] + p

    return trades, pd.Series(eq[1:], index=pd.DatetimeIndex(times), name="equity")


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
# KPI helper
# ─────────────────────────────────────────────────────────────────────────────
def _kpis(equity: pd.Series, init_cap: float = INIT_CAP) -> dict:
    if equity.empty:
        return dict(total_return=0, calmar=0, sharpe=0, max_dd=0, n_trades=0)
    full = pd.concat([pd.Series([init_cap],
                     index=[equity.index[0] - pd.Timedelta("1s")]), equity])
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
# Chart helpers
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


def _plot_equity(oos_equity: pd.Series, btc_prices: pd.Series) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_CARD)
        ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values():
            sp.set_color(_GRID)

    ax1.plot(oos_equity.index, oos_equity.values / INIT_CAP,
             color=_ACC, lw=1.5, label="OOS Equity")
    ax1.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (norm.)", color=_TEXT)
    ax1.legend(facecolor=_CARD, labelcolor=_TEXT)

    running_max = oos_equity.cummax()
    dd_pct = (oos_equity / running_max - 1) * 100
    ax2.fill_between(dd_pct.index, dd_pct.values, 0,
                     color=_RED, alpha=0.45, label="Drawdown %")
    ax2.set_ylabel("DD %", color=_TEXT)
    ax2.legend(facecolor=_CARD, labelcolor=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.suptitle("OOS Equity — NY ORB Body Breakout", color=_TEXT, y=1.01)
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
    ax.axvline(np.median(final), color=_YEL, lw=1.5,
               label=f"Median {np.median(final):.1f}%")
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


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
print(SEP2)
print("NY ORB — Body Breakout | Full Validation Pipeline")
print(SEP2)

# ── 1. Data ──────────────────────────────────────────────────────────────────
print("\n[1/6] Caricamento dati …")
raw   = fetch_extended_data(start_year=START_YEAR, start_month=1,
                             fetch_15m=True, fetch_1m=False, fetch_flow=False)
df_1h  = add_indicators(raw["1H"])
df_15m = add_indicators(raw["15M"])
print(f"  1H : {len(df_1h):,} bar  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
print(f"  15M: {len(df_15m):,} bar")

# ── 2. Array numpy ────────────────────────────────────────────────────────────
print("\n[2/6] Preparazione array …")
IDX   = df_15m.index
H_arr = IDX.hour
M_arr = IDX.minute
HI    = df_15m["high"].values
LO    = df_15m["low"].values
CL    = df_15m["close"].values
OP    = df_15m["open"].values
N     = len(df_15m)
yr_arr = np.array([t.year for t in IDX], dtype=int)

prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
atr_1h_map = df_1h["atr_14"].clip(lower=1.0).to_dict()
fallback   = df_15m["atr_14"].clip(lower=1.0).values
ATR_1H     = np.array([atr_1h_map.get(t, np.nan) for t in prev_1h], dtype=float)
ATR_1H     = np.where(np.isnan(ATR_1H), fallback, ATR_1H)

# ── 3. Raccolta eventi ────────────────────────────────────────────────────────
print("\n[3/6] Raccolta eventi NY ORB …")

events: list[dict] = []
seen_dates: set = set()

for i in range(N - MAX_HOLD - 2):
    if H_arr[i] != NY_HOUR or M_arr[i] != NY_MIN:
        continue
    d = IDX[i].normalize()
    if d in seen_dates:
        continue
    seen_dates.add(d)

    op_i    = OP[i]
    cl_i    = CL[i]
    body_hi = max(op_i, cl_i)
    body_lo = min(op_i, cl_i)
    rw      = body_hi - body_lo
    if rw < MIN_BODY:
        continue

    atr_i = ATR_1H[i]
    if np.isnan(atr_i) or atr_i <= 0:
        continue
    rba = rw / atr_i

    for j in range(i + 1, min(i + MAX_HOLD + 1, N - MAX_HOLD - 1)):
        if IDX[j].hour >= NY_END_HOUR:
            break

        direction = None
        if CL[j] > body_hi:
            direction = "long"
        elif CL[j] < body_lo:
            direction = "short"

        if direction is not None:
            events.append({
                "direction": direction,
                "entry_i":   j,
                "entry_px":  CL[j],
                "body_hi":   body_hi,
                "body_lo":   body_lo,
                "rw":        rw,
                "atr_1h":    atr_i,
                "rba":       rba,
                "year":      yr_arr[i],
                "ts":        IDX[i],
            })
            break

n_long  = sum(1 for e in events if e["direction"] == "long")
n_short = sum(1 for e in events if e["direction"] == "short")
print(f"  Totale eventi: {len(events)}  (long={n_long}, short={n_short})")

# ── 4. Walk-Forward ───────────────────────────────────────────────────────────
print("\n[4/6] Walk-Forward Backtest …")
data_start = IDX[0]
data_end   = IDX[-1]
windows    = _wf_windows(data_start, data_end)
print(f"  Finestre: {len(windows)}  ({WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / step {WF_STEP_M}m)")

# Per-window KPIs (each window starts fresh at INIT_CAP, for reporting only)
window_rets:     list[float] = []
window_meta:     list[dict]  = []
all_oos_events:  list[dict]  = []   # collected for sequential OOS equity

for wid, (tr_s, tr_e, oo_e) in enumerate(windows):
    is_evs = [e for e in events if tr_s <= e["ts"] < tr_e]
    oo_evs = [e for e in events if tr_e <= e["ts"] < oo_e]

    if len(is_evs) < 5 or len(oo_evs) < 3:
        continue

    # Per-window backtest (fresh equity, for per-window KPI table)
    oo_t, oo_eq = run_ny_orb_backtest(
        oo_evs, HI, LO, CL, IDX, N,
        tp_frac=TP_FRAC, sl_buf=SL_BUF, rba_max=RBA_MAX,
        window_id=wid,
    )
    if not oo_t:
        continue

    kp = _kpis(oo_eq)
    window_rets.append(kp["total_return"])
    window_meta.append({
        "wid":    wid,
        "tr_s":   tr_s.date(), "tr_e": tr_e.date(),
        "oo_e":   oo_e.date(),
        "n_is":   len(is_evs),
        "n_oos":  len(oo_t),
        **{k: round(v, 4) for k, v in kp.items()},
    })
    all_oos_events.extend(oo_evs)

print(f"  Finestre valide: {len(window_meta)}")

# Sequential OOS equity: single backtest on all OOS events in chronological
# order, equity compounding properly (avoids the "35×INIT_CAP" artifact).
all_oos_events_sorted = sorted(all_oos_events, key=lambda e: e["ts"])
oos_trades_all, oos_equity = run_ny_orb_backtest(
    all_oos_events_sorted, HI, LO, CL, IDX, N,
    tp_frac=TP_FRAC, sl_buf=SL_BUF, rba_max=RBA_MAX,
)
print(f"  Trade OOS (sequenziali): {len(oos_trades_all)}")

# ── 5. Statistiche OOS ───────────────────────────────────────────────────────
print("\n[5/6] Test statistici OOS …")

if not oos_trades_all:
    print("  NESSUN trade OOS.")
    sys.exit(1)

kp_oos   = _kpis(oos_equity)
oos_df   = pd.DataFrame([t.__dict__ for t in oos_trades_all])
oos_df   = oos_df.sort_values("entry_ts").reset_index(drop=True)

oos_rets  = oos_df["net_pnl"].values
wins_oos  = (oos_rets > 0).sum()
total_oos = len(oos_rets)
wr_oos    = wins_oos / total_oos

tp_dist_oos = np.mean(np.abs(oos_df["tp_price"].values - oos_df["entry_price"].values)
                      / oos_df["entry_price"].values * 100)
sl_dist_oos = np.mean(np.abs(oos_df["stop_price"].values - oos_df["entry_price"].values)
                      / oos_df["entry_price"].values * 100)
rr_oos = tp_dist_oos / sl_dist_oos if sl_dist_oos > 0 else 0
be_oos  = 1 / (1 + rr_oos) * 100 if rr_oos > 0 else 50.0

# Binomial test
binom_p   = st.binomtest(int(wins_oos), int(total_oos),
                          be_oos / 100, alternative="greater").pvalue

# t-test OOS daily returns
oos_df_tmp = oos_df.copy()
oos_df_tmp["date"] = pd.to_datetime(oos_df_tmp["exit_ts"]).dt.normalize()
daily_ret = oos_df_tmp.groupby("date")["net_pnl"].sum()
t_stat, t_p = st.ttest_1samp(daily_ret.values, 0) if len(daily_ret) > 1 else (0, 1)

print(f"  Trade OOS      : {total_oos}")
print(f"  WR             : {wr_oos:.1%}")
print(f"  Breakeven WR   : {be_oos:.1f}%")
print(f"  Calmar OOS     : {kp_oos['calmar']:.3f}")
print(f"  Sharpe OOS     : {kp_oos['sharpe']:.3f}")
print(f"  Max DD OOS     : {kp_oos['max_dd']:.1%}")
print(f"  Return OOS     : {kp_oos['total_return']:.1%}")
print(f"  Binomial p     : {binom_p:.4f}  {'✓ SIGNIFICATIVO' if binom_p < 0.05 else '✗ non significativo'}")
print(f"  t-test daily p : {t_p:.4f}")

# Breakdown per anno OOS
print(f"\n  Breakdown per anno (OOS):")
print(f"  {'Anno':>6} {'N':>5} {'WR%':>7} {'RetUSD':>10} {'Calmar':>8}")
print(f"  {SEP[:52]}")
for yr in sorted(oos_df["year"].unique()):
    sub = oos_df[oos_df["year"] == yr]
    w = (sub["net_pnl"] > 0).sum()
    wr_y = w / len(sub) * 100
    ret_y = sub["net_pnl"].sum()
    print(f"  {yr:>6} {len(sub):>5} {wr_y:>6.1f}% {ret_y:>+10.0f}")

# ── 6. Monte Carlo ────────────────────────────────────────────────────────────
print(f"\n[6/6] Monte Carlo ({N_SIMS} simulazioni) …")

mc_trades_df = oos_df[["entry_price", "exit_price", "net_pnl",
                        "gross_pnl", "total_fees"]].copy()
mc_result = run_monte_carlo(mc_trades_df, INIT_CAP, N_SIMS)

mc_fin  = mc_result["total_return"] * 100
mc_dd   = mc_result["max_drawdown"] * 100
p_profit = (mc_result["total_return"] > 0).mean()
p_ruin   = mc_result.get("p_ruin", (mc_result["total_return"] < -0.5).mean())

print(f"  P(profit)      : {p_profit:.1%}")
print(f"  P(ruin)        : {p_ruin:.1%}")
print(f"  Median return  : {np.median(mc_fin):.1f}%")
print(f"  5th pct return : {np.percentile(mc_fin, 5):.1f}%")
print(f"  95th pct DD    : {np.percentile(mc_dd, 95):.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# HTML Report
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nGenerazione report HTML …")

img_equity   = _plot_equity(oos_equity, df_1h["close"])
img_wf       = _plot_wf_returns(window_rets)
img_mc       = _plot_mc(mc_result)
img_mc_paths = _plot_mc_paths(mc_result)
img_trades   = _plot_trade_dist(oos_df)

# Verdetto finale
verdict_ok  = (kp_oos["calmar"] > 0.5
               and kp_oos["total_return"] > 0
               and binom_p < 0.05
               and p_profit > 0.5)
verdict_col = _GRN if verdict_ok else _RED
verdict_txt = "✅ STRATEGIA VALIDATA" if verdict_ok else "✗ NON VALIDATA OOS"

# Strategy card
strat_body = "".join([
    _kv("TP frac", f"{TP_FRAC}× body range"),
    _kv("SL buf", f"{SL_BUF}× ATR_1H sotto body_lo / sopra body_hi"),
    _kv("RBA max", "nessun filtro"),
    _kv("Max hold", f"{MAX_HOLD} barre 15M (~8h)"),
    _kv("Fee", f"{FEE*100:.2f}% per lato"),
    _kv("Risk per trade", f"{RISK_PCT*100:.0f}% equity"),
    _kv("Capitale iniziale", f"${INIT_CAP:,.0f}"),
])

# IS scan card
is_body = "".join([
    _kv("Trade totali IS", f"{len(events)}"),
    _kv("Long / Short", f"{n_long} / {n_short}"),
    _kv("WR IS (best combo)", "59.42%"),
    _kv("R:R IS", "0.79"),
    _kv("BE WR IS", "55.88%"),
    _kv("ExpPnL IS", "+0.0505%"),
    _kv("p-value IS (binomial)", "0.0006"),
])

# OOS card
oos_body = "".join([
    _kv("Trade OOS",     f"{total_oos}"),
    _kv("Win Rate OOS",  _style_num(wr_oos, ".1%")),
    _kv("Breakeven WR",  f"{be_oos:.1f}%"),
    _kv("Margin WR",     _style_num(wr_oos - be_oos/100, ".2%")),
    _kv("Total Return",  _style_num(kp_oos["total_return"], ".1%")),
    _kv("Calmar OOS",    _style_num(kp_oos["calmar"], ".3f")),
    _kv("Sharpe OOS",    _style_num(kp_oos["sharpe"], ".3f")),
    _kv("Max DD OOS",    _style_num(kp_oos["max_dd"], ".1%")),
    _kv("Binomial p",    f"{binom_p:.4f}"),
    _kv("t-test daily p", f"{t_p:.4f}"),
])

# Monte Carlo card
mc_body = "".join([
    _kv("Simulazioni",      f"{N_SIMS:,}"),
    _kv("P(profit)",        _style_num(p_profit, ".1%")),
    _kv("P(ruin < -50%)",   _style_num(p_ruin, ".1%", good=-1)),
    _kv("Median return",    f"{np.median(mc_fin):.1f}%"),
    _kv("5th pct return",   f"{np.percentile(mc_fin, 5):.1f}%"),
    _kv("95th pct max DD",  f"{np.percentile(mc_dd, 95):.1f}%"),
])

# WF table
wf_rows = [
    [m["wid"], f"{m['tr_s']}→{m['tr_e']}", f"{m['oo_e']}",
     m["n_is"], m["n_oos"],
     f"{m['total_return']:.1%}", f"{m['calmar']:.2f}",
     f"{m['max_dd']:.1%}"]
    for m in window_meta
]
wf_body = _table(
    ["#", "IS range", "OOS end", "N IS", "N OOS", "Ret", "Calmar", "Max DD"],
    wf_rows,
)

# Annual breakdown OOS
ann_rows = []
for yr in sorted(oos_df["year"].unique()):
    sub  = oos_df[oos_df["year"] == yr]
    w    = (sub["net_pnl"] > 0).sum()
    wr_y = w / len(sub) * 100
    ret_y = sub["net_pnl"].sum()
    color = _GRN if ret_y > 0 else _RED
    ann_rows.append([
        yr, len(sub), f"{wr_y:.1f}%",
        f'<span style="color:{color}">{ret_y:+,.0f}</span>',
    ])
annual_body = _table(["Anno", "N trade", "WR%", "Net PnL (USD)"], ann_rows)

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>NY ORB — Validation Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{_BG};color:{_TEXT};font-family:monospace;font-size:13px;padding:24px}}
  h1{{color:{_ACC};margin-bottom:8px}}
  h2{{color:{_YEL};margin:24px 0 12px;font-size:15px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .verdict{{background:{_CARD};border:2px solid {verdict_col};border-radius:10px;
            padding:20px;margin-bottom:24px;text-align:center;
            font-size:20px;font-weight:bold;color:{verdict_col}}}
</style>
</head>
<body>
<h1>NY Opening Range Breakout — Full Validation Report</h1>
<p style="color:#9e9e9e;margin-bottom:20px">
  BTCUSDT 15M &nbsp;|&nbsp; 2020-01-01 → 2026-05-31 &nbsp;|&nbsp;
  TP={TP_FRAC}×body &nbsp; SL={SL_BUF}×ATR_1H &nbsp; RBA=∞
  &nbsp;|&nbsp; WF {WF_TRAIN_M}m/{WF_OOS_M}m
</p>

<div class="verdict">{verdict_txt}</div>

<div class="grid2">
  {_card("Parametri strategia", strat_body)}
  {_card("IS Scan (best combo)", is_body)}
</div>
<div class="grid2">
  {_card("OOS Performance", oos_body)}
  {_card("Monte Carlo", mc_body)}
</div>

<h2>OOS Equity Curve</h2>
{_imgt(img_equity)}

<h2>OOS Return per finestra Walk-Forward</h2>
{_imgt(img_wf)}

<h2>Monte Carlo — Distribuzione Rendimento Finale</h2>
{_imgt(img_mc)}

<h2>Monte Carlo — Percorsi Equity</h2>
{_imgt(img_mc_paths)}

<h2>P&amp;L per Trade (OOS)</h2>
{_imgt(img_trades)}

<h2>Walk-Forward Windows</h2>
{_card("Dettaglio finestre", wf_body)}

<h2>Breakdown Annuale (OOS)</h2>
{_card("Rendimento per anno", annual_body)}

<p style="color:#555;margin-top:32px;font-size:11px">
  Generated: NY ORB Body Breakout — Full validation pipeline
  (Walk-Forward 6m/2m, Monte Carlo 5000 sims, fee {FEE*100:.2f}%/side)
</p>
</body>
</html>"""

out_dir = Path("reports")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "report_ny_orb.html"
out_path.write_text(html, encoding="utf-8")

print(f"\n{SEP2}")
print(f"Report scritto in: {out_path}")
print(SEP2)
print(f"  Calmar OOS    : {kp_oos['calmar']:+.3f}")
print(f"  Return OOS    : {kp_oos['total_return']:+.1%}")
print(f"  Max DD OOS    : {kp_oos['max_dd']:+.1%}")
print(f"  WR OOS        : {wr_oos:.1%}  (BE={be_oos:.1f}%)")
print(f"  Binomial p    : {binom_p:.4f}")
print(f"  P(profit) MC  : {p_profit:.1%}")
print(f"  Verdetto      : {verdict_txt}")
print(SEP2)
