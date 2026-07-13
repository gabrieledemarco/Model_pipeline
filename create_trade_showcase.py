#!/usr/bin/env python3
"""
create_trade_showcase.py
════════════════════════════════════════════════════════════════════════════
Publication artifacts for the SMC Structure 1H strategy (C1 baseline / C2
funding-filtered — see create_funding_momentum_filter_report.py):

  1. $100 buy-and-hold vs $100-flat-bet-per-trade equity curves (C1, C2),
     full analyzed period 2022-01 -> 2026-06.
  2. 3 illustrative trades per strategy (best winner / worst loser / a
     representative median trade — not cherry-picked wins only), each
     rendered as an OHLCV candlestick chart with the structural levels the
     strategy actually used (CHoCH bias, swing pivot high/low = the
     discount/premium zone boundary, ATR) plus entry/SL/TP/exit markers.

Equity convention ("in ogni trade investi 100", non-compounding): each trade
risks a FIXED $100 notional-equivalent regardless of running balance — the
running total is 100 + cumsum(100 * pct_return_per_trade), flat between
trades, not the risk-1%-of-current-equity compounding sizing the validation
report uses. This is a different, simpler accounting convention chosen
specifically for this comparison chart, not a re-statement of the validated
backtest's own equity curve.

No partial exits: this engine's exit_mode is single-TP (one target, one
stop, one time-stop) — there is no partial-close event to plot. Noted
explicitly on each trade chart instead of silently omitting it.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_binance_vision_funding
from src.strategy.indicators import atr as atr_fn
from src.strategy.smc import compute_smc_features, smc_trend_signal
from src.strategy.mtf_swing import causal_trend_state
from src.strategy.report_html import BG, PANEL, BORDER, WHITE, GRAY, GREEN, RED, GOLD, BLUE

plt.style.use("dark_background")

START_YEAR, START_MONTH = 2022, 1
END_YEAR, END_MONTH = 2026, 6
FEE = 0.0006
TREND_SWING_LEN, TREND_INTERNAL_LEN = 30, 5
SL_BUFFER_ATR = 0.15
MIN_RR = 1.2
FALLBACK_SL_ATR = 1.0
FALLBACK_TP_ATR = 2.0
TIME_STOP_BARS = 96
COOLDOWN_BARS = 4
FUNDING_PCTILE_WINDOW = 180
AVOID_DECILE = 0.10

OUT_DIR = Path("reports/trade_showcase")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA + SIGNAL ENGINE (identical to create_funding_momentum_filter_report.py)
# ═══════════════════════════════════════════════════════════════════════════
print("[DATA] Loading 1H OHLCV + funding rate...")
df1h = fetch_binance_vision_klines("1h", START_YEAR, START_MONTH, END_YEAR, END_MONTH, workers=6)
df1h["atr_14"] = atr_fn(df1h["high"], df1h["low"], df1h["close"], 14)
N1 = len(df1h)
IDX1 = df1h.index
HI1 = df1h["high"].values.astype(float)
LO1 = df1h["low"].values.astype(float)
CL1 = df1h["close"].values.astype(float)
OP1 = df1h["open"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0, df1h["atr_14"].shift(1).values, 1.0)

funding_8h = fetch_binance_vision_funding(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
funding_pctile_8h = funding_8h.rolling(FUNDING_PCTILE_WINDOW, min_periods=60).apply(
    lambda x: (x <= x.iloc[-1]).mean(), raw=False)
FUND_PCTILE1 = funding_pctile_8h.reindex(IDX1, method="ffill").values

smc1h = compute_smc_features(df1h, swing_len=TREND_SWING_LEN, internal_len=TREND_INTERNAL_LEN, prefix="smc")
bias1h = smc_trend_signal(smc1h, prefix="smc").values.astype(int)
zone_bull1h = ((smc1h["smc_ob_bull_in"].values == 1) | (smc1h["smc_in_discount"].values == 1))
zone_bear1h = ((smc1h["smc_ob_bear_in"].values == 1) | (smc1h["smc_in_premium"].values == 1))
struct1h = causal_trend_state(CL1, HI1, LO1, left=8, right=8)
sl_basis_low = struct1h["last_pivot_low"].values
sl_basis_high = struct1h["last_pivot_high"].values
tp_basis_high = struct1h["target_high"].values
tp_basis_low = struct1h["target_low"].values
TREND_SIGNAL1 = np.where((bias1h == 1) & zone_bull1h, 1, np.where((bias1h == -1) & zone_bear1h, -1, 0))


def simulate(mode: str) -> pd.DataFrame:
    trades: list[dict] = []
    in_pos = False
    dirn = 0
    entry_px = sl = tp = 0.0
    entry_i = 0
    hold_end = 0
    last_exit = -COOLDOWN_BARS

    for i in range(N1):
        if in_pos:
            hit = None
            if dirn == 1:
                if LO1[i] <= sl: hit = "SL"
                elif HI1[i] >= tp: hit = "TP"
            else:
                if HI1[i] >= sl: hit = "SL"
                elif LO1[i] <= tp: hit = "TP"
            if hit is None and i >= hold_end:
                hit = "TIME"
            if hit is not None:
                exit_px = sl if hit == "SL" else (tp if hit == "TP" else CL1[i])
                pct_ret = (exit_px - entry_px) / entry_px * dirn - FEE * 2
                trades.append(dict(
                    entry_time=IDX1[entry_i], exit_time=IDX1[i], entry_i=entry_i, exit_i=i,
                    direction=dirn, entry_price=entry_px, exit_price=exit_px, sl=sl, tp=tp,
                    exit_reason=hit, pct_return=pct_ret,
                    bias=int(bias1h[entry_i]), atr_entry=float(ATR1[entry_i]),
                    sl_basis=float(sl_basis_low[entry_i] if dirn == 1 else sl_basis_high[entry_i]),
                    tp_basis=float(tp_basis_high[entry_i] if dirn == 1 else tp_basis_low[entry_i]),
                ))
                in_pos = False
                last_exit = i
            continue

        if i - last_exit < COOLDOWN_BARS:
            continue
        d = int(TREND_SIGNAL1[i])
        if d == 0:
            continue

        fp = FUND_PCTILE1[i]
        if mode == "avoid" and not np.isnan(fp):
            if d == 1 and fp < AVOID_DECILE:
                continue
            if d == -1 and fp > (1 - AVOID_DECILE):
                continue

        atrv = ATR1[i]
        if atrv <= 0:
            continue
        px = CL1[i]
        sl_level = sl_basis_low[i] if d == 1 else sl_basis_high[i]
        tp_level = tp_basis_high[i] if d == 1 else tp_basis_low[i]
        slp = (sl_level - d * SL_BUFFER_ATR * atrv) if not np.isnan(sl_level) else (px - d * FALLBACK_SL_ATR * atrv)
        tpp = tp_level if not np.isnan(tp_level) else (px + d * FALLBACK_TP_ATR * atrv)
        risk = abs(px - slp)
        if risk <= 0:
            continue
        if abs(tpp - px) / risk < MIN_RR:
            tpp = px + d * max(MIN_RR * risk, FALLBACK_TP_ATR * atrv)

        dirn, entry_px, sl, tp, entry_i = d, px, slp, tpp, i
        hold_end = i + TIME_STOP_BARS
        in_pos = True

    return pd.DataFrame(trades)


print("[BACKTEST] C1 baseline + C2 avoid-extreme-funding...")
c1 = simulate("baseline")
c2 = simulate("avoid")
print(f"  C1: {len(c1)} trades   C2: {len(c2)} trades")

# ═══════════════════════════════════════════════════════════════════════════
# 2. EQUITY CURVES — $100 buy&hold vs $100-flat-bet-per-trade (non-compounding)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[EQUITY] Building $100 buy&hold vs flat-bet equity curves...")

def flat_bet_equity(trades: pd.DataFrame, base: float = 100.0) -> pd.Series:
    if trades.empty:
        return pd.Series([base], index=[IDX1[0]])
    idx = [IDX1[0]] + list(trades["exit_time"])
    vals = [base] + list(base + (trades["pct_return"] * base).cumsum())
    return pd.Series(vals, index=pd.DatetimeIndex(idx))

bh_equity = 100.0 * (CL1 / CL1[0])
bh_series = pd.Series(bh_equity, index=IDX1)
c1_equity = flat_bet_equity(c1)
c2_equity = flat_bet_equity(c2)

fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
ax.set_facecolor(PANEL)
ax.plot(bh_series.index, bh_series.values, color=GRAY, lw=1.3, label="Buy & Hold BTCUSDT ($100)")
ax.step(c1_equity.index, c1_equity.values, color=GOLD, lw=1.6, where="post",
        label=f"C1 Baseline — $100/trade ({len(c1)} trades)")
ax.step(c2_equity.index, c2_equity.values, color=BLUE, lw=1.6, where="post",
        label=f"C2 Avoid Extreme Funding — $100/trade ({len(c2)} trades)")
ax.axhline(100, color=GRAY, lw=0.7, ls=":")
ax.set_title("BTCUSDT — Buy & Hold vs SMC Structure 1H (start 100 USD, non-compounding, fixed 100 USD/trade)",
              color=WHITE, fontsize=11, pad=10)
ax.set_ylabel("Value ($)", color=GRAY, fontsize=9)
ax.tick_params(colors=GRAY, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(BORDER)
ax.grid(True, alpha=0.12, color=GRAY, linestyle="--")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.legend(fontsize=9, labelcolor=WHITE, framealpha=0.15, loc="upper left")
fig.patch.set_facecolor(BG)
fig.tight_layout()
fig.savefig(OUT_DIR / "equity_comparison.png", dpi=140, facecolor=BG)
plt.close(fig)
print(f"  saved {OUT_DIR / 'equity_comparison.png'}")
print(f"  Final: Buy&Hold=${bh_series.iloc[-1]:.2f}  C1=${c1_equity.iloc[-1]:.2f}  C2=${c2_equity.iloc[-1]:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. TRADE SELECTION — best winner / worst loser / median trade, per strategy
# ═══════════════════════════════════════════════════════════════════════════
def pick_trades(trades: pd.DataFrame) -> dict[str, pd.Series]:
    by_ret = trades.sort_values("pct_return")
    worst = by_ret.iloc[0]
    best = by_ret.iloc[-1]
    median_idx = (by_ret["pct_return"] - by_ret["pct_return"].median()).abs().idxmin()
    median = by_ret.loc[median_idx]
    return {"best_winner": best, "worst_loser": worst, "median_trade": median}

# ═══════════════════════════════════════════════════════════════════════════
# 4. TRADE CHART — OHLCV candles + structural levels + entry/SL/TP/exit
# ═══════════════════════════════════════════════════════════════════════════
def plot_trade(trade: pd.Series, label: str, strategy_name: str, fname: str) -> None:
    entry_i, exit_i = int(trade["entry_i"]), int(trade["exit_i"])
    lo_i = max(0, entry_i - 20)
    hi_i = min(N1 - 1, exit_i + 10)
    window = slice(lo_i, hi_i + 1)
    idx_w = IDX1[window]
    op_w, hi_w, lo_w, cl_w = OP1[window], HI1[window], LO1[window], CL1[window]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(PANEL)

    bar_w = pd.Timedelta(hours=0.7)
    for t, o, h, l, c in zip(idx_w, op_w, hi_w, lo_w, cl_w):
        color = GREEN if c >= o else RED
        ax.plot([t, t], [l, h], color=color, lw=0.8, zorder=2)
        rect = Rectangle((mdates.date2num(t) - bar_w.total_seconds() / 86400 / 2, min(o, c)),
                          bar_w.total_seconds() / 86400, max(abs(c - o), 1e-6),
                          facecolor=color, edgecolor=color, zorder=3)
        ax.add_patch(rect)

    d = int(trade["direction"])
    dir_label = "LONG" if d == 1 else "SHORT"
    entry_t, exit_t = trade["entry_time"], trade["exit_time"]

    # Structural levels the strategy actually used (CHoCH swing pivots =
    # the discount/premium zone boundary at entry time).
    sl_basis, tp_basis = trade["sl_basis"], trade["tp_basis"]
    if sl_basis == sl_basis:
        ax.axhline(sl_basis, color=GRAY, lw=0.9, ls="-.", alpha=0.6)
        ax.text(idx_w[0], sl_basis, " swing pivot (SL basis)", color=GRAY, fontsize=7, va="bottom")
    if tp_basis == tp_basis:
        ax.axhline(tp_basis, color=GRAY, lw=0.9, ls="-.", alpha=0.6)
        ax.text(idx_w[0], tp_basis, " swing pivot (TP basis)", color=GRAY, fontsize=7, va="bottom")

    # Entry / SL / TP / Exit
    ax.axhline(trade["entry_price"], color=WHITE, lw=1.0, ls="--", alpha=0.8)
    ax.axhline(trade["sl"], color=RED, lw=1.2, ls="--")
    ax.axhline(trade["tp"], color=GREEN, lw=1.2, ls="--")
    ax.text(idx_w[-1], trade["entry_price"], " entry", color=WHITE, fontsize=8, va="bottom", ha="right")
    ax.text(idx_w[-1], trade["sl"], " SL", color=RED, fontsize=8, va="bottom", ha="right")
    ax.text(idx_w[-1], trade["tp"], " TP", color=GREEN, fontsize=8, va="bottom", ha="right")

    marker_entry = "^" if d == 1 else "v"
    ax.scatter([entry_t], [trade["entry_price"]], marker=marker_entry, s=180,
               color=GOLD, edgecolor=WHITE, zorder=5, label="Entry")
    exit_color = GREEN if trade["pct_return"] > 0 else RED
    ax.scatter([exit_t], [trade["exit_price"]], marker="x", s=160,
               color=exit_color, linewidths=3, zorder=5, label=f"Exit ({trade['exit_reason']})")

    subtitle = (
        f"{strategy_name} — {label} — {dir_label}  |  bias(CHoCH)={'+1 bull' if trade['bias']==1 else '-1 bear'}  "
        f"ATR@entry={trade['atr_entry']:.1f}  |  return={trade['pct_return']*100:+.2f}%  "
        f"exit={trade['exit_reason']}  |  no partial exits (single-TP engine)"
    )
    ax.set_title(subtitle, color=WHITE, fontsize=9.5, pad=10)
    ax.set_ylabel("Price ($)", color=GRAY, fontsize=9)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    fig.autofmt_xdate()
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.10, color=GRAY, linestyle="--")
    ax.legend(fontsize=8, labelcolor=WHITE, framealpha=0.15, loc="best")
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"  saved {OUT_DIR / fname}")


print("\n[TRADES] Selecting + charting 3 trades per strategy...")
for strat_name, trades, prefix in [("C1 Baseline", c1, "c1"), ("C2 Avoid Extreme Funding", c2, "c2")]:
    picks = pick_trades(trades)
    for label, trade in picks.items():
        plot_trade(trade, label.replace("_", " ").title(), strat_name, f"{prefix}_{label}.png")

print("\n[DONE]")
