"""
Visualization module – produces all charts for the BTCUSDT strategy analysis.

Charts are grouped into three packages:

GROUP A – Signal Analysis
  A1  signal_score_distribution   score histogram by direction
  A2  alpha_decay                 mean fwd-return vs horizon (long/short/neutral)
  A3  signal_component_corr       heatmap of inter-component correlations
  A4  score_vs_fwd_return         scatter composite score vs 4H fwd return
  A5  score_quantile_returns      bar chart avg fwd-return per score decile

GROUP B – Market Structure
  B1  market_overview             daily price + EMAs + volume
  B2  oi_vs_price                 OI trend overlaid with price + funding rate
  B3  cyclicality_monthly         monthly return bars + heat-strip
  B4  cyclicality_dow_hour        day-of-week & hour-of-day returns
  B5  fft_cycles                  dominant cycle periods by amplitude

GROUP C – Backtest Performance
  C1  equity_curve                equity curve + drawdown panel
  C2  trade_distribution          histogram + KDE of per-trade net returns
  C3  mae_mfe_scatter             MAE vs MFE scatter coloured by outcome
  C4  trade_duration_pnl          duration vs net PnL scatter
  C5  regime_session_bars         side-by-side win-rate by regime and session
  C6  rolling_sharpe              rolling 7-day Sharpe
  C7  monthly_pnl_heatmap         month × year heat-map
  C8  score_vs_outcome            win-rate + avg PnL per score quintile
  C9  tp_sl_breakdown             stacked bar of exit reasons
  C10 signal_component_timeline   stacked-area of daily component contributions
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
WHITE   = "#e6edf3"
GRAY    = "#8b949e"
GREEN   = "#2ea043"
RED     = "#f85149"
GOLD    = "#e3b341"
BLUE    = "#58a6ff"
PURPLE  = "#d2a8ff"
ORANGE  = "#f0883e"
TEAL    = "#39d353"
PINK    = "#ff7eb6"
DPI     = 150
SAVE    = Path("reports/charts")
SAVE.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    p = SAVE / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


def _ax(ax: plt.Axes, title: str = "",
        xlabel: str = "", ylabel: str = "") -> plt.Axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.12, color=GRAY, linestyle="--")
    if title:  ax.set_title(title,  color=WHITE, fontsize=9, pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY,  fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY,  fontsize=8)
    return ax


def _fig(w=14, h=8):
    return plt.figure(figsize=(w, h), facecolor=BG)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A – Signal Analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_signal_score_distribution(signals: pd.DataFrame) -> Path:
    """A1: Histogram of composite scores coloured by signal direction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("A1 · Signal Score Distribution", color=WHITE,
                 fontsize=13, fontweight="bold")

    comp = signals["composite"]
    sig  = signals["signal"]

    # Left: histogram
    ax = _ax(axes[0], "Composite Score Frequency",
             "Composite Score", "Count")
    bins = np.linspace(comp.min() - 1, comp.max() + 1, 35)
    for val, col, lbl in [( 1, GREEN,  "Long  (≥ 5)"),
                           (-1, RED,    "Short (≤ -5)"),
                           ( 0, GRAY,   "Neutral")]:
        ax.hist(comp[sig == val], bins=bins, color=col,
                alpha=0.65, label=lbl, edgecolor="none")
    ax.axvline(5,  color=GREEN, lw=1.4, ls="--", alpha=0.8)
    ax.axvline(-5, color=RED,   lw=1.4, ls="--", alpha=0.8)
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

    # Right: signal count pie
    ax2 = _ax(axes[1], "Signal Direction Split")
    counts  = [int((sig == 1).sum()), int((sig == -1).sum()), int((sig == 0).sum())]
    labels  = [f"Long\n{counts[0]}", f"Short\n{counts[1]}", f"Neutral\n{counts[2]}"]
    explode = [0.03, 0.03, 0]
    wedges, _, autotxt = ax2.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=[GREEN, RED, GRAY], startangle=90,
        explode=explode, textprops={"color": WHITE, "fontsize": 8})
    for wt in autotxt:
        wt.set_color(BG); wt.set_fontweight("bold")
    ax2.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "A1_signal_score_distribution")


def plot_alpha_decay(decay_df: pd.DataFrame) -> Path:
    """A2: Mean forward return vs horizon per signal direction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("A2 · Alpha-Decay Analysis", color=WHITE,
                 fontsize=13, fontweight="bold")

    palette = {"Long": GREEN, "Short": RED, "Neutral": GRAY}

    # Left: mean fwd return (bars %)
    ax = _ax(axes[0], "Mean Forward Log-Return by Horizon",
             "Horizon (hours)", "Mean Log-Return")
    for sig, col in palette.items():
        sub = decay_df[decay_df["signal"] == sig].sort_values("horizon_h")
        ax.plot(sub["horizon_h"], sub["mean_ret"] * 100,
                marker="o", color=col, label=sig, lw=2, ms=5)
        ax.fill_between(sub["horizon_h"],
                        (sub["mean_ret"] - sub["std_ret"] / np.sqrt(sub["n_obs"].clip(1))) * 100,
                        (sub["mean_ret"] + sub["std_ret"] / np.sqrt(sub["n_obs"].clip(1))) * 100,
                        color=col, alpha=0.12)
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

    # Right: IC (Spearman) by horizon
    ax2 = _ax(axes[1], "Spearman IC vs Horizon (Long signal)",
              "Horizon (hours)", "Spearman IC")
    for sig, col in palette.items():
        sub = decay_df[decay_df["signal"] == sig].sort_values("horizon_h")
        ax2.bar(sub["horizon_h"] + {"Long": -0.6, "Short": 0.6, "Neutral": 0}[sig],
                sub["IC_spearman"], width=0.5, color=col, alpha=0.8, label=sig)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "A2_alpha_decay")


def plot_signal_component_correlation(signals: pd.DataFrame) -> Path:
    """A3: Heatmap of pairwise correlations between signal components."""
    cols  = ["s_weekly", "s_daily", "s_4h", "s_1h",
             "s_oi", "s_funding", "s_vol", "s_cycle", "composite"]
    avail = [c for c in cols if c in signals.columns]
    corr  = signals[avail].corr()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    fig.suptitle("A3 · Signal Component Correlation Matrix",
                 color=WHITE, fontsize=13, fontweight="bold")
    _ax(ax)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", mask=mask,
                cmap=sns.diverging_palette(10, 140, s=80, l=45, n=21),
                vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 8, "color": WHITE},
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.set_xticklabels(avail, rotation=35, ha="right", color=WHITE)
    ax.set_yticklabels(avail, rotation=0, color=WHITE)

    plt.tight_layout()
    return _save(fig, "A3_signal_component_correlation")


def plot_score_vs_fwd_return(signals: pd.DataFrame,
                              fwd: pd.DataFrame,
                              col: str = "fwd_4h") -> Path:
    """A4: Scatter of composite score vs 4H forward log-return."""
    df = pd.DataFrame({"score": signals["composite"],
                        "fwd":   fwd[col],
                        "sig":   signals["signal"]}).dropna()

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    _ax(ax, f"A4 · Composite Score vs {col.replace('_','+')} Forward Return",
        "Composite Score", "Forward Log-Return")

    cmap = {1: GREEN, -1: RED, 0: GRAY}
    for sv, col_p in cmap.items():
        sub = df[df["sig"] == sv]
        ax.scatter(sub["score"], sub["fwd"] * 100,
                   c=col_p, alpha=0.3, s=15, linewidths=0, rasterized=True)

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(df["score"], df["fwd"])
    xs = np.linspace(df["score"].min(), df["score"].max(), 100)
    ax.plot(xs, (slope * xs + intercept) * 100,
            color=GOLD, lw=2, label=f"OLS  r={r:.3f}  p={p:.3g}")
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.axvline( 5, color=GREEN, lw=1.2, ls=":", alpha=0.7)
    ax.axvline(-5, color=RED,   lw=1.2, ls=":", alpha=0.7)
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

    plt.tight_layout()
    return _save(fig, "A4_score_vs_fwd_return")


def plot_score_quantile_returns(quantile_df: pd.DataFrame) -> Path:
    """A5: Average forward return per score decile (bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    _ax(ax, "A5 · Average 4H Forward Return by Score Decile",
        "Score Decile", "Mean Forward Log-Return (%)")

    colors = [GREEN if r > 0 else RED for r in quantile_df["avg_fwd_ret"]]
    bars   = ax.bar(quantile_df.index, quantile_df["avg_fwd_ret"] * 100,
                    color=colors, alpha=0.85, width=0.7, edgecolor=BORDER)
    ax.axhline(0, color=GRAY, lw=0.9)

    # annotate n_obs
    for bar, row in zip(bars, quantile_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002 * np.sign(bar.get_height()),
                f"n={int(row.n_obs)}", ha="center", va="bottom",
                fontsize=7, color=GRAY)

    ax.set_xticks(quantile_df.index)
    ax.set_xticklabels([f"D{i+1}" for i in quantile_df.index])

    plt.tight_layout()
    return _save(fig, "A5_score_quantile_returns")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B – Market Structure
# ══════════════════════════════════════════════════════════════════════════════

def plot_market_overview(df_1d: pd.DataFrame) -> Path:
    """B1: Daily OHLCV with EMA stack and volume bars."""
    fig = _fig(16, 10)
    fig.suptitle("B1 · BTCUSDT Daily Market Overview (Price + EMA Stack + Volume)",
                 color=WHITE, fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1], hspace=0.04)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _ax(ax1, ylabel="Price (USD)")
    _ax(ax2, ylabel="ATR %")
    _ax(ax3, ylabel="Volume")

    tail = df_1d.tail(500)  # show last 500 days

    # Candles (simplified: green/red line from close)
    for _, row in tail.iterrows():
        color = GREEN if row["close"] >= row["open"] else RED
        ax1.plot([row.name, row.name], [row["low"], row["high"]],
                 color=color, lw=0.5, alpha=0.6)
        ax1.plot([row.name, row.name], [row["open"], row["close"]],
                 color=color, lw=2.0, solid_capstyle="butt")

    # EMAs
    for p, col, lbl in [(21, BLUE, "EMA-21"), (50, ORANGE, "EMA-50"),
                         (200, PURPLE, "EMA-200")]:
        col_nm = f"ema_{p}"
        if col_nm in tail:
            ax1.plot(tail.index, tail[col_nm], color=col, lw=1.2,
                     label=lbl, alpha=0.9)

    ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ATR %
    ax2.fill_between(tail.index, tail["atr_pct"], color=GOLD, alpha=0.6)
    ax2.set_ylim(bottom=0)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Volume
    vol_colors = [GREEN if r["close"] >= r["open"] else RED
                  for _, r in tail.iterrows()]
    ax3.bar(tail.index, tail["volume"], color=vol_colors,
            alpha=0.7, width=0.8)
    ax3.plot(tail.index, tail["vol_sma20"], color=BLUE, lw=1.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", color=GRAY)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, alpha=0.1, color=GRAY, linestyle="--")

    return _save(fig, "B1_market_overview")


def plot_oi_analysis(df_1d: pd.DataFrame, oi_df: pd.DataFrame,
                     funding: pd.Series) -> Path:
    """B2: Open Interest vs price, plus funding rate."""
    fig = _fig(16, 10)
    fig.suptitle("B2 · Open Interest Analysis + Funding Rate",
                 color=WHITE, fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    common = df_1d.index.intersection(oi_df.index)
    price  = df_1d.loc[common, "close"]
    oi     = oi_df.loc[common, "oi"]
    fund   = funding.reindex(common, method="ffill")

    _ax(ax1, ylabel="Price (USD)")
    ax1.plot(common, price, color=BLUE, lw=1.2, label="Close")
    ax1_r = ax1.twinx()
    ax1_r.plot(common, oi / 1e9, color=ORANGE, lw=1.2,
               alpha=0.8, label="OI ($B)", ls="--")
    ax1_r.set_ylabel("OI (Billion USD)", color=ORANGE, fontsize=8)
    ax1_r.tick_params(axis="y", colors=ORANGE, labelsize=8)
    ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # OI change colour-coded
    _ax(ax2, ylabel="OI Change (%)")
    oi_chg = oi_df.loc[common, "oi_chg"].fillna(0) * 100
    colors  = [GREEN if v > 0 else RED for v in oi_chg]
    ax2.bar(common, oi_chg, color=colors, alpha=0.7, width=0.8)
    ax2.axhline(0, color=GRAY, lw=0.8)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Funding rate
    _ax(ax3, ylabel="Funding Rate (%)")
    fund_pct = fund * 100
    ax3.fill_between(common, fund_pct,
                     where=fund_pct > 0, color=RED,    alpha=0.6)
    ax3.fill_between(common, fund_pct,
                     where=fund_pct < 0, color=GREEN, alpha=0.6)
    ax3.axhline(0, color=GRAY, lw=0.8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", color=GRAY)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, alpha=0.1, color=GRAY)

    return _save(fig, "B2_oi_analysis")


def plot_cyclicality(monthly_df: pd.DataFrame,
                     dow_df: pd.DataFrame,
                     hour_df: pd.DataFrame) -> Path:
    """B3/B4: Monthly, day-of-week, and hour-of-day seasonality."""
    fig = _fig(16, 12)
    fig.suptitle("B3-4 · BTC Seasonality & Cyclicality Patterns",
                 color=WHITE, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])   # monthly – full width
    ax2 = fig.add_subplot(gs[1, 0])   # DoW
    ax3 = fig.add_subplot(gs[1, 1])   # Hour

    # Monthly
    _ax(ax1, "Monthly Average Log-Return (BTC historical)",
        "Month", "Avg Log-Return (%)")
    colors_m = [GREEN if v > 0 else RED
                for v in monthly_df["avg_ret"]]
    ax1.bar(monthly_df.index, monthly_df["avg_ret"] * 100,
            color=colors_m, alpha=0.8, edgecolor=BORDER)
    for i, (m, row) in enumerate(monthly_df.iterrows()):
        ax1.text(i, row["avg_ret"] * 100 + 0.3 * np.sign(row["avg_ret"]),
                 f"{row['avg_ret']*100:.1f}%",
                 ha="center", va="bottom" if row["avg_ret"] >= 0 else "top",
                 color=WHITE, fontsize=7.5)
    ax1.axhline(0, color=GRAY, lw=0.8)

    # DoW
    _ax(ax2, "Day-of-Week Average Return", "Day", "Avg Log-Return (%)")
    colors_d = [GREEN if v > 0 else RED for v in dow_df["avg_ret"]]
    ax2.bar(dow_df.index, dow_df["avg_ret"] * 100,
            color=colors_d, alpha=0.8, edgecolor=BORDER)
    ax2.axhline(0, color=GRAY, lw=0.8)

    # Hour
    _ax(ax3, "Hour-of-Day Average Return (UTC)", "Hour", "Avg Log-Return (%)")
    colors_h = [GREEN if v > 0 else RED for v in hour_df["avg_ret"]]
    ax3.bar(hour_df.index, hour_df["avg_ret"] * 100,
            color=colors_h, alpha=0.8, edgecolor=BORDER)
    ax3.axhline(0, color=GRAY, lw=0.8)
    # Session boundaries
    for h_line, lbl in [(8, "London\nOpen"), (13, "NY\nOpen"), (21, "NY\nClose")]:
        ax3.axvline(h_line, color=GOLD, lw=1, ls="--", alpha=0.7)
        ax3.text(h_line, ax3.get_ylim()[1] * 0.9, lbl,
                 color=GOLD, fontsize=6.5, ha="left")

    return _save(fig, "B3_cyclicality")


def plot_fft_cycles(cycles_df: pd.DataFrame, price: pd.Series,
                    tf_label: str = "daily") -> Path:
    """B5: FFT cycle detection – amplitude spectrum and top cycles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("B5 · Dominant Price Cycles (FFT – detrended log-price)",
                 color=WHITE, fontsize=13, fontweight="bold")

    # Bar chart of top cycles
    top = cycles_df.head(10)
    ax  = _ax(axes[0], f"Top Cycles ({tf_label} bars)",
              "Period (bars)", "Amplitude")
    bars = ax.barh(top["period_bars"].astype(int).astype(str),
                   top["amplitude"], color=BLUE, alpha=0.85, edgecolor=BORDER)
    ax.invert_yaxis()
    for b, row in zip(bars, top.itertuples()):
        ax.text(b.get_width() + top["amplitude"].max() * 0.02,
                b.get_y() + b.get_height() / 2,
                f"{row.period_bars:.1f} bars",
                va="center", color=GRAY, fontsize=8)

    # Power spectrum
    ax2 = _ax(axes[1], "Full Amplitude Spectrum",
              "Period (bars)", "Amplitude")
    ax2.plot(cycles_df["period_bars"], cycles_df["amplitude"],
             color=BLUE, lw=1.2, alpha=0.8)
    ax2.fill_between(cycles_df["period_bars"], cycles_df["amplitude"],
                     color=BLUE, alpha=0.2)
    for _, row in top.head(5).iterrows():
        ax2.axvline(row["period_bars"], color=GOLD, lw=1.2, ls="--", alpha=0.7)
        ax2.text(row["period_bars"], cycles_df["amplitude"].max() * 0.9,
                 f"{int(row['period_bars'])}d",
                 color=GOLD, fontsize=7, ha="left")
    ax2.set_xlim(0, 200)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "B5_fft_cycles")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C – Backtest Performance
# ══════════════════════════════════════════════════════════════════════════════

def plot_equity_curve(equity: pd.Series, drawdown: pd.Series,
                      trades: pd.DataFrame) -> Path:
    """C1: Equity curve with drawdown panel and trade markers."""
    fig = _fig(16, 9)
    fig.suptitle("C1 · Strategy Equity Curve (BTCUSDT Multi-TF)",
                 color=WHITE, fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 2, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    _ax(ax1, ylabel="Equity (USD)")
    _ax(ax2, ylabel="Drawdown (%)")
    _ax(ax3, ylabel="Active")

    ax1.plot(equity.index, equity.values, color=BLUE, lw=1.5, label="Equity")
    ax1.fill_between(equity.index, equity.values, equity.values[0],
                     where=equity.values >= equity.values[0],
                     color=GREEN, alpha=0.08)
    ax1.fill_between(equity.index, equity.values, equity.values[0],
                     where=equity.values <  equity.values[0],
                     color=RED,   alpha=0.08)

    # Trade entry / exit markers
    if not trades.empty and "entry_ts" in trades.columns:
        for _, t in trades.iterrows():
            col = GREEN if t["net_pnl"] > 0 else RED
            if t["entry_ts"] in equity.index:
                ax1.axvline(t["entry_ts"], color=col, lw=0.4, alpha=0.4)

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.fill_between(drawdown.index, drawdown.values * 100,
                     color=RED, alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values * 100, color=RED, lw=0.8)
    ax2.set_ylim(top=0)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # In-position indicator
    inv = pd.Series(0, index=equity.index)
    if not trades.empty and "entry_ts" in trades.columns:
        for _, t in trades.iterrows():
            if pd.notna(t.get("exit_ts")):
                mask = (equity.index >= t["entry_ts"]) & (equity.index <= t["exit_ts"])
                inv[mask] = 1
    ax3.fill_between(inv.index, inv.values, color=GOLD, alpha=0.6)
    ax3.set_ylim(-0.1, 1.5)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", color=GRAY)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, alpha=0.1, color=GRAY)

    return _save(fig, "C1_equity_curve")


def plot_trade_distribution(trades: pd.DataFrame) -> Path:
    """C2: Histogram + KDE of per-trade net PnL % + stats table."""
    if trades.empty:
        return _save(_fig(10, 5), "C2_trade_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("C2 · Trade Return Distribution",
                 color=WHITE, fontsize=13, fontweight="bold")

    pnl_pct = (trades["net_pnl"] / trades["notional"] * 100).dropna()

    # Histogram
    ax = _ax(axes[0], "Net Return per Trade (%)", "Return (%)", "Count")
    bins = min(30, len(pnl_pct) // 3 + 5)
    wins   = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct <= 0]
    ax.hist(wins,   bins=bins, color=GREEN, alpha=0.7, label=f"Win  ({len(wins)})")
    ax.hist(losses, bins=bins, color=RED,   alpha=0.7, label=f"Loss ({len(losses)})")
    ax.axvline(float(pnl_pct.mean()), color=GOLD, lw=2, ls="--",
               label=f"Mean={pnl_pct.mean():.2f}%")
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

    # KDE
    ax2 = _ax(axes[1], "Return Density (KDE)", "Return (%)", "Density")
    xs = np.linspace(pnl_pct.min() - 1, pnl_pct.max() + 1, 300)
    if len(pnl_pct) > 5:
        kde = stats.gaussian_kde(pnl_pct)
        ax2.fill_between(xs, kde(xs), alpha=0.4, color=BLUE)
        ax2.plot(xs, kde(xs), color=BLUE, lw=2)
        # Normal fit
        mu, sig = float(pnl_pct.mean()), float(pnl_pct.std())
        ax2.plot(xs, stats.norm.pdf(xs, mu, sig),
                 color=ORANGE, lw=1.5, ls="--", label=f"Normal(μ={mu:.2f}, σ={sig:.2f})")
    ax2.axvline(0, color=GRAY, lw=0.8)
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "C2_trade_distribution")


def plot_mae_mfe(trades: pd.DataFrame) -> Path:
    """C3: MAE vs MFE scatter (coloured by win/loss, sized by notional)."""
    if trades.empty or "mae_pct" not in trades.columns:
        return _save(_fig(10, 6), "C3_mae_mfe")

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    _ax(ax, "C3 · MAE vs MFE by Trade Outcome",
        "Max Adverse Excursion (%)", "Max Favorable Excursion (%)")

    wins   = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    for subset, col, lbl in [(wins, GREEN, "Win"), (losses, RED, "Loss")]:
        if subset.empty:
            continue
        ax.scatter(subset["mae_pct"], subset["mfe_pct"],
                   c=col, alpha=0.6, s=40, label=lbl, linewidths=0)

    # Diagonal: trades where MFE == MAE (no edge)
    mx = max(trades["mae_pct"].max(), trades["mfe_pct"].max()) * 1.1
    ax.plot([0, mx], [0, mx], color=GRAY, lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)

    # Annotate TP/SL bands
    atr_tp1 = trades.get("tp1_price", pd.Series(dtype=float))
    ax.text(mx * 0.6, mx * 0.95, "MFE > MAE → captured more upside",
            color=GREEN, fontsize=8, alpha=0.8)
    ax.text(mx * 0.6, mx * 0.15, "MAE > MFE → stopped out too soon",
            color=RED,   fontsize=8, alpha=0.8)

    plt.tight_layout()
    return _save(fig, "C3_mae_mfe")


def plot_trade_duration_pnl(trades: pd.DataFrame) -> Path:
    """C4: Duration (hours) vs net PnL scatter."""
    if trades.empty:
        return _save(_fig(10, 6), "C4_duration_pnl")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("C4 · Trade Duration vs Outcome",
                 color=WHITE, fontsize=13, fontweight="bold")

    wins   = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    ax = _ax(axes[0], "Duration (h) vs Net PnL ($)",
             "Duration (hours)", "Net PnL (USD)")
    for sub, col, lbl in [(wins, GREEN, "Win"), (losses, RED, "Loss")]:
        if not sub.empty:
            ax.scatter(sub["duration_h"], sub["net_pnl"],
                       c=col, alpha=0.6, s=30, label=lbl)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

    # Box-plot by exit reason
    ax2 = _ax(axes[1], "PnL by Exit Reason", "Exit Reason", "Net PnL (USD)")
    if "exit_reason" in trades.columns:
        reasons = trades["exit_reason"].unique()
        data_by_reason = [trades.loc[trades["exit_reason"] == r, "net_pnl"].values
                          for r in reasons]
        bp = ax2.boxplot(data_by_reason, tick_labels=reasons, patch_artist=True,
                         medianprops=dict(color=GOLD, lw=2))
        colors_bp = [GREEN if m.get_data()[1][0] > 0 else RED
                     for m in bp["medians"]]
        for patch, col in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(col)
            patch.set_alpha(0.4)
        ax2.axhline(0, color=GRAY, lw=0.8)
        ax2.tick_params(axis="x", colors=WHITE)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "C4_trade_duration_pnl")


def plot_regime_session(regime_df: pd.DataFrame,
                        session_df: pd.DataFrame) -> Path:
    """C5: Side-by-side win-rate bars by regime and session."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("C5 · Win-Rate by Market Regime & Trading Session",
                 color=WHITE, fontsize=13, fontweight="bold")

    for ax, df, title in [(axes[0], regime_df, "By Market Regime"),
                           (axes[1], session_df, "By Session (UTC)")]:
        _ax(ax, title, "", "Win Rate (%)")
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=GRAY, transform=ax.transAxes)
            continue
        wr = (df["win_rate"] * 100).round(1)
        cols = [GREEN if v >= 50 else RED for v in wr]
        bars = ax.bar(range(len(df)), wr, color=cols, alpha=0.85,
                      edgecolor=BORDER)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=25, ha="right",
                           color=WHITE, fontsize=8)
        ax.axhline(50, color=GRAY, lw=1.2, ls="--")
        ax.set_ylim(0, 105)
        for bar, val in zip(bars, wr):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{val:.0f}%",
                    ha="center", va="bottom", color=WHITE, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "C5_regime_session")


def plot_rolling_sharpe(equity: pd.Series, window_h: int = 168) -> Path:
    """C6: Rolling 7-day Sharpe ratio over the backtest."""
    ret = equity.pct_change().dropna()
    hrs = 24 * 365
    roll = (ret.rolling(window_h).mean() /
            ret.rolling(window_h).std().replace(0, np.nan)) * np.sqrt(hrs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG,
                              sharex=True)
    fig.suptitle("C6 · Rolling 7-Day Sharpe Ratio",
                 color=WHITE, fontsize=13, fontweight="bold")

    _ax(axes[0], ylabel="Equity (USD)")
    axes[0].plot(equity.index, equity, color=BLUE, lw=1.2)
    axes[0].set_facecolor(PANEL)
    for sp in axes[0].spines.values():
        sp.set_color(BORDER)
    axes[0].grid(True, alpha=0.1, color=GRAY)
    axes[0].tick_params(colors=GRAY)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))

    _ax(axes[1], ylabel="Rolling Sharpe")
    axes[1].fill_between(roll.index, roll,
                          where=roll >= 0, color=GREEN, alpha=0.4)
    axes[1].fill_between(roll.index, roll,
                          where=roll < 0,  color=RED,   alpha=0.4)
    axes[1].plot(roll.index, roll, color=WHITE, lw=0.8)
    axes[1].axhline(0,   color=GRAY, lw=0.8)
    axes[1].axhline(1.0, color=GOLD, lw=1, ls="--", alpha=0.7,
                    label="Sharpe=1")
    axes[1].legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right", color=GRAY)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "C6_rolling_sharpe")


def plot_monthly_pnl_heatmap(monthly_pivot: pd.DataFrame) -> Path:
    """C7: Monthly P&L heat-map (year × month)."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(monthly_pivot) * 0.8 + 1)),
                           facecolor=BG)
    _ax(ax, "C7 · Monthly P&L Heatmap (%)")

    if monthly_pivot.empty:
        ax.text(0.5, 0.5, "Insufficient data (< 1 month)",
                ha="center", va="center", color=GRAY, transform=ax.transAxes)
        return _save(fig, "C7_monthly_pnl_heatmap")

    data = monthly_pivot * 100
    vmax = float(data.abs().max().max())
    sns.heatmap(data, ax=ax, annot=True, fmt=".1f", linewidths=0.5,
                cmap=sns.diverging_palette(10, 140, s=80, l=50, n=21),
                vmin=-vmax, vmax=vmax,
                annot_kws={"size": 9, "color": WHITE},
                cbar_kws={"shrink": 0.7, "label": "Return (%)"},
                linecolor=BORDER)
    ax.tick_params(colors=WHITE)
    ax.set_xlabel("Month", color=GRAY, fontsize=9)
    ax.set_ylabel("Year",  color=GRAY, fontsize=9)

    plt.tight_layout()
    return _save(fig, "C7_monthly_pnl_heatmap")


def plot_score_vs_outcome(score_out_df: pd.DataFrame) -> Path:
    """C8: Win-rate and avg PnL by signal score quintile."""
    if score_out_df.empty:
        return _save(_fig(10, 6), "C8_score_vs_outcome")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("C8 · Signal Score Quintile vs Trade Outcome",
                 color=WHITE, fontsize=13, fontweight="bold")

    ax  = _ax(axes[0], "Win Rate by Score Quintile",
              "Quintile", "Win Rate (%)")
    wr  = score_out_df["win_rate"] * 100
    ax.bar(score_out_df["score_bin"], wr,
           color=[GREEN if v >= 50 else RED for v in wr],
           alpha=0.85, edgecolor=BORDER)
    ax.axhline(50, color=GRAY, lw=1.2, ls="--")
    ax.set_ylim(0, 110)
    ax.set_xticks(score_out_df["score_bin"])
    ax.set_xticklabels([f"Q{i+1}" for i in score_out_df["score_bin"]])
    for _, row in score_out_df.iterrows():
        ax.text(row["score_bin"],
                row["win_rate"] * 100 + 2,
                f"{row['win_rate']*100:.0f}%\n(n={int(row['n'])})",
                ha="center", color=WHITE, fontsize=7.5)

    ax2 = _ax(axes[1], "Avg PnL (USD) by Score Quintile",
              "Quintile", "Avg Net PnL (USD)")
    ax2.bar(score_out_df["score_bin"], score_out_df["avg_pnl"],
            color=[GREEN if v > 0 else RED for v in score_out_df["avg_pnl"]],
            alpha=0.85, edgecolor=BORDER)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.set_xticks(score_out_df["score_bin"])
    ax2.set_xticklabels([f"Q{i+1}" for i in score_out_df["score_bin"]])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "C8_score_vs_outcome")


def plot_exit_breakdown(trades: pd.DataFrame) -> Path:
    """C9: Stacked bar chart of exit reasons and their cumulative PnL."""
    if trades.empty or "exit_reason" not in trades.columns:
        return _save(_fig(10, 6), "C9_exit_breakdown")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("C9 · Trade Exit Reason Analysis",
                 color=WHITE, fontsize=13, fontweight="bold")

    summary = (trades.groupby("exit_reason")
                     .agg(count=("net_pnl", "count"),
                          total_pnl=("net_pnl", "sum"),
                          avg_pnl=("net_pnl", "mean"))
                     .sort_values("count", ascending=False))

    # Counts
    ax = _ax(axes[0], "Exit Reason – Trade Count", "Exit Reason", "Count")
    bars = ax.bar(summary.index, summary["count"],
                  color=BLUE, alpha=0.85, edgecolor=BORDER)
    ax.set_xticklabels(summary.index, rotation=25, ha="right", color=WHITE)
    for bar, v in zip(bars, summary["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, str(v),
                ha="center", color=WHITE, fontsize=8)

    # Cumulative PnL
    ax2 = _ax(axes[1], "Exit Reason – Total Net PnL",
              "Exit Reason", "Total Net PnL (USD)")
    colors_ex = [GREEN if v > 0 else RED for v in summary["total_pnl"]]
    ax2.bar(summary.index, summary["total_pnl"],
            color=colors_ex, alpha=0.85, edgecolor=BORDER)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.set_xticklabels(summary.index, rotation=25, ha="right", color=WHITE)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "C9_exit_breakdown")


def plot_signal_timeline(signals: pd.DataFrame,
                         df_1h: pd.DataFrame) -> Path:
    """C10: Signal component contributions stacked over time."""
    score_cols = ["s_weekly", "s_daily", "s_4h", "s_1h",
                  "s_oi", "s_funding", "s_vol", "s_cycle"]
    avail = [c for c in score_cols if c in signals.columns]

    fig = _fig(16, 9)
    fig.suptitle("C10 · Signal Component Timeline (1H bars, resampled daily)",
                 color=WHITE, fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Price (1H)
    _ax(ax1, ylabel="BTC Price (USD)")
    ax1.plot(df_1h.index, df_1h["close"], color=BLUE, lw=0.9, alpha=0.9)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Stacked area of positive / negative components (resample to 4H for clarity)
    comp_d = signals[avail].resample("4h").mean()
    palette = [GREEN, BLUE, ORANGE, PURPLE, GOLD, PINK, TEAL, RED]

    _ax(ax2, ylabel="Composite Score")
    pos = comp_d.clip(lower=0)
    neg = comp_d.clip(upper=0)
    ax2.stackplot(comp_d.index, [pos[c] for c in avail],
                  labels=avail, colors=palette, alpha=0.65)
    ax2.stackplot(comp_d.index, [neg[c] for c in avail],
                  colors=palette, alpha=0.65)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6.5,
               loc="upper left", ncol=4)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Signal direction
    _ax(ax3, ylabel="Signal")
    sig_d = signals["signal"].resample("4h").last()
    ax3.fill_between(sig_d.index, sig_d,
                     where=sig_d > 0, color=GREEN, alpha=0.7, step="post")
    ax3.fill_between(sig_d.index, sig_d,
                     where=sig_d < 0, color=RED,   alpha=0.7, step="post")
    ax3.set_ylim(-1.5, 1.5)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", color=GRAY)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, alpha=0.08, color=GRAY)

    return _save(fig, "C10_signal_component_timeline")


# ══════════════════════════════════════════════════════════════════════════════
# Master runner
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(
    tf_data:   Dict,
    signals:   pd.DataFrame,
    oi_df:     pd.DataFrame,
    funding:   "pd.Series",
    bt_result: Dict,
    analytics: Dict,
) -> list[Path]:
    """
    Generate all charts and return list of saved paths.
    Errors in individual charts are caught and reported (non-fatal).
    """
    paths = []
    fwd    = analytics.get("fwd_returns")
    decay  = analytics.get("decay")
    quant  = analytics.get("score_quintile")
    month  = analytics.get("monthly_season")
    dow    = analytics.get("dow_season")
    hour   = analytics.get("hour_season")
    cycles = analytics.get("cycles")
    equity = bt_result.get("equity", pd.Series(dtype=float))
    dd     = bt_result.get("drawdown", pd.Series(dtype=float))
    trades = bt_result.get("trades", pd.DataFrame())
    month_pnl = analytics.get("monthly_pnl")
    regime_df = analytics.get("regime_df")
    sess_df   = analytics.get("session_df")
    score_out = analytics.get("score_out")

    tasks = [
        ("A1 Signal Score Distribution",   lambda: plot_signal_score_distribution(signals)),
        ("A2 Alpha Decay",                  lambda: plot_alpha_decay(decay) if decay is not None else None),
        ("A3 Component Correlation",        lambda: plot_signal_component_correlation(signals)),
        ("A4 Score vs Fwd Return",          lambda: plot_score_vs_fwd_return(signals, fwd) if fwd is not None else None),
        ("A5 Score Quantile Returns",       lambda: plot_score_quantile_returns(quant) if quant is not None else None),
        ("B1 Market Overview",              lambda: plot_market_overview(tf_data["1D"])),
        ("B2 OI Analysis",                  lambda: plot_oi_analysis(tf_data["1D"], oi_df, funding)),
        ("B3-4 Cyclicality",                lambda: plot_cyclicality(month, dow, hour)
                                                     if (month is not None and dow is not None and hour is not None) else None),
        ("B5 FFT Cycles",                   lambda: plot_fft_cycles(cycles, tf_data["1D"]["close"])
                                                     if cycles is not None else None),
        ("C1 Equity Curve",                 lambda: plot_equity_curve(equity, dd, trades)),
        ("C2 Trade Distribution",           lambda: plot_trade_distribution(trades)),
        ("C3 MAE vs MFE",                   lambda: plot_mae_mfe(trades)),
        ("C4 Duration vs PnL",              lambda: plot_trade_duration_pnl(trades)),
        ("C5 Regime & Session",             lambda: plot_regime_session(regime_df, sess_df)),
        ("C6 Rolling Sharpe",               lambda: plot_rolling_sharpe(equity)),
        ("C7 Monthly PnL Heatmap",          lambda: plot_monthly_pnl_heatmap(month_pnl)
                                                     if month_pnl is not None else None),
        ("C8 Score vs Outcome",             lambda: plot_score_vs_outcome(score_out)
                                                     if score_out is not None else None),
        ("C9 Exit Breakdown",               lambda: plot_exit_breakdown(trades)),
        ("C10 Signal Timeline",             lambda: plot_signal_timeline(signals, tf_data["1H"])),
    ]

    for name, fn in tasks:
        try:
            p = fn()
            if p is not None:
                paths.append(p)
                print(f"  ✓  {name}")
        except Exception as exc:
            print(f"  ✗  {name}: {exc}")

    return paths
