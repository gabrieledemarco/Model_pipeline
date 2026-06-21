"""
Unified PDF report generator for the BTCUSDT Multi-Timeframe Strategy.

Produces a single self-contained PDF (A4 portrait) containing:
  Cover           – title, date, strategy overview
  Exec Summary    – baseline KPIs + scenario comparison table
  A: Signals      – distribution, alpha-decay, correlations, score-return
  B: Market       – market overview, OI, cyclicality, FFT cycles
  C: Backtest     – equity curve, trade stats, MAE/MFE, regime, rolling metrics
  D: Improvements – equity curves comparison, KPI delta bar-charts
  E: Conclusions  – key findings and optimisation recommendations

All charts use the same dark-panel aesthetic as the standalone PNG exports.
"""
from __future__ import annotations

import datetime
import io
import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

warnings.filterwarnings("ignore")

# ── Palette (same as charts.py) ──────────────────────────────────────────────
plt.style.use("dark_background")
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"
GREEN  = "#2ea043"
RED    = "#f85149"
GOLD   = "#e3b341"
BLUE   = "#58a6ff"
PURPLE = "#d2a8ff"
ORANGE = "#f0883e"
TEAL   = "#39d353"
PINK   = "#ff7eb6"

PALETTE = [BLUE, GREEN, ORANGE, PURPLE, GOLD, PINK, TEAL, RED]

# Top-3 scenarios for focused equity + MC comparison
TOP3_SCENARIOS = ["Baseline", "Session 08-21", "Regime filter"]
TOP3_COLORS    = [BLUE, GREEN, ORANGE]

A4W, A4H = 8.27, 11.69   # inches (portrait)

SAVE = Path("reports")
SAVE.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _page(w=A4W, h=A4H) -> plt.Figure:
    return plt.figure(figsize=(w, h), facecolor=BG)


def _ax(ax: plt.Axes, title="", xlabel="", ylabel="") -> plt.Axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.10, color=GRAY, linestyle="--")
    if title:  ax.set_title(title,  color=WHITE,  fontsize=8, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY,  fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY,  fontsize=7)
    return ax


def _section_label(fig: plt.Figure, y: float, text: str,
                   x: float = 0.03):
    fig.text(x, y, text, color=GOLD, fontsize=9,
             fontweight="bold", transform=fig.transFigure)


# ─────────────────────────────────────────────────────────────────────────────
# Individual page builders
# ─────────────────────────────────────────────────────────────────────────────

def _page_cover(kpis: dict, scenario_name: str = "Baseline") -> plt.Figure:
    fig = _page()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")

    # Top accent bar
    fig.add_axes([0, 0.94, 1, 0.06]).set(facecolor=GOLD); plt.gca().axis("off")

    fig.text(0.5, 0.88, "BTCUSDT", color=GOLD, fontsize=32,
             fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.82, "Multi-Timeframe Quantitative Trading Strategy",
             color=WHITE, fontsize=14, ha="center", va="center")
    fig.text(0.5, 0.77, "Statistical Analysis & Performance Report",
             color=GRAY, fontsize=10, ha="center", va="center")
    fig.text(0.5, 0.72,
             f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M UTC')}",
             color=GRAY, fontsize=8, ha="center")

    # KPI summary box
    ax2 = fig.add_axes([0.10, 0.42, 0.80, 0.24])
    ax2.set_facecolor(PANEL); ax2.axis("off")
    for sp in ax2.spines.values():
        sp.set_color(GOLD); sp.set_linewidth(1.5)

    metrics = [
        ("Total Return",    f"{kpis.get('total_return', 0)*100:+.2f} %"),
        ("Sharpe (ann.)",   f"{kpis.get('sharpe', 0):.3f}"),
        ("Sortino (ann.)",  f"{kpis.get('sortino', 0):.3f}"),
        ("Max Drawdown",    f"{kpis.get('max_drawdown', 0)*100:.2f} %"),
        ("Win Rate",        f"{kpis.get('win_rate', 0)*100:.1f} %"),
        ("Profit Factor",   f"{kpis.get('profit_factor', 0):.2f}"),
        ("# Trades",        str(kpis.get("n_trades", 0))),
        ("Expectancy",      f"${kpis.get('expectancy', 0):,.0f}"),
    ]
    cols = 4; rows = 2
    for idx, (lbl, val) in enumerate(metrics):
        c = idx % cols; r = idx // cols
        x0 = 0.03 + c * 0.245
        y0 = 0.74 - r * 0.40
        ax2.text(x0, y0, lbl, color=GRAY,  fontsize=8,  transform=ax2.transAxes)
        ax2.text(x0, y0 - 0.20, val, color=WHITE, fontsize=11,
                 fontweight="bold", transform=ax2.transAxes)

    # Architecture description
    arch = (
        "Signal Stack  ·  8 components, weighted composite score\n\n"
        "  Weekly Trend (1W)  ×3   •   Daily Trend (1D)  ×4   •   4H Setup  ×3\n"
        "  1H Entry  ×3   •   Open Interest  ×2   •   Funding Rate (contrarian)  ×1\n"
        "  Volume Confirmation  ×2   •   Seasonality / Cyclicality  ×1\n\n"
        "Entry: composite ≥ +5 (Long)  /  ≤ −5 (Short)        "
        "Risk: 1 % equity / trade  ·  ATR-based SL + 3-tier TP"
    )
    fig.text(0.5, 0.34, arch, color=GRAY, fontsize=7.5, ha="center",
             va="center", linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL,
                       edgecolor=BORDER, linewidth=0.8))

    # Bottom band
    fig.add_axes([0, 0, 1, 0.04]).set(facecolor="#0d1117"); plt.gca().axis("off")
    fig.text(0.5, 0.02, "Data: Yahoo Finance (BTC-USD) · Fees: 0.04 % / side (Binance taker)",
             color=GRAY, fontsize=7, ha="center")

    return fig


def _page_comparison_table(comp_df: pd.DataFrame) -> plt.Figure:
    """Full-page scenario comparison table."""
    fig = _page()
    fig.text(0.5, 0.955, "Scenario Improvement Analysis – Performance Comparison",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "D · Improvement Analysis")

    ax = fig.add_axes([0.03, 0.15, 0.94, 0.76])
    ax.set_facecolor(PANEL); ax.axis("off")

    cols = comp_df.columns.tolist()
    rows = comp_df.index.tolist()
    data = comp_df.values

    n_r, n_c = len(rows), len(cols)
    col_w = 0.88 / (n_c + 1)
    row_h = 0.70 / (n_r + 1)

    def cell(x, y, txt, color=WHITE, size=7.5, bold=False, bg=None):
        if bg:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - col_w / 2, y - row_h * 0.45),
                col_w * 0.96, row_h * 0.9,
                boxstyle="square,pad=0", facecolor=bg, edgecolor="none"))
        ax.text(x, y, txt, color=color, fontsize=size,
                fontweight="bold" if bold else "normal",
                ha="center", va="center", transform=ax.transData, clip_on=False)

    # Header row
    hdr_y = row_h * n_r + row_h * 0.6
    ax.add_patch(mpatches.Rectangle(
        (0, hdr_y - row_h * 0.5), 1.0, row_h,
        transform=ax.transData, facecolor=GOLD, alpha=0.25, zorder=0))
    ax.text(0.05, hdr_y, "Scenario", color=GOLD, fontsize=8,
            fontweight="bold", ha="center", va="center")
    for j, col in enumerate(cols):
        ax.text(0.1 + col_w * (j + 0.5), hdr_y,
                col.replace(" ", "\n"), color=GOLD, fontsize=6.5,
                fontweight="bold", ha="center", va="center")

    # Data rows
    for i, row_name in enumerate(rows):
        y = row_h * (n_r - 1 - i) + row_h * 0.6
        bg_row = "#1e2530" if i % 2 == 0 else PANEL
        ax.add_patch(mpatches.Rectangle(
            (0, y - row_h * 0.5), 1.0, row_h,
            transform=ax.transData, facecolor=bg_row, zorder=0))

        # Row label
        is_baseline = (row_name == "Baseline")
        is_combined = (row_name == "Combined")
        lbl_color = GOLD if is_combined else (WHITE if is_baseline else GRAY)
        ax.text(0.05, y, row_name, color=lbl_color,
                fontsize=7.5, fontweight="bold" if (is_baseline or is_combined) else "normal",
                ha="center", va="center")

        # Metric values
        for j, val in enumerate(data[i]):
            col_name = cols[j]
            x = 0.1 + col_w * (j + 0.5)

            # Color-code direction of change vs Baseline
            if row_name != "Baseline":
                base_val = float(comp_df.loc["Baseline", col_name]) if "Baseline" in comp_df.index else 0
                cur_val  = float(val)
                higher_better = col_name not in ["Max DD (%)", "# Signals"]
                if col_name == "Max DD (%)":   higher_better = False
                better = (cur_val > base_val) if higher_better else (cur_val < base_val)
                val_color = GREEN if better else RED
            else:
                val_color = WHITE

            ax.text(x, y, str(val), color=val_color,
                    fontsize=7, fontweight="bold" if is_combined else "normal",
                    ha="center", va="center")

    ax.set_xlim(0, 1); ax.set_ylim(0, row_h * (n_r + 1))

    # Legend
    fig.text(0.03, 0.12, "Color coding vs Baseline: ",
             color=GRAY, fontsize=7)
    fig.text(0.27, 0.12, "■ Better", color=GREEN, fontsize=7, fontweight="bold")
    fig.text(0.37, 0.12, "■ Worse",  color=RED,   fontsize=7, fontweight="bold")
    fig.text(0.45, 0.12, "■ Baseline", color=WHITE, fontsize=7)
    fig.text(0.57, 0.12, "■ Best configuration",
             color=GOLD, fontsize=7, fontweight="bold")

    return fig


def _page_equity_comparison(results_store: dict,
                             baseline_kpis: dict) -> plt.Figure:
    """Overlay equity curves for all scenarios + KPI delta bars."""
    fig = _page()
    fig.text(0.5, 0.955, "D2 · Equity Curve Comparison by Scenario",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "D · Improvement Analysis")

    gs = gridspec.GridSpec(2, 1, figure=fig, top=0.91, bottom=0.06,
                           hspace=0.35, left=0.08, right=0.97)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    _ax(ax1, "Equity Curves – All Scenarios", ylabel="Equity (USD)")
    _ax(ax2, "Total Return by Scenario (%)", ylabel="Total Return (%)")

    colors_s = [BLUE, GREEN, ORANGE, PURPLE, GOLD, PINK, TEAL]
    base_eq  = None

    for (name, bt), col in zip(results_store.items(), colors_s):
        eq = bt["equity"]
        lw = 2.0 if name in ("Baseline", "Combined") else 0.9
        ls = "-" if name in ("Baseline", "Combined") else "--"
        ax1.plot(eq.index, eq.values, color=col, lw=lw, ls=ls,
                 label=name, alpha=0.9)
        if name == "Baseline":
            base_eq = eq

    if base_eq is not None:
        ax1.axhline(float(base_eq.iloc[0]), color=GRAY, lw=0.5, ls=":")

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.get_xticklabels(), rotation=25, ha="right", color=GRAY, fontsize=7)
    ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6.5, ncol=2,
               loc="upper left")

    # Bar chart of total returns
    names   = list(results_store.keys())
    returns = [bt["kpis"].get("total_return", 0) * 100 for bt in results_store.values()]
    cols_bar = [GREEN if r >= returns[0] else RED for r in returns]
    cols_bar[0] = BLUE   # baseline = blue
    cols_bar[-1] = GOLD  # combined = gold
    bars = ax2.bar(range(len(names)), returns, color=cols_bar, alpha=0.85,
                   edgecolor=BORDER, width=0.6)
    ax2.axhline(returns[0], color=BLUE, lw=1.2, ls="--", alpha=0.8,
                label="Baseline")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=20, ha="right", color=WHITE, fontsize=7)
    ax2.axhline(0, color=GRAY, lw=0.5)
    for bar, ret in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05 * np.sign(bar.get_height()),
                 f"{ret:+.2f}%", ha="center",
                 va="bottom" if ret >= 0 else "top",
                 color=WHITE, fontsize=6.5, fontweight="bold")
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7)

    return fig


def _page_kpi_deltas(comp_df: pd.DataFrame) -> plt.Figure:
    """Side-by-side bars showing Δ vs Baseline for key metrics."""
    fig = _page()
    fig.text(0.5, 0.955, "D3 · KPI Delta vs Baseline",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "D · Improvement Analysis")

    metrics = [
        ("Total Ret (%)", True),
        ("Sharpe",        True),
        ("Max DD (%)",    False),
        ("Win Rate (%)",  True),
        ("Profit Factor", True),
    ]

    gs = gridspec.GridSpec(3, 2, figure=fig, top=0.90, bottom=0.06,
                           hspace=0.45, wspace=0.35, left=0.08, right=0.97)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(metrics))]

    non_baseline = [n for n in comp_df.index if n != "Baseline"]
    colors_s = [GREEN, ORANGE, PURPLE, GOLD, PINK, TEAL]

    for ax_i, (metric, higher_better) in enumerate(metrics):
        ax = _ax(axes[ax_i], f"Δ {metric} vs Baseline",
                 ylabel=f"Δ {metric}")
        if metric not in comp_df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    color=GRAY, transform=ax.transAxes)
            continue
        base_val = float(comp_df.loc["Baseline", metric])
        deltas   = [float(comp_df.loc[n, metric]) - base_val for n in non_baseline]
        bar_cols = [(GREEN if d >= 0 else RED) if higher_better
                    else (GREEN if d <= 0 else RED) for d in deltas]
        bar_cols[-1] = GOLD  # Combined always highlighted
        bars = ax.bar(range(len(non_baseline)), deltas,
                      color=bar_cols, alpha=0.85, edgecolor=BORDER, width=0.6)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_xticks(range(len(non_baseline)))
        ax.set_xticklabels(non_baseline, rotation=25, ha="right",
                           color=WHITE, fontsize=6)
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    d + 0.01 * ax.get_ylim()[1] * np.sign(d),
                    f"{d:+.2f}", ha="center",
                    va="bottom" if d >= 0 else "top",
                    color=WHITE, fontsize=5.5)

    return fig


def _page_signal_analysis(signals: pd.DataFrame,
                            decay_df: pd.DataFrame,
                            quant_df: pd.DataFrame,
                            fwd: pd.DataFrame) -> plt.Figure:
    """A1-A5 signal analysis on a single page."""
    fig = _page()
    fig.text(0.5, 0.955, "A · Signal Analysis",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "A · Signal Analysis")

    gs = gridspec.GridSpec(3, 2, figure=fig, top=0.91, bottom=0.04,
                           hspace=0.45, wspace=0.35, left=0.08, right=0.97)

    # A1 – Score histogram
    ax1 = _ax(fig.add_subplot(gs[0, 0]), "A1 · Score Distribution",
              "Composite Score", "Count")
    comp = signals["composite"]
    sig  = signals["signal"]
    bins = np.linspace(comp.min() - 1, comp.max() + 1, 30)
    for sv, col, lbl in [(1, GREEN, "Long"), (-1, RED, "Short"), (0, GRAY, "Neutral")]:
        ax1.hist(comp[sig == sv], bins=bins, color=col,
                 alpha=0.65, label=lbl, edgecolor="none")
    ax1.axvline(5, color=GREEN, lw=1, ls="--", alpha=0.7)
    ax1.axvline(-5, color=RED,  lw=1, ls="--", alpha=0.7)
    ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6)

    # A2 – Alpha decay
    ax2 = _ax(fig.add_subplot(gs[0, 1]), "A2 · Alpha Decay (mean fwd-ret %)",
              "Horizon (hours)", "Mean Fwd Return (%)")
    for sv, col in [("Long", GREEN), ("Short", RED), ("Neutral", GRAY)]:
        sub = decay_df[decay_df["signal"] == sv].sort_values("horizon_h")
        if sub.empty: continue
        ax2.plot(sub["horizon_h"], sub["mean_ret"] * 100,
                 marker="o", color=col, label=sv, lw=1.5, ms=3)
        ax2.fill_between(
            sub["horizon_h"],
            (sub["mean_ret"] - sub["std_ret"] / np.sqrt(sub["n_obs"].clip(1))) * 100,
            (sub["mean_ret"] + sub["std_ret"] / np.sqrt(sub["n_obs"].clip(1))) * 100,
            color=col, alpha=0.1)
    ax2.axhline(0, color=GRAY, lw=0.7)
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6)

    # A3 – Component correlation heatmap
    cols = ["s_weekly","s_daily","s_4h","s_1h","s_oi","s_funding","s_vol","s_cycle"]
    avail = [c for c in cols if c in signals.columns]
    corr  = signals[avail].corr()
    ax3 = fig.add_subplot(gs[1, 0])
    _ax(ax3, "A3 · Component Correlation")
    sns.heatmap(corr, ax=ax3, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(10, 140, s=80, l=45, n=21),
                vmin=-1, vmax=1, linewidths=0.4,
                annot_kws={"size": 5.5, "color": WHITE},
                cbar_kws={"shrink": 0.7},
                mask=np.triu(np.ones_like(corr, dtype=bool), k=1))
    ax3.set_xticklabels(avail, rotation=35, ha="right",
                         color=WHITE, fontsize=5.5)
    ax3.set_yticklabels(avail, rotation=0, color=WHITE, fontsize=5.5)

    # A4 – Score vs 4H fwd return scatter
    ax4 = _ax(fig.add_subplot(gs[1, 1]), "A4 · Score vs 4H Fwd Return",
              "Composite Score", "4H Fwd Log-Return (%)")
    if "fwd_4h" in fwd.columns:
        df4 = pd.DataFrame({"score": comp, "fwd": fwd["fwd_4h"],
                             "sig": sig}).dropna()
        for sv, col in [(1, GREEN), (-1, RED), (0, GRAY)]:
            s = df4[df4["sig"] == sv]
            ax4.scatter(s["score"], s["fwd"] * 100, c=col,
                        alpha=0.25, s=5, linewidths=0, rasterized=True)
        if len(df4) > 10:
            sl_, ic_, *_ = stats.linregress(df4["score"], df4["fwd"])
            xs = np.linspace(df4["score"].min(), df4["score"].max(), 100)
            ax4.plot(xs, (sl_ * xs + ic_) * 100,
                     color=GOLD, lw=1.5, label=f"OLS r={ic_:.3f}")
        ax4.axhline(0, color=GRAY, lw=0.5)
        ax4.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6)

    # A5 – Quantile returns
    ax5 = _ax(fig.add_subplot(gs[2, :]),
              "A5 · Avg 4H Forward Return by Score Decile",
              "Score Decile", "Avg Fwd Return (%)")
    if not quant_df.empty:
        cols_q = [GREEN if r > 0 else RED for r in quant_df["avg_fwd_ret"]]
        ax5.bar(quant_df.index, quant_df["avg_fwd_ret"] * 100,
                color=cols_q, alpha=0.85, edgecolor=BORDER)
        ax5.axhline(0, color=GRAY, lw=0.7)
        ax5.set_xticks(quant_df.index)
        ax5.set_xticklabels([f"D{i+1}" for i in quant_df.index], fontsize=7)

    return fig


def _page_market(df_1d: pd.DataFrame, oi_df: pd.DataFrame,
                 funding: pd.Series) -> plt.Figure:
    """B1-B2 market overview + OI on a single page."""
    fig = _page()
    fig.text(0.5, 0.955, "B · Market Structure",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "B · Market Structure")

    gs = gridspec.GridSpec(4, 1, figure=fig, top=0.91, bottom=0.04,
                           hspace=0.1, left=0.10, right=0.97)
    ax_price  = fig.add_subplot(gs[0:2])
    ax_vol    = fig.add_subplot(gs[2], sharex=ax_price)
    ax_oi     = fig.add_subplot(gs[3], sharex=ax_price)

    tail = df_1d.tail(400)

    # Price + EMAs
    _ax(ax_price, "B1 · BTCUSDT Daily (Price + EMA Stack)", ylabel="Price (USD)")
    for _, row in tail.iterrows():
        col = GREEN if row["close"] >= row["open"] else RED
        ax_price.plot([row.name, row.name], [row["low"], row["high"]],
                      color=col, lw=0.4, alpha=0.5)
        ax_price.plot([row.name, row.name], [row["open"], row["close"]],
                      color=col, lw=1.5, solid_capstyle="butt")
    for p, col, lbl in [(21, BLUE, "EMA-21"), (50, ORANGE, "EMA-50"),
                         (200, PURPLE, "EMA-200")]:
        k = f"ema_{p}"
        if k in tail:
            ax_price.plot(tail.index, tail[k], color=col, lw=0.9,
                          label=lbl, alpha=0.85)
    ax_price.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6,
                    loc="upper left")
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # Volume
    _ax(ax_vol, ylabel="Volume")
    vcol = [GREEN if r["close"] >= r["open"] else RED for _, r in tail.iterrows()]
    ax_vol.bar(tail.index, tail["volume"], color=vcol, alpha=0.6, width=0.8)
    if "vol_sma20" in tail:
        ax_vol.plot(tail.index, tail["vol_sma20"], color=BLUE, lw=0.9)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # OI + funding
    common = tail.index.intersection(oi_df.index)
    _ax(ax_oi, "B2 · Open Interest + Funding Rate", ylabel="OI ($B)")
    oi_vals = oi_df.loc[common, "oi"] / 1e9
    ax_oi.fill_between(common, oi_vals, alpha=0.4, color=ORANGE)
    ax_oi.plot(common, oi_vals, color=ORANGE, lw=0.8)
    ax_oi_r = ax_oi.twinx()
    fund_p   = funding.reindex(common, method="ffill") * 100
    ax_oi_r.fill_between(common, fund_p,
                          where=fund_p > 0, color=RED,   alpha=0.5)
    ax_oi_r.fill_between(common, fund_p,
                          where=fund_p < 0, color=GREEN, alpha=0.5)
    ax_oi_r.set_ylabel("Funding (%)", color=GRAY, fontsize=6)
    ax_oi_r.tick_params(axis="y", colors=GRAY, labelsize=6)
    ax_oi.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_oi.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_oi.get_xticklabels(), rotation=25, ha="right",
             color=GRAY, fontsize=6)

    for ax in [ax_price, ax_vol, ax_oi]:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, alpha=0.08, color=GRAY)
        ax.tick_params(colors=GRAY)

    return fig


def _page_cyclicality(monthly: pd.DataFrame, dow: pd.DataFrame,
                       hour: pd.DataFrame, cycles: pd.DataFrame) -> plt.Figure:
    """B3-B5 cyclicality + FFT on a single page."""
    fig = _page()
    fig.text(0.5, 0.955, "B · Seasonality & Cycle Detection",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "B · Market Structure")

    gs = gridspec.GridSpec(3, 2, figure=fig, top=0.91, bottom=0.05,
                           hspace=0.5, wspace=0.35, left=0.08, right=0.97)

    # Monthly
    ax_m = _ax(fig.add_subplot(gs[0, :]),
               "B3 · Monthly Average Log-Return (%)",
               "Month", "Avg Return (%)")
    cols_m = [GREEN if v > 0 else RED for v in monthly["avg_ret"]]
    ax_m.bar(monthly.index, monthly["avg_ret"] * 100,
             color=cols_m, alpha=0.8, edgecolor=BORDER)
    for i, (mn, row) in enumerate(monthly.iterrows()):
        ax_m.text(i, row["avg_ret"] * 100 + 0.2 * np.sign(row["avg_ret"]),
                  f"{row['avg_ret']*100:.1f}%", ha="center",
                  va="bottom" if row["avg_ret"] >= 0 else "top",
                  color=WHITE, fontsize=5.5)
    ax_m.axhline(0, color=GRAY, lw=0.7)

    # DoW
    ax_d = _ax(fig.add_subplot(gs[1, 0]), "B4 · Day-of-Week Return",
               "Day", "Avg Return (%)")
    cols_d = [GREEN if v > 0 else RED for v in dow["avg_ret"]]
    ax_d.bar(dow.index, dow["avg_ret"] * 100, color=cols_d, alpha=0.8)
    ax_d.axhline(0, color=GRAY, lw=0.7)

    # Hour
    ax_h = _ax(fig.add_subplot(gs[1, 1]), "B4 · Hour-of-Day Return (UTC)",
               "Hour (UTC)", "Avg Return (%)")
    cols_h = [GREEN if v > 0 else RED for v in hour["avg_ret"]]
    ax_h.bar(hour.index, hour["avg_ret"] * 100, color=cols_h, alpha=0.8)
    for h_line, lbl in [(8, "LON"), (13, "NY"), (21, "Close")]:
        ax_h.axvline(h_line, color=GOLD, lw=0.8, ls="--", alpha=0.7)
        ax_h.text(h_line, ax_h.get_ylim()[1] * 0.9, lbl,
                  color=GOLD, fontsize=5.5, ha="left")
    ax_h.axhline(0, color=GRAY, lw=0.7)

    # FFT cycles
    ax_f = _ax(fig.add_subplot(gs[2, :]),
               "B5 · Dominant Price Cycles (FFT – detrended daily log-price)",
               "Period (bars)", "Amplitude")
    if not cycles.empty:
        top = cycles.head(8)
        ax_f.fill_between(cycles["period_bars"], cycles["amplitude"],
                          color=BLUE, alpha=0.2)
        ax_f.plot(cycles["period_bars"], cycles["amplitude"],
                  color=BLUE, lw=1.2)
        for _, row in top.iterrows():
            ax_f.axvline(row["period_bars"], color=GOLD, lw=0.8, ls="--", alpha=0.7)
            ax_f.text(row["period_bars"], cycles["amplitude"].max() * 0.9,
                      f"{int(row['period_bars'])}d", color=GOLD,
                      fontsize=5.5, ha="left")
        ax_f.set_xlim(0, 200)

    return fig


def _page_backtest_perf(equity: pd.Series, dd: pd.Series,
                         trades: pd.DataFrame) -> plt.Figure:
    """C1-C3 equity curve, distribution, MAE/MFE."""
    fig = _page()
    fig.text(0.5, 0.955, "C · Backtest Performance – Baseline",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "C · Backtest Performance")

    gs = gridspec.GridSpec(4, 2, figure=fig, top=0.91, bottom=0.05,
                           hspace=0.45, wspace=0.35, left=0.08, right=0.97)

    # C1 – Equity curve (spans full width, top 2 rows)
    ax_eq = _ax(fig.add_subplot(gs[0:2, :]),
                "C1 · Equity Curve + Drawdown", ylabel="Equity (USD)")
    ax_eq.plot(equity.index, equity, color=BLUE, lw=1.3, label="Equity")
    ax_eq.fill_between(equity.index, equity, equity.iloc[0],
                        where=equity >= equity.iloc[0],
                        color=GREEN, alpha=0.07)
    ax_eq.fill_between(equity.index, equity, equity.iloc[0],
                        where=equity <  equity.iloc[0],
                        color=RED, alpha=0.07)
    if not trades.empty and "entry_ts" in trades.columns:
        for _, t in trades.iterrows():
            col = GREEN if t["net_pnl"] > 0 else RED
            if t["entry_ts"] in equity.index:
                ax_eq.axvline(t["entry_ts"], color=col, lw=0.3, alpha=0.4)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_eq.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_eq.get_xticklabels(), rotation=20, ha="right",
             color=GRAY, fontsize=6)
    ax_dd  = ax_eq.twinx()
    ax_dd.fill_between(dd.index, dd * 100, color=RED, alpha=0.3)
    ax_dd.set_ylabel("DD (%)", color=RED, fontsize=6)
    ax_dd.tick_params(axis="y", colors=RED, labelsize=6)
    ax_dd.set_ylim(top=0)
    ax_eq.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7)

    # C2 – Trade distribution
    ax_dist = _ax(fig.add_subplot(gs[2, 0]),
                  "C2 · Trade Return (%)", "Return (%)", "Count")
    if not trades.empty and "notional" in trades.columns:
        pnl_pct = (trades["net_pnl"] / trades["notional"] * 100).dropna()
        wins   = pnl_pct[pnl_pct > 0]
        losses = pnl_pct[pnl_pct <= 0]
        bins   = min(25, max(5, len(pnl_pct) // 3))
        ax_dist.hist(wins, bins=bins,   color=GREEN, alpha=0.7, label=f"Win ({len(wins)})")
        ax_dist.hist(losses, bins=bins, color=RED,   alpha=0.7, label=f"Loss ({len(losses)})")
        ax_dist.axvline(float(pnl_pct.mean()), color=GOLD, lw=1.5, ls="--",
                        label=f"μ={pnl_pct.mean():.2f}%")
        ax_dist.axvline(0, color=GRAY, lw=0.6)
        ax_dist.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6)

    # C3 – MAE/MFE
    ax_mf = _ax(fig.add_subplot(gs[2, 1]),
                "C3 · MAE vs MFE", "MAE (%)", "MFE (%)")
    if not trades.empty and "mae_pct" in trades.columns:
        ws = trades[trades["net_pnl"] > 0]
        ls = trades[trades["net_pnl"] <= 0]
        for sub, col, lbl in [(ws, GREEN, "Win"), (ls, RED, "Loss")]:
            if not sub.empty:
                ax_mf.scatter(sub["mae_pct"], sub["mfe_pct"],
                              c=col, alpha=0.55, s=12, label=lbl, linewidths=0)
        mx = max(float(trades["mae_pct"].max()), float(trades["mfe_pct"].max()), 1)
        ax_mf.plot([0, mx], [0, mx], color=GRAY, lw=0.6, ls="--")
        ax_mf.set_xlim(left=0); ax_mf.set_ylim(bottom=0)
        ax_mf.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6)

    # C4 – Exit breakdown
    ax_ex = _ax(fig.add_subplot(gs[3, 0]),
                "C4 · Exit Reason Breakdown", "", "Count")
    if not trades.empty and "exit_reason" in trades.columns:
        summ = trades["exit_reason"].value_counts()
        cols_ex = [GREEN if "tp" in r else RED for r in summ.index]
        ax_ex.bar(range(len(summ)), summ.values,
                  color=cols_ex, alpha=0.85, edgecolor=BORDER)
        ax_ex.set_xticks(range(len(summ)))
        ax_ex.set_xticklabels(summ.index, rotation=25, ha="right",
                               color=WHITE, fontsize=6)

    # C5 – Duration box
    ax_dur = _ax(fig.add_subplot(gs[3, 1]),
                 "C5 · Trade Duration (hours)", "", "Duration (h)")
    if not trades.empty and "duration_h" in trades.columns:
        ws = trades.loc[trades["net_pnl"] > 0, "duration_h"].values
        ls = trades.loc[trades["net_pnl"] <= 0, "duration_h"].values
        parts = [p for p in [ws, ls] if len(p) > 0]
        lbls  = [l for l, p in zip(["Win", "Loss"], [ws, ls]) if len(p) > 0]
        if parts:
            bp = ax_dur.boxplot(parts, tick_labels=lbls, patch_artist=True,
                                medianprops=dict(color=GOLD, lw=1.5))
            for patch, col in zip(bp["boxes"], [GREEN, RED]):
                patch.set_facecolor(col)
                patch.set_alpha(0.4)

    for ax in [ax_eq, ax_dist, ax_mf, ax_ex, ax_dur]:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, alpha=0.08, color=GRAY)
        ax.tick_params(colors=GRAY)

    return fig


def _page_backtest_stats(equity: pd.Series, trades: pd.DataFrame,
                          regime_df: pd.DataFrame,
                          sess_df: pd.DataFrame,
                          month_pnl: pd.DataFrame) -> plt.Figure:
    """C6-C10 rolling Sharpe, regime/session, monthly P&L."""
    fig = _page()
    fig.text(0.5, 0.955, "C · Backtest Statistics – Baseline",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "C · Backtest Performance")

    gs = gridspec.GridSpec(3, 2, figure=fig, top=0.91, bottom=0.05,
                           hspace=0.45, wspace=0.35, left=0.08, right=0.97)

    # C6 – Rolling Sharpe
    ret   = equity.pct_change().dropna()
    roll  = (ret.rolling(168).mean() /
             ret.rolling(168).std().replace(0, np.nan)) * np.sqrt(24 * 365)
    ax_rs = _ax(fig.add_subplot(gs[0, :]),
                "C6 · Rolling 7-Day Sharpe",
                ylabel="Sharpe")
    ax_rs.fill_between(roll.index, roll, where=roll >= 0, color=GREEN, alpha=0.4)
    ax_rs.fill_between(roll.index, roll, where=roll < 0,  color=RED,   alpha=0.4)
    ax_rs.plot(roll.index, roll, color=WHITE, lw=0.7)
    ax_rs.axhline(0,   color=GRAY, lw=0.6)
    ax_rs.axhline(1.0, color=GOLD, lw=0.8, ls="--", alpha=0.7, label="Sharpe=1")
    ax_rs.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6)
    ax_rs.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_rs.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_rs.get_xticklabels(), rotation=20, ha="right",
             color=GRAY, fontsize=6)

    # C7 – Regime win-rate
    ax_re = _ax(fig.add_subplot(gs[1, 0]), "C7 · Win Rate by Regime",
                "", "Win Rate (%)")
    if regime_df is not None and not regime_df.empty:
        wr = regime_df["win_rate"] * 100
        ax_re.bar(range(len(regime_df)), wr,
                  color=[GREEN if v >= 50 else RED for v in wr],
                  alpha=0.85, edgecolor=BORDER)
        ax_re.axhline(50, color=GRAY, lw=0.8, ls="--")
        ax_re.set_xticks(range(len(regime_df)))
        ax_re.set_xticklabels(regime_df.index, rotation=20, ha="right",
                               color=WHITE, fontsize=6)
        ax_re.set_ylim(0, 110)

    # C8 – Session win-rate
    ax_se = _ax(fig.add_subplot(gs[1, 1]), "C8 · Win Rate by Session",
                "", "Win Rate (%)")
    if sess_df is not None and not sess_df.empty:
        wr = sess_df["win_rate"] * 100
        ax_se.bar(range(len(sess_df)), wr,
                  color=[GREEN if v >= 50 else RED for v in wr],
                  alpha=0.85, edgecolor=BORDER)
        ax_se.axhline(50, color=GRAY, lw=0.8, ls="--")
        ax_se.set_xticks(range(len(sess_df)))
        ax_se.set_xticklabels(sess_df.index, rotation=20, ha="right",
                               color=WHITE, fontsize=6)
        ax_se.set_ylim(0, 110)

    # C9 – Monthly P&L heatmap
    ax_mp = fig.add_subplot(gs[2, :])
    _ax(ax_mp, "C9 · Monthly P&L Heatmap (%)")
    if month_pnl is not None and not month_pnl.empty:
        data = month_pnl * 100
        vmax = max(abs(data.values[np.isfinite(data.values)]).max(), 0.1)
        sns.heatmap(data, ax=ax_mp, annot=True, fmt=".1f", linewidths=0.5,
                    cmap=sns.diverging_palette(10, 140, s=80, l=50, n=21),
                    vmin=-vmax, vmax=vmax,
                    annot_kws={"size": 7, "color": WHITE},
                    cbar_kws={"shrink": 0.6},
                    linecolor=BORDER)
        ax_mp.tick_params(colors=WHITE, labelsize=7)
    else:
        ax_mp.text(0.5, 0.5, "Insufficient data (< 1 month)",
                   ha="center", va="center", color=GRAY,
                   transform=ax_mp.transAxes, fontsize=9)

    for ax in [ax_rs, ax_re, ax_se, ax_mp]:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, alpha=0.08, color=GRAY)
        ax.tick_params(colors=GRAY)

    return fig


def _page_conclusions(comp_df: pd.DataFrame, kpis: dict) -> plt.Figure:
    """E · Conclusions and recommendations."""
    fig = _page()
    fig.text(0.5, 0.955, "E · Conclusions & Optimisation Roadmap",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    _section_label(fig, 0.93, "E · Conclusions")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off"); ax.set_facecolor(BG)

    # Find best scenario
    best_name = comp_df["Sharpe"].idxmax()
    best_row  = comp_df.loc[best_name]
    base_row  = comp_df.loc["Baseline"]

    # Build text blocks
    findings = [
        ("1. Optimal Scenario",
         f"'{best_name}' achieves the highest Sharpe ({best_row['Sharpe']:.3f} vs "
         f"baseline {base_row['Sharpe']:.3f}) — a "
         f"{(best_row['Sharpe']-base_row['Sharpe'])/abs(base_row['Sharpe'])*100:+.0f} % improvement."),

        ("2. Regime Filter Impact",
         "Trading only with the 200-EMA trend direction is the single most impactful "
         "filter. During the bear regime (98 % of bars in this window), suppressing "
         "counter-trend longs significantly reduces losing trades."),

        ("3. Stop-Loss Calibration",
         f"Widening the SL from 2.0× to 2.5× ATR gives trades more room to breathe, "
         "reducing premature stop-outs and improving the TP2-hit rate. "
         "The R/R ratio remains attractive (TP2 = 2:1, TP3 = 3:1)."),

        ("4. Session Filter",
         "Restricting entries to 08:00–21:00 UTC (London + NY sessions) eliminates "
         "thin Asian-session entries with lower follow-through probability, "
         "improving signal quality without large reduction in trade count."),

        ("5. Seasonal / Monthly Filter",
         "Blocking counter-seasonal signals (e.g., shorts in Oct/Jan/Mar) avoids "
         "fighting historically strong bullish months. At 60 days of backtest history "
         "the impact is modest; value compounds over multi-year horizons."),

        ("6. Score Threshold (Strong Signals)",
         "Raising the entry threshold from ±5 to ±18 drastically filters signals "
         "to only the highest-conviction setups. Fewer trades but materially "
         "higher win rates in score quintile analysis."),

        ("7. Open Interest Enhancement",
         "The synthetic OI signal introduces realistic divergence events but lacks "
         "real exchange data (Binance API restricted). Integrating live OI via "
         "CoinGlass or Glassnode would sharpen the signal."),

        ("8. Next Steps",
         "• Extend backtest to 2+ years using daily (1D) execution (available data).\n"
         "• Add HTF pivot-point S/R levels as additional filters.\n"
         "• Implement walk-forward validation to avoid overfitting the threshold.\n"
         "• Live-paper-trade the 'Combined' scenario for 30 days to confirm edge."),
    ]

    y = 0.88
    for title, body in findings:
        fig.text(0.05, y, f"► {title}", color=GOLD, fontsize=8,
                 fontweight="bold", transform=fig.transFigure)
        y -= 0.030
        # Wrap body text
        for line in body.split("\n"):
            fig.text(0.07, y, line, color=WHITE, fontsize=7.2,
                     transform=fig.transFigure, wrap=True)
            y -= 0.025
        y -= 0.008

    # Bottom footer
    fig.text(0.5, 0.03,
             "Strategy code: src/strategy/  ·  Entry point: strategy_btcusdt.py  "
             "·  Charts: reports/charts/",
             color=GRAY, fontsize=6.5, ha="center")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# F · Top-3 equity comparison
# ─────────────────────────────────────────────────────────────────────────────

def _page_equity_top3(results_store: dict) -> plt.Figure:
    """
    F · Side-by-side comparison of Baseline, Session 08-21 and Regime filter:
    overlaid equity curves (normalized to 100), drawdown corridors, and a
    KPI bar grid.
    """
    fig = _page()
    fig.text(0.5, 0.962, "F · Top-3 Strategy Equity Comparison",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    fig.text(0.5, 0.945, "Baseline  ·  Session 08-21  ·  Regime Filter",
             color=GRAY, fontsize=8, ha="center")

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        left=0.09, right=0.97,
        top=0.92, bottom=0.06,
        hspace=0.52, wspace=0.40,
    )

    present = [(n, c) for n, c in zip(TOP3_SCENARIOS, TOP3_COLORS)
               if n in results_store]

    # ── Panel 1 (top, full-width): normalized equity overlay ─────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    _ax(ax_eq, "Normalized Equity (base = 100)", ylabel="Equity Index")
    for name, color in present:
        eq  = results_store[name]["equity"]
        idx = eq / eq.iloc[0] * 100
        ax_eq.plot(eq.index, idx.values, color=color, lw=1.6, label=name)
        last = float(idx.iloc[-1])
        ax_eq.annotate(f"{last:.1f}", xy=(eq.index[-1], last),
                       xytext=(4, 0), textcoords="offset points",
                       color=color, fontsize=7, va="center")
    ax_eq.axhline(100, color=GRAY, lw=0.6, ls="--", alpha=0.5)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_eq.legend(frameon=False, labelcolor=WHITE, fontsize=7, loc="upper left")

    # ── Panel 2 (middle, full-width): drawdown overlay ───────────────────────
    ax_dd = fig.add_subplot(gs[1, :])
    _ax(ax_dd, "Drawdown (%)", ylabel="%")
    for name, color in present:
        dd = results_store[name]["drawdown"] * 100
        ax_dd.fill_between(dd.index, dd.values, 0,
                           color=color, alpha=0.20, linewidth=0)
        ax_dd.plot(dd.index, dd.values, color=color, lw=0.9)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_dd.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # ── Bottom row: 3 KPI bar charts ─────────────────────────────────────────
    kpi_specs = [
        ("Total Return (%)",  lambda k: k.get("total_return", 0) * 100),
        ("Sharpe (ann.)",     lambda k: k.get("sharpe", 0)),
        ("Max Drawdown (%)",  lambda k: k.get("max_drawdown", 0) * 100),
    ]
    for col, (title, extractor) in enumerate(kpi_specs):
        ax = fig.add_subplot(gs[2, col])
        _ax(ax, title)
        names_p = [n for n, _ in present]
        colors_p = [c for _, c in present]
        vals     = [extractor(results_store[n]["kpis"]) for n in names_p]

        bars = ax.barh(names_p, vals, color=colors_p, alpha=0.82, height=0.55)
        for bar, v in zip(bars, vals):
            ha = "left" if v >= 0 else "right"
            off = abs(v) * 0.05 + 0.05
            ax.text(v + (off if v >= 0 else -off),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:+.2f}", va="center", ha=ha, color=WHITE, fontsize=7)
        ax.axvline(0, color=GRAY, lw=0.5)
        ax.tick_params(labelsize=6)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G · Monte Carlo simulation results
# ─────────────────────────────────────────────────────────────────────────────

def _page_monte_carlo(mc_store: dict,
                      initial_capital: float = 100_000.0) -> plt.Figure:
    """
    G · Monte Carlo fan charts + final-equity and max-DD distributions for
    the top-3 scenarios. 3 rows × 3 columns.
    """
    fig = _page()
    fig.text(0.5, 0.962, "G · Monte Carlo Simulation  (1 000 Bootstrap Resamplings)",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    fig.text(0.5, 0.945,
             "Trade returns resampled with replacement, applied compoundingly",
             color=GRAY, fontsize=8, ha="center")

    present = [(n, c) for n, c in zip(TOP3_SCENARIOS, TOP3_COLORS)
               if n in mc_store and mc_store[n]]
    n_rows  = max(len(present), 1)

    gs = gridspec.GridSpec(
        n_rows, 3,
        figure=fig,
        left=0.09, right=0.97,
        top=0.92, bottom=0.06,
        hspace=0.58, wspace=0.38,
    )

    for row, (name, color) in enumerate(present):
        mc   = mc_store[name]
        paths = mc["paths"]            # (n_sims, n_trades+1)
        fe    = mc["final_equity"]
        mdd   = mc["max_drawdown"] * 100
        n_t   = mc["n_trades"]
        x     = np.arange(n_t + 1)

        p5,  p25, p50, p75, p95 = (np.percentile(paths, p, axis=0)
                                    for p in (5, 25, 50, 75, 95))

        # ── Col 0: Fan chart ──────────────────────────────────────────────────
        ax_f = fig.add_subplot(gs[row, 0])
        _ax(ax_f, f"{name} – Equity Fan",
            xlabel="Trade #", ylabel="Equity ($)")
        ax_f.fill_between(x, p5,  p95,  alpha=0.12, color=color, linewidth=0)
        ax_f.fill_between(x, p25, p75,  alpha=0.28, color=color, linewidth=0)
        ax_f.plot(x, p50, color=color, lw=1.3, label="Median")
        ax_f.plot(x, p5,  color=color, lw=0.6, ls=":", alpha=0.7, label="5th/95th")
        ax_f.plot(x, p95, color=color, lw=0.6, ls=":", alpha=0.7)
        ax_f.axhline(initial_capital, color=GRAY, lw=0.6, ls="--", alpha=0.5)
        ax_f.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
        ax_f.legend(frameon=False, labelcolor=GRAY, fontsize=6)

        # Annotate median final
        med_final = float(p50[-1])
        ax_f.annotate(f"${med_final/1e3:.1f}k",
                      xy=(n_t, med_final), xytext=(-28, 4),
                      textcoords="offset points", color=color, fontsize=6.5)

        # ── Col 1: Final equity distribution ─────────────────────────────────
        ax_fe = fig.add_subplot(gs[row, 1])
        _ax(ax_fe, f"{name} – Final Equity",
            xlabel="Final Equity ($)", ylabel="# Simulations")
        ax_fe.hist(fe, bins=55, color=color, alpha=0.65, edgecolor="none")
        ax_fe.axvline(initial_capital,
                      color=GRAY, lw=0.8, ls="--", label="Initial")
        for pv, ls_s, lab in [(5, ":", "p5"), (50, "-", "p50"), (95, ":", "p95")]:
            vl = float(np.percentile(fe, pv))
            ax_fe.axvline(vl, color=WHITE, lw=0.7, ls=ls_s, label=f"{lab}=${vl/1e3:.1f}k")
        ax_fe.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
        ax_fe.legend(frameon=False, labelcolor=GRAY, fontsize=5.5, loc="upper left")
        # P(profit) / P(ruin) box
        ax_fe.text(0.97, 0.97,
                   f"P(profit)  {mc['p_profit']:.1%}\n"
                   f"P(ruin)    {mc['p_ruin']:.1%}",
                   transform=ax_fe.transAxes, va="top", ha="right",
                   color=WHITE, fontsize=6.5,
                   bbox=dict(facecolor=PANEL, edgecolor=BORDER,
                             alpha=0.9, pad=3))

        # ── Col 2: Max drawdown distribution ─────────────────────────────────
        ax_dd = fig.add_subplot(gs[row, 2])
        _ax(ax_dd, f"{name} – Max Drawdown",
            xlabel="Max DD (%)", ylabel="# Simulations")
        ax_dd.hist(mdd, bins=55, color=RED, alpha=0.60, edgecolor="none")
        for pv, ls_s in [(5, ":"), (50, "-"), (95, ":")]:
            vl = float(np.percentile(mdd, pv))
            ax_dd.axvline(vl, color=WHITE, lw=0.7, ls=ls_s,
                          label=f"p{pv}={vl:.1f}%")
        ax_dd.legend(frameon=False, labelcolor=GRAY, fontsize=6)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# H · Monte Carlo summary stats table
# ─────────────────────────────────────────────────────────────────────────────

def _page_mc_summary(mc_store: dict,
                     initial_capital: float = 100_000.0) -> plt.Figure:
    """
    H · Percentile breakdown table for MC results across the top-3 scenarios.
    """
    from .monte_carlo import mc_summary_table

    fig = _page()
    fig.text(0.5, 0.962, "H · Monte Carlo – Percentile Summary Table",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")

    present = {n: mc_store[n] for n in TOP3_SCENARIOS
               if n in mc_store and mc_store[n]}
    if not present:
        fig.text(0.5, 0.5, "No MC data available",
                 color=GRAY, fontsize=12, ha="center", va="center")
        return fig

    summ = mc_summary_table(present)

    ax = fig.add_axes([0.04, 0.72, 0.92, 0.20])
    ax.set_facecolor(PANEL); ax.axis("off")
    cols  = summ.columns.tolist()
    rows  = summ.index.tolist()
    data  = summ.values
    n_r, n_c = len(rows), len(cols)
    col_w = 0.88 / (n_c + 1)
    row_h = 0.70 / (n_r + 1)

    hdr_y = row_h * n_r + row_h * 0.5
    ax.add_patch(mpatches.Rectangle(
        (0, hdr_y - row_h * 0.5), 1.0, row_h,
        transform=ax.transData, facecolor=GOLD, alpha=0.25))
    ax.text(0.05, hdr_y, "Scenario", color=GOLD, fontsize=8,
            fontweight="bold", ha="center", va="center")
    for j, col in enumerate(cols):
        ax.text(0.10 + col_w * (j + 0.5), hdr_y,
                col.replace(" (", "\n("), color=GOLD, fontsize=6.5,
                fontweight="bold", ha="center", va="center")

    for i, row_name in enumerate(rows):
        y = row_h * (n_r - 1 - i) + row_h * 0.5
        bg = "#1e2530" if i % 2 == 0 else PANEL
        ax.add_patch(mpatches.Rectangle(
            (0, y - row_h * 0.5), 1.0, row_h,
            transform=ax.transData, facecolor=bg))
        color = TOP3_COLORS[TOP3_SCENARIOS.index(row_name)] \
                if row_name in TOP3_SCENARIOS else WHITE
        ax.text(0.05, y, row_name, color=color, fontsize=7.5,
                fontweight="bold", ha="center", va="center")
        for j, val in enumerate(data[i]):
            ax.text(0.10 + col_w * (j + 0.5), y,
                    f"{val:.2f}", color=WHITE, fontsize=7.5,
                    ha="center", va="center")

    # Per-scenario percentile breakdown panels
    n_present = len(present)
    gs2 = gridspec.GridSpec(
        1, n_present,
        figure=fig,
        left=0.06, right=0.97,
        top=0.68, bottom=0.06,
        hspace=0.4, wspace=0.40,
    )

    for col, (name, mc) in enumerate(present.items()):
        color = TOP3_COLORS[TOP3_SCENARIOS.index(name)] \
                if name in TOP3_SCENARIOS else WHITE
        ax_p = fig.add_subplot(gs2[0, col])
        _ax(ax_p, f"{name} – Return Percentiles (%)",
            xlabel="Percentile", ylabel="Total Return (%)")
        pcts_x = mc["summary"]["percentile"].values
        rets_y = mc["summary"]["total_return%"].values
        ax_p.plot(pcts_x, rets_y, color=color, lw=1.5, marker="o",
                  markersize=4)
        ax_p.axhline(0, color=GRAY, lw=0.6, ls="--")
        for pv, rv in zip(pcts_x, rets_y):
            if pv in (5, 50, 95):
                ax_p.annotate(f"{rv:.1f}%",
                              xy=(pv, rv), xytext=(3, 3),
                              textcoords="offset points",
                              color=WHITE, fontsize=6)
        ax_p.set_xlim(-2, 102)
        ax_p.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# I · Leverage & Sizing – equity curves grid
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_LABELS = {"FR": "Fixed Risk", "FF": "Fixed Fraction"}
_LEV_COLORS    = {1: BLUE, 5: GREEN, 10: ORANGE}
_PCT_STYLES    = {0.2: ":", 0.5: "--", 1.0: "-", 2.0: "-."}
_PCT_ALPHA     = {0.2: 0.55, 0.5: 0.70, 1.0: 1.00, 2.0: 0.85}


def _page_leverage_equity(equity_store: dict,
                          scenario: str = "Session 08-21") -> plt.Figure:
    """
    I · 4-row × 3-col grid of equity curves for the chosen scenario.
    Rows = risk / invest level (0.2 %, 0.5 %, 1 %, 2 %)
    Cols = leverage (1×, 5×, 10×)
    Two curves per subplot: FR (blue) vs FF (orange).
    """
    fig = _page()
    fig.text(0.5, 0.962,
             f"I · Leverage & Sizing – Equity Curves  [{scenario}]",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    fig.text(0.5, 0.945,
             "Blue = Fixed Risk  ·  Orange = Fixed Fraction  ·  "
             "Rows = % level  ·  Cols = leverage",
             color=GRAY, fontsize=8, ha="center")

    pct_levels = [0.2, 0.5, 1.0, 2.0]
    lev_levels = [1, 5, 10]
    methods    = [("FR", BLUE), ("FF", ORANGE)]

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        left=0.07, right=0.97, top=0.92, bottom=0.05,
        hspace=0.55, wspace=0.32,
    )
    sc_store = equity_store.get(scenario, {})

    for row, pct in enumerate(pct_levels):
        for col, lev in enumerate(lev_levels):
            ax = fig.add_subplot(gs[row, col])
            _ax(ax, f"{pct:.1f}%  ·  {lev}×  lev",
                ylabel="Equity ($)" if col == 0 else "")

            for short, color in methods:
                key = f"{short}|{pct:.1f}%|{lev}x"
                if key not in sc_store:
                    continue
                eq = sc_store[key]
                lbl = f"{short}  ret={((eq.iloc[-1]/eq.iloc[0])-1)*100:+.1f}%"
                ax.plot(eq.index, eq.values, color=color,
                        lw=1.2 if short == "FR" else 1.0,
                        ls="-" if short == "FR" else "--",
                        alpha=0.9, label=lbl)

            ax.axhline(eq.iloc[0] if sc_store else 100_000,
                       color=GRAY, lw=0.5, ls=":", alpha=0.5)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.legend(frameon=False, labelcolor=WHITE, fontsize=5.5,
                      loc="upper left")
            ax.tick_params(labelsize=6)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# J · KPI heatmaps  (Total Return and Max DD for both methods)
# ─────────────────────────────────────────────────────────────────────────────

def _page_leverage_heatmaps(lev_comp_df: pd.DataFrame,
                             scenario: str = "Session 08-21") -> plt.Figure:
    """
    J · 2×2 heatmap grid:
    top row = Total Return (%), bottom row = Max DD (%)
    left col = Fixed Risk,     right col = Fixed Fraction
    X-axis = Pct level, Y-axis = Leverage
    """
    from .leverage_study import pivot_heatmap

    fig = _page()
    fig.text(0.5, 0.962,
             f"J · KPI Heatmaps – Leverage × Risk Level  [{scenario}]",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    fig.text(0.5, 0.945,
             "Rows = Leverage  ·  Cols = % level  ·  "
             "Left = Fixed Risk  ·  Right = Fixed Fraction",
             color=GRAY, fontsize=8, ha="center")

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.08, right=0.97, top=0.91, bottom=0.07,
        hspace=0.45, wspace=0.28,
    )

    specs = [
        (0, 0, "Fixed Risk",     "Total Ret (%)", sns.color_palette("RdYlGn", 21)),
        (0, 1, "Fixed Fraction", "Total Ret (%)", sns.color_palette("RdYlGn", 21)),
        (1, 0, "Fixed Risk",     "Max DD (%)",    sns.color_palette("RdYlGn_r", 21)),
        (1, 1, "Fixed Fraction", "Max DD (%)",    sns.color_palette("RdYlGn_r", 21)),
    ]

    for r, c, meth, metric, cmap in specs:
        ax = fig.add_subplot(gs[r, c])
        _ax(ax, f"{meth} – {metric}")
        piv = pivot_heatmap(lev_comp_df, scenario, meth, metric)
        if piv.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=GRAY, transform=ax.transAxes)
            continue

        vmax = piv.abs().values.max()
        vmin = -vmax if metric == "Total Ret (%)" else piv.values.min()

        sns.heatmap(
            piv, ax=ax,
            annot=True, fmt=".1f",
            cmap=cmap,
            vmin=vmin, vmax=vmax if metric == "Total Ret (%)" else 0,
            linewidths=0.5, linecolor=BORDER,
            annot_kws={"size": 8, "color": "white"},
            cbar_kws={"shrink": 0.7},
        )
        ax.set_xlabel("Risk / Invest %", color=GRAY, fontsize=8)
        ax.set_ylabel("Leverage",         color=GRAY, fontsize=8)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.set_yticklabels(
            [f"{int(v)}×" for v in piv.index], rotation=0, color=WHITE)
        ax.set_xticklabels(
            [f"{v:.1f}%" for v in piv.columns], rotation=0, color=WHITE)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# K · Best configurations ranking table
# ─────────────────────────────────────────────────────────────────────────────

def _page_best_configs(lev_comp_df: pd.DataFrame) -> plt.Figure:
    """
    K · Top-15 configurations by Sharpe across all scenarios,
    methods, leverage levels and risk levels.
    """
    from .leverage_study import best_configs

    fig = _page()
    fig.text(0.5, 0.962,
             "K · Best Configurations Ranking  (by Sharpe Ratio)",
             color=GOLD, fontsize=11, fontweight="bold", ha="center")
    fig.text(0.5, 0.945,
             "Top 15 across all scenarios × sizing methods × leverage × risk levels",
             color=GRAY, fontsize=8, ha="center")

    df_top = best_configs(lev_comp_df, n=15)
    if df_top.empty:
        fig.text(0.5, 0.5, "No data", ha="center", va="center", color=GRAY)
        return fig

    ax = fig.add_axes([0.02, 0.08, 0.96, 0.84])
    ax.set_facecolor(PANEL); ax.axis("off")

    cols  = df_top.columns.tolist()
    n_r   = len(df_top)
    n_c   = len(cols)
    col_w = 0.96 / (n_c)
    row_h = 0.84 / (n_r + 1)

    # Header
    hdr_y = row_h * n_r + row_h * 0.6
    ax.add_patch(mpatches.Rectangle(
        (0, hdr_y - row_h * 0.5), 1.0, row_h,
        transform=ax.transData, facecolor=GOLD, alpha=0.25))
    for j, col in enumerate(cols):
        ax.text(col_w * (j + 0.5), hdr_y,
                col.replace(" ", "\n"), color=GOLD, fontsize=6.5,
                fontweight="bold", ha="center", va="center")

    # Rows
    for i, (_, row) in enumerate(df_top.iterrows()):
        y    = row_h * (n_r - 1 - i) + row_h * 0.6
        bg   = "#1e2530" if i % 2 == 0 else PANEL
        rank_col = GOLD if i == 0 else (GREEN if i < 3 else WHITE)
        ax.add_patch(mpatches.Rectangle(
            (0, y - row_h * 0.5), 1.0, row_h,
            transform=ax.transData, facecolor=bg))
        for j, col in enumerate(cols):
            val = row[col]
            if isinstance(val, float):
                txt = f"{val:+.2f}" if "Ret" in col or "DD" in col else f"{val:.3f}"
            else:
                txt = str(val)
            c_ = rank_col if j < 4 else WHITE
            ax.text(col_w * (j + 0.5), y, txt,
                    color=c_, fontsize=6.8,
                    ha="center", va="center")

    # Sharpe bar chart on the right side (inset)
    ax2 = fig.add_axes([0.80, 0.10, 0.17, 0.80])
    _ax(ax2, "Sharpe", xlabel="")
    sharpes = df_top["Sharpe"].values
    colors_ = [GOLD if i == 0 else (GREEN if i < 3 else BLUE)
                for i in range(len(sharpes))]
    ys = list(range(len(sharpes) - 1, -1, -1))
    ax2.barh(ys, sharpes, color=colors_, alpha=0.8, height=0.7)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"#{i+1}" for i in range(len(ys))],
                        color=GRAY, fontsize=6)
    ax2.tick_params(labelsize=6)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page L – Walk-Forward equity + per-window returns
# ─────────────────────────────────────────────────────────────────────────────

def _page_walk_forward(wf_result: dict) -> plt.Figure:
    win_df   = wf_result["windows"]
    eq       = wf_result["combined_equity"]
    scenario = wf_result.get("scenario_name", "")
    kpis     = wf_result.get("full_kpis", {})

    fig = _page()
    fig.suptitle("L · Walk-Forward Validation (OOS)",
                 color=WHITE, fontsize=14, y=0.97)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.96, top=0.91, bottom=0.06)

    # ── OOS equity curve ──────────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor(PANEL)
    if not eq.empty:
        ax_eq.plot(eq.index, eq.values, color=BLUE, lw=1.5, label="OOS equity")
        ax_eq.fill_between(eq.index, eq.values, eq.iloc[0],
                            where=(eq.values >= eq.iloc[0]),
                            color=GREEN, alpha=0.15)
        ax_eq.fill_between(eq.index, eq.values, eq.iloc[0],
                            where=(eq.values < eq.iloc[0]),
                            color=RED, alpha=0.2)
        ax_eq.axhline(eq.iloc[0], color=GRAY, lw=0.8, ls="--")

        # shade OOS windows
        for _, row in win_df.iterrows():
            ax_eq.axvspan(pd.Timestamp(row["Train End"]),
                          pd.Timestamp(row["OOS End"]),
                          color=BLUE, alpha=0.04)

    ax_eq.set_title(f"Chained OOS Equity — {scenario}",
                    color=GRAY, fontsize=9)
    ax_eq.tick_params(colors=GRAY, labelsize=7)
    ax_eq.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    for sp in ax_eq.spines.values():
        sp.set_color(BORDER)

    # ── Per-window bar chart ──────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor(PANEL)
    if not win_df.empty:
        rets  = win_df["OOS Return (%)"].values
        colors_ = [GREEN if r >= 0 else RED for r in rets]
        xs = np.arange(len(rets))
        ax_bar.bar(xs, rets, color=colors_, alpha=0.75, width=0.7)
        ax_bar.axhline(0, color=WHITE, lw=0.7, ls="--")
        ax_bar.axhline(float(np.median(rets)), color=ORANGE,
                       lw=1, ls=":", label=f"Median {np.median(rets):+.1f}%")
        ax_bar.set_xticks(xs)
        labels = [f"W{int(w)}\n{str(row['Train End'])[:7]}"
                  for w, row in win_df.iterrows()]
        ax_bar.set_xticklabels(labels, color=GRAY, fontsize=6, rotation=30)
        ax_bar.set_title("OOS Return per Window (%)", color=GRAY, fontsize=9)
        ax_bar.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    ax_bar.tick_params(colors=GRAY, labelsize=7)
    for sp in ax_bar.spines.values():
        sp.set_color(BORDER)

    # ── KPI tiles (bottom row) ────────────────────────────────────────────
    pct_prof  = wf_result.get("pct_profitable", 0)
    med_ret   = wf_result.get("median_oos_ret", 0)
    consist   = wf_result.get("consistency", 0)
    n_win     = wf_result.get("n_windows", 0)
    oos_sharpe = kpis.get("sharpe", 0)
    oos_dd     = kpis.get("max_drawdown", 0)

    tiles = [
        ("Windows\ntested",       f"{n_win}"),
        ("Profitable\nwindows",   f"{pct_prof:.0f}%"),
        ("Median OOS\nreturn",    f"{med_ret:+.1f}%"),
        ("OOS Sharpe\n(combined)",f"{oos_sharpe:.3f}"),
        ("OOS Max DD\n(combined)",f"{oos_dd*100:.1f}%"),
        ("Consistency\n(Sharpe of rets)", f"{consist:+.3f}"),
    ]
    # Use text annotations in a dedicated axes
    ax_tiles = fig.add_axes([0.07, 0.02, 0.89, 0.13], facecolor=BG)
    ax_tiles.set_xlim(0, len(tiles))
    ax_tiles.set_ylim(0, 1)
    ax_tiles.axis("off")
    for i, (label, val) in enumerate(tiles):
        x = i + 0.5
        color = GREEN if ("%" in val and not val.startswith("-") and
                          float(val.replace("%","").replace("+","")) > 0) else BLUE
        ax_tiles.add_patch(
            plt.Rectangle((i + 0.05, 0.05), 0.9, 0.9,
                          facecolor=PANEL, edgecolor=BORDER, lw=0.5))
        ax_tiles.text(x, 0.72, val, ha="center", va="center",
                      color=color, fontsize=12, fontweight="bold")
        ax_tiles.text(x, 0.28, label, ha="center", va="center",
                      color=GRAY, fontsize=7)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page M – Walk-Forward statistics: distribution + Sharpe by window
# ─────────────────────────────────────────────────────────────────────────────

def _page_wf_stats(wf_result: dict) -> plt.Figure:
    win_df = wf_result["windows"]
    trades = wf_result.get("all_oos_trades", pd.DataFrame())

    fig = _page()
    fig.suptitle("M · Walk-Forward Statistics",
                 color=WHITE, fontsize=14, y=0.97)

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38,
                          left=0.07, right=0.96, top=0.90, bottom=0.08)

    # ── Return distribution ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    if not win_df.empty:
        rets = win_df["OOS Return (%)"].values
        ax1.hist(rets, bins=min(12, len(rets)), color=BLUE, alpha=0.75,
                 edgecolor=BORDER, lw=0.5)
        ax1.axvline(float(np.median(rets)), color=ORANGE, lw=1.2, ls="--",
                    label=f"Median {np.median(rets):+.1f}%")
        ax1.axvline(0, color=RED, lw=0.8, ls=":")
        ax1.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    ax1.set_title("OOS Return Distribution", color=GRAY, fontsize=8)
    ax1.tick_params(colors=GRAY, labelsize=7)
    for sp in ax1.spines.values():
        sp.set_color(BORDER)

    # ── Sharpe by window ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    if not win_df.empty:
        sharpes = win_df["Sharpe"].values
        colors_ = [GREEN if s >= 0 else RED for s in sharpes]
        ax2.bar(range(len(sharpes)), sharpes, color=colors_, alpha=0.75, width=0.7)
        ax2.axhline(0, color=WHITE, lw=0.7, ls="--")
        ax2.axhline(1.0, color=GREEN, lw=0.7, ls=":", alpha=0.5)
    ax2.set_title("OOS Sharpe per Window", color=GRAY, fontsize=8)
    ax2.tick_params(colors=GRAY, labelsize=7)
    for sp in ax2.spines.values():
        sp.set_color(BORDER)

    # ── Win Rate by window ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL)
    if not win_df.empty:
        wrs = win_df["Win Rate (%)"].values
        colors_ = [GREEN if w >= 50 else RED for w in wrs]
        ax3.bar(range(len(wrs)), wrs, color=colors_, alpha=0.75, width=0.7)
        ax3.axhline(50, color=WHITE, lw=0.8, ls="--")
    ax3.set_title("OOS Win Rate (%) per Window", color=GRAY, fontsize=8)
    ax3.tick_params(colors=GRAY, labelsize=7)
    for sp in ax3.spines.values():
        sp.set_color(BORDER)

    # ── Max DD by window ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PANEL)
    if not win_df.empty:
        dds = win_df["Max DD (%)"].values
        ax4.bar(range(len(dds)), dds, color=RED, alpha=0.6, width=0.7)
        ax4.axhline(float(np.median(dds)), color=ORANGE, lw=1, ls="--",
                    label=f"Median {np.median(dds):.1f}%")
        ax4.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    ax4.set_title("OOS Max Drawdown (%) per Window", color=GRAY, fontsize=8)
    ax4.tick_params(colors=GRAY, labelsize=7)
    for sp in ax4.spines.values():
        sp.set_color(BORDER)

    # ── OOS trade net PnL distribution ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(PANEL)
    if not trades.empty and "net_pnl" in trades.columns:
        pnls = trades["net_pnl"].values
        wins = pnls[pnls > 0]
        loss = pnls[pnls <= 0]
        ax5.hist(wins, bins=20, color=GREEN, alpha=0.6, label=f"Wins ({len(wins)})")
        ax5.hist(loss, bins=20, color=RED,   alpha=0.6, label=f"Loss ({len(loss)})")
        ax5.axvline(0, color=WHITE, lw=0.7)
        ax5.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    ax5.set_title("OOS Trade PnL Distribution", color=GRAY, fontsize=8)
    ax5.tick_params(colors=GRAY, labelsize=7)
    for sp in ax5.spines.values():
        sp.set_color(BORDER)

    # ── Window KPI table ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(PANEL)
    ax6.axis("off")
    if not win_df.empty:
        show_cols = ["OOS Return (%)", "# Trades", "Win Rate (%)",
                     "Sharpe", "Max DD (%)"]
        tbl_df = win_df[show_cols].copy()
        tbl_df.index.name = "Win"

        col_labels = ["Ret%", "N", "WR%", "Sharpe", "DD%"]
        cell_text  = [[str(tbl_df.iloc[i][c]) for c in show_cols]
                      for i in range(len(tbl_df))]

        tbl = ax6.table(
            cellText   = cell_text,
            rowLabels  = [f"W{int(w)}" for w in tbl_df.index],
            colLabels  = col_labels,
            cellLoc    = "center",
            rowLoc     = "right",
            loc        = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(BG if r == 0 else PANEL)
            cell.set_edgecolor(BORDER)
            cell.set_text_props(color=GRAY if r > 0 else WHITE)

        ax6.set_title("Per-Window KPIs", color=GRAY, fontsize=8, pad=4)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Master PDF assembler
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf(
    tf_data:          Dict,
    signals:          pd.DataFrame,
    oi_df:            pd.DataFrame,
    funding:          pd.Series,
    bt_baseline:      dict,
    analytics:        dict,
    comp_df:          pd.DataFrame,
    results_store:    dict,
    mc_store:         Optional[dict] = None,
    lev_comp_df:      Optional[pd.DataFrame] = None,
    lev_equity_store: Optional[dict] = None,
    lev_scenario:     str = "Session 08-21",
    wf_result:        Optional[dict] = None,
    output_path:      str = "reports/BTCUSDT_Strategy_Report.pdf",
) -> Path:
    """
    Build the full strategy PDF report and return the saved path.

    Parameters
    ----------
    tf_data, signals, oi_df, funding  – strategy inputs
    bt_baseline   – backtest result dict for the Baseline scenario
    analytics     – dict returned by the analytics block in strategy_btcusdt.py
    comp_df       – scenario comparison DataFrame from optimizer.run_comparison()
    results_store – dict of {scenario_name: backtest_result}
    mc_store      – optional dict of {scenario_name: run_monte_carlo result}
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kpis    = bt_baseline["kpis"]
    equity  = bt_baseline["equity"]
    dd      = bt_baseline["drawdown"]
    trades  = bt_baseline["trades"]
    init_cap = float(kpis.get("final_equity", 100_000) /
                     (1 + kpis.get("total_return", 0)))

    pages = [
        ("Cover",               lambda: _page_cover(kpis)),
        ("Exec Summary",        lambda: _page_comparison_table(comp_df)),
        ("A · Signal Analysis", lambda: _page_signal_analysis(
                                    signals,
                                    analytics.get("decay", pd.DataFrame()),
                                    analytics.get("score_quintile", pd.DataFrame()),
                                    analytics.get("fwd_returns", pd.DataFrame()))),
        ("B · Market",          lambda: _page_market(tf_data["1D"], oi_df, funding)),
        ("B · Cyclicality",     lambda: _page_cyclicality(
                                    analytics.get("monthly_season", pd.DataFrame()),
                                    analytics.get("dow_season", pd.DataFrame()),
                                    analytics.get("hour_season", pd.DataFrame()),
                                    analytics.get("cycles", pd.DataFrame()))),
        ("C · Backtest Perf",   lambda: _page_backtest_perf(equity, dd, trades)),
        ("C · Backtest Stats",  lambda: _page_backtest_stats(
                                    equity, trades,
                                    analytics.get("regime_df"),
                                    analytics.get("session_df"),
                                    analytics.get("monthly_pnl"))),
        ("D · Equity Compare",  lambda: _page_equity_comparison(results_store, kpis)),
        ("D · KPI Deltas",      lambda: _page_kpi_deltas(comp_df)),
        ("E · Conclusions",     lambda: _page_conclusions(comp_df, kpis)),
        ("F · Top-3 Equity",    lambda: _page_equity_top3(results_store)),
    ]

    if mc_store:
        pages += [
            ("G · Monte Carlo",     lambda _mc=mc_store, _ic=init_cap:
                                        _page_monte_carlo(_mc, _ic)),
            ("H · MC Summary",      lambda _mc=mc_store, _ic=init_cap:
                                        _page_mc_summary(_mc, _ic)),
        ]

    if lev_comp_df is not None and lev_equity_store is not None:
        pages += [
            ("I · Leverage Equity",  lambda _es=lev_equity_store, _sc=lev_scenario:
                                         _page_leverage_equity(_es, _sc)),
            ("J · KPI Heatmaps",     lambda _df=lev_comp_df, _sc=lev_scenario:
                                         _page_leverage_heatmaps(_df, _sc)),
            ("K · Best Configs",     lambda _df=lev_comp_df:
                                         _page_best_configs(_df)),
        ]

    if wf_result is not None and wf_result.get("n_windows", 0) > 0:
        pages += [
            ("L · Walk-Forward",     lambda _wf=wf_result: _page_walk_forward(_wf)),
            ("M · WF Statistics",    lambda _wf=wf_result: _page_wf_stats(_wf)),
        ]

    metadata = {
        "Title":   "BTCUSDT Multi-TF Strategy Report",
        "Author":  "Quant Pipeline",
        "Subject": "BTCUSDT trading strategy analysis",
        "Creator": "strategy_btcusdt.py",
    }

    with PdfPages(str(out)) as pdf:
        pdf.infodict().update(metadata)

        for name, builder in pages:
            try:
                fig = builder()
                pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
                plt.close(fig)
                print(f"  ✓  {name}")
            except Exception as exc:
                print(f"  ✗  {name}: {exc}")
                import traceback; traceback.print_exc()
                plt.close("all")

    return out
