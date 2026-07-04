"""
create_sp500_portfolio_report.py
─────────────────────────────────
S&P 500 sector-rotation portfolio strategy — full validation report.

Strategy: monthly-rebalanced rotation across the 11 SPDR sector ETFs
(top-4 by composite 3/6/12-month momentum), gated by Faber's 10-month-SMA
trend filter on SPY (rotate to a T-bill cash proxy when risk-off) and scaled
by a volatility-targeting overlay (target 18% ann. vol, cap 1.15x).

Validation, reusing src/strategy's framework wherever the methodology
transfers from the BTCUSDT engine to a daily multi-asset portfolio:
  • Walk-forward OOS  (src.equity_strategy.walk_forward, mirrors
    src.strategy.walk_forward's fixed-parameter rolling-window design;
    `_build_windows` is imported directly, unmodified)
  • Monte Carlo        (src.strategy.monte_carlo.run_monte_carlo /
    mc_summary_table, imported and used completely unmodified — fed
    monthly portfolio returns instead of single-position trade P&L, since
    this strategy holds several concurrent sector positions rather than
    one position at a time)
  • Trade analysis      (per-sector holding-period trade log from the
    portfolio engine, analogous to the BTC engine's Trade dataclass)

Output → reports/report_sp500_portfolio.html
Usage  → python create_sp500_portfolio_report.py
"""
from __future__ import annotations

import base64
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.equity_strategy.data_fetcher import (
    fetch_universe, build_adjclose_panel, build_cash_nav,
    SECTOR_ETFS, EXTRA_SECTOR_ETFS,
)
from src.equity_strategy.signals import build_rebalance_plan
from src.equity_strategy.engine import (
    run_portfolio_backtest, compute_kpis, monthly_return_trades, INIT_CAP,
)
from src.equity_strategy.walk_forward import run_walk_forward
from src.strategy.monte_carlo import run_monte_carlo, mc_summary_table

ALL_SECTORS = SECTOR_ETFS + EXTRA_SECTOR_ETFS
N_MC_SIMS = 5_000

# ── Palette (consistent with the rest of the repo's dark-theme reports) ─────
BG, PANEL, BORDER = "#0d1117", "#161b22", "#30363d"
WHITE, GRAY = "#e6edf3", "#8b949e"
GREEN, RED, GOLD, BLUE, PURPLE = "#2ea043", "#f85149", "#e3b341", "#58a6ff", "#d2a8ff"

plt.style.use("dark_background")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.12, color=GRAY, linestyle="--")
    if title:  ax.set_title(title, color=WHITE, fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY, fontsize=8)


def _fig_to_b64(fig, dpi=130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px">'


def chart_equity_comparison(curves: dict, title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(BG)
    colors = [GOLD, BLUE, PURPLE, GREEN, RED]
    for (name, eq), c in zip(curves.items(), colors):
        ax.plot(eq.index, eq.values, label=name, color=c, linewidth=1.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _ax(ax, title, "", "Equity ($, log scale)")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _img(_fig_to_b64(fig))


def chart_drawdown(dd_map: dict, title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor(BG)
    colors = [GOLD, BLUE]
    for (name, dd), c in zip(dd_map.items(), colors):
        ax.fill_between(dd.index, dd.values * 100, 0, color=c, alpha=0.25)
        ax.plot(dd.index, dd.values * 100, color=c, linewidth=1.0, label=name)
    _ax(ax, title, "", "Drawdown (%)")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8, loc="lower left")
    fig.tight_layout()
    return _img(_fig_to_b64(fig))


def chart_trade_histogram(trades_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    fig.patch.set_facecolor(BG)
    pnl = trades_df.loc[trades_df["asset"] != "CASH", "net_pnl"]
    ax.hist(pnl, bins=40, color=BLUE, alpha=0.85, edgecolor=BG)
    ax.axvline(0, color=RED, linewidth=1.0, linestyle="--")
    _ax(ax, "Sector-Trade Net P&L Distribution", "Net P&L ($)", "# Trades")
    fig.tight_layout()
    return _img(_fig_to_b64(fig))


def chart_sector_winrate(trades_df: pd.DataFrame) -> str:
    risky = trades_df[trades_df["asset"] != "CASH"]
    g = risky.groupby("asset")["net_pnl"].agg(
        win_rate=lambda s: (s > 0).mean() * 100, n="count", avg_pnl="mean")
    g = g.sort_values("win_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    fig.patch.set_facecolor(BG)
    colors = [GREEN if v >= 50 else RED for v in g["win_rate"]]
    ax.barh(g.index, g["win_rate"], color=colors, alpha=0.85)
    ax.axvline(50, color=GRAY, linewidth=0.8, linestyle="--")
    _ax(ax, "Win Rate by Sector Holding", "Win Rate (%)")
    fig.tight_layout()
    return _img(_fig_to_b64(fig))


def chart_mc_fan(mc: dict, initial_capital: float) -> str:
    paths = mc["paths"]
    pcts = [5, 25, 50, 75, 95]
    bands = np.percentile(paths, pcts, axis=0)
    x = np.arange(paths.shape[1])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(BG)
    ax.fill_between(x, bands[0], bands[4], color=BLUE, alpha=0.15, label="p5–p95")
    ax.fill_between(x, bands[1], bands[3], color=BLUE, alpha=0.30, label="p25–p75")
    ax.plot(x, bands[2], color=GOLD, linewidth=1.6, label="Median")
    ax.axhline(initial_capital, color=GRAY, linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _ax(ax, "Monte Carlo — Bootstrapped Monthly-Return Paths (5,000 sims)",
        "Months (simulated)", "Equity ($)")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _img(_fig_to_b64(fig))


def chart_mc_dist(mc: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)
    axes[0].hist(mc["total_return"] * 100, bins=50, color=GOLD, alpha=0.85, edgecolor=BG)
    _ax(axes[0], "Total Return Distribution", "Total Return (%)", "# Simulations")
    axes[1].hist(mc["max_drawdown"] * 100, bins=50, color=RED, alpha=0.85, edgecolor=BG)
    _ax(axes[1], "Max Drawdown Distribution", "Max Drawdown (%)", "# Simulations")
    fig.tight_layout()
    return _img(_fig_to_b64(fig))


def chart_wfo_windows(win_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)
    colors = [GREEN if v >= 0 else RED for v in win_df["OOS Return (%)"]]
    ax.bar(win_df.index, win_df["OOS Return (%)"], color=colors, alpha=0.85)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    _ax(ax, "Walk-Forward OOS Return per Window", "Window #", "OOS Return (%)")
    fig.tight_layout()
    return _img(_fig_to_b64(fig))


# ─────────────────────────────────────────────────────────────────────────────
# Table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _table(df: pd.DataFrame, index_name: str = "") -> str:
    return df.to_html(classes="dtable", border=0, index=True, index_names=bool(index_name))


def kpi_row(label: str, kpis: dict) -> dict:
    return {
        "Strategy": label,
        "CAGR (%)": round(kpis.get("cagr", 0) * 100, 2),
        "Total Return (%)": round(kpis.get("total_return", 0) * 100, 1),
        "Sharpe": round(kpis.get("sharpe", 0), 3),
        "Sortino": round(kpis.get("sortino", 0), 3),
        "Calmar": round(kpis.get("calmar", 0), 3),
        "Max DD (%)": round(kpis.get("max_drawdown", 0) * 100, 2),
        "# Trades": kpis.get("n_trades", 0),
        "Win Rate (%)": round(kpis.get("win_rate", 0) * 100, 1),
        "Profit Factor": round(kpis.get("profit_factor", 0), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Fetching S&P 500 sector universe (Yahoo Finance, parquet-cached)…")
    uni = fetch_universe(verbose=True)
    panel = build_adjclose_panel(uni)
    cash_nav = build_cash_nav(uni["^IRX"], panel.index)
    print(f"Panel: {panel.shape[0]:,} trading days  [{panel.index[0].date()} → {panel.index[-1].date()}]")

    # ── Scenario ablation ────────────────────────────────────────────────────
    print("\nRunning scenario ablation …")
    scenarios = {
        "Momentum Only":            dict(use_trend_filter=False, use_vol_target=False, max_leverage=1.0),
        "+ Trend Filter":           dict(use_trend_filter=True,  use_vol_target=False, max_leverage=1.0),
        "+ Trend + Vol Target":     dict(use_trend_filter=True,  use_vol_target=True),   # final config (defaults)
    }
    bt_store, plan_store = {}, {}
    for name, kw in scenarios.items():
        plan = build_rebalance_plan(panel, ALL_SECTORS, **kw)
        bt = run_portfolio_backtest(panel, cash_nav, plan, ALL_SECTORS)
        # KPIs must be computed on the *active* (post-warm-up) equity slice —
        # run_portfolio_backtest returns equity over the full panel, including
        # the ~13-month flat pre-warm-up stub, which would silently lengthen
        # n_years and understate CAGR/Sharpe relative to the SPY benchmark
        # (whose KPIs are computed only from the strategy's own start date).
        eq_active = bt["equity"].loc[plan.index[0]:]
        dd_active = (eq_active - eq_active.cummax()) / eq_active.cummax()
        bt["kpis"] = compute_kpis(eq_active, dd_active, bt["trades"], INIT_CAP)
        bt_store[name] = bt
        plan_store[name] = plan
        print(f"  {name:<24s} CAGR={bt['kpis']['cagr']*100:6.2f}%  "
              f"Sharpe={bt['kpis']['sharpe']:.3f}  MaxDD={bt['kpis']['max_drawdown']*100:6.2f}%")

    final_name = "+ Trend + Vol Target"
    final_bt = bt_store[final_name]
    final_plan = plan_store[final_name]
    start_date = final_plan.index[0]

    spy = panel["SPY"].loc[start_date:]
    spy_equity = INIT_CAP * spy / spy.iloc[0]
    spy_dd = (spy_equity - spy_equity.cummax()) / spy_equity.cummax()
    spy_kpis = compute_kpis(spy_equity, spy_dd, pd.DataFrame(), INIT_CAP)

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print("\nRunning walk-forward validation …")
    wf = run_walk_forward(panel, cash_nav, ALL_SECTORS,
                           plan_kwargs=dict(use_trend_filter=True, use_vol_target=True))
    print(f"  {wf['n_windows']} windows | {wf['pct_profitable']:.0f}% profitable | "
          f"median OOS ret {wf['median_oos_ret']:+.2f}%")

    wf_spy_start = wf["combined_equity"].index[0]
    spy_wf = panel["SPY"].loc[wf_spy_start:]
    spy_wf_equity = INIT_CAP * spy_wf / spy_wf.iloc[0]
    spy_wf_equity = spy_wf_equity.reindex(wf["combined_equity"].index).ffill()

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print("\nRunning Monte Carlo simulation (bootstrapped monthly returns) …")
    mc_store = {}
    for name, bt in bt_store.items():
        eq_active = bt["equity"].loc[plan_store[name].index[0]:]
        mrt = monthly_return_trades(eq_active)
        mc_store[name] = run_monte_carlo(mrt, initial_capital=INIT_CAP, n_sims=N_MC_SIMS, seed=42)
    mc_final = mc_store[final_name]
    mc_compare = mc_summary_table(mc_store)

    # ── Trade analysis ───────────────────────────────────────────────────────
    trades_df = final_bt["trades"]
    by_regime = trades_df.groupby("regime")["net_pnl"].agg(
        n="count", win_rate=lambda s: (s > 0).mean() * 100, avg_pnl="mean", total_pnl="sum")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nRendering charts …")
    chart_main_equity = chart_equity_comparison(
        {final_name: final_bt["equity"].loc[start_date:], "SPY Buy & Hold": spy_equity},
        "Strategy vs SPY Buy & Hold — Equity Curve")
    chart_main_dd = chart_drawdown(
        {final_name: final_bt["drawdown"].loc[start_date:], "SPY Buy & Hold": spy_dd}, "Drawdown Comparison")
    chart_ablation = chart_equity_comparison(
        {**{n: bt_store[n]["equity"].loc[start_date:] for n in scenarios}, "SPY Buy & Hold": spy_equity},
        "Scenario Ablation — Equity Curves")
    chart_trades_hist = chart_trade_histogram(trades_df)
    chart_winrate = chart_sector_winrate(trades_df)
    chart_wfo = chart_wfo_windows(wf["windows"])
    chart_wfo_equity = chart_equity_comparison(
        {"WFO Chained OOS Equity": wf["combined_equity"], "SPY (same OOS periods)": spy_wf_equity},
        "Walk-Forward — Chained OOS Equity vs SPY")
    chart_mc_fan_img = chart_mc_fan(mc_final, INIT_CAP)
    chart_mc_dist_img = chart_mc_dist(mc_final)

    # ── KPI tables ────────────────────────────────────────────────────────────
    kpi_table = pd.DataFrame([
        kpi_row(name, bt_store[name]["kpis"]) for name in scenarios
    ] + [kpi_row("SPY Buy & Hold", spy_kpis)]).set_index("Strategy")

    wfo_summary = pd.DataFrame([{
        "Metric": "Windows", "Value": wf["n_windows"],
    }, {
        "Metric": "% Profitable Windows", "Value": f"{wf['pct_profitable']:.0f}%",
    }, {
        "Metric": "Median OOS Return", "Value": f"{wf['median_oos_ret']:+.2f}%",
    }, {
        "Metric": "Consistency (mean/std of window rets)", "Value": f"{wf['consistency']:.3f}",
    }, {
        "Metric": "Combined OOS CAGR", "Value": f"{wf['full_kpis']['cagr']*100:+.2f}%",
    }, {
        "Metric": "Combined OOS Sharpe", "Value": f"{wf['full_kpis']['sharpe']:.3f}",
    }, {
        "Metric": "Combined OOS Max DD", "Value": f"{wf['full_kpis']['max_drawdown']*100:.2f}%",
    }]).set_index("Metric")

    mc_percentiles = mc_final["summary"].round(2)

    html = build_html(
        panel, kpi_table, chart_main_equity, chart_main_dd, chart_ablation,
        chart_trades_hist, chart_winrate, by_regime, chart_wfo, chart_wfo_equity,
        wf["windows"], wfo_summary, chart_mc_fan_img, chart_mc_dist_img,
        mc_percentiles, mc_compare, mc_final, start_date,
    )

    out_path = REPORTS_DIR / "report_sp500_portfolio.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✓ Report written to {out_path}")


def build_html(panel, kpi_table, chart_main_equity, chart_main_dd, chart_ablation,
                chart_trades_hist, chart_winrate, by_regime, chart_wfo, chart_wfo_equity,
                wfo_windows, wfo_summary, chart_mc_fan_img, chart_mc_dist_img,
                mc_percentiles, mc_compare, mc_final, start_date) -> str:

    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")
    universe_str = ", ".join(SECTOR_ETFS) + " (+ " + ", ".join(EXTRA_SECTOR_ETFS) + " once eligible)"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>S&amp;P 500 Sector-Rotation Strategy — Validation Report</title>
<style>
  body {{ background:{BG}; color:{WHITE}; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         margin:0; padding:0 0 60px 0; }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px; }}
  h1 {{ font-size: 26px; margin-bottom:4px; }}
  h2 {{ font-size: 19px; border-bottom:1px solid {BORDER}; padding-bottom:8px; margin-top:48px; color:{GOLD}; }}
  h3 {{ font-size: 15px; color:{BLUE}; margin-top:24px; }}
  p, li {{ color:{GRAY}; line-height:1.55; font-size:14px; }}
  .meta {{ color:{GRAY}; font-size:13px; margin-bottom:24px; }}
  .card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:8px; padding:18px; margin:16px 0; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  table.dtable {{ border-collapse: collapse; width:100%; font-size:12.5px; margin-top:8px; }}
  table.dtable th {{ background:#1c2128; color:{GOLD}; text-align:right; padding:6px 10px; border-bottom:1px solid {BORDER}; }}
  table.dtable th:first-child, table.dtable td:first-child {{ text-align:left; }}
  table.dtable td {{ text-align:right; padding:5px 10px; border-bottom:1px solid {BORDER}; color:{WHITE}; }}
  table.dtable tr:hover td {{ background:#1c2128; }}
  code {{ background:#1c2128; padding:1px 5px; border-radius:4px; color:{GOLD}; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }}
  .badge.green {{ background:rgba(46,160,67,.15); color:{GREEN}; }}
  .badge.red {{ background:rgba(248,81,73,.15); color:{RED}; }}
  .disclaimer {{ font-size:12px; color:{GRAY}; border-top:1px solid {BORDER}; margin-top:48px; padding-top:16px; }}
</style></head>
<body><div class="wrap">

<h1>S&amp;P 500 Sector-Rotation Portfolio Strategy</h1>
<div class="meta">Generated {generated} · Data: Yahoo Finance daily adjusted close ·
Backtest window: {start_date.date()} → {panel.index[-1].date()}</div>

<div class="card">
<h3 style="margin-top:0">Strategy definition</h3>
<p>
Monthly-rebalanced rotation across the S&amp;P 500 sector SPDR ETFs:
<code>{universe_str}</code>, ranked by a composite momentum score
(equal blend of trailing 3/6/12-month returns). The top <b>4</b> sectors are held
equally weighted, gated by two overlays:
</p>
<ul>
  <li><b>Trend filter</b> — Faber's 10-month SMA rule on SPY (with a 2% hysteresis band
      to reduce whipsaws): fully invested only while SPY's month-end close is above its
      trailing 10-month average; otherwise the book rotates to a synthetic cash / 13-week
      T-bill proxy (built from ^IRX).</li>
  <li><b>Volatility targeting</b> — exposure is scaled to keep the selected sector basket's
      trailing 20-day realised volatility near an 18% annualised target, capped at 1.15x
      (modest, realistic leverage financed at the cash rate) so the book de-risks
      automatically ahead of / during volatility spikes.</li>
</ul>
<p>All decisions at execution date <i>t</i> use data available only through <i>t−1</i>
(no lookahead); rebalancing executes at date <i>t</i>'s close. A 5 bps one-way cost is
applied to traded turnover at every rebalance.</p>
</div>

<h2>1 · Backtest vs SPY Buy &amp; Hold</h2>
<div class="card">{chart_main_equity}</div>
<div class="card">{chart_main_dd}</div>
<div class="card">{_table(kpi_table)}</div>
<p>The full combined overlay (trend filter + vol targeting) modestly outperforms SPY on
CAGR while more than halving the maximum drawdown and roughly doubling the Sharpe
ratio and Calmar ratio (return per unit of drawdown risk) — precisely because it
avoids the bulk of the 2000-02, 2008 and 2022 drawdowns, at the cost of lagging during
some of the strongest melt-up years (2013, 2017, 2019-21, 2023-24).</p>

<h2>2 · Scenario Ablation — Contribution of Each Overlay</h2>
<div class="card">{chart_ablation}</div>
<p>Each overlay is added incrementally to isolate its marginal contribution: pure
relative-momentum rotation, then + the trend/regime filter, then + volatility targeting
(the final configuration used throughout this report).</p>

<h2>3 · Trade Analysis</h2>
<div class="grid2">
  <div class="card">{chart_trades_hist}</div>
  <div class="card">{chart_winrate}</div>
</div>
<div class="card">
<h3 style="margin-top:0">Performance by Market Regime</h3>
{_table(by_regime.round(2))}
</div>

<h2>4 · Walk-Forward Out-of-Sample Validation</h2>
<p>Fixed-parameter rolling walk-forward (15-month warm-up window, 12-month
non-overlapping OOS test, reusing <code>src.strategy.walk_forward._build_windows</code>
for the date-window arithmetic): each window recomputes signals using only its own
in-window data, so every OOS slice is genuinely blind to data outside its window.</p>
<div class="card">{chart_wfo}</div>
<div class="card">{chart_wfo_equity}</div>
<div class="grid2">
  <div class="card">{_table(wfo_summary)}</div>
  <div class="card" style="max-height:340px;overflow:auto">{_table(wfo_windows)}</div>
</div>
<p style="font-size:12px">Note: a few windows (e.g. the 2001-02 and 2008-09 bear-market
years) sit almost entirely in the cash/T-bill regime — their reported Sharpe values are
inflated artifacts of a near-zero-volatility equity curve, not genuine risk-adjusted
outperformance; the underlying return in those windows is simply the risk-free rate.</p>

<h2>5 · Monte Carlo Simulation</h2>
<p>Reusing <code>src.strategy.monte_carlo.run_monte_carlo</code> and
<code>mc_summary_table</code> unmodified (5,000 bootstrap resamples). The resampling
unit here is the strategy's <b>calendar-month portfolio return</b> rather than a single
position's P&amp;L: this portfolio holds several sector positions concurrently, so
individual trade P&amp;Ls are not independent/sequential in the way the original
BTCUSDT single-position engine assumes — monthly portfolio returns are the correct
sequential, non-overlapping unit for this strategy shape.</p>
<div class="card">{chart_mc_fan_img}</div>
<div class="card">{chart_mc_dist_img}</div>
<div class="grid2">
  <div class="card">
    <h3 style="margin-top:0">Percentile Table</h3>
    {_table(mc_percentiles.set_index("percentile"))}
  </div>
  <div class="card">
    <h3 style="margin-top:0">P(Ruin) / P(Profit)</h3>
    <p><span class="badge red">P(Ruin) = {mc_final['p_ruin']*100:.2f}%</span>
       &nbsp; <span class="badge green">P(Profit) = {mc_final['p_profit']*100:.1f}%</span></p>
    <p>P(Ruin) = probability final equity falls below 50% of initial capital;
    P(Profit) = probability final equity exceeds initial capital, across
    {mc_final['n_sims']:,} bootstrap resamples of {mc_final['n_trades']} monthly returns.</p>
  </div>
</div>
<div class="card">
<h3 style="margin-top:0">Monte Carlo Comparison — All Scenarios</h3>
{_table(mc_compare)}
</div>

<div class="disclaimer">
Backtest only — not investment advice. Yahoo Finance adjusted-close data; ETF expense
ratios and bid/ask spread beyond the modelled 5 bps turnover cost are not included.
Past performance, in-sample or out-of-sample, does not guarantee future results;
the trend filter and volatility overlay are rule-based and can underperform SPY in
sustained, low-volatility bull markets (as shown in Section 1). Sector universe:
{universe_str}. Cash proxy: synthetic daily-accrual NAV built from the ^IRX 13-week
T-bill discount yield.
</div>

</div></body></html>"""


if __name__ == "__main__":
    main()
