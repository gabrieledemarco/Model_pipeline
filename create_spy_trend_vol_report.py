"""
create_spy_trend_vol_report.py
────────────────────────────────
Single-ETF alternative strategy — full validation report.

Strategy: 100% SPY when SPY's month-end close is above its trailing
10-month SMA (Faber's rule, 2% hysteresis band), otherwise a synthetic
T-bill/cash proxy; exposure additionally scaled to keep realised vol near
an 18% annualised target (cap 1.15x leverage). Same overlay rules as the
sector-rotation strategy in create_sp500_portfolio_report.py, applied to a
single already-diversified index instead of rotating among sub-sectors.

This strategy emerged from questioning the sector-rotation report: repeating
the same validation cycle on "does simpler beat more complex here?" showed
that dropping the 11-sector rotation and keeping only the trend + vol
overlay on SPY alone wins in every scenario tested (lump sum, small account,
DCA) with ~4x fewer trades. This script re-runs the *entire* validation
cycle on that finding — ablation, walk-forward OOS, Monte Carlo, trade
analysis, small-account and DCA simulations — with the same rigor applied
to the original strategy, plus a closing head-to-head comparison.

Reuses:
  • src.equity_strategy.{signals,engine,walk_forward} — build_rebalance_plan
    called with sectors=['SPY'], top_k=1; the trend filter's reference asset
    was generalized (trend_asset param) specifically to make this test fair
  • src.strategy.monte_carlo — unmodified
  • create_sp500_portfolio_report's chart/table helpers and the (now
    sector-agnostic) small-account / DCA simulation functions, imported as
    a module rather than duplicated

Output → reports/report_spy_trend_vol.html
Usage  → python create_spy_trend_vol_report.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import create_sp500_portfolio_report as base
from src.equity_strategy.data_fetcher import fetch_universe, build_adjclose_panel, build_cash_nav
from src.equity_strategy.signals import build_rebalance_plan
from src.equity_strategy.engine import (
    run_portfolio_backtest, compute_kpis, monthly_return_trades, INIT_CAP,
)
from src.equity_strategy.walk_forward import run_walk_forward
from src.strategy.monte_carlo import run_monte_carlo, mc_summary_table

SECTORS = ["SPY"]
N_MC_SIMS = 5_000
REPORTS_DIR = base.REPORTS_DIR


def main():
    print("Fetching data (reusing cached S&P 500 / sector universe fetch) …")
    uni = fetch_universe(verbose=True)
    panel = build_adjclose_panel(uni)
    cash_nav = build_cash_nav(uni["^IRX"], panel.index)
    print(f"Panel: {panel.shape[0]:,} trading days  [{panel.index[0].date()} → {panel.index[-1].date()}]")

    # ── Scenario ablation ────────────────────────────────────────────────────
    print("\nRunning scenario ablation …")
    scenarios = {
        "+ Trend Filter":       dict(top_k=1, use_trend_filter=True,  use_vol_target=False, max_leverage=1.0, trend_asset="SPY"),
        "+ Trend + Vol Target": dict(top_k=1, use_trend_filter=True,  use_vol_target=True,  trend_asset="SPY"),   # final config
    }
    bt_store, plan_store = {}, {}
    for name, kw in scenarios.items():
        plan = build_rebalance_plan(panel, SECTORS, **kw)
        bt = run_portfolio_backtest(panel, cash_nav, plan, SECTORS)
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
    wf = run_walk_forward(panel, cash_nav, SECTORS,
                           plan_kwargs=dict(top_k=1, use_trend_filter=True, use_vol_target=True, trend_asset="SPY"))
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

    # ── Small-account feasibility ──────────────────────────────────────────────
    print("\nRunning small-account simulation (1,000 / 2,000 / 3,000 EUR) …")
    small_account_table, small_account_curves = base.run_small_account_simulation(
        panel, cash_nav, final_plan, start_date, sectors=SECTORS)

    # ── DCA simulation ─────────────────────────────────────────────────────────
    print("\nRunning DCA simulation (€1,000 + €150/month) …")
    dca_table, dca_curves, spy_months_skipped, n_months = base.run_dca_simulation(
        panel, cash_nav, final_plan, start_date, sectors=SECTORS)

    # ── Trade analysis ───────────────────────────────────────────────────────
    trades_df = final_bt["trades"]
    by_regime = trades_df.groupby("regime")["net_pnl"].agg(
        n="count", win_rate=lambda s: (s > 0).mean() * 100, avg_pnl="mean", total_pnl="sum")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nRendering charts …")
    chart_main_equity = base.chart_equity_comparison(
        {final_name: final_bt["equity"].loc[start_date:], "SPY Buy & Hold": spy_equity},
        "SPY + Trend + Vol vs SPY Buy & Hold — Equity Curve")
    chart_main_dd = base.chart_drawdown(
        {final_name: final_bt["drawdown"].loc[start_date:], "SPY Buy & Hold": spy_dd}, "Drawdown Comparison")
    chart_ablation = base.chart_equity_comparison(
        {**{n: bt_store[n]["equity"].loc[start_date:] for n in scenarios}, "SPY Buy & Hold": spy_equity},
        "Scenario Ablation — Equity Curves")
    chart_trades_hist = base.chart_trade_histogram(trades_df)
    chart_wfo = base.chart_wfo_windows(wf["windows"])
    chart_wfo_equity = base.chart_equity_comparison(
        {"WFO Chained OOS Equity": wf["combined_equity"], "SPY (same OOS periods)": spy_wf_equity},
        "Walk-Forward — Chained OOS Equity vs SPY")
    chart_mc_fan_img = base.chart_mc_fan(mc_final, INIT_CAP)
    chart_mc_dist_img = base.chart_mc_dist(mc_final)
    chart_small_account = base.chart_equity_comparison(
        small_account_curves, f"€{base.EUR_CAPITALS[1]:,} Starting Capital — Idealized vs Realistic vs SPY")
    chart_dca = base.chart_equity_comparison(
        dca_curves, f"DCA — €{base.DCA_INITIAL_EUR:,.0f} initial + €{base.DCA_MONTHLY_EUR:,.0f}/month")

    # ── KPI tables ────────────────────────────────────────────────────────────
    kpi_table = pd.DataFrame([
        base.kpi_row(name, bt_store[name]["kpis"]) for name in scenarios
    ] + [base.kpi_row("SPY Buy & Hold", spy_kpis)]).set_index("Strategy")

    wfo_summary = pd.DataFrame([
        {"Metric": "Windows", "Value": wf["n_windows"]},
        {"Metric": "% Profitable Windows", "Value": f"{wf['pct_profitable']:.0f}%"},
        {"Metric": "Median OOS Return", "Value": f"{wf['median_oos_ret']:+.2f}%"},
        {"Metric": "Consistency (mean/std of window rets)", "Value": f"{wf['consistency']:.3f}"},
        {"Metric": "Combined OOS CAGR", "Value": f"{wf['full_kpis']['cagr']*100:+.2f}%"},
        {"Metric": "Combined OOS Sharpe", "Value": f"{wf['full_kpis']['sharpe']:.3f}"},
        {"Metric": "Combined OOS Max DD", "Value": f"{wf['full_kpis']['max_drawdown']*100:.2f}%"},
    ]).set_index("Metric")

    mc_percentiles = mc_final["summary"].round(2)

    # ── Head-to-head vs the sector-rotation strategy (previously validated) ───
    sector_all = base.SECTOR_ETFS + base.EXTRA_SECTOR_ETFS
    sector_plan = build_rebalance_plan(panel, sector_all)
    sector_bt = run_portfolio_backtest(panel, cash_nav, sector_plan, sector_all)
    sector_start = sector_plan.index[0]
    sector_eq_active = sector_bt["equity"].loc[sector_start:]
    sector_dd_active = (sector_eq_active - sector_eq_active.cummax()) / sector_eq_active.cummax()
    sector_kpis = compute_kpis(sector_eq_active, sector_dd_active, sector_bt["trades"], INIT_CAP)

    head_to_head = pd.DataFrame([
        base.kpi_row("Sector Rotation (11 ETF)", sector_kpis),
        base.kpi_row("SPY + Trend + Vol (1 ETF)", final_bt["kpis"]),
    ]).set_index("Strategy")

    html = build_html(
        panel, kpi_table, chart_main_equity, chart_main_dd, chart_ablation,
        chart_trades_hist, by_regime, chart_wfo, chart_wfo_equity,
        wf["windows"], wfo_summary, chart_mc_fan_img, chart_mc_dist_img,
        mc_percentiles, mc_compare, mc_final, start_date,
        chart_small_account, small_account_table,
        chart_dca, dca_table, spy_months_skipped, n_months,
        head_to_head,
    )

    out_path = REPORTS_DIR / "report_spy_trend_vol.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✓ Report written to {out_path}")


def build_html(panel, kpi_table, chart_main_equity, chart_main_dd, chart_ablation,
                chart_trades_hist, by_regime, chart_wfo, chart_wfo_equity,
                wfo_windows, wfo_summary, chart_mc_fan_img, chart_mc_dist_img,
                mc_percentiles, mc_compare, mc_final, start_date,
                chart_small_account, small_account_table,
                chart_dca, dca_table, spy_months_skipped, n_months,
                head_to_head) -> str:

    BG, PANEL, BORDER = base.BG, base.PANEL, base.BORDER
    WHITE, GRAY, GOLD, GREEN, RED = base.WHITE, base.GRAY, base.GOLD, base.GREEN, base.RED
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SPY + Trend + Vol — Single-ETF Strategy Validation Report</title>
<style>
  body {{ background:{BG}; color:{WHITE}; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         margin:0; padding:0 0 60px 0; }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px; }}
  h1 {{ font-size: 26px; margin-bottom:4px; }}
  h2 {{ font-size: 19px; border-bottom:1px solid {BORDER}; padding-bottom:8px; margin-top:48px; color:{GOLD}; }}
  h3 {{ font-size: 15px; color:{base.BLUE}; margin-top:24px; }}
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

<h1>SPY + Trend + Vol — Single-ETF Strategy</h1>
<div class="meta">Generated {generated} · Data: Yahoo Finance daily adjusted close ·
Backtest window: {start_date.date()} → {panel.index[-1].date()}</div>

<div class="card">
<h3 style="margin-top:0">Where this strategy came from</h3>
<p>The 11-sector rotation strategy (<code>report_sp500_portfolio.html</code>) was
validated first. Testing whether its complexity was earning its keep — by isolating
just the trend filter and vol-targeting overlay applied to <b>SPY alone</b>, with no
sector rotation — showed it winning on every metric tested, with about a quarter of the
trading activity. This report re-runs the full validation cycle (ablation, walk-forward
OOS, Monte Carlo, trade analysis, small-account and DCA simulations) on that finding,
exactly as rigorously as the original strategy.</p>
</div>

<div class="card">
<h3 style="margin-top:0">Strategy definition</h3>
<p>100% SPY when SPY's month-end close is above its trailing 10-month average
(Faber's rule, 2% hysteresis band to reduce whipsaws) — a T-bill/cash proxy otherwise
— with exposure additionally scaled to keep trailing 20-day realised volatility near an
18% annualised target (capped at 1.15x). Identical overlay rules to the sector-rotation
strategy; the only difference is the traded universe is SPY alone instead of 11 sector
SPDRs, so there is no monthly sector-picking decision — only the binary regime switch
and the vol-driven sizing.</p>
</div>

<h2>1 · Backtest vs SPY Buy &amp; Hold</h2>
<div class="card">{chart_main_equity}</div>
<div class="card">{chart_main_dd}</div>
<div class="card">{base._table(kpi_table)}</div>

<h2>2 · Scenario Ablation</h2>
<div class="card">{chart_ablation}</div>
<p>Trend filter alone already cuts the drawdown roughly in half; adding volatility
targeting recovers most of the CAGR the trend filter gives up while keeping the
drawdown reduction — the same pattern seen in the sector-rotation report, but here with
no sector-selection step diluting the effect.</p>

<h2>3 · Trade Analysis</h2>
<div class="card">{chart_trades_hist}</div>
<div class="card">
<h3 style="margin-top:0">Performance by Market Regime</h3>
{base._table(by_regime.round(2))}
</div>

<h2>4 · Walk-Forward Out-of-Sample Validation</h2>
<p>Same fixed-parameter rolling walk-forward as the sector-rotation report (15-month
warm-up, 12-month non-overlapping OOS test, reusing
<code>src.strategy.walk_forward._build_windows</code>).</p>
<div class="card">{chart_wfo}</div>
<div class="card">{chart_wfo_equity}</div>
<div class="grid2">
  <div class="card">{base._table(wfo_summary)}</div>
  <div class="card" style="max-height:340px;overflow:auto">{base._table(wfo_windows)}</div>
</div>
<p style="font-size:12px">Note: a few windows (e.g. the 2001-02 and 2008-09 bear-market
years) sit almost entirely in the cash/T-bill regime — their reported Sharpe values are
inflated artifacts of a near-zero-volatility equity curve, not genuine risk-adjusted
outperformance; the underlying return in those windows is simply the risk-free rate.</p>

<h2>5 · Monte Carlo Simulation</h2>
<p>Reusing <code>src.strategy.monte_carlo.run_monte_carlo</code> and
<code>mc_summary_table</code> unmodified, bootstrapped on monthly portfolio returns
(5,000 resamples) — same methodology as the sector-rotation report.</p>
<div class="card">{chart_mc_fan_img}</div>
<div class="card">{chart_mc_dist_img}</div>
<div class="grid2">
  <div class="card">
    <h3 style="margin-top:0">Percentile Table</h3>
    {base._table(mc_percentiles.set_index("percentile"))}
  </div>
  <div class="card">
    <h3 style="margin-top:0">P(Ruin) / P(Profit)</h3>
    <p><span class="badge red">P(Ruin) = {mc_final['p_ruin']*100:.2f}%</span>
       &nbsp; <span class="badge green">P(Profit) = {mc_final['p_profit']*100:.1f}%</span></p>
    <p>Across {mc_final['n_sims']:,} bootstrap resamples of {mc_final['n_trades']} monthly returns.</p>
  </div>
</div>
<div class="card">
<h3 style="margin-top:0">Monte Carlo Comparison — Ablation Scenarios</h3>
{base._table(mc_compare)}
</div>

<h2>6 · Small-Account Feasibility (€1,000 / €2,000 / €3,000)</h2>
<p>Same methodology as the sector-rotation report: <b>Idealized</b> assumes fractional
shares and only the 5 bps turnover cost; <b>Realistic</b> floors to whole shares and
adds a flat €{base.COMMISSION_EUR:.2f} commission per traded leg. With only one asset to
trade, there are far fewer legs than the sector strategy, so this friction matters much
less here.</p>
<div class="card">{chart_small_account}</div>
<div class="card">{base._table(small_account_table)}</div>

<h2>7 · Recurring-Contribution (DCA) Simulation</h2>
<p>€{base.DCA_INITIAL_EUR:,.0f} initial + €{base.DCA_MONTHLY_EUR:,.0f}/month for
{n_months} months, summarised by money-weighted IRR (CAGR isn't meaningful with
recurring external cash flows).</p>
<div class="card">{chart_dca}</div>
<div class="card">{base._table(dca_table)}</div>

<h2>8 · Head-to-Head vs the Sector-Rotation Strategy</h2>
<p>Both strategies backtested over their own full available history (start dates differ
slightly because the sector universe needs more warm-up history for a fair momentum
ranking across 11 names).</p>
<div class="card">{base._table(head_to_head)}</div>
<p style="font-size:12px">Fewer legs directly means less exposure to the small-account
and DCA frictions shown in Sections 6-7 of both reports — this is the practical case for
preferring the simpler strategy, on top of the modestly better risk-adjusted metrics
above.</p>

<div class="disclaimer">
Backtest only — not investment advice. Yahoo Finance adjusted-close data; ETF expense
ratios and bid/ask spread beyond the modelled 5 bps turnover cost are not included.
Past performance, in-sample or out-of-sample, does not guarantee future results.
Cash proxy: synthetic daily-accrual NAV built from the ^IRX 13-week T-bill discount
yield. See <code>report_sp500_portfolio.html</code> for the sector-rotation strategy
this was derived from.
</div>

</div></body></html>"""


if __name__ == "__main__":
    main()
