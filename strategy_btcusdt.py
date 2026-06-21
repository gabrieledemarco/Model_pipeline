#!/usr/bin/env python3
"""
BTCUSDT Multi-Timeframe Quantitative Strategy
══════════════════════════════════════════════

Architecture:
  HTF  (1W / 1D)  → macro regime + primary trend bias
  MTF  (4H)       → intermediate setup quality
  LTF  (1H)       → entry timing + volume confirmation
  OI              → positioning pressure (synthetic, realistic)
  Funding         → contrarian sentiment
  Cyclicality     → seasonal edge (monthly + DoW)

Run:
  python strategy_btcusdt.py [--no-charts]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Local modules ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher  import fetch_all_timeframes, generate_oi, generate_funding
from src.strategy.indicators    import add_indicators
from src.strategy.signals       import build_signal_matrix
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.optimizer      import run_comparison
from src.strategy.monte_carlo    import run_monte_carlo, mc_summary_table
from src.strategy.leverage_study import run_leverage_grid, best_configs, STUDY_SCENARIOS
from src.strategy.report         import generate_pdf, TOP3_SCENARIOS
from src.strategy import analytics as ana
from src.strategy import charts   as chrt


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hline(char="─", n=68): print(char * n)

def _banner(title: str):
    _hline("═")
    print(f"  {title}")
    _hline("═")


def _print_kpis(kpis: dict):
    """Pretty-print the backtest KPI block."""
    _hline()
    print(f"  {'PERFORMANCE SUMMARY':^64}")
    _hline()
    rows = [
        ("Total Return",    f"{kpis.get('total_return', 0)*100:+.2f}%"),
        ("Final Equity",    f"${kpis.get('final_equity', 0):,.0f}"),
        ("Max Drawdown",    f"{kpis.get('max_drawdown', 0)*100:.2f}%"),
        ("Sharpe (ann.)",   f"{kpis.get('sharpe', 0):.3f}"),
        ("Sortino (ann.)",  f"{kpis.get('sortino', 0):.3f}"),
        ("Calmar",          f"{kpis.get('calmar', 0):.3f}"),
        ("# Trades",        str(kpis.get("n_trades", 0))),
        ("Win Rate",        f"{kpis.get('win_rate', 0)*100:.1f}%"),
        ("Avg Win",         f"${kpis.get('avg_win', 0):,.0f}"),
        ("Avg Loss",        f"${kpis.get('avg_loss', 0):,.0f}"),
        ("Profit Factor",   f"{kpis.get('profit_factor', 0):.2f}"),
        ("Expectancy/trade",f"${kpis.get('expectancy', 0):,.0f}"),
        ("Avg Duration",    f"{kpis.get('avg_duration_h', 0):.1f} h"),
        ("Total Fees",      f"${kpis.get('total_fees', 0):,.0f}"),
    ]
    for label, value in rows:
        print(f"  {label:<22}  {value:>20}")
    _hline()


def _print_regime(trades: pd.DataFrame):
    if trades.empty or "regime" not in trades.columns:
        return
    print("\n  Win-Rate by Regime:")
    for regime, grp in trades.groupby("regime"):
        n = len(grp)
        wr = (grp["net_pnl"] > 0).mean() * 100
        print(f"    {regime:<10}  {n:>3} trades   WR={wr:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-charts", action="store_true",
                    help="Skip individual PNG chart generation")
    ap.add_argument("--no-report", action="store_true",
                    help="Skip unified PDF report generation")
    ap.add_argument("--mc-sims", type=int, default=1000,
                    help="Number of Monte Carlo simulations (default: 1000)")
    args = ap.parse_args()

    t0 = time.time()
    _banner("BTCUSDT Multi-Timeframe Quant Strategy")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/10] Fetching OHLCV data …")
    tf_data = fetch_all_timeframes("BTC-USD")
    if "1H" not in tf_data or len(tf_data["1H"]) < 50:
        sys.exit("ERROR: Insufficient 1H data. Check internet connection.")

    # ── 2. Indicators ────────────────────────────────────────────────────────
    print("\n[2/10] Computing technical indicators …")
    for tf in tf_data:
        tf_data[tf] = add_indicators(tf_data[tf])

    # Synthetic OI and funding on the daily timeframe
    oi_df   = generate_oi(tf_data["1D"]["close"])
    funding = generate_funding(tf_data["1D"]["close"])
    print(f"  OI range: ${oi_df['oi'].min()/1e9:.1f}B – ${oi_df['oi'].max()/1e9:.1f}B")

    # ── 3. Signals ───────────────────────────────────────────────────────────
    print("\n[3/10] Building multi-timeframe signal matrix …")
    signals = build_signal_matrix(tf_data, oi_df, funding)

    n_long    = int((signals["signal"] == 1).sum())
    n_short   = int((signals["signal"] == -1).sum())
    n_neutral = int((signals["signal"] == 0).sum())
    print(f"  Signals on 1H index: {len(signals)} bars")
    print(f"  Long={n_long}  Short={n_short}  Neutral={n_neutral}")
    print(f"  Composite range: [{signals['composite'].min():.1f}, "
          f"{signals['composite'].max():.1f}]")

    # ── 4. Backtest ──────────────────────────────────────────────────────────
    print("\n[4/10] Running backtest …")
    bt = run_backtest(tf_data["1H"], signals)

    _print_kpis(bt["kpis"])
    _print_regime(bt["trades"])

    trades = bt["trades"]
    if not trades.empty:
        print(f"\n  Exit breakdown:")
        for reason, grp in trades.groupby("exit_reason"):
            pnl_sum = grp["net_pnl"].sum()
            print(f"    {reason:<15}  {len(grp):>3} trades  "
                  f"ΣPnL=${pnl_sum:+,.0f}")

    # ── 5. Analytics ─────────────────────────────────────────────────────────
    print("\n[5/10] Running statistical analysis …")

    fwd_returns = ana.forward_returns(tf_data["1H"])
    decay_df    = ana.alpha_decay(signals, fwd_returns)
    quant_df    = ana.score_quintile_returns(signals, fwd_returns)
    month_seas  = ana.monthly_seasonality(tf_data["1D"]["close"])
    dow_seas    = ana.dow_seasonality(tf_data["1D"]["close"])
    hour_seas   = ana.hour_seasonality(tf_data["1H"]["close"])
    cycles_df   = ana.detect_cycles(tf_data["1D"]["close"])
    rolling     = ana.compute_rolling_metrics(bt["equity"])
    month_pnl   = ana.monthly_pnl(bt["equity"])
    regime_df   = ana.regime_stats(trades)
    sess_df     = ana.session_stats(trades)
    score_out   = ana.score_vs_outcome(trades)

    analytics_bundle = {
        "fwd_returns":    fwd_returns,
        "decay":          decay_df,
        "score_quintile": quant_df,
        "monthly_season": month_seas,
        "dow_season":     dow_seas,
        "hour_season":    hour_seas,
        "cycles":         cycles_df,
        "monthly_pnl":    month_pnl,
        "regime_df":      regime_df,
        "session_df":     sess_df,
        "score_out":      score_out,
    }

    # Print alpha-decay summary
    print("\n  Alpha-Decay (Long signals) – mean fwd return vs horizon:")
    long_decay = decay_df[decay_df["signal"] == "Long"].sort_values("horizon_h")
    for _, row in long_decay.iterrows():
        bar = "█" * max(0, int(row["mean_ret"] * 2000))
        sig = "**" if row["p_value"] < 0.05 else "  "
        print(f"    {row['horizon_h']:>3}h  {row['mean_ret']*100:+.3f}%  "
              f"IC={row['IC_spearman']:+.3f}  p={row['p_value']:.3f} {sig}  {bar}")

    print("\n  Top dominant cycles (daily bars):")
    for _, row in cycles_df.head(5).iterrows():
        print(f"    Period={row['period_bars']:.1f} bars  "
              f"Amplitude={row['amplitude']:.4f}")

    print("\n  Monthly seasonality (historical avg):")
    for month_name, row in month_seas.iterrows():
        bar = "▪" * max(0, int(row["avg_ret"] * 500))
        neg = "▾" * max(0, int(-row["avg_ret"] * 500))
        print(f"    {month_name:<4}  {row['avg_ret']*100:+.2f}%  "
              f"{'▸' if row['avg_ret'] > 0 else '◂'}{bar}{neg}")

    # ── 6. Scenario comparison ───────────────────────────────────────────────
    print("\n[6/10] Running improvement scenario comparison …")
    comp_df, results_store = run_comparison(tf_data["1H"], signals)

    print("\n  Scenario Comparison:")
    _hline()
    # Print a concise comparison table
    key_cols = ["# Trades", "Win Rate (%)", "Total Ret (%)", "Sharpe",
                "Max DD (%)", "Profit Factor", "Final Equity ($)"]
    available = [c for c in key_cols if c in comp_df.columns]
    print(f"  {'Scenario':<22}", end="")
    for col in available:
        print(f"  {col:>16}", end="")
    print()
    _hline()
    for scenario, row in comp_df.iterrows():
        print(f"  {scenario:<22}", end="")
        for col in available:
            val = row.get(col, 0)
            print(f"  {val:>16}", end="")
        print()
    _hline()

    # ── 7. Monte Carlo simulation ─────────────────────────────────────────────
    print(f"\n[7/10] Running Monte Carlo ({args.mc_sims:,} simulations) …")
    mc_store: dict = {}
    init_cap = INIT_CAP
    for sc_name in TOP3_SCENARIOS:
        if sc_name not in results_store:
            continue
        sc_trades = results_store[sc_name]["trades"]
        if sc_trades is None or sc_trades.empty:
            continue
        mc_store[sc_name] = run_monte_carlo(
            sc_trades, initial_capital=init_cap, n_sims=args.mc_sims)
        mc = mc_store[sc_name]
        p50_ret = float(np.percentile(mc["total_return"] * 100, 50))
        p5_ret  = float(np.percentile(mc["total_return"] * 100, 5))
        p95_ret = float(np.percentile(mc["total_return"] * 100, 95))
        p50_dd  = float(np.percentile(mc["max_drawdown"] * 100, 50))
        print(f"  {sc_name:<20}  "
              f"Ret p50={p50_ret:+.1f}%  [p5={p5_ret:+.1f}%, p95={p95_ret:+.1f}%]  "
              f"MDD p50={p50_dd:.1f}%  "
              f"P(profit)={mc['p_profit']:.1%}")

    if mc_store:
        summ = mc_summary_table(mc_store)
        print("\n  MC Summary Table:")
        _hline()
        print(summ.to_string())
        _hline()

    # ── 9. Leverage & Sizing Grid ────────────────────────────────────────────
    print("\n[9/10] Running leverage × sizing grid "
          "(2 methods × 4 risk levels × 3 leverage) …")
    lev_comp_df, lev_equity_store = run_leverage_grid(tf_data["1H"], signals)

    print("\n  Top-5 configurations by Sharpe:")
    _hline()
    top5 = best_configs(lev_comp_df, n=5)
    print(top5.to_string())
    _hline()

    # ── 8. PNG Charts ────────────────────────────────────────────────────────
    if not args.no_charts:
        print("\n[8/10] Generating individual PNG charts …")
        saved = chrt.generate_all(
            tf_data   = tf_data,
            signals   = signals,
            oi_df     = oi_df,
            funding   = funding,
            bt_result = bt,
            analytics = analytics_bundle,
        )
        print(f"  {len(saved)} charts saved to: reports/charts/")
    else:
        print("\n[8/10] PNG charts skipped (--no-charts).")

    # ── 8. Unified PDF Report ─────────────────────────────────────────────────
    if not args.no_report:
        print("\n[10/10] Generating unified PDF report …")
        pdf_path = generate_pdf(
            tf_data          = tf_data,
            signals          = signals,
            oi_df            = oi_df,
            funding          = funding,
            bt_baseline      = bt,
            analytics        = analytics_bundle,
            comp_df          = comp_df,
            results_store    = results_store,
            mc_store         = mc_store if mc_store else None,
            lev_comp_df      = lev_comp_df,
            lev_equity_store = lev_equity_store,
            lev_scenario     = "Session 08-21",
            output_path      = "reports/BTCUSDT_Strategy_Report.pdf",
        )
        print(f"  PDF report saved to: {pdf_path}")
    else:
        print("\n[10/10] PDF report skipped (--no-report).")

    elapsed = time.time() - t0
    _banner(f"Done in {elapsed:.1f}s  ·  Report: reports/BTCUSDT_Strategy_Report.pdf")

    return {
        "tf_data":           tf_data,
        "signals":           signals,
        "oi_df":             oi_df,
        "funding":           funding,
        "backtest":          bt,
        "analytics":         analytics_bundle,
        "comp_df":           comp_df,
        "results_store":     results_store,
        "mc_store":          mc_store,
        "lev_comp_df":       lev_comp_df,
        "lev_equity_store":  lev_equity_store,
    }


if __name__ == "__main__":
    main()
