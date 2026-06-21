#!/usr/bin/env python3
"""
BTCUSDT Multi-Timeframe Quantitative Strategy
══════════════════════════════════════════════

Architecture:
  HTF  (1W / 1D)  → macro regime + primary trend bias
  MTF  (4H)       → intermediate setup quality
  LTF  (1H)       → entry timing + volume confirmation
  OI/Basis        → real premiumIndexKlines basis from Binance Vision (fallback: synthetic)
  Funding         → real from Binance Vision CDN (fallback: synthetic)
  Cyclicality     → seasonal edge (monthly + DoW)

Data sources:
  1H OHLCV  : Binance Vision CDN (real BTCUSDT perp, 2022-present)
  4H        : resampled from 1H
  1D / 1W   : Yahoo Finance (4 years, macro context)
  Funding   : Binance Vision CDN (real, 8-hourly)
  OI/Basis  : Binance Vision premiumIndexKlines 1H (real, since 2020; fallback: synthetic)

Run:
  python strategy_btcusdt.py [--no-charts] [--no-report] [--mc-sims N]
                             [--quick]  # skip WFO for fast iteration
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

from src.strategy.data_fetcher   import (fetch_extended_data, fetch_real_funding,
                                          fetch_real_oi, generate_oi, generate_funding,
                                          fetch_all_timeframes)
from src.strategy.indicators     import add_indicators
from src.strategy.signals        import build_signal_matrix
from src.strategy.engine         import run_backtest, INIT_CAP
from src.strategy.optimizer      import run_comparison
from src.strategy.monte_carlo    import run_monte_carlo, mc_summary_table
from src.strategy.walk_forward   import run_walk_forward
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
    ap.add_argument("--quick", action="store_true",
                    help="Skip walk-forward (faster iteration)")
    args = ap.parse_args()

    STEPS = 13
    t0 = time.time()
    _banner("BTCUSDT Multi-Timeframe Quant Strategy  [2022 – present]")

    # ── 1. Extended data (Binance Vision + yfinance) ─────────────────────────
    print(f"\n[1/{STEPS}] Fetching 2-year multi-TF dataset …")
    try:
        tf_data = fetch_extended_data(start_year=2022, start_month=1, workers=8)
    except Exception as e:
        print(f"  ⚠ fetch_extended_data failed ({e}), falling back to yfinance")
        tf_data = fetch_all_timeframes("BTC-USD")

    if "1H" not in tf_data or len(tf_data["1H"]) < 200:
        sys.exit("ERROR: Insufficient 1H data. Check internet connection.")

    df_1h_full = tf_data["1H"]
    print(f"\n  1H dataset: {len(df_1h_full):,} bars  "
          f"[{df_1h_full.index[0].date()} → {df_1h_full.index[-1].date()}]")

    # ── 2. Indicators ────────────────────────────────────────────────────────
    print(f"\n[2/{STEPS}] Computing technical indicators …")
    for tf in tf_data:
        if not tf_data[tf].empty:
            tf_data[tf] = add_indicators(tf_data[tf])

    # ── 3. OI (real basis) + Funding ────────────────────────────────────────
    print(f"\n[3/{STEPS}] Loading OI (basis) and funding data …")

    # Real basis from premiumIndexKlines (replaces synthetic OI)
    premium_1h, oi_is_real = fetch_real_oi(df_1h_full)
    oi_label = "Binance Vision premiumIndex (real basis)" if oi_is_real else "synthetic (fallback)"
    print(f"  OI/Basis: {oi_label}", end="")
    if oi_is_real:
        print(f"  bars={len(premium_1h):,}  "
              f"range=[{premium_1h.min()*100:.3f}%, {premium_1h.max()*100:.3f}%]")
    else:
        print()

    # Keep synthetic OI DataFrame for fallback path in build_signal_matrix
    oi_df = generate_oi(tf_data["1D"]["close"])

    funding, is_real = fetch_real_funding(df_1h_full, tf_data["1D"])
    src_label = "Binance Vision (real)" if is_real else "synthetic (fallback)"
    print(f"  Funding : {src_label}  "
          f"mean={funding.mean()*100:.4f}%  "
          f"range=[{funding.min()*100:.3f}%, {funding.max()*100:.3f}%]")

    # ── 4. Signals ───────────────────────────────────────────────────────────
    print(f"\n[4/{STEPS}] Building multi-timeframe signal matrix …")
    signals = build_signal_matrix(tf_data, oi_df, funding,
                                  premium_1h=premium_1h if oi_is_real else None)

    n_long    = int((signals["signal"] == 1).sum())
    n_short   = int((signals["signal"] == -1).sum())
    n_neutral = int((signals["signal"] == 0).sum())
    print(f"  Signals on 1H index: {len(signals):,} bars")
    print(f"  Long={n_long}  Short={n_short}  Neutral={n_neutral}")
    print(f"  Composite range: [{signals['composite'].min():.1f}, "
          f"{signals['composite'].max():.1f}]")

    # ── 5. Baseline backtest (full dataset) ──────────────────────────────────
    print(f"\n[5/{STEPS}] Running baseline backtest (full dataset) …")
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

    # ── 6. Analytics ─────────────────────────────────────────────────────────
    print(f"\n[6/{STEPS}] Running statistical analysis …")

    fwd_returns = ana.forward_returns(tf_data["1H"])
    decay_df    = ana.alpha_decay(signals, fwd_returns)
    quant_df    = ana.score_quintile_returns(signals, fwd_returns)
    month_seas  = ana.monthly_seasonality(tf_data["1D"]["close"])
    dow_seas    = ana.dow_seasonality(tf_data["1D"]["close"])
    hour_seas   = ana.hour_seasonality(tf_data["1H"]["close"])
    cycles_df   = ana.detect_cycles(tf_data["1D"]["close"])
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

    print("\n  Alpha-Decay (Long signals) – mean fwd return vs horizon:")
    long_decay = decay_df[decay_df["signal"] == "Long"].sort_values("horizon_h")
    for _, row in long_decay.iterrows():
        bar = "█" * max(0, int(row["mean_ret"] * 2000))
        sig = "**" if row["p_value"] < 0.05 else "  "
        print(f"    {row['horizon_h']:>3}h  {row['mean_ret']*100:+.3f}%  "
              f"IC={row['IC_spearman']:+.3f}  p={row['p_value']:.3f} {sig}  {bar}")

    print("\n  Monthly seasonality (historical avg):")
    for month_name, row in month_seas.iterrows():
        bar = "▪" * max(0, int(row["avg_ret"] * 500))
        neg = "▾" * max(0, int(-row["avg_ret"] * 500))
        print(f"    {month_name:<4}  {row['avg_ret']*100:+.2f}%  "
              f"{'▸' if row['avg_ret'] > 0 else '◂'}{bar}{neg}")

    # ── 7. Scenario comparison ───────────────────────────────────────────────
    print(f"\n[7/{STEPS}] Running scenario comparison …")
    comp_df, results_store = run_comparison(tf_data["1H"], signals)

    print("\n  Scenario Comparison:")
    _hline()
    key_cols  = ["# Trades", "Win Rate (%)", "Total Ret (%)", "Sharpe",
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
            print(f"  {row.get(col, 0):>16}", end="")
        print()
    _hline()

    # ── 8. Monte Carlo (in-sample, Top-3 scenarios) ──────────────────────────
    print(f"\n[8/{STEPS}] Running Monte Carlo ({args.mc_sims:,} sims, in-sample) …")
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

    # ── 9. Walk-Forward Validation ───────────────────────────────────────────
    wf_result: dict = {}
    if not args.quick:
        print(f"\n[9/{STEPS}] Running walk-forward validation …")
        print(f"  (train=6m, test=2m, step=2m, scenario=Session 08-21)")
        wf_result = run_walk_forward(
            df_1h          = tf_data["1H"],
            signals_full   = signals,
            scenario_name  = "Session 08-21",
            train_months   = 6,
            test_months    = 2,
            step_months    = 2,
            initial_capital = init_cap,
        )
    else:
        print(f"\n[9/{STEPS}] Walk-forward skipped (--quick).")

    # ── 10. MC on OOS trades ─────────────────────────────────────────────────
    wf_mc: dict = {}
    if wf_result and not wf_result.get("all_oos_trades", pd.DataFrame()).empty:
        print(f"\n[10/{STEPS}] Running Monte Carlo on OOS trades …")
        oos_tr = wf_result["all_oos_trades"]
        wf_mc  = run_monte_carlo(oos_tr, initial_capital=init_cap, n_sims=args.mc_sims)
        p50_r  = float(np.percentile(wf_mc["total_return"] * 100, 50))
        p5_r   = float(np.percentile(wf_mc["total_return"] * 100, 5))
        p95_r  = float(np.percentile(wf_mc["total_return"] * 100, 95))
        print(f"  OOS MC: Ret p50={p50_r:+.1f}%  [p5={p5_r:+.1f}%, p95={p95_r:+.1f}%]  "
              f"P(profit)={wf_mc['p_profit']:.1%}  P(ruin)={wf_mc['p_ruin']:.1%}")
    else:
        print(f"\n[10/{STEPS}] OOS MC skipped (no OOS trades).")

    # ── 11. Leverage & Sizing Grid ───────────────────────────────────────────
    print(f"\n[11/{STEPS}] Running leverage × sizing grid "
          "(2 methods × 4 risk levels × 3 leverage) …")
    lev_comp_df, lev_equity_store = run_leverage_grid(tf_data["1H"], signals)

    print("\n  Top-5 configurations by Sharpe:")
    _hline()
    print(best_configs(lev_comp_df, n=5).to_string())
    _hline()

    # ── 12. PNG Charts ───────────────────────────────────────────────────────
    if not args.no_charts:
        print(f"\n[12/{STEPS}] Generating individual PNG charts …")
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
        print(f"\n[12/{STEPS}] PNG charts skipped (--no-charts).")

    # ── 13. Unified PDF Report ────────────────────────────────────────────────
    if not args.no_report:
        print(f"\n[13/{STEPS}] Generating unified PDF report …")
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
            wf_result        = wf_result if wf_result else None,
            output_path      = "reports/BTCUSDT_Strategy_Report.pdf",
        )
        print(f"  PDF report saved to: {pdf_path}")
    else:
        print(f"\n[13/{STEPS}] PDF report skipped (--no-report).")

    elapsed = time.time() - t0
    bars_1h  = len(tf_data["1H"])
    date_range = (f"{tf_data['1H'].index[0].date()} → "
                  f"{tf_data['1H'].index[-1].date()}")
    _banner(f"Done in {elapsed:.1f}s  ·  {bars_1h:,} 1H bars [{date_range}]"
            f"  ·  Report: reports/BTCUSDT_Strategy_Report.pdf")

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
        "wf_result":         wf_result,
        "wf_mc":             wf_mc,
        "lev_comp_df":       lev_comp_df,
        "lev_equity_store":  lev_equity_store,
    }


if __name__ == "__main__":
    main()
