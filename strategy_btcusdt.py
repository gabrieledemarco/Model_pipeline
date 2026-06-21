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
from src.strategy.engine        import run_backtest
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
                    help="Skip chart generation (faster)")
    args = ap.parse_args()

    t0 = time.time()
    _banner("BTCUSDT Multi-Timeframe Quant Strategy")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/6] Fetching OHLCV data …")
    tf_data = fetch_all_timeframes("BTC-USD")
    if "1H" not in tf_data or len(tf_data["1H"]) < 50:
        sys.exit("ERROR: Insufficient 1H data. Check internet connection.")

    # ── 2. Indicators ────────────────────────────────────────────────────────
    print("\n[2/6] Computing technical indicators …")
    for tf in tf_data:
        tf_data[tf] = add_indicators(tf_data[tf])

    # Synthetic OI and funding on the daily timeframe
    oi_df   = generate_oi(tf_data["1D"]["close"])
    funding = generate_funding(tf_data["1D"]["close"])
    print(f"  OI range: ${oi_df['oi'].min()/1e9:.1f}B – ${oi_df['oi'].max()/1e9:.1f}B")

    # ── 3. Signals ───────────────────────────────────────────────────────────
    print("\n[3/6] Building multi-timeframe signal matrix …")
    signals = build_signal_matrix(tf_data, oi_df, funding)

    n_long    = int((signals["signal"] == 1).sum())
    n_short   = int((signals["signal"] == -1).sum())
    n_neutral = int((signals["signal"] == 0).sum())
    print(f"  Signals on 1H index: {len(signals)} bars")
    print(f"  Long={n_long}  Short={n_short}  Neutral={n_neutral}")
    print(f"  Composite range: [{signals['composite'].min():.1f}, "
          f"{signals['composite'].max():.1f}]")

    # ── 4. Backtest ──────────────────────────────────────────────────────────
    print("\n[4/6] Running backtest …")
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
    print("\n[5/6] Running statistical analysis …")

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

    # ── 6. Charts ────────────────────────────────────────────────────────────
    if not args.no_charts:
        print("\n[6/6] Generating charts …")
        saved = chrt.generate_all(
            tf_data   = tf_data,
            signals   = signals,
            oi_df     = oi_df,
            funding   = funding,
            bt_result = bt,
            analytics = analytics_bundle,
        )
        print(f"\n  {len(saved)} charts saved to: reports/charts/")
    else:
        print("\n[6/6] Charts skipped (--no-charts).")

    elapsed = time.time() - t0
    _banner(f"Done in {elapsed:.1f}s  ·  Charts: reports/charts/")

    # Return for programmatic use
    return {
        "tf_data":    tf_data,
        "signals":    signals,
        "oi_df":      oi_df,
        "funding":    funding,
        "backtest":   bt,
        "analytics":  analytics_bundle,
    }


if __name__ == "__main__":
    main()
