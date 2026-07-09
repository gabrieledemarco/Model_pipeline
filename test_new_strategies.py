#!/usr/bin/env python3
"""
New Strategy Testing Suite
══════════════════════════
Tests all new scenario variants (ADX Filter, OBV Enhanced, DD Control,
Adaptive Vol, Trend Quality, Ultra Select) alongside existing scenarios,
runs Walk-Forward Validation on the top candidates, and evaluates with
Monte Carlo to identify the most robust strategies for live deployment.

Run:
  python test_new_strategies.py [--quick]
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher  import (fetch_extended_data, fetch_real_funding,
                                         fetch_real_oi, generate_oi)
from src.strategy.indicators    import add_indicators
from src.strategy.signals       import build_signal_matrix
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.optimizer     import SCENARIOS, apply_filters, run_comparison
from src.strategy.monte_carlo   import run_monte_carlo
from src.strategy.walk_forward  import run_walk_forward

import argparse

HLINE = "─" * 72
DLINE = "═" * 72

def _banner(t): print(f"\n{DLINE}\n  {t}\n{DLINE}")
def _hdr(t):    print(f"\n{HLINE}\n  {t}\n{HLINE}")


# ── Calmar ratio helper ───────────────────────────────────────────────────────

def calmar(kpis: dict) -> float:
    dd = kpis.get("max_drawdown", 0)
    ret = kpis.get("total_return", 0)
    return ret / abs(dd) if dd < 0 else 0.0


# ── Ranking function: balanced score for live deployment ─────────────────────

def live_score(kpis: dict) -> float:
    """
    Composite rank score prioritising:
      - Low drawdown (40 %)
      - Profit factor > 1 (30 %)
      - Trade-level Sharpe proxy (30 %)
    Penalises Max DD > 40 % heavily (live trading cannot survive -40 % equity).
    """
    dd  = kpis.get("max_drawdown", -1.0)
    pf  = kpis.get("profit_factor", 0.0)
    ret = kpis.get("total_return", 0.0)
    n   = kpis.get("n_trades", 0)

    if n < 30:          # too few trades → statistically unreliable
        return -999.0
    if dd < -0.60:      # hard reject: drawdown > 60 %
        return -999.0

    dd_score  = 1.0 + dd        # -84 % → 0.16 ; -20 % → 0.80
    pf_score  = min(pf - 1.0, 1.0)  # PF 1.21 → 0.21 ; PF 2.0 → 1.0
    ret_score = min(ret, 5.0) / 5.0  # cap at +500 %

    return 0.40 * dd_score + 0.30 * pf_score + 0.30 * ret_score


# ── Full comparison across ALL scenarios ─────────────────────────────────────

def run_full_comparison(df_1h, signals, initial_capital=INIT_CAP):
    rows = []
    results_store = {}

    for name, cfg in SCENARIOS.items():
        filt = apply_filters(signals, cfg)
        bt   = run_backtest(df_1h, filt, initial_capital,
                            atr_sl_override=cfg.atr_sl,
                            dd_pause_pct=cfg.dd_pause_pct,
                            adaptive_vol=cfg.adaptive_vol)
        kpis = bt["kpis"]
        results_store[name] = bt

        n_sig = int((filt["signal"] != 0).sum())
        score = live_score(kpis)

        rows.append({
            "Scenario":        name,
            "# Signals":       n_sig,
            "# Trades":        kpis.get("n_trades", 0),
            "Win Rate (%)":    round(kpis.get("win_rate", 0) * 100, 1),
            "Total Ret (%)":   round(kpis.get("total_return", 0) * 100, 2),
            "Profit Factor":   round(kpis.get("profit_factor", 0), 2),
            "Sharpe":          round(kpis.get("sharpe", 0), 3),
            "Max DD (%)":      round(kpis.get("max_drawdown", 0) * 100, 2),
            "Calmar":          round(calmar(kpis), 2),
            "Expectancy ($)":  round(kpis.get("expectancy", 0), 0),
            "Live Score":      round(score, 3),
        })

    df = pd.DataFrame(rows).set_index("Scenario").sort_values("Live Score", ascending=False)
    return df, results_store


# ── Walk-Forward for top N candidates ────────────────────────────────────────

def run_wfo_candidates(df_1h, signals, candidates, initial_capital=INIT_CAP):
    results = {}
    for name in candidates:
        _hdr(f"WFO: {name}")
        wf = run_walk_forward(
            df_1h=df_1h,
            signals_full=signals,
            scenario_name=name,
            train_months=6,
            test_months=2,
            step_months=2,
            initial_capital=initial_capital,
        )
        results[name] = wf
    return results


# ── Monte Carlo on OOS trades ─────────────────────────────────────────────────

def run_mc_oos(wfo_results, initial_capital=INIT_CAP, n_sims=1000):
    mc_results = {}
    for name, wf in wfo_results.items():
        oos_tr = wf.get("all_oos_trades", pd.DataFrame())
        if oos_tr.empty:
            continue
        mc = run_monte_carlo(oos_tr, initial_capital=initial_capital, n_sims=n_sims)
        mc_results[name] = mc
        p50  = np.percentile(mc["total_return"] * 100, 50)
        p5   = np.percentile(mc["total_return"] * 100,  5)
        p95  = np.percentile(mc["total_return"] * 100, 95)
        p50d = np.percentile(mc["max_drawdown"] * 100, 50)
        print(f"  {name:<22}  Ret p50={p50:+.1f}%  "
              f"[p5={p5:+.1f}%, p95={p95:+.1f}%]  "
              f"MDD p50={p50d:.1f}%  "
              f"P(profit)={mc['p_profit']:.1%}  P(ruin)={mc['p_ruin']:.2%}")
    return mc_results


# ── Print comparison table ────────────────────────────────────────────────────

def print_comparison(df: pd.DataFrame):
    cols = ["# Trades", "Win Rate (%)", "Total Ret (%)", "Profit Factor",
            "Sharpe", "Max DD (%)", "Calmar", "Expectancy ($)", "Live Score"]
    avail = [c for c in cols if c in df.columns]
    hdr = f"  {'Scenario':<22}"
    for c in avail:
        hdr += f"  {c:>14}"
    print(hdr)
    print(HLINE)
    for sc, row in df.iterrows():
        line = f"  {sc:<22}"
        for c in avail:
            v = row.get(c, 0)
            line += f"  {v:>14}"
        print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Skip WFO (faster iteration)")
    ap.add_argument("--mc-sims", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    _banner("New Strategy Comparison Suite  [BTCUSDT 2022–present]")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/5] Fetching multi-TF data …")
    tf_data = fetch_extended_data(start_year=2022, start_month=1, workers=8)
    print(f"  1H raw: {len(tf_data['1H']):,} bars  "
          f"[{tf_data['1H'].index[0].date()} → {tf_data['1H'].index[-1].date()}]")

    # ── 2. Indicators + Signals ───────────────────────────────────────────────
    print("\n[2/5] Computing indicators & signal matrix …")
    for tf in tf_data:
        if not tf_data[tf].empty:
            tf_data[tf] = add_indicators(tf_data[tf])

    # df_1h must be retrieved AFTER add_indicators (add_indicators returns a copy)
    df_1h = tf_data["1H"]
    print(f"  1H enriched: {len(df_1h):,} bars, {len(df_1h.columns)} columns")

    premium_1h, oi_is_real = fetch_real_oi(df_1h)
    oi_df = generate_oi(tf_data["1D"]["close"])
    funding, _ = fetch_real_funding(df_1h, tf_data["1D"])

    signals = build_signal_matrix(tf_data, oi_df, funding,
                                  premium_1h=premium_1h if oi_is_real else None)

    print(f"  ADX cols in signals: {[c for c in signals.columns if 'adx' in c or 'obv' in c or 'di_' in c]}")

    # ── 3. Full scenario comparison ───────────────────────────────────────────
    print("\n[3/5] Running full scenario comparison (all scenarios) …")
    comp_df, results_store = run_full_comparison(df_1h, signals)

    _hdr("STRATEGY COMPARISON  (sorted by Live Score)")
    print_comparison(comp_df)

    # Separate original vs new
    new_scenarios = {"ADX Filter", "OBV Enhanced", "DD Control",
                     "Adaptive Vol", "Trend Quality", "Ultra Select"}
    print(f"\n  {'─'*40}")
    print("  NEW strategies highlighted:")
    for sc in comp_df.index:
        if sc in new_scenarios:
            row = comp_df.loc[sc]
            print(f"  ★ {sc:<22}  DD={row['Max DD (%)']:+.1f}%  "
                  f"PF={row['Profit Factor']:.2f}  "
                  f"Ret={row['Total Ret (%)']:+.1f}%  "
                  f"Score={row['Live Score']:.3f}")

    # ── 4. Walk-Forward on top 3 candidates ──────────────────────────────────
    if not args.quick:
        top3 = comp_df[comp_df["Live Score"] > -999].head(3).index.tolist()
        print(f"\n[4/5] Walk-Forward on top-3 candidates: {top3} …")
        wfo_results = run_wfo_candidates(df_1h, signals, top3)

        # ── 5. Monte Carlo on OOS trades ──────────────────────────────────────
        _hdr("Monte Carlo — OOS Trades (1000 simulations)")
        mc_results = run_mc_oos(wfo_results, n_sims=args.mc_sims)

        # Final summary table
        _hdr("WALK-FORWARD SUMMARY")
        for name, wf in wfo_results.items():
            fk = wf.get("full_kpis", {})
            print(f"  {name:<24}  "
                  f"Windows profitable: {wf.get('pct_profitable', 0):.0f}%  "
                  f"OOS Sharpe: {fk.get('sharpe', 0):.3f}  "
                  f"OOS Return: {fk.get('total_return', 0)*100:+.1f}%  "
                  f"OOS MaxDD: {fk.get('max_drawdown', 0)*100:.1f}%")
    else:
        print("\n[4/5] WFO skipped (--quick).")
        print("[5/5] OOS MC skipped (--quick).")
        wfo_results = {}
        mc_results  = {}

    elapsed = time.time() - t0
    _banner(f"Done in {elapsed:.1f}s")

    return comp_df, results_store, wfo_results


if __name__ == "__main__":
    main()
