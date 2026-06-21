"""
Walk-Forward Optimization (WFO) for the BTCUSDT strategy.

Design (fixed-parameter rolling WFO):
  – Parameters are NOT re-optimized per window (fixed scenario).
  – Each window: 6-month warm-up (indicator context) + 2-month OOS test.
  – Step = 2 months → non-overlapping OOS periods.
  – OOS equity curves are chained end-to-end to produce a continuous
    out-of-sample equity series covering the full dataset.

Why fixed-parameter WFO?
  The strategy has no tunable parameters beyond scenario selection.
  The WFO tests temporal stability: does the strategy produce consistent
  returns across different 2-month market regimes?
"""
from __future__ import annotations

from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine   import run_backtest, INIT_CAP, _compute_kpis
from .optimizer import SCENARIOS, apply_filters


# ─────────────────────────────────────────────────────────────────────────────
# Window builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_windows(
    start: pd.Timestamp,
    end:   pd.Timestamp,
    train_months: int,
    test_months:  int,
    step_months:  int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (train_start, train_end, test_end) tuples."""
    windows = []
    tr_start = start
    while True:
        tr_end  = tr_start + relativedelta(months=train_months)
        te_end  = tr_end   + relativedelta(months=test_months)
        if te_end > end:
            break
        windows.append((tr_start, tr_end, te_end))
        tr_start += relativedelta(months=step_months)
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Per-window runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_window(
    df_1h:    pd.DataFrame,
    signals:  pd.DataFrame,
    tr_start: pd.Timestamp,
    tr_end:   pd.Timestamp,
    te_end:   pd.Timestamp,
    initial_capital: float,
) -> Optional[dict]:
    """
    Run backtest on [tr_start, te_end), extract OOS slice [tr_end, te_end).
    Returns None if data insufficient.
    """
    # Full window slice (train + test) for indicator warm-up
    df_w  = df_1h[(df_1h.index >= tr_start) & (df_1h.index < te_end)]
    sig_w = signals[(signals.index >= tr_start) & (signals.index < te_end)]

    if len(df_w) < 200 or len(sig_w) < 50:
        return None

    # Align
    common = df_w.index.intersection(sig_w.index)
    df_w   = df_w.reindex(common)
    sig_w  = sig_w.reindex(common)

    try:
        bt = run_backtest(df_w, sig_w, initial_capital)
    except Exception:
        return None

    equity = bt["equity"]
    trades = bt["trades"]

    # OOS slice: test period only
    oos_mask   = equity.index >= tr_end
    oos_equity = equity[oos_mask]

    if len(oos_equity) < 5:
        return None

    # OOS trades
    oos_trades = pd.DataFrame()
    if not trades.empty and "entry_ts" in trades.columns:
        oos_trades = trades[trades["entry_ts"] >= tr_end].copy()

    # OOS KPIs
    oos_ret   = float(oos_equity.iloc[-1] / oos_equity.iloc[0] - 1)
    oos_n     = len(oos_trades)
    oos_wr    = float((oos_trades["net_pnl"] > 0).mean()) if oos_n > 0 else 0.0
    oos_pf    = 0.0
    if oos_n > 0:
        w = oos_trades["net_pnl"][oos_trades["net_pnl"] > 0]
        l = oos_trades["net_pnl"][oos_trades["net_pnl"] <= 0]
        if len(w) > 0 and len(l) > 0:
            oos_pf = float(w.sum() / (-l.sum()))

    # OOS Sharpe (annualised from hourly bars)
    hrs_per_yr = 24 * 365
    bar_ret    = oos_equity.pct_change().dropna()
    oos_sharpe = 0.0
    if len(bar_ret) > 2 and bar_ret.std() > 0:
        oos_sharpe = float(bar_ret.mean() * hrs_per_yr /
                           (bar_ret.std() * np.sqrt(hrs_per_yr)))

    oos_dd = float((oos_equity / oos_equity.cummax() - 1).min())

    return {
        "tr_start":   tr_start,
        "tr_end":     tr_end,
        "te_end":     te_end,
        "oos_equity": oos_equity,
        "oos_trades": oos_trades,
        "oos_ret":    oos_ret,
        "oos_n":      oos_n,
        "oos_wr":     oos_wr,
        "oos_pf":     oos_pf,
        "oos_sharpe": oos_sharpe,
        "oos_dd":     oos_dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main WFO runner
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward(
    df_1h:           pd.DataFrame,
    signals_full:    pd.DataFrame,
    scenario_name:   str   = "Session 08-21",
    train_months:    int   = 6,
    test_months:     int   = 2,
    step_months:     int   = 2,
    initial_capital: float = INIT_CAP,
) -> dict:
    """
    Rolling walk-forward backtest with non-overlapping OOS windows.

    Parameters
    ----------
    df_1h          : Full 1H OHLCV dataset with indicators.
    signals_full   : Pre-computed signal matrix on full dataset.
    scenario_name  : Which scenario to apply (from optimizer.SCENARIOS).
    train_months   : Warm-up / indicator context window.
    test_months    : OOS test window length.
    step_months    : How far to slide between windows (= test_months for
                     non-overlapping OOS).
    initial_capital: Starting capital per window (equity is normalised).

    Returns
    -------
    dict with keys:
      windows         : DataFrame – per-window KPIs
      combined_equity : pd.Series – chained OOS equity (realistic compound)
      all_oos_trades  : pd.DataFrame – all OOS trades concatenated
      n_windows       : int
      pct_profitable  : float – % of windows with positive OOS return
      median_oos_ret  : float – median OOS return (%)
      consistency     : float – Sharpe of per-window returns (stability metric)
    """
    # Apply scenario filters to the pre-computed full signal matrix
    cfg             = SCENARIOS[scenario_name]
    filtered_signals = apply_filters(signals_full, cfg)

    # Build windows
    start   = df_1h.index[0]
    end     = df_1h.index[-1]
    windows = _build_windows(start, end, train_months, test_months, step_months)

    print(f"  Scenario : {scenario_name}")
    print(f"  Windows  : {len(windows)}  "
          f"(train={train_months}m, test={test_months}m, step={step_months}m)")
    print(f"  Period   : {start.date()} → {end.date()}")
    print()

    results: List[dict] = []
    for i, (tr_start, tr_end, te_end) in enumerate(windows):
        res = _run_window(df_1h, filtered_signals,
                          tr_start, tr_end, te_end, initial_capital)
        if res is None:
            print(f"  Win {i+1:02d}: {tr_start.date()}–{te_end.date()}  SKIPPED")
            continue

        results.append(res)
        sign = "+" if res["oos_ret"] >= 0 else ""
        print(f"  Win {i+1:02d}: "
              f"train {tr_start.date()}–{tr_end.date()} │ "
              f"OOS {tr_end.date()}–{te_end.date()} │ "
              f"ret={sign}{res['oos_ret']*100:.1f}%  "
              f"n={res['oos_n']}  WR={res['oos_wr']*100:.0f}%  "
              f"Sharpe={res['oos_sharpe']:.2f}")

    if not results:
        return {"windows": pd.DataFrame(), "combined_equity": pd.Series(dtype=float),
                "all_oos_trades": pd.DataFrame(), "n_windows": 0,
                "pct_profitable": 0.0, "median_oos_ret": 0.0, "consistency": 0.0}

    # Chain OOS equity curves (compound capital)
    chained   : List[pd.Series] = []
    running   = float(initial_capital)
    for res in results:
        eq_norm = res["oos_equity"] / res["oos_equity"].iloc[0] * running
        chained.append(eq_norm)
        running = float(eq_norm.iloc[-1])

    combined_equity = pd.concat(chained)
    combined_equity = combined_equity[~combined_equity.index.duplicated(keep="last")]

    # Concatenate all OOS trades
    trade_parts = [r["oos_trades"] for r in results if not r["oos_trades"].empty]
    all_oos_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()

    # Window-level summary DataFrame
    rows = []
    for i, res in enumerate(results):
        rows.append({
            "Window":          i + 1,
            "Train Start":     res["tr_start"].date(),
            "Train End":       res["tr_end"].date(),
            "OOS End":         res["te_end"].date(),
            "OOS Return (%)":  round(res["oos_ret"] * 100, 2),
            "# Trades":        res["oos_n"],
            "Win Rate (%)":    round(res["oos_wr"] * 100, 1),
            "Profit Factor":   round(res["oos_pf"], 2),
            "Sharpe":          round(res["oos_sharpe"], 3),
            "Max DD (%)":      round(res["oos_dd"] * 100, 2),
        })
    win_df = pd.DataFrame(rows).set_index("Window")

    oos_rets = win_df["OOS Return (%)"].values
    pct_profitable = float((oos_rets > 0).mean() * 100)
    median_oos_ret = float(np.median(oos_rets))
    consistency    = (float(np.mean(oos_rets) / np.std(oos_rets))
                      if np.std(oos_rets) > 0 else 0.0)

    # Full OOS KPIs (on combined equity)
    dd_combined = (combined_equity / combined_equity.cummax() - 1)
    full_kpis   = _compute_kpis(combined_equity, dd_combined,
                                 all_oos_trades, initial_capital)

    print(f"\n  ── OOS Summary ──────────────────────────────────────────────")
    print(f"  Windows profitable : {pct_profitable:.0f}%  ({(oos_rets>0).sum()}/{len(oos_rets)})")
    print(f"  Median OOS return  : {median_oos_ret:+.2f}%")
    print(f"  Consistency (Sharpe of window returns): {consistency:+.3f}")
    print(f"  Combined OOS Sharpe: {full_kpis.get('sharpe', 0):.3f}")
    print(f"  Combined OOS Return: {full_kpis.get('total_return', 0)*100:+.2f}%")
    print(f"  Combined Max DD    : {full_kpis.get('max_drawdown', 0)*100:.2f}%")

    return {
        "windows":          win_df,
        "combined_equity":  combined_equity,
        "all_oos_trades":   all_oos_trades,
        "n_windows":        len(results),
        "pct_profitable":   pct_profitable,
        "median_oos_ret":   median_oos_ret,
        "consistency":      consistency,
        "full_kpis":        full_kpis,
        "scenario_name":    scenario_name,
    }
