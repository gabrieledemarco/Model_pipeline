"""
Walk-Forward Optimization (WFO) for the S&P 500 sector-rotation strategy.

Same fixed-parameter rolling-window design as src.strategy.walk_forward for
the BTCUSDT strategy (no per-window re-fitting — the strategy has no tunable
parameters beyond the pre-specified rule set; this tests temporal stability,
not parameter search). `_build_windows` is reused directly since it is pure
calendar arithmetic, independent of asset class or bar frequency.

Design: each window recomputes signals using *only* data inside the window
(train_months of warm-up + test_months of OOS test), so the OOS slice is
genuinely out-of-sample with respect to that window's own signal history —
consistent with how the BTC WFO isolates each window.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.strategy.walk_forward import _build_windows  # pure date arithmetic, asset-agnostic

from .engine import run_portfolio_backtest, compute_kpis, INIT_CAP
from .signals import build_rebalance_plan


def _run_window(
    panel: pd.DataFrame,
    cash_nav: pd.Series,
    sectors: List[str],
    tr_start: pd.Timestamp,
    tr_end: pd.Timestamp,
    te_end: pd.Timestamp,
    initial_capital: float,
    plan_kwargs: dict | None = None,
) -> Optional[dict]:
    """Run the strategy on [tr_start, te_end) using only in-window data; extract OOS slice [tr_end, te_end)."""
    idx = panel.index
    df_w = panel[(idx >= tr_start) & (idx < te_end)]
    if len(df_w) < 300:
        return None

    cash_w = cash_nav.reindex(df_w.index)
    plan_w = build_rebalance_plan(df_w, sectors, **(plan_kwargs or {}))
    if plan_w.empty:
        return None

    bt = run_portfolio_backtest(df_w, cash_w, plan_w, sectors, initial_capital)
    equity = bt["equity"]
    trades = bt["trades"]

    oos_mask = equity.index >= tr_end
    oos_equity = equity[oos_mask]
    if len(oos_equity) < 5:
        return None

    oos_trades = pd.DataFrame()
    if not trades.empty:
        oos_trades = trades[trades["entry_ts"] >= tr_end].copy()

    oos_dd = (oos_equity / oos_equity.cummax() - 1)
    oos_kpis = compute_kpis(oos_equity, oos_dd, oos_trades, float(oos_equity.iloc[0]))

    return {
        "tr_start": tr_start, "tr_end": tr_end, "te_end": te_end,
        "oos_equity": oos_equity, "oos_trades": oos_trades, "oos_kpis": oos_kpis,
    }


def run_walk_forward(
    panel: pd.DataFrame,
    cash_nav: pd.Series,
    sectors: List[str],
    train_months: int = 15,
    test_months: int = 12,
    step_months: int = 12,
    initial_capital: float = INIT_CAP,
    plan_kwargs: dict | None = None,
) -> dict:
    """
    Rolling walk-forward backtest with non-overlapping annual OOS windows.

    Returns dict with keys: windows, combined_equity, all_oos_trades,
    n_windows, pct_profitable, median_oos_ret, consistency, full_kpis.
    """
    start, end = panel.index[0], panel.index[-1]
    windows = _build_windows(start, end, train_months, test_months, step_months)

    results: List[dict] = []
    for tr_start, tr_end, te_end in windows:
        res = _run_window(panel, cash_nav, sectors, tr_start, tr_end, te_end,
                           initial_capital, plan_kwargs)
        if res is not None:
            results.append(res)

    if not results:
        return {"windows": pd.DataFrame(), "combined_equity": pd.Series(dtype=float),
                "all_oos_trades": pd.DataFrame(), "n_windows": 0,
                "pct_profitable": 0.0, "median_oos_ret": 0.0, "consistency": 0.0}

    chained: List[pd.Series] = []
    running = float(initial_capital)
    for res in results:
        eq_norm = res["oos_equity"] / res["oos_equity"].iloc[0] * running
        chained.append(eq_norm)
        running = float(eq_norm.iloc[-1])

    combined_equity = pd.concat(chained)
    combined_equity = combined_equity[~combined_equity.index.duplicated(keep="last")]

    trade_parts = [r["oos_trades"] for r in results if not r["oos_trades"].empty]
    all_oos_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()

    rows = []
    for i, res in enumerate(results):
        k = res["oos_kpis"]
        rows.append({
            "Window": i + 1,
            "Train Start": res["tr_start"].date(),
            "OOS Start": res["tr_end"].date(),
            "OOS End": res["te_end"].date(),
            "OOS Return (%)": round(k.get("total_return", 0) * 100, 2),
            "# Trades": k.get("n_trades", 0),
            "Win Rate (%)": round(k.get("win_rate", 0) * 100, 1),
            "Sharpe": round(k.get("sharpe", 0), 3),
            "Max DD (%)": round(k.get("max_drawdown", 0) * 100, 2),
        })
    win_df = pd.DataFrame(rows).set_index("Window")

    oos_rets = win_df["OOS Return (%)"].values
    pct_profitable = float((oos_rets > 0).mean() * 100)
    median_oos_ret = float(np.median(oos_rets))
    consistency = (float(np.mean(oos_rets) / np.std(oos_rets))
                   if np.std(oos_rets) > 0 else 0.0)

    dd_combined = (combined_equity / combined_equity.cummax() - 1)
    full_kpis = compute_kpis(combined_equity, dd_combined, all_oos_trades, initial_capital)

    return {
        "windows": win_df,
        "combined_equity": combined_equity,
        "all_oos_trades": all_oos_trades,
        "n_windows": len(results),
        "pct_profitable": pct_profitable,
        "median_oos_ret": median_oos_ret,
        "consistency": consistency,
        "full_kpis": full_kpis,
    }
