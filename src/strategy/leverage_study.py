"""
Leverage and position-sizing grid study for the BTCUSDT strategy.

Runs a full factorial grid:
  Sizing methods : fixed_risk (FR)  ×  fixed_fraction (FF)
  Risk / invest %: 0.2 %  0.5 %  1.0 %  2.0 %
  Leverage       : 1×   5×   10×
  Scenarios      : Baseline  ·  Session 08-21  ·  Regime filter

Fixed Risk (FR):
    size so that hitting the ATR stop-loss costs exactly ``size_pct`` % of
    equity, regardless of leverage. Leverage only raises the ceiling on how
    large the position can physically get (relevant at high risk levels).

Fixed Fraction (FF):
    invest ``size_pct × leverage`` of equity as position notional.
    At 1 % + 10× leverage → 10 % of equity in notional.
    The actual dollar-risk depends on the ATR stop-distance.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .engine   import run_backtest, INIT_CAP, ATR_SL
from .optimizer import SCENARIOS, apply_filters

# ─────────────────────────────────────────────────────────────────────────────
# Grid constants
# ─────────────────────────────────────────────────────────────────────────────

RISK_LEVELS     : List[float] = [0.002, 0.005, 0.010, 0.020]   # 0.2 % … 2 %
LEVERAGE_LEVELS : List[int]   = [1, 5, 10]
SIZING_METHODS  : List[str]   = ["fixed_risk", "fixed_fraction"]
STUDY_SCENARIOS : List[str]   = ["Baseline", "Session 08-21", "Regime filter"]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_leverage_grid(
    df_1h:           pd.DataFrame,
    signals:         pd.DataFrame,
    initial_capital: float = INIT_CAP,
    risk_levels:     List[float] = RISK_LEVELS,
    leverage_levels: List[int]   = LEVERAGE_LEVELS,
    sizing_methods:  List[str]   = SIZING_METHODS,
    scenario_names:  List[str]   = STUDY_SCENARIOS,
    atr_sl:          float       = ATR_SL,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.Series]]]:
    """
    Run the full leverage × sizing × risk grid and collect KPIs.

    Returns
    -------
    comp_df      : DataFrame indexed by (Scenario, Method, Pct%, Leverage)
    equity_store : nested dict  equity_store[scenario][key] = equity Series
                   where key = "FR|1.0%|5x" or "FF|0.5%|10x" etc.
    """
    rows         : List[dict]                          = []
    equity_store : Dict[str, Dict[str, pd.Series]]    = {s: {} for s in scenario_names}

    for sc_name in scenario_names:
        cfg   = SCENARIOS[sc_name]
        filt  = apply_filters(signals, cfg)

        for method in sizing_methods:
            for pct in risk_levels:
                for lev in leverage_levels:
                    key = f"{'FR' if method=='fixed_risk' else 'FF'}|{pct*100:.1f}%|{lev}x"
                    try:
                        bt = run_backtest(
                            df_1h, filt, initial_capital,
                            atr_sl_override = cfg.atr_sl,
                            sizing_method   = method,
                            size_pct        = pct,
                            leverage        = float(lev),
                        )
                    except Exception as exc:
                        print(f"  ⚠ {sc_name} {key}: {exc}")
                        continue

                    kpis = bt["kpis"]
                    equity_store[sc_name][key] = bt["equity"]

                    # effective risk per trade (as % of initial equity) –
                    # meaningful for FF where it varies with ATR
                    avg_notional = float(
                        bt["trades"]["notional"].mean()
                        if not bt["trades"].empty else 0
                    )
                    eff_risk = float(
                        bt["trades"]["total_fees"].mean() / initial_capital
                        if not bt["trades"].empty else 0
                    )  # approximate: dominated by SL losses

                    rows.append({
                        "Scenario":       sc_name,
                        "Method":         "Fixed Risk" if method == "fixed_risk" else "Fixed Fraction",
                        "Pct (%)":        round(pct * 100, 1),
                        "Leverage":       lev,
                        "# Trades":       int(kpis.get("n_trades", 0)),
                        "Win Rate (%)":   round(kpis.get("win_rate", 0) * 100, 1),
                        "Total Ret (%)":  round(kpis.get("total_return", 0) * 100, 2),
                        "Ann. Ret (%)":   round(kpis.get("total_return", 0) * 100, 2),
                        "Sharpe":         round(kpis.get("sharpe", 0), 3),
                        "Sortino":        round(kpis.get("sortino", 0), 3),
                        "Calmar":         round(kpis.get("calmar", 0), 3),
                        "Max DD (%)":     round(kpis.get("max_drawdown", 0) * 100, 2),
                        "Profit Factor":  round(kpis.get("profit_factor", 0), 2),
                        "Expectancy ($)": round(kpis.get("expectancy", 0), 0),
                        "Final Equity":   round(kpis.get("final_equity", 0), 0),
                        "Avg Notional":   round(avg_notional, 0),
                    })

    comp_df = pd.DataFrame(rows)
    return comp_df, equity_store


# ─────────────────────────────────────────────────────────────────────────────
# Analytics helpers
# ─────────────────────────────────────────────────────────────────────────────

def pivot_heatmap(
    comp_df:  pd.DataFrame,
    scenario: str,
    method:   str,   # "Fixed Risk" or "Fixed Fraction"
    metric:   str,   # column name in comp_df
) -> pd.DataFrame:
    """
    Return a pivot table: rows = Leverage, cols = Pct (%), values = metric.
    Suitable for seaborn heatmap.
    """
    sub = comp_df[(comp_df["Scenario"] == scenario) & (comp_df["Method"] == method)].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot(index="Leverage", columns="Pct (%)", values=metric)
    return pivot.sort_index(ascending=False)   # leverage descending (10 top)


def best_configs(comp_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Top-N configurations by Sharpe ratio (across all scenarios and methods)."""
    return (
        comp_df.sort_values("Sharpe", ascending=False)
               .head(n)
               .reset_index(drop=True)[
                   ["Scenario", "Method", "Pct (%)", "Leverage",
                    "Total Ret (%)", "Sharpe", "Sortino", "Max DD (%)",
                    "Profit Factor", "Win Rate (%)", "Final Equity"]
               ]
    )
