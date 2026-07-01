"""
Scenario comparison engine for the BTCUSDT strategy.

Applies improvement filters to the signal matrix and runs parallel
backtests to measure incremental impact of each enhancement.

Improvements tested
───────────────────
1. Regime filter       – trade only in direction of daily 200-EMA regime
                          (bear → shorts only; bull → longs only)
2. Strong signals      – raise composite threshold to ±18
3. Wide stop loss      – ATR_SL 2.0 → 2.5 (more breathing room)
4. Session filter      – signals only 08:00-21:00 UTC (London + NY)
5. Monthly filter      – block counter-seasonal entries
6. Combined            – all of the above
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import run_backtest, INIT_CAP


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    name:             str   = "Baseline"
    long_threshold:   float = 5.0
    short_threshold:  float = -5.0
    regime_filter:    bool  = False          # only trade with 200-EMA trend
    session_hours:    Optional[Tuple[int,int]] = None  # (8, 21) = London+NY
    monthly_filter:   bool  = False          # block counter-seasonal entries
    atr_sl:           float = 2.0
    # ── New parameters ───────────────────────────────────────────────────────
    adx_threshold:    float = 0.0            # min ADX required for entry (0=off)
    use_obv:          bool  = False          # add s_obv (weight=1) to composite
    dd_pause_pct:     float = 0.0            # suspend entries on DD > X (0=off)
    adaptive_vol:     bool  = False          # scale risk by ATR regime


# Pre-defined scenarios (ordered for display)
SCENARIOS: Dict[str, ScenarioConfig] = {
    # ── Original scenarios ───────────────────────────────────────────────────
    "Baseline":       ScenarioConfig("Baseline"),
    "Regime filter":  ScenarioConfig("Regime filter",   regime_filter=True),
    "Strong (≥±18)":  ScenarioConfig("Strong (≥±18)",  long_threshold=18.0,
                                      short_threshold=-18.0),
    "Wide SL (2.5×)": ScenarioConfig("Wide SL (2.5×)", atr_sl=2.5),
    "Session 08-21":  ScenarioConfig("Session 08-21",
                                      session_hours=(8, 21)),
    "Monthly filter": ScenarioConfig("Monthly filter",  monthly_filter=True),
    "Combined":       ScenarioConfig("Combined",
                                      long_threshold=18.0, short_threshold=-18.0,
                                      regime_filter=True, session_hours=(8, 21),
                                      monthly_filter=True, atr_sl=2.5),
    # ── New strategies ───────────────────────────────────────────────────────
    "ADX Filter":     ScenarioConfig("ADX Filter",
                                      adx_threshold=25.0,
                                      session_hours=(8, 21)),
    "OBV Enhanced":   ScenarioConfig("OBV Enhanced",
                                      use_obv=True,
                                      session_hours=(8, 21)),
    "DD Control":     ScenarioConfig("DD Control",
                                      session_hours=(8, 21),
                                      regime_filter=True,
                                      dd_pause_pct=0.15),
    "Adaptive Vol":   ScenarioConfig("Adaptive Vol",
                                      session_hours=(8, 21),
                                      adaptive_vol=True),
    "Trend Quality":  ScenarioConfig("Trend Quality",
                                      adx_threshold=25.0,
                                      session_hours=(8, 21),
                                      regime_filter=True,
                                      use_obv=True),
    "Ultra Select":   ScenarioConfig("Ultra Select",
                                      long_threshold=30.0,
                                      short_threshold=-30.0,
                                      session_hours=(8, 21),
                                      regime_filter=True,
                                      adx_threshold=20.0),
}

# Months where BTC historically performs strongly (avoid counter-seasonal shorts)
BULL_MONTHS   = {1, 3, 4, 7, 10, 11}    # Jan, Mar, Apr, Jul, Oct, Nov
BEAR_MONTHS   = {6, 8}                   # Jun, Aug


# ─────────────────────────────────────────────────────────────────────────────
# Filter application
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(signals: pd.DataFrame,
                  cfg: ScenarioConfig) -> pd.DataFrame:
    """
    Return a copy of *signals* with the signal column filtered by *cfg*.
    All other columns (component scores, composite) are preserved unchanged
    for analytics; only 'signal' is zeroed out when a filter blocks the trade.
    """
    out = signals.copy()

    # ── 1. Working composite (optionally boosted by OBV signal) ──────────────
    # OBV adds ±1 to composite — raises effective threshold by 1 point for
    # entries that lack institutional accumulation/distribution confirmation.
    composite = out["composite"].copy()
    if cfg.use_obv and "s_obv" in out.columns:
        composite = composite + out["s_obv"] * 1

    # ── 2. Threshold filter ──────────────────────────────────────────────────
    sig_l = (composite >= cfg.long_threshold).astype(int)
    sig_s = -(composite <= cfg.short_threshold).astype(int)
    sig   = (sig_l + sig_s).clip(-1, 1)

    # ── 3. Regime filter ────────────────────────────────────────────────────
    if cfg.regime_filter and "regime" in out.columns:
        bull_mask = out["regime"] == "bull"
        bear_mask = out["regime"] == "bear"
        sig[bear_mask & (sig == 1)]  = 0
        sig[bull_mask & (sig == -1)] = 0

    # ── 4. ADX trend quality filter ──────────────────────────────────────────
    # Block entries in low-momentum / choppy market regimes.
    # Uses DI+ vs DI- direction alignment in addition to ADX threshold.
    if cfg.adx_threshold > 0 and "adx_14" in out.columns:
        weak_trend = out["adx_14"] < cfg.adx_threshold
        sig[weak_trend] = 0
        # DI directional alignment: longs need DI+ > DI-, shorts need DI- > DI+
        if "di_plus" in out.columns and "di_minus" in out.columns:
            di_misalign_long  = (sig == 1)  & (out["di_plus"]  < out["di_minus"])
            di_misalign_short = (sig == -1) & (out["di_minus"] < out["di_plus"])
            sig[di_misalign_long | di_misalign_short] = 0

    # ── 5. Session filter ────────────────────────────────────────────────────
    if cfg.session_hours is not None:
        h_s, h_e = cfg.session_hours
        not_session = ~((out.index.hour >= h_s) & (out.index.hour < h_e))
        sig[not_session] = 0

    # ── 6. Monthly filter ────────────────────────────────────────────────────
    if cfg.monthly_filter:
        month = out.index.month
        sig[(sig == -1) & np.isin(month, list(BULL_MONTHS))] = 0
        sig[(sig ==  1) & np.isin(month, list(BEAR_MONTHS))] = 0

    out["signal"] = sig
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(df_1h: pd.DataFrame,
                   signals: pd.DataFrame,
                   initial_capital: float = INIT_CAP) -> pd.DataFrame:
    """
    Run all scenarios, collect KPIs, and return a comparison DataFrame.
    Rows = scenarios, columns = KPI metrics.
    """
    rows = []
    results_store: Dict[str, dict] = {}

    for name, cfg in SCENARIOS.items():
        filt = apply_filters(signals, cfg)
        bt   = run_backtest(df_1h, filt, initial_capital,
                            atr_sl_override=cfg.atr_sl,
                            dd_pause_pct=cfg.dd_pause_pct,
                            adaptive_vol=cfg.adaptive_vol)
        kpis = bt["kpis"]
        kpis["equity"]  = bt["equity"]
        kpis["trades"]  = bt["trades"]
        kpis["drawdown"] = bt["drawdown"]
        results_store[name] = bt

        n_sig = int((filt["signal"] != 0).sum())
        rows.append({
            "Scenario":       name,
            "# Signals":      n_sig,
            "# Trades":       kpis.get("n_trades", 0),
            "Win Rate (%)":   round(kpis.get("win_rate", 0) * 100, 1),
            "Total Ret (%)":  round(kpis.get("total_return", 0) * 100, 2),
            "Sharpe":         round(kpis.get("sharpe", 0), 3),
            "Sortino":        round(kpis.get("sortino", 0), 3),
            "Max DD (%)":     round(kpis.get("max_drawdown", 0) * 100, 2),
            "Profit Factor":  round(kpis.get("profit_factor", 0), 2),
            "Expectancy ($)": round(kpis.get("expectancy", 0), 0),
            "Final Equity ($)": round(kpis.get("final_equity", 0), 0),
        })

    comparison = pd.DataFrame(rows).set_index("Scenario")
    return comparison, results_store
