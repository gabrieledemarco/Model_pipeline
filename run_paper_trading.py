"""
run_paper_trading.py — Start the live paper-trading engine.

Usage
─────
  python run_paper_trading.py [--strategies S1,S2,...] [--capital N] [--verbose]

Enhancement variants (default — from create_enhancements_report.py, WF 2020-2026)
────────────────────────────────────────────────────────────────────────────────
  baseline    Fixed 1% risk/trade, threshold=3.0, session 08-21
              Calmar=0.830  Max DD=37.5%  Return=+31.1%

  vol         Vol-scaled sizing targeting 20% annual vol
              Calmar=1.473  Max DD=53.6%  Return=+79.0%

  vol_ddhalt  Vol scaling + circuit breaker at 15% DD (size ×0.5, pause at ×1.5)
              Calmar=0.748  Max DD=40.9%  Return=+30.6%

  vol_ma      Vol scaling + daily EMA-200 regime filter
              Calmar=0.743  Max DD=54.2%  Return=+40.3%

Legacy variants (kept for backward compatibility)
─────────────────────────────────────────────────
  max_return  balanced  min_dd  conservative

Examples
────────
  python run_paper_trading.py
  python run_paper_trading.py --strategies vol,vol_ddhalt --capital 10000
  python run_paper_trading.py --strategies baseline,vol --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.live.paper_trader  import PaperTrader
from src.live.runner        import LiveRunner
from src.live.monitor       import start_monitor
from src.strategy.optimizer import ScenarioConfig

# ── Shared scenario config (matching create_enhancements_report.py baseline) ──
_ENHANCEMENT_SCENARIO = ScenarioConfig(
    name            = "Session 08-21 (T=3)",
    session_hours   = (8, 21),
    long_threshold  = 3.0,
    short_threshold = -3.0,
)

# ── Daily EMA-200 regime signal filter ───────────────────────────────────────

def _ma_regime_filter(signal: int, composite: float,
                      df_1h, df_1d) -> tuple[int, float]:
    """Suppress longs below daily EMA(200) and shorts above it."""
    if df_1d is None or df_1d.empty or "ema_200" not in df_1d.columns:
        return signal, composite
    ema200 = float(df_1d["ema_200"].iloc[-1])
    close  = float(df_1h["close"].iloc[-1])
    if signal == -1 and close > ema200:
        return 0, composite
    if signal ==  1 and close < ema200:
        return 0, composite
    return signal, composite


# ── Strategy catalogue ────────────────────────────────────────────────────────

STRATEGY_CATALOGUE: dict[str, dict] = {
    # ── Enhancement variants (primary) ──────────────────────────────────────
    "baseline": {
        "label":         "Baseline (T=3, fixed risk)",
        "vol_target":    None,
        "dd_halt_pct":   None,
        "scenario_cfg":  _ENHANCEMENT_SCENARIO,
        "signal_filter": None,
        "backtest":      "Calmar=0.830  DD=37.5%  ret=+31.1%",
    },
    "vol": {
        "label":         "Vol Sizing (vol_target=0.20)",
        "vol_target":    0.20,
        "dd_halt_pct":   None,
        "scenario_cfg":  _ENHANCEMENT_SCENARIO,
        "signal_filter": None,
        "backtest":      "Calmar=1.473  DD=53.6%  ret=+79.0%",
    },
    "vol_ddhalt": {
        "label":         "Vol + DD Halt 15%",
        "vol_target":    0.20,
        "dd_halt_pct":   0.15,
        "scenario_cfg":  _ENHANCEMENT_SCENARIO,
        "signal_filter": None,
        "backtest":      "Calmar=0.748  DD=40.9%  ret=+30.6%",
    },
    "vol_ma": {
        "label":         "Vol + MA Filter (EMA-200d)",
        "vol_target":    0.20,
        "dd_halt_pct":   None,
        "scenario_cfg":  _ENHANCEMENT_SCENARIO,
        "signal_filter": _ma_regime_filter,
        "backtest":      "Calmar=0.743  DD=54.2%  ret=+40.3%",
    },
    # ── Legacy variants ──────────────────────────────────────────────────────
    "max_return": {
        "label":         "Max Return (no controls)",
        "vol_target":    None,
        "dd_halt_pct":   None,
        "scenario_cfg":  None,
        "signal_filter": None,
        "backtest":      "legacy",
    },
    "min_dd": {
        "label":         "Min Drawdown (Vol15+CB15)",
        "vol_target":    0.15,
        "dd_halt_pct":   0.15,
        "scenario_cfg":  None,
        "signal_filter": None,
        "backtest":      "legacy",
    },
    "balanced": {
        "label":         "Balanced (Vol20+CB15)",
        "vol_target":    0.20,
        "dd_halt_pct":   0.15,
        "scenario_cfg":  None,
        "signal_filter": None,
        "backtest":      "legacy",
    },
    "conservative": {
        "label":         "Conservative (Vol20+CB15+Score≥8)",
        "vol_target":    0.20,
        "dd_halt_pct":   0.15,
        "scenario_cfg":  None,
        "signal_filter": None,
        "backtest":      "legacy",
    },
}

# Default: the 4 enhancement variants
DEFAULT_STRATEGIES = ["baseline", "vol", "vol_ddhalt", "vol_ma"]


def _build_runners(
    strategy_keys:   list[str],
    initial_capital: float,
) -> tuple[dict[str, PaperTrader], dict[str, callable], ScenarioConfig | None]:
    """
    Returns (traders, signal_filters, scenario_cfg).
    All selected variants must share a scenario_cfg; we use the first one found.
    """
    traders:        dict[str, PaperTrader] = {}
    signal_filters: dict[str, callable]   = {}
    scenario_cfg = None

    for key in strategy_keys:
        if key not in STRATEGY_CATALOGUE:
            print(f"  [warn] Unknown strategy '{key}' — skipping")
            continue
        cfg = STRATEGY_CATALOGUE[key]

        traders[key] = PaperTrader(
            label           = cfg["label"],
            initial_capital = initial_capital,
            vol_target      = cfg["vol_target"],
            dd_halt_pct     = cfg["dd_halt_pct"],
        )

        if cfg["signal_filter"] is not None:
            signal_filters[key] = cfg["signal_filter"]

        if cfg["scenario_cfg"] is not None and scenario_cfg is None:
            scenario_cfg = cfg["scenario_cfg"]

    return traders, signal_filters, scenario_cfg


async def main(
    strategy_keys:   list[str],
    initial_capital: float,
    verbose:         bool,
) -> None:
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  BTCUSDT Live Paper Trading — Enhancement Variants              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n  Capital : {initial_capital:,.0f} USDT per variant")
    print(f"  Variants: {', '.join(strategy_keys)}\n")

    for key in strategy_keys:
        cfg = STRATEGY_CATALOGUE.get(key, {})
        bt  = cfg.get("backtest", "")
        print(f"  [{key:<12}] {cfg.get('label','?')}")
        if bt and bt != "legacy":
            print(f"               WF backtest: {bt}")

    print()

    traders, signal_filters, scenario_cfg = _build_runners(strategy_keys, initial_capital)
    if not traders:
        print("No valid strategies selected. Exiting.")
        return

    runner = LiveRunner(
        traders        = traders,
        signal_filters = signal_filters or None,
        scenario_cfg   = scenario_cfg,
        verbose        = verbose,
    )

    # ── Start HTTP monitor (Render exposes $PORT; local default 8080) ─────────
    port = int(os.environ.get("PORT", 8080))
    await start_monitor(traders, runner.last_signal, port=port)

    await runner.bootstrap()
    await runner.run()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BTCUSDT live paper-trading engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--strategies", "-s",
        default=",".join(DEFAULT_STRATEGIES),
        help=f"Comma-separated list of strategies (default: all). "
             f"Available: {', '.join(STRATEGY_CATALOGUE)}",
    )
    p.add_argument(
        "--capital", "-c",
        type=float,
        default=100_000.0,
        help="Initial paper capital in USDT per strategy (default: 100000)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every 15m and 1m bar close (adds noise)",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: WARNING)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Env-var overrides (used by Render — set in dashboard → Environment)
    strategies_env = os.environ.get("PT_STRATEGIES", "").strip()
    capital_env    = os.environ.get("PT_CAPITAL",    "").strip()
    log_level_env  = os.environ.get("PT_LOG_LEVEL",  "").strip()

    raw_keys  = strategies_env if strategies_env else args.strategies
    capital   = float(capital_env) if capital_env else args.capital
    log_level = log_level_env if log_level_env else args.log_level

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    print("\nPress Ctrl+C to stop.\n")
    try:
        asyncio.run(main(keys, capital, args.verbose))
    except KeyboardInterrupt:
        print("\nStopped by user.")
