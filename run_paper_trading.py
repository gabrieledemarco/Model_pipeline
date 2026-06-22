"""
run_paper_trading.py — Start the live paper-trading engine.

Usage
─────
  python run_paper_trading.py [--strategies S1,S2,...] [--capital N] [--verbose]

Strategies available
─────────────────────
  max_return    Extended signals, no risk controls  (IS +298%, OOS +191%)
  min_dd        Notional 10% + Vol 15% + CB 15%    (IS DD -21.8%)
  balanced      Notional 20% + Vol 20% + CB 15%    (best Calmar, OOS DD -17%)
  conservative  balanced + min_score ≥ 8            (fewest trades, tightest risk)

Examples
────────
  python run_paper_trading.py
  python run_paper_trading.py --strategies balanced,min_dd --capital 10000
  python run_paper_trading.py --strategies max_return --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.live.paper_trader import PaperTrader
from src.live.runner       import LiveRunner


# ─── Strategy catalogue ───────────────────────────────────────────────────────
# Tuned from the backtest / WFO comparison study (see reports/comparison_report.html).
STRATEGY_CATALOGUE: dict[str, dict] = {
    "max_return": {
        "label":            "Max Return (no controls)",
        "max_notional_pct": 1.00,
        "vol_target":       None,
        "dd_halt_pct":      None,
        "min_score":        None,
    },
    "min_dd": {
        "label":            "Min Drawdown (Cap10+Vol15+CB15)",
        "max_notional_pct": 0.10,
        "vol_target":       0.15,
        "dd_halt_pct":      0.15,
        "min_score":        None,
    },
    "balanced": {
        "label":            "Balanced (Cap20+Vol20+CB15)",
        "max_notional_pct": 0.20,
        "vol_target":       0.20,
        "dd_halt_pct":      0.15,
        "min_score":        None,
    },
    "conservative": {
        "label":            "Conservative (Cap20+Vol20+CB15+Score≥8)",
        "max_notional_pct": 0.20,
        "vol_target":       0.20,
        "dd_halt_pct":      0.15,
        "min_score":        8.0,
    },
}

DEFAULT_STRATEGIES = list(STRATEGY_CATALOGUE.keys())


def _build_traders(
    strategy_keys: list[str],
    initial_capital: float,
) -> dict[str, PaperTrader]:
    traders: dict[str, PaperTrader] = {}
    for key in strategy_keys:
        if key not in STRATEGY_CATALOGUE:
            print(f"  [warn] Unknown strategy '{key}' — skipping")
            continue
        cfg = STRATEGY_CATALOGUE[key]
        traders[key] = PaperTrader(
            label            = cfg["label"],
            initial_capital  = initial_capital,
            max_notional_pct = cfg["max_notional_pct"],
            vol_target       = cfg["vol_target"],
            dd_halt_pct      = cfg["dd_halt_pct"],
            min_score        = cfg["min_score"],
        )
    return traders


async def main(
    strategy_keys:   list[str],
    initial_capital: float,
    verbose:         bool,
) -> None:
    # ── Print config ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  BTCUSDT Live Paper Trading")
    print(f"  Scenario : Session 08-21 (08:00–21:00 UTC)")
    print(f"  Capital  : {initial_capital:,.0f} USDT per strategy")
    print(f"  Strategies: {', '.join(strategy_keys)}")
    print("=" * 70)

    for key in strategy_keys:
        cfg = STRATEGY_CATALOGUE.get(key, {})
        print(f"  [{key}] {cfg.get('label', '?')}")
        print(f"         Cap={cfg.get('max_notional_pct',1)*100:.0f}%  "
              f"Vol={cfg.get('vol_target') or 'off'}  "
              f"CB={cfg.get('dd_halt_pct') or 'off'}  "
              f"Score≥{cfg.get('min_score') or 'off'}")

    # ── Build traders ─────────────────────────────────────────────────────────
    traders = _build_traders(strategy_keys, initial_capital)
    if not traders:
        print("No valid strategies selected. Exiting.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = LiveRunner(traders, verbose=verbose)
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

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    keys = [k.strip() for k in args.strategies.split(",") if k.strip()]

    print("\nPress Ctrl+C to stop.\n")
    try:
        asyncio.run(main(keys, args.capital, args.verbose))
    except KeyboardInterrupt:
        print("\nStopped by user.")
