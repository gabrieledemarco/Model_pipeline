#!/usr/bin/env python3
"""
BTCUSDT Live Trader — Bybit testnet / Bitget simulated trading
═══════════════════════════════════════════════════════════════════════════════

Runs the Strong (≥±18) strategy on exchange testnet.

Setup (Bybit testnet)
──────────────────────
   export BYBIT_TESTNET_API_KEY=<key>
   export BYBIT_TESTNET_API_SECRET=<secret>

Setup (Bitget simulated trading)
─────────────────────────────────
   export BITGET_API_KEY=bg_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   export BITGET_API_SECRET=<hex_secret>
   export BITGET_PASSPHRASE=<passphrase>

Usage
─────
   # Bybit testnet (default):
   python live_trader.py --exchange bybit

   # Bitget simulated trading:
   python live_trader.py --exchange bitget

   # Dry run — segnali senza ordini:
   python live_trader.py --dry-run

   # Capitale fisso override:
   python live_trader.py --capital 10000

   # Scenario alternativo:
   python live_trader.py --scenario "Ultra Select"

   # Strategy id esplicito (per lanciare più strategie in parallelo):
   python live_trader.py --scenario "Strong (≥±18)" --exchange bybit --strategy-id strong_18_bybit

   # Wyckoff Spring/Upthrust (1H, stessa scaletta SL/TP ad ATR):
   python live_trader.py --strategy-type wyckoff --exchange bybit

   # ICT Silver Bullet NY AM+PM (15M, TP/SL su FVG + time-stop 8h):
   python live_trader.py --strategy-type ict --exchange bybit

Ogni processo scrive esclusivamente in logs/strategies/{strategy_id}/
(live_trader.log, trades.csv, position.json, meta.json). Se --strategy-id
non è passato viene derivato automaticamente da scenario/strategy-type + exchange.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.live.bitget_client  import BitgetClient
from src.live.bybit_client   import BybitClient
from src.live.trader         import LiveTrader
from src.live import position_state as ps
from src.live.strategy_id    import slugify_strategy_id

STRATEGIES_ROOT = ROOT / "logs" / "strategies"


def _setup_logging(log_dir: Path, level: str = "INFO"):
    fmt = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"
    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "live_trader.log", encoding="utf-8"),
        ],
    )


def _build_signal_source(strategy_type: str, scenario: str):
    """Return (label, signal_fn, ws_interval) for the requested strategy type.

    signal_fn : () -> (signal, composite, atr, close, exit_plan, diagnostics)
    — see LiveTrader's docstring for the exit_plan/diagnostics contract.
    Imports are local so a missing/broken strategy module only breaks the
    strategy types that actually need it, not the whole CLI.
    """
    if strategy_type == "composite":
        from src.live.data_live import compute_live_signal
        from src.strategy.optimizer import SCENARIOS

        cfg = SCENARIOS[scenario]

        def signal_fn():
            signal, composite, atr, close = compute_live_signal(scenario=scenario)
            if signal > 0:
                reason = f"Composite={composite:+.1f} ≥ long threshold {cfg.long_threshold:+.1f} — LONG"
            elif signal < 0:
                reason = f"Composite={composite:+.1f} ≤ short threshold {cfg.short_threshold:+.1f} — SHORT"
            else:
                reason = (f"Composite={composite:+.1f} within [{cfg.short_threshold:+.1f}, "
                          f"{cfg.long_threshold:+.1f}] — no entry")
            diagnostics = {
                "reason": reason,
                "metrics": {
                    "composite": composite,
                    "long_threshold": cfg.long_threshold,
                    "short_threshold": cfg.short_threshold,
                },
            }
            return signal, composite, atr, close, None, diagnostics

        return scenario, signal_fn, "1h"

    if strategy_type == "wyckoff":
        from src.live.wyckoff_live import compute_wyckoff_signal

        def signal_fn():
            signal, composite, atr, close, diagnostics = compute_wyckoff_signal()
            return signal, composite, atr, close, None, diagnostics

        return "Wyckoff Spring/Upthrust", signal_fn, "1h"

    if strategy_type == "ict":
        from src.live.ict_live import compute_ict_signal
        return "ICT Silver Bullet NY AM+PM", compute_ict_signal, "15m"

    raise ValueError(f"Unknown strategy_type: {strategy_type!r}")


def _write_meta(log_dir: Path, strategy_id: str, scenario: str, exchange: str,
                 dry_run: bool, capital: float | None) -> None:
    meta = {
        "strategy_id": strategy_id,
        "scenario":    scenario,
        "exchange":    exchange,
        "dry_run":     dry_run,
        "capital":     capital,
        "pid":         os.getpid(),
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "host":        socket.gethostname(),
    }
    (log_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="BTCUSDT Live Trader — Bitget Simulated Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--exchange", default="bybit",
                    choices=["bybit", "bitget"],
                    help="Exchange testnet da usare (default: bybit)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Calcola segnali e logga senza inviare ordini")
    ap.add_argument("--capital", type=float, default=None,
                    help="Override equity in USDT (default: fetched dal conto)")
    ap.add_argument("--scenario", default="Strong (≥±18)",
                    help="Scenario strategia (solo per --strategy-type composite; "
                         "default: 'Strong (≥±18)')")
    ap.add_argument("--strategy-type", default="composite",
                    choices=["composite", "wyckoff", "ict"],
                    help="Fonte del segnale: 'composite' (score multi-TF, "
                         "parametrizzato da --scenario), 'wyckoff' (Spring/"
                         "Upthrust 1H), 'ict' (Silver Bullet NY AM+PM 15M). "
                         "Default: composite")
    ap.add_argument("--strategy-id", default=None,
                    help="Identificatore univoco della strategia (default: "
                         "slug auto-derivato da scenario+exchange, es. "
                         "'strong_18_bybit'). Determina la sottodirectory "
                         "logs/strategies/{strategy_id}/ usata per log, "
                         "trades.csv, position.json e meta.json.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()


async def _main(args: argparse.Namespace, log_dir: Path, strategy_id: str, label: str,
                 signal_fn, ws_interval: str):
    log = logging.getLogger("live_trader")

    if args.exchange == "bybit":
        api_key    = os.environ.get("BYBIT_TESTNET_API_KEY",    "")
        api_secret = os.environ.get("BYBIT_TESTNET_API_SECRET", "")
        if not args.dry_run and not (api_key and api_secret):
            log.error(
                "Credenziali Bybit mancanti.\n"
                "Imposta BYBIT_TESTNET_API_KEY e BYBIT_TESTNET_API_SECRET\n"
                "oppure usa --dry-run per girare senza ordini."
            )
            sys.exit(1)
        client = BybitClient(api_key, api_secret)
    else:
        api_key    = os.environ.get("BITGET_API_KEY",    "")
        api_secret = os.environ.get("BITGET_API_SECRET", "")
        passphrase = os.environ.get("BITGET_PASSPHRASE", "")
        if not args.dry_run and not (api_key and api_secret and passphrase):
            log.error(
                "Credenziali Bitget mancanti.\n"
                "Imposta BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE\n"
                "oppure usa --dry-run per girare senza ordini."
            )
            sys.exit(1)
        client = BitgetClient(api_key, api_secret, passphrase)

    ps.configure(
        state_file  = log_dir / "position.json",
        strategy_id = strategy_id,
        scenario    = label,
        exchange    = args.exchange,
    )
    _write_meta(
        log_dir     = log_dir,
        strategy_id = strategy_id,
        scenario    = label,
        exchange    = args.exchange,
        dry_run     = args.dry_run,
        capital     = args.capital,
    )

    trader = LiveTrader(
        client           = client,
        dry_run          = args.dry_run,
        scenario         = label,
        capital_override = args.capital,
        log_dir          = log_dir,
        strategy_id      = strategy_id,
        signal_fn        = signal_fn,
        ws_interval      = ws_interval,
    )

    loop = asyncio.get_running_loop()

    def _shutdown(signum, frame):
        log.info("Signal %s — shutting down …", signal.Signals(signum).name)
        trader.stop()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("═" * 62)
    log.info("  BTCUSDT Live Trader  |  Strategy: %s", label)
    log.info("  Strategy ID: %s", strategy_id)
    exchange_label = {
        "bybit":  "Bybit USDT-Perps (Testnet)",
        "bitget": "Bitget USDT-Futures (Simulated Trading)",
    }[args.exchange]
    log.info("  Exchange:  %s", exchange_label)
    log.info("  Signals:   Binance FAPI production data")
    log.info("  Mode:      %s", "DRY RUN" if args.dry_run else "LIVE TESTNET")
    if args.capital:
        log.info("  Capital:   $%.2f (override)", args.capital)
    log.info("═" * 62)

    try:
        await trader.run()
    except asyncio.CancelledError:
        log.info("Exiting cleanly.")


def _refuse_if_already_running(log_dir: Path, strategy_id: str) -> None:
    """Abort startup if a live process for this strategy_id is still alive.

    Two concurrent processes writing the same logs/strategies/{id}/ files
    (trades.csv, position.json) with no locking would silently interleave
    or clobber each other's state.
    """
    meta_path = log_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        old_pid = json.loads(meta_path.read_text()).get("pid")
    except Exception:
        return
    if old_pid is None:
        return
    try:
        os.kill(int(old_pid), 0)
    except (OSError, ValueError):
        return  # old_pid not alive (or malformed) — safe to proceed
    print(
        f"ERROR: strategy_id '{strategy_id}' already has a running process "
        f"(pid {old_pid}). Stop it first or pass a different --strategy-id.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    args = parse_args()

    label, signal_fn, ws_interval = _build_signal_source(args.strategy_type, args.scenario)

    strategy_id = args.strategy_id or slugify_strategy_id(label, args.exchange)
    log_dir = STRATEGIES_ROOT / strategy_id
    log_dir.mkdir(parents=True, exist_ok=True)

    _refuse_if_already_running(log_dir, strategy_id)
    _setup_logging(log_dir, args.log_level)
    asyncio.run(_main(args, log_dir, strategy_id, label, signal_fn, ws_interval))


if __name__ == "__main__":
    main()
