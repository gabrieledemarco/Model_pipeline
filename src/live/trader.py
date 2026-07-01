"""
Live trader — event loop generico per strategie BTCUSDT.

Il motore (entry/exit/sizing/position-state) è strategia-agnostico: la fonte
del segnale è iniettata tramite `signal_fn` (composite-score, Wyckoff,
ICT Silver Bullet, ...), la cadenza tramite `ws_interval` (1H per
composite/Wyckoff, 15M per ICT).

Architettura
────────────
  WebSocket  wss://fstream.binance.com/ws/btcusdt@kline_{ws_interval}
       │
       └─► on bar CLOSE ─► signal_fn() ─► entry / flat logic

  Background monitor (ogni POLL_SECS):
       └─► BitgetClient.get_mark_price() ─► check SL / TP / time-stop ─► gestisce posizione

Esecuzione ordini: Bitget USDT-Futures (simulated trading / testnet)
Dati di mercato:   Binance FAPI production (no key — stessa pipeline del backtest)
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import websockets

from src.live.bitget_client   import BitgetClient, SYMBOL
from src.live.data_live       import compute_live_signal
from src.live import position_state as ps

# ── Costanti (specchio di engine.py) ─────────────────────────────────────────
ATR_SL   = 2.0
ATR_TP1  = 2.0
ATR_TP2  = 4.0
ATR_TP3  = 6.0
RISK_PCT = 0.01    # 1 % equity per trade
FEE      = 0.0006  # 0.06 % taker Bitget USDT-Futures

WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/btcusdt@kline_{interval}"
POLL_SECS = 30
HEARTBEAT_EVERY_N_POLLS = 10   # ~5 min at POLL_SECS=30 — keeps live_trader.log
                               # from going silent for up to an hour between
                               # candle closes so the dashboard's stall
                               # detector has a reliable, tight signal.

# If neither the WS loop nor the monitor loop have made progress in this many
# seconds, the asyncio event loop is assumed to be stuck (e.g. a synchronous
# network call that outlived its own timeout, or a silently-dead WebSocket
# the library's ping/pong keepalive failed to detect). Comfortably longer
# than the worst-case legitimate blocking window (a handful of sequential
# 10-30s HTTP calls), short enough to recover quickly under a supervisor
# (systemd Restart=always) that reloads position.json on relaunch.
WATCHDOG_TIMEOUT_SECS = 360

# Default location (single-strategy fallback). LiveTrader is normally
# constructed with an explicit log_dir so each concurrent strategy instance
# writes to its own logs/strategies/{strategy_id}/ directory.
LOG_DIR = Path(__file__).parent.parent.parent / "logs"

log = logging.getLogger("live.trader")


# ── Trade logger ──────────────────────────────────────────────────────────────

class TradeLog:
    FIELDS = [
        "strategy_id",
        "timestamp", "event", "direction", "price", "qty",
        "sl", "tp1", "tp2", "tp3", "composite", "atr",
        "pnl_net", "equity", "note",
    ]

    def __init__(self, path: Path, strategy_id: str = ""):
        self.path        = path
        self.strategy_id = strategy_id
        self._is_new     = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **kwargs) -> None:
        row = {f: kwargs.get(f, "") for f in self.FIELDS}
        row["timestamp"]   = datetime.now(timezone.utc).isoformat()
        row["strategy_id"] = self.strategy_id
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if self._is_new:
                w.writeheader()
                self._is_new = False
            w.writerow(row)
        printable = {k: v for k, v in row.items() if v != ""}
        log.info("[LOG] %s", json.dumps(printable))


# ── Position sizing (specchio di engine.py) ───────────────────────────────────

def _size(equity: float, atr: float, entry_price: float) -> float:
    risk_usd      = equity * RISK_PCT
    risk_per_unit = atr * ATR_SL
    qty           = risk_usd / risk_per_unit
    max_qty       = equity * 0.95 / entry_price
    return max(round(min(qty, max_qty), 3), 0.001)


def _size_from_stop(
    equity: float, entry_price: float, sl_price: float, atr: float,
    min_sl_atr_floor: float = 0.0, max_leverage: float = 1.0,
) -> float:
    """Generic risk-based sizing from an explicit stop price, for strategies
    whose SL isn't a fixed ATR multiple (e.g. ICT Silver Bullet's SL sits on
    the FVG boundary). Mirrors the backtest sizing exactly:
      sl_dist = max(|entry - sl|, atr × min_sl_atr_floor)   [prevents a very
                 tight structural stop from blowing up position size]
      qty     = min(risk_usd / sl_dist, equity × max_leverage / entry)
    """
    risk_usd = equity * RISK_PCT
    sl_dist  = max(abs(entry_price - sl_price), atr * min_sl_atr_floor)
    if sl_dist <= 0:
        return 0.001
    qty     = risk_usd / sl_dist
    max_qty = equity * max_leverage / entry_price
    return max(round(min(qty, max_qty), 3), 0.001)


def _levels(direction: int, entry: float, atr: float):
    if direction == 1:
        return (entry - atr * ATR_SL,
                entry + atr * ATR_TP1,
                entry + atr * ATR_TP2,
                entry + atr * ATR_TP3)
    else:
        return (entry + atr * ATR_SL,
                entry - atr * ATR_TP1,
                entry - atr * ATR_TP2,
                entry - atr * ATR_TP3)


# ── Core trader ───────────────────────────────────────────────────────────────

class LiveTrader:
    def __init__(
        self,
        client: BitgetClient,
        dry_run: bool = False,
        scenario: str = "Strong (≥±18)",
        capital_override: Optional[float] = None,
        log_dir: Optional[Path] = None,
        strategy_id: str = "",
        signal_fn: Optional[Callable[[], tuple]] = None,
        ws_interval: str = "1h",
    ):
        """
        signal_fn : () -> (signal, composite, atr, close, exit_plan, diagnostics)
            Pluggable signal source, decoupling the trading engine from any
            specific strategy. `exit_plan` is either None (use the standard
            2/2/4/6×ATR ladder — composite-score and Wyckoff strategies) or a
            dict {"sl": float, "tp": float, "min_sl_atr_floor": float,
            "max_leverage": float, "time_stop_hours": float} for strategies
            with their own validated single-TP/SL + time-stop exit rule
            (e.g. ICT Silver Bullet). `diagnostics` is either None or a dict
            {"reason": str, "metrics": {...}} describing the conditions that
            led to the decision — persisted to logs/strategies/{id}/
            analysis.json each bar close for the dashboard to display;
            purely informational, the engine never reads its contents.
            Defaults to the original composite-score signal (no diagnostics)
            for backward compatibility when not provided.
        ws_interval : Binance kline stream interval driving on_candle_close()
            cadence — "1h" for composite/Wyckoff, "15m" for ICT Silver Bullet.
        """
        self.client   = client
        self.dry_run  = dry_run
        self.scenario = scenario
        self.cap_override = capital_override
        self.strategy_id  = strategy_id
        self.ws_interval  = ws_interval
        self.signal_fn = signal_fn or (
            lambda: (*compute_live_signal(scenario=self.scenario), None, None)
        )

        log_dir = Path(log_dir) if log_dir is not None else LOG_DIR
        self.log_dir    = log_dir
        self.trade_log = TradeLog(log_dir / "trades.csv", strategy_id=strategy_id)
        self.state     = ps.load()
        self._running  = True
        self._last_heartbeat = time.time()

    def _touch(self) -> None:
        """Record forward progress. Read by the watchdog thread — see run()."""
        self._last_heartbeat = time.time()

    def _watchdog(self) -> None:
        """Runs in its own OS thread, not on the asyncio event loop.

        A stuck event loop (e.g. blocked inside a synchronous call, or a
        WebSocket the library's keepalive failed to notice is dead) would
        also block any asyncio-based watchdog task — this has to live
        outside the loop to be able to detect and react to that case.
        """
        while self._running:
            time.sleep(30)
            stale_for = time.time() - self._last_heartbeat
            if stale_for > WATCHDOG_TIMEOUT_SECS:
                log.critical(
                    "Watchdog: no progress in %.0fs (limit %ds) — event loop "
                    "appears stuck. Forcing process exit for supervisor restart.",
                    stale_for, WATCHDOG_TIMEOUT_SECS,
                )
                os._exit(1)  # bypass asyncio cleanup: the loop may be wedged

    # ── Equity helper ─────────────────────────────────────────────────────────

    def _equity(self) -> float:
        if self.cap_override:
            return self.cap_override
        if self.dry_run:
            return 10_000.0
        try:
            return self.client.get_available_balance()
        except Exception as e:
            log.warning("Balance fetch failed: %s", e)
            return 10_000.0

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _enter(self, direction: int, composite: float, atr: float, entry_price: float,
               exit_plan: Optional[dict] = None):
        equity = self._equity()

        if exit_plan is None:
            # Standard 3-tier ATR ladder (composite-score, Wyckoff).
            qty = _size(equity, atr, entry_price)
            sl, tp1, tp2, tp3 = _levels(direction, entry_price, atr)
            exit_mode = "ladder"
            time_stop_at = ""
        else:
            # Strategy-provided single TP/SL (e.g. ICT Silver Bullet).
            sl = exit_plan["sl"]
            tp1 = tp2 = tp3 = exit_plan["tp"]

            # Sanity-check the stop is actually on the risk side of entry.
            # A signal module bug that ever inverted ref_lo/ref_hi would
            # otherwise be applied silently — exiting at the wrong price
            # with no error raised. Refuse the trade instead.
            wrong_side = (direction == 1 and sl >= entry_price) or \
                         (direction == -1 and sl <= entry_price)
            if wrong_side:
                log.error(
                    "Refusing entry: exit_plan SL=%.2f is not on the risk "
                    "side of entry=%.2f for direction=%+d — signal module bug?",
                    sl, entry_price, direction,
                )
                return

            qty = _size_from_stop(
                equity, entry_price, sl, atr,
                min_sl_atr_floor=exit_plan.get("min_sl_atr_floor", 0.0),
                max_leverage=exit_plan.get("max_leverage", 1.0),
            )
            exit_mode = "single_tp"
            time_stop_hours = exit_plan.get("time_stop_hours")
            time_stop_at = (
                (datetime.now(timezone.utc) + timedelta(hours=time_stop_hours)).isoformat()
                if time_stop_hours else ""
            )

        dir_label = "LONG" if direction == 1 else "SHORT"
        log.info(
            "ENTRY %s  qty=%.3f BTC  @%.1f  SL=%.1f  TP1=%.1f  TP2=%.1f  TP3=%.1f  score=%.1f  exit_mode=%s",
            dir_label, qty, entry_price, sl, tp1, tp2, tp3, composite, exit_mode,
        )

        sl_order_id = None
        if not self.dry_run:
            try:
                self.client.place_market_entry(direction, qty)
                log.info("Market entry order placed")
                # Place native TPSL stop-loss
                sl_order_id = self.client.place_stop_loss(direction, sl, qty)
                if sl_order_id:
                    log.info("SL TPSL order placed: id=%s  triggerPrice=%.1f", sl_order_id, sl)
                else:
                    log.warning("SL order placement returned no orderId")
            except Exception as e:
                log.error("Entry order failed: %s", e)
                return

        self.state = ps.open_position(
            direction=direction, entry_price=entry_price, size=qty,
            sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            composite=composite, atr=atr, equity=equity,
            sl_order_id=sl_order_id if sl_order_id else None,
            exit_mode=exit_mode, time_stop_at=time_stop_at,
        )
        self.trade_log.write(
            event="ENTRY", direction=direction, price=entry_price,
            qty=qty, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            composite=composite, atr=atr, equity=equity,
        )

    # ── Full exit ─────────────────────────────────────────────────────────────

    def _exit_full(self, exit_price: float, reason: str):
        state = self.state
        if not state.active:
            return

        gross   = state.direction * state.size_remaining * (exit_price - state.entry_price)
        fees    = state.size_remaining * exit_price * FEE * 2
        net_pnl = gross - fees + state.realized_pnl

        log.info("EXIT [%s]  dir=%+d  qty=%.3f  @%.1f  net_pnl=%.2f USDT",
                 reason, state.direction, state.size_remaining, exit_price, net_pnl)

        if not self.dry_run:
            # Cancel any open SL TPSL orders first
            try:
                self.client.cancel_all_tpsl()
            except Exception:
                pass
            try:
                self.client.place_market_close(state.direction, state.size_remaining)
            except Exception as e:
                # The native SL/TP order placed at entry can trigger and
                # close the position on the exchange before our own
                # market-close request lands — the exchange then rejects
                # our reduce-only order because there's nothing left to
                # reduce (e.g. Bybit 110017 "current position is zero").
                # Confirm against the exchange rather than guessing from
                # the error text: if it's genuinely flat, finalize local
                # state to match (otherwise this repeats every poll
                # forever and the trade never gets recorded). If the
                # exchange still shows a real position, something else
                # went wrong — leave state untouched and retry next poll,
                # since forcing "closed" while a position is still open
                # risks a duplicate entry on the next signal.
                try:
                    still_open = abs(self.client.get_position_size()) > 1e-9
                except Exception:
                    still_open = True  # can't confirm — assume worst case
                if still_open:
                    log.error("Exit order failed: %s", e)
                    return
                log.warning(
                    "Exit order failed (%s) but exchange position is already "
                    "flat — native SL/TP likely triggered first. Finalizing "
                    "local state to match.", e,
                )

        self.trade_log.write(
            event=f"EXIT_{reason.upper()}",
            direction=state.direction, price=exit_price,
            qty=state.size_remaining, pnl_net=round(net_pnl, 2),
            composite=state.composite,
        )
        self.state = ps.close_position()

    # ── Partial close ─────────────────────────────────────────────────────────

    def _partial_close(self, frac: float, exit_price: float, label: str):
        state = self.state
        qty   = max(round(state.size_remaining * frac, 3), 0.001)

        gross = state.direction * qty * (exit_price - state.entry_price)
        fees  = qty * exit_price * FEE
        net   = gross - fees
        state.realized_pnl  += net
        state.size_remaining = round(state.size_remaining - qty, 3)

        log.info("PARTIAL [%s]  qty=%.3f  @%.1f  net=%.2f", label, qty, exit_price, net)

        if not self.dry_run:
            try:
                self.client.place_market_close(state.direction, qty)
            except Exception as e:
                log.error("Partial close failed: %s", e)

        self.trade_log.write(
            event=f"PARTIAL_{label.upper()}",
            direction=state.direction, price=exit_price,
            qty=qty, pnl_net=round(net, 2),
        )
        ps.update(state)

    # ── Move SL to break-even ─────────────────────────────────────────────────

    def _move_sl_to_be(self):
        state  = self.state
        new_sl = state.entry_price
        log.info("SL → break-even  %.1f", new_sl)

        if not self.dry_run:
            # Cancel old SL, place new one at BE
            if state.sl_order_id:
                self.client.cancel_tpsl_order(str(state.sl_order_id))
            new_id = self.client.place_stop_loss(
                state.direction, new_sl, state.size_remaining)
            state.sl_order_id = str(new_id) if new_id else None
            log.info("New SL TPSL placed at BE=%.1f  id=%s", new_sl, new_id)

        state.sl       = new_sl
        state.be_moved = True
        ps.update(state)

    # ── SL/TP monitor (chiamato ogni POLL_SECS) ───────────────────────────────

    def check_position(self) -> Optional[float]:
        """Returns the fetched mark price (for heartbeat logging), or None
        if flat / the fetch failed."""
        if not self.state.active:
            return None

        try:
            mark = self.client.get_mark_price()
        except Exception as e:
            log.warning("Mark price fetch failed: %s", e)
            return None

        state = self.state
        d     = state.direction

        log.debug("Monitor  mark=%.1f  sl=%.1f  tp1=%.1f  tp2=%.1f  tp3=%.1f",
                  mark, state.sl, state.tp1, state.tp2, state.tp3)

        # ── Time-stop (any exit_mode) ────────────────────────────────────────
        if state.time_stop_at:
            try:
                deadline = datetime.fromisoformat(state.time_stop_at)
            except ValueError:
                deadline = None
            if deadline is not None and datetime.now(timezone.utc) >= deadline:
                self._exit_full(mark, "time_stop")
                return mark

        # ── Single TP/SL (e.g. ICT Silver Bullet) ────────────────────────────
        if state.exit_mode == "single_tp":
            if d == 1:
                if mark <= state.sl:
                    self._exit_full(state.sl, "stop_loss")
                elif mark >= state.tp1:
                    self._exit_full(state.tp1, "take_profit")
            else:
                if mark >= state.sl:
                    self._exit_full(state.sl, "stop_loss")
                elif mark <= state.tp1:
                    self._exit_full(state.tp1, "take_profit")
            return mark

        # ── LONG (3-tier ATR ladder) ──────────────────────────────────────────
        if d == 1:
            if mark <= state.sl:
                self._exit_full(state.sl, "stop_loss"); return
            if mark >= state.tp3 and state.tp1_hit and state.tp2_hit:
                self._exit_full(state.tp3, "tp3"); return
            if mark >= state.tp2 and state.tp1_hit and not state.tp2_hit:
                self._partial_close(0.5, state.tp2, "tp2")
                state.tp2_hit = True; ps.update(state); return
            if mark >= state.tp1 and not state.tp1_hit:
                self._partial_close(0.5, state.tp1, "tp1")
                state.tp1_hit = True; ps.update(state)
                self._move_sl_to_be(); return

        # ── SHORT ─────────────────────────────────────────────────────────────
        elif d == -1:
            if mark >= state.sl:
                self._exit_full(state.sl, "stop_loss"); return
            if mark <= state.tp3 and state.tp1_hit and state.tp2_hit:
                self._exit_full(state.tp3, "tp3"); return
            if mark <= state.tp2 and state.tp1_hit and not state.tp2_hit:
                self._partial_close(0.5, state.tp2, "tp2")
                state.tp2_hit = True; ps.update(state); return
            if mark <= state.tp1 and not state.tp1_hit:
                self._partial_close(0.5, state.tp1, "tp1")
                state.tp1_hit = True; ps.update(state)
                self._move_sl_to_be(); return

        return mark

    # ── On bar chiusa (cadenza = self.ws_interval) ────────────────────────────

    def on_candle_close(self):
        log.info("── %s bar closed — computing signal ──", self.ws_interval)
        try:
            signal, composite, atr, close, exit_plan, diagnostics = self.signal_fn()
        except Exception as e:
            log.error("Signal computation failed: %s", e)
            return

        log.info("Signal=%+d  composite=%.2f  atr=%.1f  close=%.1f",
                 signal, composite, atr, close)

        self._write_analysis(diagnostics, signal, composite, atr, close)

        self.trade_log.write(
            event="SIGNAL", direction=signal, price=close,
            composite=composite, atr=atr,
        )

        if not self.state.active:
            if signal != 0:
                self._enter(signal, composite, atr, close, exit_plan)
        else:
            log.info("Position active (dir=%+d) — SL/TP manage exit", self.state.direction)

    def _write_analysis(self, diagnostics: Optional[dict], signal: int,
                         composite: float, atr: float, close: float) -> None:
        """Persist the latest signal-computation diagnostics for the
        dashboard. Purely informational — never read back by the engine."""
        if diagnostics is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "composite": composite,
            "atr": atr,
            "close": close,
            "reason": diagnostics.get("reason", ""),
            "metrics": diagnostics.get("metrics", {}),
        }
        try:
            tmp = self.log_dir / "analysis.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2, default=str))
            tmp.replace(self.log_dir / "analysis.json")
        except Exception as e:
            log.warning("Could not write analysis.json: %s", e)

    # ── WebSocket loop ────────────────────────────────────────────────────────

    async def _ws_loop(self):
        backoff = 5
        ws_url = WS_URL_TEMPLATE.format(interval=self.ws_interval)
        while self._running:
            try:
                log.info("WebSocket connecting to %s", ws_url)
                async with websockets.connect(
                    ws_url, ping_interval=20, ping_timeout=20,
                ) as ws:
                    backoff = 5
                    self._touch()
                    async for raw in ws:
                        self._touch()
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        if msg.get("k", {}).get("x"):   # candle closed
                            self.on_candle_close()
                            self._touch()
            except Exception as e:
                log.warning("WS error: %s — retry in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # ── Background monitor loop ───────────────────────────────────────────────

    async def _monitor_loop(self):
        # Between hourly candle closes this bot otherwise stays silent at
        # INFO level (SL/TP checks only log at DEBUG), which made a healthy,
        # quiet process indistinguishable from a stuck one when watching the
        # log file alone. A periodic heartbeat closes that gap.
        polls_since_heartbeat = 0
        while self._running:
            await asyncio.sleep(POLL_SECS)
            self._touch()
            mark = self.check_position() if self.state.active else None

            polls_since_heartbeat += 1
            if polls_since_heartbeat >= HEARTBEAT_EVERY_N_POLLS:
                polls_since_heartbeat = 0
                if mark is not None:
                    log.info("heartbeat — alive, position active, mark=%.1f", mark)
                else:
                    log.info("heartbeat — alive, flat" if not self.state.active
                              else "heartbeat — alive, position active (mark unavailable)")

    # ── Start ─────────────────────────────────────────────────────────────────

    async def run(self):
        log.info("LiveTrader START  dry_run=%s  scenario=%s", self.dry_run, self.scenario)

        if not self.dry_run:
            if not self.client.ping():
                raise RuntimeError("Cannot reach Bitget — check credentials / network")
            try:
                self.client.set_one_way_mode()
                self.client.set_margin_mode("isolated")
                self.client.set_leverage(1)
                log.info("Bitget account configured: isolated, 1× leverage, one-way mode")
            except Exception as e:
                log.warning("Account setup warning: %s", e)

        if self.state.active:
            log.info("Resuming open position  dir=%+d  entry=%.1f  size=%.3f",
                     self.state.direction, self.state.entry_price, self.state.size_remaining)

        threading.Thread(target=self._watchdog, daemon=True).start()

        # Calcola il segnale subito all'avvio
        self.on_candle_close()
        self._touch()

        await asyncio.gather(self._ws_loop(), self._monitor_loop())

    def stop(self):
        self._running = False
        log.info("LiveTrader stopping …")
