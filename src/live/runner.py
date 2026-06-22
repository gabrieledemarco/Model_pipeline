"""
Live paper-trading runner.

Bootstrap → signal loop → paper traders.

Architecture
───────────────────────────────────────────────────────────────────────────────
• 1H buffer   : rolling window of 1H OHLCV + indicators (all history cached)
• 15M buffer  : last 300 bars of 15m OHLCV + indicators (for s_15m)
• 1M buffer   : last 200 bars of 1m OHLCV + indicators  (for s_1m)
• Resample 1H → 1W, 1D, 4H on each 1H bar close for HTF signals
• Premium + funding: refreshed every hour via REST in a background task
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger("live.runner")

# ── strategy modules ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.indicators import add_indicators
from src.strategy.signals    import build_signal_matrix, LONG_THRESH, SHORT_THRESH
from src.strategy.optimizer  import SCENARIOS, apply_filters

from src.live.bootstrap   import (bootstrap_all, fetch_funding,
                                   fetch_premium_1h, fetch_rest_bars)
from src.live.feed        import stream_klines, KlineBar
from src.live.paper_trader import PaperTrader

STATE_PATH = Path("data/paper_trading_state.json")
SCENARIO   = "Session 08-21"
RESAMPLE_AGG = {"open": "first", "high": "max",
                "low":  "min",   "close": "last", "volume": "sum"}

# Rolling window sizes
MAX_1H  = 5_000    # ~7 months of hourly bars
MAX_4H  = 3_000    # ~500 days of 4H bars
MAX_1D  = 2_000    # ~5.5 years of daily bars
MAX_15M = 600      # ~6 days
MAX_1M  = 250      # ~4 hours (enough for vol_ratio / log_ret)


class LiveRunner:
    """
    Parameters
    ----------
    traders  : dict label → PaperTrader
    verbose  : print every 15m/1m event (adds noise)
    """

    def __init__(
        self,
        traders: Dict[str, PaperTrader],
        verbose: bool = False,
    ) -> None:
        self.traders = traders
        self.verbose = verbose

        # Rolling OHLCV windows (raw, no indicators yet)
        self._buf_1h:  pd.DataFrame = pd.DataFrame()
        self._buf_4h:  pd.DataFrame = pd.DataFrame()
        self._buf_1d:  pd.DataFrame = pd.DataFrame()
        self._buf_15m: pd.DataFrame = pd.DataFrame()
        self._buf_1m:  pd.DataFrame = pd.DataFrame()

        # Auxiliary series (refreshed hourly)
        self._funding:    pd.Series = pd.Series(dtype=float)
        self._premium_1h: pd.Series = pd.Series(dtype=float)

        self._last_print_hour = -1

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    async def bootstrap(self, verbose: bool = True) -> None:
        print("\n── Bootstrap ────────────────────────────────────────────────")
        raw = bootstrap_all(verbose=verbose)
        self._buf_1h  = raw["1H"]
        self._buf_4h  = raw.get("4H", pd.DataFrame())
        self._buf_1d  = raw.get("1D", pd.DataFrame())
        self._buf_15m = raw["15M"]
        self._buf_1m  = raw["1M"]
        self._funding    = raw["funding"]
        self._premium_1h = raw["premium_1h"]
        print(f"  Bootstrap complete.  Ready to trade.")

    # ── Main async loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start feed and process events forever."""
        # Background task: refresh funding + premium every hour
        asyncio.create_task(self._refresh_loop())

        print("\n── Live feed ────────────────────────────────────────────────")
        async for bar in stream_klines(["1h", "15m", "1m"]):
            await self._on_bar(bar)

    # ── Bar handler ───────────────────────────────────────────────────────────

    async def _on_bar(self, bar: KlineBar) -> None:
        ts = datetime.utcfromtimestamp(bar.open_time / 1000)
        row = pd.DataFrame([{
            "open": bar.open, "high": bar.high,
            "low": bar.low,   "close": bar.close, "volume": bar.volume,
        }], index=pd.DatetimeIndex([ts]))

        if bar.interval == "1m":
            self._buf_1m = pd.concat([self._buf_1m, row]).tail(MAX_1M)
            if self.verbose:
                print(f"  [1m close] {ts}  close={bar.close:,.0f}")

        elif bar.interval == "15m":
            self._buf_15m = pd.concat([self._buf_15m, row]).tail(MAX_15M)
            if self.verbose:
                print(f"  [15m close] {ts}  close={bar.close:,.0f}")

        elif bar.interval == "1h":
            self._buf_1h = pd.concat([self._buf_1h, row]).tail(MAX_1H)
            await self._on_1h_close(ts, bar)

    async def _on_1h_close(self, ts: datetime, bar: KlineBar) -> None:
        """Full signal computation + paper-trader update on each 1H close."""

        # ── Enrich 1H buffer ────────────────────────────────────────────────
        df_1h = add_indicators(self._buf_1h)

        # ── Real HTF buffers (no resampling) ────────────────────────────────
        raw_4h = self._buf_4h if not self._buf_4h.empty else pd.DataFrame()
        raw_1d = self._buf_1d if not self._buf_1d.empty else pd.DataFrame()
        df_4h = add_indicators(raw_4h) if len(raw_4h) > 10 else raw_4h
        df_1d = add_indicators(raw_1d) if len(raw_1d) > 10 else raw_1d
        # 1W from real 1D futures (resample only the aggregation, not TF data)
        df_1w = (raw_1d.resample("W", label="left", closed="left")
                       .agg(RESAMPLE_AGG).dropna(subset=["close"])
                 if not raw_1d.empty else pd.DataFrame())
        df_1w = add_indicators(df_1w) if len(df_1w) > 10 else df_1w

        tf_data = {"1W": df_1w, "1D": df_1d, "4H": df_4h, "1H": df_1h}

        # ── Enrich sub-1H buffers ────────────────────────────────────────────
        df_15m_ind = (add_indicators(self._buf_15m)
                      if len(self._buf_15m) > 50 else None)
        df_1m_ind  = (add_indicators(self._buf_1m)
                      if len(self._buf_1m)  > 50 else None)

        # ── Synthetic OI (generate from daily close if premium unavailable) ──
        from src.strategy.data_fetcher import generate_oi
        oi_df = generate_oi(df_1d["close"]) if len(df_1d) > 10 else pd.DataFrame()

        # ── Build signal matrix ──────────────────────────────────────────────
        try:
            signals = build_signal_matrix(
                tf_data,
                oi_df,
                self._funding,
                premium_1h = (self._premium_1h
                              if len(self._premium_1h) > 100 else None),
                df_15m     = df_15m_ind,
                df_1m      = df_1m_ind,
            )
        except Exception as exc:
            log.warning("Signal matrix build failed: %s", exc)
            return

        # ── Apply scenario (session) filter ─────────────────────────────────
        cfg     = SCENARIOS[SCENARIO]
        signals = apply_filters(signals, cfg)

        # ── Current bar's signal ─────────────────────────────────────────────
        if signals.empty or ts not in signals.index:
            return

        row_sig   = signals.loc[ts]
        signal    = int(row_sig["signal"])
        composite = float(row_sig["composite"])

        atr_val  = float(df_1h["atr_14"].iloc[-1]) if "atr_14" in df_1h.columns else 1.0
        rvol_val = float(df_1h["rvol_20"].iloc[-1]) if "rvol_20" in df_1h.columns else 0.20

        # ── Update all paper traders ─────────────────────────────────────────
        ohlc = bar

        print(f"\n[{ts}] 1H close  "
              f"O={ohlc.open:,.0f} H={ohlc.high:,.0f} "
              f"L={ohlc.low:,.0f} C={ohlc.close:,.0f}  "
              f"sig={signal:+d}  score={composite:.1f}  "
              f"ATR={atr_val:.0f}  rvol={rvol_val:.3f}")

        for trader in self.traders.values():
            events = trader.on_bar(
                ts       = ts,
                open_px  = ohlc.open,
                high_px  = ohlc.high,
                low_px   = ohlc.low,
                close_px = ohlc.close,
                atr      = atr_val,
                rvol     = rvol_val,
                signal   = signal,
                composite= composite,
            )
            for ev in events:
                print(f"  [{trader.label}] {ev}")

        # ── Dashboard every 4 bars (or on first bar) ─────────────────────────
        if ts.hour % 4 == 0 or self._last_print_hour < 0:
            self._last_print_hour = ts.hour
            self._print_dashboard(ts)
            self._save_state(ts)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def _print_dashboard(self, ts: datetime) -> None:
        print(f"\n{'─'*80}")
        print(f"  PAPER TRADING DASHBOARD  {ts}  (UTC)  Scenario: {SCENARIO}")
        print(f"{'─'*80}")
        for trader in self.traders.values():
            print(f"  {trader.summary()}")
        print(f"{'─'*80}")

    def _save_state(self, ts: datetime) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "timestamp": ts.isoformat(),
            "scenario":  SCENARIO,
            "traders":   {k: t.to_dict() for k, t in self.traders.items()},
        }
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))

    # ── Background refresh ────────────────────────────────────────────────────

    async def _refresh_loop(self) -> None:
        """Refresh funding, premium, and HTF bars every hour."""
        while True:
            await asyncio.sleep(3600)
            try:
                new_funding = await asyncio.to_thread(fetch_funding)
                if not new_funding.empty:
                    self._funding = new_funding
                new_prem = await asyncio.to_thread(fetch_premium_1h)
                if not new_prem.empty:
                    self._premium_1h = new_prem
                # Top up 1H buffer
                if not self._buf_1h.empty:
                    new_1h = await asyncio.to_thread(
                        fetch_rest_bars, "1H", self._buf_1h.index[-1])
                    if not new_1h.empty:
                        self._buf_1h = (pd.concat([self._buf_1h, new_1h])
                                        .tail(MAX_1H))
                # Top up real 4H buffer
                if not self._buf_4h.empty:
                    new_4h = await asyncio.to_thread(
                        fetch_rest_bars, "4H", self._buf_4h.index[-1])
                    if not new_4h.empty:
                        self._buf_4h = (pd.concat([self._buf_4h, new_4h])
                                        .drop_duplicates().tail(MAX_4H))
                # Top up real 1D buffer
                if not self._buf_1d.empty:
                    new_1d = await asyncio.to_thread(
                        fetch_rest_bars, "1D", self._buf_1d.index[-1])
                    if not new_1d.empty:
                        self._buf_1d = (pd.concat([self._buf_1d, new_1d])
                                        .drop_duplicates().tail(MAX_1D))
                log.info("Refreshed funding, premium, and HTF bars")
            except Exception as exc:
                log.warning("Refresh failed: %s", exc)
