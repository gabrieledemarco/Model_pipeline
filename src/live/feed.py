"""
Binance Futures WebSocket kline feed.

Subscribes to btcusdt@kline_{interval} streams via the combined-stream
endpoint.  Yields a KlineBar whenever a bar is *closed* (k.x == true).

Reconnects automatically with exponential back-off on any error.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import websockets
import websockets.exceptions

log = logging.getLogger("live.feed")

FSTREAM = "wss://fstream.binance.com/stream"
DEFAULT_INTERVALS = ["1h", "15m", "1m"]


@dataclass(frozen=True)
class KlineBar:
    """One closed OHLCV bar received from the WebSocket."""
    interval:  str    # "1h" | "15m" | "1m"
    open_time: int    # epoch ms
    open:  float
    high:  float
    low:   float
    close: float
    volume: float
    n_trades: int


def _parse(msg: str) -> KlineBar | None:
    """Parse a combined-stream message; return KlineBar if bar is closed."""
    try:
        data = json.loads(msg)
        k = data["data"]["k"]
        if not k["x"]:       # bar still open
            return None
        return KlineBar(
            interval  = k["i"],
            open_time = int(k["t"]),
            open      = float(k["o"]),
            high      = float(k["h"]),
            low       = float(k["l"]),
            close     = float(k["c"]),
            volume    = float(k["v"]),
            n_trades  = int(k["n"]),
        )
    except (KeyError, ValueError, TypeError):
        return None


async def stream_klines(
    intervals: list[str] = DEFAULT_INTERVALS,
    max_retries: int = 0,           # 0 = infinite
) -> AsyncIterator[KlineBar]:
    """
    Async generator that yields closed KlineBars for the given intervals.

    Usage::

        async for bar in stream_klines(["1h", "15m", "1m"]):
            process(bar)
    """
    streams = "/".join(f"btcusdt@kline_{i}" for i in intervals)
    uri     = f"{FSTREAM}?streams={streams}"

    attempt = 0
    while True:
        try:
            log.info("Connecting to %s", uri)
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=30,
                open_timeout=30,
            ) as ws:
                attempt = 0   # reset back-off on successful connect
                print(f"  [feed] Connected — streams: {', '.join(intervals)}")
                async for raw in ws:
                    bar = _parse(raw)
                    if bar is not None:
                        yield bar

        except (websockets.exceptions.ConnectionClosed,
                ConnectionError, OSError, asyncio.TimeoutError) as exc:
            attempt += 1
            wait = min(2 ** attempt, 60)
            print(f"  [feed] Disconnected ({exc}). "
                  f"Reconnecting in {wait}s … (attempt {attempt})")
            await asyncio.sleep(wait)
            if max_retries and attempt >= max_retries:
                raise RuntimeError(f"Feed: too many reconnects ({attempt})") from exc

        except Exception as exc:
            log.error("Feed: unexpected error: %s", exc, exc_info=True)
            await asyncio.sleep(10)
