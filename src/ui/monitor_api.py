"""FastAPI backend for the read-only live-strategy monitoring dashboard.

Serves real-time push updates over WebSockets (replacing an earlier
Streamlit dashboard that only refreshed every 15s via full page reload).

Strictly read-only: this module (and everything it imports from
`src.ui.monitor_core`) only ever reads files under logs/strategies/ and
inspects local processes via psutil. It must NEVER import anything from
`src.live` (the trader/exchange client modules) and must never place,
modify, or close trades.

Run from the project root with:

    .venv/bin/python -m uvicorn src.ui.monitor_api:app --host 0.0.0.0 --port 8501
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make `src` importable regardless of how uvicorn was invoked.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.monitor_core import (  # noqa: E402
    StrategyInfo,
    build_equity_series,
    compute_daily_pnl,
    compute_performance_stats,
    compute_r_multiples,
    compute_session_stats,
    current_equity,
    discover_strategies,
    list_trade_episodes,
    load_analysis,
    load_log_tail,
    load_position,
    load_trades,
    realized_pnl_total,
)

logger = logging.getLogger("monitor_api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

TICK_SECONDS = 2
EQUITY_CURVE_CAP = 500
TRADES_LIMIT = 100

app = FastAPI(title="Live Strategy Monitor (read-only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# JSON safety helpers
# --------------------------------------------------------------------------
def _json_safe(obj):
    """Recursively coerce a value tree into something json.dumps can emit,
    replacing NaN/Infinity with None and converting numpy/pandas scalars,
    Timestamps, and datetimes into plain JSON-friendly types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    # pandas NA / NaT / other scalar "missing" markers.
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _dumps(payload: dict) -> str:
    return json.dumps(_json_safe(payload))


# --------------------------------------------------------------------------
# Shared payload builders (used by both WS endpoints)
# --------------------------------------------------------------------------
def _position_desc(position: dict) -> str:
    if not position or not position.get("active"):
        return "flat"
    direction = position.get("direction")
    size = position.get("size_remaining")
    if size is None:
        size = position.get("size_total")
    dir_str = "LONG" if (direction is not None and direction > 0) else "SHORT"
    return f"{dir_str} {size}"


def _equity_curve_points(trades: pd.DataFrame, cap: int = EQUITY_CURVE_CAP) -> list[dict]:
    series = build_equity_series(trades)
    if series.empty:
        return []
    if len(series) > cap:
        series = series.iloc[-cap:]
    return [{"t": idx, "e": val} for idx, val in series.items()]


def _trades_records(trades: pd.DataFrame, limit: int = TRADES_LIMIT) -> list[dict]:
    if trades.empty:
        return []
    df = trades
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)
    df = df.head(limit).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(
            lambda x: x.isoformat() if pd.notna(x) else None
        )
    df = df.where(pd.notna(df), None)
    return df.to_dict("records")


def _strategy_overview_entry(info: StrategyInfo, today) -> tuple[dict, Optional[list[dict]], int, float, int]:
    """Build the per-strategy overview dict plus side values needed for the
    aggregate block: (entry, equity_curve_points_or_None, trades_today_count,
    realized_pnl, is_open_position)."""
    trades = load_trades(info.dir_path)
    position = load_position(info.dir_path)
    equity = current_equity(trades, info.meta)
    realized = realized_pnl_total(trades)

    trades_today_count = 0
    if not trades.empty and "timestamp" in trades.columns and "event" in trades.columns:
        # Exclude SIGNAL rows (logged every bar close, no trade) — only
        # ENTRY/PARTIAL_*/EXIT_* are actual trade activity.
        real_trades = trades[trades["event"] != "SIGNAL"]
        ts = real_trades["timestamp"].dropna()
        if not ts.empty:
            trades_today_count = int((ts.dt.date == today).sum())

    curve_points = _equity_curve_points(trades)

    entry = {
        "strategy_id": info.strategy_id,
        "scenario": info.meta.get("scenario"),
        "exchange": info.meta.get("exchange"),
        "dry_run": info.meta.get("dry_run"),
        "status": info.status,
        "stalled": info.stalled,
        "uptime_seconds": info.uptime_seconds,
        "cpu_percent": info.cpu_percent,
        "rss_mb": info.rss_mb,
        "position_desc": _position_desc(position),
        "equity": equity,
        "realized_pnl": realized,
        # Raw fields so the frontend can compute live unrealized P&L against
        # its own already-open BTCUSDT mark-price feed, without this
        # (strictly read-only, no-exchange-access) backend needing to fetch
        # a mark price itself.
        "position_direction": position.get("direction") if position.get("active") else None,
        "position_entry_price": position.get("entry_price") if position.get("active") else None,
        "position_size_remaining": position.get("size_remaining") if position.get("active") else None,
    }
    is_open = 1 if position.get("active") else 0
    return entry, (curve_points or None), trades_today_count, (realized if realized is not None else 0.0), is_open


def build_overview_payload() -> dict:
    server_time = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).date()

    strategies_out: list[dict] = []
    equity_curves: dict[str, list[dict]] = {}
    total_realized_pnl = 0.0
    open_positions = 0
    trades_today = 0

    try:
        infos = discover_strategies()
    except Exception:
        logger.exception("discover_strategies() failed while building overview payload")
        infos = []

    for info in infos:
        try:
            entry, curve_points, day_count, realized, is_open = _strategy_overview_entry(info, today)
        except Exception:
            logger.warning("Skipping strategy '%s' in overview tick due to error", info.strategy_id, exc_info=True)
            continue
        strategies_out.append(entry)
        if curve_points:
            equity_curves[info.strategy_id] = curve_points
        try:
            if realized is not None and math.isfinite(float(realized)):
                total_realized_pnl += float(realized)
        except (TypeError, ValueError):
            pass
        open_positions += is_open
        trades_today += day_count

    return {
        "type": "overview",
        "server_time": server_time,
        "strategies": strategies_out,
        "aggregate": {
            "total_realized_pnl": total_realized_pnl,
            "open_positions": open_positions,
            "trades_today": trades_today,
            "equity_curves": equity_curves,
        },
    }


def build_detail_payload(info: StrategyInfo) -> dict:
    trades = load_trades(info.dir_path)
    position = load_position(info.dir_path)
    if not position:
        position = {"active": False}

    return {
        "type": "detail",
        "strategy_id": info.strategy_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "meta": info.meta,
        "status": info.status,
        "stalled": info.stalled,
        "uptime_seconds": info.uptime_seconds,
        "cpu_percent": info.cpu_percent,
        "rss_mb": info.rss_mb,
        "position": position,
        "equity_curve": _equity_curve_points(trades),
        "trades": _trades_records(trades),
        "stats": compute_performance_stats(trades),
        "r_multiples": compute_r_multiples(trades),
        "daily_pnl": compute_daily_pnl(trades),
        "session_stats": compute_session_stats(trades),
        "trade_episodes": list_trade_episodes(trades),
        "log_tail": load_log_tail(info.dir_path),
        "analysis": load_analysis(info.dir_path),
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/overview")
async def ws_overview(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                payload = build_overview_payload()
            except Exception:
                logger.exception("Failed to build overview payload; sending empty tick")
                payload = {
                    "type": "overview",
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "strategies": [],
                    "aggregate": {
                        "total_realized_pnl": 0.0,
                        "open_positions": 0,
                        "trades_today": 0,
                        "equity_curves": {},
                    },
                }
            try:
                await websocket.send_text(_dumps(payload))
            except (WebSocketDisconnect, RuntimeError):
                return
            await asyncio.sleep(TICK_SECONDS)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Unexpected error in /ws/overview loop; closing connection")
        return


@app.websocket("/ws/strategy/{strategy_id}")
async def ws_strategy_detail(websocket: WebSocket, strategy_id: str):
    await websocket.accept()

    async def _not_found():
        try:
            await websocket.send_text(
                _dumps({"type": "error", "message": f"strategy '{strategy_id}' not found"})
            )
        except (WebSocketDisconnect, RuntimeError):
            pass
        try:
            await websocket.close(code=1008)
        except RuntimeError:
            pass

    try:
        while True:
            try:
                infos = discover_strategies()
            except Exception:
                logger.exception("discover_strategies() failed in /ws/strategy/%s loop", strategy_id)
                infos = []

            info = next((i for i in infos if i.strategy_id == strategy_id), None)
            if info is None:
                await _not_found()
                return

            try:
                payload = build_detail_payload(info)
            except Exception:
                logger.warning(
                    "Failed building detail payload for '%s' this tick; sending minimal payload",
                    strategy_id,
                    exc_info=True,
                )
                payload = {
                    "type": "detail",
                    "strategy_id": info.strategy_id,
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "meta": info.meta,
                    "status": info.status,
                    "stalled": info.stalled,
                    "uptime_seconds": info.uptime_seconds,
                    "cpu_percent": info.cpu_percent,
                    "rss_mb": info.rss_mb,
                    "position": {"active": False},
                    "equity_curve": [],
                    "trades": [],
                    "stats": {"insufficient_data": True},
                    "r_multiples": [],
                    "daily_pnl": {},
                    "session_stats": [],
                    "trade_episodes": [],
                    "log_tail": [],
                    "analysis": {},
                }

            try:
                await websocket.send_text(_dumps(payload))
            except (WebSocketDisconnect, RuntimeError):
                return

            await asyncio.sleep(TICK_SECONDS)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Unexpected error in /ws/strategy/%s loop; closing connection", strategy_id)
        return


# Mount static frontend last so the /api and /ws routes above take
# precedence over the catch-all static handler. html=True serves
# static/index.html for `/` once the frontend agent has populated it;
# until then this simply 404s on `/`, which is expected.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
