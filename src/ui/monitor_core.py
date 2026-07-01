"""Read-only data layer for live-strategy monitoring, shared by the FastAPI
live dashboard (src/ui/monitor_api.py).

Strictly read-only: only reads files under logs/strategies/{strategy_id}/
(meta.json, position.json, trades.csv, live_trader.log) and inspects local
processes via psutil to check whether the corresponding live_trader.py
process is still alive. Do NOT import anything from src.live (trader/exchange
clients) here — this module must remain incapable of touching an exchange or
writing to the logs/ tree.
"""
from __future__ import annotations

import glob
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover - surfaced by the API layer
    psutil = None

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_GLOB = str(ROOT / "logs" / "strategies" / "*" / "meta.json")

STALL_THRESHOLD_SECONDS = 10 * 60         # log considered stale after 10 min
LOG_TAIL_LINES = 200

# psutil.Process objects reused across polls, keyed by pid — cpu_percent()
# only becomes meaningful on the *second* call on the *same* Process object.
_PROC_CACHE: dict[int, "psutil.Process"] = {}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class StrategyInfo:
    strategy_id: str
    dir_path: Path
    meta: dict = field(default_factory=dict)
    meta_ok: bool = False
    status: str = "unknown"       # "running" | "stopped" | "unknown"
    stalled: bool = False
    uptime_seconds: Optional[float] = None
    cpu_percent: Optional[float] = None
    rss_mb: Optional[float] = None


# --------------------------------------------------------------------------
# Discovery & health
# --------------------------------------------------------------------------
def _safe_load_json(path: Path) -> Optional[dict]:
    """Load JSON, returning None (not raising) on any failure.

    Files may be mid-write by a live process, truncated, or briefly absent —
    none of that should ever crash the dashboard.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def discover_strategies() -> list[StrategyInfo]:
    """Glob logs/strategies/*/meta.json and build a StrategyInfo per hit."""
    infos: list[StrategyInfo] = []
    for meta_path_str in sorted(glob.glob(STRATEGIES_GLOB)):
        meta_path = Path(meta_path_str)
        strategy_dir = meta_path.parent
        strategy_id = strategy_dir.name
        meta = _safe_load_json(meta_path)
        info = StrategyInfo(
            strategy_id=strategy_id,
            dir_path=strategy_dir,
            meta=meta or {},
            meta_ok=meta is not None,
        )
        _evaluate_health(info)
        infos.append(info)
    return infos


def _cached_process(pid: int) -> "psutil.Process":
    proc = _PROC_CACHE.get(pid)
    if proc is None:
        proc = psutil.Process(pid)
        proc.cpu_percent(interval=None)  # prime the baseline
        _PROC_CACHE[pid] = proc
    return proc


def _evaluate_health(info: StrategyInfo) -> None:
    """Populate status/stalled/uptime/cpu/rss on the given StrategyInfo."""
    pid = info.meta.get("pid")
    alive = False
    proc = None

    if psutil is None:
        info.status = "unknown"
        return

    if pid is not None:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = None

    if pid is not None and psutil.pid_exists(pid):
        try:
            proc = _cached_process(pid)
            cmdline = " ".join(proc.cmdline())
            if "live_trader.py" in cmdline:
                alive = True
            else:
                # PID reused by an unrelated process.
                alive = False
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            alive = False

    if pid is None:
        info.status = "unknown"
    elif alive:
        info.status = "running"
        try:
            info.cpu_percent = proc.cpu_percent(interval=None)
            info.rss_mb = proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    else:
        info.status = "stopped"

    # Uptime from meta's started_at, regardless of live status (best effort).
    started_at = info.meta.get("started_at")
    if started_at:
        try:
            started_dt = _parse_ts(started_at)
            if started_dt is not None:
                now = datetime.now(started_dt.tzinfo or timezone.utc)
                info.uptime_seconds = (now - started_dt).total_seconds()
        except Exception:
            pass

    # Stall detection: process alive but log hasn't been touched recently.
    if info.status == "running":
        log_path = info.dir_path / "live_trader.log"
        try:
            if log_path.exists():
                mtime = log_path.stat().st_mtime
                age = datetime.now().timestamp() - mtime
                if age > STALL_THRESHOLD_SECONDS:
                    info.stalled = True
        except OSError:
            pass


def status_label(info: StrategyInfo) -> str:
    if info.stalled:
        return "stalled"
    return info.status


def _fmt_uptime(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


# --------------------------------------------------------------------------
# File loaders (all read-only)
# --------------------------------------------------------------------------
def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


_TRADE_COLS = [
    "strategy_id", "timestamp", "event", "direction", "price", "qty",
    "sl", "tp1", "tp2", "tp3", "composite", "atr", "pnl_net", "equity",
    "note",
]


def load_trades(strategy_dir: Path) -> pd.DataFrame:
    """Load trades.csv defensively; return an empty frame with expected
    columns if the file is missing, empty, or malformed."""
    path = Path(strategy_dir) / "trades.csv"
    if not path.exists():
        return pd.DataFrame(columns=_TRADE_COLS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=_TRADE_COLS)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def load_position(strategy_dir: Path) -> dict:
    data = _safe_load_json(Path(strategy_dir) / "position.json")
    return data or {}


def load_analysis(strategy_dir: Path) -> dict:
    """Latest signal-computation diagnostics (indicators + reasoning) written
    by the trader each bar close — see LiveTrader._write_analysis(). Empty
    dict if the strategy hasn't produced one yet (e.g. just started)."""
    data = _safe_load_json(Path(strategy_dir) / "analysis.json")
    return data or {}


def load_log_tail(strategy_dir: Path, n_lines: int = LOG_TAIL_LINES) -> list[str]:
    """Return the last n_lines of live_trader.log with bounded memory use.

    live_trader.log has no rotation and grows unboundedly, so this streams
    line-by-line into a maxlen deque instead of loading the whole file via
    readlines().
    """
    path = Path(strategy_dir) / "live_trader.log"
    if not path.exists():
        return []
    try:
        with open(path, "r", errors="replace") as f:
            return [line.rstrip("\n") for line in deque(f, maxlen=n_lines)]
    except Exception:
        return []


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def current_equity(trades: pd.DataFrame, meta: dict) -> float:
    if not trades.empty and "equity" in trades.columns:
        eq_series = trades["equity"].dropna()
        if not eq_series.empty:
            return float(eq_series.iloc[-1])
    return float(meta.get("capital", 0.0) or 0.0)


def realized_pnl_total(trades: pd.DataFrame) -> float:
    if trades.empty or "pnl_net" not in trades.columns:
        return 0.0
    return float(trades["pnl_net"].dropna().sum())


def compute_performance_stats(trades: pd.DataFrame) -> dict:
    """Win rate / avg win/loss / max drawdown / Sharpe from trades.csv.

    Uses only pandas math on pnl_net / equity columns — kept intentionally
    simple. Returns {'insufficient_data': True} when there isn't enough
    closed-trade history to compute anything meaningful.
    """
    if trades.empty or "pnl_net" not in trades.columns:
        return {"insufficient_data": True}

    closed = trades.dropna(subset=["pnl_net"])
    closed = closed[closed["pnl_net"] != 0]
    if len(closed) < 2:
        return {"insufficient_data": True}

    wins = closed[closed["pnl_net"] > 0]["pnl_net"]
    losses = closed[closed["pnl_net"] < 0]["pnl_net"]
    win_rate = len(wins) / len(closed) if len(closed) else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0

    # Max drawdown & Sharpe from the equity curve (percentage returns).
    max_dd = None
    sharpe = None
    if "equity" in trades.columns:
        eq = trades["equity"].dropna()
        if len(eq) >= 2:
            running_max = eq.cummax()
            dd = (eq - running_max) / running_max
            max_dd = float(dd.min())
            rets = eq.pct_change().dropna()
            if len(rets) >= 2 and rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * (252 ** 0.5))

    return {
        "insufficient_data": False,
        "n_trades": len(closed),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


def build_equity_series(trades: pd.DataFrame) -> pd.Series:
    """Return a timestamp-indexed equity series, empty if unavailable."""
    if trades.empty or "equity" not in trades.columns or "timestamp" not in trades.columns:
        return pd.Series(dtype=float)
    df = trades.dropna(subset=["equity", "timestamp"]).sort_values("timestamp")
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["equity"].values, index=pd.DatetimeIndex(df["timestamp"]))
