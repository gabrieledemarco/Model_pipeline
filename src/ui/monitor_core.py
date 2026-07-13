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


def _pair_closed_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair each PARTIAL_*/EXIT_* row with its most recent preceding ENTRY to
    recover the risk distance (|entry_price - sl|) needed for R-multiples,
    and recover each row's TRUE incremental pnl.

    Quirk in src/live/trader.py this has to correct for: `_partial_close`
    logs that leg's own net pnl, but `_exit_full` logs
    `gross - fees + state.realized_pnl` — i.e. the final EXIT_* row's
    `pnl_net` is the CUMULATIVE total for the whole trade (this leg plus
    every prior partial already folded in), not that leg's own contribution.
    Summing raw `pnl_net` across a trade's rows double-counts every partial
    that preceded a final exit. Fix: track cumulative partial pnl per
    episode and subtract it back out of the terminal EXIT_* row so every
    row in the returned frame holds its own true incremental pnl — safe to
    sum, safe to R-multiple, safe to bucket by day/session.

    trades.csv has no trade_id column. This assumes at most one open position
    at a time per strategy (true for every strategy this dashboard monitors)
    and walks rows in chronological order, tracking the active ENTRY's
    entry_price/sl until the next ENTRY replaces it. Returns one row per
    closed/partial event: timestamp, event, direction, pnl_net (true,
    incremental), r_multiple, entry_time (the ISO-parseable timestamp of the
    ENTRY it was paired with, for grouping into per-trade episodes — see
    list_trade_episodes). r_multiple is None when sl/qty aren't available.
    """
    cols = ["timestamp", "event", "direction", "price", "qty", "pnl_net", "fee", "r_multiple", "entry_time"]
    required = {"event", "timestamp", "price", "sl", "qty", "pnl_net"}
    if trades.empty or not required.issubset(trades.columns):
        return pd.DataFrame(columns=cols)

    df = trades.dropna(subset=["timestamp"]).sort_values("timestamp")
    rows = []
    basis = None  # {"entry_price": float, "sl": float, "entry_time": Timestamp}
    partial_pnl_so_far = 0.0
    for _, row in df.iterrows():
        event = row.get("event")
        if event == "ENTRY":
            sl, entry_price = row.get("sl"), row.get("price")
            basis = {
                "entry_price": float(entry_price), "sl": float(sl), "entry_time": row["timestamp"],
            } if pd.notna(sl) and pd.notna(entry_price) else None
            partial_pnl_so_far = 0.0
            continue
        if not isinstance(event, str) or not (event.startswith("EXIT") or event.startswith("PARTIAL")):
            continue
        raw_pnl = row.get("pnl_net")
        if pd.isna(raw_pnl):
            continue
        raw_pnl = float(raw_pnl)
        if event.startswith("PARTIAL"):
            true_pnl = raw_pnl  # already this leg's own incremental net
            partial_pnl_so_far += true_pnl
        else:  # EXIT_* — logged value is cumulative, subtract prior partials back out
            true_pnl = raw_pnl - partial_pnl_so_far

        r_multiple = None
        if basis is not None:
            risk_distance = abs(basis["entry_price"] - basis["sl"])
            qty = row.get("qty")
            if risk_distance > 0 and pd.notna(qty) and qty:
                r_multiple = true_pnl / (risk_distance * float(qty))
        qty = row.get("qty")
        price = row.get("price")
        direction = row.get("direction")

        # Fee isn't logged as its own column — trader.py computes it inline
        # and only writes the net result. Derive it instead of guessing
        # which of trader.py's two (asymmetric, partial vs full-exit) fee
        # formulas applies: gross (pure price move, no fees) minus the true
        # net we already recovered above is exactly the fee charged,
        # whatever formula produced it.
        fee = None
        if basis is not None and pd.notna(qty) and pd.notna(price) and pd.notna(direction):
            gross = float(direction) * float(qty) * (float(price) - basis["entry_price"])
            fee = gross - true_pnl

        rows.append({
            "timestamp": row["timestamp"],
            "event": event,
            "direction": direction,
            "price": float(price) if pd.notna(price) else None,
            "qty": float(qty) if pd.notna(qty) else None,
            "pnl_net": true_pnl,
            "fee": fee,
            "r_multiple": r_multiple,
            "entry_time": basis["entry_time"] if basis is not None else None,
        })
    return pd.DataFrame(rows, columns=cols)


def realized_pnl_total(trades: pd.DataFrame) -> float:
    """Sum of every trade's TRUE realized pnl — routed through
    _pair_closed_trades so a trade with partial closes isn't double-counted
    (see that function's docstring)."""
    closed = _pair_closed_trades(trades)
    if closed.empty:
        return 0.0
    return float(closed["pnl_net"].sum())


def compute_r_multiples(trades: pd.DataFrame) -> list[dict]:
    """R-multiple per closed/partial exit: pnl_net / (risk distance * qty)."""
    closed = _pair_closed_trades(trades)
    if closed.empty:
        return []
    closed = closed.dropna(subset=["r_multiple"])
    return [
        {"t": row["timestamp"].isoformat(), "r": float(row["r_multiple"]), "event": row["event"]}
        for _, row in closed.iterrows()
    ]


def compute_daily_pnl(trades: pd.DataFrame) -> dict:
    """Realized pnl_net summed per calendar day (UTC), for a calendar heatmap."""
    closed = _pair_closed_trades(trades)
    if closed.empty:
        return {}
    grouped = closed.groupby(closed["timestamp"].dt.date)["pnl_net"].sum()
    return {d.isoformat(): float(v) for d, v in grouped.items()}


# Approximate UTC session windows for the day/session breakdown. Not exact
# ICT kill-zone boundaries — coarse enough to spot session-level edge.
_SESSION_BOUNDS = [
    ("Asia", 0, 7),
    ("London", 7, 12),
    ("NY AM", 12, 16),
    ("NY PM", 16, 20),
    ("Late", 20, 24),
]


def _session_for_hour(hour: int) -> str:
    for name, start, end in _SESSION_BOUNDS:
        if start <= hour < end:
            return name
    return "Late"


def compute_session_stats(trades: pd.DataFrame) -> list[dict]:
    """Win rate / avg R / total pnl per UTC session bucket, in session order."""
    closed = _pair_closed_trades(trades)
    if closed.empty:
        return []
    closed = closed.copy()
    closed["session"] = closed["timestamp"].dt.hour.map(_session_for_hour)

    out = []
    for name, _, _ in _SESSION_BOUNDS:
        bucket = closed[closed["session"] == name]
        if bucket.empty:
            continue
        wins = bucket[bucket["pnl_net"] > 0]
        rs = bucket["r_multiple"].dropna()
        out.append({
            "session": name,
            "trades": int(len(bucket)),
            "win_rate": len(wins) / len(bucket),
            "avg_r": float(rs.mean()) if not rs.empty else None,
            "total_pnl": float(bucket["pnl_net"].sum()),
        })
    return out


def list_trade_episodes(trades: pd.DataFrame) -> list[dict]:
    """One self-contained record per ENTRY that has at least one close, for
    the trade-replay chart: entry terms (price/sl/tp1-3/qty) plus every
    close event, so the frontend can plot markers and floating PnL without
    re-deriving the ENTRY<->close pairing itself."""
    closed = _pair_closed_trades(trades)
    if closed.empty:
        return []
    closed = closed.dropna(subset=["entry_time"])
    if closed.empty:
        return []

    entries = trades[trades["event"] == "ENTRY"].dropna(subset=["timestamp"])
    entries_by_time = {row["timestamp"]: row for _, row in entries.iterrows()}

    episodes = []
    for entry_time, group in closed.groupby("entry_time"):
        entry_row = entries_by_time.get(entry_time)
        if entry_row is None:
            continue
        qty_total = entry_row.get("qty")
        equity_at_entry = entry_row.get("equity")
        episodes.append({
            "entry_time": entry_time.isoformat(),
            "direction": entry_row.get("direction"),
            "entry_price": float(entry_row["price"]) if pd.notna(entry_row.get("price")) else None,
            "sl": float(entry_row["sl"]) if pd.notna(entry_row.get("sl")) else None,
            "tp1": float(entry_row["tp1"]) if pd.notna(entry_row.get("tp1")) else None,
            "tp2": float(entry_row["tp2"]) if pd.notna(entry_row.get("tp2")) else None,
            "tp3": float(entry_row["tp3"]) if pd.notna(entry_row.get("tp3")) else None,
            "qty_total": float(qty_total) if pd.notna(qty_total) else None,
            "equity_at_entry": float(equity_at_entry) if pd.notna(equity_at_entry) else None,
            "closes": [
                {
                    "time": row["timestamp"].isoformat(),
                    "event": row["event"],
                    "price": row["price"],
                    "qty": row["qty"],
                    "pnl_net": float(row["pnl_net"]),
                    "fee": None if pd.isna(row["fee"]) else float(row["fee"]),
                    "r_multiple": None if pd.isna(row["r_multiple"]) else float(row["r_multiple"]),
                }
                for _, row in group.sort_values("timestamp").iterrows()
            ],
            "pnl_total": float(group["pnl_net"].sum()),
            "fees_total": float(group["fee"].dropna().sum()) if group["fee"].notna().any() else None,
            # One risk-adjusted R for the whole trade (pnl_total against the
            # entry-defined risk on the full size) — distinct from each
            # individual leg's own R already in `closes`.
            "r_multiple": (
                float(group["pnl_net"].sum() / (abs(entry_row["price"] - entry_row["sl"]) * qty_total))
                if pd.notna(entry_row.get("sl")) and pd.notna(qty_total)
                and abs(entry_row["price"] - entry_row["sl"]) > 0 and qty_total
                else None
            ),
        })

    episodes.sort(key=lambda e: e["entry_time"])
    return episodes


def compute_performance_stats(trades: pd.DataFrame) -> dict:
    """Win rate / avg win/loss / max drawdown / Sharpe from trades.csv.

    win_rate/avg_win/avg_loss/profit_factor are computed per TRADE (one
    entry -> one total, summing its true incremental legs via
    _pair_closed_trades — see that function's docstring for why a trade
    with a partial close can't just be summed from raw pnl_net), not per
    close-row, so a partial+final pair counts as one win or loss, not two.

    Returns {'insufficient_data': True} when there isn't enough closed-trade
    history to compute anything meaningful.
    """
    if trades.empty or "pnl_net" not in trades.columns:
        return {"insufficient_data": True}

    closed = _pair_closed_trades(trades)
    if closed.empty:
        return {"insufficient_data": True}

    episode_pnl = closed.groupby("entry_time")["pnl_net"].sum()
    episode_pnl = episode_pnl[episode_pnl != 0]
    if len(episode_pnl) < 2:
        return {"insufficient_data": True}

    wins = episode_pnl[episode_pnl > 0]
    losses = episode_pnl[episode_pnl < 0]
    win_rate = len(wins) / len(episode_pnl)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0

    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(-losses.sum()) if not losses.empty else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

    r_values = closed["r_multiple"].dropna()
    expectancy = float(r_values.mean()) if not r_values.empty else None

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
        "n_trades": len(episode_pnl),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }


def build_equity_series(trades: pd.DataFrame) -> pd.Series:
    """Return a timestamp-indexed equity series, empty if unavailable."""
    if trades.empty or "equity" not in trades.columns or "timestamp" not in trades.columns:
        return pd.Series(dtype=float)
    df = trades.dropna(subset=["equity", "timestamp"]).sort_values("timestamp")
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["equity"].values, index=pd.DatetimeIndex(df["timestamp"]))
