"""
Persistent position state for the live trader.

Stores the current position (if any) in a JSON file so the trader
can survive process restarts without losing track of open trades.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Default location (single-strategy fallback). Overridden per-process via
# configure() so each concurrent strategy instance writes to its own
# logs/strategies/{strategy_id}/position.json.
STATE_FILE = Path(__file__).parent.parent.parent / "logs" / "position.json"

# Self-describing fields stamped onto every PositionState written by this
# process, set via configure() at startup.
_STRATEGY_ID: str = ""
_SCENARIO: str = ""
_EXCHANGE: str = ""


def configure(state_file: Path, strategy_id: str, scenario: str, exchange: str) -> None:
    """Point this module at a per-strategy position.json and stamp metadata.

    Must be called once at process startup, before any load()/save() calls.
    """
    global STATE_FILE, _STRATEGY_ID, _SCENARIO, _EXCHANGE
    STATE_FILE = Path(state_file)
    _STRATEGY_ID = strategy_id
    _SCENARIO = scenario
    _EXCHANGE = exchange


@dataclass
class PositionState:
    # Core position fields
    active:           bool  = False
    direction:        int   = 0       # +1 long / -1 short
    entry_price:      float = 0.0
    size_total:       float = 0.0     # BTC at entry
    size_remaining:   float = 0.0     # BTC still open

    # Price levels
    sl:   float = 0.0
    tp1:  float = 0.0
    tp2:  float = 0.0
    tp3:  float = 0.0

    # Progression flags
    tp1_hit: bool = False
    tp2_hit: bool = False
    be_moved: bool = False            # SL moved to break-even after TP1

    # Exchange order tracking
    sl_order_id: Optional[str] = None

    # Metadata
    entry_time:   str   = ""
    composite:    float = 0.0
    atr_at_entry: float = 0.0
    equity_at_entry: float = 0.0

    # Realized P&L during current trade
    realized_pnl: float = 0.0        # net after fees

    # Exit model: "ladder" (default, 3-tier TP1/TP2/TP3 + BE trail — used by
    # the composite-score and Wyckoff strategies) or "single_tp" (one TP, one
    # SL, optional time-stop — used by strategies with their own validated
    # exit rule, e.g. ICT Silver Bullet's TP=3x FVG width / SL=FVG edge).
    exit_mode:    str = "ladder"
    time_stop_at: str = ""           # ISO8601 UTC deadline; "" = no time-stop

    # Self-describing fields (dashboard identification)
    strategy_id: str = ""
    scenario:    str = ""
    exchange:    str = ""


def _stamp(state: PositionState) -> PositionState:
    """Attach the current process's strategy_id/scenario/exchange metadata."""
    state.strategy_id = _STRATEGY_ID
    state.scenario     = _SCENARIO
    state.exchange     = _EXCHANGE
    return state


def load() -> PositionState:
    """Load position from disk. Returns flat PositionState only if file absent.

    Deliberately does NOT swallow read/parse errors on an existing file: if
    position.json exists but is unreadable or corrupted, a real open position
    could be silently forgotten and re-entered on top of an existing one on
    the exchange. Failing loudly here forces manual inspection instead.
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return _stamp(PositionState())
    data = json.loads(STATE_FILE.read_text())
    return _stamp(PositionState(**data))


def save(state: PositionState) -> None:
    """Persist position to disk atomically."""
    _stamp(state)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(STATE_FILE)


def open_position(
    direction: int,
    entry_price: float,
    size: float,
    sl: float,
    tp1: float,
    tp2: float,
    tp3: float,
    composite: float,
    atr: float,
    equity: float,
    sl_order_id: Optional[str] = None,
    exit_mode: str = "ladder",
    time_stop_at: str = "",
) -> PositionState:
    state = PositionState(
        active          = True,
        direction       = direction,
        entry_price     = entry_price,
        size_total      = size,
        size_remaining  = size,
        sl              = sl,
        tp1             = tp1,
        tp2             = tp2,
        tp3             = tp3,
        sl_order_id     = sl_order_id,
        entry_time      = datetime.now(timezone.utc).isoformat(),
        composite       = composite,
        atr_at_entry    = atr,
        equity_at_entry = equity,
        exit_mode       = exit_mode,
        time_stop_at    = time_stop_at,
    )
    save(state)
    return state


def close_position() -> PositionState:
    """Mark position as closed and persist."""
    state = PositionState()
    save(state)
    return state


def update(state: PositionState) -> None:
    """Persist updated state."""
    save(state)
