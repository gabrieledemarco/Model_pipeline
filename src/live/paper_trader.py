"""
Paper-trading state machine — mirrors the backtest engine logic exactly.

One PaperTrader instance per strategy variant.  Call on_bar() for every
closed 1H bar in chronological order; the trader handles entries, partial
TP closes, stop loss, and all risk controls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

# Constants from backtest engine
ATR_SL  = 2.0
ATR_TP1 = 2.0
ATR_TP2 = 4.0
ATR_TP3 = 6.0
RISK_PCT = 0.01
FEE      = 0.0004   # 0.04 % per side taker


# ─── Internal state ───────────────────────────────────────────────────────────

@dataclass
class _Position:
    direction:  int    # +1 long / -1 short
    entry_ts:   datetime
    entry_px:   float
    qty_full:   float   # total BTC units at entry
    qty_rem:    float   # remaining after partial closes
    sl:         float
    tp1: float; tp2: float; tp3: float
    tp1_hit:    bool = False
    tp2_hit:    bool = False
    acc_gross:  float = 0.0
    acc_fees:   float = 0.0
    entry_fee:  float = 0.0
    worst_px:   float = 0.0   # MAE tracker
    best_px:    float = 0.0   # MFE tracker
    score:      float = 0.0


@dataclass
class CompletedTrade:
    entry_ts:   str
    exit_ts:    str
    direction:  int
    entry_px:   float
    exit_px:    float
    size:       float
    net_pnl:    float
    exit_reason: str
    duration_h: int
    tp1_hit:    bool
    tp2_hit:    bool
    score:      float


# ─── PaperTrader ──────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Parameters
    ----------
    label            : display name
    initial_capital  : starting equity (USDT)
    max_notional_pct : hard cap on notional as fraction of equity (default 1.0 = off)
    vol_target       : annualised vol target (e.g. 0.20); None = off
    dd_halt_pct      : drawdown circuit-breaker threshold; None = off
    min_score        : minimum |composite| to open a trade; None = off
    atr_sl           : ATR multiplier for stop loss (default 2.0)
    """

    def __init__(
        self,
        label: str,
        initial_capital: float = 100_000.0,
        max_notional_pct: float = 1.0,
        vol_target: float | None = None,
        dd_halt_pct: float | None = None,
        min_score: float | None = None,
        atr_sl: float = ATR_SL,
    ) -> None:
        self.label           = label
        self.initial_capital = initial_capital
        self.max_notional    = max_notional_pct
        self.vol_target      = vol_target
        self.dd_halt_pct     = dd_halt_pct
        self.min_score       = min_score
        self.atr_sl          = atr_sl

        self.cash            = float(initial_capital)
        self._pos: Optional[_Position] = None
        self._pending_dir    = 0          # signal direction from prev bar
        self._pending_score  = 0.0
        self._peak_equity    = float(initial_capital)

        self.trades: list[CompletedTrade] = []
        self._equity_history: list[tuple[str, float]] = []   # (iso_ts, equity)

    # ── Public interface ────────────────────────────────────────────────────

    def on_bar(
        self,
        ts:        datetime,
        open_px:   float,
        high_px:   float,
        low_px:    float,
        close_px:  float,
        atr:       float,
        rvol:      float,
        signal:    int,     # -1 / 0 / +1 (after scenario filters)
        composite: float,
    ) -> list[str]:
        """
        Process one closed 1H bar.  Returns list of human-readable event strings.
        Order mirrors the backtest engine:
          1. If in position → check exits with this bar's OHLC
          2. If pending signal → enter at this bar's OPEN
          3. New signal from this bar → store as pending for next bar
          4. Update equity mark
        """
        events: list[str] = []

        # ── 1. Exit checks ──────────────────────────────────────────────────
        if self._pos is not None:
            events += self._check_exits(ts, open_px, high_px, low_px, close_px)

        # ── 2. Pending entry (signal fired on previous bar → execute at open) ─
        if self._pos is None and self._pending_dir != 0:
            ev = self._try_entry(ts, open_px, atr, rvol, self._pending_score)
            if ev:
                events.append(ev)
        self._pending_dir   = 0
        self._pending_score = 0.0

        # ── 3. New signal → queue for next bar ─────────────────────────────
        if self._pos is None and signal != 0:
            self._pending_dir   = signal
            self._pending_score = composite

        # ── 4. Mark equity ──────────────────────────────────────────────────
        eq = self._mark_equity(close_px)
        self._peak_equity = max(self._peak_equity, eq)
        self._equity_history.append((ts.isoformat(), round(eq, 2)))

        return events

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        if self._equity_history:
            return self._equity_history[-1][1]
        return self.cash

    @property
    def pnl(self) -> float:
        return self.equity - self.initial_capital

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.initial_capital * 100

    @property
    def max_drawdown(self) -> float:
        if not self._equity_history:
            return 0.0
        vals = [v for _, v in self._equity_history]
        peak = self.initial_capital
        worst = 0.0
        for v in vals:
            peak  = max(peak, v)
            dd    = (v - peak) / peak
            worst = min(worst, dd)
        return worst * 100    # returns negative %

    @property
    def current_dd(self) -> float:
        """Current drawdown from peak (negative %)."""
        eq = self.equity
        return (eq - self._peak_equity) / self._peak_equity * 100

    @property
    def in_position(self) -> bool:
        return self._pos is not None

    def summary(self) -> str:
        """One-line status string for the dashboard."""
        pos_str = ""
        if self._pos is not None:
            p  = self._pos
            unr = p.direction * p.qty_rem * (
                self._equity_history[-1][1] - p.entry_px
                if self._equity_history else 0.0
            )
            pos_str = (
                f"  {'LONG' if p.direction == 1 else 'SHORT'} "
                f"{p.qty_rem:.4f} BTC @ {p.entry_px:,.0f} "
                f"SL={p.sl:,.0f}  UnrPnL={unr:+,.0f}"
            )
        n = len(self.trades)
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        wr   = wins / n * 100 if n else 0
        return (
            f"{self.label:<32} "
            f"Equity={self.equity:>12,.0f}  "
            f"PnL={self.pnl_pct:>+7.2f}%  "
            f"DD={self.current_dd:>+6.2f}%  "
            f"Trades={n}  WR={wr:.0f}%"
            + pos_str
        )

    def to_dict(self) -> dict:
        """Serialisable state snapshot for JSON persistence."""
        return {
            "label":            self.label,
            "equity":           self.equity,
            "cash":             round(self.cash, 4),
            "pnl_pct":          round(self.pnl_pct, 4),
            "max_drawdown":     round(self.max_drawdown, 4),
            "current_dd":       round(self.current_dd, 4),
            "n_trades":         len(self.trades),
            "in_position":      self.in_position,
            "pending_signal":   self._pending_dir,
            "recent_trades":    [
                {
                    "exit_ts":    t.exit_ts,
                    "direction":  "LONG" if t.direction == 1 else "SHORT",
                    "net_pnl":    round(t.net_pnl, 2),
                    "reason":     t.exit_reason,
                }
                for t in self.trades[-5:]
            ],
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _mark_equity(self, close_px: float) -> float:
        if self._pos is None:
            return self.cash
        p   = self._pos
        unr = p.direction * p.qty_rem * (close_px - p.entry_px)
        return self.cash + unr

    def _try_entry(
        self,
        ts:        datetime,
        open_px:   float,
        atr:       float,
        rvol:      float,
        composite: float,
    ) -> str | None:
        direction = self._pending_dir

        # ── min_score filter ────────────────────────────────────────────────
        if self.min_score is not None and abs(composite) < self.min_score:
            return None

        # ── circuit breaker ─────────────────────────────────────────────────
        size_scale = 1.0
        if self.dd_halt_pct is not None:
            dd_frac = (self.equity - self._peak_equity) / self._peak_equity
            if dd_frac < -(self.dd_halt_pct * 1.5):
                return None     # full pause
            elif dd_frac < -self.dd_halt_pct:
                size_scale = 0.5

        # ── effective size fraction (with vol targeting) ─────────────────────
        eff_pct = RISK_PCT
        if self.vol_target is not None:
            rv         = max(float(rvol), 1e-6)
            vol_scalar = float(np.clip(self.vol_target / rv, 0.25, 3.0))
            eff_pct    = RISK_PCT * vol_scalar

        # ── position sizing ─────────────────────────────────────────────────
        risk_per_unit = self.atr_sl * atr
        if risk_per_unit < 1e-6:
            return None

        qty = (self.cash * eff_pct) / risk_per_unit

        # notional cap
        if self.max_notional < 1.0:
            cap = self.cash * self.max_notional / open_px
            qty = min(qty, cap)

        # leverage guard (1× implicit)
        max_qty = self.cash * 0.95 / open_px
        qty = min(qty, max_qty) * size_scale

        if qty < 1e-8:
            return None

        # ── set stops and targets ───────────────────────────────────────────
        sl_d   = self.atr_sl  * atr
        tp1_d  = ATR_TP1      * atr
        tp2_d  = ATR_TP2      * atr
        tp3_d  = ATR_TP3      * atr
        sign   = direction

        sl  = open_px - sign * sl_d
        tp1 = open_px + sign * tp1_d
        tp2 = open_px + sign * tp2_d
        tp3 = open_px + sign * tp3_d

        entry_fee = qty * open_px * FEE
        self.cash -= entry_fee

        self._pos = _Position(
            direction  = direction,
            entry_ts   = ts,
            entry_px   = open_px,
            qty_full   = qty,
            qty_rem    = qty,
            sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            entry_fee  = entry_fee,
            acc_fees   = entry_fee,
            worst_px   = open_px,
            best_px    = open_px,
            score      = composite,
        )

        side = "LONG" if direction == 1 else "SHORT"
        return (f"ENTER {side} {qty:.4f} BTC @ {open_px:,.0f}  "
                f"SL={sl:,.0f}  TP1={tp1:,.0f}  score={composite:.1f}")

    def _exit_partial(self, exit_px: float, frac: float) -> None:
        p     = self._pos
        q     = p.qty_rem * frac
        gross = p.direction * q * (exit_px - p.entry_px)
        fee   = q * exit_px * FEE
        p.acc_gross += gross
        p.acc_fees  += fee
        self.cash   += gross - fee
        p.qty_rem   -= q

    def _exit_full(self, exit_px: float, reason: str, ts: datetime) -> str:
        p = self._pos
        self._exit_partial(exit_px, 1.0)   # close remainder

        net = p.acc_gross - p.acc_fees

        if p.direction == 1:
            mae = (p.entry_px - p.worst_px) / p.entry_px * 100
            mfe = (p.best_px  - p.entry_px) / p.entry_px * 100
        else:
            mae = (p.worst_px - p.entry_px) / p.entry_px * 100
            mfe = (p.entry_px - p.best_px)  / p.entry_px * 100

        dur = max(0, int((ts - p.entry_ts).total_seconds() // 3600))

        self.trades.append(CompletedTrade(
            entry_ts    = p.entry_ts.isoformat(),
            exit_ts     = ts.isoformat(),
            direction   = p.direction,
            entry_px    = p.entry_px,
            exit_px     = exit_px,
            size        = p.qty_full,
            net_pnl     = round(net, 4),
            exit_reason = reason,
            duration_h  = dur,
            tp1_hit     = p.tp1_hit,
            tp2_hit     = p.tp2_hit,
            score       = p.score,
        ))

        side = "LONG" if p.direction == 1 else "SHORT"
        self._pos = None
        return (f"EXIT {side} @ {exit_px:,.0f}  "
                f"reason={reason}  net={net:+,.2f}  dur={dur}h")

    def _check_exits(
        self,
        ts:       datetime,
        open_px:  float,
        high_px:  float,
        low_px:   float,
        close_px: float,
    ) -> list[str]:
        p      = self._pos
        events: list[str] = []

        # update MAE/MFE trackers
        if p.direction == 1:
            p.worst_px = min(p.worst_px, low_px)
            p.best_px  = max(p.best_px,  high_px)
        else:
            p.worst_px = max(p.worst_px, high_px)
            p.best_px  = min(p.best_px,  low_px)

        if p.direction == 1:    # ─── LONG ──────────────────────────────────

            if low_px <= p.sl:
                events.append(self._exit_full(p.sl, "stop_loss", ts))
                return events

            if high_px >= p.tp3:
                if not p.tp1_hit:
                    self._exit_partial(p.tp1, 0.5); p.tp1_hit = True; p.sl = p.entry_px
                if not p.tp2_hit:
                    self._exit_partial(p.tp2, 0.5); p.tp2_hit = True
                events.append(self._exit_full(p.tp3, "tp3", ts))
                return events

            if high_px >= p.tp2:
                if not p.tp1_hit:
                    self._exit_partial(p.tp1, 0.5); p.tp1_hit = True; p.sl = p.entry_px
                if not p.tp2_hit:
                    self._exit_partial(p.tp2, 0.5); p.tp2_hit = True
                if p.qty_rem > 1e-12:
                    events.append(self._exit_full(p.tp2, "tp2", ts))
                return events

            if high_px >= p.tp1 and not p.tp1_hit:
                self._exit_partial(p.tp1, 0.5)
                p.tp1_hit = True
                p.sl      = p.entry_px   # trail to break-even
                events.append(f"  TP1 hit @ {p.tp1:,.0f}  (50% closed, SL → BE)")

        else:                   # ─── SHORT ─────────────────────────────────

            if high_px >= p.sl:
                events.append(self._exit_full(p.sl, "stop_loss", ts))
                return events

            if low_px <= p.tp3:
                if not p.tp1_hit:
                    self._exit_partial(p.tp1, 0.5); p.tp1_hit = True; p.sl = p.entry_px
                if not p.tp2_hit:
                    self._exit_partial(p.tp2, 0.5); p.tp2_hit = True
                events.append(self._exit_full(p.tp3, "tp3", ts))
                return events

            if low_px <= p.tp2:
                if not p.tp1_hit:
                    self._exit_partial(p.tp1, 0.5); p.tp1_hit = True; p.sl = p.entry_px
                if not p.tp2_hit:
                    self._exit_partial(p.tp2, 0.5); p.tp2_hit = True
                if p.qty_rem > 1e-12:
                    events.append(self._exit_full(p.tp2, "tp2", ts))
                return events

            if low_px <= p.tp1 and not p.tp1_hit:
                self._exit_partial(p.tp1, 0.5)
                p.tp1_hit = True
                p.sl      = p.entry_px
                events.append(f"  TP1 hit @ {p.tp1:,.0f}  (50% closed, SL → BE)")

        return events
