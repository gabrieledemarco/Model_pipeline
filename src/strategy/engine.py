"""
BTCUSDT multi-timeframe strategy backtester.

Execution model
───────────────
• Signals are generated on bar *close* (end-of-bar, no look-ahead).
• Entry executes at the *next* bar *open* (realistic: you can't trade the
  candle that produced the signal).
• Risk-based position sizing: 1 % of current equity per trade.
• Three-tier profit target:
    TP1 (½ position) : entry ± 2 × ATR  →  R/R 1:1
    TP2 (¼ position) : entry ± 4 × ATR  →  R/R 1:2
    TP3 (¼ position) : entry ± 6 × ATR  →  R/R 1:3
• Stop loss at 2 × ATR (worst-case scenario order).
• After TP1 hit: stop moves to break-even on remaining.
• Fees: 0.04 % per side (Binance USDT-perp taker).
• Maximum one position at a time (no pyramiding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────
ATR_SL   = 2.0    # stop distance as ATR multiple
ATR_TP1  = 2.0    # 1st target (close 50 %)
ATR_TP2  = 4.0    # 2nd target (close 25 %)
ATR_TP3  = 6.0    # 3rd target / runner (close 25 %)
RISK_PCT = 0.01   # risk per trade as fraction of equity
FEE      = 0.0004 # 0.04 % one-way taker fee
INIT_CAP = 100_000.0


@dataclass
class Trade:
    entry_ts:     pd.Timestamp
    exit_ts:      Optional[pd.Timestamp] = None
    direction:    int   = 0            # +1 long / -1 short
    entry_price:  float = 0.0
    exit_price:   float = 0.0
    stop_price:   float = 0.0
    tp1_price:    float = 0.0
    tp2_price:    float = 0.0
    tp3_price:    float = 0.0
    size:         float = 0.0          # BTC units
    notional:     float = 0.0          # USDT at entry
    gross_pnl:    float = 0.0
    total_fees:   float = 0.0
    net_pnl:      float = 0.0
    exit_reason:  str   = ""
    score:        float = 0.0
    regime:       str   = ""
    duration_h:   int   = 0
    tp1_hit:      bool  = False
    tp2_hit:      bool  = False
    mae_pct:      float = 0.0          # max adverse excursion (%, positive)
    mfe_pct:      float = 0.0          # max favorable excursion (%, positive)


def run_backtest(
    df_1h: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = INIT_CAP,
    atr_sl_override: float | None = None,
    sizing_method: str = "fixed_risk",
    size_pct: float | None = None,
    leverage: float = 1.0,
) -> dict:
    """
    Event-driven backtest on 1H OHLCV bars.

    Parameters
    ----------
    df_1h : pd.DataFrame
        1H bars with columns [open, high, low, close] + all indicators.
    signals : pd.DataFrame
        Aligned signal matrix from signals.build_signal_matrix().
    initial_capital : float
    atr_sl_override : float | None
        Override ATR_SL multiplier.
    sizing_method : str
        "fixed_risk"     – size so that the SL costs exactly *size_pct* of equity.
        "fixed_fraction" – invest *size_pct* × leverage of equity per trade (notional).
    size_pct : float | None
        Fraction of equity used for sizing (default: RISK_PCT = 0.01).
    leverage : float
        Maximum position notional as a multiple of equity (default: 1.0).
        Caps position to equity × leverage / price.  No margin-call / liquidation
        modelling – the ATR stop-loss is assumed to execute without slippage.

    Returns
    -------
    dict  with keys: equity, drawdown, trades, kpis
    """
    _atr_sl   = atr_sl_override if atr_sl_override is not None else ATR_SL
    _size_pct = size_pct if size_pct is not None else RISK_PCT
    _leverage = max(float(leverage), 1.0)

    n = len(df_1h)
    equity_arr = np.full(n, float(initial_capital))
    cash = float(initial_capital)

    o   = df_1h["open"].to_numpy(float)
    h   = df_1h["high"].to_numpy(float)
    l   = df_1h["low"].to_numpy(float)
    c   = df_1h["close"].to_numpy(float)
    atr = df_1h["atr_14"].to_numpy(float)

    sig_arr   = signals["signal"].to_numpy(int)
    comp_arr  = signals["composite"].to_numpy(float)
    regime_arr = signals["regime"].to_numpy(object) if "regime" in signals.columns \
                 else np.full(n, "unknown", dtype=object)

    trades: List[Trade] = []

    # ── Active position state ────────────────────────────────────────────────
    IN_POS   = False
    direction = 0
    size_full = 0.0          # full BTC units
    size_rem  = 0.0          # remaining BTC (after partial closes)
    ep        = 0.0          # entry price
    sl        = 0.0
    tp1, tp2, tp3 = 0.0, 0.0, 0.0
    tp1_hit = tp2_hit = False
    entry_ts  = None
    entry_i   = 0
    score     = 0.0
    regime    = ""
    notional  = 0.0
    entry_fee = 0.0
    acc_fees  = 0.0
    acc_gross = 0.0
    worst     = 0.0          # tracks adverse extreme price
    best      = 0.0          # tracks favorable extreme price

    def _mark() -> float:
        if not IN_POS:
            return cash
        mark = c[i]
        return cash + direction * size_rem * (mark - ep)

    def _exit_partial(exit_px: float, frac: float) -> float:
        """Close *frac* of remaining position. Returns net PnL of this leg."""
        nonlocal size_rem, cash, acc_gross, acc_fees
        q = size_rem * frac
        gross = direction * q * (exit_px - ep)
        fee   = q * exit_px * FEE
        acc_gross += gross
        acc_fees  += fee
        cash      += gross - fee
        size_rem  -= q
        return gross - fee

    def _exit_full(exit_px: float, reason: str, bar_i: int):
        nonlocal IN_POS, direction, size_rem, size_full, cash
        nonlocal ep, sl, tp1, tp2, tp3, tp1_hit, tp2_hit
        nonlocal entry_ts, entry_i, score, regime, notional, entry_fee
        nonlocal acc_gross, acc_fees, worst, best

        # close remainder
        _exit_partial(exit_px, 1.0)

        # MAE / MFE as % of entry
        if direction == 1:
            mae = (ep - worst) / ep * 100
            mfe = (best  - ep) / ep * 100
        else:
            mae = (worst - ep) / ep * 100
            mfe = (ep   - best) / ep * 100

        total_fees = acc_fees + entry_fee
        net = acc_gross - total_fees

        trades.append(Trade(
            entry_ts    = entry_ts,
            exit_ts     = df_1h.index[bar_i],
            direction   = direction,
            entry_price = ep,
            exit_price  = exit_px,
            stop_price  = sl,
            tp1_price   = tp1,
            tp2_price   = tp2,
            tp3_price   = tp3,
            size        = size_full,
            notional    = notional,
            gross_pnl   = acc_gross,
            total_fees  = total_fees,
            net_pnl     = net,
            exit_reason = reason,
            score       = score,
            regime      = regime,
            duration_h  = bar_i - entry_i,
            tp1_hit     = tp1_hit,
            tp2_hit     = tp2_hit,
            mae_pct     = max(mae, 0.0),
            mfe_pct     = max(mfe, 0.0),
        ))

        # reset
        IN_POS  = False
        direction = 0
        size_rem = size_full = 0.0
        tp1_hit = tp2_hit = False
        acc_gross = acc_fees = 0.0

    # ── Main loop ────────────────────────────────────────────────────────────
    for i in range(1, n):

        if IN_POS:
            # update extreme trackers
            if direction == 1:
                worst = min(worst, l[i])
                best  = max(best,  h[i])
            else:
                worst = max(worst, h[i])
                best  = min(best,  l[i])

            # ── Check exits (worst-case order) ────────────────────────────
            if direction == 1:        # ─── LONG ───
                if l[i] <= sl:
                    _exit_full(sl, "stop_loss", i); continue

                # TP3 direct hit (skip TP1/TP2)
                if h[i] >= tp3:
                    if not tp1_hit:
                        _exit_partial(tp1, 0.5)
                        tp1_hit = True
                        sl = ep          # move to BE
                    if not tp2_hit:
                        _exit_partial(tp2, 0.5)
                        tp2_hit = True
                    _exit_full(tp3, "tp3", i); continue

                if h[i] >= tp2:
                    if not tp1_hit:
                        _exit_partial(tp1, 0.5)
                        tp1_hit = True
                        sl = ep
                    if not tp2_hit:
                        _exit_partial(tp2, 0.5)
                        tp2_hit = True
                    if size_rem > 1e-12:
                        _exit_full(tp2, "tp2", i)
                    continue

                if h[i] >= tp1 and not tp1_hit:
                    _exit_partial(tp1, 0.5)
                    tp1_hit = True
                    sl = ep              # trail stop to BE

            else:                     # ─── SHORT ───
                if h[i] >= sl:
                    _exit_full(sl, "stop_loss", i); continue

                if l[i] <= tp3:
                    if not tp1_hit:
                        _exit_partial(tp1, 0.5)
                        tp1_hit = True
                        sl = ep
                    if not tp2_hit:
                        _exit_partial(tp2, 0.5)
                        tp2_hit = True
                    _exit_full(tp3, "tp3", i); continue

                if l[i] <= tp2:
                    if not tp1_hit:
                        _exit_partial(tp1, 0.5)
                        tp1_hit = True
                        sl = ep
                    if not tp2_hit:
                        _exit_partial(tp2, 0.5)
                        tp2_hit = True
                    if size_rem > 1e-12:
                        _exit_full(tp2, "tp2", i)
                    continue

                if l[i] <= tp1 and not tp1_hit:
                    _exit_partial(tp1, 0.5)
                    tp1_hit = True
                    sl = ep

        # ── New position entry ────────────────────────────────────────────
        if not IN_POS and sig_arr[i - 1] != 0:
            new_sig   = int(sig_arr[i - 1])
            entry_px  = o[i]            # fill at next-bar open
            curr_atr  = atr[i - 1]

            if curr_atr > 0 and entry_px > 0:
                # leverage-adjusted max notional
                max_qty = cash * _leverage * 0.95 / entry_px

                if sizing_method == "fixed_fraction":
                    # invest _size_pct × leverage of equity as notional
                    qty = cash * _size_pct * _leverage / entry_px
                else:  # fixed_risk (default)
                    # size so the SL costs exactly _size_pct × equity
                    risk_per_unit = curr_atr * _atr_sl
                    qty = (cash * _size_pct) / risk_per_unit

                qty = min(qty, max_qty)
                qty = max(qty, 1e-12)

                IN_POS    = True
                direction = new_sig
                ep        = entry_px
                size_full = qty
                size_rem  = qty
                notional  = qty * entry_px
                entry_fee = notional * FEE
                cash     -= entry_fee
                entry_ts  = df_1h.index[i]
                entry_i   = i
                score     = float(comp_arr[i - 1])
                regime    = str(regime_arr[i - 1])
                acc_gross = acc_fees = 0.0
                tp1_hit = tp2_hit = False

                if new_sig == 1:
                    sl  = ep - curr_atr * _atr_sl
                    tp1 = ep + curr_atr * ATR_TP1
                    tp2 = ep + curr_atr * ATR_TP2
                    tp3 = ep + curr_atr * ATR_TP3
                    worst = ep; best = ep
                else:
                    sl  = ep + curr_atr * _atr_sl
                    tp1 = ep - curr_atr * ATR_TP1
                    tp2 = ep - curr_atr * ATR_TP2
                    tp3 = ep - curr_atr * ATR_TP3
                    worst = ep; best = ep

        equity_arr[i] = _mark()

    # close any open position at last bar
    if IN_POS:
        _exit_full(c[-1], "eob", n - 1)

    equity  = pd.Series(equity_arr, index=df_1h.index)
    running_max = equity.cummax()
    dd_abs = equity - running_max
    dd_pct = dd_abs / running_max

    trades_df = (pd.DataFrame([t.__dict__ for t in trades])
                 if trades else pd.DataFrame())

    return {
        "equity":   equity,
        "drawdown": dd_pct,
        "trades":   trades_df,
        "kpis":     _compute_kpis(equity, dd_pct, trades_df, initial_capital),
    }


def _compute_kpis(equity: pd.Series, drawdown: pd.Series,
                  trades_df: pd.DataFrame, init_cap: float) -> dict:
    """Comprehensive performance metrics."""
    final   = float(equity.iloc[-1])
    tot_ret = final / init_cap - 1.0

    bar_ret = equity.pct_change().dropna()
    hrs_per_year = 24 * 365

    ann_vol = float(bar_ret.std() * np.sqrt(hrs_per_year))
    ann_ret = float(bar_ret.mean() * hrs_per_year)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    neg = bar_ret[bar_ret < 0]
    sortino = (ann_ret / (neg.std() * np.sqrt(hrs_per_year))
               if len(neg) > 1 else 0.0)

    max_dd  = float(drawdown.min())
    calmar  = (tot_ret / abs(max_dd)) if max_dd < 0 else 0.0

    if len(trades_df) == 0:
        return dict(total_return=tot_ret, final_equity=final,
                    sharpe=sharpe, sortino=sortino, calmar=calmar,
                    max_drawdown=max_dd, n_trades=0,
                    win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                    avg_duration_h=0, expectancy=0)

    wins     = trades_df["net_pnl"] > 0
    n        = len(trades_df)
    n_w      = int(wins.sum())
    n_l      = n - n_w
    wr       = n_w / n

    avg_w    = float(trades_df.loc[wins,  "net_pnl"].mean()) if n_w else 0.0
    avg_l    = float(trades_df.loc[~wins, "net_pnl"].mean()) if n_l else 0.0
    pf       = (avg_w * n_w / (-avg_l * n_l)) if (n_l > 0 and avg_l < 0) else 0.0
    exp      = wr * avg_w + (1 - wr) * avg_l

    avg_dur  = float(trades_df["duration_h"].mean()) if n else 0.0

    return dict(
        total_return=tot_ret,
        final_equity=final,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        n_trades=n,
        n_wins=n_w,
        n_losses=n_l,
        win_rate=wr,
        avg_win=avg_w,
        avg_loss=avg_l,
        profit_factor=pf,
        expectancy=exp,
        avg_duration_h=avg_dur,
        total_fees=float(trades_df["total_fees"].sum()),
    )
