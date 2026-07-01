"""
Donchian Channel Turtle Breakout — BTCUSDT 1H.

Classic Turtle Trading rules adapted for perpetual futures:
  Entry long  : close crosses ABOVE rolling max of last n_entry bars
  Entry short : close crosses BELOW rolling min of last n_entry bars

Uses the same ATR-based SL/TP as the composite system via run_backtest().
Crossing is one-bar only (signal fires on the bar of the breakout, not
on every subsequent bar above/below the channel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_turtle_signals(
    df_1h: pd.DataFrame,
    n_entry: int = 20,
    session_hours: tuple = (8, 21),
) -> pd.DataFrame:
    """
    Build Donchian breakout signal DataFrame compatible with run_backtest().

    Parameters
    ----------
    df_1h         : 1H OHLCV DataFrame with atr_14 column.
    n_entry       : lookback bars for channel high/low (default 20).
    session_hours : (start, end) UTC hours for session filter.

    Returns
    -------
    DataFrame with columns: signal (int), composite (float).
    """
    close = df_1h["close"]

    # Shift by 1: bar T sees max/min of bars T-n .. T-1 (no lookahead)
    roll_max = close.shift(1).rolling(n_entry, min_periods=n_entry).max()
    roll_min = close.shift(1).rolling(n_entry, min_periods=n_entry).min()

    # Breakout: current close crosses the channel
    broke_up   = close > roll_max
    broke_down = close < roll_min

    # Session filter
    in_session = (df_1h.index.hour >= session_hours[0]) & (df_1h.index.hour < session_hours[1])

    sig = pd.Series(0, index=df_1h.index)
    sig[in_session & broke_up]   =  1
    sig[in_session & broke_down] = -1

    # Composite proxy: distance from channel mid, normalised by channel width
    channel_mid   = (roll_max + roll_min) / 2.0
    channel_width = (roll_max - roll_min).replace(0, np.nan)
    composite = ((close - channel_mid) / (channel_width / 2.0)).fillna(0.0) * sig.abs()
    composite = composite.clip(-10, 10)

    out = pd.DataFrame({"signal": sig.astype(int), "composite": composite}, index=df_1h.index)
    return out
