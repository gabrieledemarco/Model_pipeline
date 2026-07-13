"""
Causal multi-timeframe swing/trend detection.

Identifies pivot highs/lows on a higher-timeframe (HTF) series using a
classic fractal definition (left/right bars), determines trend direction
from the HH/HL vs LH/LL pivot sequence, and tracks the nearest *unbroken*
prior pivot in the trade direction — used as a price target by systems
that want to "ride a lower-timeframe trend start back to the last HTF
swing extreme."

Causality: a pivot at bar i is only known (usable) starting from bar
i+right (it takes `right` bars of subsequent price action to confirm a
local extreme). All derived state (trend_state, target_high, target_low)
is exposed per-bar as of when it become knowable — safe to merge onto a
finer timeframe with a plain as-of join without introducing look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int) -> list[dict]:
    """
    Fractal pivot detection. Returns a list of dicts sorted by confirm_idx:
        bar_idx     : index of the actual extreme bar
        confirm_idx : index at which the pivot becomes known (bar_idx+right)
        price       : high or low value at the pivot
        kind        : +1 for pivot high, -1 for pivot low
    """
    n = len(high)
    pivots = []
    for i in range(left, n - right):
        window_h = high[i - left:i + right + 1]
        if high[i] == window_h.max() and np.argmax(window_h) == left:
            pivots.append(dict(bar_idx=i, confirm_idx=i + right, price=high[i], kind=1))
        window_l = low[i - left:i + right + 1]
        if low[i] == window_l.min() and np.argmin(window_l) == left:
            pivots.append(dict(bar_idx=i, confirm_idx=i + right, price=low[i], kind=-1))
    pivots.sort(key=lambda p: p["confirm_idx"])
    return pivots


def causal_trend_state(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    left: int = 5,
    right: int = 5,
) -> pd.DataFrame:
    """
    Build a per-bar causal trend-state table from pivot structure.

    Returns a DataFrame (same length as input) with columns:
        trend_state : +1 bullish (HH & HL), -1 bearish (LH & LL), 0 otherwise
        target_high  : nearest confirmed pivot-high price still above the
                       bar's own close (NaN if none / already broken)
        target_low   : nearest confirmed pivot-low price still below the
                       bar's own close (NaN if none / already broken)
        last_pivot_high, last_pivot_low : most recent confirmed pivot price
                       of each kind regardless of whether broken (for
                       diagnostics/plotting)

    All values at bar i only use information confirmable by bar i (no
    look-ahead) — pivots with confirm_idx > i are not yet visible.
    """
    n = len(close)
    pivots = find_pivots(high, low, left, right)

    trend_state = np.zeros(n, dtype=int)
    target_high = np.full(n, np.nan)
    target_low  = np.full(n, np.nan)
    last_ph_arr = np.full(n, np.nan)
    last_pl_arr = np.full(n, np.nan)

    highs_seen: list[float] = []   # confirmed pivot-high prices, in time order
    lows_seen:  list[float] = []   # confirmed pivot-low prices, in time order
    state = 0
    p_idx = 0
    n_pivots = len(pivots)

    for i in range(n):
        # Absorb any pivots confirmed exactly at this bar
        while p_idx < n_pivots and pivots[p_idx]["confirm_idx"] == i:
            piv = pivots[p_idx]
            if piv["kind"] == 1:
                highs_seen.append(piv["price"])
            else:
                lows_seen.append(piv["price"])
            p_idx += 1

            if len(highs_seen) >= 2 and len(lows_seen) >= 2:
                hh = highs_seen[-1] > highs_seen[-2]
                hl = lows_seen[-1]  > lows_seen[-2]
                lh = highs_seen[-1] < highs_seen[-2]
                ll = lows_seen[-1]  < lows_seen[-2]
                if hh and hl:
                    state = 1
                elif lh and ll:
                    state = -1
                # mixed (HH & LL, or LH & HL) -> keep previous state (transitional)

        trend_state[i] = state
        if highs_seen:
            last_ph_arr[i] = highs_seen[-1]
            # nearest unbroken pivot high: scan backward for the first one
            # still above current close
            for ph in reversed(highs_seen):
                if ph > close[i]:
                    target_high[i] = ph
                    break
        if lows_seen:
            last_pl_arr[i] = lows_seen[-1]
            for pl in reversed(lows_seen):
                if pl < close[i]:
                    target_low[i] = pl
                    break

    return pd.DataFrame({
        "trend_state": trend_state,
        "target_high": target_high,
        "target_low": target_low,
        "last_pivot_high": last_ph_arr,
        "last_pivot_low": last_pl_arr,
    })


def align_htf_to_ltf(htf_index: pd.DatetimeIndex, htf_df: pd.DataFrame,
                      ltf_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    As-of merge: each LTF timestamp inherits the state of the most recently
    *closed* HTF bar strictly at or before it. Causal by construction since
    htf_df's own columns are already only using information knowable as of
    each HTF bar's close.
    """
    htf = htf_df.copy()
    # rename_axis(None) before reset_index(): callers may pass an index with
    # a name (e.g. "open_time" from src.live.data_live's klines fetchers),
    # which would make reset_index() emit that name as the column instead of
    # "index", leaving the rename below a no-op and merge_asof's on="ts"
    # unresolvable.
    htf.index = htf_index
    # merge_asof requires both join keys to share the exact same datetime64
    # unit (ms/us/ns). Callers routinely build htf_index via arithmetic like
    # `some_index + pd.Timedelta(days=1)` (the day+1 causality trick used by
    # volume_profile.py / the vol-regime daily bucket), which silently
    # upcasts the result to a different unit than ltf_index — normalize here
    # once so every caller doesn't have to remember to cast back.
    if htf.index.dtype != ltf_index.dtype:
        htf.index = htf.index.astype(ltf_index.dtype)
    ltf_frame = pd.DataFrame(index=ltf_index).rename_axis(None)
    merged = pd.merge_asof(
        ltf_frame.reset_index().rename(columns={"index": "ts"}),
        htf.rename_axis(None).reset_index().rename(columns={"index": "ts"}),
        on="ts", direction="backward",
    )
    merged.index = ltf_index
    return merged.drop(columns=["ts"])
