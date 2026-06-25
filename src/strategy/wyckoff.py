"""
Wyckoff Pattern Signal Generator — BTCUSDT 1H.

Methodology
───────────────────────────────────────────────────────────────────────────────
1. Range Detection
   Trading range = ADX < adx_max AND price bandwidth < range_width_max.
   Boundaries: rolling max(high) and min(low) over the last n_range bars,
   shifted by 1 bar to avoid look-ahead.

2. Volume Bias
   Accumulation : ranging + OBV above its 21-bar EMA  (smart money absorbing)
   Distribution : ranging + OBV below its 21-bar EMA  (smart money selling)

3. Manipulation Events
   Spring   : bar's low pierces range_low AND closes back inside range
              on high volume (vol_ratio ≥ threshold), in accumulation context.
   Upthrust : bar's high pierces range_high AND closes back inside range
              on high volume, in distribution context.

4. Entry
   Signal fires on the bar AFTER the manipulation event (next-bar open
   execution — no look-ahead).

5. Composite Score (positive for longs, negative for shorts)
   = ±(penetration_depth / ATR) × vol_ratio  × 2
   Higher score = deeper spring/upthrust + larger volume surge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_wyckoff_signals(
    df_1h: pd.DataFrame,
    n_range: int = 24,
    adx_max: float = 25.0,
    range_width_max: float = 0.10,
    vol_threshold: float = 1.3,
    session_hours: tuple[int, int] | None = (8, 21),
) -> pd.DataFrame:
    """
    Build Wyckoff spring/upthrust signals from enriched 1H OHLCV data.

    Parameters
    ----------
    df_1h           : DataFrame with OHLCV + indicators from add_indicators()
    n_range         : look-back bars for range boundary detection (default 24 = 1 day)
    adx_max         : ADX ceiling for "ranging" classification (default 25)
    range_width_max : max (range_high − range_low) / range_low fraction (default 0.10)
    vol_threshold   : vol_ratio required to confirm spring/upthrust (default 1.3)
    session_hours   : (start_h, end_h) UTC session filter; None = no filter

    Returns
    -------
    pd.DataFrame  columns: signal (int), composite (float), regime (str)
                  index aligned to df_1h.index
    """
    idx = df_1h.index
    hi  = df_1h["high"]
    lo  = df_1h["low"]
    cl  = df_1h["close"]
    atr = (df_1h["atr_14"].clip(lower=1.0)
           if "atr_14" in df_1h.columns
           else pd.Series(1.0, index=idx))

    # ── 1. Range boundaries (shift by 1 → zero look-ahead) ──────────────────
    min_periods = max(n_range // 2, 5)
    range_high  = hi.shift(1).rolling(n_range, min_periods=min_periods).max()
    range_low   = lo.shift(1).rolling(n_range, min_periods=min_periods).min()
    range_width = (range_high - range_low) / range_low.clip(lower=1.0)

    # ── 2. Ranging regime ────────────────────────────────────────────────────
    adx_col    = (df_1h["adx"]
                  if "adx" in df_1h.columns
                  else pd.Series(50.0, index=idx))
    is_ranging = (adx_col < adx_max) & (range_width < range_width_max) & range_high.notna()

    # ── 3. Volume bias (OBV vs its 21-bar EMA) ───────────────────────────────
    if "obv_trend" in df_1h.columns:
        obv_trend = df_1h["obv_trend"]
    elif "obv" in df_1h.columns and "obv_ema21" in df_1h.columns:
        obv_trend = np.sign(df_1h["obv"] - df_1h["obv_ema21"])
    else:
        obv_trend = pd.Series(0.0, index=idx)

    accum_bias  = is_ranging & (obv_trend > 0)
    distrib_bias = is_ranging & (obv_trend < 0)

    vol_ratio = (df_1h["vol_ratio"]
                 if "vol_ratio" in df_1h.columns
                 else pd.Series(1.0, index=idx))

    # ── 4. Manipulation events ────────────────────────────────────────────────
    is_spring = (
        is_ranging
        & accum_bias
        & (lo  < range_low)
        & (cl  >= range_low)
        & (vol_ratio >= vol_threshold)
    )

    is_upthrust = (
        is_ranging
        & distrib_bias
        & (hi  > range_high)
        & (cl  <= range_high)
        & (vol_ratio >= vol_threshold)
    )

    # ── 5. Signals fire on the next bar ──────────────────────────────────────
    long_trigger  = is_spring.shift(1).fillna(False).astype(bool)
    short_trigger = is_upthrust.shift(1).fillna(False).astype(bool)

    # ── 6. Composite score ────────────────────────────────────────────────────
    spring_depth    = ((range_low  - lo) / atr).clip(0.2, 4.0)
    upthrust_height = ((hi - range_high) / atr).clip(0.2, 4.0)
    vol_factor      = vol_ratio.clip(1.0, 4.0)

    # Shift event-bar scores to signal bar (next bar)
    score_long  = (spring_depth    * vol_factor * 2.0).shift(1).fillna(0.0)
    score_short = (upthrust_height * vol_factor * 2.0).shift(1).fillna(0.0)

    # ── 7. Assemble output ────────────────────────────────────────────────────
    signal    = pd.Series(0,   index=idx, dtype=int)
    composite = pd.Series(0.0, index=idx)

    signal[long_trigger]     =  1
    signal[short_trigger]    = -1
    composite[long_trigger]  =  score_long[long_trigger].clip(lower=0.1)
    composite[short_trigger] = -score_short[short_trigger].clip(lower=0.1)

    # Rare same-bar conflict → flatten
    conflict = long_trigger & short_trigger
    signal[conflict]    = 0
    composite[conflict] = 0.0

    # ── 8. Session filter ─────────────────────────────────────────────────────
    if session_hours is not None:
        h_s, h_e   = session_hours
        not_session = ~((idx.hour >= h_s) & (idx.hour < h_e))
        signal[not_session]    = 0
        composite[not_session] = 0.0

    # ── 9. Regime label ───────────────────────────────────────────────────────
    regime = pd.Series("trending", index=idx)
    regime[is_ranging] = "ranging"

    return pd.DataFrame(
        {"signal": signal, "composite": composite, "regime": regime},
        index=idx,
    )
