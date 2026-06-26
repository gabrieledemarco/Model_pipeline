"""
orb_ict.py — ICT-style Asian Range Sweep + ORB Signal Generator
================================================================

Multi-timeframe logic:
  1H  → computes Asian Range (00:00–06:59 UTC) + NR compression filter
  15M → detects London KZ (07:00–09:59 UTC) sweep of Asian levels
        and NY KZ (13:00–15:59 UTC) sweep of full-session levels

Signal definition:
  LONG  : 15M bar low dips below Asian Low AND bar closes back above AL
           (false breakout below support in accumulation context)
  SHORT : 15M bar high pierces Asian High AND bar closes back below AH
           (false breakout above resistance in distribution context)

Entry  : next 15M bar open (zero look-ahead — signal on bar close,
          execution on next open)
Score  : compression_ratio × penetration_depth / ATR_15M × 2
          compression_ratio = rolling_20d_median_asian_range / today_asian_range
          (higher when today is unusually tight = Crabel NR logic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Session windows (UTC hours, half-open [start, end))
# ─────────────────────────────────────────────────────────────────────────────
ASIA_START   = 0    # 00:00 UTC
ASIA_END     = 7    # 06:59 UTC  (7 bars of 1H = 28 bars of 15M)
LONDON_START = 7    # 07:00 UTC
LONDON_END   = 10   # 09:59 UTC  (3 bars of 1H = 12 bars of 15M)
NY_START     = 13   # 13:00 UTC
NY_END       = 16   # 15:59 UTC  (3 bars of 1H = 12 bars of 15M)


def build_asian_range(df_1h: pd.DataFrame,
                      nr_lookback: int = 20) -> pd.DataFrame:
    """
    Compute the daily Asian session range from 1H bars.

    Returns a DataFrame indexed by DATE (not datetime) with columns:
        asian_high      : max(high) of bars 00-06 UTC
        asian_low       : min(low)  of bars 00-06 UTC
        asian_range     : asian_high - asian_low (absolute)
        asian_range_pct : asian_range / asian_low * 100
        atr_1h          : mean ATR(14) of Asian session bars (proxy for daily ATR)
        nr_rank         : percentile rank of today's range in last nr_lookback days
        is_nr           : True if range is in bottom 25% (NR filter = Crabel Q1)
        compression_ratio: rolling median / today range (score multiplier)
    """
    mask  = (df_1h.index.hour >= ASIA_START) & (df_1h.index.hour < ASIA_END)
    df_as = df_1h[mask].copy()
    df_as["date"] = df_as.index.date

    def agg_day(g):
        g = g.sort_index()
        return pd.Series({
            "asian_high":   g["high"].max(),
            "asian_low":    g["low"].min(),
            "atr_1h":       g["atr_14"].mean() if "atr_14" in g.columns else np.nan,
        })

    daily = (df_as.groupby("date")
               .apply(agg_day, include_groups=False)
               .reset_index())
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()
    # Only keep complete sessions
    bar_cnt = df_as.groupby("date").size()
    bar_cnt.index = pd.to_datetime(bar_cnt.index)
    daily = daily[bar_cnt >= (ASIA_END - ASIA_START)].copy()

    daily["asian_range"]     = daily["asian_high"] - daily["asian_low"]
    daily["asian_range_pct"] = daily["asian_range"] / daily["asian_low"].clip(lower=1) * 100

    # Rolling NR percentile (over last nr_lookback days)
    roll_med = daily["asian_range"].rolling(nr_lookback, min_periods=max(nr_lookback//2, 5)).median()
    roll_q25 = daily["asian_range"].rolling(nr_lookback, min_periods=max(nr_lookback//2, 5)).quantile(0.25)

    daily["nr_rank"]          = daily["asian_range"].rolling(nr_lookback, min_periods=5).rank(pct=True)
    daily["is_nr"]            = daily["asian_range"] <= roll_q25
    daily["compression_ratio"] = (roll_med / daily["asian_range"].clip(lower=1)).clip(0.5, 4.0)

    return daily


def _detect_sweeps_in_window(df_15m: pd.DataFrame,
                              asian_daily: pd.DataFrame,
                              window_start_h: int,
                              window_end_h: int,
                              atr_threshold: float = 1.5,
                              ) -> pd.DataFrame:
    """
    For each 15M bar in [window_start_h, window_end_h) UTC:
      - LONG signal  : bar.low < asian_low  AND bar.close >= asian_low
      - SHORT signal : bar.high > asian_high AND bar.close <= asian_high

    The bar's date key is used to look up the Asian Range for THAT day.
    Signal is placed on the event bar; entry is next bar open (shift(1) done
    in the caller).

    Returns DataFrame aligned to df_15m.index with columns:
        signal     : raw event bar signal (1, -1, 0) — NOT yet shifted
        composite  : event bar composite score
        asian_high : reference level
        asian_low  : reference level
        is_nr      : whether today is an NR day
    """
    idx = df_15m.index
    mask_win = (idx.hour >= window_start_h) & (idx.hour < window_end_h)

    hi  = df_15m["high"]
    lo  = df_15m["low"]
    cl  = df_15m["close"]
    atr = (df_15m["atr_14"].clip(lower=1.0)
           if "atr_14" in df_15m.columns
           else pd.Series(1.0, index=idx))

    # Broadcast daily Asian levels to 15M bars (by date)
    date_idx = idx.normalize()                # midnight of each bar's day
    ah_series = date_idx.map(
        asian_daily["asian_high"].to_dict()).astype(float)
    al_series = date_idx.map(
        asian_daily["asian_low"].to_dict()).astype(float)
    cr_series = date_idx.map(
        asian_daily["compression_ratio"].fillna(1.0).to_dict()).astype(float)
    nr_series = date_idx.map(
        asian_daily["is_nr"].fillna(False).to_dict()).astype(bool)

    ah = pd.Series(ah_series.values, index=idx)
    al = pd.Series(al_series.values, index=idx)
    cr = pd.Series(cr_series.values, index=idx)
    nr = pd.Series(nr_series.values, index=idx)

    valid = mask_win & ah.notna() & al.notna()

    # Penetration depth
    long_pen  = (al - lo).clip(lower=0)   # how far below AL (positive)
    short_pen = (hi - ah).clip(lower=0)   # how far above AH (positive)

    is_long_sweep  = valid & (lo  < al) & (cl >= al) & (long_pen  > 0)
    is_short_sweep = valid & (hi  > ah) & (cl <= ah) & (short_pen > 0)

    # Composite score: compression × depth/ATR × 2
    score_long  = (cr * (long_pen  / atr).clip(0.1, 4.0) * 2.0)
    score_short = (cr * (short_pen / atr).clip(0.1, 4.0) * 2.0)

    signal    = pd.Series(0,   index=idx, dtype=int)
    composite = pd.Series(0.0, index=idx)

    signal[is_long_sweep]     =  1
    signal[is_short_sweep]    = -1
    composite[is_long_sweep]  =  score_long[is_long_sweep].clip(lower=0.1)
    composite[is_short_sweep] = -score_short[is_short_sweep].clip(lower=0.1)

    # Conflict: same bar gets both → flatten
    conflict = is_long_sweep & is_short_sweep
    signal[conflict]    = 0
    composite[conflict] = 0.0

    return pd.DataFrame({
        "signal":     signal,
        "composite":  composite,
        "asian_high": ah,
        "asian_low":  al,
        "is_nr":      nr,
    }, index=idx)


def build_orb_ict_signals(
    df_15m: pd.DataFrame,
    asian_daily: pd.DataFrame,
    use_london_kz: bool = True,
    use_ny_kz: bool = False,
    nr_filter: bool = True,
) -> pd.DataFrame:
    """
    Build ICT-style Asian Range Sweep signals on 15M data.

    Parameters
    ----------
    df_15m        : 15M OHLCV DataFrame with indicators (atr_14 required)
    asian_daily   : output of build_asian_range() — daily Asian Range levels
    use_london_kz : include London KZ sweeps (07:00–09:59 UTC)
    use_ny_kz     : include NY KZ sweeps (13:00–15:59 UTC)
    nr_filter     : if True, only emit signals on NR (compression) days

    Returns
    -------
    pd.DataFrame with columns: signal (int), composite (float), regime (str)
    Index aligned to df_15m.index. Signal fires on event bar; engine must
    execute on the NEXT bar (shift handled here via .shift(1)).
    """
    idx = df_15m.index
    signal_raw    = pd.Series(0,   index=idx, dtype=int)
    composite_raw = pd.Series(0.0, index=idx)
    nr_mask_raw   = pd.Series(False, index=idx)

    if use_london_kz:
        sw = _detect_sweeps_in_window(
            df_15m, asian_daily, LONDON_START, LONDON_END)
        # merge: take first non-zero signal if both KZ active
        mask = sw["signal"] != 0
        signal_raw[mask]    = sw["signal"][mask]
        composite_raw[mask] = sw["composite"][mask]
        nr_mask_raw[mask]   = sw["is_nr"][mask]

    if use_ny_kz:
        sw_ny = _detect_sweeps_in_window(
            df_15m, asian_daily, NY_START, NY_END)
        mask_ny = (sw_ny["signal"] != 0) & (signal_raw == 0)
        signal_raw[mask_ny]    = sw_ny["signal"][mask_ny]
        composite_raw[mask_ny] = sw_ny["composite"][mask_ny]
        nr_mask_raw[mask_ny]   = sw_ny["is_nr"][mask_ny]

    # Apply NR filter: suppress signals on non-NR days if requested
    if nr_filter:
        not_nr = (signal_raw != 0) & ~nr_mask_raw
        signal_raw[not_nr]    = 0
        composite_raw[not_nr] = 0.0

    # Shift 1 bar forward: signal detected at bar T → entry at bar T+1 open
    signal    = signal_raw.shift(1).fillna(0).astype(int)
    composite = composite_raw.shift(1).fillna(0.0)

    # Regime label: "london_kz" / "ny_kz" / "out"
    h = idx.hour
    regime = pd.Series("out", index=idx)
    regime[(h >= LONDON_START) & (h < LONDON_END)] = "london_kz"
    regime[(h >= NY_START)     & (h < NY_END)]     = "ny_kz"

    return pd.DataFrame({
        "signal":    signal,
        "composite": composite,
        "regime":    regime,
    }, index=idx)
