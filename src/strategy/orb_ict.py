"""
orb_ict.py — ICT-style Asian Range Sweep Signal Generator (v2)
==============================================================

Empirical evidence (test_ict_correlation.py, BTCUSDT 2020-2026):
  - London KZ (07:00-09:59 UTC): 52.9% directional accuracy at 15m,
    54.0% at 4h — statistically significant (p<0.001 at 4h)
  - NY KZ (13:00-15:59 UTC): 48-49%, NOT significant — excluded
  - Penetration depth: NEGATIVE Spearman vs forward return (deeper sweep
    → worse outcome) → only shallow sweeps are kept (depth/ATR ≤ threshold)
  - NR filter and compression_ratio: NOT statistically significant → removed

Signal definition:
  LONG  : London KZ 15M bar low < Asian Low AND bar closes back above AL
           AND penetration depth < max_depth_atr × ATR (shallow sweep)
  SHORT : London KZ 15M bar high > Asian High AND bar closes back below AH
           AND penetration depth < max_depth_atr × ATR (shallow sweep)

Entry  : next 15M bar open (zero look-ahead)
Exit   : ATR-based SL and TP (optimised in IS walk-forward grid)
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
NY_START     = 13   # 13:00 UTC  (kept for reference only — no signal)
NY_END       = 16   # 15:59 UTC


def build_asian_range(df_1h: pd.DataFrame,
                      nr_lookback: int = 20) -> pd.DataFrame:
    """
    Compute the daily Asian session range from 1H bars.

    Returns a DataFrame indexed by DATE with columns:
        asian_high  : max(high) of bars 00-06 UTC
        asian_low   : min(low)  of bars 00-06 UTC
        asian_range : asian_high - asian_low
        atr_1h      : mean ATR(14) of Asian session bars
    """
    mask  = (df_1h.index.hour >= ASIA_START) & (df_1h.index.hour < ASIA_END)
    df_as = df_1h[mask].copy()
    df_as["date"] = df_as.index.date

    def agg_day(g):
        g = g.sort_index()
        return pd.Series({
            "asian_high": g["high"].max(),
            "asian_low":  g["low"].min(),
            "atr_1h":     g["atr_14"].mean() if "atr_14" in g.columns else np.nan,
        })

    daily = (df_as.groupby("date")
               .apply(agg_day, include_groups=False)
               .reset_index())
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()

    # Require complete Asian sessions
    bar_cnt = df_as.groupby("date").size()
    bar_cnt.index = pd.to_datetime(bar_cnt.index)
    daily = daily[bar_cnt >= (ASIA_END - ASIA_START)].copy()

    daily["asian_range"] = daily["asian_high"] - daily["asian_low"]
    daily["asian_range_pct"] = (
        daily["asian_range"] / daily["asian_low"].clip(lower=1) * 100)

    return daily


def build_orb_ict_signals(
    df_15m: pd.DataFrame,
    asian_daily: pd.DataFrame,
    max_depth_atr: float = 0.5,
) -> pd.DataFrame:
    """
    Build London KZ Asian Range Sweep signals on 15M data.

    Only shallow sweeps are emitted: penetration_depth / ATR_15M ≤ max_depth_atr.
    Empirically, deep sweeps have negative directional predictability
    (Spearman r = -0.039 at 4h horizon, p=0.004).

    Parameters
    ----------
    df_15m        : 15M OHLCV DataFrame with atr_14 indicator
    asian_daily   : output of build_asian_range()
    max_depth_atr : max penetration depth in ATR units (default 0.5)

    Returns
    -------
    pd.DataFrame — signal (int), composite (float), regime (str)
    Aligned to df_15m.index. Signal on event bar T; entry at T+1 open.
    """
    idx = df_15m.index
    h   = idx.hour

    hi  = df_15m["high"]
    lo  = df_15m["low"]
    cl  = df_15m["close"]
    atr = (df_15m["atr_14"].clip(lower=1.0)
           if "atr_14" in df_15m.columns
           else pd.Series(1.0, index=idx))

    # Broadcast daily Asian levels to 15M bars
    date_idx  = idx.normalize()
    ah_series = date_idx.map(asian_daily["asian_high"].to_dict()).astype(float)
    al_series = date_idx.map(asian_daily["asian_low"].to_dict()).astype(float)

    ah = pd.Series(ah_series.values, index=idx)
    al = pd.Series(al_series.values, index=idx)

    in_london = (h >= LONDON_START) & (h < LONDON_END)
    valid     = in_london & ah.notna() & al.notna()

    long_pen  = (al - lo).clip(lower=0)
    short_pen = (hi - ah).clip(lower=0)

    depth_atr_long  = long_pen  / atr
    depth_atr_short = short_pen / atr

    is_long  = (valid
                & (lo < al) & (cl >= al)
                & (long_pen > 0)
                & (depth_atr_long <= max_depth_atr))

    is_short = (valid
                & (hi > ah) & (cl <= ah)
                & (short_pen > 0)
                & (depth_atr_short <= max_depth_atr))

    # Flat score (equal weighting — scoring was empirically contradicted)
    signal    = pd.Series(0,   index=idx, dtype=int)
    composite = pd.Series(1.0, index=idx)

    signal[is_long]  =  1
    signal[is_short] = -1

    # Conflict on same bar: flatten
    conflict = is_long & is_short
    signal[conflict]    = 0
    composite[conflict] = 0.0
    composite[signal == 0] = 0.0

    # Shift 1 bar: signal at T → entry at T+1 open (zero look-ahead)
    signal    = signal.shift(1).fillna(0).astype(int)
    composite = composite.shift(1).fillna(0.0)

    regime = pd.Series("out", index=idx)
    regime[(h >= LONDON_START) & (h < LONDON_END)] = "london_kz"

    return pd.DataFrame({
        "signal":    signal,
        "composite": composite,
        "regime":    regime,
    }, index=idx)
