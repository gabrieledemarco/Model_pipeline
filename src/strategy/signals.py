"""
Multi-timeframe signal scoring for BTCUSDT.

Each module returns a continuous score on its own timeframe.
All scores are then forward-filled and aligned to the 1H base index via
pd.merge_asof, so every 1H bar carries the most recent HTF view.

Signal component weights (tuned empirically):
  weekly_trend  : 3   – macro bias sets the tide
  daily_trend   : 4   – primary trend filter; highest weight
  4h_setup      : 3   – intermediate context and setup quality
  1h_entry      : 3   – precise entry timing on the working TF
  oi_signal     : 2   – positioning pressure from open interest (real basis)
  funding       : 1   – contrarian sentiment from funding rate
  volume        : 2   – volume confirmation / divergence
  cyclicality   : 1   – time-based seasonal edge
  15m_entry     : 2   – intraday precision entry (RSI-7 + MACD, forward-filled)
  1m_entry      : 1   – microstructure confirmation (high-vol bar direction)

Entry thresholds: ±5 (loose), ±8 (strong).
15m and 1m signals are forward-filled to 1H; they contribute 0 when data
is unavailable, so the system degrades gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Individual modules
# ─────────────────────────────────────────────────────────────────────────────

def weekly_trend(df: pd.DataFrame) -> pd.Series:
    """
    Score: -3 to +3
      ±2  EMA-13 vs EMA-34 alignment
      ±1  price vs EMA-13
    """
    s = pd.Series(0.0, index=df.index)
    s += np.where(df["ema_13"] > df["ema_34"], 2.0, -2.0)
    s += np.where(df["close"]  > df["ema_13"], 1.0, -1.0)
    return s.rename("s_weekly")


def daily_trend(df: pd.DataFrame) -> pd.Series:
    """
    Score: -4 to +4
      ±2  EMA stack (21/50/200)
      ±1  RSI-14 zone (>55 / <45)
      ±1  MACD histogram sign
    """
    s = pd.Series(0.0, index=df.index)

    # EMA stack: count how many of 3 conditions hold
    bull = (df["ema_bull_21_50"] + df["ema_bull_50_200"] + df["price_vs_ema21"])
    bear = ((~df["ema_bull_21_50"].astype(bool)).astype(int) +
            (~df["ema_bull_50_200"].astype(bool)).astype(int) +
            (~df["price_vs_ema21"].astype(bool)).astype(int))

    s += np.where(bull >= 2, 2.0, np.where(bull == 1, 1.0, 0.0))
    s -= np.where(bear >= 2, 2.0, np.where(bear == 1, 1.0, 0.0))

    s += np.where(df["rsi_14"] > 55, 1.0, np.where(df["rsi_14"] < 45, -1.0, 0.0))
    s += np.sign(df["macd_hist"].fillna(0))
    return s.rename("s_daily")


def fourfour_setup(df: pd.DataFrame) -> pd.Series:
    """
    Score: -3 to +3
      ±1  EMA-21 vs EMA-50
      ±1  RSI-14 zone
      ±1  MACD histogram sign
    """
    s = pd.Series(0.0, index=df.index)
    s += np.where(df["ema_21"] > df["ema_50"], 1.0, -1.0)
    s += np.where(df["rsi_14"] > 55, 1.0, np.where(df["rsi_14"] < 45, -1.0, 0.0))
    s += np.sign(df["macd_hist"].fillna(0))
    return s.rename("s_4h")


def one_hour_entry(df: pd.DataFrame) -> pd.Series:
    """
    Score: -3 to +3
      ±1  RSI zone
      ±1  MACD histogram sign
      ±1  Volume spike confirming direction
    """
    s = pd.Series(0.0, index=df.index)
    s += np.where(df["rsi_14"] > 55, 1.0, np.where(df["rsi_14"] < 45, -1.0, 0.0))
    s += np.sign(df["macd_hist"].fillna(0))
    # Volume spike: vol > 1.5× average + directional confirmation
    spike_dir = np.where(df["vol_ratio"] > 1.5, np.sign(df["log_ret"]), 0.0)
    s += spike_dir
    return s.rename("s_1h")


def oi_signal(df_oi: pd.DataFrame, df_price: pd.DataFrame) -> pd.Series:
    """
    Open Interest confirmation / divergence signal (synthetic OI fallback).
    Score: -2 to +2

      Price ↑ + OI ↑  →  +2  (new longs entering, confirmed bull)
      Price ↓ + OI ↓  →  -2  (longs liquidating, confirmed bear)
      Price ↓ + OI ↑  →  -1  (shorts building, bear but less clean)
      Price ↑ + OI ↓  →  +1  (short squeeze, weakly bullish)
    """
    p_chg  = df_price["close"].pct_change()
    oi_chg = df_oi["oi"].pct_change()

    common = df_price.index.intersection(df_oi.index)
    p_c  = p_chg.reindex(common)
    oi_c = oi_chg.reindex(common)

    s = pd.Series(0.0, index=common)
    s[ (p_c > 0) & (oi_c > 0)] =  2.0
    s[(p_c <= 0) & (oi_c <= 0)] = -2.0
    s[(p_c <= 0) & (oi_c > 0)] = -1.0
    s[ (p_c > 0) & (oi_c <= 0)] =  1.0

    return s.reindex(df_price.index, fill_value=0).rename("s_oi")


def basis_oi_signal(premium_1h: pd.Series, df_price_1h: pd.DataFrame) -> pd.Series:
    """
    Real OI proxy using Binance Vision premiumIndexKlines (basis = futures - spot).
    Score: -2 to +2  (same semantics as oi_signal)

    Logic mirrors OI divergence but using basis changes as positioning proxy:
      Price ↑ + Basis ↑  →  +2  longs aggressively bidding up futures (confirmed bull)
      Price ↓ + Basis ↓  →  -2  shorts pushing futures below spot (confirmed bear)
      Price ↓ + Basis ↑  →  -1  shorts entering but paying premium (bear, less clean)
      Price ↑ + Basis ↓  →  +1  short squeeze; longs not chasing (weakly bullish)

    Additionally: extreme absolute basis (>0.15%) → contrarian overlay
      Basis > +0.15%  →  cap score at 0 (longs overextended, not a buy)
      Basis < -0.10%  →  cap score at 0 (shorts overextended, not a sell)
    """
    p_chg = df_price_1h["close"].pct_change()

    # Align basis to 1H price index via forward-fill
    prem_idx = premium_1h.index.astype("datetime64[s]")
    price_idx = df_price_1h.index.astype("datetime64[s]")
    prem_df = pd.DataFrame({"ts": prem_idx, "v": premium_1h.values}).sort_values("ts")
    base_df = pd.DataFrame({"ts": price_idx})
    merged = pd.merge_asof(base_df, prem_df, on="ts", direction="backward")
    basis = pd.Series(merged["v"].fillna(0.0).values,
                      index=df_price_1h.index, name="basis")

    b_chg = basis.diff()

    s = pd.Series(0.0, index=df_price_1h.index)
    s[(p_chg > 0)  & (b_chg > 0)]  =  2.0
    s[(p_chg <= 0) & (b_chg <= 0)] = -2.0
    s[(p_chg <= 0) & (b_chg > 0)]  = -1.0
    s[(p_chg > 0)  & (b_chg <= 0)] =  1.0

    # Contrarian cap: extreme basis means market is overcrowded
    s[basis >  0.0015] = np.minimum(s[basis >  0.0015], 0.0)
    s[basis < -0.0010] = np.maximum(s[basis < -0.0010], 0.0)

    return s.rename("s_oi")


def fifteen_min_entry(df: pd.DataFrame) -> pd.Series:
    """
    15-minute precision entry signal.
    Score: -2 to +2
      ±1  RSI-7 zone (fast RSI for short-TF momentum)
      ±1  MACD histogram sign
    """
    s = pd.Series(0.0, index=df.index)
    s += np.where(df["rsi_7"] > 60, 1.0, np.where(df["rsi_7"] < 40, -1.0, 0.0))
    s += np.sign(df["macd_hist"].fillna(0))
    return s.rename("s_15m")


def one_min_entry(df: pd.DataFrame) -> pd.Series:
    """
    1-minute microstructure signal.
    Score: -1 to +1
    High-volume bars (vol_ratio > 2.0) confirm direction via log_ret sign.
    """
    s = pd.Series(0.0, index=df.index)
    mask = df["vol_ratio"] > 2.0
    s[mask] = np.sign(df.loc[mask, "log_ret"]).clip(-1, 1)
    return s.rename("s_1m")


def funding_signal(funding: pd.Series) -> pd.Series:
    """
    Contrarian funding signal.
    Score: -1 to +1

      Funding > +0.05 %  →  longs overextended  →  -1
      Funding < -0.05 %  →  shorts overextended  →  +1
    """
    s = pd.Series(0.0, index=funding.index)
    s[funding >  0.0005] = -1.0
    s[funding < -0.0005] =  1.0
    return s.rename("s_funding")


def volume_signal(df: pd.DataFrame) -> pd.Series:
    """
    Volume confirmation signal (on-bar direction × volume).
    Score: -1 to +1

      High volume + up candle   → +1
      High volume + down candle → -1
      Normal volume             →  0
    """
    s = pd.Series(0.0, index=df.index)
    s[df["vol_ratio"] > 1.3] = np.sign(df["log_ret"][df["vol_ratio"] > 1.3])
    return s.rename("s_vol")


def cyclicality_signal(df: pd.DataFrame) -> pd.Series:
    """
    Seasonal / time-based edge.
    Score: -1 to +1

    Monthly effect (BTC historical):
      Bullish months  : Oct, Nov, Dec, Jan, Apr  → +0.5
      Bearish months  : May, Jun, Sep             → -0.5

    Day-of-week effect:
      Mon–Wed historically stronger  → +0.5
      Sat–Sun weaker                 → -0.5
    """
    BULL_MONTHS  = {10, 11, 12, 1, 4}
    BEAR_MONTHS  = {5, 6, 9}
    BULL_DAYS    = {0, 1, 2}           # Mon, Tue, Wed
    WEAK_DAYS    = {5, 6}              # Sat, Sun

    month = df.index.month
    dow   = df.index.dayofweek

    m_score = np.where(np.isin(month, list(BULL_MONTHS)), 0.5,
              np.where(np.isin(month, list(BEAR_MONTHS)), -0.5, 0.0))
    d_score = np.where(np.isin(dow, list(BULL_DAYS)),  0.5,
              np.where(np.isin(dow, list(WEAK_DAYS)), -0.5, 0.0))

    return pd.Series(np.clip(m_score + d_score, -1, 1),
                     index=df.index, name="s_cycle")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "s_weekly":  3,
    "s_daily":   4,
    "s_4h":      3,
    "s_1h":      3,
    "s_oi":      2,
    "s_funding": 1,
    "s_vol":     2,
    "s_cycle":   1,
    "s_15m":     2,  # 15-minute entry precision
    "s_1m":      1,  # 1-minute microstructure
}

LONG_THRESH   = 5.0
SHORT_THRESH  = -5.0
STRONG_THRESH = 8.0


def _align(signal: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
    """Forward-fill a higher-TF signal to the target (1H) index."""
    # Normalise both sides to second precision to avoid ms-vs-s merge errors
    # (Binance Vision returns datetime64[s], yfinance returns datetime64[ms/ns])
    tgt = target_index.astype("datetime64[s]")
    sig_idx = pd.DatetimeIndex(signal.index).astype("datetime64[s]")
    s_df = pd.DataFrame({"ts": sig_idx, "v": signal.values}).sort_values("ts")
    base = pd.DataFrame({"ts": tgt})
    merged = pd.merge_asof(base, s_df, on="ts", direction="backward")
    return merged["v"].fillna(0.0).to_numpy()


def build_signal_matrix(
    tf_data:    Dict[str, pd.DataFrame],
    oi_df:      pd.DataFrame,
    funding:    pd.Series,
    premium_1h: pd.Series = None,
    df_15m:     pd.DataFrame = None,
    df_1m:      pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute all signal components and produce a composite score on the 1H index.

    Lookahead-free alignment rules
    ───────────────────────────────
    Every HTF bar is indexed at its open_time but contains close data at
    open_time + bar_duration.  Without correction, a 1H bar at T would
    receive a 4H signal whose close is T+4h (future data).

    Fix: shift each HTF signal index forward by its bar duration before
    merge_asof.  This makes merge_asof find the last bar that CLOSED at
    or before the 1H bar's open_time — no look-ahead.

      Weekly  → shift +7 days   (bar closes end-of-week)
      Daily   → shift +1 day    (bar closes at midnight next day)
      4H      → shift +4 hours
      Funding → shift +1 day    (daily-resampled series)
      OI syn. → shift +1 day    (daily-frequency synthetic OI)

    Sub-1H signals (15m, 1m) are aligned from the 1H close_time (= open
    + 1h) after shifting by their bar duration, so we use the last bar
    that closed within the 1H bar — not the bar that just opened.

    Parameters
    ----------
    tf_data     : multi-TF OHLCV DataFrames with indicators (keys: 1W, 1D, 4H, 1H)
    oi_df       : synthetic OI DataFrame (fallback when premium_1h is None/empty)
    funding     : daily funding rate Series
    premium_1h  : real basis series from Binance Vision premiumIndexKlines (1H)
    df_15m      : 15-minute OHLCV with indicators (forward-filled to 1H)
    df_1m       : 1-minute OHLCV with indicators (forward-filled to 1H)

    Returns a DataFrame indexed to 1H bars with columns:
      s_weekly, s_daily, s_4h, s_1h, s_oi, s_funding, s_vol, s_cycle,
      s_15m, s_1m, composite, signal (−1 / 0 / +1), strong (bool),
      regime (str), oi_source ('real_basis' | 'synthetic')
    """
    df_1w = tf_data["1W"]
    df_1d = tf_data["1D"]
    df_4h = tf_data["4H"]
    df_1h = tf_data["1H"]

    base = df_1h.index
    # 1H close_time: aligns sub-1H signals to the last bar closed within each 1H
    close_times = base + pd.Timedelta(hours=1)

    # OI / basis signal: use real basis when available
    use_real_basis = (premium_1h is not None and
                      not premium_1h.empty and
                      len(premium_1h) > 100)

    if use_real_basis:
        s_oi_raw = basis_oi_signal(premium_1h, df_1h)
        oi_source = "real_basis"
    else:
        s_oi_raw = oi_signal(oi_df, df_1d)
        oi_source = "synthetic"

    # 15m signal (0 when data unavailable)
    has_15m = (df_15m is not None and not df_15m.empty and
               "rsi_7" in df_15m.columns and len(df_15m) > 100)
    s_15m_raw = fifteen_min_entry(df_15m) if has_15m else pd.Series(0.0, index=base)

    # 1m signal (0 when data unavailable)
    has_1m = (df_1m is not None and not df_1m.empty and
              "vol_ratio" in df_1m.columns and len(df_1m) > 100)
    s_1m_raw = one_min_entry(df_1m) if has_1m else pd.Series(0.0, index=base)

    def _shift(series: pd.Series, delta: pd.Timedelta) -> pd.Series:
        """Relabel signal index by +delta so merge_asof finds the last CLOSED bar."""
        s = series.copy()
        s.index = s.index + delta
        return s

    out = pd.DataFrame(index=base)

    # ── HTF signals: shift index forward by bar duration ─────────────────────
    out["s_weekly"] = _align(_shift(weekly_trend(df_1w),   pd.Timedelta(weeks=1)),  base)
    out["s_daily"]  = _align(_shift(daily_trend(df_1d),    pd.Timedelta(days=1)),   base)
    out["s_4h"]     = _align(_shift(fourfour_setup(df_4h), pd.Timedelta(hours=4)),  base)

    # OI: real basis is 1H-frequency (no shift needed); synthetic is daily (shift +1d)
    if use_real_basis:
        out["s_oi"] = _align(s_oi_raw, base)
    else:
        out["s_oi"] = _align(_shift(s_oi_raw, pd.Timedelta(days=1)), base)

    # Funding: daily-resampled → shift +1d
    out["s_funding"] = _align(
        _shift(funding_signal(funding), pd.Timedelta(days=1)), base)

    # ── Sub-1H signals: shift by bar duration + align from 1H close_time ─────
    # Picks the last 15m/1m bar that CLOSED within (not just started in) the 1H.
    out["s_15m"] = _align(_shift(s_15m_raw, pd.Timedelta(minutes=15)), close_times)
    out["s_1m"]  = _align(_shift(s_1m_raw,  pd.Timedelta(minutes=1)),  close_times)

    # ── Same-TF 1H signals: direct assignment, no shift ──────────────────────
    out["s_1h"]    = one_hour_entry(df_1h).reindex(base, fill_value=0).values
    out["s_vol"]   = volume_signal(df_1h).reindex(base, fill_value=0).values
    out["s_cycle"] = cyclicality_signal(df_1h).reindex(base, fill_value=0).values

    out["oi_source"] = oi_source
    out["has_15m"]   = int(has_15m)
    out["has_1m"]    = int(has_1m)

    # Weighted composite
    out["composite"] = sum(out[k] * WEIGHTS[k] for k in WEIGHTS)

    # Thresholded signal
    out["signal"] = np.where(
        out["composite"] >= LONG_THRESH, 1,
        np.where(out["composite"] <= SHORT_THRESH, -1, 0),
    ).astype(int)

    out["strong"] = (out["composite"].abs() >= STRONG_THRESH).astype(int)

    # Regime from daily 200-EMA: shift +1d (daily bar not closed until next midnight)
    bull_regime = (df_1d["close"] > df_1d["ema_200"]).astype(int)
    out["regime"] = _align(
        _shift(bull_regime.map({1: "bull", 0: "bear"}).rename("regime"),
               pd.Timedelta(days=1)),
        base,
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 15M base-timeframe variant
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_matrix_15m(
    tf_data:    Dict[str, pd.DataFrame],
    oi_df:      pd.DataFrame,
    funding:    pd.Series,
    premium_1h: pd.Series = None,
) -> pd.DataFrame:
    """
    Same composite signal as build_signal_matrix() but evaluated on the 15M index.

    Key differences vs the 1H version:
    - s_1h  : 1H entry signal is now an HTF source (shifted +1H before align)
    - s_15m : computed directly on the 15M same-TF bars (no shift)
    - s_vol / s_cycle: computed from 15M bars
    - All HTF shifts are identical (weekly +7d, daily +1d, 4H +4H, funding +1d)
    """
    df_1w  = tf_data["1W"]
    df_1d  = tf_data["1D"]
    df_4h  = tf_data["4H"]
    df_1h  = tf_data["1H"]
    df_15m = tf_data["15M"]

    base = df_15m.index

    def _shift(series: pd.Series, delta: pd.Timedelta) -> pd.Series:
        s = series.copy()
        s.index = s.index + delta
        return s

    use_real_basis = (premium_1h is not None and
                      not premium_1h.empty and len(premium_1h) > 100)

    if use_real_basis:
        s_oi_raw  = basis_oi_signal(premium_1h, df_1h)
        oi_source = "real_basis"
    else:
        s_oi_raw  = oi_signal(oi_df, df_1d)
        oi_source = "synthetic"

    out = pd.DataFrame(index=base)

    # HTF signals: shift index forward by bar duration so merge_asof returns
    # the last bar that was fully closed at the 15M bar's open_time.
    out["s_weekly"] = _align(_shift(weekly_trend(df_1w),   pd.Timedelta(weeks=1)),  base)
    out["s_daily"]  = _align(_shift(daily_trend(df_1d),    pd.Timedelta(days=1)),   base)
    out["s_4h"]     = _align(_shift(fourfour_setup(df_4h), pd.Timedelta(hours=4)),  base)
    out["s_1h"]     = _align(_shift(one_hour_entry(df_1h), pd.Timedelta(hours=1)),  base)

    # OI: real basis is 1H-indexed → shift +1H; synthetic is daily → shift +1d
    if use_real_basis:
        out["s_oi"] = _align(_shift(s_oi_raw, pd.Timedelta(hours=1)), base)
    else:
        out["s_oi"] = _align(_shift(s_oi_raw, pd.Timedelta(days=1)),  base)

    out["s_funding"] = _align(
        _shift(funding_signal(funding), pd.Timedelta(days=1)), base)

    # Same-TF 15M signals: direct computation on 15M bars, no shift needed
    out["s_15m"]   = fifteen_min_entry(df_15m).reindex(base, fill_value=0).values
    out["s_vol"]   = volume_signal(df_15m).reindex(base, fill_value=0).values
    out["s_cycle"] = cyclicality_signal(df_15m).reindex(base, fill_value=0).values
    out["s_1m"]    = 0.0   # no 1M data; contributes 0 to composite

    out["oi_source"] = oi_source
    out["has_15m"]   = 1
    out["has_1m"]    = 0

    # Weighted composite (same WEIGHTS dict as the 1H version)
    out["composite"] = sum(out[k] * WEIGHTS[k] for k in WEIGHTS)

    out["signal"] = np.where(
        out["composite"] >= LONG_THRESH,  1,
        np.where(out["composite"] <= SHORT_THRESH, -1, 0),
    ).astype(int)

    out["strong"] = (out["composite"].abs() >= STRONG_THRESH).astype(int)

    bull_regime = (df_1d["close"] > df_1d["ema_200"]).astype(int)
    out["regime"] = _align(
        _shift(bull_regime.map({1: "bull", 0: "bear"}).rename("regime"),
               pd.Timedelta(days=1)),
        base,
    )

    return out
