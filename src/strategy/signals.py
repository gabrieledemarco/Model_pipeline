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
  oi_signal     : 2   – positioning pressure from open interest
  funding       : 1   – contrarian sentiment from funding rate
  volume        : 2   – volume confirmation / divergence
  cyclicality   : 1   – time-based seasonal edge

Max raw composite ≈ ±19 (all components maxed simultaneously).
Entry thresholds are set at ±5 (loose) and ±8 (strong).
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
    Open Interest confirmation / divergence signal.
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
    tf_data: Dict[str, pd.DataFrame],
    oi_df:   pd.DataFrame,
    funding: pd.Series,
) -> pd.DataFrame:
    """
    Compute all signal components and produce a composite score on the 1H index.

    Returns a DataFrame indexed to 1H bars with columns:
      s_weekly, s_daily, s_4h, s_1h, s_oi, s_funding, s_vol, s_cycle,
      composite, signal (−1 / 0 / +1), strong (bool), regime (str)
    """
    df_1w = tf_data["1W"]
    df_1d = tf_data["1D"]
    df_4h = tf_data["4H"]
    df_1h = tf_data["1H"]

    base = df_1h.index

    # Native-TF signals
    raw = {
        "s_weekly":  weekly_trend(df_1w),
        "s_daily":   daily_trend(df_1d),
        "s_4h":      fourfour_setup(df_4h),
        "s_1h":      one_hour_entry(df_1h),
        "s_oi":      oi_signal(oi_df, df_1d),
        "s_funding": funding_signal(funding),
        "s_vol":     volume_signal(df_1h),
        "s_cycle":   cyclicality_signal(df_1h),
    }

    out = pd.DataFrame(index=base)

    # HTF signals: align to 1H via forward-fill
    for key in ("s_weekly", "s_daily", "s_4h", "s_oi", "s_funding"):
        out[key] = _align(raw[key], base)

    # Same-TF signals: direct assignment
    for key in ("s_1h", "s_vol", "s_cycle"):
        out[key] = raw[key].reindex(base, fill_value=0).values

    # Weighted composite
    out["composite"] = sum(out[k] * WEIGHTS[k] for k in WEIGHTS)

    # Thresholded signal
    out["signal"] = np.where(
        out["composite"] >= LONG_THRESH, 1,
        np.where(out["composite"] <= SHORT_THRESH, -1, 0),
    ).astype(int)

    out["strong"] = (out["composite"].abs() >= STRONG_THRESH).astype(int)

    # Regime from daily 200-EMA
    bull_regime = (df_1d["close"] > df_1d["ema_200"]).astype(int)
    out["regime"] = _align(
        bull_regime.map({1: "bull", 0: "bear"}).rename("regime"), base
    )

    return out
