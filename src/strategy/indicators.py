"""
Pure-numpy / pandas technical indicators.
No external TA library dependency – all computed from first principles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal_p: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_e  = series.ewm(span=fast,   adjust=False).mean()
    slow_e  = series.ewm(span=slow,   adjust=False).mean()
    line    = fast_e - slow_e
    sig     = line.ewm(span=signal_p, adjust=False).mean()
    hist    = line - sig
    return line, sig, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up   = high.diff()
    down = (-low.diff())

    dm_p = np.where((up > down) & (up > 0), up, 0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)

    tr_v = atr(high, low, close, period)
    sm_p = pd.Series(dm_p, index=close.index).ewm(com=period - 1, adjust=False).mean()
    sm_m = pd.Series(dm_m, index=close.index).ewm(com=period - 1, adjust=False).mean()

    di_p = (100 * sm_p / tr_v.replace(0, np.nan)).fillna(0)
    di_m = (100 * sm_m / tr_v.replace(0, np.nan)).fillna(0)
    denom = (di_p + di_m).replace(0, np.nan)
    dx    = (100 * (di_p - di_m).abs() / denom).fillna(0)
    adx_v = dx.ewm(com=period - 1, adjust=False).mean()
    return adx_v, di_p, di_m


def bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0
              ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid   = sma(series, period)
    std   = series.rolling(period, min_periods=1).std().fillna(0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct   = ((series - lower) / (upper - lower).replace(0, np.nan)).fillna(0.5)
    return upper, mid, lower, pct


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_p: int = 14, d_p: int = 3) -> tuple[pd.Series, pd.Series]:
    lo  = low.rolling(k_p, min_periods=1).min()
    hi  = high.rolling(k_p, min_periods=1).max()
    k   = (100 * (close - lo) / (hi - lo).replace(0, np.nan)).fillna(50)
    d   = k.rolling(d_p, min_periods=1).mean()
    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


# ─────────────────────────────────────────────────────────────────────────────
# Full enrichment pipeline
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append all technical indicators to an OHLCV DataFrame in-place copy."""
    df = df.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ── EMAs ─────────────────────────────────────────────────────────────────
    for p in [8, 13, 21, 34, 50, 100, 200]:
        df[f"ema_{p}"] = ema(c, p)

    # ── Trend structure ───────────────────────────────────────────────────────
    df["ema_bull_13_34"]  = (df["ema_13"]  > df["ema_34"] ).astype(int)
    df["ema_bull_21_50"]  = (df["ema_21"]  > df["ema_50"] ).astype(int)
    df["ema_bull_50_200"] = (df["ema_50"]  > df["ema_200"]).astype(int)
    df["price_vs_ema21"]  = (c > df["ema_21"]).astype(int)
    df["price_vs_ema50"]  = (c > df["ema_50"]).astype(int)
    df["price_vs_ema200"] = (c > df["ema_200"]).astype(int)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi_7"]  = rsi(c, 7)
    df["rsi_14"] = rsi(c, 14)

    df["macd"], df["macd_sig"], df["macd_hist"] = macd(c)
    df["macd_cross_up"]   = ((df["macd"] > df["macd_sig"]) &
                              (df["macd"].shift(1) <= df["macd_sig"].shift(1))).astype(int)
    df["macd_cross_down"] = ((df["macd"] < df["macd_sig"]) &
                              (df["macd"].shift(1) >= df["macd_sig"].shift(1))).astype(int)

    df["stoch_k"], df["stoch_d"] = stochastic(h, l, c)

    # ── Volatility / ATR ─────────────────────────────────────────────────────
    df["atr_14"]  = atr(h, l, c, 14)
    df["atr_pct"] = (df["atr_14"] / c * 100).fillna(0)

    # ── ADX ──────────────────────────────────────────────────────────────────
    df["adx"], df["di_plus"], df["di_minus"] = adx(h, l, c)
    df["adx_trending"] = (df["adx"] > 25).astype(int)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df["bb_up"], df["bb_mid"], df["bb_lo"], df["bb_pct"] = bollinger(c)
    df["bb_squeeze"] = (df["bb_up"] - df["bb_lo"]) / df["bb_mid"]

    # ── Volume ───────────────────────────────────────────────────────────────
    df["vol_sma20"]  = sma(v, 20)
    df["vol_ratio"]  = (v / df["vol_sma20"].replace(0, np.nan)).fillna(1.0)
    df["obv"]        = obv(c, v)
    df["obv_ema21"]  = ema(df["obv"], 21)
    df["obv_trend"]  = np.sign(df["obv"] - df["obv_ema21"])
    df["vwap"]       = vwap(h, l, c, v)
    df["vwap_dev"]   = (c - df["vwap"]) / df["vwap"]

    # ── Returns & vol regime ──────────────────────────────────────────────────
    df["log_ret"]   = np.log(c).diff().fillna(0)
    df["ret_1"]     = c.pct_change().fillna(0)
    df["rvol_20"]   = df["log_ret"].rolling(20, min_periods=1).std() * np.sqrt(252)

    # ── Candle anatomy ────────────────────────────────────────────────────────
    df["body"]       = (c - o).abs()
    df["upper_wick"] = h - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - l
    df["is_bull"]    = (c > o).astype(int)

    # ── Price dynamics ────────────────────────────────────────────────────────
    df["price_accel"] = (c.pct_change() - c.pct_change().shift(1)).fillna(0)
    df["ema21_slope"] = ((df["ema_21"] - df["ema_21"].shift(4)) /
                          df["ema_21"].shift(4).replace(0, np.nan)).fillna(0)
    df["rvol_ratio"]  = (df["rvol_20"] /
                          df["rvol_20"].rolling(120, min_periods=10).mean()
                          .replace(0, np.nan)).fillna(1.0)

    # ── Taker flow / CVD (only present when taker_buy_base is available) ──────
    if "taker_buy_base" in df.columns:
        taker_buy  = df["taker_buy_base"]
        taker_sell = v - taker_buy
        raw_delta  = taker_buy - taker_sell
        df["cvd"]           = raw_delta.cumsum()
        df["cvd_ema21"]     = ema(df["cvd"], 21)
        df["cvd_div"]       = df["cvd"] - df["cvd_ema21"]
        df["cvd_slope_4"]   = (df["cvd"] - df["cvd"].shift(4)).fillna(0)
        df["flow_ratio"]    = (taker_buy / v.replace(0, np.nan)).fillna(0.5)
        df["flow_imb_8"]    = df["flow_ratio"].rolling(8, min_periods=1).mean() - 0.5
        # Negative = CVD going down while price goes up (or vice versa): divergence
        df["cvd_price_div"] = (np.sign(df["cvd_slope_4"]) *
                                np.sign((c - c.shift(4)).fillna(0))).fillna(0)

    # ── Trade count (only present when n_trades is available) ────────────────
    if "n_trades" in df.columns:
        nt = df["n_trades"]
        df["n_trades_ratio"] = (nt / nt.rolling(20, min_periods=1).mean()
                                 .replace(0, np.nan)).fillna(1.0)

    return df
