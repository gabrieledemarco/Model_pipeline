"""
Bias-free feature matrix builder for ML gate model.

For each 1H bar at open_time T, all features are computed exclusively from
data available at T — no future information.  HTF alignment follows the same
_shift() + merge_asof pattern used in signals.py.

Feature groups
──────────────
  1H indicators       : rsi, macd_hist, adx, bb_pct, vol_ratio, atr_pct …
  4H indicators (lag) : same set, shifted +4h before merge_asof
  1D indicators (lag) : same set, shifted +1d before merge_asof
  1W indicators (lag) : EMA structure + RSI, shifted +7d before merge_asof
  Signal scores       : the 9 component scores + composite (already bias-free)
  Time features       : hour, day_of_week, month, session flag
  Lagged returns      : log_ret at T-1 … T-8
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


# ── Shared alignment helper (mirrors signals.py) ──────────────────────────────

def _align_htf(signal_df: pd.DataFrame, cols: list[str],
               base: pd.DatetimeIndex, shift: pd.Timedelta) -> pd.DataFrame:
    """
    Forward-fill HTF indicator columns to the 1H base index.
    Shifts the HTF index by +shift so merge_asof finds the last CLOSED bar.
    """
    tgt  = base.astype("datetime64[s]")
    src  = (signal_df.index + shift).astype("datetime64[s]")
    vals = signal_df[cols].copy()
    vals.index = src

    result = pd.DataFrame(index=base)
    for col in cols:
        s = vals[col].reset_index()
        s.columns = ["ts", "v"]
        b = pd.DataFrame({"ts": tgt})
        merged = pd.merge_asof(b, s, on="ts", direction="backward")
        result[col] = merged["v"].fillna(0.0).values
    return result


# ── Feature names (declared for downstream use) ────────────────────────────────

SIGNAL_FEATURES = [
    "s_weekly", "s_daily", "s_4h", "s_1h", "s_oi",
    "s_funding", "s_vol", "s_cycle", "s_15m", "composite",
]

_1H_INDICATORS = [
    "rsi_14", "rsi_7", "macd_hist", "adx", "di_plus", "di_minus",
    "bb_pct", "vol_ratio", "log_ret", "atr_pct", "rvol_20",
    "ema_bull_21_50", "ema_bull_50_200", "price_vs_ema200",
    "stoch_k", "obv_trend", "vwap_dev",
    # price dynamics (always computed)
    "price_accel", "ema21_slope", "rvol_ratio",
    # taker flow / CVD (present only when fetch_flow=True was used)
    "cvd_div", "cvd_slope_4", "flow_ratio", "flow_imb_8", "cvd_price_div",
    "n_trades_ratio",
]

_4H_INDICATORS = [
    "rsi_14", "macd_hist", "adx", "di_plus", "di_minus",
    "bb_pct", "ema_bull_21_50", "ema_bull_50_200", "vol_ratio",
]

_1D_INDICATORS = [
    "rsi_14", "macd_hist", "adx", "ema_bull_21_50",
    "ema_bull_50_200", "price_vs_ema200", "bb_pct",
]

_1W_INDICATORS = [
    "rsi_14", "macd_hist", "ema_bull_13_34",
]

TIME_FEATURES = ["hour", "dayofweek", "month", "in_session"]
LAG_FEATURES  = [f"lag_ret_{i}" for i in range(1, 9)]


def build_feature_matrix(
    tf_data:  Dict[str, pd.DataFrame],   # TF → DataFrame with indicators
    signals:  pd.DataFrame,               # bias-free signal matrix (1H index)
) -> pd.DataFrame:
    """
    Produce a single DataFrame aligned to the 1H index containing all features
    needed by the ML gate.

    Parameters
    ----------
    tf_data  : dict with keys '1W', '1D', '4H', '1H' (all with add_indicators applied)
    signals  : output of build_signal_matrix() — carries signal scores + composite

    Returns
    -------
    pd.DataFrame  indexed to 1H bars, one column per feature
    """
    df_1h = tf_data["1H"]
    df_4h = tf_data.get("4H", pd.DataFrame())
    df_1d = tf_data.get("1D", pd.DataFrame())
    df_1w = tf_data.get("1W", pd.DataFrame())

    base  = df_1h.index
    feats = pd.DataFrame(index=base)

    # ── 1H indicators (same TF, no shift needed) ─────────────────────────────
    for col in _1H_INDICATORS:
        if col in df_1h.columns:
            feats[f"1h_{col}"] = df_1h[col].reindex(base).fillna(0).values

    # ── Lagged 1H returns ─────────────────────────────────────────────────────
    for i in range(1, 9):
        feats[f"lag_ret_{i}"] = df_1h["log_ret"].shift(i).reindex(base).fillna(0).values

    # ── 4H indicators (shift +4h) ─────────────────────────────────────────────
    if not df_4h.empty:
        avail = [c for c in _4H_INDICATORS if c in df_4h.columns]
        if avail:
            htf = _align_htf(df_4h, avail, base, pd.Timedelta(hours=4))
            htf.columns = [f"4h_{c}" for c in htf.columns]
            feats = pd.concat([feats, htf], axis=1)

    # ── 1D indicators (shift +1d) ─────────────────────────────────────────────
    if not df_1d.empty:
        avail = [c for c in _1D_INDICATORS if c in df_1d.columns]
        if avail:
            htf = _align_htf(df_1d, avail, base, pd.Timedelta(days=1))
            htf.columns = [f"1d_{c}" for c in htf.columns]
            feats = pd.concat([feats, htf], axis=1)

    # ── 1W indicators (shift +7d) ─────────────────────────────────────────────
    if not df_1w.empty:
        avail = [c for c in _1W_INDICATORS if c in df_1w.columns]
        if avail:
            htf = _align_htf(df_1w, avail, base, pd.Timedelta(weeks=1))
            htf.columns = [f"1w_{c}" for c in htf.columns]
            feats = pd.concat([feats, htf], axis=1)

    # ── Signal component scores ───────────────────────────────────────────────
    for col in SIGNAL_FEATURES:
        if col in signals.columns:
            feats[col] = signals[col].reindex(base).fillna(0).values

    # ── Time features ─────────────────────────────────────────────────────────
    feats["hour"]       = base.hour.astype(float)
    feats["dayofweek"]  = base.dayofweek.astype(float)
    feats["month"]      = base.month.astype(float)
    feats["in_session"] = ((base.hour >= 8) & (base.hour < 21)).astype(float)

    # ── ATR-normalised distance from recent swing lows/highs ─────────────────
    if "close" in df_1h.columns and "high" in df_1h.columns and "atr_14" in df_1h.columns:
        atr  = df_1h["atr_14"]
        hi20 = df_1h["high"].rolling(20, min_periods=1).max()
        lo20 = df_1h["low"].rolling(20, min_periods=1).min()
        c    = df_1h["close"]
        feats["dist_hi20"] = ((hi20 - c) / atr.replace(0, np.nan)).fillna(0).reindex(base).values
        feats["dist_lo20"] = ((c - lo20) / atr.replace(0, np.nan)).fillna(0).reindex(base).values

    # ── Cross-timeframe signal divergence ─────────────────────────────────────
    sig_available = [c for c in ["s_1h", "s_4h", "s_daily", "s_weekly", "s_oi", "s_funding"]
                     if c in signals.columns]
    if "s_1h" in signals.columns and "s_4h" in signals.columns:
        feats["tf_agree_1h_4h"] = (np.sign(signals["s_1h"]) *
                                    np.sign(signals["s_4h"])).reindex(base).fillna(0).values
    if "s_1h" in signals.columns and "s_daily" in signals.columns:
        feats["tf_agree_1h_1d"] = (np.sign(signals["s_1h"]) *
                                    np.sign(signals["s_daily"])).reindex(base).fillna(0).values
    if sig_available:
        bull_mask = signals[sig_available].gt(0)
        feats["n_bullish_sig"]  = bull_mask.sum(axis=1).reindex(base).fillna(0).astype(float).values
        feats["composite_abs"]  = signals["composite"].abs().reindex(base).fillna(0).values

    # ── Return autocorrelation (fast rolling corr with shifted series) ────────
    log_ret = df_1h["log_ret"]
    feats["autocorr_lag1"] = (log_ret.rolling(24, min_periods=12)
                               .corr(log_ret.shift(1))
                               .reindex(base).fillna(0).values)
    feats["autocorr_lag4"] = (log_ret.rolling(24, min_periods=12)
                               .corr(log_ret.shift(4))
                               .reindex(base).fillna(0).values)

    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
    return feats
