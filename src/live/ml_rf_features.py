"""
Shared causal feature engineering for the ML RandomForest 8h strategy —
see docs/ML_RF_8H_STRATEGY_SPEC.md section 3 for the full spec.

Used by both scripts/train_ml_rf_8h.py (fit) and src/live/ml_rf_live.py
(inference) so the two can never silently drift out of sync — the exact
same function builds the feature matrix in both places.

23 features, all causal (no look-ahead):
  - OHLCV (6): ret_1h, ret_4h, ret_24h, atr_pct, vol_ratio, range_pct
  - MTF pivots (12): {m5,m15,h1,d1}_{trend,dist_high,dist_low}
  - HMM regime (5): hmm_state, hmm_prob_bear/side/bull, hmm_duration
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.hmm_regime import HMM_FEATURE_NAMES, predict_hmm_features

PIVOT_LR = {"5m": 12, "15m": 8, "1h": 6, "1d": 3}


def _pivot_features_on_1h(df_tf: pd.DataFrame, left_right: int, tf_label: str,
                           idx_1h: pd.DatetimeIndex) -> pd.DataFrame:
    close = df_tf["close"].values.astype(float)
    high  = df_tf["high"].values.astype(float)
    low   = df_tf["low"].values.astype(float)
    state = causal_trend_state(close, high, low, left_right, left_right)
    atr_tf = (df_tf["high"] - df_tf["low"]).rolling(14).mean().bfill().values
    state["atr"]   = atr_tf
    state["close"] = close

    aligned = align_htf_to_ltf(df_tf.index, state, idx_1h)
    atr_on_1h   = np.where(aligned["atr"].values > 0, aligned["atr"].values, 1.0)
    close_on_1h = aligned["close"].values
    dist_high = (aligned["target_high"].values - close_on_1h) / atr_on_1h
    dist_low  = (close_on_1h - aligned["target_low"].values) / atr_on_1h
    dist_high = np.where(np.isnan(dist_high), 10.0, np.clip(dist_high, 0, 10))
    dist_low  = np.where(np.isnan(dist_low), 10.0, np.clip(dist_low, 0, 10))

    return pd.DataFrame({
        f"{tf_label}_trend":     aligned["trend_state"].values,
        f"{tf_label}_dist_high": dist_high,
        f"{tf_label}_dist_low":  dist_low,
    }, index=idx_1h)


def build_features(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1d: pd.DataFrame,
    hmm_model,
    hmm_sorted_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Build the 23-feature causal matrix aligned on df_1h's index.

    df_1h must already have `add_indicators()` applied (needs atr_14/atr_pct).
    hmm_model/hmm_sorted_idx: as returned by src.strategy.hmm_regime.fit_hmm(),
    fitted on data strictly preceding df_1h's *live* (unfitted) tail — see
    scripts/train_ml_rf_8h.py for the fit procedure.
    """
    idx_1h = df_1h.index
    close  = df_1h["close"].values.astype(float)
    high   = df_1h["high"].values.astype(float)
    low    = df_1h["low"].values.astype(float)

    close_s = pd.Series(close, index=idx_1h)
    vol_s   = pd.Series(df_1h["volume"].values.astype(float), index=idx_1h)
    ohlcv_feats = pd.DataFrame({
        "ret_1h":    np.log(close_s / close_s.shift(1)),
        "ret_4h":    np.log(close_s / close_s.shift(4)),
        "ret_24h":   np.log(close_s / close_s.shift(24)),
        "atr_pct":   df_1h["atr_pct"].values,
        "vol_ratio": (vol_s / vol_s.rolling(20).mean()).values,
        "range_pct": (high - low) / close,
    }, index=idx_1h)

    pivot_feats = pd.concat([
        _pivot_features_on_1h(df_5m,  PIVOT_LR["5m"],  "m5",  idx_1h),
        _pivot_features_on_1h(df_15m, PIVOT_LR["15m"], "m15", idx_1h),
        _pivot_features_on_1h(df_1h,  PIVOT_LR["1h"],  "h1",  idx_1h),
        _pivot_features_on_1h(df_1d,  PIVOT_LR["1d"],  "d1",  idx_1h),
    ], axis=1)

    hmm_feats = predict_hmm_features(hmm_model, hmm_sorted_idx, df_1h)

    feats = pd.concat([ohlcv_feats, pivot_feats, hmm_feats], axis=1)
    return feats


FEATURE_NAMES = (
    ["ret_1h", "ret_4h", "ret_24h", "atr_pct", "vol_ratio", "range_pct"]
    + [f"{tf}_{col}" for tf in ("m5", "m15", "h1", "d1")
       for col in ("trend", "dist_high", "dist_low")]
    + HMM_FEATURE_NAMES
)
