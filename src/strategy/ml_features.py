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

# ── SMC multi-TF configuration ────────────────────────────────────────────────

# Swing detection length (n bars on each side of pivot) per TF
SMC_SWING_LENGTHS: Dict[str, int] = {
    "1W":  4,   # ~1 month on each side
    "1D":  10,  # ~2 weeks
    "4H":  15,  # ~2.5 days
    "1H":  20,  # ~20 hours
    "15M": 20,  # ~5 hours
}

# Shift applied before merge_asof so bar T uses the last CLOSED HTF bar
SMC_TF_SHIFTS: Dict[str, pd.Timedelta] = {
    "1W":  pd.Timedelta(weeks=1),
    "1D":  pd.Timedelta(days=1),
    "4H":  pd.Timedelta(hours=4),
    "1H":  pd.Timedelta(hours=1),
    "15M": pd.Timedelta(minutes=15),
}

# TFs to compute SMC on for each base TF
SMC_TFS_FOR_BASE: Dict[str, list] = {
    "1H":  ["1W", "1D", "4H", "1H"],
    "15M": ["1W", "1D", "4H", "1H", "15M"],
}

SMC_N_EVENTS = 5   # track last N events per event type


def _smc_col_list(prefix: str, n_events: int, include_internal: bool) -> list[str]:
    """
    Build the list of SMC column names to extract from a compute_smc_features() result.
    All names are fully-prefixed (e.g. "smc_4h_trend").
    """
    p    = prefix + "_"
    cols: list[str] = []

    # Structural trend
    cols.append(f"{p}trend")

    # External structure events (last n_events each)
    for ev in ("choch_bull", "choch_bear", "bos_bull", "bos_bear"):
        for k in range(1, n_events + 1):
            cols.append(f"{p}bars_since_{ev}_{k}")

    # FVG events (last n_events each)
    for ev in ("fvg_bull", "fvg_bear"):
        for k in range(1, n_events + 1):
            cols.append(f"{p}bars_since_{ev}_{k}")

    # Active state flags
    cols += [
        f"{p}ob_bull_in", f"{p}ob_bear_in",
        f"{p}fvg_bull_active", f"{p}fvg_bear_active",
        f"{p}in_premium", f"{p}in_discount", f"{p}zone_pct",
        f"{p}dist_to_sh", f"{p}dist_to_sl",
        f"{p}eq_high", f"{p}eq_low",
    ]

    # Internal structure (same-TF only, higher resolution)
    if include_internal:
        cols.append(f"{p}int_trend")
        for ev in ("int_choch_bull", "int_choch_bear", "int_bos_bull", "int_bos_bear"):
            for k in range(1, n_events + 1):
                cols.append(f"{p}bars_since_{ev}_{k}")

    return cols


def _compute_smc_aligned(
    tf_data: Dict[str, pd.DataFrame],
    base:    pd.DatetimeIndex,
    base_tf: str = "1H",
    n_events: int = SMC_N_EVENTS,
) -> pd.DataFrame:
    """
    Compute SMC features for every TF and align them to the base index.

    For each TF the SMC DataFrame is computed with the appropriate swing length,
    then forward-filled to `base` via merge_asof (shifted by the bar duration so
    only the last CLOSED bar is visible — no look-ahead).

    Parameters
    ----------
    tf_data   : dict {TF → OHLCV DataFrame with indicators}.
    base      : target DatetimeIndex (1H or 15M).
    base_tf   : "1H" or "15M".
    n_events  : number of past events to track per type (default SMC_N_EVENTS=5).

    Returns a DataFrame aligned to `base` with all SMC columns.
    """
    # Import here to avoid circular imports at module load time
    from src.strategy.smc import compute_smc_features  # noqa: PLC0415

    tfs_to_use = SMC_TFS_FOR_BASE.get(base_tf, ["1W", "1D", "4H", "1H"])
    all_parts: list[pd.DataFrame] = []

    for tf in tfs_to_use:
        df_tf = tf_data.get(tf)
        if df_tf is None or df_tf.empty or len(df_tf) < 2 * SMC_SWING_LENGTHS[tf] + 10:
            continue

        swing_len = SMC_SWING_LENGTHS[tf]
        # Same TF as base uses shorter internal swing for extra resolution context
        internal_len = 5
        prefix = f"smc_{tf.lower()}"

        try:
            smc_tf = compute_smc_features(
                df_tf,
                swing_len=swing_len,
                internal_len=internal_len,
                n_last_events=n_events,
                prefix=prefix,
            )
        except Exception:
            continue

        # Build the column list we want to extract
        is_same_tf = (tf == base_tf)
        want_cols  = _smc_col_list(prefix, n_events, include_internal=is_same_tf)
        avail_cols = [c for c in want_cols if c in smc_tf.columns]

        if not avail_cols:
            continue

        # Align to base index using merge_asof (shift by bar duration = last closed bar)
        shift = SMC_TF_SHIFTS[tf]
        if tf == base_tf and shift == pd.Timedelta(0):
            # Same TF with zero shift: direct reindex
            aligned = smc_tf[avail_cols].reindex(base).fillna(0.0)
        else:
            aligned = _align_htf(smc_tf, avail_cols, base, shift)

        all_parts.append(aligned)

    if not all_parts:
        return pd.DataFrame(index=base)

    return pd.concat(all_parts, axis=1)


def build_feature_matrix(
    tf_data:     Dict[str, pd.DataFrame],   # TF → DataFrame with indicators
    signals:     pd.DataFrame,               # bias-free signal matrix (base-TF index)
    base_tf:     str  = "1H",               # base timeframe key: "1H" or "15M"
    include_smc: bool = False,              # add multi-TF SMC features
) -> pd.DataFrame:
    """
    Produce a single DataFrame aligned to the base-TF index containing all
    features needed by the ML gate.

    Parameters
    ----------
    tf_data  : dict with keys '1W', '1D', '4H', '1H' and optionally '15M'
    signals  : output of build_signal_matrix[_15m]() — indexed at base_tf
    base_tf  : '1H' (default) or '15M'

    Returns
    -------
    pd.DataFrame  indexed to base-TF bars, one column per feature
    """
    df_base = tf_data[base_tf]
    df_1h   = tf_data.get("1H", pd.DataFrame())
    df_4h   = tf_data.get("4H", pd.DataFrame())
    df_1d   = tf_data.get("1D", pd.DataFrame())
    df_1w   = tf_data.get("1W", pd.DataFrame())

    base     = df_base.index
    base_pfx = base_tf.lower()   # "1h" or "15m"
    feats    = pd.DataFrame(index=base)

    # ── Base-TF indicators (same TF, no shift needed) ─────────────────────────
    for col in _1H_INDICATORS:
        if col in df_base.columns:
            feats[f"{base_pfx}_{col}"] = df_base[col].reindex(base).fillna(0).values

    # ── Lagged base-TF returns ────────────────────────────────────────────────
    log_ret_base = df_base["log_ret"] if "log_ret" in df_base.columns \
                   else pd.Series(0.0, index=base)
    for i in range(1, 9):
        feats[f"lag_ret_{i}"] = log_ret_base.shift(i).reindex(base).fillna(0).values

    # ── When base is 15M: add 1H as an intermediate HTF layer (shift +1H) ────
    if base_tf == "15M" and not df_1h.empty:
        avail = [c for c in _4H_INDICATORS if c in df_1h.columns]
        if avail:
            htf = _align_htf(df_1h, avail, base, pd.Timedelta(hours=1))
            htf.columns = [f"1h_{c}" for c in htf.columns]
            feats = pd.concat([feats, htf], axis=1)

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
    if ("close" in df_base.columns and "high" in df_base.columns
            and "atr_14" in df_base.columns):
        atr  = df_base["atr_14"]
        hi20 = df_base["high"].rolling(20, min_periods=1).max()
        lo20 = df_base["low"].rolling(20, min_periods=1).min()
        c    = df_base["close"]
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

    # ── Return autocorrelation — use base-TF returns ──────────────────────────
    # Window: 96 bars = ~1 day at 15M, ~4 days at 1H. min_periods=48.
    ac_window = 96 if base_tf == "15M" else 24
    ac_min    = ac_window // 2
    feats["autocorr_lag1"] = (log_ret_base.rolling(ac_window, min_periods=ac_min)
                               .corr(log_ret_base.shift(1))
                               .reindex(base).fillna(0).values)
    feats["autocorr_lag4"] = (log_ret_base.rolling(ac_window, min_periods=ac_min)
                               .corr(log_ret_base.shift(4))
                               .reindex(base).fillna(0).values)

    # ── Multi-TF SMC features (optional) ─────────────────────────────────────
    if include_smc:
        smc_block = _compute_smc_aligned(tf_data, base, base_tf)
        if not smc_block.empty:
            feats = pd.concat([feats, smc_block], axis=1)

    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
    return feats
