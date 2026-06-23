"""
Smart Money Concepts (SMC) features — Python implementation.

Ported and simplified from LuxAlgo Smart Money Concepts (Pine Script v5).
All features are strictly point-in-time with no lookahead bias:
  - Pivot detection confirmed n bars after the actual swing via rolling max/min.
  - Structure breaks detected on the close of the crossing bar.
  - FVGs and OBs use only past data at each evaluation point.

Key concepts
────────────
  BOS   (Break of Structure)   – close breaks last swing in trend direction
  CHoCH (Change of Character)  – close breaks last swing AGAINST current trend
  Order Blocks                 – last bearish candle before bullish BOS/CHoCH
                                  last bullish candle before bearish BOS/CHoCH
  Fair Value Gaps              – 3-bar imbalance (low[i] > high[i-2] / high[i] < low[i-2])
  Premium / Discount / Equilibrium – zones based on last swing high/low range
  Equal Highs / Equal Lows    – two pivots within threshold × ATR

Public API
──────────
  compute_smc_features(df, swing_len=50, internal_len=5, prefix="smc")
      → pd.DataFrame  aligned to df.index

  smc_composite_score(smc_feats, prefix="smc")
      → pd.Series  ∈ [-3, +3]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.indicators import atr as _atr_fn


# ─────────────────────────────────────────────────────────────────────────────
# Pivot detection  (vectorized, lookahead-free)
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_high(high: pd.Series, n: int) -> pd.Series:
    """
    True at bar T when high[T-n] is the maximum over the window [T-2n, T].
    Confirmation lag = n bars; no future data used.
    """
    w  = 2 * n + 1
    rm = high.rolling(w, min_periods=w).max()
    return (high.shift(n) >= rm) & rm.notna()


def _pivot_low(low: pd.Series, n: int) -> pd.Series:
    """True at bar T when low[T-n] is the minimum over the window [T-2n, T]."""
    w  = 2 * n + 1
    rm = low.rolling(w, min_periods=w).min()
    return (low.shift(n) <= rm) & rm.notna()


# ─────────────────────────────────────────────────────────────────────────────
# Structure scan  (sequential O(N) loop)
# ─────────────────────────────────────────────────────────────────────────────

def _structure_scan(
    high:   np.ndarray,
    low:    np.ndarray,
    close:  np.ndarray,
    open_:  np.ndarray,
    ph:     np.ndarray,  # bool – confirmed pivot high at index i
    pl:     np.ndarray,  # bool – confirmed pivot low  at index i
    n:      int,         # confirmation lag (pivot is at i-n)
    max_ob: int = 3,     # max active order blocks to track
) -> dict[str, np.ndarray]:
    """
    Sequential bar scan: detect BOS / CHoCH events and track order blocks.
    At bar i, only data from bars ≤ i is used.
    """
    N = len(close)

    trend      = np.zeros(N, dtype=np.int8)
    bos_bull   = np.zeros(N, dtype=np.int8)
    bos_bear   = np.zeros(N, dtype=np.int8)
    choch_bull = np.zeros(N, dtype=np.int8)
    choch_bear = np.zeros(N, dtype=np.int8)
    sh_arr     = np.full(N, np.nan)
    sl_arr     = np.full(N, np.nan)
    ob_bull_lo = np.full(N, np.nan)
    ob_bull_hi = np.full(N, np.nan)
    ob_bear_lo = np.full(N, np.nan)
    ob_bear_hi = np.full(N, np.nan)
    ob_bull_in = np.zeros(N, dtype=np.int8)
    ob_bear_in = np.zeros(N, dtype=np.int8)

    cur_trend = 0
    cur_sh    = np.nan   # active swing high level being watched
    cur_sl    = np.nan   # active swing low  level being watched

    bull_obs: list[tuple[float, float]] = []  # (lo, hi) demand zones
    bear_obs: list[tuple[float, float]] = []  # (lo, hi) supply zones

    for i in range(N):
        # Update pending swing levels from newly confirmed pivots
        if ph[i] and i >= n:
            cur_sh = high[i - n]   # pivot price confirmed now
        if pl[i] and i >= n:
            cur_sl = low[i - n]

        c = close[i]
        broke_up   = (not np.isnan(cur_sh)) and c > cur_sh
        broke_down = (not np.isnan(cur_sl)) and c < cur_sl

        # ── Classify structural break ─────────────────────────────────────
        if broke_up:
            if cur_trend <= 0:
                choch_bull[i] = 1
                cur_trend = 1
            else:
                bos_bull[i] = 1
            # Bullish OB: last bearish candle before this break (look back ≤30 bars)
            for j in range(i - 1, max(i - 30, -1), -1):
                if close[j] < open_[j]:
                    bull_obs.append((low[j], high[j]))
                    if len(bull_obs) > max_ob:
                        bull_obs.pop(0)
                    break
            cur_sh = np.nan   # level consumed

        if broke_down:
            if cur_trend >= 0:
                choch_bear[i] = 1
                cur_trend = -1
            else:
                bos_bear[i] = 1
            # Bearish OB: last bullish candle before this break
            for j in range(i - 1, max(i - 30, -1), -1):
                if close[j] > open_[j]:
                    bear_obs.append((low[j], high[j]))
                    if len(bear_obs) > max_ob:
                        bear_obs.pop(0)
                    break
            cur_sl = np.nan

        # ── Deactivate mitigated OBs ──────────────────────────────────────
        # Bull OB expires when price closes below its low
        bull_obs = [(lo, hi) for lo, hi in bull_obs if c >= lo]
        # Bear OB expires when price closes above its high
        bear_obs = [(lo, hi) for lo, hi in bear_obs if c <= hi]

        # ── Record state ──────────────────────────────────────────────────
        trend[i]  = cur_trend
        sh_arr[i] = cur_sh
        sl_arr[i] = cur_sl

        if bull_obs:
            lo_, hi_ = bull_obs[-1]
            ob_bull_lo[i] = lo_
            ob_bull_hi[i] = hi_
            ob_bull_in[i] = 1 if lo_ <= c <= hi_ else 0

        if bear_obs:
            lo_, hi_ = bear_obs[-1]
            ob_bear_lo[i] = lo_
            ob_bear_hi[i] = hi_
            ob_bear_in[i] = 1 if lo_ <= c <= hi_ else 0

    return dict(
        trend=trend,
        bos_bull=bos_bull, bos_bear=bos_bear,
        choch_bull=choch_bull, choch_bear=choch_bear,
        sh_level=sh_arr, sl_level=sl_arr,
        ob_bull_lo=ob_bull_lo, ob_bull_hi=ob_bull_hi, ob_bull_in=ob_bull_in,
        ob_bear_lo=ob_bear_lo, ob_bear_hi=ob_bear_hi, ob_bear_in=ob_bear_in,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fair Value Gaps  (vectorized + sequential fill tracking)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fvg(
    high:        pd.Series,
    low:         pd.Series,
    close:       pd.Series,
    atr:         pd.Series,
    min_atr_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Detect bullish/bearish FVGs and track whether the gap has been filled.

    Bullish FVG at bar i: low[i] > high[i-2]   (gap up)
    Bearish FVG at bar i: high[i] < low[i-2]   (gap down)
    """
    idx = high.index
    N   = len(idx)

    fvg_bull_size = (low  - high.shift(2)).clip(lower=0.0)
    fvg_bear_size = (low.shift(2) - high).clip(lower=0.0)

    bull_flag = (fvg_bull_size > min_atr_pct * atr)
    bear_flag = (fvg_bear_size > min_atr_pct * atr)

    # FVG zone boundaries
    bflo = high.shift(2).where(bull_flag)   # bull FVG low  = prior bar's high
    bfhi = low.where(bull_flag)              # bull FVG high = current bar's low
    brlo = high.where(bear_flag)             # bear FVG low  = current bar's high
    brhi = low.shift(2).where(bear_flag)     # bear FVG high = prior bar's low

    fvg_bull_act = np.zeros(N, dtype=np.int8)
    fvg_bear_act = np.zeros(N, dtype=np.int8)

    bf  = bull_flag.values
    brf = bear_flag.values
    bflo_a = bflo.values;  bfhi_a = bfhi.values
    brlo_a = brlo.values;  brhi_a = brhi.values
    c_a = close.values;    l_a = low.values;  h_a = high.values

    cur_bull: tuple[float, float] | None = None
    cur_bear: tuple[float, float] | None = None

    for i in range(N):
        # Register new FVGs
        if bf[i] and not np.isnan(bflo_a[i]):
            cur_bull = (float(bflo_a[i]), float(bfhi_a[i]))
        if brf[i] and not np.isnan(brlo_a[i]):
            cur_bear = (float(brlo_a[i]), float(brhi_a[i]))

        # Deactivate when filled
        if cur_bull is not None and l_a[i] < cur_bull[0]:
            cur_bull = None
        if cur_bear is not None and h_a[i] > cur_bear[1]:
            cur_bear = None

        # Active = price inside the gap
        if cur_bull is not None:
            fvg_bull_act[i] = 1 if cur_bull[0] <= c_a[i] <= cur_bull[1] else 0
        if cur_bear is not None:
            fvg_bear_act[i] = 1 if cur_bear[0] <= c_a[i] <= cur_bear[1] else 0

    return pd.DataFrame({
        "fvg_bull":        bull_flag.astype(np.int8).values,
        "fvg_bear":        bear_flag.astype(np.int8).values,
        "fvg_bull_active": fvg_bull_act,
        "fvg_bear_active": fvg_bear_act,
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Premium / Discount / Equilibrium zones  (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_zones(
    close:    pd.Series,
    sh_level: np.ndarray,
    sl_level: np.ndarray,
) -> pd.DataFrame:
    """Zone membership based on the last swing high/low range."""
    sh  = pd.Series(sh_level, index=close.index).ffill()
    sl  = pd.Series(sl_level, index=close.index).ffill()
    rng = (sh - sl).clip(lower=1e-8)
    pct = ((close - sl) / rng).clip(0.0, 1.0).fillna(0.5)

    return pd.DataFrame({
        "in_premium":     (pct > 0.75).astype(np.int8),
        "in_discount":    (pct < 0.25).astype(np.int8),
        "in_equilibrium": ((pct >= 0.40) & (pct <= 0.60)).astype(np.int8),
        "zone_pct":       pct,
    }, index=close.index)


# ─────────────────────────────────────────────────────────────────────────────
# Equal Highs / Equal Lows  (sequential)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_equal_levels(
    high:      pd.Series,
    low:       pd.Series,
    atr:       pd.Series,
    ph:        np.ndarray,
    pl:        np.ndarray,
    n:         int,
    threshold: float = 0.125,
) -> pd.DataFrame:
    """Two consecutive pivots within threshold × ATR → equal high/low."""
    N = len(high)
    eq_high = np.zeros(N, dtype=np.int8)
    eq_low  = np.zeros(N, dtype=np.int8)

    high_a = high.values;  low_a = low.values;  atr_a = atr.values

    prev_sh = np.nan;  prev_sl = np.nan

    for i in range(N):
        if ph[i] and i >= n:
            cur_sh = high_a[i - n]
            if not np.isnan(prev_sh) and abs(cur_sh - prev_sh) <= threshold * atr_a[i]:
                eq_high[i] = 1
            prev_sh = cur_sh
        if pl[i] and i >= n:
            cur_sl = low_a[i - n]
            if not np.isnan(prev_sl) and abs(cur_sl - prev_sl) <= threshold * atr_a[i]:
                eq_low[i] = 1
            prev_sl = cur_sl

    return pd.DataFrame({"eq_high": eq_high, "eq_low": eq_low}, index=high.index)


# ─────────────────────────────────────────────────────────────────────────────
# Distance features  (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def _distance_features(
    close:     pd.Series,
    atr:       pd.Series,
    sh_level:  np.ndarray,
    sl_level:  np.ndarray,
    ob_bull_lo: np.ndarray,
    ob_bull_hi: np.ndarray,
    ob_bear_lo: np.ndarray,
    ob_bear_hi: np.ndarray,
) -> pd.DataFrame:
    c   = close.values
    a   = atr.values + 1e-8

    dist_sh = (sh_level - c) / a   # +ve → swing high above close
    dist_sl = (c - sl_level) / a   # +ve → swing low  below close

    ob_bull_mid = (ob_bull_lo + ob_bull_hi) / 2.0
    ob_bear_mid = (ob_bear_lo + ob_bear_hi) / 2.0

    # +ve → close above OB mid (not in zone); -ve → below OB mid (in zone/below)
    d_ob_bull = np.where(np.isnan(ob_bull_mid), 0.0, (c - ob_bull_mid) / a)
    d_ob_bear = np.where(np.isnan(ob_bear_mid), 0.0, (ob_bear_mid - c) / a)

    idx = close.index
    return pd.DataFrame({
        "dist_to_sh":      dist_sh,
        "dist_to_sl":      dist_sl,
        "dist_to_ob_bull": d_ob_bull,
        "dist_to_ob_bear": d_ob_bear,
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Bars-since helper  (sequential)
# ─────────────────────────────────────────────────────────────────────────────

def _bars_since(arr: np.ndarray, cap: int) -> np.ndarray:
    """Number of bars since last nonzero value; cap at `cap` if never happened."""
    N   = len(arr)
    res = np.full(N, cap, dtype=np.float32)
    cnt = cap
    for i in range(N):
        if arr[i]:
            cnt = 0
        res[i] = cnt
        if cnt < cap:
            cnt += 1
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def compute_smc_features(
    df:           pd.DataFrame,
    swing_len:    int   = 50,
    internal_len: int   = 5,
    fvg_min_atr:  float = 0.0,
    eq_threshold: float = 0.125,
    prefix:       str   = "smc",
) -> pd.DataFrame:
    """
    Compute all SMC features for an OHLCV DataFrame.

    df must have columns: open, high, low, close, volume.
    If 'atr_14' is present it is reused; otherwise ATR(14) is computed.

    Parameters
    ----------
    swing_len    : confirmation lag for external pivots (default 50).
    internal_len : confirmation lag for internal pivots (default 5).
    fvg_min_atr  : minimum FVG size as fraction of ATR (0 = any gap).
    eq_threshold : equal-level detection tolerance (× ATR).
    prefix       : column prefix; "" for no prefix, "smc" → "smc_bos_bull".

    Returns a DataFrame with ~25 SMC feature columns, same index as df.
    """
    h = df["high"];   l = df["low"]
    c = df["close"];  o = df["open"]
    a = df["atr_14"] if "atr_14" in df.columns else _atr_fn(h, l, c, 14)

    # ── Pivot detection ────────────────────────────────────────────────────
    ph_ext = _pivot_high(h, swing_len)
    pl_ext = _pivot_low(l, swing_len)
    ph_int = _pivot_high(h, internal_len)
    pl_int = _pivot_low(l, internal_len)

    # ── Structure scans ────────────────────────────────────────────────────
    ext = _structure_scan(
        h.values, l.values, c.values, o.values,
        ph_ext.values, pl_ext.values, swing_len,
    )
    int_ = _structure_scan(
        h.values, l.values, c.values, o.values,
        ph_int.values, pl_int.values, internal_len,
    )

    # ── FVGs ──────────────────────────────────────────────────────────────
    fvg = _compute_fvg(h, l, c, a, min_atr_pct=fvg_min_atr)

    # ── Zones ─────────────────────────────────────────────────────────────
    zones = _compute_zones(c, ext["sh_level"], ext["sl_level"])

    # ── Equal levels ──────────────────────────────────────────────────────
    eq = _compute_equal_levels(h, l, a, ph_ext.values, pl_ext.values,
                                swing_len, eq_threshold)

    # ── Distance features ─────────────────────────────────────────────────
    dists = _distance_features(
        c, a,
        ext["sh_level"], ext["sl_level"],
        ext["ob_bull_lo"], ext["ob_bull_hi"],
        ext["ob_bear_lo"], ext["ob_bear_hi"],
    )

    # ── Recency (bars since last event) ───────────────────────────────────
    idx = df.index
    cap = len(idx)
    bs_bos_bull   = _bars_since(ext["bos_bull"],   cap)
    bs_bos_bear   = _bars_since(ext["bos_bear"],   cap)
    bs_choch_bull = _bars_since(ext["choch_bull"], cap)
    bs_choch_bear = _bars_since(ext["choch_bear"], cap)

    # ── Assemble output ───────────────────────────────────────────────────
    p   = (prefix + "_") if prefix else ""
    out = pd.DataFrame(index=idx)

    # External structure
    for k in ("trend", "bos_bull", "bos_bear", "choch_bull", "choch_bear",
               "ob_bull_in", "ob_bear_in"):
        out[f"{p}{k}"] = ext[k]

    # Internal structure
    for k in ("trend", "bos_bull", "bos_bear", "choch_bull", "choch_bear"):
        out[f"{p}int_{k}"] = int_[k]

    # Recency
    out[f"{p}bars_since_bos_bull"]   = bs_bos_bull
    out[f"{p}bars_since_bos_bear"]   = bs_bos_bear
    out[f"{p}bars_since_choch_bull"] = bs_choch_bull
    out[f"{p}bars_since_choch_bear"] = bs_choch_bear

    # FVGs
    for col in fvg.columns:
        out[f"{p}{col}"] = fvg[col].values

    # Zones
    for col in zones.columns:
        out[f"{p}{col}"] = zones[col].values

    # Equal levels
    out[f"{p}eq_high"] = eq["eq_high"].values
    out[f"{p}eq_low"]  = eq["eq_low"].values

    # Distances
    for col in dists.columns:
        out[f"{p}{col}"] = dists[col].values

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Composite signal score from SMC features
# ─────────────────────────────────────────────────────────────────────────────

def smc_composite_score(smc_feats: pd.DataFrame, prefix: str = "smc") -> pd.Series:
    """
    Convert SMC binary events into a continuous score ∈ [-3, +3].

    Scoring
    ───────
      +1 / -1  structural trend (+1 bull, -1 bear)
      +1 / -1  recent CHoCH (within last 10 bars)
      +0.5/-0.5 price in discount / premium zone
      +0.5/-0.5 price inside active bull / bear order block

    Maximum absolute value = 3.0.
    """
    p = (prefix + "_") if prefix else ""

    def _get(name: str) -> pd.Series:
        col = f"{p}{name}"
        if col in smc_feats.columns:
            return smc_feats[col].fillna(0.0)
        return pd.Series(0.0, index=smc_feats.index)

    s = pd.Series(0.0, index=smc_feats.index)

    s += _get("trend").astype(float)

    choch_bull_recent = (_get("bars_since_choch_bull") <= 10).astype(float)
    choch_bear_recent = (_get("bars_since_choch_bear") <= 10).astype(float)
    s += choch_bull_recent
    s -= choch_bear_recent

    s += _get("in_discount").astype(float) * 0.5
    s -= _get("in_premium").astype(float)  * 0.5

    s += _get("ob_bull_in").astype(float) * 0.5
    s -= _get("ob_bear_in").astype(float) * 0.5

    return s.clip(-3.0, 3.0).rename("s_smc")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone SMC trend signal (for Path C / simple trend following)
# ─────────────────────────────────────────────────────────────────────────────

def smc_trend_signal(smc_feats: pd.DataFrame, prefix: str = "smc") -> pd.Series:
    """
    Pure structure-based directional signal: +1 bull, -1 bear, 0 neutral.
    Transitions on CHoCH; stays flat until a CHoCH is confirmed.
    """
    p = (prefix + "_") if prefix else ""
    trend_col = f"{p}trend"
    if trend_col in smc_feats.columns:
        return smc_feats[trend_col].astype(float).rename("s_smc_trend")
    return pd.Series(0.0, index=smc_feats.index, name="s_smc_trend")
