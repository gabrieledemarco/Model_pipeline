"""
Daily Volume Profile — POC/Value-Area/shape classification + real order-flow
(CVD/delta) features, built from intraday bars with real taker_buy_base
(Binance Vision klines — NOT the synthetic generate_oi/generate_funding
fallbacks in data_fetcher.py).

Used by create_daily_volume_profile_report.py (Daily Volume Profile
Exhaustion Reversal strategy research).

Causality contract
───────────────────
Each daily profile row is indexed by (day + 1 calendar day, 00:00 UTC) —
i.e. the timestamp at which that day's profile FIRST becomes fully known,
not the day it describes. This lets src.strategy.mtf_swing.align_htf_to_ltf
(backward as-of merge, already used for the 30m→15m bias in
create_ltf_smc_mtf_report.py) pick it up correctly: an intraday bar at any
point during day D+1 sees day D's completed profile, never its own
in-progress one. Indexing by day D's own open (the "natural" OHLCV
convention) would reintroduce the exact lookahead bug class documented in
docs/VALIDATED_STRATEGIES_SPEC.md's HMM 4H ffill correction — a backward
as-of merge against a same-day-open-indexed row would match day D's own
(still-incomplete) profile for intraday bars later that same day.

Profile construction
──────────────────────
Volume-at-price via each bar's typical price ((H+L+C)/3) binned into
`n_bins` equal-width bins across the day's [low, high] — the standard
simplified/fast approach (vs. distributing each bar's volume across its
full H-L range), appropriate at 5m granularity (~288 bars/day).

Shape classification (top/mid/bottom thirds of the day's price range, by
volume share):
    b-shape  : bottom-heavy (>=40% volume in bottom third, <=25% in top)
               → value was accepted low; POI = demand zone [day_low, VAL]
    P-shape  : top-heavy (mirror of b-shape)
               → POI = supply zone [VAH, day_high]
    B-shape  : bimodal (two local volume maxima separated by a valley
               bin < 50% of the smaller peak) → balance/rotation, no
               exhaustion-reversal edge — excluded from the strategy
    D-shape  : single central POC, none of the above → balanced/normal
               day — also excluded (a directionally skewed prior day is a
               precondition for a "trend exhaustion" trade to mean anything)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 50
VALUE_AREA_PCT = 0.70
SHAPE_HEAVY_THRESHOLD = 0.40   # bottom/top third volume share to call b/P-shape
SHAPE_THIN_THRESHOLD = 0.25    # opposite third volume share must be <= this
BIMODAL_VALLEY_RATIO = 0.50    # valley bin must be < this fraction of the smaller peak


def _value_area(bin_vol: np.ndarray, poc_idx: int, target_pct: float) -> tuple[int, int]:
    """Classic expand-from-POC value-area algorithm: grow [lo, hi] outward,
    each step adding whichever adjacent side has more volume, until
    cumulative volume >= target_pct of total."""
    total = bin_vol.sum()
    if total <= 0:
        return poc_idx, poc_idx
    lo = hi = poc_idx
    acc = bin_vol[poc_idx]
    n = len(bin_vol)
    while acc < target_pct * total and (lo > 0 or hi < n - 1):
        left_vol = bin_vol[lo - 1] if lo > 0 else -1.0
        right_vol = bin_vol[hi + 1] if hi < n - 1 else -1.0
        if right_vol >= left_vol:
            hi += 1
            acc += bin_vol[hi]
        else:
            lo -= 1
            acc += bin_vol[lo]
    return lo, hi


def _classify_shape(bin_vol: np.ndarray) -> tuple[str, float, float, float]:
    n = len(bin_vol)
    third = max(1, n // 3)
    bottom_pct = bin_vol[:third].sum() / bin_vol.sum() if bin_vol.sum() > 0 else 0.0
    top_pct = bin_vol[-third:].sum() / bin_vol.sum() if bin_vol.sum() > 0 else 0.0
    mid_pct = 1.0 - bottom_pct - top_pct

    if bottom_pct >= SHAPE_HEAVY_THRESHOLD and top_pct <= SHAPE_THIN_THRESHOLD:
        return "b", top_pct, mid_pct, bottom_pct
    if top_pct >= SHAPE_HEAVY_THRESHOLD and bottom_pct <= SHAPE_THIN_THRESHOLD:
        return "P", top_pct, mid_pct, bottom_pct

    # Bimodal check: local maxima of a SMOOTHED profile (raw 50-bin histograms
    # are noisy enough that almost every day has some bump-neighbor-bump by
    # chance — smoothing + minimum peak separation + minimum peak mass avoids
    # over-labeling ordinary D-shape days as bimodal).
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(bin_vol, kernel, mode="same")
    min_sep = max(3, n // 6)
    min_peak_mass = 0.12 * bin_vol.sum()
    peaks = [i for i in range(1, n - 1)
             if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
             and bin_vol[i] >= min_peak_mass]
    if len(peaks) >= 2:
        peaks_sorted = sorted(peaks, key=lambda i: -smoothed[i])
        p1 = peaks_sorted[0]
        p2 = next((p for p in peaks_sorted[1:] if abs(p - p1) >= min_sep), None)
        if p2 is not None:
            lo, hi = sorted([p1, p2])
            valley = bin_vol[lo:hi + 1].min()
            smaller_peak = min(bin_vol[p1], bin_vol[p2])
            if smaller_peak > 0 and valley < BIMODAL_VALLEY_RATIO * smaller_peak:
                return "B", top_pct, mid_pct, bottom_pct

    return "D", top_pct, mid_pct, bottom_pct


def compute_daily_profiles(
    df_intraday: pd.DataFrame, n_bins: int = N_BINS, value_area_pct: float = VALUE_AREA_PCT,
) -> pd.DataFrame:
    """
    Build one volume profile per calendar UTC day from intraday OHLCV(+taker_buy_base).

    df_intraday must have columns: open, high, low, close, volume, and
    optionally taker_buy_base (real order-flow — from
    src.strategy.data_fetcher.fetch_binance_vision_taker_flow; if absent,
    buy/sell split columns are NaN and downstream order-flow features
    degrade gracefully to "unavailable", never silently fabricated).

    Returns a DataFrame indexed by (day + 1 day, 00:00) — see module
    docstring's Causality contract — with columns: day (the date it
    describes), day_open/high/low/close, poc, vah, val, shape,
    top_pct/mid_pct/bottom_pct, demand_lo/demand_hi, supply_lo/supply_hi.
    """
    has_flow = "taker_buy_base" in df_intraday.columns
    typical_full = (df_intraday["high"] + df_intraday["low"] + df_intraday["close"]) / 3.0
    day_key = df_intraday.index.floor("D")

    rows = []
    for day, g in df_intraday.groupby(day_key):
        if len(g) < 20:   # partial/incomplete day (e.g. first or last in the fetched range)
            continue
        day_low, day_high = float(g["low"].min()), float(g["high"].max())
        if day_high <= day_low:
            continue

        tp = typical_full.loc[g.index].values
        vol = g["volume"].values.astype(float)
        edges = np.linspace(day_low, day_high, n_bins + 1)
        bin_idx = np.clip(np.digitize(tp, edges) - 1, 0, n_bins - 1)

        bin_vol = np.zeros(n_bins)
        np.add.at(bin_vol, bin_idx, vol)
        if has_flow:
            buy_vol = g["taker_buy_base"].values.astype(float)
            bin_buy = np.zeros(n_bins)
            np.add.at(bin_buy, bin_idx, buy_vol)
        else:
            bin_buy = np.full(n_bins, np.nan)

        if bin_vol.sum() <= 0:
            continue
        poc_idx = int(np.argmax(bin_vol))
        va_lo_idx, va_hi_idx = _value_area(bin_vol, poc_idx, value_area_pct)
        bin_mid = (edges[:-1] + edges[1:]) / 2.0

        shape, top_pct, mid_pct, bottom_pct = _classify_shape(bin_vol)

        poc = float(bin_mid[poc_idx])
        vah = float(bin_mid[va_hi_idx])
        val = float(bin_mid[va_lo_idx])
        day_delta = float(2 * np.nansum(bin_buy) - bin_vol.sum()) if has_flow else float("nan")

        demand_lo, demand_hi = (day_low, val) if shape == "b" else (np.nan, np.nan)
        supply_lo, supply_hi = (vah, day_high) if shape == "P" else (np.nan, np.nan)

        rows.append(dict(
            profile_ts=day + pd.Timedelta(days=1),
            day=day, day_open=float(g["open"].iloc[0]), day_high=day_high,
            day_low=day_low, day_close=float(g["close"].iloc[-1]),
            poc=poc, vah=vah, val=val,
            shape=shape, top_pct=top_pct, mid_pct=mid_pct, bottom_pct=bottom_pct,
            day_delta=day_delta,
            demand_lo=demand_lo, demand_hi=demand_hi,
            supply_lo=supply_lo, supply_hi=supply_hi,
        ))

    out = pd.DataFrame(rows).set_index("profile_ts").sort_index()
    # Adding a plain Timedelta to a ms/us-precision DatetimeIndex (day_key's
    # dtype, inherited from df_intraday.index) silently upcasts the result
    # to a different time unit — cast back so align_htf_to_ltf's merge_asof
    # (which requires matching key dtypes) doesn't choke on it downstream.
    out.index = out.index.astype(df_intraday.index.dtype)
    return out


def compute_bar_delta(df_intraday: pd.DataFrame) -> pd.Series:
    """Per-bar order-flow delta = buy_volume - sell_volume, from real
    taker_buy_base (buy_vol - (total_vol - buy_vol) = 2*buy_vol - total_vol).
    NaN if taker_buy_base isn't available (never fabricated)."""
    if "taker_buy_base" not in df_intraday.columns:
        return pd.Series(np.nan, index=df_intraday.index, name="delta")
    return (2 * df_intraday["taker_buy_base"] - df_intraday["volume"]).rename("delta")
