"""
ICT Silver Bullet (NY AM + PM) Signal Generator — BTCUSDT 15M.

Methodology (ported faithfully from the validated research code in
`create_ict_suite_report.py::collect_silver_bullet_events()` — see
`docs/ICT_SILVER_BULLET_STRATEGY.md` for the full write-up and Walk-Forward /
Monte-Carlo validation results: NY AM+PM combined killzone, OOS return
+46.1%, MaxDD -22.4%, P(profit) MC 84.5%).
───────────────────────────────────────────────────────────────────────────────
1. Killzones (UTC, half-open [start, end) )
   NY AM = 14:00-15:00, NY PM = 18:00-19:00 (combined — the validated setup).
   Only bars whose (hour, minute) fall in one of these windows are eligible
   to be the FVG-*forming* bar.

2. Fair Value Gap (FVG) detection
   Bullish FVG at bar i : high[i-1] < low[i+1]
       gap zone: fbot = high[i-1], ftop = low[i+1], fsz = ftop - fbot
   Bearish FVG at bar i : low[i-1] > high[i+1]
       gap zone: ftop = low[i-1],  fbot = high[i+1], fsz = ftop - fbot
   Size filter: fsz >= min_fvg_atr_frac * atr_1h[i]   (default 5% of ATR_1H)

3. Entry (re-entry into the gap)
   Scan forward from j = i+2 up to max_age (default 16) bars:
     Bullish: if close[j] < fbot  -> invalidated, stop scanning (no event)
              elif low[j] <= ftop -> entry fires at j,
                    entry_px = min(close[j], ftop), direction = long
     Bearish: if close[j] > ftop  -> invalidated, stop scanning (no event)
              elif high[j] >= fbot -> entry fires at j,
                    entry_px = max(close[j], fbot), direction = short
   The event (signal, ref_lo/ref_hi/ref_size/atr_1h) is stamped onto the
   *entry* bar j (not the FVG-forming bar i) — that's the bar a live/loop
   consumer would see fire "now".

4. ATR_1H mapping
   `atr_1h` at 15M bar i is the ATR-14 of the most recently *completed* 1H
   bar at or before bar i's timestamp — i.e. the 1H bar that CONTAINS bar i
   is deliberately excluded (it may still be forming); we look one hour back
   from the floor of bar i's timestamp, mirroring the original script's
   `prev_1h = IDX.floor("h") - pd.Timedelta("1h")` / `atr_1h_map` lookup.
   Implemented here via `pd.merge_asof` against df_1h's 1H-indexed ATR series
   (cleaner than replicating the original's raw dict-lookup), which achieves
   an equivalent "most recent completed 1H bar" join.

5. Composite score
   This strategy uses flat scoring (confirmed by the original script: entries
   are gated purely by killzone + FVG structure, no continuous score) —
   composite is 1.0 on a signal bar and 0.0 otherwise.

Exit rule (NOT the standard ATR ladder used elsewhere in this codebase):
   tp_px = entry + direction * 3.0 * ref_size
   sl_px = ref_lo - 0.5*atr_1h  (long)  or  ref_hi + 0.5*atr_1h  (short)
   Sizing floor sl_dist = max(|entry-sl_px|, atr_1h*0.5), leverage cap 5x.
   Max holding 32 bars of 15M (8h) -> time-stop exit.
   This module only emits the signal + reference levels; the actual TP/SL/
   sizing/time-stop math is applied by the live wiring layer
   (src/live/ict_live.py) since it needs live equity/entry-fill context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Killzone windows (UTC) — NY AM + NY PM combined (the validated setup).
KZ_NY_AM    = (14, 0, 15, 0)
KZ_NY_PM    = (18, 0, 19, 0)
KZ_NY_COMBO = [KZ_NY_AM, KZ_NY_PM]


def _map_atr_1h(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.Series:
    """
    For every 15M bar, look up the ATR-14 of the most recently *completed*
    1H bar strictly before the 15M bar's containing hour (i.e. the 1H bar
    whose open_time == floor(ts, '1h') - 1h).

    Equivalent to the original script's:
        prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
        atr_1h_map = df_1h["atr_14"].clip(lower=1.0).to_dict()
        ATR_1H     = [atr_1h_map.get(t, nan) for t in prev_1h]  (+ 15M fallback)
    """
    atr_1h_series = df_1h["atr_14"].clip(lower=1.0).sort_index()
    atr_1h_series.index = atr_1h_series.index.astype("datetime64[ns]")
    prev_1h = (df_15m.index.floor("h") - pd.Timedelta("1h")).astype("datetime64[ns]")

    lookup = pd.DataFrame({"prev_1h": prev_1h}, index=df_15m.index)
    mapped = pd.merge_asof(
        lookup.sort_values("prev_1h"),
        atr_1h_series.rename("atr_1h").to_frame(),
        left_on="prev_1h", right_index=True,
        direction="backward",
    )
    mapped = mapped.reindex(lookup.index)  # merge_asof sorted -> restore original order
    # Require an *exact* hour match (mirrors dict.get(t, nan) — no stale reuse
    # across gaps); merge_asof(direction="backward") with an exact key present
    # in atr_1h_series will match exactly when present, else fall back to nan.
    exact = mapped["prev_1h"].isin(atr_1h_series.index)
    atr_1h = mapped["atr_1h"].where(exact)

    # Fallback: 15M's own ATR-14 (clipped), matching the original script's
    # `fallback = df_15m["atr_14"].clip(lower=1.0).values`.
    if "atr_14" in df_15m.columns:
        fallback = df_15m["atr_14"].clip(lower=1.0)
    else:
        fallback = pd.Series(1.0, index=df_15m.index)
    atr_1h = atr_1h.fillna(fallback)
    atr_1h.index = df_15m.index
    return atr_1h


def build_silver_bullet_signals(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    min_fvg_atr_frac: float = 0.05,
    max_age: int = 16,
    killzones: list[tuple[int, int, int, int]] | None = None,
) -> pd.DataFrame:
    """
    Build ICT Silver Bullet (FVG-in-killzone) signals from 15M OHLCV, using
    ATR-14 computed on 1H bars for FVG-size filtering and stop distance.

    Parameters
    ----------
    df_15m           : 15M OHLCV with `atr_14` column (from add_indicators()).
    df_1h            : 1H OHLCV with `atr_14` column (from add_indicators()).
    min_fvg_atr_frac : minimum FVG width as a fraction of ATR_1H (default 0.05).
    max_age          : max bars to wait for re-entry into the gap (default 16 = 4h).
    killzones        : list of (start_h, start_m, end_h, end_m) UTC windows;
                       defaults to NY AM+PM combined (the validated setup).

    Returns
    -------
    pd.DataFrame aligned to df_15m.index with columns:
      signal    : int   {-1, 0, 1}
      composite : float always 1.0 on a signal bar, else 0.0 (flat scoring)
      ref_lo    : float FVG lower edge (NaN when signal==0)
      ref_hi    : float FVG upper edge (NaN when signal==0)
      ref_size  : float FVG width = ref_hi - ref_lo (NaN when signal==0)
      atr_1h    : float ATR_1H at the entry bar (NaN when signal==0)
    """
    if killzones is None:
        killzones = KZ_NY_COMBO

    idx = df_15m.index
    n   = len(df_15m)

    hi = df_15m["high"].to_numpy(dtype=float)
    lo = df_15m["low"].to_numpy(dtype=float)
    cl = df_15m["close"].to_numpy(dtype=float)

    h_arr = np.array(idx.hour,   dtype=int)
    m_arr = np.array(idx.minute, dtype=int)

    atr_1h = _map_atr_1h(df_15m, df_1h).to_numpy(dtype=float)

    def in_killzone(i: int) -> bool:
        h, m = h_arr[i], m_arr[i]
        t = h * 60 + m
        for (sh, sm, eh, em) in killzones:
            if (sh * 60 + sm) <= t < (eh * 60 + em):
                return True
        return False

    signal    = np.zeros(n, dtype=int)
    composite = np.zeros(n, dtype=float)
    ref_lo    = np.full(n, np.nan)
    ref_hi    = np.full(n, np.nan)
    ref_size  = np.full(n, np.nan)
    ref_atr   = np.full(n, np.nan)

    # Candidate FVG-forming bar i needs i-1 and i+1 already closed (so i
    # ranges over [1, n-2]); the re-entry scan then looks forward from i+2,
    # capped by max_age and by the array bounds.
    for i in range(1, n - 1):
        if not in_killzone(i):
            continue
        atr = atr_1h[i]
        if np.isnan(atr) or atr <= 0:
            continue

        scan_end = min(i + 2 + max_age, n)

        # Bullish FVG
        if hi[i - 1] < lo[i + 1]:
            fbot, ftop = hi[i - 1], lo[i + 1]
            fsz = ftop - fbot
            if fsz >= min_fvg_atr_frac * atr:
                for j in range(i + 2, scan_end):
                    if cl[j] < fbot:
                        break  # invalidated
                    if lo[j] <= ftop:
                        signal[j]    = 1
                        composite[j] = 1.0
                        ref_lo[j]    = fbot
                        ref_hi[j]    = ftop
                        ref_size[j]  = fsz
                        ref_atr[j]   = atr_1h[j]
                        break

        # Bearish FVG
        if lo[i - 1] > hi[i + 1]:
            ftop, fbot = lo[i - 1], hi[i + 1]
            fsz = ftop - fbot
            if fsz >= min_fvg_atr_frac * atr:
                for j in range(i + 2, scan_end):
                    if cl[j] > ftop:
                        break  # invalidated
                    if hi[j] >= fbot:
                        signal[j]    = -1
                        composite[j] = 1.0
                        ref_lo[j]    = fbot
                        ref_hi[j]    = ftop
                        ref_size[j]  = fsz
                        ref_atr[j]   = atr_1h[j]
                        break

    return pd.DataFrame(
        {
            "signal": signal,
            "composite": composite,
            "ref_lo": ref_lo,
            "ref_hi": ref_hi,
            "ref_size": ref_size,
            "atr_1h": ref_atr,
        },
        index=idx,
    )
