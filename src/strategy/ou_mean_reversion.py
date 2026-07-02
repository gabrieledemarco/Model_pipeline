"""
S07 OU Mean Reversion Signal Generator — BTCUSDT 1H.

Methodology
───────────────────────────────────────────────────────────────────────────────
1. OU (Ornstein-Uhlenbeck) z-score
   For every bar `i`, fit an OLS regression on the trailing `window` (default
   30) log-close bars:  dx_t = alpha + beta * x_{t-1}.
   The fit is only usable when beta < 0 (mean-reverting, kappa = -beta > 0):
       mu       = -alpha / beta            (estimated equilibrium level)
       sigma_eq = std(residuals) / sqrt(-2*beta)   (population std, ddof=0)
       z[i]     = (x[i] - mu) / sigma_eq
   If beta >= 0, or the window is degenerate (near-zero variance / near-zero
   sigma_eq), the bar is left as NaN — "not computable this bar".

2. Entry rule
   LONG  : z[i] < -1.0  AND  close[i] > sma30[i]
   SHORT : z[i] > +1.0  AND  close[i] < sma30[i]
   (mean-reversion trades are only taken in the direction of the intermediate
   trend, per SMA30 — i.e. "buy the dip within an uptrend", not counter-trend).

3. Composite score
   A simple magnitude proxy for logging/display only (NOT used in the entry
   decision, which is purely the z / sma30 threshold rule above):
       composite = -z   on a LONG signal bar   (more negative z → larger +composite)
       composite = -z   on a SHORT signal bar  (more positive z → larger -composite)
   i.e. composite = -z whenever signal != 0, kept at 0.0 when flat — this is
   sign-consistent with `signal` by construction.

This is a loop-based, per-bar OLS fit (deliberately not vectorized): each bar
needs its own independent 30-element regression, mirroring the verified
reference implementation in create_mtf_hmm_report.py (`_ou_zscore()` /
`S07_ou_mean_reversion()`) exactly, to avoid any risk of numerical divergence
from the validated offline result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ou_zscore(log_close: np.ndarray, window: int = 30) -> np.ndarray:
    """Per-bar OU z-score on a trailing `window`-bar log-close series.

    Mirrors create_mtf_hmm_report.py's `_ou_zscore()` (lines 202-214 at time
    of writing) exactly — do not "clean up" the math without re-verifying
    against that reference, this is the validated formula.
    """
    n = len(log_close)
    z = np.full(n, np.nan)
    for i in range(window, n):
        x = log_close[i - window:i]
        dy = x[1:] - x[:-1]
        xx = x[:-1]
        xx_m = xx.mean()
        sxx = np.dot(xx - xx_m, xx - xx_m)
        if sxx < 1e-12:
            continue
        beta = np.dot(xx - xx_m, dy - dy.mean()) / sxx
        if beta >= 0:
            continue          # not mean-reverting this window — skip (leave NaN)
        alpha = dy.mean() - beta * xx_m
        seq = (dy - (alpha + beta * xx)).std() / np.sqrt(-2.0 * beta)
        if seq < 1e-12:
            continue
        z[i] = (x[-1] - (-alpha / beta)) / seq
    return z


def build_ou_signals(df_1h: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Build OU mean-reversion signals from enriched 1H OHLCV data.

    Parameters
    ----------
    df_1h  : DataFrame with OHLCV + indicators from add_indicators()
             (only `close` is required).
    window : rolling window (bars) for both the OU fit and the SMA (default 30).

    Returns
    -------
    pd.DataFrame aligned to df_1h.index with columns:
        signal    : int   {-1, 0, 1}
        composite : float  -z on signal bars, 0.0 when flat
        z         : float  OU z-score, NaN if not computable this bar
        sma30     : float  window-bar simple moving average of close
    """
    idx = df_1h.index
    close = df_1h["close"]

    log_close = np.log(close.values.astype(float) + 1e-9)
    z = _ou_zscore(log_close, window=window)

    sma = close.rolling(window, min_periods=window).mean()
    sma_vals = sma.values

    close_vals = close.values.astype(float)

    with np.errstate(invalid="ignore"):
        long_trigger = (z < -1.0) & (close_vals > sma_vals)
        short_trigger = (z > 1.0) & (close_vals < sma_vals)
    long_trigger = np.nan_to_num(long_trigger, nan=False).astype(bool)
    short_trigger = np.nan_to_num(short_trigger, nan=False).astype(bool)

    signal = np.zeros(len(idx), dtype=int)
    signal[long_trigger] = 1
    signal[short_trigger] = -1

    composite = np.zeros(len(idx), dtype=float)
    composite[long_trigger] = -z[long_trigger]
    composite[short_trigger] = -z[short_trigger]

    return pd.DataFrame(
        {"signal": signal, "composite": composite, "z": z, "sma30": sma_vals},
        index=idx,
    )
