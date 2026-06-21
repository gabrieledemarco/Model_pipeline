"""
Multi-timeframe OHLCV data fetcher for BTCUSDT.

Uses Yahoo Finance (BTC-USD) for OHLCV across all timeframes.
Open Interest and Funding Rate are synthetically generated with
realistic statistical properties (trending + divergence events),
matching the distributional characteristics of Binance USDT-perp data.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

SYMBOL = "BTC-USD"

# Timeframe fetch config: (yfinance interval, look-back period)
TF_CONFIG: Dict[str, Tuple[str, str]] = {
    "1W":  ("1wk", "4y"),
    "1D":  ("1d",  "4y"),
    "4H":  ("4h",  "60d"),
    "1H":  ("1h",  "60d"),
    "15M": ("15m", "60d"),
}


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, normalise, drop NaN rows."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_ohlcv(timeframe: str = "1D", symbol: str = SYMBOL) -> pd.DataFrame:
    """Download OHLCV bars for *timeframe* from Yahoo Finance."""
    interval, period = TF_CONFIG[timeframe]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(symbol, interval=interval, period=period,
                          progress=False, auto_adjust=True)
    return _clean(raw)


def fetch_all_timeframes(symbol: str = SYMBOL) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for all configured timeframes. Returns dict keyed by TF label."""
    result: Dict[str, pd.DataFrame] = {}
    for tf in TF_CONFIG:
        try:
            df = fetch_ohlcv(tf, symbol)
            result[tf] = df
            print(f"  [{tf:>3s}]  {len(df):5d} bars  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
        except Exception as exc:
            print(f"  [{tf:>3s}]  FAILED: {exc}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Open Interest
# ─────────────────────────────────────────────────────────────────────────────

def generate_oi(price_series: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic Open Interest correlated with BTCUSDT price.

    Model:
    - OI grows during high-volatility trending phases.
    - OI declines during low-vol choppy phases (position reduction).
    - 12 % of bars produce a divergence event (OI and price move opposite),
      mirroring real liquidation / cascade dynamics.
    - Floor at $5B to prevent degenerate values.
    """
    rng = np.random.default_rng(seed)
    n = len(price_series)
    price = price_series.to_numpy(dtype=float)

    # 20-bar rolling realised vol as trend-strength proxy
    log_ret = np.diff(np.log(price), prepend=np.log(price[0]))
    rvol = np.array([
        float(np.std(log_ret[max(0, i - 19): i + 1])) for i in range(n)
    ])
    median_rvol = np.median(rvol)

    oi = np.empty(n, dtype=float)
    oi[0] = 1e10 * (price[0] / 50_000)  # scale with price level

    for i in range(1, n):
        trending = rvol[i] > median_rvol
        # Base change: +0.5 % in trend, -0.2 % in chop, plus Gaussian noise
        base = (0.005 if trending else -0.002) + rng.normal(0, 0.008)
        if rng.random() < 0.12:   # divergence event
            base = -base
        oi[i] = max(oi[i - 1] * (1.0 + np.clip(base, -0.06, 0.06)), 5e9)

    oi_s = pd.Series(oi, index=price_series.index, name="oi")
    return pd.DataFrame({
        "oi":       oi_s,
        "oi_chg":   oi_s.pct_change().fillna(0),
        "oi_sma7":  oi_s.rolling(7, min_periods=1).mean(),
        "oi_trend": np.sign(oi_s - oi_s.rolling(7, min_periods=1).mean()),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Funding Rate
# ─────────────────────────────────────────────────────────────────────────────

def generate_funding(price_series: pd.Series, seed: int = 42) -> pd.Series:
    """
    Generate synthetic 8-hour funding rates for BTCUSDT perpetuals.

    Properties:
    - Mean-reverting around 0.01 % (neutral / slightly long-biased).
    - Driven by 20-bar price momentum.
    - Clipped to [-0.3 %, +0.3 %] (realistic Binance range).
    """
    rng = np.random.default_rng(seed)
    n = len(price_series)
    price = price_series.to_numpy(dtype=float)

    mom = np.zeros(n, dtype=float)
    for i in range(20, n):
        mom[i] = (price[i] - price[i - 20]) / price[i - 20]

    funding = np.zeros(n, dtype=float)
    funding[0] = 0.0001  # neutral start

    for i in range(1, n):
        rev   = -0.30 * funding[i - 1]        # mean reversion
        trend = 0.01  * mom[i]                # momentum
        noise = rng.normal(0, 0.0001)
        funding[i] = np.clip(funding[i - 1] + rev + trend + noise,
                             -0.003, 0.003)

    return pd.Series(funding, index=price_series.index, name="funding_rate")
