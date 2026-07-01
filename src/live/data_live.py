"""
Live multi-timeframe data fetcher.

Pulls OHLCV from Binance production FAPI (no API key required) and
feeds it into the same indicator / signal pipeline used for backtesting,
ensuring the live signal computation is byte-for-byte identical.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Make sure parent package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.indicators import add_indicators
from src.strategy.signals    import build_signal_matrix
from src.strategy.optimizer  import SCENARIOS, apply_filters
from src.strategy.data_fetcher import generate_oi, generate_funding

FAPI        = "https://fapi.binance.com"
SYMBOL      = "BTCUSDT"
SCENARIO    = "Strong (≥±18)"

KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


# ── OHLCV from FAPI (production, no key) ─────────────────────────────────────

def _klines(interval: str, limit: int, symbol: str = SYMBOL) -> pd.DataFrame:
    r = requests.get(
        f"{FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=KLINES_COLS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_tf_data(symbol: str = SYMBOL) -> Dict[str, pd.DataFrame]:
    """
    Fetch and resample multi-TF OHLCV.
    Returns dict keyed: '1H', '4H', '1D', '1W'
    """
    df_1h = _klines("1h", 500, symbol)    # ~3 weeks, enough for all indicators
    df_1d = _klines("1d", 300, symbol)    # ~10 months (for EMA-200)
    df_1w = _klines("1w", 104, symbol)    # ~2 years

    df_4h = (df_1h
             .resample("4h", label="left", closed="left")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last",  "volume": "sum"})
             .dropna(subset=["open"]))

    return {"1H": df_1h, "4H": df_4h, "1D": df_1d, "1W": df_1w}


# ── Premium index (real OI proxy) ────────────────────────────────────────────

def fetch_premium(symbol: str = SYMBOL, limit: int = 500) -> pd.Series:
    """Fetch BTCUSDT perpetual premium index klines from production FAPI."""
    r = requests.get(
        f"{FAPI}/fapi/v1/premiumIndexKlines",
        params={"symbol": symbol, "interval": "1h", "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.Series(dtype=float, name="premium")
    df = pd.DataFrame(data, columns=KLINES_COLS)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    return pd.to_numeric(df["close"], errors="coerce").rename("premium")


# ── Funding rate ──────────────────────────────────────────────────────────────

def fetch_funding(symbol: str = SYMBOL, limit: int = 100) -> pd.Series:
    """Fetch recent funding rates from production FAPI."""
    r = requests.get(
        f"{FAPI}/fapi/v1/fundingRate",
        params={"symbol": symbol, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.Series(dtype=float, name="funding_rate")

    idx  = pd.to_datetime([d["fundingTime"] for d in data], unit="ms", utc=True).tz_localize(None)
    vals = [float(d["fundingRate"]) for d in data]
    s    = pd.Series(vals, index=idx, name="funding_rate").sort_index()
    return s


# ── Full signal pipeline ──────────────────────────────────────────────────────

def compute_live_signal(
    symbol: str = SYMBOL,
    scenario: str = SCENARIO,
) -> Tuple[int, float, float, float]:
    """
    Fetch live data, run the full backtest signal pipeline, and return
    the signal for the latest completed 1H bar.

    Returns
    -------
    signal    : int   – +1 long / -1 short / 0 flat
    composite : float – raw composite score
    atr_14    : float – ATR-14 for position sizing
    last_price: float – close price of latest bar
    """
    # 1. Fetch OHLCV
    tf_data = fetch_tf_data(symbol)

    # 2. Add indicators (same function used in backtest)
    for tf in tf_data:
        tf_data[tf] = add_indicators(tf_data[tf])

    df_1h = tf_data["1H"]

    # 3. OI proxy: try real premium, fall back to synthetic
    premium_1h = pd.Series(dtype=float)
    try:
        premium_1h = fetch_premium(symbol)
    except Exception:
        pass

    oi_is_real = len(premium_1h) > 50

    # Synthetic OI on 1D close (fallback)
    oi_df  = generate_oi(tf_data["1D"]["close"])

    # 4. Funding rate
    funding = pd.Series(dtype=float)
    try:
        funding = fetch_funding(symbol)
    except Exception:
        funding = generate_funding(tf_data["1D"]["close"])

    # 5. Build signal matrix (identical to backtest)
    signals = build_signal_matrix(
        tf_data,
        oi_df,
        funding,
        premium_1h=premium_1h if oi_is_real else None,
    )

    # 6. Apply scenario filters
    cfg  = SCENARIOS[scenario]
    filt = apply_filters(signals, cfg)

    # Use signal from bar[-2] (last *completed* bar before current live bar)
    last_signal    = int(filt["signal"].iloc[-2])
    last_composite = float(filt["composite"].iloc[-2])
    last_atr       = float(df_1h["atr_14"].iloc[-2])
    last_close     = float(df_1h["close"].iloc[-2])

    return last_signal, last_composite, last_atr, last_close
