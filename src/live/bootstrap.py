"""
Bootstrap historical bars for live paper trading.

Priority order for each timeframe:
  1. Local parquet cache (built by the historical fetcher).
  2. Binance Futures REST API for bars after the last cached timestamp.

For 1m we skip the full cache (2M+ rows) and only fetch the last 500
bars from REST — enough to compute vol_ratio / log_ret for the s_1m signal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
FAPI      = "https://fapi.binance.com/fapi/v1"

# REST interval strings for each internal TF key
_REST_INTERVAL = {"1H": "1h", "15M": "15m", "1M": "1m"}
# Cache filename fragment for each TF key
_CACHE_FRAG    = {"1H": "1h", "15M": "15m", "1M": "1m"}

RESAMPLE_AGG = {
    "open": "first", "high": "max",
    "low": "min",  "close": "last", "volume": "sum",
}

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _parse_klines(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=_KLINE_COLS)
    df["open_time"] = (pd.to_datetime(df["open_time"], unit="ms", utc=True)
                       .dt.tz_localize(None))
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return (df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
              .sort_index())


def fetch_rest_bars(tf_key: str,
                    since: Optional[pd.Timestamp] = None,
                    limit: int = 1000) -> pd.DataFrame:
    """Pull klines from Binance Futures REST (no API key required)."""
    interval = _REST_INTERVAL[tf_key]
    params: dict = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    if since is not None:
        params["startTime"] = int(since.timestamp() * 1000)
    try:
        r = requests.get(f"{FAPI}/klines", params=params, timeout=30)
        r.raise_for_status()
        df = _parse_klines(r.json())
        # Drop the still-open (incomplete) bar: its close_time is in the future
        if not df.empty:
            df = df.iloc[:-1]
        return df
    except Exception as exc:
        print(f"    [bootstrap] REST {tf_key} failed: {exc}")
        return pd.DataFrame()


def _load_cache(tf_key: str) -> pd.DataFrame:
    """Concatenate all parquet files in cache for the given TF."""
    frag = _CACHE_FRAG[tf_key]
    files = sorted(CACHE_DIR.glob(f"BTCUSDT-{frag}-*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs).sort_index()
    return df[~df.index.duplicated(keep="last")]


def bootstrap_tf(tf_key: str, skip_cache: bool = False) -> pd.DataFrame:
    """
    Return a complete OHLCV DataFrame for *tf_key*.
    Loads the cache first, then appends missing recent bars via REST.
    """
    if skip_cache:
        cached = pd.DataFrame()
    else:
        cached = _load_cache(tf_key)

    if not cached.empty:
        since = cached.index[-1]
        recent = fetch_rest_bars(tf_key, since=since, limit=1000)
        if not recent.empty:
            df = pd.concat([cached, recent])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = cached
    else:
        limit = {"1H": 1500, "15M": 1500, "1M": 500}[tf_key]
        df = fetch_rest_bars(tf_key, limit=limit)

    return df


def fetch_funding(limit: int = 200) -> pd.Series:
    """Fetch recent perpetual funding rates from REST API."""
    try:
        r = requests.get(f"{FAPI}/fundingRate",
                         params={"symbol": "BTCUSDT", "limit": limit},
                         timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return pd.Series(dtype=float)
        idx  = (pd.to_datetime([x["fundingTime"] for x in rows], unit="ms", utc=True)
                .tz_localize(None))
        vals = [float(x["fundingRate"]) for x in rows]
        return pd.Series(vals, index=idx, name="funding").sort_index()
    except Exception as exc:
        print(f"    [bootstrap] Funding fetch failed: {exc}")
        return pd.Series(dtype=float)


def fetch_premium_1h(limit: int = 600) -> pd.Series:
    """
    Fetch 1H premium-index (basis) klines from Binance.
    This is used as the real OI proxy in build_signal_matrix.
    """
    try:
        r = requests.get(f"{FAPI}/premiumIndexKlines",
                         params={"symbol": "BTCUSDT", "interval": "1h",
                                 "limit": limit},
                         timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return pd.Series(dtype=float)
        idx  = (pd.to_datetime([x[0] for x in rows], unit="ms", utc=True)
                .tz_localize(None))
        vals = [float(x[4]) for x in rows]   # close = end-of-bar premium %
        return pd.Series(vals, index=idx, name="premium").sort_index()
    except Exception as exc:
        print(f"    [bootstrap] Premium index fetch failed: {exc}")
        return pd.Series(dtype=float)


def bootstrap_all(verbose: bool = True) -> dict:
    """
    Load all required data for live signal computation.

    Returns
    -------
    dict with keys:
      "1H"         → pd.DataFrame (full history from cache + recent REST)
      "15M"        → pd.DataFrame
      "1M"         → pd.DataFrame (last ~500 bars only)
      "funding"    → pd.Series
      "premium_1h" → pd.Series (real basis proxy for OI signal)
    """
    out: dict = {}

    for tf in ["1H", "15M"]:
        if verbose:
            print(f"  [{tf}] loading cache + REST … ", end="", flush=True)
        df = bootstrap_tf(tf, skip_cache=False)
        out[tf] = df
        if verbose and not df.empty:
            print(f"{len(df):,} bars  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
        elif verbose:
            print("empty")

    # 1m: skip full cache to avoid loading 2M rows; REST gives enough context
    if verbose:
        print("  [1M] REST only (last 500 bars) … ", end="", flush=True)
    df_1m = bootstrap_tf("1M", skip_cache=True)
    out["1M"] = df_1m
    if verbose and not df_1m.empty:
        print(f"{len(df_1m)} bars  [{df_1m.index[0]} → {df_1m.index[-1]}]")
    elif verbose:
        print("empty")

    if verbose:
        print("  Funding rates … ", end="", flush=True)
    out["funding"] = fetch_funding()
    if verbose:
        print(f"{len(out['funding'])} records")

    if verbose:
        print("  Premium index … ", end="", flush=True)
    out["premium_1h"] = fetch_premium_1h()
    if verbose:
        print(f"{len(out['premium_1h'])} records")

    return out
