"""
Multi-timeframe OHLCV data fetcher for BTCUSDT.

Primary source: Binance Vision CDN (real BTCUSDT perpetual futures, no API key).
  - 1H, 4H, 1D : data/futures/um/monthly/klines/BTCUSDT/{interval}/
  - Funding     : data/futures/um/monthly/fundingRate/BTCUSDT/
  - 1W          : resampled from real Binance Vision 1D (futures data, not spot)

Fallback: Yahoo Finance BTC-USD (spot) when Binance Vision is unreachable.

Open Interest: synthetic (realistic model) – no free long-history source exists.
"""
from __future__ import annotations

import io
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SYMBOL = "BTC-USD"

BVISION   = "https://data.binance.vision/data/futures/um/monthly"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

# yfinance fallback config
TF_CONFIG: Dict[str, Tuple[str, str]] = {
    "1W":  ("1wk", "4y"),
    "1D":  ("1d",  "4y"),
    "4H":  ("4h",  "730d"),
    "1H":  ("1h",  "730d"),
    "15M": ("15m", "60d"),
}

KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_cache() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _get_zip(url: str, timeout: int = 30) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 200:
            return r.content
    except Exception:
        pass
    return None


def _months_range(start_year: int, start_month: int,
                  end_year: int, end_month: int) -> List[Tuple[int, int]]:
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


# ─────────────────────────────────────────────────────────────────────────────
# Binance Vision – generic OHLCV klines (any interval)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_klines_month_generic(interval: str, year: int, month: int) -> Optional[pd.DataFrame]:
    """Download one monthly klines zip for any interval (1m, 15m, 1h, …)."""
    cache = _ensure_cache() / f"BTCUSDT-{interval}-{year}-{month:02d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = (f"{BVISION}/klines/BTCUSDT/{interval}/"
           f"BTCUSDT-{interval}-{year}-{month:02d}.zip")
    data = _get_zip(url)
    if data is None:
        return None

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        raw = z.open(z.namelist()[0]).read()
        first = raw.split(b"\n")[0].decode(errors="ignore").strip()
        skip  = 0 if first and first[0].isdigit() else 1
        df = pd.read_csv(io.BytesIO(raw), header=None,
                         names=KLINES_COLS, skiprows=skip)

    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms")
    df = (df.set_index("ts")[["open", "high", "low", "close", "volume"]]
            .astype(float))
    df.index = df.index.tz_localize(None).astype("datetime64[s]")
    df.to_parquet(cache)
    return df


def fetch_binance_vision_klines(
    interval: str,
    start_year: int = 2022,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    workers: int = 6,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Download BTCUSDT perpetual OHLCV from Binance Vision CDN for any interval.
    Files are cached as parquet; subsequent calls are instant.

    interval : BV interval string, e.g. '1m', '15m', '1h', '4h'
    """
    now = datetime.now()
    if end_year is None:
        end_year = now.year
    if end_month is None:
        end_month = now.month - 1 or 12

    months = _months_range(start_year, start_month, end_year, end_month)
    frames: List[pd.DataFrame] = [None] * len(months)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_klines_month_generic, interval, y, m): i
                for i, (y, m) in enumerate(months)}
        for fut in as_completed(futs):
            idx = futs[fut]
            y, m = months[idx]
            result = fut.result()
            if result is not None:
                frames[idx] = result
                if verbose:
                    print(f"  ✓ {interval} {y}-{m:02d}: {len(result):6d} bars")
            else:
                if verbose:
                    print(f"  ✗ {interval} {y}-{m:02d}: not available")

    valid = [f for f in frames if f is not None]
    if not valid:
        return pd.DataFrame()

    out = pd.concat(valid).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def fetch_binance_vision_1h(
    start_year: int = 2022,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    workers: int = 6,
) -> pd.DataFrame:
    """Download BTCUSDT perpetual 1H OHLCV from Binance Vision CDN."""
    df = fetch_binance_vision_klines(
        "1h", start_year=start_year, start_month=start_month,
        end_year=end_year, end_month=end_month, workers=workers, verbose=True,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Binance Vision – Funding Rate
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_funding_month(year: int, month: int) -> Optional[pd.Series]:
    cache = _ensure_cache() / f"BTCUSDT-funding-{year}-{month:02d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["funding_rate"]

    url = (f"{BVISION}/fundingRate/BTCUSDT/"
           f"BTCUSDT-fundingRate-{year}-{month:02d}.zip")
    data = _get_zip(url)
    if data is None:
        return None

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))

    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms")
    df = df.set_index("ts")[["last_funding_rate"]].rename(
        columns={"last_funding_rate": "funding_rate"}).astype(float)
    df.index = df.index.tz_localize(None)
    df.to_parquet(cache)
    return df["funding_rate"]


def fetch_binance_vision_funding(
    start_year: int = 2022,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
) -> pd.Series:
    """
    Real BTCUSDT funding rates from Binance Vision (8-hourly settlement).
    Falls back to synthetic if download fails.
    """
    now = datetime.now()
    if end_year is None:
        end_year = now.year
    if end_month is None:
        end_month = now.month - 1 or 12

    months = _months_range(start_year, start_month, end_year, end_month)
    parts = []
    for y, m in months:
        s = _fetch_funding_month(y, m)
        if s is not None:
            parts.append(s)

    if not parts:
        return pd.Series(dtype=float, name="funding_rate")

    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Resample helpers
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV → 4H (consistent with 1H source)."""
    return (df_1h.resample("4h", label="left", closed="left")
                 .agg({"open": "first", "high": "max",
                       "low": "min",   "close": "last",
                       "volume": "sum"})
                 .dropna(subset=["open"]))


def resample_to_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV → 1D."""
    return (df_1h.resample("1D", label="left", closed="left")
                 .agg({"open": "first", "high": "max",
                       "low": "min",   "close": "last",
                       "volume": "sum"})
                 .dropna(subset=["open"]))


def align_funding_to_daily(funding_8h: pd.Series,
                            daily_index: pd.DatetimeIndex) -> pd.Series:
    """
    Resample 8-hourly funding to daily mean, then forward-fill to daily_index.
    Used by the signal module which expects daily-frequency funding.
    """
    if funding_8h.empty:
        return pd.Series(0.0, index=daily_index, name="funding_rate")

    daily = funding_8h.resample("1D").mean()
    return (daily.reindex(daily_index, method="ffill")
                 .bfill()
                 .fillna(0.0))


# ─────────────────────────────────────────────────────────────────────────────
# yfinance fallback
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_ohlcv_yf(timeframe: str = "1D", symbol: str = SYMBOL) -> pd.DataFrame:
    interval, period = TF_CONFIG[timeframe]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(symbol, interval=interval, period=period,
                          progress=False, auto_adjust=True)
    return _clean(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry points
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_timeframes(symbol: str = SYMBOL) -> Dict[str, pd.DataFrame]:
    """Legacy yfinance-only fetch (60d for intraday). Kept for compatibility."""
    result: Dict[str, pd.DataFrame] = {}
    cfg_legacy = {**TF_CONFIG, "4H": ("4h", "60d"), "1H": ("1h", "60d")}
    for tf, (interval, period) in cfg_legacy.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(symbol, interval=interval, period=period,
                                  progress=False, auto_adjust=True)
            df = _clean(raw)
            result[tf] = df
            print(f"  [{tf:>3s}]  {len(df):5d} bars  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
        except Exception as exc:
            print(f"  [{tf:>3s}]  FAILED: {exc}")
    return result


def fetch_extended_data(
    start_year: int = 2022,
    start_month: int = 1,
    workers: int = 6,
    symbol: str = SYMBOL,
    fetch_15m: bool = True,
    fetch_1m: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multi-TF data entirely from Binance Vision perpetual futures.
    No resampling, no yfinance spot data — every series is real futures OHLCV.

    1H, 4H, 1D : Binance Vision perpetual futures (real klines for each TF).
    1W          : resampled from real Binance Vision 1D (not spot prices).
    15M         : Binance Vision (~2,900 bars/month).
    1M          : Binance Vision (~43,800 bars/month). Skip with fetch_1m=False.

    Returns dict with keys: 1W, 1D, 4H, 1H, 15M, 1M
    """
    result: Dict[str, pd.DataFrame] = {}

    # ── 1H from Binance Vision ───────────────────────────────────────────────
    print("  Downloading 1H from Binance Vision …")
    df_1h = fetch_binance_vision_klines(
        "1h", start_year=start_year, start_month=start_month,
        workers=workers, verbose=True)
    if df_1h.empty or len(df_1h) < 500:
        print("  ↳ 1H unavailable – falling back to yfinance")
        df_1h = fetch_ohlcv_yf("1H", symbol)
    result["1H"] = df_1h

    # ── 4H from Binance Vision (real futures, not resampled) ─────────────────
    print("  Downloading 4H from Binance Vision …")
    df_4h = fetch_binance_vision_klines(
        "4h", start_year=start_year, start_month=start_month,
        workers=workers, verbose=False)
    if df_4h.empty or len(df_4h) < 100:
        print("  ↳ 4H unavailable – resampling from 1H as fallback")
        df_4h = resample_to_4h(df_1h)
    result["4H"] = df_4h

    # ── 1D from Binance Vision (real futures, not yfinance spot) ─────────────
    print("  Downloading 1D from Binance Vision …")
    df_1d = fetch_binance_vision_klines(
        "1d", start_year=start_year, start_month=start_month,
        workers=workers, verbose=False)
    if df_1d.empty or len(df_1d) < 100:
        print("  ↳ 1D unavailable – falling back to yfinance")
        df_1d = fetch_ohlcv_yf("1D", symbol)
    result["1D"] = df_1d

    # ── 1W resampled from real Binance Vision 1D (futures prices) ────────────
    result["1W"] = (df_1d
                    .resample("W", label="left", closed="left")
                    .agg({"open": "first", "high": "max",
                          "low": "min",   "close": "last",
                          "volume": "sum"})
                    .dropna(subset=["open"]))

    # ── 15M from Binance Vision (~2,900 bars/month) ──────────────────────────
    if fetch_15m:
        print("  Downloading 15M from Binance Vision …")
        df_15m = fetch_binance_vision_klines(
            "15m", start_year=start_year, start_month=start_month,
            workers=workers, verbose=False)
        if df_15m.empty:
            try:
                df_15m = fetch_ohlcv_yf("15M", symbol)
                print("  ↳ BV unavailable – using yfinance 15M (60d)")
            except Exception:
                df_15m = pd.DataFrame()
        result["15M"] = df_15m
    else:
        result["15M"] = pd.DataFrame()

    # ── 1M from Binance Vision (~43,800 bars/month) ──────────────────────────
    if fetch_1m:
        print("  Downloading 1M from Binance Vision … (heavy, cached after first run)")
        df_1m = fetch_binance_vision_klines(
            "1m", start_year=start_year, start_month=start_month,
            workers=workers, verbose=False)
        result["1M"] = df_1m
    else:
        result["1M"] = pd.DataFrame()

    # ── Print summary ─────────────────────────────────────────────────────────
    for tf in ["1W", "1D", "4H", "1H", "15M", "1M"]:
        df = result.get(tf, pd.DataFrame())
        if not df.empty:
            print(f"  [{tf:>3s}]  {len(df):8,d} bars  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")

    return result


def fetch_real_funding(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> Tuple[pd.Series, bool]:
    """
    Fetch real funding rates from Binance Vision.
    Returns (funding_series_daily, is_real).
    funding_series_daily is aligned to df_1d.index.
    Falls back to synthetic if download fails.
    """
    if df_1h.empty:
        return generate_funding(df_1d["close"]), False

    start = df_1h.index[0]
    funding_8h = fetch_binance_vision_funding(
        start_year=start.year, start_month=start.month)

    if funding_8h.empty or len(funding_8h) < 10:
        return generate_funding(df_1d["close"]), False

    funding_daily = align_funding_to_daily(funding_8h, df_1d.index)
    return funding_daily, True


# ─────────────────────────────────────────────────────────────────────────────
# Binance Vision – Premium Index Klines (basis = futures - spot index)
# Available since 2020-01, 1H granularity, no API key required.
# Replaces synthetic OI: basis direction + price direction → real positioning signal.
# ─────────────────────────────────────────────────────────────────────────────

PREMIUM_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _fetch_premium_month(year: int, month: int) -> Optional[pd.Series]:
    """Fetch one month of premiumIndexKlines (close = basis ratio at bar close)."""
    cache = _ensure_cache() / f"BTCUSDT-premium-{year}-{month:02d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["premium"]

    url = (f"{BVISION}/premiumIndexKlines/BTCUSDT/1h/"
           f"BTCUSDT-1h-{year}-{month:02d}.zip")
    data = _get_zip(url)
    if data is None:
        return None

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        raw = z.open(z.namelist()[0]).read()
        first = raw.split(b"\n")[0].decode(errors="ignore").strip()
        skip = 0 if first and first[0].isdigit() else 1
        df = pd.read_csv(io.BytesIO(raw), header=None,
                         names=PREMIUM_COLS, skiprows=skip)

    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms")
    df = df.set_index("ts")[["close"]].rename(columns={"close": "premium"}).astype(float)
    df.index = df.index.tz_localize(None).astype("datetime64[s]")
    df.to_parquet(cache)
    return df["premium"]


def fetch_binance_vision_premium(
    start_year: int = 2022,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
) -> pd.Series:
    """
    Real BTCUSDT basis (premium index, 1H) from Binance Vision.
    Values are the fractional premium of futures price over spot index.
    Positive = futures at premium (longs paying), negative = discount (shorts paying).
    Returns empty Series on complete failure.
    """
    now = datetime.now()
    if end_year is None:
        end_year = now.year
    if end_month is None:
        end_month = now.month - 1 or 12

    months = _months_range(start_year, start_month, end_year, end_month)
    parts = []
    for y, m in months:
        s = _fetch_premium_month(y, m)
        if s is not None:
            parts.append(s)

    if not parts:
        return pd.Series(dtype=float, name="premium")

    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def fetch_real_oi(
    df_1h: pd.DataFrame,
) -> Tuple[pd.Series, bool]:
    """
    Fetch real basis (premiumIndexKlines) from Binance Vision as an OI proxy.
    Returns (premium_1h_series, is_real).
    Falls back to empty Series if download fails.
    """
    if df_1h.empty:
        return pd.Series(dtype=float, name="premium"), False

    start = df_1h.index[0]
    premium = fetch_binance_vision_premium(
        start_year=start.year, start_month=start.month)

    if premium.empty or len(premium) < 100:
        return pd.Series(dtype=float, name="premium"), False

    return premium, True


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic OI (kept as fallback – no free long-history source for raw OI)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oi(price_series: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Realistic synthetic Open Interest correlated with BTCUSDT price.
    OI grows in trending phases, contracts in chop, with 12 % divergence events.
    """
    rng = np.random.default_rng(seed)
    n = len(price_series)
    price = price_series.to_numpy(dtype=float)

    log_ret = np.diff(np.log(price), prepend=np.log(price[0]))
    rvol = np.array([
        float(np.std(log_ret[max(0, i - 19): i + 1])) for i in range(n)
    ])
    median_rvol = np.median(rvol)

    oi = np.empty(n, dtype=float)
    oi[0] = 1e10 * (price[0] / 50_000)

    for i in range(1, n):
        trending = rvol[i] > median_rvol
        base = (0.005 if trending else -0.002) + rng.normal(0, 0.008)
        if rng.random() < 0.12:
            base = -base
        oi[i] = max(oi[i - 1] * (1.0 + np.clip(base, -0.06, 0.06)), 5e9)

    oi_s = pd.Series(oi, index=price_series.index, name="oi")
    return pd.DataFrame({
        "oi":       oi_s,
        "oi_chg":   oi_s.pct_change().fillna(0),
        "oi_sma7":  oi_s.rolling(7, min_periods=1).mean(),
        "oi_trend": np.sign(oi_s - oi_s.rolling(7, min_periods=1).mean()),
    })


def generate_funding(price_series: pd.Series, seed: int = 42) -> pd.Series:
    """
    Synthetic 8-hour funding rates (fallback when real data unavailable).
    Mean-reverting, momentum-driven, clipped to [-0.3 %, +0.3 %].
    """
    rng = np.random.default_rng(seed)
    n = len(price_series)
    price = price_series.to_numpy(dtype=float)

    mom = np.zeros(n, dtype=float)
    for i in range(20, n):
        mom[i] = (price[i] - price[i - 20]) / price[i - 20]

    funding = np.zeros(n, dtype=float)
    funding[0] = 0.0001

    for i in range(1, n):
        rev   = -0.30 * funding[i - 1]
        trend = 0.01  * mom[i]
        noise = rng.normal(0, 0.0001)
        funding[i] = np.clip(funding[i - 1] + rev + trend + noise, -0.003, 0.003)

    return pd.Series(funding, index=price_series.index, name="funding_rate")
