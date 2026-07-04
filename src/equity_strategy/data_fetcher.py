"""
Daily OHLCV data fetcher for the S&P 500 sector-rotation strategy.

Source: Yahoo Finance "chart" REST endpoint, queried directly via `requests`
(no yfinance dependency for downloads — its curl_cffi backend does not
tolerate TLS-terminating proxies). Parquet-cached like the BTC data fetcher
in src/strategy/data_fetcher.py.

Universe
────────
Benchmark        : SPY   (S&P 500 ETF, dividend/split adjusted)
Rotation sectors  : XLK XLF XLV XLY XLP XLE XLI XLB XLU  (SPDR sectors, live since 1998-12)
Risk-free proxy   : ^IRX  (13-week T-bill discount yield) → synthetic daily-accrual NAV
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache_equity"

SECTOR_ETFS: List[str] = [
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU",
]
# Later-inception sectors, included opportunistically once they have enough
# history for the momentum lookback (XLRE launched 2015-10, XLC 2018-06).
# GICS moved Google/Meta/Netflix into Communication Services in 2018, so
# excluding XLC would structurally blind the strategy to a large slice of
# the post-2018 market-cap leadership.
EXTRA_SECTOR_ETFS: List[str] = ["XLRE", "XLC"]
BENCHMARK = "SPY"
RISK_FREE_PROXY = "^IRX"

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _ensure_cache() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _fetch_chart(symbol: str, retries: int = 3) -> Optional[dict]:
    params = dict(period1=0, period2=int(datetime.now(timezone.utc).timestamp()),
                  interval="1d", events="div,split")
    url = _CHART_URL.format(symbol=symbol)
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()["chart"]["result"][0]
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_daily_ohlcv(symbol: str, force: bool = False) -> pd.DataFrame:
    """Fetch full-history daily OHLCV (+ adjclose) for one symbol, parquet-cached."""
    cache = _ensure_cache() / f"{symbol.replace('^', '_')}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = _fetch_chart(symbol)
    if raw is None:
        raise RuntimeError(f"Failed to fetch data for {symbol}")

    ts = raw["timestamp"]
    q = raw["indicators"]["quote"][0]
    adj = raw["indicators"].get("adjclose", [{}])[0].get("adjclose")

    df = pd.DataFrame({
        "open":  q["open"],
        "high":  q["high"],
        "low":   q["low"],
        "close": q["close"],
        "volume": q["volume"],
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None))
    df.index.name = "date"
    df["adjclose"] = adj if adj is not None else df["close"]

    df = df.dropna(subset=["close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # normalise to midnight (daily bars)
    df.index = df.index.normalize()
    df.to_parquet(cache)
    return df


def fetch_universe(force: bool = False, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Fetch benchmark + sector ETFs + risk-free proxy. Returns {symbol: df}."""
    symbols = [BENCHMARK] + SECTOR_ETFS + EXTRA_SECTOR_ETFS + [RISK_FREE_PROXY]
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_daily_ohlcv(sym, force=force)
        out[sym] = df
        if verbose:
            print(f"  [{sym:>5s}]  {len(df):5,d} bars  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
    return out


def build_cash_nav(irx_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """
    Synthetic daily-accrual NAV for a cash / T-bill-equivalent position, built
    from the ^IRX 13-week T-bill discount-yield index (quoted in yield points,
    e.g. 5.25 = 5.25 %/yr). Converted to a daily simple-interest accrual factor
    and compounded into a price-like series starting at 100.0.
    """
    yld = irx_df["close"].reindex(index).ffill().bfill() / 100.0   # fraction/yr
    daily_factor = 1.0 + yld / 252.0
    nav = daily_factor.cumprod() * 100.0
    return nav.rename("CASH")


def build_adjclose_panel(universe: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Assemble an aligned daily adjclose panel for SPY + sector ETFs (business days).

    Core sectors (live since 1998) must be fully populated; later-inception
    sectors (XLRE, XLC) are kept with leading NaNs so the signal layer can
    opportunistically include them once they clear the momentum warm-up —
    dropping those rows outright would truncate the whole backtest to 2018.
    """
    core_symbols = [BENCHMARK] + SECTOR_ETFS
    extra_symbols = [s for s in EXTRA_SECTOR_ETFS if s in universe]
    frames = {s: universe[s]["adjclose"] for s in core_symbols + extra_symbols}
    panel = pd.DataFrame(frames)
    # keep dates where the benchmark trades; ffill short gaps (holidays mismatch)
    panel = panel.sort_index().ffill(limit=3)
    panel = panel.dropna(subset=core_symbols, how="any")
    return panel
