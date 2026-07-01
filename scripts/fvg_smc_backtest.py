"""
FVG + SMC Multi-Timeframe Backtest
===================================
Timeframes : 1m (raw), 15m (LTF execution), 4H (HTF bias / FVG zones)
Strategy   : Fair Value Gap + Smart Money Concepts alignment
Design     : Zero look-ahead bias, event-driven mitigation, sequential scan

Data source: ccxt / Binance (public, no key required).
             Falls back to realistic synthetic GBM OHLCV when network is
             unavailable (e.g. sandboxed CI environments).
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    symbol: str = "BTC/USDT"
    since_str: str = "2024-01-01T00:00:00Z"
    until_str: str = "2024-07-01T00:00:00Z"
    # Risk
    risk_reward: float = 3.0       # 1:RR
    risk_pct: float = 0.01         # fraction of equity at risk per trade
    initial_equity: float = 10_000.0
    # FVG mitigation: "wick" = any wick touch, "close" = full-body close through
    mitigation_mode: str = "wick"
    # Limit order expiry (in LTF bars)
    order_expiry_bars: int = 20
    # Max simultaneous open trades
    max_open_trades: int = 3
    # IC indicator lookback (bars)
    ic_lookback: int = 100
    # ccxt exchange id (public OHLCV, no key needed)
    exchange_id: str = "binance"
    # Cache dir for downloaded data
    cache_dir: str = "data/fvg_cache"
    # Force synthetic data (skips network fetch)
    use_synthetic: bool = False


CFG = BacktestConfig()


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA (GBM-based, realistic BTC-like OHLCV)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_ohlcv(n_bars: int, freq_minutes: int,
                           start_price: float = 42_000.0,
                           annual_vol: float = 0.80,
                           seed: int = 42) -> pd.DataFrame:
    """
    Geometric Brownian Motion OHLCV with intra-bar simulation.
    Produces realistic OHLCV (no H<O, L>C artifacts).
    """
    rng = np.random.default_rng(seed)
    dt = freq_minutes / (365 * 24 * 60)          # fraction of year per bar
    sigma_bar = annual_vol * np.sqrt(dt)

    log_ret = rng.normal(0.0, sigma_bar, n_bars)
    closes = start_price * np.exp(np.cumsum(log_ret))
    opens  = np.empty(n_bars)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    # High / Low: simulate as max/min of 4 intra-bar sub-ticks
    sub = rng.normal(0.0, sigma_bar / 2, (n_bars, 4))
    intra = np.cumsum(sub, axis=1)
    intra_prices = opens[:, None] * np.exp(intra)
    highs  = np.maximum(intra_prices.max(axis=1), np.maximum(opens, closes))
    lows   = np.minimum(intra_prices.min(axis=1), np.minimum(opens, closes))
    volume = rng.uniform(10, 500, n_bars)

    start_ts = pd.Timestamp("2024-01-01", tz="UTC")
    idx = pd.date_range(start_ts, periods=n_bars, freq=f"{freq_minutes}min", tz="UTC")

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


def _make_all_synthetic(cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Generating synthetic OHLCV data …")
    # 6 months worth of bars
    n_15m = 6 * 30 * 24 * 4      # ~17 280 bars
    n_1m  = n_15m * 15            # ~259 200 bars
    n_4h  = n_15m // 16           # ~1 080 bars

    df_1m  = _make_synthetic_ohlcv(n_1m,  1,   seed=1)
    df_15m = _make_synthetic_ohlcv(n_15m, 15,  seed=2)
    df_4h  = _make_synthetic_ohlcv(n_4h,  240, seed=3)

    log.info("Synthetic: 1m=%d | 15m=%d | 4H=%d", len(df_1m), len(df_15m), len(df_4h))
    return df_1m, df_15m, df_4h


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (ccxt with SSL fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_ohlcv(exchange, symbol: str, timeframe: str,
                 since_ms: int, until_ms: int) -> pd.DataFrame:
    all_rows = []
    cursor   = since_ms
    limit    = 1000
    while cursor < until_ms:
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except Exception as exc:
            import ccxt
            if isinstance(exc, ccxt.RateLimitExceeded):
                time.sleep(5)
                raw = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            else:
                raise
        if not raw:
            break
        all_rows.extend(raw)
        cursor = raw[-1][0] + 1
        if len(raw) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = (df[df["ts"] < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
            .drop_duplicates("ts").sort_values("ts").set_index("ts"))
    return df


def fetch_data(cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cfg.use_synthetic:
        return _make_all_synthetic(cfg)

    os.makedirs(cfg.cache_dir, exist_ok=True)
    sym_slug = cfg.symbol.replace("/", "")
    date_tag = f"{cfg.since_str[:10]}_{cfg.until_str[:10]}"
    paths = {tf: f"{cfg.cache_dir}/{sym_slug}_{tf}_{date_tag}.csv"
             for tf in ["1m", "15m", "1h"]}

    try:
        import ccxt as ccxt_lib
        exchange = getattr(ccxt_lib, cfg.exchange_id)({"enableRateLimit": True})
        since_ms = exchange.parse8601(cfg.since_str)
        until_ms = exchange.parse8601(cfg.until_str)

        frames: dict[str, pd.DataFrame] = {}
        for tf, path in paths.items():
            if os.path.exists(path):
                log.info("Cache hit: %s", path)
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=True)
            else:
                log.info("Downloading %s %s …", cfg.symbol, tf)
                df = _fetch_ohlcv(exchange, cfg.symbol, tf, since_ms, until_ms)
                df.to_csv(path)
                log.info("Saved %d rows → %s", len(df), path)
            frames[tf] = df

        df_4h = (
            frames["1h"]
            .resample("4h", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
        )
        return frames["1m"], frames["15m"], df_4h

    except Exception as exc:
        log.warning("Network fetch failed (%s). Falling back to synthetic data.", exc)
        return _make_all_synthetic(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# FVG DETECTION — vectorised, zero look-ahead
# ─────────────────────────────────────────────────────────────────────────────

def compute_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects FVGs using only fully-closed bars (i-2, i-1, i).

    Bullish FVG : Low[i]  > High[i-2]  → zone [High[i-2] (bot), Low[i] (top)]
    Bearish FVG : High[i] < Low[i-2]   → zone [High[i] (bot),   Low[i-2] (top)]

    fvg_top / fvg_bot are the zone boundaries for whichever FVG is detected.
    """
    h2 = df["high"].shift(2)
    l2 = df["low"].shift(2)
    h  = df["high"]
    l  = df["low"]

    bull = l > h2
    bear = h < l2

    # Exclusive labels: if both fire simultaneously (rare), skip
    valid_bull = bull & ~bear
    valid_bear = bear & ~bull

    fvg_top = np.where(valid_bull, l,  np.where(valid_bear, l2, np.nan))
    fvg_bot = np.where(valid_bull, h2, np.where(valid_bear, h,  np.nan))

    out = df.copy()
    out["fvg_bull"] = valid_bull
    out["fvg_bear"] = valid_bear
    out["fvg_top"]  = fvg_top.astype(float)
    out["fvg_bot"]  = fvg_bot.astype(float)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SIGNIFICANCE & IC INDICATOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_indicator(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    Information Coefficient (IC) style indicator.

    signal_strength  : normalised price position inside the FVG zone  [-1, 1]
                       +1 = price at zone top (FVG fully tested from below)
                       -1 = price at zone bot
                       NaN when no active zone

    ic_rolling       : rolling Pearson correlation between signal_strength
                       and the 1-bar FORWARD log-return (computed on history
                       only — NOT used for live order decisions)

    ic_zscore        : (ic_rolling − rolling_mean) / rolling_std
                       a z-score > 1.5 suggests the signal is above its
                       historical predictive power baseline

    NOTE: fwd_ret_1b uses shift(-1) — this is intentionally look-ahead ONLY
    for the offline IC statistic. It is never used inside the trade logic.
    """
    out = df.copy()

    # FVG midpoint / range (use whichever zone is active; top/bot already set)
    fvg_mid   = (out["fvg_top"] + out["fvg_bot"]) / 2.0
    fvg_range = (out["fvg_top"] - out["fvg_bot"]).abs().replace(0.0, np.nan)

    out["signal_raw"] = ((out["close"] - fvg_mid) / fvg_range).clip(-1.0, 1.0)

    # Forward return — OFFLINE metric only
    out["fwd_ret_1b"] = np.log(out["close"].shift(-1) / out["close"])

    # Rolling Pearson IC  (vectorised, no loop)
    # pandas rolling.corr() requires a plain Series (not another Rolling object)
    out["ic_rolling"] = (
        out["signal_raw"]
        .rolling(lookback, min_periods=lookback // 2)
        .corr(out["fwd_ret_1b"])
    )

    # IC z-score over the same rolling window
    ic_roll = out["ic_rolling"].rolling(lookback, min_periods=lookback // 2)
    ic_mean = ic_roll.mean()
    ic_std  = ic_roll.std().replace(0.0, np.nan)
    out["ic_zscore"] = (out["ic_rolling"] - ic_mean) / ic_std

    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTF → LTF ALIGNMENT (zero look-ahead, event-driven mitigation)
# ─────────────────────────────────────────────────────────────────────────────

def align_htf_fvg_to_ltf(df_htf: pd.DataFrame,
                           df_ltf: pd.DataFrame,
                           mitigation_mode: str = "wick") -> pd.DataFrame:
    """
    Propagates 4H FVG zones onto the LTF bar-by-bar with strict temporal ordering.

    Anti-look-ahead guarantee
    ─────────────────────────
    A 4H bar labelled at open-time T becomes visible to the strategy only at
    T + 4h (= bar close time). We shift the zone's publication timestamp to
    bar_close = bar_open + 4h before merging, ensuring no LTF bar at time
    ≤ T can observe the zone.

    Mitigation
    ──────────
    A bull zone is invalidated when price (wick or close, per mode) drops
    below bot; a bear zone when price rises above top.  Both checks happen
    before any new signal is generated on that bar (conservative).

    Output columns added to df_ltf:
      htf_bull_active, htf_bull_top, htf_bull_bot
      htf_bear_active, htf_bear_top, htf_bear_bot
    """
    ltf = df_ltf.copy()

    # Pre-allocate output arrays (faster than .at[] in a loop)
    n = len(ltf)
    htf_bull_active = np.zeros(n, dtype=bool)
    htf_bear_active = np.zeros(n, dtype=bool)
    htf_bull_top    = np.full(n, np.nan)
    htf_bull_bot    = np.full(n, np.nan)
    htf_bear_top    = np.full(n, np.nan)
    htf_bear_bot    = np.full(n, np.nan)

    # Build sorted queues of new zones with their visibility timestamp
    freq_h = 4
    bull_q = (
        df_htf[df_htf["fvg_bull"]][["fvg_top", "fvg_bot"]]
        .copy()
        .assign(vis_ts=lambda d: d.index + pd.Timedelta(hours=freq_h))
        .reset_index(drop=True)
    )
    bear_q = (
        df_htf[df_htf["fvg_bear"]][["fvg_top", "fvg_bot"]]
        .copy()
        .assign(vis_ts=lambda d: d.index + pd.Timedelta(hours=freq_h))
        .reset_index(drop=True)
    )

    bull_ptr = 0
    bear_ptr = 0
    active_bulls: list[dict] = []
    active_bears: list[dict] = []

    ltf_arr = ltf[["high", "low", "close"]].to_numpy()
    ltf_idx = ltf.index  # DatetimeIndex

    for i in range(n):
        ts     = ltf_idx[i]
        h, l, c = ltf_arr[i]

        # ── Arrive new zones whose visibility timestamp ≤ current bar ─────
        while bull_ptr < len(bull_q) and bull_q.at[bull_ptr, "vis_ts"] <= ts:
            r = bull_q.iloc[bull_ptr]
            active_bulls.append({"top": r["fvg_top"], "bot": r["fvg_bot"]})
            bull_ptr += 1

        while bear_ptr < len(bear_q) and bear_q.at[bear_ptr, "vis_ts"] <= ts:
            r = bear_q.iloc[bear_ptr]
            active_bears.append({"top": r["fvg_top"], "bot": r["fvg_bot"]})
            bear_ptr += 1

        # ── Mitigate zones (price breaches opposite extreme) ──────────────
        if mitigation_mode == "wick":
            active_bulls = [z for z in active_bulls if l >= z["bot"]]
            active_bears = [z for z in active_bears if h <= z["top"]]
        else:
            active_bulls = [z for z in active_bulls if c >= z["bot"]]
            active_bears = [z for z in active_bears if c <= z["top"]]

        # ── Tag bar with the most-recent active zone ──────────────────────
        if active_bulls:
            z = active_bulls[-1]
            htf_bull_active[i] = True
            htf_bull_top[i]    = z["top"]
            htf_bull_bot[i]    = z["bot"]

        if active_bears:
            z = active_bears[-1]
            htf_bear_active[i] = True
            htf_bear_top[i]    = z["top"]
            htf_bear_bot[i]    = z["bot"]

    ltf["htf_bull_active"] = htf_bull_active
    ltf["htf_bull_top"]    = htf_bull_top
    ltf["htf_bull_bot"]    = htf_bull_bot
    ltf["htf_bear_active"] = htf_bear_active
    ltf["htf_bear_top"]    = htf_bear_top
    ltf["htf_bear_bot"]    = htf_bear_bot

    return ltf


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    direction:    str
    entry_ts:     pd.Timestamp
    entry_price:  float
    sl:           float
    tp:           float
    qty:          float
    exit_ts:      Optional[pd.Timestamp] = None
    exit_price:   Optional[float]        = None
    pnl:          Optional[float]        = None
    exit_reason:  str                    = ""


def _pnl(direction: str, entry: float, exit_p: float, qty: float) -> float:
    return (exit_p - entry) * qty if direction == "long" else (entry - exit_p) * qty


def run_backtest(df_ltf: pd.DataFrame, cfg: BacktestConfig
                 ) -> tuple[list[Trade], pd.Series]:
    """
    Sequential, event-driven backtest — bar-by-bar on the aligned LTF frame.

    Order of operations per bar (prevents same-bar conflicts):
      1. Check pending limit orders for fill (use current bar's L/H).
      2. Manage open trades: SL / TP check (conservative: if both hit, SL wins).
      3. Record equity snapshot.
      4. Generate new signals for the *next* bar's limit orders.
    """
    equity   = cfg.initial_equity
    eq_curve: list[dict] = []

    open_trades:   list[Trade] = []
    pending_lims:  list[dict]  = []
    closed_trades: list[Trade] = []

    # State for LTF FVG tracking
    last_bull_fvg: Optional[dict] = None   # {top, bot, bar_idx}
    last_bear_fvg: Optional[dict] = None

    arr = df_ltf[["open", "high", "low", "close",
                   "fvg_bull", "fvg_bear", "fvg_top", "fvg_bot",
                   "htf_bull_active", "htf_bull_top", "htf_bull_bot",
                   "htf_bear_active", "htf_bear_top", "htf_bear_bot"]].to_numpy()

    col = {c: i for i, c in enumerate(
        ["open", "high", "low", "close",
         "fvg_bull", "fvg_bear", "fvg_top", "fvg_bot",
         "htf_bull_active", "htf_bull_top", "htf_bull_bot",
         "htf_bear_active", "htf_bear_top", "htf_bear_bot"]
    )}
    idx_arr = df_ltf.index.to_numpy()   # numpy datetime64 array

    for i in range(len(arr)):
        ts = pd.Timestamp(idx_arr[i])
        row = arr[i]
        o, h, l, c = row[col["open"]], row[col["high"]], row[col["low"]], row[col["close"]]

        # ── 1. Fill pending limit orders ──────────────────────────────────
        still_pending = []
        for lim in pending_lims:
            filled = (lim["direction"] == "long"  and l <= lim["px"]) or \
                     (lim["direction"] == "short" and h >= lim["px"])

            if filled and len(open_trades) < cfg.max_open_trades:
                open_trades.append(Trade(
                    direction=lim["direction"],
                    entry_ts=ts, entry_price=lim["px"],
                    sl=lim["sl"], tp=lim["tp"], qty=lim["qty"],
                ))
            else:
                lim["ttl"] -= 1
                if lim["ttl"] > 0:
                    still_pending.append(lim)
        pending_lims = still_pending

        # ── 2. Manage open trades ─────────────────────────────────────────
        still_open = []
        for t in open_trades:
            hit_sl = (t.direction == "long"  and l <= t.sl) or \
                     (t.direction == "short" and h >= t.sl)
            hit_tp = (t.direction == "long"  and h >= t.tp) or \
                     (t.direction == "short" and l <= t.tp)

            if hit_sl and hit_tp:
                hit_tp = False  # conservative: assume worst case

            if hit_sl:
                t.exit_ts, t.exit_price = ts, t.sl
                t.pnl = _pnl(t.direction, t.entry_price, t.sl, t.qty)
                t.exit_reason = "SL"
                equity += t.pnl
                closed_trades.append(t)
            elif hit_tp:
                t.exit_ts, t.exit_price = ts, t.tp
                t.pnl = _pnl(t.direction, t.entry_price, t.tp, t.qty)
                t.exit_reason = "TP"
                equity += t.pnl
                closed_trades.append(t)
            else:
                still_open.append(t)
        open_trades = still_open

        # ── 3. Equity snapshot ────────────────────────────────────────────
        eq_curve.append({"ts": ts, "equity": equity})

        # ── 4. Signal generation (based on CURRENT bar data only) ─────────

        # 4a. Update LTF FVG state
        if row[col["fvg_bull"]]:
            last_bull_fvg = {"top": row[col["fvg_top"]], "bot": row[col["fvg_bot"]], "i": i}
        if row[col["fvg_bear"]]:
            last_bear_fvg = {"top": row[col["fvg_top"]], "bot": row[col["fvg_bot"]], "i": i}

        # Invalidate LTF FVG if price closed through it
        if last_bull_fvg is not None and c < last_bull_fvg["bot"]:
            last_bull_fvg = None
        if last_bear_fvg is not None and c > last_bear_fvg["top"]:
            last_bear_fvg = None

        # 4b. HTF alignment check: price inside HTF FVG zone
        htf_bull_ok = bool(row[col["htf_bull_active"]]) and \
                      row[col["htf_bull_bot"]] <= c <= row[col["htf_bull_top"]]
        htf_bear_ok = bool(row[col["htf_bear_active"]]) and \
                      row[col["htf_bear_bot"]] <= c <= row[col["htf_bear_top"]]

        total_active = len(pending_lims) + len(open_trades)

        # 4c. LONG setup: LTF bull FVG formed inside HTF bull zone
        if (htf_bull_ok
                and last_bull_fvg is not None
                and last_bull_fvg["i"] == i          # FVG formed THIS bar
                and total_active < cfg.max_open_trades):

            lim_px = last_bull_fvg["top"]
            # SL: low of the i-2 bar relative to the FVG formation bar
            sl_idx = max(0, i - 2)
            sl     = float(arr[sl_idx][col["low"]])

            if sl < lim_px:
                risk_d  = lim_px - sl
                tp      = lim_px + cfg.risk_reward * risk_d
                qty     = (equity * cfg.risk_pct) / risk_d
                pending_lims.append({
                    "direction": "long", "px": lim_px,
                    "sl": sl, "tp": tp, "qty": qty,
                    "ttl": cfg.order_expiry_bars,
                })

        # 4d. SHORT setup: LTF bear FVG formed inside HTF bear zone
        if (htf_bear_ok
                and last_bear_fvg is not None
                and last_bear_fvg["i"] == i
                and total_active < cfg.max_open_trades):

            lim_px = last_bear_fvg["bot"]
            sl_idx = max(0, i - 2)
            sl     = float(arr[sl_idx][col["high"]])

            if sl > lim_px:
                risk_d  = sl - lim_px
                tp      = lim_px - cfg.risk_reward * risk_d
                qty     = (equity * cfg.risk_pct) / risk_d
                pending_lims.append({
                    "direction": "short", "px": lim_px,
                    "sl": sl, "tp": tp, "qty": qty,
                    "ttl": cfg.order_expiry_bars,
                })

    # Force-close residuals at last price
    last_c  = float(arr[-1][col["close"]])
    last_ts = pd.Timestamp(idx_arr[-1])
    for t in open_trades:
        t.exit_ts, t.exit_price = last_ts, last_c
        t.pnl = _pnl(t.direction, t.entry_price, last_c, t.qty)
        t.exit_reason = "EOD"
        equity += t.pnl
        closed_trades.append(t)

    equity_series = (pd.DataFrame(eq_curve)
                       .set_index("ts")["equity"]
                       .rename("equity"))
    return closed_trades, equity_series


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade],
                    eq: pd.Series,
                    initial_equity: float) -> dict:
    if not trades:
        return {"error": "No trades executed — check data alignment & FVG occurrence"}

    pnls   = np.array([t.pnl   for t in trades], dtype=float)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate      = len(wins) / len(pnls)
    gross_profit  = wins.sum()           if len(wins)   > 0 else 0.0
    gross_loss    = abs(losses.sum())    if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    # Drawdown on equity curve
    eq_arr = eq.values
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak
    max_dd = float(dd.min())

    # Annualised metrics (15m bars → 365*24*4 bars/year)
    bars_year = 365 * 24 * 4
    ret       = eq.pct_change().dropna()
    ann_ret   = (eq.iloc[-1] / initial_equity) ** (bars_year / max(1, len(eq))) - 1
    sharpe    = (ret.mean() / ret.std() * np.sqrt(bars_year)
                 if ret.std() > 0 else np.nan)
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Total Trades":     len(trades),
        "Winners / Losers": f"{len(wins)} / {len(losses)}",
        "Win Rate":         f"{win_rate:.1%}",
        "Profit Factor":    f"{profit_factor:.2f}",
        "Net PnL":          f"${pnls.sum():,.2f}",
        "Avg Win":          f"${wins.mean():,.2f}"  if len(wins)   > 0 else "—",
        "Avg Loss":         f"${losses.mean():,.2f}" if len(losses) > 0 else "—",
        "Expectancy/trade": f"${pnls.mean():,.2f}",
        "Max Drawdown":     f"{max_dd:.1%}",
        "Sharpe (annual)":  f"{sharpe:.2f}",
        "Calmar":           f"{calmar:.2f}",
        "Ann. Return":      f"{ann_ret:.1%}",
    }


def trade_log(trades: list[Trade]) -> pd.DataFrame:
    rows = [{
        "entry_ts":    t.entry_ts,
        "exit_ts":     t.exit_ts,
        "direction":   t.direction,
        "entry":       round(t.entry_price, 2),
        "sl":          round(t.sl, 2),
        "tp":          round(t.tp, 2),
        "exit_price":  round(t.exit_price, 2) if t.exit_price else None,
        "pnl_usd":     round(t.pnl, 4)        if t.pnl is not None else None,
        "reason":      t.exit_reason,
    } for t in trades]
    df = pd.DataFrame(rows)
    if not df.empty:
        sign = df["direction"].map({"long": 1, "short": -1})
        df["rr"] = (
            sign * (df["exit_price"] - df["entry"])
            / (sign * (df["entry"] - df["sl"])).abs().replace(0, np.nan)
        ).round(2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(eq: pd.Series, df_ltf: pd.DataFrame,
                 tdf: pd.DataFrame, cfg: BacktestConfig) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(
        f"FVG+SMC Backtest — {cfg.symbol}  "
        f"{cfg.since_str[:10]} → {cfg.until_str[:10]}",
        fontsize=13, fontweight="bold",
    )

    # ── Equity curve ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eq.index, eq.values, color="steelblue", lw=1.3)
    ax.fill_between(eq.index, eq.values, eq.values.min(), alpha=0.12, color="steelblue")
    ax.axhline(cfg.initial_equity, color="gray", lw=0.8, linestyle="--", label="Start equity")
    ax.set_title("Equity Curve")
    ax.set_ylabel("USD")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(alpha=0.25)

    # ── Price + entries (last 1 000 LTF bars) ────────────────────────────
    ax = axes[1]
    tail = df_ltf.tail(1_000)
    ax.plot(tail.index, tail["close"], color="gray", lw=0.7, label="Close 15m")
    if not tdf.empty:
        longs  = tdf[tdf["direction"] == "long"]
        shorts = tdf[tdf["direction"] == "short"]
        ax.scatter(pd.to_datetime(longs["entry_ts"]),  longs["entry"],
                   marker="^", color="lime",   s=50, zorder=5, label="Long entry")
        ax.scatter(pd.to_datetime(shorts["entry_ts"]), shorts["entry"],
                   marker="v", color="tomato", s=50, zorder=5, label="Short entry")
    ax.set_title("15m Price + Trade Entries (last 1000 bars)")
    ax.set_ylabel("Price (USDT)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── IC z-score ────────────────────────────────────────────────────────
    ax = axes[2]
    if "ic_zscore" in df_ltf.columns:
        ic_tail = df_ltf["ic_zscore"].dropna().tail(2_000)
        ax.plot(ic_tail.index, ic_tail.values, color="mediumpurple", lw=0.8)
        ax.axhline(0,    color="black", lw=0.7)
        ax.axhline(1.5,  color="green", lw=0.8, ls="--", label="z=+1.5 (high signal quality)")
        ax.axhline(-1.5, color="red",   lw=0.8, ls="--", label="z=−1.5")
        ax.set_title("IC Z-Score — Signal Significance")
        ax.set_ylabel("Z-score")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = "results/fvg_smc_backtest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Chart → %s", out)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: BacktestConfig = CFG) -> None:

    # 1. Data
    log.info("=== Step 1 — Data ===")
    df_1m, df_15m, df_4h = fetch_data(cfg)
    log.info("1m: %d bars | 15m: %d bars | 4H: %d bars",
             len(df_1m), len(df_15m), len(df_4h))

    # 2. FVGs
    log.info("=== Step 2 — FVG detection ===")
    df_4h_fvg  = compute_fvg(df_4h)
    df_15m_fvg = compute_fvg(df_15m)
    log.info("4H FVGs  → bull: %d  bear: %d",
             df_4h_fvg["fvg_bull"].sum(), df_4h_fvg["fvg_bear"].sum())
    log.info("15m FVGs → bull: %d  bear: %d",
             df_15m_fvg["fvg_bull"].sum(), df_15m_fvg["fvg_bear"].sum())

    # 3. Align HTF → LTF
    log.info("=== Step 3 — HTF/LTF alignment ===")
    df_aligned = align_htf_fvg_to_ltf(df_4h_fvg, df_15m_fvg, cfg.mitigation_mode)
    active_bull_bars = df_aligned["htf_bull_active"].sum()
    active_bear_bars = df_aligned["htf_bear_active"].sum()
    log.info("LTF bars inside active HTF zone → bull: %d  bear: %d",
             active_bull_bars, active_bear_bars)

    # 4. IC indicator
    log.info("=== Step 4 — IC indicator ===")
    df_aligned = compute_ic_indicator(df_aligned, lookback=cfg.ic_lookback)

    # 5. Backtest
    log.info("=== Step 5 — Backtest ===")
    trades, eq_curve = run_backtest(df_aligned, cfg)
    log.info("Trades closed: %d", len(trades))

    # 6. Metrics
    metrics = compute_metrics(trades, eq_curve, cfg.initial_equity)
    sep = "═" * 52
    print(f"\n{sep}")
    print("  BACKTEST RESULTS — FVG + SMC Multi-Timeframe")
    print(sep)
    for k, v in metrics.items():
        print(f"  {k:<24} {v}")
    print(sep)

    # 7. Trade log
    os.makedirs("results", exist_ok=True)
    tdf = trade_log(trades)
    tdf.to_csv("results/fvg_smc_trades.csv", index=False)
    log.info("Trade log → results/fvg_smc_trades.csv  (%d rows)", len(tdf))
    if not tdf.empty:
        print("\nLast 10 trades:")
        print(tdf.tail(10).to_string(index=False))

    # 8. IC summary
    if "ic_rolling" in df_aligned.columns:
        ic = df_aligned["ic_rolling"].dropna()
        print(f"\nIC Summary (all {len(ic)} bars with active FVG):")
        print(f"  Mean IC  : {ic.mean():.4f}   (>0 = signal has positive edge)")
        print(f"  Std  IC  : {ic.std():.4f}")
        print(f"  IC > 0   : {(ic > 0).mean():.1%}")
        df_aligned[["signal_raw", "ic_rolling", "ic_zscore"]].dropna() \
            .to_csv("results/ic_indicator.csv")
        log.info("IC data → results/ic_indicator.csv")

    # 9. Plot
    log.info("=== Step 6 — Plot ===")
    plot_results(eq_curve, df_aligned, tdf, cfg)
    log.info("Done.")


if __name__ == "__main__":
    main()
