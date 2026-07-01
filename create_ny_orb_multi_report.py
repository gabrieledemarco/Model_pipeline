"""
create_ny_orb_multi_report.py
==============================
NY Opening Range Breakout — EURUSD & XAUUSD
Data: Dukascopy 1M BID candles (bi5 format) → resampled 15M + 1H
Pipeline: IS scan → Walk-Forward (6m/2m/2m) → Monte Carlo → HTML report

bi5 binary format (24 bytes/record, big-endian):
  uint32  seconds from midnight UTC
  uint32  Open  × point_value
  uint32  Close × point_value
  uint32  Low   × point_value
  uint32  High  × point_value
  float32 Volume (ticks)
"""
from __future__ import annotations

import base64
import io
import lzma
import struct
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.monte_carlo import run_monte_carlo

# ─────────────────────────────────────────────────────────────────────────────
# Instrument config
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "EURUSD": {
        "point_value": 100_000,
        "min_body":    0.0001,    # 1 pip minimum (very permissive)
        "fee":         0.00004,   # ~0.5 pip/side at ~1.05 EURUSD
        "label":       "EUR/USD",
        "price_fmt":   ".5f",
    },
    "XAUUSD": {
        "point_value": 1_000,
        "min_body":    0.10,      # $0.10 minimum
        "fee":         0.0001,    # ~$0.20/oz per side at $2000
        "label":       "XAU/USD (Gold)",
        "price_fmt":   ".3f",
    },
}

# Strategy params (IS-optimised per instrument at runtime)
NY_HOUR     = 14
NY_MIN      = 30
NY_END_HOUR = 20
MAX_HOLD    = 32       # 15M bars = 8h
IC_HORIZON  = 16       # 15M bars = 4h forward return
RISK_PCT    = 0.01
INIT_CAP    = 100_000.0

# IS scan grid
TP_FRAC_GRID = [1.0, 1.5, 2.0, 3.0]
SL_BUF_GRID  = [0.00, 0.25, 0.50, 1.00]
RBA_MAX_GRID = [999]   # body/ATR filter off (no reliable ATR filter for forex)

# Walk-forward
WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2
START_YEAR  = 2020

N_SIMS = 5_000

DATA_DIR  = Path("data/dukascopy")
REPORT_DIR = Path("reports")

DUKA_BASE = "https://datafeed.dukascopy.com/datafeed"

# HTML colours
_BG   = "#0f1117"
_CARD = "#12151f"
_GRID = "#1e2130"
_TEXT = "#e0e0e0"
_ACC  = "#42a5f5"
_GRN  = "#66bb6a"
_RED  = "#ef5350"
_YEL  = "#ffd54f"
_ORG  = "#ffa726"

SEP  = "─" * 72
SEP2 = "═" * 72


# ─────────────────────────────────────────────────────────────────────────────
# Dukascopy downloader
# ─────────────────────────────────────────────────────────────────────────────
def _decode_bi5(raw_bytes: bytes, point_value: int) -> list[tuple]:
    data = lzma.decompress(raw_bytes)
    n = len(data) // 24
    out = []
    for i in range(n):
        ts_s, o, c, lo, hi, v = struct.unpack(">IIIIIf", data[i * 24 : i * 24 + 24])
        out.append((ts_s, o / point_value, hi / point_value, lo / point_value,
                    c / point_value, float(v)))
    return out


def _fetch_day(session: requests.Session, pair: str, d: date,
               point_value: int) -> Optional[pd.DataFrame]:
    month_0idx = d.month - 1
    url = (f"{DUKA_BASE}/{pair}/{d.year}/{month_0idx:02d}/"
           f"{d.day:02d}/BID_candles_min_1.bi5")
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200 or len(resp.content) < 50:
            return None
        records = _decode_bi5(resp.content, point_value)
        if not records:
            return None
        base = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz="UTC")
        rows = [(base + pd.Timedelta(seconds=int(r[0])), r[1], r[2], r[3], r[4], r[5])
                for r in records]
        df = pd.DataFrame(rows,
                          columns=["datetime", "open", "high", "low", "close", "volume"])
        df = df.set_index("datetime")
        # Drop bars where all OHLC are 0 (market closed periods)
        df = df[df["close"] > 0]
        return df if not df.empty else None
    except Exception:
        return None


def fetch_dukascopy_1m(pair: str, start_year: int, point_value: int) -> pd.DataFrame:
    cache = DATA_DIR / f"{pair}_1m_{start_year}.parquet"
    if cache.exists():
        print(f"  Loading {pair} 1M from cache …")
        return pd.read_parquet(cache)

    print(f"  Downloading {pair} from Dukascopy ({start_year}–now) …")
    start = date(start_year, 1, 1)
    end   = date.today() - timedelta(days=1)
    all_days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
    weekdays = [d for d in all_days if d.weekday() < 5]

    results: list[pd.DataFrame] = []
    total = len(weekdays)
    done  = 0

    with requests.Session() as session, ThreadPoolExecutor(max_workers=25) as ex:
        futs = {ex.submit(_fetch_day, session, pair, d, point_value): d
                for d in weekdays}
        for fut in as_completed(futs):
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{total} …")
            df = fut.result()
            if df is not None:
                results.append(df)

    if not results:
        raise ValueError(f"No data for {pair}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_all = pd.concat(results).sort_index()
    df_all.to_parquet(cache)
    print(f"    Saved: {cache}  ({len(df_all):,} bars)")
    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# Resample helpers
# ─────────────────────────────────────────────────────────────────────────────
_RESAMPLE_AGG = {"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"}


def _resample(df_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = (df_1m.resample(freq, label="left", closed="left")
               .agg(_RESAMPLE_AGG).dropna(subset=["open", "close"]))
    return df[df["close"] > 0]


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def prepare_frames(df_1m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_15m = _resample(df_1m, "15min")
    df_1h  = _resample(df_1m, "1h")
    df_1h["atr_14"] = _atr(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    df_15m["atr_14"] = _atr(df_15m["high"], df_15m["low"], df_15m["close"], 14)
    return df_15m, df_1h


# ─────────────────────────────────────────────────────────────────────────────
# NY ORB event collector
# ─────────────────────────────────────────────────────────────────────────────
def collect_events(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                   min_body: float) -> list[dict]:
    IDX   = df_15m.index
    H_arr = np.array(IDX.hour,   dtype=int)
    M_arr = np.array(IDX.minute, dtype=int)
    HI    = df_15m["high"].values
    LO    = df_15m["low"].values
    CL    = df_15m["close"].values
    OP    = df_15m["open"].values
    N     = len(df_15m)
    yr_arr = np.array([t.year for t in IDX], dtype=int)

    # ATR 1H aligned to previous completed 1H bar
    prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
    atr_1h_map = df_1h["atr_14"].clip(lower=1e-8).to_dict()
    fallback   = df_15m["atr_14"].clip(lower=1e-8).values
    ATR_1H     = np.array([atr_1h_map.get(t, np.nan) for t in prev_1h], dtype=float)
    ATR_1H     = np.where(np.isnan(ATR_1H), fallback, ATR_1H)

    events: list[dict] = []
    seen_dates: set = set()

    for i in range(N - MAX_HOLD - 2):
        if H_arr[i] != NY_HOUR or M_arr[i] != NY_MIN:
            continue
        d = IDX[i].normalize()
        if d in seen_dates:
            continue
        seen_dates.add(d)

        body_hi = max(OP[i], CL[i])
        body_lo = min(OP[i], CL[i])
        rw = body_hi - body_lo
        if rw < min_body:
            continue

        atr_i = ATR_1H[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue
        rba = rw / atr_i

        for j in range(i + 1, min(i + MAX_HOLD + 1, N - MAX_HOLD - 1)):
            if IDX[j].hour >= NY_END_HOUR:
                break
            direction = None
            if CL[j] > body_hi:
                direction = "long"
            elif CL[j] < body_lo:
                direction = "short"
            if direction is not None:
                events.append({
                    "direction": direction,
                    "entry_i":   j,
                    "entry_px":  CL[j],
                    "body_hi":   body_hi,
                    "body_lo":   body_lo,
                    "rw":        rw,
                    "atr_1h":    atr_i,
                    "rba":       rba,
                    "year":      yr_arr[i],
                    "ts":        IDX[i],
                })
                break

    return events, HI, LO, CL, IDX, N


# ─────────────────────────────────────────────────────────────────────────────
# IS parameter scan
# ─────────────────────────────────────────────────────────────────────────────
def is_scan(events: list[dict], HI: np.ndarray, LO: np.ndarray,
             CL: np.ndarray) -> tuple[pd.DataFrame, dict]:
    # Pre-cache price paths
    paths: list[tuple] = []
    for ev in events:
        s = ev["entry_i"] + 1
        paths.append((HI[s : s + MAX_HOLD], LO[s : s + MAX_HOLD]))

    results = []
    for tp_frac, sl_buf, rba_max in product(TP_FRAC_GRID, SL_BUF_GRID, RBA_MAX_GRID):
        wins = losses = n_valid = 0
        total_tp = total_sl = 0.0

        for ev, (ph, pl) in zip(events, paths):
            if ev["rba"] > rba_max:
                continue
            entry = ev["entry_px"]
            rw = ev["rw"]
            atr = ev["atr_1h"]
            if ev["direction"] == "long":
                tp_px = entry + tp_frac * rw
                sl_px = ev["body_lo"] - sl_buf * atr
                if sl_px >= entry or tp_px <= entry:
                    continue
                sl_d = entry - sl_px
                tp_d = tp_px - entry
                hit_tp = hit_sl = False
                for h, l in zip(ph, pl):
                    if h >= tp_px and not hit_sl:
                        hit_tp = True; break
                    if l <= sl_px:
                        hit_sl = True; break
            else:
                tp_px = entry - tp_frac * rw
                sl_px = ev["body_hi"] + sl_buf * atr
                if sl_px <= entry or tp_px >= entry:
                    continue
                sl_d = sl_px - entry
                tp_d = entry - tp_px
                hit_tp = hit_sl = False
                for h, l in zip(ph, pl):
                    if l <= tp_px and not hit_sl:
                        hit_tp = True; break
                    if h >= sl_px:
                        hit_sl = True; break

            if sl_d <= 0 or tp_d <= 0:
                continue
            n_valid += 1
            total_tp += tp_d / entry * 100
            total_sl += sl_d / entry * 100
            if hit_tp:
                wins += 1
            elif hit_sl:
                losses += 1

        if n_valid < 10:
            continue
        wr   = wins / n_valid * 100
        rr   = (total_tp / n_valid) / (total_sl / n_valid) if total_sl > 0 else 0
        be   = 1 / (1 + rr) * 100 if rr > 0 else 50.0
        avg_tp = total_tp / n_valid
        avg_sl = total_sl / n_valid
        exp = wr / 100 * avg_tp - (1 - wr / 100) * avg_sl
        p_val = st.binomtest(int(round(wr / 100 * n_valid)), n_valid,
                              be / 100, alternative="greater").pvalue
        results.append({"tp_frac": tp_frac, "sl_buf": sl_buf, "rba_max": rba_max,
                         "n": n_valid, "wr": round(wr, 2), "rr": round(rr, 2),
                         "be": round(be, 2), "exp": round(exp, 5),
                         "p_val": round(p_val, 4)})

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res, {"tp_frac": 2.0, "sl_buf": 0.25, "rba_max": 999.0}
    best = df_res.sort_values("exp", ascending=False).iloc[0].to_dict()
    return df_res, best


# ─────────────────────────────────────────────────────────────────────────────
# IC computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_ic(events: list[dict], CL: np.ndarray, N: int) -> tuple[float, float]:
    signals, fwd_rets = [], []
    for ev in events:
        ei  = ev["entry_i"]
        end = min(ei + IC_HORIZON, N - 1)
        fwd = (CL[end] - ev["entry_px"]) / ev["entry_px"] * 100.0
        sig = 1.0 if ev["direction"] == "long" else -1.0
        signals.append(sig)
        fwd_rets.append(sig * fwd)
    if len(signals) < 10:
        return 0.0, 1.0
    ic, ic_p = st.spearmanr(signals, fwd_rets)
    return float(ic), float(ic_p)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine (fixed-fractional risk, per-instrument fee)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    direction:   int
    entry_price: float
    exit_price:  float
    stop_price:  float
    tp_price:    float
    net_pnl:     float
    gross_pnl:   float
    total_fees:  float
    exit_reason: str
    year:        int
    window_id:   int


def run_backtest(
    events: list[dict],
    HI: np.ndarray, LO: np.ndarray, CL: np.ndarray,
    IDX: pd.DatetimeIndex, n_bars: int,
    tp_frac: float, sl_buf: float, rba_max: float,
    fee: float,
    initial_capital: float = INIT_CAP,
    window_id: int = 0,
) -> tuple[list[Trade], pd.Series]:
    equity = float(initial_capital)
    trades: list[Trade] = []

    for ev in events:
        if ev["rba"] > rba_max:
            continue
        entry_px = ev["entry_px"]
        rw       = ev["rw"]
        atr_1h   = ev["atr_1h"]
        ei       = ev["entry_i"]
        direction = 1 if ev["direction"] == "long" else -1

        if direction == 1:
            tp_px = entry_px + tp_frac * rw
            sl_px = ev["body_lo"] - sl_buf * atr_1h
        else:
            tp_px = entry_px - tp_frac * rw
            sl_px = ev["body_hi"] + sl_buf * atr_1h

        sl_dist = abs(entry_px - sl_px)
        tp_dist = abs(tp_px - entry_px)
        if sl_dist <= 0 or tp_dist <= 0:
            continue
        if direction == 1 and sl_px >= entry_px:
            continue
        if direction == -1 and sl_px <= entry_px:
            continue

        at_risk   = equity * RISK_PCT
        qty       = at_risk / sl_dist
        notional  = qty * entry_px
        entry_fee = notional * fee

        start    = ei + 1
        hit_tp = hit_sl = False
        exit_bar = min(start + MAX_HOLD, n_bars - 1)
        exit_px  = float(CL[exit_bar])

        for k in range(start, min(start + MAX_HOLD, n_bars)):
            bh, bl = float(HI[k]), float(LO[k])
            if direction == 1:
                if bl <= sl_px:
                    hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bh >= tp_px:
                    hit_tp = True; exit_px = tp_px; exit_bar = k; break
            else:
                if bh >= sl_px:
                    hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bl <= tp_px:
                    hit_tp = True; exit_px = tp_px; exit_bar = k; break

        gross_pnl = direction * qty * (exit_px - entry_px)
        exit_fee  = qty * exit_px * fee
        net_pnl   = gross_pnl - entry_fee - exit_fee
        equity   += net_pnl
        reason    = "tp" if hit_tp else ("sl" if hit_sl else "time")

        trades.append(Trade(
            entry_ts    = IDX[ei],
            exit_ts     = IDX[exit_bar],
            direction   = direction,
            entry_price = entry_px,
            exit_price  = exit_px,
            stop_price  = sl_px,
            tp_price    = tp_px,
            gross_pnl   = gross_pnl,
            total_fees  = entry_fee + exit_fee,
            net_pnl     = net_pnl,
            exit_reason = reason,
            year        = ev["year"],
            window_id   = window_id,
        ))

    if not trades:
        return trades, pd.Series(dtype=float)

    eq = np.empty(len(trades) + 1)
    eq[0] = initial_capital
    for k, t in enumerate(trades):
        eq[k + 1] = eq[k] + t.net_pnl

    return trades, pd.Series(eq[1:], index=pd.DatetimeIndex([t.exit_ts for t in trades]))


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────
def _wf_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
    from dateutil.relativedelta import relativedelta
    wins, cur = [], start
    while True:
        tr_e = cur + relativedelta(months=WF_TRAIN_M)
        oo_e = tr_e + relativedelta(months=WF_OOS_M)
        if oo_e > end:
            break
        wins.append((cur, tr_e, oo_e))
        cur = cur + relativedelta(months=WF_STEP_M)
    return wins


def _kpis(equity: pd.Series, init_cap: float = INIT_CAP) -> dict:
    if equity.empty:
        return dict(total_return=0, calmar=0, sharpe=0, max_dd=0, n_trades=0)
    full = pd.concat([pd.Series([init_cap],
                     index=[equity.index[0] - pd.Timedelta("1s")]), equity])
    dd   = (full / full.cummax() - 1).min()
    ret  = full.iloc[-1] / init_cap - 1
    rets = full.pct_change().dropna()
    vol  = rets.std() * np.sqrt(365 * 24)
    ann  = rets.mean() * 365 * 24
    sharpe = ann / vol if vol > 0 else 0.0
    calmar = ret / abs(dd) if dd < 0 else 0.0
    return dict(total_return=ret, calmar=calmar, sharpe=sharpe,
                max_dd=dd, n_trades=len(equity))


def run_wf(events: list[dict],
           HI, LO, CL, IDX, N,
           tp_frac: float, sl_buf: float, rba_max: float, fee: float):
    data_start = IDX[0]
    data_end   = IDX[-1]
    windows    = _wf_windows(data_start, data_end)

    window_rets, window_meta, all_oos_events = [], [], []

    for wid, (tr_s, tr_e, oo_e) in enumerate(windows):
        is_evs = [e for e in events if tr_s <= e["ts"] < tr_e]
        oo_evs = [e for e in events if tr_e <= e["ts"] < oo_e]
        if len(is_evs) < 5 or len(oo_evs) < 3:
            continue

        oo_t, oo_eq = run_backtest(
            oo_evs, HI, LO, CL, IDX, N,
            tp_frac=tp_frac, sl_buf=sl_buf, rba_max=rba_max, fee=fee,
            window_id=wid,
        )
        if not oo_t:
            continue

        kp = _kpis(oo_eq)
        window_rets.append(kp["total_return"])
        window_meta.append({"wid": wid, "tr_s": tr_s.date(), "tr_e": tr_e.date(),
                             "oo_e": oo_e.date(), "n_is": len(is_evs),
                             "n_oos": len(oo_t),
                             **{k: round(v, 4) for k, v in kp.items()}})
        all_oos_events.extend(oo_evs)

    all_oos_sorted = sorted(all_oos_events, key=lambda e: e["ts"])
    oos_trades, oos_equity = run_backtest(
        all_oos_sorted, HI, LO, CL, IDX, N,
        tp_frac=tp_frac, sl_buf=sl_buf, rba_max=rba_max, fee=fee,
    )
    return oos_trades, oos_equity, window_rets, window_meta, windows


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _imgt(b64: str, w: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{w};border-radius:6px">'


def _style(v: float, fmt: str = ".1%", good: float = 0) -> str:
    color = _GRN if v > good else _RED
    return f'<span style="color:{color}">{v:{fmt}}</span>'


def _plot_equity(oos_equity: pd.Series, label: str, color: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_CARD)
        ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values():
            sp.set_color(_GRID)

    ax1.plot(oos_equity.index, oos_equity.values / INIT_CAP,
             color=color, lw=1.5, label=label)
    ax1.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (norm.)", color=_TEXT)
    ax1.legend(facecolor=_CARD, labelcolor=_TEXT)

    rm = oos_equity.cummax()
    dd = (oos_equity / rm - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, color=_RED, alpha=0.45, label="DD %")
    ax2.set_ylabel("DD %", color=_TEXT)
    ax2.legend(facecolor=_CARD, labelcolor=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle(f"OOS Equity — {label}", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_wf(window_rets: list[float], label: str, color: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values():
        sp.set_color(_GRID)
    clrs = [_GRN if r > 0 else _RED for r in window_rets]
    ax.bar(range(len(window_rets)), [r * 100 for r in window_rets],
           color=clrs, alpha=0.8)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_xlabel("OOS Window #", color=_TEXT)
    ax.set_ylabel("Return %", color=_TEXT)
    ax.set_title(f"OOS per-window returns — {label}", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_mc(mc_result: dict, label: str) -> str:
    final = mc_result["total_return"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values():
        sp.set_color(_GRID)
    ax.hist(final, bins=80, color=_ACC, alpha=0.7, edgecolor="none")
    ax.axvline(0, color=_RED, lw=1.5, ls="--", label="Breakeven")
    ax.axvline(np.median(final), color=_YEL, lw=1.5,
               label=f"Median {np.median(final):.1f}%")
    ax.set_xlabel("Total Return %", color=_TEXT)
    ax.set_ylabel("Count", color=_TEXT)
    ax.set_title(f"Monte Carlo — {label} ({N_SIMS} sims)", color=_TEXT)
    ax.legend(facecolor=_CARD, labelcolor=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────
def _card(title: str, body: str, accent: str = _ACC) -> str:
    return f"""
<div style="background:{_CARD};border:1px solid {_GRID};border-radius:10px;
     padding:18px 22px;margin-bottom:18px">
  <h3 style="color:{accent};margin-top:0;margin-bottom:12px">{title}</h3>
  {body}
</div>"""


def _kv(label: str, value: str) -> str:
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:4px 0">'
            f'<span style="color:#9e9e9e">{label}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{value}</span></div>')


def _table(headers: list[str], rows: list[list]) -> str:
    th = "".join(f'<th style="padding:5px 8px;text-align:right;color:{_ACC}'
                 f';border-bottom:1px solid {_GRID}">{h}</th>' for h in headers)
    body = ""
    for row in rows:
        tds = "".join(f'<td style="padding:4px 8px;text-align:right;'
                      f'color:{_TEXT}">{c}</td>' for c in row)
        body += f"<tr>{tds}</tr>"
    return (f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


# ═════════════════════════════════════════════════════════════════════════════
# Per-instrument pipeline
# ═════════════════════════════════════════════════════════════════════════════
def run_instrument(pair: str, cfg: dict) -> dict:
    label       = cfg["label"]
    point_value = cfg["point_value"]
    min_body    = cfg["min_body"]
    fee         = cfg["fee"]

    print(f"\n{SEP2}")
    print(f"  {label} ({pair})")
    print(SEP2)

    # 1. Data
    print("[1/5] Caricamento dati …")
    df_1m          = fetch_dukascopy_1m(pair, START_YEAR, point_value)
    df_15m, df_1h  = prepare_frames(df_1m)
    print(f"  1M:  {len(df_1m):,} bar  ({df_1m.index[0].date()} → {df_1m.index[-1].date()})")
    print(f"  15M: {len(df_15m):,} bar")
    print(f"  1H:  {len(df_1h):,} bar")

    # 2. Events
    print("[2/5] Raccolta eventi NY ORB …")
    events, HI, LO, CL, IDX, N = collect_events(df_15m, df_1h, min_body)
    n_long  = sum(1 for e in events if e["direction"] == "long")
    n_short = sum(1 for e in events if e["direction"] == "short")
    print(f"  Totale: {len(events)}  (long={n_long}, short={n_short})")
    if events:
        rbas = [e["rba"] for e in events]
        print(f"  rba  mean={np.mean(rbas):.2f}  p25={np.percentile(rbas,25):.2f}"
              f"  p50={np.percentile(rbas,50):.2f}  p75={np.percentile(rbas,75):.2f}")

    # IC
    ic, ic_p = compute_ic(events, CL, N)
    print(f"  IC (Spearman): {ic:.4f}  p={ic_p:.4f}"
          f"  {'✓' if ic_p < 0.05 else '✗'}")

    # 3. IS scan
    print("[3/5] IS parameter scan …")
    df_scan, best = is_scan(events, HI, LO, CL)
    tp_frac = best["tp_frac"]
    sl_buf  = best["sl_buf"]
    rba_max = best["rba_max"]
    print(f"  Best: tp_frac={tp_frac}  sl_buf={sl_buf}  rba_max={rba_max}")
    print(f"        N={int(best['n'])}  WR={best['wr']:.1f}%  "
          f"BE={best['be']:.1f}%  ExpPnL={best['exp']:+.5f}%  p={best['p_val']:.4f}")

    # 4. Walk-Forward
    print("[4/5] Walk-Forward …")
    oos_trades, oos_equity, window_rets, window_meta, windows = run_wf(
        events, HI, LO, CL, IDX, N, tp_frac, sl_buf, rba_max, fee,
    )
    print(f"  Finestre: {len(windows)}  |  Valide: {len(window_meta)}"
          f"  |  OOS trades: {len(oos_trades)}")

    if not oos_trades:
        print("  NESSUN trade OOS.")
        return {}

    kp_oos   = _kpis(oos_equity)
    oos_df   = pd.DataFrame([t.__dict__ for t in oos_trades])
    oos_df   = oos_df.sort_values("entry_ts").reset_index(drop=True)
    wins_oos = (oos_df["net_pnl"] > 0).sum()
    wr_oos   = wins_oos / len(oos_df)

    tp_d = np.mean(np.abs(oos_df["tp_price"] - oos_df["entry_price"]) / oos_df["entry_price"] * 100)
    sl_d = np.mean(np.abs(oos_df["stop_price"] - oos_df["entry_price"]) / oos_df["entry_price"] * 100)
    rr_oos = tp_d / sl_d if sl_d > 0 else 0
    be_oos  = 1 / (1 + rr_oos) * 100 if rr_oos > 0 else 50.0
    binom_p = st.binomtest(int(wins_oos), len(oos_df), be_oos / 100, alternative="greater").pvalue

    oos_tmp = oos_df.copy()
    oos_tmp["date"] = pd.to_datetime(oos_tmp["exit_ts"]).dt.normalize()
    daily = oos_tmp.groupby("date")["net_pnl"].sum()
    t_stat, t_p = st.ttest_1samp(daily.values, 0) if len(daily) > 1 else (0, 1)

    print(f"  Return: {kp_oos['total_return']:+.1%}  Max DD: {kp_oos['max_dd']:.1%}"
          f"  WR: {wr_oos:.1%}  BE: {be_oos:.1f}%  binom p={binom_p:.4f}")

    # 5. Monte Carlo
    print("[5/5] Monte Carlo …")
    mc_df  = oos_df[["net_pnl", "gross_pnl", "total_fees", "entry_price", "exit_price"]].copy()
    mc_res = run_monte_carlo(mc_df, INIT_CAP, N_SIMS)
    mc_fin = mc_res["total_return"] * 100
    mc_dd  = mc_res["max_drawdown"] * 100
    p_profit = float((mc_res["total_return"] > 0).mean())
    p_ruin   = mc_res.get("p_ruin", float((mc_res["total_return"] < -0.5).mean()))
    print(f"  P(profit)={p_profit:.1%}  P(ruin)={p_ruin:.1%}"
          f"  Median ret={np.median(mc_fin):.1f}%")

    return {
        "pair": pair, "label": label, "fee": fee, "cfg": cfg,
        "events": events, "df_15m": df_15m,
        "best": best, "df_scan": df_scan,
        "ic": ic, "ic_p": ic_p,
        "oos_df": oos_df, "oos_equity": oos_equity,
        "kp_oos": kp_oos, "wr_oos": wr_oos, "be_oos": be_oos,
        "binom_p": binom_p, "t_p": t_p, "rr_oos": rr_oos,
        "window_rets": window_rets, "window_meta": window_meta,
        "mc_res": mc_res, "mc_fin": mc_fin, "mc_dd": mc_dd,
        "p_profit": p_profit, "p_ruin": p_ruin,
        "n_long": n_long, "n_short": n_short,
    }


# ═════════════════════════════════════════════════════════════════════════════
# HTML Report
# ═════════════════════════════════════════════════════════════════════════════
def build_html(results: dict[str, dict]) -> str:
    colors = {"EURUSD": _ACC, "XAUUSD": _ORG}

    # Summary table
    sum_rows = []
    for pair, r in results.items():
        if not r:
            continue
        kp = r["kp_oos"]
        verdict = "✅" if (kp["total_return"] > 0 and r["binom_p"] < 0.05
                           and r["p_profit"] > 0.5) else "✗"
        sum_rows.append([
            f'<span style="color:{colors[pair]}">{r["label"]}</span>',
            len(r["events"]),
            f'{r["ic"]:.4f} (p={r["ic_p"]:.3f})',
            f'{r["best"]["tp_frac"]} / {r["best"]["sl_buf"]}',
            f'{r["best"]["wr"]:.1f}% (BE {r["best"]["be"]:.1f}%)',
            f'{r["best"]["exp"]:+.5f}%',
            len(r["oos_df"]),
            f'{r["wr_oos"]:.1%} (BE {r["be_oos"]:.1f}%)',
            f'{kp["total_return"]:+.1%}',
            f'{kp["max_dd"]:.1%}',
            f'{kp["calmar"]:+.2f}',
            f'{r["binom_p"]:.4f}',
            f'{r["p_profit"]:.1%} / {r["p_ruin"]:.1%}',
            verdict,
        ])

    summary_table = _table(
        ["Instrument", "Events IS", "IC (Spearman)", "TP/SL best",
         "WR% IS", "ExpPnL IS", "Trades OOS", "WR% OOS",
         "Return OOS", "Max DD", "Calmar", "Binom p",
         "P(profit)/P(ruin)", "Verdict"],
        sum_rows,
    )

    # Per-instrument sections
    sections = ""
    for pair, r in results.items():
        if not r:
            sections += f"<h2>{pair} — no data</h2>"
            continue
        col = colors[pair]
        kp  = r["kp_oos"]

        img_eq = _plot_equity(r["oos_equity"], r["label"], col)
        img_wf = _plot_wf(r["window_rets"], r["label"], col)
        img_mc = _plot_mc(r["mc_res"], r["label"])

        verdict_ok  = (kp["total_return"] > 0 and r["binom_p"] < 0.05
                       and r["p_profit"] > 0.5)
        verdict_col = _GRN if verdict_ok else _RED
        verdict_txt = "✅ VALIDATA" if verdict_ok else "✗ NON VALIDATA"

        oos_df = r["oos_df"]
        ann_rows = []
        for yr in sorted(oos_df["year"].unique()):
            sub = oos_df[oos_df["year"] == yr]
            w = (sub["net_pnl"] > 0).sum()
            wr_y = w / len(sub) * 100
            ret_y = sub["net_pnl"].sum()
            c = _GRN if ret_y > 0 else _RED
            ann_rows.append([yr, len(sub), f"{wr_y:.1f}%",
                              f'<span style="color:{c}">{ret_y:+,.0f}</span>'])
        annual_tbl = _table(["Anno", "N trade", "WR%", "Net PnL (USD)"], ann_rows)

        strat_body = "".join([
            _kv("Best tp_frac",   f"{r['best']['tp_frac']}× body"),
            _kv("Best sl_buf",    f"{r['best']['sl_buf']}× ATR_1H"),
            _kv("Fee",            f"{r['fee']*100:.4f}% per side"),
            _kv("IC (Spearman)",  f"{r['ic']:.4f}  p={r['ic_p']:.4f}"),
            _kv("IS Events",      f"{len(r['events'])}  (Long={r['n_long']}, Short={r['n_short']})"),
            _kv("IS WR%",         f"{r['best']['wr']:.1f}% (BE {r['best']['be']:.1f}%)"),
            _kv("IS ExpPnL",      f"{r['best']['exp']:+.5f}%  p={r['best']['p_val']:.4f}"),
        ])
        oos_body = "".join([
            _kv("OOS Trades",      str(len(oos_df))),
            _kv("WR OOS",          _style(r["wr_oos"], ".1%")),
            _kv("Breakeven WR",    f"{r['be_oos']:.1f}%"),
            _kv("R:R OOS",         f"{r['rr_oos']:.2f}"),
            _kv("Total Return",    _style(kp["total_return"], ".1%")),
            _kv("Max DD",          _style(kp["max_dd"], ".1%")),
            _kv("Calmar",          _style(kp["calmar"], ".3f")),
            _kv("Sharpe",          _style(kp["sharpe"], ".3f")),
            _kv("Binomial p",      f"{r['binom_p']:.4f}"),
            _kv("t-test daily p",  f"{r['t_p']:.4f}"),
        ])
        mc_body = "".join([
            _kv("P(profit)",       _style(r["p_profit"], ".1%")),
            _kv("P(ruin <-50%)",   _style(r["p_ruin"], ".1%", good=-1)),
            _kv("Median return",   f"{np.median(r['mc_fin']):.1f}%"),
            _kv("5th pct return",  f"{np.percentile(r['mc_fin'], 5):.1f}%"),
            _kv("95th pct DD",     f"{np.percentile(r['mc_dd'], 95):.1f}%"),
        ])

        sections += f"""
<hr style="border-color:{_GRID};margin:32px 0">
<h2 style="color:{col}">{r["label"]} — NY ORB Body Breakout</h2>
<div style="background:{_CARD};border:2px solid {verdict_col};border-radius:8px;
     padding:14px;margin-bottom:20px;text-align:center;font-size:18px;
     font-weight:bold;color:{verdict_col}">{verdict_txt}</div>
<div class="grid2">
  {_card("Strategia & IS Scan", strat_body, col)}
  {_card("OOS Performance", oos_body, col)}
</div>
<div class="grid2">
  {_card(f"Monte Carlo ({N_SIMS:,} sims)", mc_body, col)}
  {_card("Breakdown Annuale (OOS)", annual_tbl, col)}
</div>
<h3 style="color:{col}">OOS Equity Curve</h3>
{_imgt(img_eq)}
<h3 style="color:{col}">OOS Per-Window Returns</h3>
{_imgt(img_wf)}
<h3 style="color:{col}">Monte Carlo — Distribuzione Rendimento</h3>
{_imgt(img_mc)}
"""

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>NY ORB Multi-Symbol Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{_BG};color:{_TEXT};font-family:monospace;font-size:13px;padding:24px}}
  h1{{color:{_ACC};margin-bottom:8px}}
  h2{{color:{_YEL};margin:20px 0 10px;font-size:15px}}
  h3{{color:{_TEXT};margin:14px 0 8px;font-size:13px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
  table{{overflow-x:auto;display:block}}
</style>
</head>
<body>
<h1>NY ORB Body Breakout — EURUSD & XAUUSD</h1>
<p style="color:#9e9e9e;margin-bottom:20px">
  Dukascopy 15M BID data &nbsp;|&nbsp; {START_YEAR}-01-01 → now &nbsp;|&nbsp;
  WF {WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / step {WF_STEP_M}m &nbsp;|&nbsp;
  Risk {RISK_PCT*100:.0f}%/trade &nbsp;|&nbsp; Cap ${INIT_CAP:,.0f}
</p>

<h2>Comparison Summary</h2>
{_card("Multi-Symbol IS/OOS Comparison", summary_table)}

{sections}

<p style="color:#555;margin-top:32px;font-size:11px">
  NY ORB Multi-Symbol Pipeline — Dukascopy BID 1M → 15M/1H |
  Walk-Forward {WF_TRAIN_M}m/{WF_OOS_M}m | MC {N_SIMS:,} bootstrap sims
</p>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
print(SEP2)
print("NY ORB Body Breakout — Multi-Symbol: EURUSD & XAUUSD")
print(SEP2)

results: dict[str, dict] = {}
for pair, cfg in INSTRUMENTS.items():
    results[pair] = run_instrument(pair, cfg)

# Build and save report
print(f"\n{SEP2}")
print("Generazione report HTML …")
REPORT_DIR.mkdir(exist_ok=True)
html = build_html(results)
out_path = REPORT_DIR / "report_ny_orb_multi.html"
out_path.write_text(html, encoding="utf-8")
print(f"Report scritto in: {out_path}")
print(SEP2)
