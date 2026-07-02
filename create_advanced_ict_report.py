"""
create_advanced_ict_report.py
==============================
Validation pipeline for 3 advanced ICT strategies on BTCUSDT 15M (2020-2026):

  1. IFVG  — Inverse Fair Value Gap
             A violated standard FVG inverts its role:
             bullish FVG broken downward → becomes resistance (short on retest)
             bearish FVG broken upward  → becomes support   (long on retest)
             SL: beyond highest/lowest extreme of the violation move

  2. MTF   — Multi-Timeframe Combo (4H → 15M cascade)
             4H EMA bias → 15M CHoCH/BOS confirmation → first 15M FVG entry
             SL: beyond CHoCH structural extreme

  3. LTF   — LTF Logic Flow  (Grab → Shift → Entry → Confirm)
             TS  = Turtle Soup liquidity sweep (wick beyond swing H/L, close back)
             CISD = first strong reversal candle after sweep
             IFVG = first FVG in new direction after CISD
             Entry inside IFVG on retracement
             SL: beyond TS sweep extreme (sl_base)

Same validation protocol as ICT suite:
  IC (Spearman) → IS scan (tp/sl grid) → Walk-Forward (6m/2m) → Monte Carlo (5000 sims)
"""
from __future__ import annotations

import base64
import io
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.monte_carlo  import run_monte_carlo

# ─────────────────────────────────────────────────────────────────────────────
# Config — identical to ICT suite
# ─────────────────────────────────────────────────────────────────────────────
INIT_CAP     = 100_000.0
RISK_PCT     = 0.01
FEE          = 0.0004       # 4 bps per side (Binance taker)
MIN_SL_ATR   = 0.50         # sizing floor: ≥ 0.5× ATR_1H (caps notional)
MAX_LEV      = 5.0          # hard cap: notional ≤ 5× equity
MAX_HOLD     = 32           # 15M bars = 8 h
IC_HORIZON   = 16           # 15M bars = 4 h forward return for IC
START_YEAR   = 2020
N_SIMS       = 5_000

WF_TRAIN_M   = 6
WF_OOS_M     = 2
WF_STEP_M    = 2

TP_FRAC_GRID = [1.0, 1.5, 2.0, 3.0]
SL_BUF_GRID  = [0.0, 0.25, 0.5, 1.0]

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
_PRP  = "#ab47bc"; _TEA  = "#26a69a"; _PNK  = "#ec407a"

MODEL_COLORS = {
    "IFVG": "#ff7043",
    "MTF":  "#26c6da",
    "LTF":  "#ab47bc",
}

SEP  = "─" * 68
SEP2 = "═" * 68


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("Advanced ICT Strategies — BTCUSDT 15M | 2020-2026")
print(SEP2)
print("\n[DATA] Caricamento …")

raw    = fetch_extended_data(start_year=START_YEAR, start_month=1,
                              fetch_15m=True, fetch_1m=False, fetch_flow=False)
df_1h  = add_indicators(raw["1H"])
df_15m = add_indicators(raw["15M"])

print(f"  1H : {len(df_1h):,} bar  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
print(f"  15M: {len(df_15m):,} bar")

# ── 15M arrays
IDX    = df_15m.index
HI     = df_15m["high"].values
LO     = df_15m["low"].values
CL     = df_15m["close"].values
OP     = df_15m["open"].values
N      = len(df_15m)
yr_arr = np.array([t.year for t in IDX], dtype=int)

prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
atr_1h_map = df_1h["atr_14"].clip(lower=1.0).to_dict()
fallback   = df_15m["atr_14"].clip(lower=1.0).values
ATR_1H     = np.array([atr_1h_map.get(t, np.nan) for t in prev_1h], dtype=float)
ATR_1H     = np.where(np.isnan(ATR_1H), fallback, ATR_1H)

# ── 4H resampled (for MTF bias)
df_4h = df_1h.resample("4h").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna()
ema4h     = df_4h["close"].ewm(span=20, adjust=False).mean()
bias4h    = (df_4h["close"] >= ema4h)          # True = bullish
bias4h_d  = bias4h.to_dict()

def _4h_bias(ts15: pd.Timestamp):
    """Return 4H EMA bias for the completed 4H bar ending before ts15."""
    prev_4h = ts15.floor("4h") - pd.Timedelta("4h")
    for delta in (0, 4, 8):
        key = prev_4h - pd.Timedelta(hours=delta)
        if key in bias4h_d:
            return bias4h_d[key]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Generic backtest engine
# Events carry: entry_i, entry_px, direction, ref_lo, ref_hi, ref_size,
#               atr_1h, year, ts
# Optional "sl_base": overrides FVG boundary for structural SL
#   long:  sl_px = sl_base - sl_buf * atr   (sl_base = structural low)
#   short: sl_px = sl_base + sl_buf * atr   (sl_base = structural high)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_ts: pd.Timestamp; exit_ts: pd.Timestamp
    direction: int; entry_price: float; exit_price: float
    net_pnl: float; gross_pnl: float; total_fees: float
    exit_reason: str; year: int; window_id: int


def _sl_px(ev, sl_buf):
    d   = 1 if ev["direction"] == "long" else -1
    atr = ev["atr_1h"]
    if "sl_base" in ev:
        return ev["sl_base"] - sl_buf * atr if d == 1 else ev["sl_base"] + sl_buf * atr
    return ev["ref_lo"] - sl_buf * atr if d == 1 else ev["ref_hi"] + sl_buf * atr


def run_backtest(events, tp_frac, sl_buf, initial_capital=INIT_CAP, window_id=0):
    equity = float(initial_capital)
    trades = []
    for ev in events:
        if equity < initial_capital * 0.005:
            break
        entry    = ev["entry_px"]
        ref_size = max(ev["ref_size"], 1e-6)
        atr      = ev["atr_1h"]
        d        = 1 if ev["direction"] == "long" else -1

        tp_px = entry + d * tp_frac * ref_size
        sl_p  = _sl_px(ev, sl_buf)

        sl_dist = abs(entry - sl_p)
        tp_dist = abs(tp_px - entry)
        if sl_dist <= 0 or tp_dist <= 0:
            continue
        if d == 1 and sl_p >= entry:
            continue
        if d == -1 and sl_p <= entry:
            continue

        sl_dist  = max(sl_dist, atr * MIN_SL_ATR)
        at_risk  = equity * RISK_PCT
        qty      = at_risk / sl_dist
        qty      = min(qty, equity * MAX_LEV / entry)
        notional = qty * entry
        e_fee    = notional * FEE

        ei    = ev["entry_i"]
        start = ei + 1
        hit_tp = hit_sl = False
        exit_bar = min(start + MAX_HOLD, N - 1)
        exit_px  = float(CL[exit_bar])

        for k in range(start, min(start + MAX_HOLD, N)):
            bh, bl = float(HI[k]), float(LO[k])
            if d == 1:
                if bl <= sl_p:  hit_sl = True; exit_px = sl_p;  exit_bar = k; break
                if bh >= tp_px: hit_tp = True; exit_px = tp_px; exit_bar = k; break
            else:
                if bh >= sl_p:  hit_sl = True; exit_px = sl_p;  exit_bar = k; break
                if bl <= tp_px: hit_tp = True; exit_px = tp_px; exit_bar = k; break

        gross = d * qty * (exit_px - entry)
        x_fee = qty * exit_px * FEE
        net   = gross - e_fee - x_fee
        equity += net
        trades.append(Trade(
            entry_ts=IDX[ei], exit_ts=IDX[exit_bar],
            direction=d, entry_price=entry, exit_price=exit_px,
            gross_pnl=gross, total_fees=e_fee + x_fee, net_pnl=net,
            exit_reason="tp" if hit_tp else ("sl" if hit_sl else "time"),
            year=ev["year"], window_id=window_id,
        ))

    if not trades:
        return trades, pd.Series(dtype=float)
    eq = np.empty(len(trades) + 1); eq[0] = initial_capital
    for k, t in enumerate(trades): eq[k + 1] = eq[k] + t.net_pnl
    return trades, pd.Series(eq[1:], index=pd.DatetimeIndex([t.exit_ts for t in trades]))


# ─────────────────────────────────────────────────────────────────────────────
# IS scan
# ─────────────────────────────────────────────────────────────────────────────
def is_scan(events):
    paths = [(HI[ev["entry_i"] + 1: ev["entry_i"] + 1 + MAX_HOLD],
              LO[ev["entry_i"] + 1: ev["entry_i"] + 1 + MAX_HOLD]) for ev in events]

    results = []
    for tp_frac, sl_buf in product(TP_FRAC_GRID, SL_BUF_GRID):
        wins = losses = n_valid = 0
        sum_tp_pct = sum_sl_pct = 0.0
        for ev, (ph, pl) in zip(events, paths):
            entry    = ev["entry_px"]
            ref_size = max(ev["ref_size"], 1e-6)
            atr      = ev["atr_1h"]
            d        = 1 if ev["direction"] == "long" else -1
            tp_px    = entry + d * tp_frac * ref_size
            sl_p     = _sl_px(ev, sl_buf)
            sl_dist  = abs(entry - sl_p)
            tp_dist  = abs(tp_px - entry)
            if sl_dist <= 0 or tp_dist <= 0: continue
            if d == 1 and sl_p >= entry:  continue
            if d == -1 and sl_p <= entry: continue
            n_valid += 1
            sum_tp_pct += tp_dist / entry * 100
            sum_sl_pct += sl_dist / entry * 100
            hit_tp = hit_sl = False
            for h, l in zip(ph, pl):
                if d == 1:
                    if l <= sl_p:  hit_sl = True; break
                    if h >= tp_px: hit_tp = True; break
                else:
                    if h >= sl_p:  hit_sl = True; break
                    if l <= tp_px: hit_tp = True; break
            if hit_tp: wins += 1
            elif hit_sl: losses += 1

        if n_valid < 10: continue
        wr  = wins / n_valid * 100
        rr  = (sum_tp_pct / n_valid) / (sum_sl_pct / n_valid) if sum_sl_pct > 0 else 0
        be  = 1 / (1 + rr) * 100 if rr > 0 else 50.0
        avg_tp = sum_tp_pct / n_valid
        avg_sl = sum_sl_pct / n_valid
        exp    = wr / 100 * avg_tp - (1 - wr / 100) * avg_sl
        p      = st.binomtest(int(round(wr / 100 * n_valid)), n_valid,
                              be / 100, alternative="greater").pvalue
        results.append(dict(tp_frac=tp_frac, sl_buf=sl_buf, n=n_valid,
                            wr=round(wr, 2), rr=round(rr, 2), be=round(be, 2),
                            exp=round(exp, 5), p_val=round(p, 4)))

    df = pd.DataFrame(results)
    if df.empty:
        return df, dict(tp_frac=2.0, sl_buf=0.5, n=0, wr=0, rr=0, be=50, exp=0, p_val=1)
    best = df.sort_values("exp", ascending=False).iloc[0].to_dict()
    return df, best


# ─────────────────────────────────────────────────────────────────────────────
# IC
# ─────────────────────────────────────────────────────────────────────────────
def compute_ic(events):
    sigs, fwds = [], []
    for ev in events:
        ei  = ev["entry_i"]
        end = min(ei + IC_HORIZON, N - 1)
        fwd = (CL[end] - ev["entry_px"]) / ev["entry_px"] * 100
        sig = 1.0 if ev["direction"] == "long" else -1.0
        sigs.append(sig); fwds.append(sig * fwd)
    if len(sigs) < 10: return 0.0, 1.0
    ic, p = st.spearmanr(sigs, fwds)
    return float(ic), float(p)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward
# ─────────────────────────────────────────────────────────────────────────────
def _kpis(equity, init_cap=INIT_CAP):
    if equity.empty:
        return dict(total_return=0, calmar=0, sharpe=0, max_dd=0)
    full  = pd.concat([pd.Series([init_cap],
                      index=[equity.index[0] - pd.Timedelta("1s")]), equity])
    dd    = (full / full.cummax() - 1).min()
    ret   = full.iloc[-1] / init_cap - 1
    rets  = full.pct_change().dropna()
    vol   = rets.std() * np.sqrt(365 * 24)
    ann   = rets.mean() * 365 * 24
    sharpe = ann / vol if vol > 0 else 0.0
    calmar = ret / abs(dd) if dd < 0 else 0.0
    return dict(total_return=ret, calmar=calmar, sharpe=sharpe, max_dd=dd)


def run_wf(events, tp_frac, sl_buf):
    from dateutil.relativedelta import relativedelta
    start, end = IDX[0], IDX[-1]
    wins, cur = [], start
    while True:
        tr_e = cur + relativedelta(months=WF_TRAIN_M)
        oo_e = tr_e + relativedelta(months=WF_OOS_M)
        if oo_e > end: break
        wins.append((cur, tr_e, oo_e))
        cur = cur + relativedelta(months=WF_STEP_M)

    wr_list, all_oos = [], []
    for wid, (tr_s, tr_e, oo_e) in enumerate(wins):
        is_evs = [e for e in events if tr_s <= e["ts"] < tr_e]
        oo_evs = [e for e in events if tr_e <= e["ts"] < oo_e]
        if len(is_evs) < 5 or len(oo_evs) < 3: continue
        oo_t, oo_eq = run_backtest(oo_evs, tp_frac, sl_buf, window_id=wid)
        if not oo_t: continue
        kp = _kpis(oo_eq)
        wr_list.append(kp["total_return"])
        all_oos.extend(oo_evs)

    oos_sorted = sorted(all_oos, key=lambda e: e["ts"])
    oos_trades, oos_equity = run_backtest(oos_sorted, tp_frac, sl_buf)
    return oos_trades, oos_equity, wr_list, len(wins)


# ─────────────────────────────────────────────────────────────────────────────
# ── STRATEGY 1: IFVG (Inverse Fair Value Gap) ────────────────────────────────
#
# Bullish FVG violated downward → IFVG resistance → SHORT on retest
#   · FVG: HI[i-1] < LO[i+1]
#   · Violation: a candle body CLOSES below HI[i-1] (fbot) within lookback bars
#   · Entry SHORT: first retest where HI[j] ≥ fbot (price re-enters zone from below)
#   · sl_base: highest HI during the violation move (SL above this)
#
# Bearish FVG violated upward → IFVG support → LONG on retest
#   · FVG: LO[i-1] > HI[i+1]
#   · Violation: a candle body CLOSES above LO[i-1] (ftop) within lookback bars
#   · Entry LONG: first retest where LO[j] ≤ ftop (price re-enters zone from above)
#   · sl_base: lowest LO during the violation move (SL below this)
# ─────────────────────────────────────────────────────────────────────────────
def collect_ifvg_events(min_fvg_atr_frac=0.05, viol_lookback=8, max_age=16):
    events = []
    for i in range(2, N - viol_lookback - max_age - MAX_HOLD - 5):
        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0:
            continue

        # ── Bullish FVG violated downward → IFVG short ──────────────────────
        if HI[i - 1] < LO[i + 1]:
            fbot = HI[i - 1]; ftop = LO[i + 1]; fsz = ftop - fbot
            if fsz < min_fvg_atr_frac * atr:
                continue

            viol_idx = -1
            viol_hi  = ftop   # track highest high of violation move for SL
            for k in range(i + 2, min(i + 2 + viol_lookback, N)):
                viol_hi = max(viol_hi, HI[k])
                # Invalidate: price closes far above FVG (strong continuation)
                if CL[k] > ftop + fsz:
                    break
                if CL[k] < fbot:    # body closed BELOW bottom of bullish FVG
                    viol_idx = k
                    break

            if viol_idx < 0:
                continue

            for j in range(viol_idx + 1, min(viol_idx + 1 + max_age, N - MAX_HOLD)):
                if HI[j] >= fbot:   # price re-enters violated FVG from below
                    entry_px = min(CL[j], ftop)
                    events.append(dict(
                        direction="short", entry_i=j, entry_px=entry_px,
                        ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                        atr_1h=ATR_1H[j],
                        sl_base=viol_hi,   # SL above highest point of violation move
                        year=yr_arr[i], ts=IDX[i],
                    ))
                    break

        # ── Bearish FVG violated upward → IFVG long ─────────────────────────
        if LO[i - 1] > HI[i + 1]:
            ftop = LO[i - 1]; fbot = HI[i + 1]; fsz = ftop - fbot
            if fsz < min_fvg_atr_frac * atr:
                continue

            viol_idx = -1
            viol_lo  = fbot
            for k in range(i + 2, min(i + 2 + viol_lookback, N)):
                viol_lo = min(viol_lo, LO[k])
                if CL[k] < fbot - fsz:
                    break
                if CL[k] > ftop:    # body closed ABOVE top of bearish FVG
                    viol_idx = k
                    break

            if viol_idx < 0:
                continue

            for j in range(viol_idx + 1, min(viol_idx + 1 + max_age, N - MAX_HOLD)):
                if LO[j] <= ftop:   # price re-enters violated FVG from above
                    entry_px = max(CL[j], fbot)
                    events.append(dict(
                        direction="long", entry_i=j, entry_px=entry_px,
                        ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                        atr_1h=ATR_1H[j],
                        sl_base=viol_lo,   # SL below lowest point of violation move
                        year=yr_arr[i], ts=IDX[i],
                    ))
                    break

    return events


# ─────────────────────────────────────────────────────────────────────────────
# ── STRATEGY 2: MTF Combo (4H bias → 15M CHoCH → 15M FVG entry) ─────────────
#
# 4H bias: close ≥ EMA(20) on 4H bars → bullish; else bearish
# CHoCH on 15M (in direction of 4H bias):
#   · Bullish: recent bars pulled back ≥ pullback_atr × ATR_1H; current bar
#     closes ABOVE the 3-bar swing high (break of structure upward)
#   · Bearish: symmetric
# Entry: first 15M FVG in bias direction within max_fvg_age bars of CHoCH
# sl_base: lowest LO (bullish) or highest HI (bearish) of the CHoCH pullback
# ─────────────────────────────────────────────────────────────────────────────
def collect_mtf_events(swing_bars=3, pullback_atr=0.5, max_fvg_age=12, max_age=8):
    events = []
    for i in range(swing_bars + 20, N - max_fvg_age - max_age - MAX_HOLD - 5):
        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0:
            continue

        bias = _4h_bias(IDX[i])
        if bias is None:
            continue

        recent_hi = np.max(HI[i - swing_bars:i])
        recent_lo = np.min(LO[i - swing_bars:i])

        if bias:    # ── Bullish 4H bias: look for bullish CHoCH ───────────
            pullback = CL[i - swing_bars] - CL[i - 1]
            if pullback < pullback_atr * atr:
                continue
            if CL[i] <= recent_hi:
                continue    # no break of structure

            sl_extreme = np.min(LO[i - swing_bars:i + 1])   # structural low

            for m in range(i + 1, min(i + max_fvg_age, N - MAX_HOLD - max_age - 3)):
                if HI[m - 1] < LO[m + 1]:   # bullish FVG
                    fbot = HI[m - 1]; ftop = LO[m + 1]; fsz = ftop - fbot
                    if fsz < 0.05 * ATR_1H[m]:
                        continue
                    for j in range(m + 2, min(m + 2 + max_age, N - MAX_HOLD)):
                        if LO[j] <= ftop:    # retracement into FVG
                            entry_px = max(CL[j], fbot)
                            events.append(dict(
                                direction="long", entry_i=j, entry_px=entry_px,
                                ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                                atr_1h=ATR_1H[j],
                                sl_base=sl_extreme,
                                year=yr_arr[i], ts=IDX[i],
                            ))
                            break
                    break   # only first FVG after CHoCH

        else:       # ── Bearish 4H bias: look for bearish CHoCH ──────────
            pullback = CL[i - 1] - CL[i - swing_bars]
            if pullback < pullback_atr * atr:
                continue
            if CL[i] >= recent_lo:
                continue

            sl_extreme = np.max(HI[i - swing_bars:i + 1])

            for m in range(i + 1, min(i + max_fvg_age, N - MAX_HOLD - max_age - 3)):
                if LO[m - 1] > HI[m + 1]:   # bearish FVG
                    ftop = LO[m - 1]; fbot = HI[m + 1]; fsz = ftop - fbot
                    if fsz < 0.05 * ATR_1H[m]:
                        continue
                    for j in range(m + 2, min(m + 2 + max_age, N - MAX_HOLD)):
                        if HI[j] >= fbot:    # retracement into FVG
                            entry_px = min(CL[j], ftop)
                            events.append(dict(
                                direction="short", entry_i=j, entry_px=entry_px,
                                ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                                atr_1h=ATR_1H[j],
                                sl_base=sl_extreme,
                                year=yr_arr[i], ts=IDX[i],
                            ))
                            break
                    break

    return events


# ─────────────────────────────────────────────────────────────────────────────
# ── STRATEGY 3: LTF Logic Flow (Grab → Shift → Entry → Confirm) ──────────────
#
# TS  (Turtle Soup): wick sweeps N-bar swing H/L but candle CLOSES back inside
#   · Bearish sweep: HI[i] > swing_hi AND CL[i] < swing_hi
#   · Bullish sweep: LO[i] < swing_lo AND CL[i] > swing_lo
# CISD: first candle after sweep with body ≥ cisd_frac × ATR in reversal direction
# IFVG: first FVG in reversal direction within fvg_lookback bars of CISD
# Entry: retracement into IFVG
# sl_base: sweep extreme (the wick high for short, wick low for long)
# ─────────────────────────────────────────────────────────────────────────────
def collect_ltf_events(sweep_lookback=16, min_wick_atr=0.10,
                       cisd_frac=0.25, fvg_lookback=8, max_age=16):
    events = []
    for i in range(sweep_lookback + 2, N - fvg_lookback - max_age - MAX_HOLD - 8):
        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0:
            continue

        swing_hi = np.max(HI[i - sweep_lookback:i])
        swing_lo = np.min(LO[i - sweep_lookback:i])

        # ── Bearish sweep → expect move DOWN ─────────────────────────────────
        if HI[i] > swing_hi and CL[i] < swing_hi:
            if HI[i] - swing_hi < min_wick_atr * atr:
                continue
            ts_extreme = HI[i]   # SL above the sweep high

            # CISD: first bearish candle after sweep
            cisd_idx = -1
            for k in range(i + 1, min(i + 5, N)):
                if (OP[k] - CL[k]) >= cisd_frac * atr:
                    cisd_idx = k; break
            if cisd_idx < 0:
                continue

            # First bearish FVG after CISD
            for m in range(cisd_idx + 1, min(cisd_idx + 1 + fvg_lookback, N - max_age - MAX_HOLD - 3)):
                if LO[m - 1] > HI[m + 1]:
                    ftop = LO[m - 1]; fbot = HI[m + 1]; fsz = ftop - fbot
                    if fsz < 0.05 * ATR_1H[m]:
                        continue
                    for j in range(m + 2, min(m + 2 + max_age, N - MAX_HOLD)):
                        if HI[j] >= fbot:    # retracement up into bearish FVG
                            entry_px = min(CL[j], ftop)
                            events.append(dict(
                                direction="short", entry_i=j, entry_px=entry_px,
                                ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                                atr_1h=ATR_1H[j],
                                sl_base=ts_extreme,
                                year=yr_arr[i], ts=IDX[i],
                            ))
                            break
                    break

        # ── Bullish sweep → expect move UP ───────────────────────────────────
        elif LO[i] < swing_lo and CL[i] > swing_lo:
            if swing_lo - LO[i] < min_wick_atr * atr:
                continue
            ts_extreme = LO[i]   # SL below the sweep low

            cisd_idx = -1
            for k in range(i + 1, min(i + 5, N)):
                if (CL[k] - OP[k]) >= cisd_frac * atr:
                    cisd_idx = k; break
            if cisd_idx < 0:
                continue

            for m in range(cisd_idx + 1, min(cisd_idx + 1 + fvg_lookback, N - max_age - MAX_HOLD - 3)):
                if HI[m - 1] < LO[m + 1]:
                    fbot = HI[m - 1]; ftop = LO[m + 1]; fsz = ftop - fbot
                    if fsz < 0.05 * ATR_1H[m]:
                        continue
                    for j in range(m + 2, min(m + 2 + max_age, N - MAX_HOLD)):
                        if LO[j] <= ftop:    # retracement down into bullish FVG
                            entry_px = max(CL[j], fbot)
                            events.append(dict(
                                direction="long", entry_i=j, entry_px=entry_px,
                                ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                                atr_1h=ATR_1H[j],
                                sl_base=ts_extreme,
                                year=yr_arr[i], ts=IDX[i],
                            ))
                            break
                    break

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers (same dark theme as ICT suite)
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _plot_equity(oos_equity, label, color):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_CARD); ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values(): sp.set_color(_GRID)
    ax1.plot(oos_equity.index, oos_equity.values / INIT_CAP, color=color, lw=1.5)
    ax1.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (norm.)", color=_TEXT)
    rm = oos_equity.cummax()
    dd = (oos_equity / rm - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, color=_RED, alpha=0.4)
    ax2.set_ylabel("DD %", color=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle(f"OOS Equity — {label}", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_wf_bars(wr_list, label, color):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    clrs = [_GRN if r > 0 else _RED for r in wr_list]
    ax.bar(range(len(wr_list)), [r * 100 for r in wr_list], color=clrs, alpha=0.8)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_title(f"WF per-window OOS returns — {label}", color=_TEXT)
    ax.set_ylabel("Ret %", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_mc_hist(mc_res, label):
    fin = mc_res["total_return"] * 100
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    ax.hist(fin, bins=60, color=_ACC, alpha=0.7, edgecolor="none")
    ax.axvline(0, color=_RED, lw=1.5, ls="--")
    ax.axvline(float(np.median(fin)), color=_YEL, lw=1.5,
               label=f"Median {np.median(fin):.1f}%")
    ax.set_xlabel("Total Return %", color=_TEXT)
    ax.set_title(f"Monte Carlo — {label}", color=_TEXT)
    ax.legend(facecolor=_CARD, labelcolor=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _card(title, body, accent=_ACC):
    return (f'<div style="background:{_CARD};border:1px solid {_GRID};'
            f'border-radius:8px;padding:16px 20px;margin-bottom:14px">'
            f'<h3 style="color:{accent};margin:0 0 10px">{title}</h3>{body}</div>')


def _kv(label, value):
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:3px 0">'
            f'<span style="color:#9e9e9e">{label}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{value}</span></div>')


def _style(v, fmt=".1%", good=0):
    c = _GRN if v > good else _RED
    return f'<span style="color:{c}">{v:{fmt}}</span>'


def _imgt(b64):
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:6px;margin-bottom:10px">')


def _table(headers, rows):
    th = "".join(f'<th style="padding:5px 8px;text-align:right;color:{_ACC};'
                 f'border-bottom:1px solid {_GRID}">{h}</th>' for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td style="padding:4px 8px;text-align:right;color:{_TEXT}">{c}</td>'
                         for c in r) + "</tr>" for r in rows)
    return (f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


# ═════════════════════════════════════════════════════════════════════════════
# Run pipeline for each strategy
# ═════════════════════════════════════════════════════════════════════════════
MODELS = [
    ("IFVG", "IFVG — Inverse Fair Value Gap",              collect_ifvg_events),
    ("MTF",  "MTF Combo — 4H bias → 15M CHoCH → FVG",     collect_mtf_events),
    ("LTF",  "LTF Logic Flow — TS → CISD → IFVG → DOL",   collect_ltf_events),
]

results = {}

for key, name, collector in MODELS:
    col = MODEL_COLORS[key]
    print(f"\n{SEP}")
    print(f"  [{key}] {name}")
    print(SEP)

    print("  Raccolta eventi …")
    events = collector()
    n_l = sum(1 for e in events if e["direction"] == "long")
    n_s = sum(1 for e in events if e["direction"] == "short")
    print(f"  Totale: {len(events):,}  (long={n_l}, short={n_s})")

    if len(events) < 20:
        print("  Troppo pochi eventi — skip")
        results[key] = {"name": name, "key": key, "empty": True}
        continue

    ic, ic_p = compute_ic(events)
    print(f"  IC={ic:.4f}  p={ic_p:.4f}  {'✓' if ic_p < 0.05 else '✗'}")

    print("  IS scan …")
    df_scan, best = is_scan(events)
    tp_f, sl_b = best["tp_frac"], best["sl_buf"]
    print(f"  Best: tp={tp_f}  sl={sl_b}  WR={best['wr']:.1f}%  "
          f"BE={best['be']:.1f}%  ExpPnL={best['exp']:+.5f}%  p={best['p_val']:.4f}")

    print("  Walk-Forward …")
    oos_trades, oos_equity, wr_list, n_wins = run_wf(events, tp_f, sl_b)
    print(f"  OOS trades={len(oos_trades)}  WF windows={n_wins}")

    if not oos_trades:
        results[key] = {"name": name, "key": key, "empty": True}
        continue

    kp     = _kpis(oos_equity)
    oos_df = pd.DataFrame([t.__dict__ for t in oos_trades]).sort_values("entry_ts")
    wins_n = (oos_df["net_pnl"] > 0).sum()
    wr_oos = wins_n / len(oos_df)
    rr     = best["rr"]
    be_oos = 1 / (1 + rr) * 100 if rr > 0 else 50.0

    binom_p = st.binomtest(int(wins_n), len(oos_df),
                           be_oos / 100, alternative="greater").pvalue
    oos_tmp = oos_df.copy()
    oos_tmp["date"] = pd.to_datetime(oos_tmp["exit_ts"]).dt.normalize()
    daily = oos_tmp.groupby("date")["net_pnl"].sum()
    _, t_p = st.ttest_1samp(daily.values, 0) if len(daily) > 1 else (0, 1)

    print(f"  Return={kp['total_return']:+.1%}  MaxDD={kp['max_dd']:.1%}"
          f"  WR={wr_oos:.1%}  BE≈{be_oos:.1f}%  binom_p={binom_p:.4f}")

    print("  Monte Carlo …")
    mc_res   = run_monte_carlo(oos_df[["net_pnl", "gross_pnl", "total_fees"]], INIT_CAP, N_SIMS)
    p_profit = float((mc_res["total_return"] > 0).mean())
    p_ruin   = float((mc_res["total_return"] < -0.5).mean())
    print(f"  P(profit)={p_profit:.1%}  P(ruin)={p_ruin:.1%}")

    results[key] = dict(
        name=name, key=key, empty=False, color=col,
        events=events, n_long=n_l, n_short=n_s,
        ic=ic, ic_p=ic_p,
        best=best, df_scan=df_scan,
        oos_trades=oos_trades, oos_equity=oos_equity, oos_df=oos_df,
        kp=kp, wr_oos=wr_oos, be_oos=be_oos, binom_p=binom_p, t_p=t_p,
        wr_list=wr_list, n_wins=n_wins,
        mc_res=mc_res, p_profit=p_profit, p_ruin=p_ruin,
    )


# ═════════════════════════════════════════════════════════════════════════════
# HTML Report
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\nGenerazione report HTML …")

# Summary table
sum_rows = []
for key, r in results.items():
    col = MODEL_COLORS[key]
    if r.get("empty"):
        sum_rows.append([f'<span style="color:{col}">{r["name"]}</span>',
                         "—", "—", "—", "—", "—", "—", "—", "—"])
        continue
    kp = r["kp"]
    ok = kp["total_return"] > 0 and r["binom_p"] < 0.05 and r["p_profit"] > 0.5
    verdict = f'<span style="color:{"#66bb6a" if ok else "#ef5350"}">{"✅" if ok else "✗"}</span>'
    sum_rows.append([
        f'<span style="color:{col}">{r["name"]}</span>',
        f'{len(r["events"]):,}',
        f'{r["ic"]:.4f} (p={r["ic_p"]:.3f})',
        f'{r["best"]["tp_frac"]} / {r["best"]["sl_buf"]}',
        f'{r["best"]["wr"]:.1f}% (BE {r["best"]["be"]:.1f}%)',
        f'{len(r["oos_df"])}',
        f'{r["wr_oos"]:.1%}',
        f'{kp["total_return"]:+.1%}  /  {kp["max_dd"]:.1%}',
        verdict,
    ])

summary_tbl = _table(
    ["Strategy", "Events IS", "IC (Spearman)", "TP / SL best",
     "WR IS", "Trades OOS", "WR OOS", "Return / MaxDD OOS", "Verdict"],
    sum_rows,
)

# Per-strategy sections
sections = ""
strategy_descriptions = {
    "IFVG": (
        "Un FVG standard viene violato (il corpo di una candela chiude attraverso la zona): "
        "il FVG bullish violato al ribasso diventa resistenza (short sul retest); "
        "il FVG bearish violato al rialzo diventa supporto (long sul retest). "
        "SL: oltre il punto estremo del movimento che ha violato il FVG."
    ),
    "MTF": (
        "Cascata 4H → 15M: bias direzionale da EMA(20) su 4H; "
        "conferma strutturale via CHoCH/BOS su 15M (rottura del 3-bar swing in direzione bias "
        "dopo un pullback ≥ 0.5× ATR); ingresso nel primo FVG 15M formatosi dopo il CHoCH. "
        "SL: sotto il minimo strutturale del pullback (long) / sopra il massimo (short)."
    ),
    "LTF": (
        "Sequenza algoritmica completa: TS (Turtle Soup) = wick oltre il N-bar swing H/L "
        "con chiusura all'interno → CISD (prima candela forte nella nuova direzione, "
        "corpo ≥ 0.25× ATR) → IFVG (primo FVG nella nuova direzione dopo CISD) → "
        "ingresso sul retracement nel FVG. SL: oltre l'estremo dello spike TS."
    ),
}

for key, r in results.items():
    col  = MODEL_COLORS[key]
    name = r["name"]
    desc = strategy_descriptions.get(key, "")

    if r.get("empty"):
        sections += f'<h2 style="color:{col}">{name} — nessun evento sufficiente</h2>'
        continue

    kp  = r["kp"]
    ok  = kp["total_return"] > 0 and r["binom_p"] < 0.05 and r["p_profit"] > 0.5
    vc  = _GRN if ok else _RED
    vt  = "✅ VALIDATO" if ok else "✗ NON VALIDATO"

    img_eq = _plot_equity(r["oos_equity"], name, col)
    img_wf = _plot_wf_bars(r["wr_list"],   name, col)
    img_mc = _plot_mc_hist(r["mc_res"],    name)

    mc_fin = r["mc_res"]["total_return"] * 100
    mc_dd  = r["mc_res"]["max_drawdown"] * 100

    info_body = "".join([
        _kv("Logica", desc[:120] + "…"),
        _kv("Events IS", f'{len(r["events"]):,}  (L={r["n_long"]}, S={r["n_short"]})'),
        _kv("IC (Spearman)", f'{r["ic"]:.4f}  p={r["ic_p"]:.4f}'),
        _kv("Best tp/sl",  f'tp={r["best"]["tp_frac"]}  sl_buf={r["best"]["sl_buf"]}'),
        _kv("IS WR",       f'{r["best"]["wr"]:.1f}%  BE={r["best"]["be"]:.1f}%  '
                           f'ExpPnL={r["best"]["exp"]:+.5f}%  p={r["best"]["p_val"]:.4f}'),
        _kv("OOS trades",  str(len(r["oos_df"]))),
        _kv("WR OOS",      _style(r["wr_oos"], ".1%")),
        _kv("Total Return", _style(kp["total_return"], ".1%")),
        _kv("Max DD",      _style(kp["max_dd"], ".1%")),
        _kv("Binom p",     f'{r["binom_p"]:.4f}'),
        _kv("MC P(profit)", _style(r["p_profit"], ".1%", good=0.5)),
        _kv("MC P(ruin)",  f'{r["p_ruin"]:.1%}'),
        _kv("MC median",   f'{float(np.median(mc_fin)):.1f}%'),
        _kv("MC p5–p95",   f'{float(np.percentile(mc_fin,5)):.1f}% → '
                           f'{float(np.percentile(mc_fin,95)):.1f}%'),
        _kv("WF windows",  f'{r["n_wins"]} tot / {sum(1 for x in r["wr_list"] if x>0)} profit'),
        _kv("Verdict",     f'<b style="color:{vc}">{vt}</b>'),
    ])

    sections += f"""
<h2 style="color:{col};margin-top:40px">{name}</h2>
<p style="color:#9e9e9e;font-size:13px">{desc}</p>
{_card("Metriche", info_body, col)}
{_imgt(img_eq)}
{_imgt(img_wf)}
{_imgt(img_mc)}
"""

# Comparison equity overlay
fig_all, ax_all = plt.subplots(figsize=(12, 5))
fig_all.patch.set_facecolor(_BG); ax_all.set_facecolor(_CARD)
ax_all.tick_params(colors=_TEXT)
for sp in ax_all.spines.values(): sp.set_color(_GRID)
for key, r in results.items():
    if not r.get("empty") and not r["oos_equity"].empty:
        eq = r["oos_equity"]
        ax_all.plot(eq.index, eq.values / INIT_CAP, color=r["color"],
                    lw=1.5, label=r["key"])
ax_all.axhline(1.0, color=_GRID, lw=0.8, ls="--")
ax_all.set_title("OOS Equity — confronto strategie", color=_TEXT)
ax_all.set_ylabel("Equity (norm.)", color=_TEXT)
ax_all.legend(facecolor=_CARD, labelcolor=_TEXT, loc="upper left")
fig_all.tight_layout()
img_all = _fig_to_b64(fig_all)

html = f"""<!DOCTYPE html><html lang="it"><head>
<meta charset="UTF-8">
<title>Advanced ICT Strategies — BTCUSDT 15M</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{_BG};color:{_TEXT};font-family:monospace;padding:24px}}
  h1{{color:{_ACC};margin-bottom:6px}}
  h2{{margin-top:32px;margin-bottom:8px}}
  p{{margin-bottom:8px;line-height:1.5}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:16px}}
</style></head><body>
<h1>Advanced ICT Strategies — BTCUSDT 15M | 2020-2026</h1>
<p style="color:#9e9e9e">Validation pipeline: IC → IS scan (4×4 grid) →
Walk-Forward 35 windows (6m/2m) → Monte Carlo 5000 sims</p>
<p style="color:#9e9e9e">Dataset: {len(df_15m):,} barre · fee 4bps/side ·
risk 1%/trade · ATR floor 0.5× · max leva 5×</p>

<h2 style="color:{_ACC};margin-top:24px">Riepilogo</h2>
{summary_tbl}

<h2 style="color:{_ACC};margin-top:24px">Equity OOS — Confronto</h2>
{_imgt(img_all)}

{sections}

<p style="color:#555;font-size:11px;margin-top:40px">
  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC — BTCUSDT 15M 2020-2026
</p>
</body></html>"""

out_path = Path("reports/report_advanced_ict.html")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"\n✅ Report salvato: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
print(SEP2)
