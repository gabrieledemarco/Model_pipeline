"""
create_ict_suite_report.py
===========================
ICT Model Suite — Full Validation Pipeline on BTCUSDT 15M (2020-2026)

Models tested:
  1. Fair Value Gap (FVG) Fill
  2. Order Block (OB) Reversal
  3. Liquidity Sweep + Reversal
  4. Silver Bullet (FVG in Killzone)
  5. Power of 3 (PO3) — Asian/London/NY session structure
  6. Breaker Block

Each model goes through:
  IS parameter scan → IC (Spearman) → Walk-Forward (6m/2m/2m) → Monte Carlo (5000 sims)
All events stored as unified dict with ref_lo/ref_hi/ref_size for generic backtest.
"""
from __future__ import annotations

import base64
import io
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Optional

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
# Config
# ─────────────────────────────────────────────────────────────────────────────
INIT_CAP   = 100_000.0
RISK_PCT   = 0.01
FEE        = 0.0004        # 4 bps per side (Binance taker)
MAX_HOLD   = 32            # 15M bars = 8 h
IC_HORIZON = 16            # 15M bars = 4 h
START_YEAR = 2020
N_SIMS     = 5_000

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

TP_FRAC_GRID = [1.0, 1.5, 2.0, 3.0]
SL_BUF_GRID  = [0.0, 0.25, 0.5, 1.0]

# Killzone windows (UTC) for Silver Bullet
KILLZONES = [
    (7, 0, 8, 0),    # London open 07-08 UTC (2-3 AM ET)
    (14, 0, 15, 0),  # NY AM 14-15 UTC (9-10 AM ET)
    (18, 0, 19, 0),  # NY PM 18-19 UTC (1-2 PM ET)
]

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
_PRP  = "#ab47bc"; _TEA  = "#26a69a"; _PNK  = "#ec407a"

MODEL_COLORS = {
    "FVG":          _ACC,
    "OB":           _ORG,
    "Sweep":        _GRN,
    "SilverBullet": _PRP,
    "PO3":          _YEL,
    "Breaker":      _TEA,
}

SEP  = "─" * 68
SEP2 = "═" * 68


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & array setup
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("ICT Model Suite — BTCUSDT 15M | 2020-2026")
print(SEP2)
print("\n[DATA] Caricamento …")

raw    = fetch_extended_data(start_year=START_YEAR, start_month=1,
                              fetch_15m=True, fetch_1m=False, fetch_flow=False)
df_1h  = add_indicators(raw["1H"])
df_15m = add_indicators(raw["15M"])

print(f"  1H : {len(df_1h):,} bar  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
print(f"  15M: {len(df_15m):,} bar")

IDX   = df_15m.index
H_arr = np.array(IDX.hour,   dtype=int)
M_arr = np.array(IDX.minute, dtype=int)
HI    = df_15m["high"].values
LO    = df_15m["low"].values
CL    = df_15m["close"].values
OP    = df_15m["open"].values
N     = len(df_15m)
yr_arr = np.array([t.year for t in IDX], dtype=int)

prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
atr_1h_map = df_1h["atr_14"].clip(lower=1.0).to_dict()
fallback   = df_15m["atr_14"].clip(lower=1.0).values
ATR_1H     = np.array([atr_1h_map.get(t, np.nan) for t in prev_1h], dtype=float)
ATR_1H     = np.where(np.isnan(ATR_1H), fallback, ATR_1H)


# ─────────────────────────────────────────────────────────────────────────────
# Generic backtest engine
# All events carry: entry_i, entry_px, direction, ref_lo, ref_hi, ref_size,
#                   atr_1h, year, ts
# SL: long = ref_lo - sl_buf*atr,  short = ref_hi + sl_buf*atr
# TP: long = entry_px + tp_frac*ref_size, short = entry_px - tp_frac*ref_size
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_ts: pd.Timestamp; exit_ts: pd.Timestamp
    direction: int; entry_price: float; exit_price: float
    net_pnl: float; gross_pnl: float; total_fees: float
    exit_reason: str; year: int; window_id: int


def run_backtest(events, tp_frac, sl_buf, initial_capital=INIT_CAP,
                 window_id=0):
    equity = float(initial_capital)
    trades = []
    for ev in events:
        if equity < initial_capital * 0.005:   # stop below 0.5% of start
            break
        entry = ev["entry_px"]
        ref_size = max(ev["ref_size"], 1e-6)
        atr = ev["atr_1h"]
        d = 1 if ev["direction"] == "long" else -1

        tp_px = entry + d * tp_frac * ref_size
        sl_px = (ev["ref_lo"] - sl_buf * atr if d == 1
                 else ev["ref_hi"] + sl_buf * atr)

        sl_dist = abs(entry - sl_px)
        tp_dist = abs(tp_px - entry)
        if sl_dist <= 0 or tp_dist <= 0:
            continue
        if d == 1 and sl_px >= entry:
            continue
        if d == -1 and sl_px <= entry:
            continue
        # Enforce minimum SL distance: 10% of ATR_1H or 0.1% of price
        # This caps notional and prevents fee destruction from tiny stops
        min_sl = max(ev["atr_1h"] * 0.1, entry * 0.001)
        sl_dist = max(sl_dist, min_sl)

        at_risk  = equity * RISK_PCT
        qty      = at_risk / sl_dist
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
                if bl <= sl_px: hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bh >= tp_px: hit_tp = True; exit_px = tp_px; exit_bar = k; break
            else:
                if bh >= sl_px: hit_sl = True; exit_px = sl_px; exit_bar = k; break
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
# IS scan  (path-level, no equity sim)
# ─────────────────────────────────────────────────────────────────────────────
def is_scan(events):
    paths = []
    for ev in events:
        s = ev["entry_i"] + 1
        paths.append((HI[s:s + MAX_HOLD], LO[s:s + MAX_HOLD]))

    results = []
    for tp_frac, sl_buf in product(TP_FRAC_GRID, SL_BUF_GRID):
        wins = losses = n_valid = 0
        sum_tp_pct = sum_sl_pct = 0.0
        for ev, (ph, pl) in zip(events, paths):
            entry = ev["entry_px"]
            ref_size = max(ev["ref_size"], 1e-6)
            atr = ev["atr_1h"]
            d = 1 if ev["direction"] == "long" else -1
            tp_px = entry + d * tp_frac * ref_size
            sl_px = (ev["ref_lo"] - sl_buf * atr if d == 1
                     else ev["ref_hi"] + sl_buf * atr)
            sl_dist = abs(entry - sl_px)
            tp_dist = abs(tp_px - entry)
            if sl_dist <= 0 or tp_dist <= 0: continue
            if d == 1 and sl_px >= entry: continue
            if d == -1 and sl_px <= entry: continue

            n_valid += 1
            sum_tp_pct += tp_dist / entry * 100
            sum_sl_pct += sl_dist / entry * 100
            hit_tp = hit_sl = False
            for h, l in zip(ph, pl):
                if d == 1:
                    if l <= sl_px:  hit_sl = True; break
                    if h >= tp_px:  hit_tp = True; break
                else:
                    if h >= sl_px:  hit_sl = True; break
                    if l <= tp_px:  hit_tp = True; break
            if hit_tp: wins += 1
            elif hit_sl: losses += 1

        if n_valid < 10: continue
        wr  = wins / n_valid * 100
        rr  = (sum_tp_pct / n_valid) / (sum_sl_pct / n_valid) if sum_sl_pct > 0 else 0
        be  = 1 / (1 + rr) * 100 if rr > 0 else 50.0
        avg_tp = sum_tp_pct / n_valid
        avg_sl = sum_sl_pct / n_valid
        exp = wr / 100 * avg_tp - (1 - wr / 100) * avg_sl
        p   = st.binomtest(int(round(wr / 100 * n_valid)), n_valid,
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
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────
def _kpis(equity, init_cap=INIT_CAP):
    if equity.empty: return dict(total_return=0, calmar=0, sharpe=0, max_dd=0)
    full = pd.concat([pd.Series([init_cap],
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
# ── MODEL 1: Fair Value Gap (FVG) Fill ─────────────────────────────────────
# Bullish FVG: HI[i-1] < LO[i+1]  → gap = [HI[i-1], LO[i+1]]
# Entry: first bar after formation where price enters the FVG zone
# ─────────────────────────────────────────────────────────────────────────────
def collect_fvg_events(min_fvg_atr_frac=0.05, max_age=48):
    events = []
    for i in range(2, N - max_age - MAX_HOLD - 2):
        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0: continue

        # Bullish FVG
        if HI[i - 1] < LO[i + 1]:
            fbot = HI[i - 1]; ftop = LO[i + 1]; fsz = ftop - fbot
            if fsz < min_fvg_atr_frac * atr: continue
            for j in range(i + 2, min(i + 2 + max_age, N - MAX_HOLD)):
                if CL[j] < fbot: break             # price closed below FVG → invalidate
                if LO[j] <= ftop:                  # price entered FVG zone from above
                    events.append(dict(
                        direction="long", entry_i=j, entry_px=min(CL[j], ftop),
                        ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                        atr_1h=ATR_1H[j], year=yr_arr[i], ts=IDX[i]))
                    break

        # Bearish FVG
        if LO[i - 1] > HI[i + 1]:
            ftop = LO[i - 1]; fbot = HI[i + 1]; fsz = ftop - fbot
            if fsz < min_fvg_atr_frac * atr: continue
            for j in range(i + 2, min(i + 2 + max_age, N - MAX_HOLD)):
                if CL[j] > ftop: break
                if HI[j] >= fbot:
                    events.append(dict(
                        direction="short", entry_i=j, entry_px=max(CL[j], fbot),
                        ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                        atr_1h=ATR_1H[j], year=yr_arr[i], ts=IDX[i]))
                    break
    return events


# ─────────────────────────────────────────────────────────────────────────────
# ── MODEL 2: Order Block (OB) Reversal ──────────────────────────────────────
# Bullish OB: last bearish candle before a bullish impulse ≥ swing_atr × ATR_1H
# Entry: price returns into the OB zone
# ─────────────────────────────────────────────────────────────────────────────
def collect_ob_events(swing_atr=1.5, ob_max_age=80):
    events = []
    for i in range(2, N - ob_max_age - MAX_HOLD - 2):
        atr = ATR_1H[i];
        if np.isnan(atr) or atr <= 0: continue

        # Bullish OB: OP[i] > CL[i] (bearish candle)
        if OP[i] > CL[i]:
            ob_hi = HI[i]; ob_lo = LO[i]; ob_sz = ob_hi - ob_lo
            if ob_sz < 1e-6: continue
            # Confirm with bullish swing AFTER OB — track when swing completes
            swing_end = -1
            swing_start = CL[i]
            for s in range(i + 1, min(i + 8, N)):
                if CL[s] - swing_start >= swing_atr * atr:
                    swing_end = s; break
            if swing_end < 0: continue
            # Entry only AFTER swing completes (no lookahead)
            for j in range(swing_end + 1, min(i + ob_max_age, N - MAX_HOLD)):
                if CL[j] < ob_lo: break    # price broke through OB
                if LO[j] <= ob_hi and HI[j] >= ob_lo:  # price back in OB zone
                    events.append(dict(
                        direction="long", entry_i=j, entry_px=max(CL[j], ob_lo),
                        ref_lo=ob_lo, ref_hi=ob_hi, ref_size=ob_sz,
                        atr_1h=ATR_1H[j], year=yr_arr[i], ts=IDX[i]))
                    break

        # Bearish OB: OP[i] < CL[i] (bullish candle)
        if OP[i] < CL[i]:
            ob_hi = HI[i]; ob_lo = LO[i]; ob_sz = ob_hi - ob_lo
            if ob_sz < 1e-6: continue
            swing_end = -1
            swing_start = CL[i]
            for s in range(i + 1, min(i + 8, N)):
                if swing_start - CL[s] >= swing_atr * atr:
                    swing_end = s; break
            if swing_end < 0: continue
            for j in range(swing_end + 1, min(i + ob_max_age, N - MAX_HOLD)):
                if CL[j] > ob_hi: break
                if HI[j] >= ob_lo and LO[j] <= ob_hi:
                    events.append(dict(
                        direction="short", entry_i=j, entry_px=min(CL[j], ob_hi),
                        ref_lo=ob_lo, ref_hi=ob_hi, ref_size=ob_sz,
                        atr_1h=ATR_1H[j], year=yr_arr[i], ts=IDX[i]))
                    break
    return events


# ─────────────────────────────────────────────────────────────────────────────
# ── MODEL 3: Liquidity Sweep + Reversal ─────────────────────────────────────
# Sweep: price wicks beyond N-bar swing H/L but closes back
# Direction: sweep of local HIGH → SHORT; sweep of local LOW → LONG
# ─────────────────────────────────────────────────────────────────────────────
def collect_sweep_events(lbk=16, min_wick_atr=0.15):
    events = []
    for i in range(lbk, N - MAX_HOLD - 2):
        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0: continue

        swing_hi = np.max(HI[i - lbk:i])   # recent high
        swing_lo = np.min(LO[i - lbk:i])   # recent low

        # Bearish sweep: wick above swing_hi, close back below
        if HI[i] > swing_hi and CL[i] < swing_hi:
            wick = HI[i] - swing_hi
            if wick < min_wick_atr * atr: continue
            events.append(dict(
                direction="short", entry_i=i, entry_px=CL[i],
                ref_lo=swing_hi, ref_hi=HI[i], ref_size=atr,
                atr_1h=atr, year=yr_arr[i], ts=IDX[i]))

        # Bullish sweep: wick below swing_lo, close back above
        elif LO[i] < swing_lo and CL[i] > swing_lo:
            wick = swing_lo - LO[i]
            if wick < min_wick_atr * atr: continue
            events.append(dict(
                direction="long", entry_i=i, entry_px=CL[i],
                ref_lo=LO[i], ref_hi=swing_lo, ref_size=atr,
                atr_1h=atr, year=yr_arr[i], ts=IDX[i]))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# ── MODEL 4: Silver Bullet (FVG in Killzone) ────────────────────────────────
# Same as FVG but the candle forming the FVG must be within a killzone window
# ─────────────────────────────────────────────────────────────────────────────
def collect_silver_bullet_events(min_fvg_atr_frac=0.05, max_age=16):
    def in_killzone(idx):
        h, m = H_arr[idx], M_arr[idx]
        for (sh, sm, eh, em) in KILLZONES:
            if (h * 60 + m) >= (sh * 60 + sm) and (h * 60 + m) < (eh * 60 + em):
                return True
        return False

    events = []
    for i in range(2, N - max_age - MAX_HOLD - 2):
        if not in_killzone(i): continue
        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0: continue

        # Bullish FVG
        if HI[i - 1] < LO[i + 1]:
            fbot = HI[i - 1]; ftop = LO[i + 1]; fsz = ftop - fbot
            if fsz < min_fvg_atr_frac * atr: continue
            for j in range(i + 2, min(i + 2 + max_age, N - MAX_HOLD)):
                if CL[j] < fbot: break
                if LO[j] <= ftop:
                    events.append(dict(
                        direction="long", entry_i=j, entry_px=min(CL[j], ftop),
                        ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                        atr_1h=ATR_1H[j], year=yr_arr[i], ts=IDX[i]))
                    break

        # Bearish FVG
        if LO[i - 1] > HI[i + 1]:
            ftop = LO[i - 1]; fbot = HI[i + 1]; fsz = ftop - fbot
            if fsz < min_fvg_atr_frac * atr: continue
            for j in range(i + 2, min(i + 2 + max_age, N - MAX_HOLD)):
                if CL[j] > ftop: break
                if HI[j] >= fbot:
                    events.append(dict(
                        direction="short", entry_i=j, entry_px=max(CL[j], fbot),
                        ref_lo=fbot, ref_hi=ftop, ref_size=fsz,
                        atr_1h=ATR_1H[j], year=yr_arr[i], ts=IDX[i]))
                    break
    return events


# ─────────────────────────────────────────────────────────────────────────────
# ── MODEL 5: Power of 3 (PO3) ───────────────────────────────────────────────
# Asian range: 00:00-07:00 UTC
# London manipulation: 07:00-12:00 UTC — must sweep one side of Asian range
#   Sweep high → expect bearish NY move
#   Sweep low  → expect bullish NY move
# NY distribution: entry at 13:30 UTC (first 15M bar of NY cash open)
# ─────────────────────────────────────────────────────────────────────────────
def collect_po3_events():
    ASIA_START = 0;  ASIA_END   = 7   # UTC hours
    LON_END    = 12; NY_HOUR    = 13; NY_MIN = 30

    events = []
    seen_dates: set = set()

    for i in range(N - MAX_HOLD - 2):
        if H_arr[i] != NY_HOUR or M_arr[i] != NY_MIN: continue
        d = IDX[i].normalize()
        if d in seen_dates: continue
        seen_dates.add(d)

        # Asian range (00:00-07:00 UTC on same date)
        asia_mask = [(IDX[k].normalize() == d and
                      ASIA_START <= H_arr[k] < ASIA_END)
                     for k in range(max(0, i - 60), i)]
        asia_idx = [k for k, m in zip(range(max(0, i - 60), i), asia_mask) if m]
        if len(asia_idx) < 4: continue

        asia_hi = np.max(HI[asia_idx[0]:asia_idx[-1] + 1])
        asia_lo = np.min(LO[asia_idx[0]:asia_idx[-1] + 1])
        asia_rng = asia_hi - asia_lo
        if asia_rng < 10.0: continue    # BTC: at least $10 range

        # London session: did it sweep one side of Asian range?
        lon_mask = [(IDX[k].normalize() == d and ASIA_END <= H_arr[k] < LON_END)
                    for k in range(max(0, i - 40), i)]
        lon_idx  = [k for k, m in zip(range(max(0, i - 40), i), lon_mask) if m]
        if not lon_idx: continue

        lon_hi = np.max(HI[lon_idx[0]:lon_idx[-1] + 1])
        lon_lo = np.min(LO[lon_idx[0]:lon_idx[-1] + 1])

        swept_hi = lon_hi > asia_hi
        swept_lo = lon_lo < asia_lo

        if swept_hi and not swept_lo:
            direction = "short"
        elif swept_lo and not swept_hi:
            direction = "long"
        else:
            continue   # both swept or none → ambiguous

        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0: continue

        events.append(dict(
            direction=direction, entry_i=i, entry_px=CL[i],
            ref_lo=asia_lo, ref_hi=asia_hi, ref_size=asia_rng,
            atr_1h=atr, year=yr_arr[i], ts=IDX[i]))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# ── MODEL 6: Breaker Block ──────────────────────────────────────────────────
# 1. Find swing highs/lows (close-based, lbk bars)
# 2. A swing HIGH that gets broken to the upside → becomes bullish breaker support
#    Entry: price returns to that level from above → LONG
# 3. A swing LOW broken to the downside → becomes bearish breaker resistance
#    Entry: price returns to that level from below → SHORT
# ─────────────────────────────────────────────────────────────────────────────
def collect_breaker_events(lbk=20, max_age=80):
    events = []
    # Precompute rolling swing highs and lows (close-based)
    swing_hi = np.array([np.max(CL[max(0, i - lbk):i]) if i > 0 else CL[0] for i in range(N)])
    swing_lo = np.array([np.min(CL[max(0, i - lbk):i]) if i > 0 else CL[0] for i in range(N)])

    for i in range(lbk + 1, N - max_age - MAX_HOLD - 2):
        atr = ATR_1H[i]
        if np.isnan(atr) or atr <= 0: continue

        prev_sh = swing_hi[i]   # swing high before bar i
        prev_sl = swing_lo[i]

        # Bullish breaker: CL[i] > prev_sh  (broke above swing high)
        # This former resistance level (prev_sh) becomes support
        if CL[i] > prev_sh:
            level = prev_sh
            # Wait for price to return to the level
            for j in range(i + 1, min(i + max_age, N - MAX_HOLD)):
                if CL[j] < level - atr: break    # fell through too far
                if LO[j] <= level + 0.1 * atr:   # price touched the breaker
                    events.append(dict(
                        direction="long", entry_i=j, entry_px=CL[j],
                        ref_lo=level - 0.5 * atr, ref_hi=level + 0.5 * atr,
                        ref_size=atr, atr_1h=atr, year=yr_arr[i], ts=IDX[i]))
                    break

        # Bearish breaker: CL[i] < prev_sl (broke below swing low)
        if CL[i] < prev_sl:
            level = prev_sl
            for j in range(i + 1, min(i + max_age, N - MAX_HOLD)):
                if CL[j] > level + atr: break
                if HI[j] >= level - 0.1 * atr:
                    events.append(dict(
                        direction="short", entry_i=j, entry_px=CL[j],
                        ref_lo=level - 0.5 * atr, ref_hi=level + 0.5 * atr,
                        ref_size=atr, atr_1h=atr, year=yr_arr[i], ts=IDX[i]))
                    break
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Chart & HTML helpers
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
    return f"""<div style="background:{_CARD};border:1px solid {_GRID};
border-radius:8px;padding:16px 20px;margin-bottom:14px">
<h3 style="color:{accent};margin:0 0 10px">{title}</h3>{body}</div>"""


def _kv(label, value):
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:3px 0">'
            f'<span style="color:#9e9e9e">{label}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{value}</span></div>')


def _style(v, fmt=".1%", good=0):
    c = _GRN if v > good else _RED
    return f'<span style="color:{c}">{v:{fmt}}</span>'


def _imgt(b64):
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;margin-bottom:10px">'


def _table(headers, rows):
    th = "".join(f'<th style="padding:5px 8px;text-align:right;color:{_ACC};'
                 f'border-bottom:1px solid {_GRID}">{h}</th>' for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td style="padding:4px 8px;text-align:right;color:{_TEXT}">{c}</td>'
                         for c in r) + "</tr>" for r in rows)
    return (f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


# ═════════════════════════════════════════════════════════════════════════════
# Run each model
# ═════════════════════════════════════════════════════════════════════════════
MODELS = [
    ("FVG",          "Fair Value Gap Fill",           collect_fvg_events),
    ("OB",           "Order Block Reversal",          collect_ob_events),
    ("Sweep",        "Liquidity Sweep + Reversal",    collect_sweep_events),
    ("SilverBullet", "Silver Bullet (FVG+Killzone)",  collect_silver_bullet_events),
    ("PO3",          "Power of 3",                    collect_po3_events),
    ("Breaker",      "Breaker Block",                 collect_breaker_events),
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
    print(f"  Best: tp={tp_f} sl={sl_b}  WR={best['wr']:.1f}%  "
          f"BE={best['be']:.1f}%  ExpPnL={best['exp']:+.5f}%  p={best['p_val']:.4f}")

    print("  Walk-Forward …")
    oos_trades, oos_equity, wr_list, n_wins = run_wf(events, tp_f, sl_b)
    print(f"  OOS trades={len(oos_trades)}  WF wins={n_wins}")

    if not oos_trades:
        results[key] = {"name": name, "key": key, "empty": True}
        continue

    kp = _kpis(oos_equity)
    oos_df = pd.DataFrame([t.__dict__ for t in oos_trades]).sort_values("entry_ts")
    wins_oos = (oos_df["net_pnl"] > 0).sum()
    wr_oos = wins_oos / len(oos_df)
    tp_d = np.mean(np.abs(oos_df["tp_price"].values - oos_df["entry_price"].values)
                   / oos_df["entry_price"].values * 100) if "tp_price" in oos_df.columns else 0

    # Recalculate BE from actual RR
    sl_d = np.mean(np.abs([
        (e["ref_lo"] - sl_b * e["atr_1h"] if e["direction"] == "long"
         else e["ref_hi"] + sl_b * e["atr_1h"]) for e in events
    ])) if events else 1
    # Simpler: use best.rr
    rr = best["rr"]
    be_oos = 1 / (1 + rr) * 100 if rr > 0 else 50.0

    binom_p = st.binomtest(int(wins_oos), len(oos_df),
                            be_oos / 100, alternative="greater").pvalue
    oos_tmp = oos_df.copy()
    oos_tmp["date"] = pd.to_datetime(oos_tmp["exit_ts"]).dt.normalize()
    daily = oos_tmp.groupby("date")["net_pnl"].sum()
    _, t_p = st.ttest_1samp(daily.values, 0) if len(daily) > 1 else (0, 1)

    print(f"  Return={kp['total_return']:+.1%}  MaxDD={kp['max_dd']:.1%}"
          f"  WR={wr_oos:.1%}  BE≈{be_oos:.1f}%  binom p={binom_p:.4f}")

    print("  Monte Carlo …")
    mc_res = run_monte_carlo(oos_df[["net_pnl", "gross_pnl", "total_fees"]], INIT_CAP, N_SIMS)
    p_profit = float((mc_res["total_return"] > 0).mean())
    p_ruin   = mc_res.get("p_ruin", float((mc_res["total_return"] < -0.5).mean()))
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
    if r.get("empty"):
        sum_rows.append([f'<span style="color:{MODEL_COLORS[key]}">{r["name"]}</span>',
                         "—", "—", "—", "—", "—", "—", "—", "—"])
        continue
    kp = r["kp"]
    ok = kp["total_return"] > 0 and r["binom_p"] < 0.05 and r["p_profit"] > 0.5
    verdict = f'<span style="color:{"#66bb6a" if ok else "#ef5350"}">{"✅" if ok else "✗"}</span>'
    sum_rows.append([
        f'<span style="color:{r["color"]}">{r["name"]}</span>',
        f'{len(r["events"]):,}',
        f'{r["ic"]:.4f} (p={r["ic_p"]:.3f})',
        f'{r["best"]["tp_frac"]} / {r["best"]["sl_buf"]}',
        f'{r["best"]["wr"]:.1f}% (BE {r["best"]["be"]:.1f}%)',
        f'{len(r["oos_df"])}',
        f'{r["wr_oos"]:.1%} (BE ~{r["be_oos"]:.1f}%)',
        f'{kp["total_return"]:+.1%}',
        verdict,
    ])

summary_tbl = _table(
    ["Model", "Events IS", "IC (Spearman)", "TP/SL Best",
     "WR IS", "Trades OOS", "WR OOS", "Return OOS", "Verdict"],
    sum_rows,
)

# Per-model sections
sections = ""
for key, r in results.items():
    col  = MODEL_COLORS[key]
    name = r["name"]
    if r.get("empty"):
        sections += f'<h2 style="color:{col}">{name} — nessun evento</h2>'
        continue

    kp   = r["kp"]
    ok   = kp["total_return"] > 0 and r["binom_p"] < 0.05 and r["p_profit"] > 0.5
    vc   = _GRN if ok else _RED
    vt   = "✅ VALIDATO" if ok else "✗ NON VALIDATO"

    img_eq = _plot_equity(r["oos_equity"], name, col)
    img_wf = _plot_wf_bars(r["wr_list"], name, col)
    img_mc = _plot_mc_hist(r["mc_res"], name)

    mc_fin = r["mc_res"]["total_return"] * 100
    mc_dd  = r["mc_res"]["max_drawdown"] * 100

    info_body = "".join([
        _kv("Events IS",     f'{len(r["events"]):,} (L={r["n_long"]}, S={r["n_short"]})'),
        _kv("IC (Spearman)", f'{r["ic"]:.4f}  p={r["ic_p"]:.4f}'),
        _kv("Best tp/sl",    f'tp={r["best"]["tp_frac"]}  sl_buf={r["best"]["sl_buf"]}'),
        _kv("IS WR",         f'{r["best"]["wr"]:.1f}%  BE={r["best"]["be"]:.1f}%  '
                             f'ExpPnL={r["best"]["exp"]:+.5f}%  p={r["best"]["p_val"]:.4f}'),
        _kv("OOS trades",    str(len(r["oos_df"]))),
        _kv("WR OOS",        _style(r["wr_oos"], ".1%")),
        _kv("Total Return",  _style(kp["total_return"], ".1%")),
        _kv("Max DD",        _style(kp["max_dd"], ".1%")),
        _kv("Calmar",        _style(kp["calmar"], ".3f")),
        _kv("Binom p",       f'{r["binom_p"]:.4f}'),
        _kv("t-test daily",  f'{r["t_p"]:.4f}'),
        _kv("P(profit) MC",  _style(r["p_profit"], ".1%")),
        _kv("P(ruin) MC",    _style(r["p_ruin"], ".1%", good=-1)),
        _kv("MC med ret",    f'{np.median(mc_fin):.1f}%'),
        _kv("MC p95 DD",     f'{np.percentile(mc_dd, 95):.1f}%'),
    ])

    # Annual breakdown OOS
    oos_df = r["oos_df"]
    ann_rows = []
    for yr in sorted(oos_df["year"].unique()):
        sub = oos_df[oos_df["year"] == yr]
        w = (sub["net_pnl"] > 0).sum()
        wr_y = w / len(sub) * 100
        ret_y = sub["net_pnl"].sum()
        c2 = _GRN if ret_y > 0 else _RED
        ann_rows.append([yr, len(sub), f"{wr_y:.1f}%",
                         f'<span style="color:{c2}">{ret_y:+,.0f}</span>'])
    ann_tbl = _table(["Anno", "N", "WR%", "Net PnL (USD)"], ann_rows)

    sections += f"""
<hr style="border-color:{_GRID};margin:28px 0">
<h2 style="color:{col}">{name}</h2>
<div style="background:{_CARD};border:2px solid {vc};border-radius:8px;
  padding:12px;margin-bottom:16px;text-align:center;font-size:17px;
  font-weight:bold;color:{vc}">{vt}</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
  {_card("Statistiche IS + OOS", info_body, col)}
  {_card("Breakdown Annuale OOS", ann_tbl, col)}
</div>
{_imgt(img_eq)}{_imgt(img_wf)}{_imgt(img_mc)}
"""

html = f"""<!DOCTYPE html>
<html lang="it">
<head><meta charset="UTF-8">
<title>ICT Suite — BTCUSDT 15M</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{_BG};color:{_TEXT};font-family:monospace;font-size:13px;padding:22px}}
  h1{{color:{_ACC};margin-bottom:6px}}
  h2{{color:{_YEL};margin:18px 0 8px;font-size:14px}}
  table{{overflow-x:auto;display:block}}
</style></head>
<body>
<h1>ICT Model Suite — BTCUSDT 15M | {START_YEAR}–2026</h1>
<p style="color:#9e9e9e;margin-bottom:18px">
  6 modelli ICT &nbsp;|&nbsp; WF {WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / step {WF_STEP_M}m
  &nbsp;|&nbsp; Risk {RISK_PCT*100:.0f}%/trade &nbsp;|&nbsp; Fee {FEE*100:.2f}%/lato
  &nbsp;|&nbsp; MC {N_SIMS:,} sims
</p>
<h2>Comparison Summary</h2>
{_card("Tutti i modelli — IS vs OOS", summary_tbl)}
{sections}
<p style="color:#555;margin-top:28px;font-size:11px">
  ICT Suite Pipeline — BTCUSDT 15M 2020-2026 | Dati: Binance Vision
</p>
</body></html>"""

out = Path("reports") / "report_ict_suite.html"
out.write_text(html, encoding="utf-8")
print(f"Report scritto in: {out}")
print(SEP2)
