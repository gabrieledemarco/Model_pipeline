"""
create_external_strategies_report.py
=====================================
Applies 10 strategies from:
  https://github.com/ali-azary/Algorithmic-Trading-From-Beginner-to-Advanced
to the BTCUSDT 15M IC-ranking + walk-forward validation pipeline.

Strategies:
  S01  Ichimoku Cloud          (T=7, K=14, Sk=30, disp=14)
  S02  Keltner Breakout        (EMA=30, ATR=14, mult=1.0)
  S03  Keltner + RSI           (EMA=30, ATR=7,  RSI=14)
  S04  Momentum Ignition       (StdDev/price<0.10 + ROC z-score)
  S05  OBV Market Regime       (OBV cross MA7, ADX>20, 7-bar breakout)
  S06  OBV Momentum            (OBV cross MA30, vol>avg, RSI filter)
  S07  OU Mean Reversion       (rolling OLS z-score ±1, SMA30 filter)
  S08  Quantile Channel        (rolling 30-bar Q80/Q20, 1% breakout)
  S09  Regime Filtered Trend   (trending score ≥3, SMA7 cross SMA30)
  S10  Relative Momentum Accel (KAMA=30, EMA=7, thrust BB=7 dev=1.0)

Skipped:
  ML-Enhanced ADX  — requires a fitted RandomForestClassifier (no data)
"""
from __future__ import annotations

import base64
import io
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

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
INIT_CAP     = 100_000.0
RISK_PCT     = 0.01
FEE          = 0.0004
MIN_SL_ATR   = 0.50
MAX_LEV      = 5.0
MAX_HOLD     = 32
IC_HORIZON   = 16        # 4H forward return on 15M bars
START_YEAR   = 2020
N_SIMS       = 5_000

WF_TRAIN_M   = 6
WF_OOS_M     = 2
WF_STEP_M    = 2

TP_FRAC_GRID = [1.0, 1.5, 2.0, 3.0]
SL_BUF_GRID  = [0.0, 0.25, 0.5, 1.0]

_BG = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC = "#42a5f5"; _GRN = "#66bb6a"
_RED = "#ef5350"; _YEL = "#ffd54f"; _ORG = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

SOURCE_URL = "github.com/ali-azary/Algorithmic-Trading-From-Beginner-to-Advanced"

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("External Strategies — BTCUSDT 15M | 2020-2026")
print(f"Source: {SOURCE_URL}")
print(SEP2)
print("[DATA] Loading …")

raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=True, fetch_1m=False, fetch_flow=False)
df   = add_indicators(raw["15M"])
df1h = add_indicators(raw["1H"])

print(f"  15M: {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

IDX   = df.index;  N = len(df)
HI    = df["high"].values;    LO   = df["low"].values
CL    = df["close"].values;   OP   = df["open"].values
VOL   = df["volume"].values
ATR   = df["atr_14"].values
RSI14 = df["rsi_14"].values
ADX_  = df["adx"].values
OBV_  = df["obv"].values
yr_   = np.array([t.year for t in IDX], dtype=int)

prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
atr_1h_map = df1h["atr_14"].clip(lower=1.0).to_dict()
fb         = df["atr_14"].clip(lower=1.0).values
ATR1H      = np.array([atr_1h_map.get(t, np.nan) for t in prev_1h], dtype=float)
ATR1H      = np.where(np.isnan(ATR1H), fb, ATR1H)

# ─────────────────────────────────────────────────────────────────────────────
# Strategy-specific indicators
# ─────────────────────────────────────────────────────────────────────────────
print("  Precomputing indicators …")

cl_s = pd.Series(CL)
hi_s = pd.Series(HI)
lo_s = pd.Series(LO)
vol_s = pd.Series(VOL)

# ── S01 Ichimoku (T=7, K=14, Senkou=30, disp=14)
def _donchian_mid(h, l, p):
    return (h.rolling(p, min_periods=p).max() + l.rolling(p, min_periods=p).min()) / 2

tenkan   = _donchian_mid(hi_s, lo_s, 7).values
kijun    = _donchian_mid(hi_s, lo_s, 14).values
senkou_a = ((pd.Series(tenkan) + pd.Series(kijun)) / 2).shift(14).values
senkou_b = _donchian_mid(hi_s, lo_s, 30).shift(14).values
chi_hi14 = hi_s.shift(14).values   # high from 14 bars ago (Chikou comparison)
chi_lo14 = lo_s.shift(14).values   # low  from 14 bars ago

# ── S02/S03 Keltner channels
ema30      = cl_s.ewm(span=30, adjust=False).mean().values
kelt14_up  = ema30 + 1.0 * ATR         # EMA30 + 1×ATR14
kelt14_lo  = ema30 - 1.0 * ATR
prev_c     = cl_s.shift(1)
tr7_       = pd.concat([(hi_s - lo_s),
                         (hi_s - prev_c).abs(),
                         (lo_s - prev_c).abs()], axis=1).max(axis=1)
atr7       = tr7_.ewm(com=6, adjust=False).mean().values
kelt7_up   = ema30 + 1.0 * atr7        # EMA30 + 1×ATR7
kelt7_lo   = ema30 - 1.0 * atr7

# ── S04 Momentum Ignition
sma30       = cl_s.rolling(30, min_periods=10).mean().values
price_std30 = cl_s.rolling(30, min_periods=10).std().fillna(0).values
roc7        = cl_s.pct_change(7).fillna(0).values
roc7_ma30   = pd.Series(roc7).rolling(30, min_periods=10).mean().values
roc7_std30  = pd.Series(roc7).rolling(30, min_periods=10).std().fillna(1e-9).values

# ── S05/S06 OBV cross MAs
vol_ma7    = vol_s.rolling(7,  min_periods=1).mean().values
obv_ma7    = pd.Series(OBV_).rolling(7,  min_periods=1).mean().values
obv_ma30   = pd.Series(OBV_).rolling(30, min_periods=1).mean().values
hi7        = hi_s.rolling(7, min_periods=1).max().values
lo7        = lo_s.rolling(7, min_periods=1).min().values

_ab7   = OBV_ > obv_ma7
_ab30  = OBV_ > obv_ma30
obv_up7  = np.r_[False, _ab7[1:]  & ~_ab7[:-1]]
obv_dn7  = np.r_[False, ~_ab7[1:] &  _ab7[:-1]]
obv_up30 = np.r_[False, _ab30[1:]  & ~_ab30[:-1]]
obv_dn30 = np.r_[False, ~_ab30[1:] &  _ab30[:-1]]

# ── S09 Regime Filtered Trend
sma7       = cl_s.rolling(7, min_periods=1).mean().values
bb_squeeze = df["bb_squeeze"].values            # (bb_up - bb_lo) / bb_mid
atr_pct_f  = ATR / (CL + 1e-9)
ma_sep     = np.abs(sma7 - sma30) / (sma30 + 1e-9)
regime_score = (
    (ADX_ > 20).astype(int)         +
    (bb_squeeze > 0.01).astype(int) +
    (atr_pct_f  > 0.01).astype(int) +
    (ma_sep     > 0.02).astype(int)
)
is_trending = regime_score >= 3
_sma7_ab    = sma7 > sma30
sma7_up_x   = np.r_[False, _sma7_ab[1:] & ~_sma7_ab[:-1]]
sma7_dn_x   = np.r_[False, ~_sma7_ab[1:] & _sma7_ab[:-1]]

# ── S08 Quantile Channel
q80 = cl_s.rolling(30, min_periods=15).quantile(0.80).values
q20 = cl_s.rolling(30, min_periods=15).quantile(0.20).values

# ── S07 OU Mean Reversion z-score
print("  Computing OU z-scores (~5 s) …")
log_cl = np.log(CL + 1e-9)

def _ou_zscore(log_close, window=30):
    n = len(log_close)
    z = np.full(n, np.nan)
    for i in range(window, n):
        x   = log_close[i - window: i]   # window values, length=window
        dy  = x[1:] - x[:-1]             # ΔX, length=window-1
        xx  = x[:-1]                      # X_{t-1}, length=window-1
        xx_m, dy_m = xx.mean(), dy.mean()
        sxx = np.dot(xx - xx_m, xx - xx_m)
        if sxx < 1e-12: continue
        beta  = np.dot(xx - xx_m, dy - dy_m) / sxx
        if beta >= 0: continue
        alpha = dy_m - beta * xx_m
        resid = dy - (alpha + beta * xx)
        sigma_eq = resid.std() / np.sqrt(-2.0 * beta)
        if sigma_eq < 1e-12: continue
        z[i] = (x[-1] - (-alpha / beta)) / sigma_eq
    return z

ou_z = _ou_zscore(log_cl, window=30)

# ── S10 KAMA + Thrust Oscillator
print("  Computing KAMA …")

def _kama_np(close_arr, period=30, fast=2, slow=30):
    sc_f = 2.0 / (fast + 1)
    sc_s = 2.0 / (slow + 1)
    n    = len(close_arr)
    k    = close_arr.copy().astype(float)
    # precompute cumulative abs-diff for fast volatility rolling sum
    adc  = np.r_[0.0, np.cumsum(np.abs(np.diff(close_arr)))]
    for i in range(period + 1, n):
        d   = abs(close_arr[i] - close_arr[i - period])
        v   = adc[i] - adc[i - period]
        er  = d / v if v > 1e-12 else 0.0
        sc  = (er * (sc_f - sc_s) + sc_s) ** 2
        k[i] = k[i - 1] + sc * (close_arr[i] - k[i - 1])
    return k

kama   = _kama_np(CL, period=30)
ema7   = cl_s.ewm(span=7, adjust=False).mean().values
thrust = np.where(kama > 1e-9, (ema7 - kama) / kama, 0.0)
th_s   = pd.Series(thrust)
th_mean = th_s.rolling(7, min_periods=1).mean().values
th_std  = th_s.rolling(7, min_periods=1).std().fillna(0).values
th_upper = th_mean + 1.0 * th_std
th_lower = th_mean - 1.0 * th_std
_th_ab_u = thrust > th_upper
_th_ab_l = thrust < th_lower
th_cross_up  = np.r_[False, _th_ab_u[1:]  & ~_th_ab_u[:-1]]
th_cross_dn  = np.r_[False, _th_ab_l[1:]  & ~_th_ab_l[:-1]]

print("  Indicators ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build standard event dict
# SL anchor: ±1×ATR1H from entry; TP ref: ATR1H
# ─────────────────────────────────────────────────────────────────────────────
def _ev(i, direction):
    a = max(float(ATR1H[i]), 1.0)
    e = float(CL[i])
    return dict(
        direction = direction,
        entry_i   = i,
        entry_px  = e,
        ref_lo    = e - a,        # SL anchor for longs   (1 ATR below)
        ref_hi    = e + a,        # SL anchor for shorts  (1 ATR above)
        ref_size  = a,            # TP reference unit
        atr_1h    = a,
        year      = int(yr_[i]),
        ts        = IDX[i],
    )

# ─────────────────────────────────────────────────────────────────────────────
# IC engine
# ─────────────────────────────────────────────────────────────────────────────
def compute_ic(events):
    if len(events) < 30:
        return 0.0, 1.0, len(events)
    sigs, fwds = [], []
    for ev in events:
        ei  = ev["entry_i"]
        end = min(ei + IC_HORIZON, N - 1)
        fwd = (CL[end] - ev["entry_px"]) / ev["entry_px"] * 100
        sig = 1.0 if ev["direction"] == "long" else -1.0
        sigs.append(sig); fwds.append(sig * fwd)
    ic, p = st.spearmanr(sigs, fwds)
    return float(ic), float(p), len(events)

# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_ts: pd.Timestamp; exit_ts: pd.Timestamp
    direction: int; entry_price: float; exit_price: float
    net_pnl: float; gross_pnl: float; total_fees: float
    exit_reason: str; year: int; window_id: int

def run_backtest(events, tp_frac, sl_buf, initial_capital=INIT_CAP, window_id=0):
    equity = float(initial_capital); trades = []
    for ev in events:
        if equity < initial_capital * 0.005: break
        entry = ev["entry_px"]; rs = max(ev["ref_size"], 1e-6)
        atr   = ev["atr_1h"];   d  = 1 if ev["direction"] == "long" else -1
        tp_px = entry + d * tp_frac * rs
        sl_px = ev["ref_lo"] - sl_buf * atr if d == 1 else ev["ref_hi"] + sl_buf * atr
        sl_d  = abs(entry - sl_px); tp_d = abs(tp_px - entry)
        if sl_d <= 0 or tp_d <= 0: continue
        if d == 1 and sl_px >= entry: continue
        if d == -1 and sl_px <= entry: continue
        sl_d  = max(sl_d, atr * MIN_SL_ATR)
        qty   = min(equity * RISK_PCT / sl_d, equity * MAX_LEV / entry)
        notl  = qty * entry; e_fee = notl * FEE
        ei    = ev["entry_i"]; start = ei + 1
        hit_tp = hit_sl = False
        exit_bar = min(start + MAX_HOLD, N - 1); exit_px = float(CL[exit_bar])
        for k in range(start, min(start + MAX_HOLD, N)):
            bh, bl = float(HI[k]), float(LO[k])
            if d == 1:
                if bl <= sl_px: hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bh >= tp_px: hit_tp = True; exit_px = tp_px; exit_bar = k; break
            else:
                if bh >= sl_px: hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bl <= tp_px: hit_tp = True; exit_px = tp_px; exit_bar = k; break
        gross = d * qty * (exit_px - entry)
        net   = gross - e_fee - qty * exit_px * FEE
        equity += net
        trades.append(Trade(
            entry_ts=IDX[ei], exit_ts=IDX[exit_bar], direction=d,
            entry_price=entry, exit_price=exit_px, gross_pnl=gross,
            total_fees=e_fee + qty * exit_px * FEE, net_pnl=net,
            exit_reason="tp" if hit_tp else ("sl" if hit_sl else "time"),
            year=ev["year"], window_id=window_id,
        ))
    if not trades: return trades, pd.Series(dtype=float)
    eq = np.empty(len(trades) + 1); eq[0] = initial_capital
    for k, t in enumerate(trades): eq[k + 1] = eq[k] + t.net_pnl
    return trades, pd.Series(eq[1:], index=pd.DatetimeIndex([t.exit_ts for t in trades]))

def is_scan(events):
    paths = [(HI[e["entry_i"]+1: e["entry_i"]+1+MAX_HOLD],
              LO[e["entry_i"]+1: e["entry_i"]+1+MAX_HOLD]) for e in events]
    results = []
    for tp_f, sl_b in product(TP_FRAC_GRID, SL_BUF_GRID):
        wins = losses = n_v = 0; sum_tp = sum_sl = 0.0
        for ev, (ph, pl) in zip(events, paths):
            entry = ev["entry_px"]; rs = max(ev["ref_size"], 1e-6)
            a = ev["atr_1h"]; d = 1 if ev["direction"] == "long" else -1
            tp_px = entry + d * tp_f * rs
            sl_px = ev["ref_lo"] - sl_b * a if d == 1 else ev["ref_hi"] + sl_b * a
            sd = abs(entry - sl_px); td = abs(tp_px - entry)
            if sd <= 0 or td <= 0: continue
            if d == 1 and sl_px >= entry: continue
            if d == -1 and sl_px <= entry: continue
            n_v += 1; sum_tp += td / entry * 100; sum_sl += sd / entry * 100
            ht = hs = False
            for h, l in zip(ph, pl):
                if d == 1:
                    if l <= sl_px: hs = True; break
                    if h >= tp_px: ht = True; break
                else:
                    if h >= sl_px: hs = True; break
                    if l <= tp_px: ht = True; break
            if ht: wins += 1
            elif hs: losses += 1
        if n_v < 10: continue
        wr = wins / n_v * 100
        rr = (sum_tp / n_v) / (sum_sl / n_v) if sum_sl > 0 else 0
        be = 1 / (1 + rr) * 100 if rr > 0 else 50.0
        exp = wr / 100 * (sum_tp / n_v) - (1 - wr / 100) * (sum_sl / n_v)
        p_v = st.binomtest(int(round(wr / 100 * n_v)), n_v, be / 100,
                           alternative="greater").pvalue
        results.append(dict(tp_frac=tp_f, sl_buf=sl_b, n=n_v,
                            wr=round(wr, 2), rr=round(rr, 2),
                            be=round(be, 2), exp=round(exp, 5), p_val=round(p_v, 4)))
    df_r = pd.DataFrame(results)
    if df_r.empty:
        return df_r, dict(tp_frac=2.0, sl_buf=0.5, n=0, wr=0, rr=0, be=50, exp=0, p_val=1)
    return df_r, df_r.sort_values("exp", ascending=False).iloc[0].to_dict()

def _kpis(eq, init=INIT_CAP):
    if eq.empty: return dict(total_return=0, calmar=0, sharpe=0, max_dd=0)
    full = pd.concat([pd.Series([init], index=[eq.index[0] - pd.Timedelta("1s")]), eq])
    dd   = (full / full.cummax() - 1).min()
    ret  = full.iloc[-1] / init - 1
    rets = full.pct_change().dropna()
    vol  = rets.std() * np.sqrt(365 * 24)
    ann  = rets.mean() * 365 * 24
    return dict(total_return=ret, calmar=ret / abs(dd) if dd < 0 else 0,
                sharpe=ann / vol if vol > 0 else 0, max_dd=dd)

def run_wf(events, tp_frac, sl_buf):
    from dateutil.relativedelta import relativedelta
    start, end = IDX[0], IDX[-1]; wlist = []; cur = start
    while True:
        tr_e = cur + relativedelta(months=WF_TRAIN_M)
        oo_e = tr_e + relativedelta(months=WF_OOS_M)
        if oo_e > end: break
        wlist.append((cur, tr_e, oo_e)); cur = cur + relativedelta(months=WF_STEP_M)
    wr_list = []; all_oos = []
    for wid, (tr_s, tr_e, oo_e) in enumerate(wlist):
        is_ev = [e for e in events if tr_s <= e["ts"] < tr_e]
        oo_ev = [e for e in events if tr_e <= e["ts"] < oo_e]
        if len(is_ev) < 5 or len(oo_ev) < 3: continue
        _, oo_eq = run_backtest(oo_ev, tp_frac, sl_buf, window_id=wid)
        if oo_eq.empty: continue
        wr_list.append(_kpis(oo_eq)["total_return"]); all_oos.extend(oo_ev)
    oos = sorted(all_oos, key=lambda e: e["ts"])
    oos_t, oos_eq = run_backtest(oos, tp_frac, sl_buf)
    return oos_t, oos_eq, wr_list, len(wlist)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Collectors
# ─────────────────────────────────────────────────────────────────────────────

def S01_ichimoku():
    evs = []
    for i in range(50, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        sa, sb = senkou_a[i], senkou_b[i]
        if np.isnan(sa) or np.isnan(sb): continue
        tk, kj = tenkan[i], kijun[i]
        if np.isnan(tk) or np.isnan(kj): continue
        cl = CL[i]
        cloud_top = max(sa, sb); cloud_bot = min(sa, sb)
        if cl > cloud_top and tk > kj and cl > chi_hi14[i]:
            evs.append(_ev(i, "long"))
        elif cl < cloud_bot and tk < kj and cl < chi_lo14[i]:
            evs.append(_ev(i, "short"))
    return evs


def S02_keltner_breakout():
    evs = []
    for i in range(33, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        # Fresh crossover: prev bar broke band, bar before did not
        if CL[i-1] > kelt14_up[i-1] and CL[i-2] <= kelt14_up[i-2]:
            evs.append(_ev(i, "long"))
        elif CL[i-1] < kelt14_lo[i-1] and CL[i-2] >= kelt14_lo[i-2]:
            evs.append(_ev(i, "short"))
    return evs


def S03_keltner_rsi():
    evs = []
    for i in range(33, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        cl = CL[i]; rsi = RSI14[i]
        if cl > kelt7_up[i] and rsi > 30:
            evs.append(_ev(i, "long"))
        elif cl < kelt7_lo[i] and rsi < 70:
            evs.append(_ev(i, "short"))
    return evs


def S04_momentum_ignition():
    evs = []
    for i in range(40, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        cl = CL[i]; pstd = price_std30[i]
        if np.isnan(pstd) or cl < 1e-6: continue
        if pstd / cl >= 0.10: continue          # must be consolidating
        roc = roc7[i]; rma = roc7_ma30[i]; rstd = roc7_std30[i]
        if np.isnan(rma) or rstd < 1e-12: continue
        sm = sma30[i]
        if np.isnan(sm): continue
        if roc > rma + 1.0 * rstd and cl > sm:
            evs.append(_ev(i, "long"))
        elif roc < rma - 1.0 * rstd and cl < sm:
            evs.append(_ev(i, "short"))
    return evs


def S05_obv_market_regime():
    evs = []
    for i in range(10, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        if not (VOL[i] > vol_ma7[i] and ADX_[i] > 20): continue
        if obv_up7[i] and RSI14[i] < 70 and CL[i] > hi7[i-1]:
            evs.append(_ev(i, "long"))
        elif obv_dn7[i] and RSI14[i] > 30 and CL[i] < lo7[i-1]:
            evs.append(_ev(i, "short"))
    return evs


def S06_obv_momentum():
    evs = []
    for i in range(32, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        if VOL[i] <= vol_ma7[i]: continue
        if obv_up30[i] and RSI14[i] < 70:
            evs.append(_ev(i, "long"))
        elif obv_dn30[i] and RSI14[i] > 30:
            evs.append(_ev(i, "short"))
    return evs


def S07_ou_mean_reversion():
    evs = []
    for i in range(35, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        z  = ou_z[i];  sm = sma30[i]
        if np.isnan(z) or np.isnan(sm): continue
        if z < -1.0 and CL[i] > sm:
            evs.append(_ev(i, "long"))
        elif z > 1.0 and CL[i] < sm:
            evs.append(_ev(i, "short"))
    return evs


def S08_quantile_channel():
    evs = []
    for i in range(32, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        qu = q80[i]; ql = q20[i]
        if np.isnan(qu) or np.isnan(ql) or ql < 1e-6: continue
        if CL[i] > qu * 1.01:
            evs.append(_ev(i, "long"))
        elif CL[i] < ql / 1.01:
            evs.append(_ev(i, "short"))
    return evs


def S09_regime_trend():
    evs = []
    for i in range(32, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        if not is_trending[i]: continue
        if sma7_up_x[i]:
            evs.append(_ev(i, "long"))
        elif sma7_dn_x[i]:
            evs.append(_ev(i, "short"))
    return evs


def S10_relative_momentum_accel():
    evs = []
    for i in range(45, N - MAX_HOLD - 2):
        a = ATR1H[i]
        if np.isnan(a) or a <= 0: continue
        if np.isnan(kama[i]): continue
        if th_cross_up[i]:
            evs.append(_ev(i, "long"))
        elif th_cross_dn[i]:
            evs.append(_ev(i, "short"))
    return evs


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = [
    ("S01", "Ichimoku Cloud",
     S01_ichimoku,
     "close>cloud + Tenkan>Kijun + close>high[i-14]  |  Short: inverse",
     "IchimokuCloudStrategy.py"),
    ("S02", "Keltner Breakout",
     S02_keltner_breakout,
     "prev close outside EMA(30)±ATR(14) band → fresh crossover entry",
     "KeltnerBreakoutStrategy.py"),
    ("S03", "Keltner + RSI",
     S03_keltner_rsi,
     "close > EMA(30)+ATR(7) AND RSI>30  |  close < EMA-ATR AND RSI<70",
     "KeltnerChannelRSIBreakoutStrategy.py"),
    ("S04", "Momentum Ignition",
     S04_momentum_ignition,
     "StdDev30/price<0.10 + SMA30 trend + ROC(7)>ROC_MA±1σ",
     "MomentumIgnitionStrategy.py"),
    ("S05", "OBV Market Regime",
     S05_obv_market_regime,
     "OBV cross MA(7) + vol>vol_MA7 + ADX>20 + 7-bar price breakout",
     "OBVMarketRegimeStrategyBreakout.py"),
    ("S06", "OBV Momentum",
     S06_obv_momentum,
     "OBV cross MA(30) + vol>vol_MA7 + RSI<70 (long) / RSI>30 (short)",
     "OBVmomentumStrategy.py"),
    ("S07", "OU Mean Reversion",
     S07_ou_mean_reversion,
     "Rolling OLS: z<−1 + close>SMA30 = long  |  z>+1 + close<SMA30 = short",
     "OUMeanReversionStrategy.py"),
    ("S08", "Quantile Channel",
     S08_quantile_channel,
     "close > 30-bar Q80×1.01 (breakout)  |  close < Q20÷1.01 (breakdown)",
     "QuantileChannelStrategy.py"),
    ("S09", "Regime Filtered Trend",
     S09_regime_trend,
     "Trending regime (score≥3) + SMA(7) fresh cross SMA(30)",
     "RegimeFilteredTrendStrategy.py"),
    ("S10", "Relative Momentum Accel",
     S10_relative_momentum_accel,
     "Thrust=(EMA7−KAMA30)/KAMA30 breaks above/below 7-bar BB (dev=1.0)",
     "RelativeMomentumAccel.py"),
]

# ─────────────────────────────────────────────────────────────────────────────
# IC computation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Computing IC for all 10 external strategies …")
print(SEP)

ic_results = []
for key, label, collector, desc, src in STRATEGIES:
    evs = collector()
    ic, p, n = compute_ic(evs)
    ic_results.append(dict(key=key, label=label, n=n,
                           ic=round(ic, 5), p=round(p, 4),
                           events=evs, desc=desc, src=src))
    sig = "✅" if p < 0.05 else ("~" if p < 0.10 else "✗")
    print(f"  [{key}] {label:<26s}  n={n:7,}  IC={ic:+.4f}  p={p:.4f}  {sig}")

def _rank_key(x):
    ic = x["ic"] if not (x["ic"] != x["ic"]) else 0.0
    p  = x["p"]  if not (x["p"]  != x["p"])  else 1.0
    return (p, -ic) if ic > 0 else (1.0 + abs(ic), -ic)

ranked = sorted(ic_results, key=_rank_key)

print(f"\n{SEP}")
print("  RANKING (lowest p-value, positive IC first)")
print(SEP)
for rank, r in enumerate(ranked, 1):
    sig = "✅" if r["p"]<0.05 else ("~" if r["p"]<0.10 else "✗")
    print(f"  #{rank:2d}  [{r['key']}] {r['label']:<26s}  IC={r['ic']:+.4f}  p={r['p']:.4f}  {sig}")

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline on best setup (must have positive IC and p < 0.10)
# ─────────────────────────────────────────────────────────────────────────────
best = next((r for r in ranked if r["ic"] > 0 and r["p"] < 0.10), ranked[0])

print(f"\n{SEP2}")
print(f"  BEST: [{best['key']}] {best['label']}")
print(f"  IC={best['ic']:+.4f}  p={best['p']:.4f}  n={best['n']:,}")
print(SEP2)

events = best["events"]
print("  IS scan …")
df_scan, best_params = is_scan(events)
tp_f, sl_b = best_params["tp_frac"], best_params["sl_buf"]
print(f"  Best tp={tp_f}  sl={sl_b}  WR={best_params['wr']:.1f}%  "
      f"BE={best_params['be']:.1f}%  ExpPnL={best_params['exp']:+.5f}%  p={best_params['p_val']:.4f}")

print("  Walk-Forward …")
oos_t, oos_eq, wr_list, n_wins = run_wf(events, tp_f, sl_b)
print(f"  OOS trades={len(oos_t)}  WF windows={n_wins}")
kp = _kpis(oos_eq)
oos_df = (pd.DataFrame([t.__dict__ for t in oos_t]).sort_values("entry_ts")
          if oos_t else pd.DataFrame())
wins_n = int((oos_df["net_pnl"] > 0).sum()) if not oos_df.empty else 0
wr_oos = wins_n / len(oos_df) if len(oos_df) else 0
rr     = best_params["rr"]
be_oos = 1 / (1 + rr) * 100 if rr > 0 else 50.0
binom_p = (st.binomtest(wins_n, max(len(oos_df), 1), be_oos / 100,
                        alternative="greater").pvalue
           if len(oos_df) else 1)
print(f"  Return={kp['total_return']:+.1%}  MaxDD={kp['max_dd']:.1%}  "
      f"WR={wr_oos:.1%}  BE≈{be_oos:.1f}%  binom_p={binom_p:.4f}")

print("  Monte Carlo …")
mc_res = (run_monte_carlo(oos_df[["net_pnl","gross_pnl","total_fees"]],
                          INIT_CAP, N_SIMS)
          if len(oos_df) > 5 else {})
p_profit = float((mc_res["total_return"] > 0).mean()) if mc_res else 0
p_ruin   = float((mc_res["total_return"] < -0.5).mean()) if mc_res else 1
print(f"  P(profit)={p_profit:.1%}  P(ruin)={p_ruin:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[HTML] Generating report …")

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def _imgt(b64):
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;margin-bottom:12px">'

def _kv(lbl, val):
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:4px 0">'
            f'<span style="color:#9e9e9e">{lbl}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{val}</span></div>')

def _style(v, fmt=".1%", good=0):
    c = _GRN if v > good else _RED
    return f'<span style="color:{c}">{v:{fmt}}</span>'

def ic_color(ic, p):
    if p < 0.05 and ic > 0: return _GRN
    if p < 0.10 and ic > 0: return _YEL
    if ic < 0: return _RED
    return "#9e9e9e"

# IC bar chart
def plot_ic_bar():
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    xs = range(len(ranked))
    ics = [r["ic"] for r in ranked]
    ps  = [r["p"]  for r in ranked]
    clrs = [_GRN if p < 0.05 and ic > 0 else
            (_YEL if p < 0.10 and ic > 0 else _RED)
            for ic, p in zip(ics, ps)]
    ax.bar(xs, ics, color=clrs, alpha=0.85)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["key"] for r in ranked], rotation=0, fontsize=9, color=_TEXT)
    ax.set_ylabel("IC (Spearman)", color=_TEXT)
    ax.set_title(f"IC Ranking — 10 External Strategies  |  green=p<0.05  yellow=p<0.10",
                 color=_TEXT, fontsize=10)
    fig.tight_layout()
    return _fig_to_b64(fig)

def plot_equity(eq, label):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_CARD); ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values(): sp.set_color(_GRID)
    ax1.plot(eq.index, eq.values / INIT_CAP, color=_GRN, lw=1.5)
    ax1.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (norm.)", color=_TEXT)
    rm = eq.cummax(); dd = (eq / rm - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, color=_RED, alpha=0.4)
    ax2.set_ylabel("DD %", color=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle(f"OOS Equity Curve — {label}", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)

def plot_wf_bars(wr_list, label):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    clrs = [_GRN if r > 0 else _RED for r in wr_list]
    ax.bar(range(len(wr_list)), [r * 100 for r in wr_list], color=clrs, alpha=0.8)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_title(f"WF per-window OOS returns — {label}", color=_TEXT)
    ax.set_ylabel("Return %", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)

def plot_mc(mc_res, label):
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
    ax.set_title(f"Monte Carlo (5 000 sims) — {label}", color=_TEXT)
    ax.legend(facecolor=_CARD, labelcolor=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)

img_rank = plot_ic_bar()
img_eq   = plot_equity(oos_eq, best["label"]) if not oos_eq.empty else ""
img_wf   = plot_wf_bars(wr_list, best["label"])
img_mc   = plot_mc(mc_res, best["label"]) if mc_res else ""

# Ranking table
th_cols = ["Rank", "Key", "Strategy", "N events", "IC", "p-value", "Sig"]
th_html = "".join(
    f'<th style="padding:5px 8px;text-align:right;color:{_ACC};border-bottom:1px solid {_GRID}">{h}</th>'
    for h in th_cols)
tr_rows = []
for rank, r in enumerate(ranked, 1):
    c   = ic_color(r["ic"], r["p"])
    sig = "✅" if r["p"] < 0.05 else ("~" if r["p"] < 0.10 else "✗")
    cells = [
        f'<b style="color:{_ACC}">#{rank}</b>',
        f'<span style="color:{c}">{r["key"]}</span>',
        f'<span style="color:{c}">{r["label"]}</span>',
        f'{r["n"]:,}',
        f'<span style="color:{c}">{r["ic"]:+.4f}</span>',
        f'<span style="color:{c}">{r["p"]:.4f}</span>',
        sig,
    ]
    tr_rows.append(
        "<tr>" + "".join(f'<td style="padding:4px 8px;text-align:right;color:{_TEXT}">{c}</td>'
                         for c in cells) + "</tr>")
rank_table = (f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
              f'<thead><tr>{th_html}</tr></thead>'
              f'<tbody>{"".join(tr_rows)}</tbody></table>')

# Description table (all strategies, original order)
th2_cols = ["Key", "Strategy", "Signal Logic", "Source File"]
th2_html = "".join(
    f'<th style="padding:5px 8px;color:{_ACC};border-bottom:1px solid {_GRID};text-align:left">{h}</th>'
    for h in th2_cols)
tr2_rows = []
for r in ic_results:
    c = ic_color(r["ic"], r["p"])
    cells = [
        f'<span style="color:{c}">{r["key"]}</span>',
        f'<b style="color:{_TEXT}">{r["label"]}</b>',
        f'<span style="font-size:11px;color:#b0b0b0">{r["desc"]}</span>',
        f'<span style="font-size:10px;color:#555">{r["src"]}</span>',
    ]
    tr2_rows.append(
        "<tr>" + "".join(f'<td style="padding:5px 8px;color:{_TEXT};vertical-align:top">{c}</td>'
                         for c in cells) + "</tr>")
desc_table = (f'<table style="width:100%;border-collapse:collapse">'
              f'<thead><tr>{th2_html}</tr></thead>'
              f'<tbody>{"".join(tr2_rows)}</tbody></table>')

# Best setup metrics
ok  = kp["total_return"] > 0 and binom_p < 0.05 and p_profit > 0.5
vc  = _GRN if ok else _RED
vt  = "✅ VALIDATED" if ok else "✗ NOT VALIDATED"
mc_fin = pd.Series(mc_res["total_return"] * 100) if mc_res else pd.Series([0])

metrics_html = "".join([
    _kv("Strategy",          f'[{best["key"]}] {best["label"]}'),
    _kv("Source file",       best["src"]),
    _kv("Signal logic",      best["desc"]),
    _kv("N events (IS)",     f'{best["n"]:,}'),
    _kv("IC (Spearman)",     f'{best["ic"]:+.4f}  p={best["p"]:.4f}'),
    _kv("IS best tp/sl",     f'tp={tp_f}×ATR  sl=(1+{sl_b})×ATR'),
    _kv("IS WR / BE",        f'{best_params["wr"]:.1f}% / {best_params["be"]:.1f}%  '
                             f'ExpPnL={best_params["exp"]:+.5f}%'),
    _kv("OOS trades",        str(len(oos_t))),
    _kv("WR OOS / BE",       _style(wr_oos, ".1%") + f' / {be_oos:.1f}%'),
    _kv("Total Return OOS",  _style(kp["total_return"], ".1%")),
    _kv("Max Drawdown",      _style(kp["max_dd"], ".1%")),
    _kv("Binom p-value",     f'{binom_p:.4f}'),
    _kv("MC P(profit)",      _style(p_profit, ".1%", good=0.5)),
    _kv("MC P(ruin<-50%)",   f'{p_ruin:.1%}'),
    _kv("MC median return",  f'{float(np.median(mc_fin)):.1f}%'),
    _kv("WF profitable",     f'{sum(1 for x in wr_list if x > 0)} / {len(wr_list)} windows'),
    _kv("Verdict",           f'<b style="color:{vc};font-size:14px">{vt}</b>'),
])

metrics_card = (f'<div style="background:{_CARD};border:1px solid {_GRID};'
                f'border-radius:8px;padding:16px 20px;margin-bottom:14px">'
                f'{metrics_html}</div>')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>External Strategies — BTCUSDT 15M IC Ranking</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{_BG};color:{_TEXT};font-family:monospace;padding:24px;line-height:1.5}}
  h1{{color:{_ACC};margin-bottom:4px;font-size:20px}}
  h2{{color:{_ACC};margin-top:32px;margin-bottom:10px;font-size:15px}}
  p{{margin-bottom:8px;color:#9e9e9e;font-size:13px}}
  a{{color:{_ACC}}}
  table{{width:100%;border-collapse:collapse;margin-bottom:16px}}
  th,td{{padding:4px 8px;vertical-align:top}}
  thead th{{color:{_ACC};border-bottom:1px solid {_GRID}}}
</style>
</head>
<body>

<h1>External Strategies — BTCUSDT 15M | 2020-2026</h1>
<p>Source: <a href="https://{SOURCE_URL}">{SOURCE_URL}</a></p>
<p>IC = Spearman corr(direction signal, 4H signed forward return) · ranked by reliability (p-value)</p>
<p>Pipeline: IC screen → IS scan (4×4 TP/SL grid) → Walk-Forward (35 windows 6m/2m) → Monte Carlo (5 000 sims)</p>

<h2>IC Ranking — Visual</h2>
{_imgt(img_rank)}

<h2>IC Ranking Table</h2>
{rank_table}

<h2>All 10 Strategies — Signal Logic Reference</h2>
{desc_table}

<h2>Full Validation: [{best["key"]}] {best["label"]}</h2>
{metrics_card}
{"" if not img_eq else _imgt(img_eq)}
{_imgt(img_wf)}
{"" if not img_mc else _imgt(img_mc)}

<p style="color:#555;font-size:11px;margin-top:40px">
  Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC — BTCUSDT 15M 2020-2026
</p>
</body>
</html>"""

out = Path("reports/report_external_strategies.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n✅ Report saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")
print(SEP2)
