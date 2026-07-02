"""
create_mtf_report.py
=====================
Multi-Timeframe (MTF) experiment — BTCUSDT 1H signals, 2020-2026.

Two configurations compared:
  BASE      1H entry · 1H ATR sizing · no filter        (reference)
  MTF_FILT  1H entry · 4H ATR sizing · 4H trend filter  (noise cut + wider stops)

Fee maths with 4H ATR (ATR_4H ≈ 1.6% vs 0.4% for 1H):
  tp=2×sl=0.5  →  BE_fee = (0.8+0.08)/(3.2+0.8) = 22%   [was 47% with 1H ATR]
  tp=3×sl=0.5  →  BE_fee = (0.8+0.08)/(4.8+0.8) = 16%
  tp=5×sl=0.5  →  BE_fee = (0.8+0.08)/(8.0+0.8) = 10%

4H trend filter: long only if 1H close > 4H EMA30, short only if below.
Full pipeline (IS scan → WFO → MC) on MTF_FILT strategies with IC>0, p<0.05.
"""
from __future__ import annotations

import base64, io, sys, warnings
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

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP   = 100_000.0
RISK_PCT   = 0.01
FEE        = 0.0004
FEE_RT_PCT = FEE * 2 * 100       # 0.08%
MIN_SL_ATR = 0.25
MAX_LEV    = 5.0
MAX_HOLD   = 96                   # 4 days: 4H ATR targets need more resolution time
IC_HORIZON = 16                   # 16H forward return (unchanged)
START_YEAR = 2020
N_SIMS     = 5_000

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

TP_FRAC_GRID = [1.0, 2.0, 3.0, 5.0]
SL_FRAC_GRID = [0.25, 0.5, 0.75, 1.0]

_BG = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC = "#42a5f5"; _GRN = "#66bb6a"
_RED = "#ef5350"; _YEL = "#ffd54f"
SEP  = "─" * 70
SEP2 = "═" * 70

# ── Data ──────────────────────────────────────────────────────────────────────
print(SEP2)
print("MTF Experiment — BTCUSDT 1H signals × 4H sizing + trend filter")
print(SEP2)
print("[DATA] Loading …")

raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df   = add_indicators(raw["1H"])
df4h = add_indicators(raw["4H"])

print(f"  1H : {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  4H : {len(df4h):,} bars  ({df4h.index[0].date()} → {df4h.index[-1].date()})")

IDX   = df.index;  N = len(df)
HI    = df["high"].values;   LO  = df["low"].values
CL    = df["close"].values;  VOL = df["volume"].values
ATR   = df["atr_14"].values
RSI14 = df["rsi_14"].values
ADX_  = df["adx"].values
OBV_  = df["obv"].values
yr_   = np.array([t.year for t in IDX], dtype=int)
ATR1H = np.where(ATR > 0, ATR, 1.0)

# ── 4H indicators mapped to 1H timestamps ────────────────────────────────────
print("  Mapping 4H → 1H …")
atr4h_raw = df4h["atr_14"].shift(1).reindex(IDX, method="ffill")    # shift: use prev closed 4H bar
ATR4H     = np.where(atr4h_raw.values > 0, atr4h_raw.values, ATR1H)

ema30_4h = (df4h["close"].ewm(span=30, adjust=False).mean()
            .shift(1)                                                   # shift: use prev closed 4H bar
            .reindex(IDX, method="ffill").values)

# 4H trend-alignment filter
trend_long_4h  = CL > ema30_4h   # long allowed: price above 4H EMA30
trend_short_4h = CL < ema30_4h   # short allowed: price below 4H EMA30

# ── 1H indicator precomputation ───────────────────────────────────────────────
print("  Precomputing 1H indicators …")
cl_s = pd.Series(CL); hi_s = pd.Series(HI)
lo_s = pd.Series(LO); vol_s = pd.Series(VOL)

def _donchian_mid(h, l, p):
    return (h.rolling(p, min_periods=p).max() + l.rolling(p, min_periods=p).min()) / 2

tenkan   = _donchian_mid(hi_s, lo_s, 7).values
kijun    = _donchian_mid(hi_s, lo_s, 14).values
senkou_a = ((pd.Series(tenkan) + pd.Series(kijun)) / 2).shift(14).values
senkou_b = _donchian_mid(hi_s, lo_s, 30).shift(14).values
chi_hi14 = hi_s.shift(14).values
chi_lo14 = lo_s.shift(14).values

ema30      = cl_s.ewm(span=30, adjust=False).mean().values
kelt14_up  = ema30 + 1.0 * ATR;  kelt14_lo = ema30 - 1.0 * ATR
prev_c     = cl_s.shift(1)
tr7_ = pd.concat([(hi_s-lo_s),(hi_s-prev_c).abs(),(lo_s-prev_c).abs()],axis=1).max(axis=1)
atr7 = tr7_.ewm(com=6, adjust=False).mean().values
kelt7_up = ema30 + 1.0*atr7;  kelt7_lo = ema30 - 1.0*atr7

sma30       = cl_s.rolling(30, min_periods=10).mean().values
price_std30 = cl_s.rolling(30, min_periods=10).std().fillna(0).values
roc7        = cl_s.pct_change(7).fillna(0).values
roc7_ma30   = pd.Series(roc7).rolling(30, min_periods=10).mean().values
roc7_std30  = pd.Series(roc7).rolling(30, min_periods=10).std().fillna(1e-9).values

vol_ma7  = vol_s.rolling(7,  min_periods=1).mean().values
obv_ma7  = pd.Series(OBV_).rolling(7,  min_periods=1).mean().values
obv_ma30 = pd.Series(OBV_).rolling(30, min_periods=1).mean().values
hi7 = hi_s.rolling(7, min_periods=1).max().values
lo7 = lo_s.rolling(7, min_periods=1).min().values
_ab7  = OBV_ > obv_ma7;  _ab30 = OBV_ > obv_ma30
obv_up7  = np.r_[False, _ab7[1:]  & ~_ab7[:-1]]
obv_dn7  = np.r_[False, ~_ab7[1:] &  _ab7[:-1]]
obv_up30 = np.r_[False, _ab30[1:]  & ~_ab30[:-1]]
obv_dn30 = np.r_[False, ~_ab30[1:] &  _ab30[:-1]]

sma7       = cl_s.rolling(7, min_periods=1).mean().values
bb_squeeze = df["bb_squeeze"].values
atr_pct_f  = ATR / (CL + 1e-9)
ma_sep     = np.abs(sma7 - sma30) / (sma30 + 1e-9)
regime_score = (
    (ADX_ > 20).astype(int) + (bb_squeeze > 0.01).astype(int) +
    (atr_pct_f > 0.01).astype(int) + (ma_sep > 0.02).astype(int)
)
is_trending_3 = regime_score >= 3;  is_trending_2 = regime_score >= 2
_sma7_ab  = sma7 > sma30
sma7_up_x = np.r_[False, _sma7_ab[1:] & ~_sma7_ab[:-1]]
sma7_dn_x = np.r_[False, ~_sma7_ab[1:] & _sma7_ab[:-1]]

q80 = cl_s.rolling(30, min_periods=15).quantile(0.80).values
q20 = cl_s.rolling(30, min_periods=15).quantile(0.20).values

print("  Computing OU z-scores …")
log_cl = np.log(CL + 1e-9)
def _ou_zscore(lc, window=30):
    n = len(lc); z = np.full(n, np.nan)
    for i in range(window, n):
        x = lc[i-window:i]; dy = x[1:]-x[:-1]; xx = x[:-1]
        xx_m = xx.mean(); sxx = np.dot(xx-xx_m, xx-xx_m)
        if sxx < 1e-12: continue
        beta = np.dot(xx-xx_m, dy-dy.mean()) / sxx
        if beta >= 0: continue
        alpha = dy.mean() - beta*xx_m
        resid = dy-(alpha+beta*xx); seq = resid.std()/np.sqrt(-2.0*beta)
        if seq < 1e-12: continue
        z[i] = (x[-1]-(-alpha/beta))/seq
    return z
ou_z = _ou_zscore(log_cl, window=30)

print("  Computing KAMA …")
def _kama_np(ca, period=30, fast=2, slow=30):
    sc_f = 2.0/(fast+1); sc_s = 2.0/(slow+1)
    k = ca.copy().astype(float)
    adc = np.r_[0.0, np.cumsum(np.abs(np.diff(ca)))]
    for i in range(period+1, len(ca)):
        d = abs(ca[i]-ca[i-period]); v = adc[i]-adc[i-period]
        er = d/v if v > 1e-12 else 0.0
        sc = (er*(sc_f-sc_s)+sc_s)**2
        k[i] = k[i-1]+sc*(ca[i]-k[i-1])
    return k
kama = _kama_np(CL, period=30)
ema7 = cl_s.ewm(span=7, adjust=False).mean().values
thrust = np.where(kama > 1e-9, (ema7-kama)/kama, 0.0)
th_s = pd.Series(thrust)
th_mean = th_s.rolling(7, min_periods=1).mean().values
th_std  = th_s.rolling(7, min_periods=1).std().fillna(0).values
th_upper = th_mean+1.0*th_std;  th_lower = th_mean-1.0*th_std
_tab_u = thrust > th_upper;  _tab_l = thrust < th_lower
th_cross_up = np.r_[False, _tab_u[1:] & ~_tab_u[:-1]]
th_cross_dn = np.r_[False, _tab_l[1:] & ~_tab_l[:-1]]
print("  Indicators ready.")

# ── Event factory — global _ev_atr swapped per config ────────────────────────
_ev_atr = ATR1H   # default; replaced with ATR4H for MTF configs

def _ev(i, direction):
    a = max(float(_ev_atr[i]), 1.0)
    e = float(CL[i])
    return dict(direction=direction, entry_i=i, entry_px=e,
                ref_lo=e-a, ref_hi=e+a, ref_size=a,
                atr_1h=a, year=int(yr_[i]), ts=IDX[i])

# ── IC engine ─────────────────────────────────────────────────────────────────
def compute_ic(events):
    if len(events) < 30: return 0.0, 1.0, len(events)
    sigs, fwds = [], []
    for ev in events:
        ei = ev["entry_i"]; end = min(ei+IC_HORIZON, N-1)
        fwd = (CL[end]-ev["entry_px"])/ev["entry_px"]*100
        sig = 1.0 if ev["direction"] == "long" else -1.0
        sigs.append(sig); fwds.append(sig*fwd)
    ic, p = st.spearmanr(sigs, fwds)
    return float(ic), float(p), len(events)

# ── Backtest engine ───────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_ts: pd.Timestamp; exit_ts: pd.Timestamp
    direction: int; entry_price: float; exit_price: float
    net_pnl: float; gross_pnl: float; total_fees: float
    exit_reason: str; year: int; window_id: int

def run_backtest(events, tp_frac, sl_frac, initial_capital=INIT_CAP, window_id=0):
    equity = float(initial_capital); trades = []
    for ev in events:
        if equity < initial_capital * 0.005: break
        entry = ev["entry_px"]; atr = ev["atr_1h"]
        d = 1 if ev["direction"] == "long" else -1
        tp_px = entry + d*tp_frac*atr
        sl_px = entry - d*sl_frac*atr
        sl_d  = max(abs(entry-sl_px), atr*MIN_SL_ATR)
        tp_d  = abs(tp_px-entry)
        if sl_d <= 0 or tp_d <= 0: continue
        qty  = min(equity*RISK_PCT/sl_d, equity*MAX_LEV/entry)
        notl = qty*entry; e_fee = notl*FEE
        ei = ev["entry_i"]; start = ei+1
        hit_tp = hit_sl = False
        exit_bar = min(start+MAX_HOLD, N-1); exit_px = float(CL[exit_bar])
        for k in range(start, min(start+MAX_HOLD, N)):
            bh, bl = float(HI[k]), float(LO[k])
            if d == 1:
                if bl <= sl_px: hit_sl=True; exit_px=sl_px; exit_bar=k; break
                if bh >= tp_px: hit_tp=True; exit_px=tp_px; exit_bar=k; break
            else:
                if bh >= sl_px: hit_sl=True; exit_px=sl_px; exit_bar=k; break
                if bl <= tp_px: hit_tp=True; exit_px=tp_px; exit_bar=k; break
        gross = d*qty*(exit_px-entry)
        net   = gross - e_fee - qty*exit_px*FEE
        equity += net
        trades.append(Trade(entry_ts=IDX[ei], exit_ts=IDX[exit_bar], direction=d,
                            entry_price=entry, exit_price=exit_px, gross_pnl=gross,
                            total_fees=e_fee+qty*exit_px*FEE, net_pnl=net,
                            exit_reason="tp" if hit_tp else ("sl" if hit_sl else "time"),
                            year=ev["year"], window_id=window_id))
    if not trades: return trades, pd.Series(dtype=float)
    eq = np.empty(len(trades)+1); eq[0] = initial_capital
    for k, t in enumerate(trades): eq[k+1] = eq[k]+t.net_pnl
    return trades, pd.Series(eq[1:], index=pd.DatetimeIndex([t.exit_ts for t in trades]))

def is_scan(events):
    paths = [(HI[e["entry_i"]+1:e["entry_i"]+1+MAX_HOLD],
              LO[e["entry_i"]+1:e["entry_i"]+1+MAX_HOLD]) for e in events]
    results = []
    for tp_f, sl_f in product(TP_FRAC_GRID, SL_FRAC_GRID):
        wins = losses = n_v = 0; sum_tp = sum_sl = 0.0
        for ev, (ph, pl) in zip(events, paths):
            entry = ev["entry_px"]; a = ev["atr_1h"]
            d = 1 if ev["direction"] == "long" else -1
            tp_px = entry+d*tp_f*a;  sl_px = entry-d*sl_f*a
            td = abs(tp_px-entry);   sd = abs(entry-sl_px)
            if sd <= 0 or td <= 0: continue
            n_v += 1; sum_tp += td/entry*100; sum_sl += sd/entry*100
            ht = hs = False
            for h, l in zip(ph, pl):
                if d == 1:
                    if l <= sl_px: hs=True; break
                    if h >= tp_px: ht=True; break
                else:
                    if h >= sl_px: hs=True; break
                    if l <= tp_px: ht=True; break
            if ht: wins += 1
            elif hs: losses += 1
        if n_v < 10: continue
        wr = wins/n_v*100; avg_tp = sum_tp/n_v; avg_sl = sum_sl/n_v
        rr = avg_tp/avg_sl if avg_sl > 0 else 0
        be     = 1/(1+rr)*100 if rr > 0 else 50.0
        be_fee = (avg_sl+FEE_RT_PCT)/(avg_tp+avg_sl)*100
        exp_adj = wr/100*(avg_tp-FEE_RT_PCT)-(1-wr/100)*(avg_sl+FEE_RT_PCT)
        pv = st.binomtest(int(round(wr/100*n_v)), n_v, be/100,
                          alternative="greater").pvalue
        results.append(dict(tp_frac=tp_f, sl_frac=sl_f, n=n_v, wr=round(wr,2),
                            rr=round(rr,2), be=round(be,2), be_fee=round(be_fee,2),
                            exp=round(exp_adj,5), p_val=round(pv,4)))
    df_r = pd.DataFrame(results)
    if df_r.empty:
        return df_r, dict(tp_frac=2.0,sl_frac=0.5,n=0,wr=0,rr=0,be=50,be_fee=50,exp=0,p_val=1)
    return df_r, df_r.sort_values("exp", ascending=False).iloc[0].to_dict()

def _kpis(eq, init=INIT_CAP):
    if eq.empty: return dict(total_return=0,calmar=0,sharpe=0,max_dd=0)
    full = pd.concat([pd.Series([init],index=[eq.index[0]-pd.Timedelta("1s")]),eq])
    dd   = (full/full.cummax()-1).min()
    ret  = full.iloc[-1]/init-1
    rets = full.pct_change().dropna()
    vol  = rets.std()*np.sqrt(365*24)
    ann  = rets.mean()*365*24
    return dict(total_return=ret, calmar=ret/abs(dd) if dd < 0 else 0,
                sharpe=ann/vol if vol > 0 else 0, max_dd=dd)

def run_wf(events, tp_frac, sl_frac):
    from dateutil.relativedelta import relativedelta
    start, end = IDX[0], IDX[-1]; wlist = []; cur = start
    while True:
        tr_e = cur+relativedelta(months=WF_TRAIN_M)
        oo_e = tr_e+relativedelta(months=WF_OOS_M)
        if oo_e > end: break
        wlist.append((cur,tr_e,oo_e)); cur = cur+relativedelta(months=WF_STEP_M)
    wr_list = []; all_oos = []
    for wid,(tr_s,tr_e,oo_e) in enumerate(wlist):
        is_ev = [e for e in events if tr_s <= e["ts"] < tr_e]
        oo_ev = [e for e in events if tr_e <= e["ts"] < oo_e]
        if len(is_ev) < 5 or len(oo_ev) < 3: continue
        _, oo_eq = run_backtest(oo_ev, tp_frac, sl_frac, window_id=wid)
        if oo_eq.empty: continue
        wr_list.append(_kpis(oo_eq)["total_return"]); all_oos.extend(oo_ev)
    oos = sorted(all_oos, key=lambda e: e["ts"])
    oos_t, oos_eq = run_backtest(oos, tp_frac, sl_frac)
    return oos_t, oos_eq, wr_list, len(wlist)

# ── Strategy collectors (use module-level _ev, which reads _ev_atr) ───────────
def S01_ichimoku():
    evs = []
    for i in range(50, N-MAX_HOLD-2):
        if _ev_atr[i] <= 0: continue
        sa,sb = senkou_a[i],senkou_b[i]
        if np.isnan(sa) or np.isnan(sb): continue
        tk,kj = tenkan[i],kijun[i]
        if np.isnan(tk) or np.isnan(kj): continue
        cl=CL[i]; ctp=max(sa,sb); cbt=min(sa,sb)
        if cl>ctp and tk>kj and cl>chi_hi14[i]:   evs.append(_ev(i,"long"))
        elif cl<cbt and tk<kj and cl<chi_lo14[i]: evs.append(_ev(i,"short"))
    return evs

def S02_keltner_breakout():
    evs = []
    for i in range(33, N-MAX_HOLD-2):
        if _ev_atr[i] <= 0: continue
        if CL[i-1]>kelt14_up[i-1] and CL[i-2]<=kelt14_up[i-2]: evs.append(_ev(i,"long"))
        elif CL[i-1]<kelt14_lo[i-1] and CL[i-2]>=kelt14_lo[i-2]: evs.append(_ev(i,"short"))
    return evs

def S03_keltner_rsi():
    evs = []
    for i in range(33, N-MAX_HOLD-2):
        if _ev_atr[i] <= 0: continue
        if CL[i]>kelt7_up[i] and RSI14[i]>30:   evs.append(_ev(i,"long"))
        elif CL[i]<kelt7_lo[i] and RSI14[i]<70: evs.append(_ev(i,"short"))
    return evs

def S04_momentum_ignition():
    evs = []
    for i in range(40, N-MAX_HOLD-2):
        if _ev_atr[i] <= 0: continue
        cl=CL[i]; ps=price_std30[i]
        if np.isnan(ps) or cl<1e-6 or ps/cl>=0.10: continue
        roc=roc7[i]; rm=roc7_ma30[i]; rs=roc7_std30[i]
        if np.isnan(rm) or rs<1e-12: continue
        sm=sma30[i]
        if np.isnan(sm): continue
        if roc>rm+rs and cl>sm:      evs.append(_ev(i,"long"))
        elif roc<rm-rs and cl<sm:    evs.append(_ev(i,"short"))
    return evs

def S05_obv_market_regime():
    evs = []
    for i in range(10, N-MAX_HOLD-2):
        if _ev_atr[i] <= 0: continue
        if not (VOL[i]>vol_ma7[i] and ADX_[i]>20): continue
        if obv_up7[i] and RSI14[i]<70 and CL[i]>hi7[i-1]:  evs.append(_ev(i,"long"))
        elif obv_dn7[i] and RSI14[i]>30 and CL[i]<lo7[i-1]:evs.append(_ev(i,"short"))
    return evs

def S06_obv_momentum():
    evs = []
    for i in range(32, N-MAX_HOLD-2):
        if _ev_atr[i]<=0 or VOL[i]<=vol_ma7[i]: continue
        if obv_up30[i] and RSI14[i]<70:   evs.append(_ev(i,"long"))
        elif obv_dn30[i] and RSI14[i]>30: evs.append(_ev(i,"short"))
    return evs

def S07_ou_mean_reversion():
    evs = []
    for i in range(35, N-MAX_HOLD-2):
        if _ev_atr[i] <= 0: continue
        z=ou_z[i]; sm=sma30[i]
        if np.isnan(z) or np.isnan(sm): continue
        if z<-1.0 and CL[i]>sm:   evs.append(_ev(i,"long"))
        elif z>1.0 and CL[i]<sm:  evs.append(_ev(i,"short"))
    return evs

def S08_quantile_channel():
    evs = []
    for i in range(32, N-MAX_HOLD-2):
        if _ev_atr[i] <= 0: continue
        qu=q80[i]; ql=q20[i]
        if np.isnan(qu) or np.isnan(ql) or ql<1e-6: continue
        if CL[i]>qu*1.01:      evs.append(_ev(i,"long"))
        elif CL[i]<ql/1.01:   evs.append(_ev(i,"short"))
    return evs

def S09_regime_trend(min_score=3):
    mask = is_trending_3 if min_score>=3 else is_trending_2
    evs  = []
    for i in range(32, N-MAX_HOLD-2):
        if _ev_atr[i]<=0 or not mask[i]: continue
        if sma7_up_x[i]:    evs.append(_ev(i,"long"))
        elif sma7_dn_x[i]:  evs.append(_ev(i,"short"))
    return evs

def S10_relative_momentum_accel():
    evs = []
    for i in range(45, N-MAX_HOLD-2):
        if _ev_atr[i]<=0 or np.isnan(kama[i]): continue
        if th_cross_up[i]:   evs.append(_ev(i,"long"))
        elif th_cross_dn[i]: evs.append(_ev(i,"short"))
    return evs

STRATEGIES = [
    ("S01","Ichimoku Cloud",          S01_ichimoku,
     "close>cloud+Tenkan>Kijun+chikou","IchimokuCloudStrategy.py"),
    ("S02","Keltner Breakout",         S02_keltner_breakout,
     "fresh close outside EMA30±ATR14","KeltnerBreakoutStrategy.py"),
    ("S03","Keltner + RSI",            S03_keltner_rsi,
     "close>EMA30+ATR7 AND RSI>30","KeltnerChannelRSIBreakoutStrategy.py"),
    ("S04","Momentum Ignition",        S04_momentum_ignition,
     "ROC7 > ROC_MA±1σ AND SMA30 trend","MomentumIgnitionStrategy.py"),
    ("S05","OBV Market Regime",        S05_obv_market_regime,
     "OBV cross MA7 + vol + ADX>20","OBVMarketRegimeStrategyBreakout.py"),
    ("S06","OBV Momentum",             S06_obv_momentum,
     "OBV cross MA30 + vol filter","OBVmomentumStrategy.py"),
    ("S07","OU Mean Reversion",        S07_ou_mean_reversion,
     "OU z-score <-1/>+1 + SMA30 align","OUMeanReversionStrategy.py"),
    ("S08","Quantile Channel",         S08_quantile_channel,
     "close > Q80×1.01  |  close < Q20÷1.01","QuantileChannelStrategy.py"),
    ("S09","Regime Trend (strict≥3)",  lambda: S09_regime_trend(3),
     "score≥3 + SMA7 cross SMA30","RegimeFilteredTrendStrategy.py"),
    ("S09b","Regime Trend (relaxed≥2)",lambda: S09_regime_trend(2),
     "score≥2 + SMA7 cross SMA30","RegimeFilteredTrendStrategy.py"),
    ("S10","Rel. Momentum Accel",      S10_relative_momentum_accel,
     "KAMA thrust breaks 7-bar BB","RelativeMomentumAccel.py"),
]

# ── 4H trend filter ───────────────────────────────────────────────────────────
def apply_trend_filter(evs):
    return [e for e in evs if (
        (e["direction"]=="long"  and bool(trend_long_4h[e["entry_i"]])) or
        (e["direction"]=="short" and bool(trend_short_4h[e["entry_i"]]))
    )]

# ── IC for both configs ───────────────────────────────────────────────────────
def _rk(x):
    ic = x["ic"] if x["ic"]==x["ic"] else 0.0
    p  = x["p"]  if x["p"]==x["p"]   else 1.0
    return (p,-ic) if ic>0 else (1.0+abs(ic),-ic)

base_ic   = []
filt_ic   = []

print(f"\n{SEP}")
print("  IC — BASE (1H ATR, no filter) vs MTF_FILT (4H ATR + 4H trend filter)")
print(SEP)
print(f"  {'Key':6s} {'Strategy':26s} {'BASE_n':>7s} {'BASE_IC':>8s} {'BASE_p':>7s}  "
      f"{'FILT_n':>7s} {'FILT_IC':>8s} {'FILT_p':>7s}")
print(f"  {'-'*6} {'-'*26} {'-'*7} {'-'*8} {'-'*7}  {'-'*7} {'-'*8} {'-'*7}")

_ev_atr = ATR1H   # BASE: 1H ATR
for key, label, collector, desc, src in STRATEGIES:
    evs  = collector()
    ic, p, n = compute_ic(evs)
    base_ic.append(dict(key=key,label=label,n=n,ic=round(ic,5),p=round(p,4),
                        events=evs,desc=desc,src=src))

_ev_atr = ATR4H   # MTF: 4H ATR
for (key,label,collector,desc,src), base_r in zip(STRATEGIES, base_ic):
    evs_raw = collector()
    evs  = apply_trend_filter(evs_raw)
    ic, p, n = compute_ic(evs)
    filt_ic.append(dict(key=key,label=label,n=n,ic=round(ic,5),p=round(p,4),
                        events=evs,desc=desc,src=src))
    b = base_r
    sig_b = "✅" if b["p"]<0.05 else ("~" if b["p"]<0.10 else "✗")
    sig_f = "✅" if p<0.05 else ("~" if p<0.10 else "✗")
    diff  = f"{ic-b['ic']:+.4f}"
    print(f"  [{key:4s}] {label:26s}  "
          f"{b['n']:7,} {b['ic']:+8.4f} {b['p']:7.4f}{sig_b}  "
          f"{n:7,} {ic:+8.4f} {p:7.4f}{sig_f}  Δ={diff}")

_ev_atr = ATR1H   # reset

ranked_base = sorted(base_ic, key=_rk)
ranked_filt = sorted(filt_ic, key=_rk)

print(f"\n{SEP}")
print("  MTF_FILT RANKING")
print(SEP)
for rank, r in enumerate(ranked_filt, 1):
    sig = "✅" if r["p"]<0.05 else ("~" if r["p"]<0.10 else "✗")
    print(f"  #{rank:2d}  [{r['key']:4s}] {r['label']:26s}  IC={r['ic']:+.4f}  p={r['p']:.4f}  {sig}")

# ── Full pipeline on MTF_FILT significant strategies ──────────────────────────
top_sig = [r for r in ranked_filt if r["ic"]>0 and r["p"]<0.05]
if not top_sig:
    top_sig = ranked_filt[:3]

pipeline_results = []
for cand in top_sig:
    print(f"\n{SEP2}")
    print(f"  PIPELINE: [{cand['key']}] {cand['label']}")
    print(f"  IC={cand['ic']:+.4f}  p={cand['p']:.4f}  n={cand['n']:,}  (4H ATR + trend filter)")
    print(SEP2)

    evs = cand["events"]
    print("  IS scan …")
    df_scan, bp = is_scan(evs)
    tp_f, sl_f  = bp["tp_frac"], bp["sl_frac"]
    print(f"  Best tp={tp_f}×ATR4H  sl={sl_f}×ATR4H  WR={bp['wr']:.1f}%  "
          f"BE={bp['be']:.1f}%  BE(fee-adj)={bp['be_fee']:.1f}%  "
          f"ExpPnL(adj)={bp['exp']:+.5f}%  p={bp['p_val']:.4f}")

    print("  Walk-Forward …")
    oos_t, oos_eq, wr_list, n_wins = run_wf(evs, tp_f, sl_f)
    kp = _kpis(oos_eq)
    oos_df = (pd.DataFrame([t.__dict__ for t in oos_t]).sort_values("entry_ts")
              if oos_t else pd.DataFrame())
    wins_n = int((oos_df["net_pnl"]>0).sum()) if not oos_df.empty else 0
    n_oos  = len(oos_df)
    wr_oos = wins_n/n_oos if n_oos else 0
    rr     = bp["rr"]
    be_oos = 1/(1+rr)*100 if rr>0 else 50.0
    binom_p = (st.binomtest(wins_n, max(n_oos,1), be_oos/100,
                            alternative="greater").pvalue if n_oos else 1)
    print(f"  OOS trades={n_oos}  Return={kp['total_return']:+.1%}  "
          f"MaxDD={kp['max_dd']:.1%}  WR={wr_oos:.1%}  BE≈{be_oos:.1f}%  "
          f"binom_p={binom_p:.4f}")

    print("  Monte Carlo …")
    mc = (run_monte_carlo(oos_df[["net_pnl","gross_pnl","total_fees"]],
                          INIT_CAP, N_SIMS)
          if n_oos > 5 else {})
    p_profit = float((mc["total_return"]>0).mean()) if mc else 0
    p_ruin   = float((mc["total_return"]<-0.5).mean()) if mc else 1
    print(f"  P(profit)={p_profit:.1%}  P(ruin)={p_ruin:.1%}")

    pipeline_results.append(dict(
        cand=cand, bp=bp, tp_f=tp_f, sl_f=sl_f,
        oos_t=oos_t, oos_eq=oos_eq, oos_df=oos_df,
        kp=kp, wr_list=wr_list, wins_n=wins_n, wr_oos=wr_oos,
        be_oos=be_oos, binom_p=binom_p, mc=mc,
        p_profit=p_profit, p_ruin=p_ruin,
    ))

# ── HTML report ───────────────────────────────────────────────────────────────
print(f"\n[HTML] Generating report …")

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()

def _imgt(b64):
    return (f'<img src="data:image/png;base64,{b64}"'
            f' style="width:100%;border-radius:6px;margin-bottom:12px">')

def _kv(lbl, val):
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:4px 0">'
            f'<span style="color:#9e9e9e">{lbl}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{val}</span></div>')

def _style(v, fmt=".1%", good=0):
    c = _GRN if v>good else _RED
    return f'<span style="color:{c}">{v:{fmt}}</span>'

def ic_color(ic, p):
    if p<0.05 and ic>0: return _GRN
    if p<0.10 and ic>0: return _YEL
    return _RED if ic<0 else "#9e9e9e"

# IC comparison chart
def plot_ic_compare():
    labels = [r["key"] for r in ranked_filt]
    ics_b  = [next(x["ic"] for x in base_ic if x["key"]==k) for k in labels]
    ics_f  = [r["ic"] for r in ranked_filt]
    ps_f   = [r["p"]  for r in ranked_filt]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(13,4))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    ax.bar(x-w/2, ics_b, width=w, label="BASE (1H ATR, no filter)",
           color="#42a5f5", alpha=0.6)
    clrs_f = [_GRN if p<0.05 else (_YEL if p<0.10 else _RED)
              for ic,p in zip(ics_f, ps_f)]
    ax.bar(x+w/2, ics_f, width=w, label="MTF_FILT (4H ATR + trend filter)",
           color=clrs_f, alpha=0.85)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, color=_TEXT)
    ax.set_ylabel("IC (Spearman)", color=_TEXT)
    ax.set_title("IC Comparison — BASE vs MTF_FILT  |  green=p<0.05", color=_TEXT)
    ax.legend(facecolor=_CARD, labelcolor=_TEXT, fontsize=9)
    fig.tight_layout(); return _fig_to_b64(fig)

# IC table
def _ic_table():
    ths = ["Key","Strategy","BASE n","BASE IC","BASE p","FILT n","FILT IC","FILT p","ΔIC"]
    th  = "".join(f'<th style="padding:5px 8px;text-align:right;color:{_ACC};'
                  f'border-bottom:1px solid {_GRID}">{h}</th>' for h in ths)
    rows = []
    for b, f in zip(base_ic, filt_ic):
        cb = ic_color(b["ic"],b["p"]); cf = ic_color(f["ic"],f["p"])
        delta = f["ic"]-b["ic"]
        dc    = _GRN if delta>0 else _RED
        sig_b = "✅" if b["p"]<0.05 else ("~" if b["p"]<0.10 else "✗")
        sig_f = "✅" if f["p"]<0.05 else ("~" if f["p"]<0.10 else "✗")
        cs = [f'<span style="color:{_ACC}">{b["key"]}</span>',
              b["label"],
              f'{b["n"]:,}',
              f'<span style="color:{cb}">{b["ic"]:+.4f}</span>',
              f'<span style="color:{cb}">{b["p"]:.4f} {sig_b}</span>',
              f'{f["n"]:,}',
              f'<span style="color:{cf}">{f["ic"]:+.4f}</span>',
              f'<span style="color:{cf}">{f["p"]:.4f} {sig_f}</span>',
              f'<span style="color:{dc}">{delta:+.4f}</span>']
        rows.append("<tr>"+"".join(
            f'<td style="padding:4px 8px;text-align:right;color:{_TEXT}">{x}</td>'
            for x in cs)+"</tr>")
    return (f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>')

def plot_equity(eq, label):
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,5),sharex=True,
                                  gridspec_kw={"height_ratios":[2,1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1,ax2):
        ax.set_facecolor(_CARD); ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values(): sp.set_color(_GRID)
    ax1.plot(eq.index, eq.values/INIT_CAP, color=_GRN, lw=1.5)
    ax1.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (norm.)", color=_TEXT)
    rm = eq.cummax(); dd = (eq/rm-1)*100
    ax2.fill_between(dd.index, dd.values, 0, color=_RED, alpha=0.4)
    ax2.set_ylabel("DD %", color=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle(f"OOS Equity — {label}", color=_TEXT); fig.tight_layout()
    return _fig_to_b64(fig)

def plot_wf_bars(wr_list, label):
    fig,ax = plt.subplots(figsize=(9,2.8))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    clrs = [_GRN if r>0 else _RED for r in wr_list]
    ax.bar(range(len(wr_list)),[r*100 for r in wr_list],color=clrs,alpha=0.8)
    ax.axhline(0,color=_GRID,lw=0.8)
    ax.set_title(f"WF OOS per-window — {label}",color=_TEXT)
    ax.set_ylabel("Return %",color=_TEXT); fig.tight_layout()
    return _fig_to_b64(fig)

def plot_mc(mc_res, label):
    fin = mc_res["total_return"]*100
    fig,ax = plt.subplots(figsize=(8,3))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    ax.hist(fin, bins=60, color=_ACC, alpha=0.7, edgecolor="none")
    ax.axvline(0, color=_RED, lw=1.5, ls="--")
    ax.axvline(float(np.median(fin)),color=_YEL,lw=1.5,
               label=f"Median {np.median(fin):.1f}%")
    ax.set_xlabel("Total Return %",color=_TEXT)
    ax.set_title(f"Monte Carlo (5 000 sims) — {label}",color=_TEXT)
    ax.legend(facecolor=_CARD,labelcolor=_TEXT); fig.tight_layout()
    return _fig_to_b64(fig)

img_cmp = plot_ic_compare()

def _pipeline_card(res):
    cand = res["cand"]; kp = res["kp"]
    ok = kp["total_return"]>0 and res["binom_p"]<0.05 and res["p_profit"]>0.5
    vc = _GRN if ok else _RED; vt = "✅ VALIDATED" if ok else "✗ NOT VALIDATED"
    mc_fin = pd.Series(res["mc"]["total_return"]*100) if res["mc"] else pd.Series([0])
    m = "".join([
        _kv("Strategy",      f'[{cand["key"]}] {cand["label"]}'),
        _kv("IC (Spearman)", f'{cand["ic"]:+.4f}  p={cand["p"]:.4f}  n={cand["n"]:,}'),
        _kv("Sizing",        "4H ATR (symmetric framework)"),
        _kv("Filter",        "4H trend alignment: long if close>EMA30_4H, short if below"),
        _kv("IS best params",f'tp={res["tp_f"]}×ATR4H  sl={res["sl_f"]}×ATR4H'),
        _kv("IS WR / BE / BE(fee)",
            f'{res["bp"]["wr"]:.1f}% / {res["bp"]["be"]:.1f}% / {res["bp"]["be_fee"]:.1f}%  '
            f'ExpPnL(adj)={res["bp"]["exp"]:+.5f}%'),
        _kv("OOS trades",    str(len(res["oos_t"]))),
        _kv("WR OOS / BE",   _style(res["wr_oos"],".1%")+f' / {res["be_oos"]:.1f}%'),
        _kv("Total Return OOS", _style(kp["total_return"],".1%")),
        _kv("Max Drawdown",  _style(kp["max_dd"],".1%")),
        _kv("Binom p",       f'{res["binom_p"]:.4f}'),
        _kv("MC P(profit)",  _style(res["p_profit"],".1%",good=0.5)),
        _kv("MC P(ruin<-50%)",f'{res["p_ruin"]:.1%}'),
        _kv("MC median",     f'{float(np.median(mc_fin)):.1f}%'),
        _kv("WF profit windows",
            f'{sum(1 for x in res["wr_list"] if x>0)} / {len(res["wr_list"])}'),
        _kv("Verdict",       f'<b style="color:{vc};font-size:14px">{vt}</b>'),
    ])
    eq_img = _imgt(plot_equity(res["oos_eq"],cand["label"])) if not res["oos_eq"].empty else ""
    wf_img = _imgt(plot_wf_bars(res["wr_list"],cand["label"]))
    mc_img = _imgt(plot_mc(res["mc"],cand["label"])) if res["mc"] else ""
    return (f'<div style="background:{_CARD};border:1px solid {_GRID};'
            f'border-radius:8px;padding:16px 20px;margin-bottom:20px">'
            f'<h3 style="color:{_ACC};margin:0 0 10px">'
            f'[{cand["key"]}] {cand["label"]}</h3>{m}</div>'
            f'{eq_img}{wf_img}{mc_img}')

pipeline_html = "\n".join(_pipeline_card(r) for r in pipeline_results)

# ATR comparison table
atr4h_median = float(np.nanmedian(ATR4H))
atr1h_median = float(np.nanmedian(ATR1H))
rows_be = []
for tp_f in [1.0,2.0,3.0,5.0]:
    for sl_f in [0.25,0.5,1.0]:
        tp1h = tp_f*atr1h_median/CL[-1]*100; sl1h = sl_f*atr1h_median/CL[-1]*100
        tp4h = tp_f*atr4h_median/CL[-1]*100; sl4h = sl_f*atr4h_median/CL[-1]*100
        be1h_fee = (sl1h+FEE_RT_PCT)/(tp1h+sl1h)*100 if (tp1h+sl1h)>0 else 50
        be4h_fee = (sl4h+FEE_RT_PCT)/(tp4h+sl4h)*100 if (tp4h+sl4h)>0 else 50
        c1 = _RED if be1h_fee>55 else (_YEL if be1h_fee>40 else _GRN)
        c4 = _RED if be4h_fee>55 else (_YEL if be4h_fee>40 else _GRN)
        rows_be.append(f"<tr>"
            f"<td style='padding:4px 8px;color:{_TEXT}'>{tp_f}×ATR / {sl_f}×ATR</td>"
            f"<td style='padding:4px 8px;color:{c1};text-align:right'>{be1h_fee:.1f}%</td>"
            f"<td style='padding:4px 8px;color:{c4};text-align:right'>{be4h_fee:.1f}%</td>"
            f"<td style='padding:4px 8px;color:{_GRN if be4h_fee<be1h_fee else _RED};text-align:right'>"
            f"{be4h_fee-be1h_fee:+.1f}pp</td></tr>")
be_table = (f'<table style="width:auto;border-collapse:collapse;font-size:12px;margin-bottom:16px">'
            f'<thead><tr>'
            f'<th style="padding:5px 8px;color:{_ACC}">TP / SL</th>'
            f'<th style="padding:5px 8px;color:{_ACC};text-align:right">BE_fee (1H ATR)</th>'
            f'<th style="padding:5px 8px;color:{_ACC};text-align:right">BE_fee (4H ATR)</th>'
            f'<th style="padding:5px 8px;color:{_ACC};text-align:right">Improvement</th>'
            f'</tr></thead><tbody>{"".join(rows_be)}</tbody></table>')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MTF Experiment — BTCUSDT 1H × 4H sizing</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{_BG};color:{_TEXT};font-family:monospace;padding:24px;line-height:1.5}}
  h1{{color:{_ACC};margin-bottom:4px;font-size:20px}}
  h2{{color:{_ACC};margin-top:32px;margin-bottom:10px;font-size:15px}}
  h3{{color:{_ACC};font-size:13px}}
  p{{margin-bottom:8px;color:#9e9e9e;font-size:13px}}
  a{{color:{_ACC}}}
  table{{width:100%;border-collapse:collapse;margin-bottom:16px}}
  th,td{{padding:4px 8px;vertical-align:top}}
  thead th{{color:{_ACC};border-bottom:1px solid {_GRID}}}
</style>
</head>
<body>
<h1>MTF Experiment — BTCUSDT 1H signals × 4H ATR sizing | 2020-2026</h1>
<p>Source: <a href="https://github.com/ali-azary/Algorithmic-Trading-From-Beginner-to-Advanced">
github.com/ali-azary/Algorithmic-Trading-From-Beginner-to-Advanced</a></p>
<p>Entry signals on 1H bars (unchanged). SL/TP sized with 4H ATR (~4× wider than 1H ATR).
4H trend-alignment filter: longs only if 1H close &gt; 4H EMA30, shorts only if below.</p>
<p>MAX_HOLD extended to 96 bars (4 days) to allow wider targets to resolve.</p>

<h2>Why 4H ATR sizing helps — fee-adjusted BE comparison</h2>
<p>ATR_1H ≈ {atr1h_median:.0f} USD ({atr1h_median/CL[-1]*100:.2f}% of price) &nbsp;|&nbsp;
ATR_4H ≈ {atr4h_median:.0f} USD ({atr4h_median/CL[-1]*100:.2f}% of price) &nbsp;|&nbsp;
Fee RT = {FEE_RT_PCT:.2f}%</p>
{be_table}

<h2>IC Comparison — BASE vs MTF_FILT</h2>
{_imgt(img_cmp)}
{_ic_table()}

<h2>Full Pipeline — MTF_FILT significant strategies (IC&gt;0, p&lt;0.05)</h2>
{pipeline_html}

<p style="color:#555;font-size:11px;margin-top:40px">
  Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC — BTCUSDT 1H 2020-2026 | MTF experiment
</p>
</body>
</html>"""

out = Path("reports/report_mtf.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n✅ Report saved: {out}  ({out.stat().st_size/1024:.0f} KB)")
print(SEP2)
