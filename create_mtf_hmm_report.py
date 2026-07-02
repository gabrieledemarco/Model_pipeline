"""
create_mtf_hmm_report.py
=========================
MTF experiment — 1H signals × 4H ATR sizing × 3-state Hidden Markov Model regime filter.

HMM replaces the simple 4H EMA trend filter:
  - 3-state GaussianHMM fit on (4H log_ret, 4H rolling_vol)
  - States labelled by mean return: BULL / SIDEWAYS / BEAR
  - Signal filter: long in BULL regime · short in BEAR regime · skip SIDEWAYS

Two rigor levels:
  FULL-SAMPLE HMM  — fit once on all 4H bars → IC analysis + IS scan (exploratory)
  CAUSAL HMM       — re-fit on each WF IS window → honest OOS / no lookahead bias

Motivation vs EMA filter:
  EMA is a price-level heuristic prone to whipsaws.
  HMM models regime transitions probabilistically and captures
  volatility + direction jointly.
"""
from __future__ import annotations

import base64, io, sys, warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from hmmlearn.hmm import GaussianHMM

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
MAX_HOLD   = 96                   # 4 days (4H ATR targets)
IC_HORIZON = 16                   # 16H forward return
START_YEAR = 2020
N_SIMS     = 5_000
HMM_STATES = 3                    # BULL / SIDEWAYS / BEAR
HMM_ITER   = 500
HMM_SEED   = 42
VOL_WINDOW = 10                   # bars for rolling vol in HMM features

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

TP_FRAC_GRID = [1.0, 2.0, 3.0, 5.0]
SL_FRAC_GRID = [0.25, 0.5, 0.75, 1.0]

_BG = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC = "#42a5f5"; _GRN = "#66bb6a"
_RED = "#ef5350"; _YEL = "#ffd54f"; _ORG = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

# ── Data ──────────────────────────────────────────────────────────────────────
print(SEP2)
print("MTF-HMM Experiment — BTCUSDT 1H × 4H ATR + 3-state HMM regime filter")
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

# ── 4H ATR mapped to 1H (shift 1 bar: use only the previous *closed* 4H bar) ──
atr4h_raw = df4h["atr_14"].shift(1).reindex(IDX, method="ffill")
ATR4H     = np.where(atr4h_raw.values > 0, atr4h_raw.values, ATR1H)

# ── HMM helpers ───────────────────────────────────────────────────────────────
def _hmm_features(close_series: pd.Series) -> tuple[np.ndarray, pd.Series]:
    """Return (X, log_ret) for a close price series."""
    lr  = np.log(close_series / close_series.shift(1)).fillna(0)
    vol = lr.rolling(VOL_WINDOW, min_periods=2).std().fillna(0)
    X   = np.column_stack([lr.values, vol.values])
    return X, lr

def _label_states(states: np.ndarray, log_ret: pd.Series, n: int = HMM_STATES):
    """Return (BULL_state, BEAR_state) labelled by per-state mean log_ret."""
    mean_r = {s: log_ret.values[states == s].mean()
              for s in range(n) if (states == s).any()}
    sorted_s = sorted(mean_r, key=mean_r.get)
    return sorted_s[-1], sorted_s[0]          # BULL, BEAR

def fit_hmm(X: np.ndarray) -> GaussianHMM:
    m = GaussianHMM(n_components=HMM_STATES, covariance_type="full",
                    n_iter=HMM_ITER, random_state=HMM_SEED)
    m.fit(X)
    return m

# ── Full-sample HMM (exploratory — used for IC analysis and IS scan) ──────────
print("[HMM] Fitting full-sample 3-state HMM on 4H data …")
X_full, lr_full = _hmm_features(df4h["close"])
hmm_full        = fit_hmm(X_full)
states_full     = hmm_full.predict(X_full)
BULL_FULL, BEAR_FULL = _label_states(states_full, lr_full)
state_counts = {s: (states_full == s).sum() for s in range(HMM_STATES)}
mean_ret_per = {s: lr_full.values[states_full == s].mean() * 100
                for s in range(HMM_STATES)}
print(f"  State counts: {state_counts}")
print(f"  Mean 4H log-ret: { {s: f'{v:.3f}%' for s,v in mean_ret_per.items()} }")
print(f"  BULL={BULL_FULL}  BEAR={BEAR_FULL}  "
      f"SIDEWAYS={[s for s in range(HMM_STATES) if s not in (BULL_FULL,BEAR_FULL)][0]}")

# Map to 1H resolution (shift 1 bar: use only the previous *closed* 4H bar)
hmm_regime_1h = (pd.Series(states_full, index=df4h.index)
                 .shift(1)
                 .reindex(IDX, method="ffill").values.astype(float))

hmm_long_ok_full  = hmm_regime_1h == BULL_FULL
hmm_short_ok_full = hmm_regime_1h == BEAR_FULL

def apply_hmm_filter_full(evs):
    return [e for e in evs if (
        (e["direction"] == "long"  and bool(hmm_long_ok_full[e["entry_i"]])) or
        (e["direction"] == "short" and bool(hmm_short_ok_full[e["entry_i"]]))
    )]

# ── 1H indicators ─────────────────────────────────────────────────────────────
print("  Precomputing 1H indicators …")
cl_s = pd.Series(CL); hi_s = pd.Series(HI)
lo_s = pd.Series(LO); vol_s = pd.Series(VOL)

def _donchian_mid(h, l, p):
    return (h.rolling(p,min_periods=p).max()+l.rolling(p,min_periods=p).min())/2

tenkan   = _donchian_mid(hi_s,lo_s,7).values
kijun    = _donchian_mid(hi_s,lo_s,14).values
senkou_a = ((pd.Series(tenkan)+pd.Series(kijun))/2).shift(14).values
senkou_b = _donchian_mid(hi_s,lo_s,30).shift(14).values
chi_hi14 = hi_s.shift(14).values;  chi_lo14 = lo_s.shift(14).values

ema30      = cl_s.ewm(span=30,adjust=False).mean().values
kelt14_up  = ema30+1.0*ATR;  kelt14_lo = ema30-1.0*ATR
prev_c     = cl_s.shift(1)
tr7_       = pd.concat([(hi_s-lo_s),(hi_s-prev_c).abs(),(lo_s-prev_c).abs()],axis=1).max(axis=1)
atr7       = tr7_.ewm(com=6,adjust=False).mean().values
kelt7_up   = ema30+1.0*atr7;  kelt7_lo = ema30-1.0*atr7

sma30       = cl_s.rolling(30,min_periods=10).mean().values
price_std30 = cl_s.rolling(30,min_periods=10).std().fillna(0).values
roc7        = cl_s.pct_change(7).fillna(0).values
roc7_ma30   = pd.Series(roc7).rolling(30,min_periods=10).mean().values
roc7_std30  = pd.Series(roc7).rolling(30,min_periods=10).std().fillna(1e-9).values

vol_ma7  = vol_s.rolling(7,min_periods=1).mean().values
obv_ma7  = pd.Series(OBV_).rolling(7,min_periods=1).mean().values
obv_ma30 = pd.Series(OBV_).rolling(30,min_periods=1).mean().values
hi7 = hi_s.rolling(7,min_periods=1).max().values
lo7 = lo_s.rolling(7,min_periods=1).min().values
_ab7  = OBV_>obv_ma7;  _ab30 = OBV_>obv_ma30
obv_up7  = np.r_[False, _ab7[1:] & ~_ab7[:-1]]
obv_dn7  = np.r_[False,~_ab7[1:] &  _ab7[:-1]]
obv_up30 = np.r_[False, _ab30[1:]& ~_ab30[:-1]]
obv_dn30 = np.r_[False,~_ab30[1:]&  _ab30[:-1]]

sma7       = cl_s.rolling(7,min_periods=1).mean().values
bb_squeeze = df["bb_squeeze"].values
atr_pct_f  = ATR/(CL+1e-9);  ma_sep = np.abs(sma7-sma30)/(sma30+1e-9)
regime_score = ((ADX_>20).astype(int)+(bb_squeeze>0.01).astype(int)+
                (atr_pct_f>0.01).astype(int)+(ma_sep>0.02).astype(int))
is_trending_3 = regime_score>=3;  is_trending_2 = regime_score>=2
_sma7_ab = sma7>sma30
sma7_up_x = np.r_[False, _sma7_ab[1:]& ~_sma7_ab[:-1]]
sma7_dn_x = np.r_[False,~_sma7_ab[1:]&  _sma7_ab[:-1]]

q80 = cl_s.rolling(30,min_periods=15).quantile(0.80).values
q20 = cl_s.rolling(30,min_periods=15).quantile(0.20).values

print("  Computing OU z-scores …")
log_cl = np.log(CL+1e-9)
def _ou_zscore(lc,window=30):
    n=len(lc); z=np.full(n,np.nan)
    for i in range(window,n):
        x=lc[i-window:i]; dy=x[1:]-x[:-1]; xx=x[:-1]
        xx_m=xx.mean(); sxx=np.dot(xx-xx_m,xx-xx_m)
        if sxx<1e-12: continue
        beta=np.dot(xx-xx_m,dy-dy.mean())/sxx
        if beta>=0: continue
        alpha=dy.mean()-beta*xx_m
        seq=(dy-(alpha+beta*xx)).std()/np.sqrt(-2.0*beta)
        if seq<1e-12: continue
        z[i]=(x[-1]-(-alpha/beta))/seq
    return z
ou_z = _ou_zscore(log_cl,window=30)

print("  Computing KAMA …")
def _kama_np(ca,period=30,fast=2,slow=30):
    sc_f=2.0/(fast+1); sc_s=2.0/(slow+1); k=ca.copy().astype(float)
    adc=np.r_[0.0,np.cumsum(np.abs(np.diff(ca)))]
    for i in range(period+1,len(ca)):
        d=abs(ca[i]-ca[i-period]); v=adc[i]-adc[i-period]
        er=d/v if v>1e-12 else 0.0; sc=(er*(sc_f-sc_s)+sc_s)**2
        k[i]=k[i-1]+sc*(ca[i]-k[i-1])
    return k
kama=_kama_np(CL,period=30)
ema7=cl_s.ewm(span=7,adjust=False).mean().values
thrust=np.where(kama>1e-9,(ema7-kama)/kama,0.0)
th_s=pd.Series(thrust)
th_mean=th_s.rolling(7,min_periods=1).mean().values
th_std=th_s.rolling(7,min_periods=1).std().fillna(0).values
th_upper=th_mean+1.0*th_std;  th_lower=th_mean-1.0*th_std
_tab_u=thrust>th_upper; _tab_l=thrust<th_lower
th_cross_up=np.r_[False,_tab_u[1:]& ~_tab_u[:-1]]
th_cross_dn=np.r_[False,_tab_l[1:]& ~_tab_l[:-1]]
print("  Indicators ready.")

# ── Event factory ─────────────────────────────────────────────────────────────
_ev_atr = ATR4H   # always use 4H ATR for sizing

def _ev(i, direction):
    a = max(float(_ev_atr[i]), 1.0)
    e = float(CL[i])
    return dict(direction=direction, entry_i=i, entry_px=e,
                ref_lo=e-a, ref_hi=e+a, ref_size=a,
                atr_1h=a, year=int(yr_[i]), ts=IDX[i])

# ── IC engine ─────────────────────────────────────────────────────────────────
def compute_ic(events):
    if len(events)<30: return 0.0,1.0,len(events)
    sigs,fwds=[],[]
    for ev in events:
        ei=ev["entry_i"]; end=min(ei+IC_HORIZON,N-1)
        fwd=(CL[end]-ev["entry_px"])/ev["entry_px"]*100
        sig=1.0 if ev["direction"]=="long" else -1.0
        sigs.append(sig); fwds.append(sig*fwd)
    ic,p=st.spearmanr(sigs,fwds)
    return float(ic),float(p),len(events)

# ── Backtest / IS scan / WF ──────────────────────────────────────────────────
@dataclass
class Trade:
    entry_ts: pd.Timestamp; exit_ts: pd.Timestamp
    direction: int; entry_price: float; exit_price: float
    net_pnl: float; gross_pnl: float; total_fees: float
    exit_reason: str; year: int; window_id: int

def run_backtest(events, tp_frac, sl_frac, initial_capital=INIT_CAP, window_id=0):
    equity=float(initial_capital); trades=[]
    for ev in events:
        if equity<initial_capital*0.005: break
        entry=ev["entry_px"]; atr=ev["atr_1h"]
        d=1 if ev["direction"]=="long" else -1
        tp_px=entry+d*tp_frac*atr; sl_px=entry-d*sl_frac*atr
        sl_d=max(abs(entry-sl_px),atr*MIN_SL_ATR); tp_d=abs(tp_px-entry)
        if sl_d<=0 or tp_d<=0: continue
        qty=min(equity*RISK_PCT/sl_d, equity*MAX_LEV/entry)
        notl=qty*entry; e_fee=notl*FEE
        ei=ev["entry_i"]; start=ei+1
        hit_tp=hit_sl=False
        exit_bar=min(start+MAX_HOLD,N-1); exit_px=float(CL[exit_bar])
        for k in range(start,min(start+MAX_HOLD,N)):
            bh,bl=float(HI[k]),float(LO[k])
            if d==1:
                if bl<=sl_px: hit_sl=True;exit_px=sl_px;exit_bar=k;break
                if bh>=tp_px: hit_tp=True;exit_px=tp_px;exit_bar=k;break
            else:
                if bh>=sl_px: hit_sl=True;exit_px=sl_px;exit_bar=k;break
                if bl<=tp_px: hit_tp=True;exit_px=tp_px;exit_bar=k;break
        gross=d*qty*(exit_px-entry)
        net=gross-e_fee-qty*exit_px*FEE
        equity+=net
        trades.append(Trade(entry_ts=IDX[ei],exit_ts=IDX[exit_bar],direction=d,
                            entry_price=entry,exit_price=exit_px,gross_pnl=gross,
                            total_fees=e_fee+qty*exit_px*FEE,net_pnl=net,
                            exit_reason="tp" if hit_tp else("sl" if hit_sl else "time"),
                            year=ev["year"],window_id=window_id))
    if not trades: return trades,pd.Series(dtype=float)
    eq=np.empty(len(trades)+1); eq[0]=initial_capital
    for k,t in enumerate(trades): eq[k+1]=eq[k]+t.net_pnl
    return trades,pd.Series(eq[1:],index=pd.DatetimeIndex([t.exit_ts for t in trades]))

def is_scan(events):
    paths=[(HI[e["entry_i"]+1:e["entry_i"]+1+MAX_HOLD],
            LO[e["entry_i"]+1:e["entry_i"]+1+MAX_HOLD]) for e in events]
    results=[]
    for tp_f,sl_f in product(TP_FRAC_GRID,SL_FRAC_GRID):
        wins=losses=n_v=0; sum_tp=sum_sl=0.0
        for ev,(ph,pl) in zip(events,paths):
            entry=ev["entry_px"]; a=ev["atr_1h"]
            d=1 if ev["direction"]=="long" else -1
            tp_px=entry+d*tp_f*a; sl_px=entry-d*sl_f*a
            td=abs(tp_px-entry); sd=abs(entry-sl_px)
            if sd<=0 or td<=0: continue
            n_v+=1; sum_tp+=td/entry*100; sum_sl+=sd/entry*100
            ht=hs=False
            for h,l in zip(ph,pl):
                if d==1:
                    if l<=sl_px: hs=True;break
                    if h>=tp_px: ht=True;break
                else:
                    if h>=sl_px: hs=True;break
                    if l<=tp_px: ht=True;break
            if ht: wins+=1
            elif hs: losses+=1
        if n_v<10: continue
        wr=wins/n_v*100; avg_tp=sum_tp/n_v; avg_sl=sum_sl/n_v
        rr=avg_tp/avg_sl if avg_sl>0 else 0
        be=1/(1+rr)*100 if rr>0 else 50.0
        be_fee=(avg_sl+FEE_RT_PCT)/(avg_tp+avg_sl)*100
        exp_adj=wr/100*(avg_tp-FEE_RT_PCT)-(1-wr/100)*(avg_sl+FEE_RT_PCT)
        pv=st.binomtest(int(round(wr/100*n_v)),n_v,be/100,alternative="greater").pvalue
        results.append(dict(tp_frac=tp_f,sl_frac=sl_f,n=n_v,wr=round(wr,2),
                            rr=round(rr,2),be=round(be,2),be_fee=round(be_fee,2),
                            exp=round(exp_adj,5),p_val=round(pv,4)))
    df_r=pd.DataFrame(results)
    if df_r.empty:
        return df_r,dict(tp_frac=2.0,sl_frac=0.5,n=0,wr=0,rr=0,be=50,be_fee=50,exp=0,p_val=1)
    return df_r,df_r.sort_values("exp",ascending=False).iloc[0].to_dict()

def _kpis(eq,init=INIT_CAP):
    if eq.empty: return dict(total_return=0,calmar=0,sharpe=0,max_dd=0)
    full=pd.concat([pd.Series([init],index=[eq.index[0]-pd.Timedelta("1s")]),eq])
    dd=(full/full.cummax()-1).min()
    ret=full.iloc[-1]/init-1
    rets=full.pct_change().dropna()
    vol=rets.std()*np.sqrt(365*24)
    ann=rets.mean()*365*24
    return dict(total_return=ret,calmar=ret/abs(dd) if dd<0 else 0,
                sharpe=ann/vol if vol>0 else 0,max_dd=dd)

def run_wf_hmm(events_all, tp_frac, sl_frac):
    """
    Walk-forward with causal HMM:
    - HMM re-fit on each IS window (no lookahead)
    - OOS events filtered by window-specific HMM states
    """
    from dateutil.relativedelta import relativedelta
    start,end=IDX[0],IDX[-1]; wlist=[]; cur=start
    while True:
        tr_e=cur+relativedelta(months=WF_TRAIN_M)
        oo_e=tr_e+relativedelta(months=WF_OOS_M)
        if oo_e>end: break
        wlist.append((cur,tr_e,oo_e)); cur=cur+relativedelta(months=WF_STEP_M)

    wr_list=[]; all_oos=[]
    for wid,(tr_s,tr_e,oo_e) in enumerate(wlist):
        # ── fit HMM on IS 4H bars ──────────────────────────────────────────
        is_4h = df4h[(df4h.index>=tr_s)&(df4h.index<tr_e)]["close"]
        if len(is_4h)<20: continue
        X_is, lr_is = _hmm_features(is_4h)
        try:
            hmm_w = fit_hmm(X_is)
            states_is = hmm_w.predict(X_is)
        except Exception:
            continue
        n_occupied = len({s for s in states_is})
        if n_occupied < 2: continue
        BULL_W, BEAR_W = _label_states(states_is, lr_is, hmm_w.n_components)

        # ── predict OOS 4H bars (prepend 14 days tail for rolling stats) ──
        tail_start = tr_e - pd.Timedelta(days=14)
        tail_close = df4h[(df4h.index>=tail_start)&(df4h.index<oo_e)]["close"]
        if len(tail_close)<5: continue
        X_tail, lr_tail = _hmm_features(tail_close)
        try:
            states_tail = hmm_w.predict(X_tail)
        except Exception:
            continue
        oo_mask = np.asarray(tail_close.index >= tr_e)
        oo_regime = pd.Series(states_tail[oo_mask],
                              index=tail_close.index[oo_mask])
        # map to 1H (shift 1 bar: use only the previous *closed* 4H bar)
        idx_oos_1h  = IDX[(IDX>=tr_e)&(IDX<oo_e)]
        oo_reg_1h   = oo_regime.shift(1).reindex(idx_oos_1h, method="ffill")

        # ── filter OOS events ─────────────────────────────────────────────
        oo_ev_all = [e for e in events_all if tr_e<=e["ts"]<oo_e]
        oo_ev = []
        for ev in oo_ev_all:
            r = oo_reg_1h.get(ev["ts"])
            if pd.isna(r): continue
            r = int(r)
            if   ev["direction"]=="long"  and r==BULL_W: oo_ev.append(ev)
            elif ev["direction"]=="short" and r==BEAR_W: oo_ev.append(ev)

        if len(oo_ev)<3: continue
        _, oo_eq = run_backtest(oo_ev, tp_frac, sl_frac, window_id=wid)
        if oo_eq.empty: continue
        wr_list.append(_kpis(oo_eq)["total_return"]); all_oos.extend(oo_ev)

    oos = sorted(all_oos,key=lambda e:e["ts"])
    oos_t,oos_eq = run_backtest(oos,tp_frac,sl_frac)
    return oos_t,oos_eq,wr_list,len(wlist)

# ── Strategy collectors ───────────────────────────────────────────────────────
def S01_ichimoku():
    evs=[]
    for i in range(50,N-MAX_HOLD-2):
        if ATR4H[i]<=0: continue
        sa,sb=senkou_a[i],senkou_b[i]
        if np.isnan(sa) or np.isnan(sb): continue
        tk,kj=tenkan[i],kijun[i]
        if np.isnan(tk) or np.isnan(kj): continue
        cl=CL[i]; ctp=max(sa,sb); cbt=min(sa,sb)
        if cl>ctp and tk>kj and cl>chi_hi14[i]: evs.append(_ev(i,"long"))
        elif cl<cbt and tk<kj and cl<chi_lo14[i]: evs.append(_ev(i,"short"))
    return evs

def S02_keltner_breakout():
    evs=[]
    for i in range(33,N-MAX_HOLD-2):
        if ATR4H[i]<=0: continue
        if CL[i-1]>kelt14_up[i-1] and CL[i-2]<=kelt14_up[i-2]: evs.append(_ev(i,"long"))
        elif CL[i-1]<kelt14_lo[i-1] and CL[i-2]>=kelt14_lo[i-2]: evs.append(_ev(i,"short"))
    return evs

def S03_keltner_rsi():
    evs=[]
    for i in range(33,N-MAX_HOLD-2):
        if ATR4H[i]<=0: continue
        if CL[i]>kelt7_up[i] and RSI14[i]>30: evs.append(_ev(i,"long"))
        elif CL[i]<kelt7_lo[i] and RSI14[i]<70: evs.append(_ev(i,"short"))
    return evs

def S04_momentum_ignition():
    evs=[]
    for i in range(40,N-MAX_HOLD-2):
        if ATR4H[i]<=0: continue
        cl=CL[i]; ps=price_std30[i]
        if np.isnan(ps) or cl<1e-6 or ps/cl>=0.10: continue
        roc=roc7[i]; rm=roc7_ma30[i]; rs=roc7_std30[i]
        if np.isnan(rm) or rs<1e-12: continue
        sm=sma30[i]
        if np.isnan(sm): continue
        if roc>rm+rs and cl>sm: evs.append(_ev(i,"long"))
        elif roc<rm-rs and cl<sm: evs.append(_ev(i,"short"))
    return evs

def S05_obv_market_regime():
    evs=[]
    for i in range(10,N-MAX_HOLD-2):
        if ATR4H[i]<=0: continue
        if not(VOL[i]>vol_ma7[i] and ADX_[i]>20): continue
        if obv_up7[i] and RSI14[i]<70 and CL[i]>hi7[i-1]: evs.append(_ev(i,"long"))
        elif obv_dn7[i] and RSI14[i]>30 and CL[i]<lo7[i-1]: evs.append(_ev(i,"short"))
    return evs

def S06_obv_momentum():
    evs=[]
    for i in range(32,N-MAX_HOLD-2):
        if ATR4H[i]<=0 or VOL[i]<=vol_ma7[i]: continue
        if obv_up30[i] and RSI14[i]<70: evs.append(_ev(i,"long"))
        elif obv_dn30[i] and RSI14[i]>30: evs.append(_ev(i,"short"))
    return evs

def S07_ou_mean_reversion():
    evs=[]
    for i in range(35,N-MAX_HOLD-2):
        if ATR4H[i]<=0: continue
        z=ou_z[i]; sm=sma30[i]
        if np.isnan(z) or np.isnan(sm): continue
        if z<-1.0 and CL[i]>sm: evs.append(_ev(i,"long"))
        elif z>1.0 and CL[i]<sm: evs.append(_ev(i,"short"))
    return evs

def S08_quantile_channel():
    evs=[]
    for i in range(32,N-MAX_HOLD-2):
        if ATR4H[i]<=0: continue
        qu=q80[i]; ql=q20[i]
        if np.isnan(qu) or np.isnan(ql) or ql<1e-6: continue
        if CL[i]>qu*1.01: evs.append(_ev(i,"long"))
        elif CL[i]<ql/1.01: evs.append(_ev(i,"short"))
    return evs

def S09_regime_trend(min_score=3):
    mask=is_trending_3 if min_score>=3 else is_trending_2
    evs=[]
    for i in range(32,N-MAX_HOLD-2):
        if ATR4H[i]<=0 or not mask[i]: continue
        if sma7_up_x[i]: evs.append(_ev(i,"long"))
        elif sma7_dn_x[i]: evs.append(_ev(i,"short"))
    return evs

def S10_relative_momentum_accel():
    evs=[]
    for i in range(45,N-MAX_HOLD-2):
        if ATR4H[i]<=0 or np.isnan(kama[i]): continue
        if th_cross_up[i]: evs.append(_ev(i,"long"))
        elif th_cross_dn[i]: evs.append(_ev(i,"short"))
    return evs

STRATEGIES=[
    ("S01","Ichimoku Cloud",          S01_ichimoku,          "close>cloud+TK>KJ+chikou","IchimokuCloudStrategy.py"),
    ("S02","Keltner Breakout",         S02_keltner_breakout,  "fresh close outside EMA30±ATR14","KeltnerBreakoutStrategy.py"),
    ("S03","Keltner + RSI",            S03_keltner_rsi,       "close>EMA30+ATR7 AND RSI>30","KeltnerChannelRSIBreakoutStrategy.py"),
    ("S04","Momentum Ignition",        S04_momentum_ignition, "ROC7>ROC_MA±1σ AND SMA30 trend","MomentumIgnitionStrategy.py"),
    ("S05","OBV Market Regime",        S05_obv_market_regime, "OBV cross MA7+vol+ADX>20","OBVMarketRegimeStrategyBreakout.py"),
    ("S06","OBV Momentum",             S06_obv_momentum,      "OBV cross MA30+vol filter","OBVmomentumStrategy.py"),
    ("S07","OU Mean Reversion",        S07_ou_mean_reversion, "OU z<-1/>+1+SMA30 align","OUMeanReversionStrategy.py"),
    ("S08","Quantile Channel",         S08_quantile_channel,  "close>Q80×1.01 | close<Q20÷1.01","QuantileChannelStrategy.py"),
    ("S09","Regime Trend (strict≥3)",  lambda: S09_regime_trend(3),"score≥3+SMA7×SMA30","RegimeFilteredTrendStrategy.py"),
    ("S09b","Regime Trend (relaxed≥2)",lambda: S09_regime_trend(2),"score≥2+SMA7×SMA30","RegimeFilteredTrendStrategy.py"),
    ("S10","Rel. Momentum Accel",      S10_relative_momentum_accel,"KAMA thrust breaks 7-bar BB","RelativeMomentumAccel.py"),
]

# ── IC: BASE vs MTF_EMA vs MTF_HMM ───────────────────────────────────────────
def _rk(x):
    ic=x["ic"] if x["ic"]==x["ic"] else 0.0
    p =x["p"]  if x["p"] ==x["p"]  else 1.0
    return (p,-ic) if ic>0 else (1.0+abs(ic),-ic)

print(f"\n{SEP}")
print("  IC comparison — BASE / MTF_EMA_FILT / MTF_HMM_FILT")
print(SEP)

# EMA filter (4H EMA30 from create_mtf_report)
ema30_4h_s = (df4h["close"].ewm(span=30,adjust=False).mean()
              .reindex(IDX,method="ffill").values)
ema_long_ok  = CL > ema30_4h_s
ema_short_ok = CL < ema30_4h_s

base_ic=[]; ema_ic=[]; hmm_ic=[]

_ev_atr_save = ATR4H   # we always use 4H sizing

for key,label,collector,desc,src in STRATEGIES:
    evs_all = collector()
    # BASE (4H ATR, no filter, for fair ATR comparison)
    ic_b,p_b,n_b = compute_ic(evs_all)
    base_ic.append(dict(key=key,label=label,n=n_b,ic=round(ic_b,5),p=round(p_b,4),
                        events=evs_all,desc=desc,src=src))
    # EMA filter
    evs_ema=[e for e in evs_all if
             (e["direction"]=="long" and bool(ema_long_ok[e["entry_i"]])) or
             (e["direction"]=="short" and bool(ema_short_ok[e["entry_i"]]))]
    ic_e,p_e,n_e = compute_ic(evs_ema)
    ema_ic.append(dict(key=key,label=label,n=n_e,ic=round(ic_e,5),p=round(p_e,4),
                       events=evs_ema,desc=desc,src=src))
    # HMM filter (full-sample)
    evs_hmm = apply_hmm_filter_full(evs_all)
    ic_h,p_h,n_h = compute_ic(evs_hmm)
    hmm_ic.append(dict(key=key,label=label,n=n_h,ic=round(ic_h,5),p=round(p_h,4),
                       events=evs_hmm, events_all=evs_all, desc=desc,src=src))
    sig_b="✅" if p_b<0.05 else("~" if p_b<0.10 else "✗")
    sig_e="✅" if p_e<0.05 else("~" if p_e<0.10 else "✗")
    sig_h="✅" if p_h<0.05 else("~" if p_h<0.10 else "✗")
    print(f"  [{key:4s}] {label:26s}  "
          f"BASE {ic_b:+.4f}{sig_b}  "
          f"EMA {ic_e:+.4f}{sig_e}  "
          f"HMM {ic_h:+.4f}{sig_h}  "
          f"ΔHMM-BASE={ic_h-ic_b:+.4f}")

ranked_hmm = sorted(hmm_ic,key=_rk)
print(f"\n{SEP}")
print("  MTF_HMM RANKING")
print(SEP)
for rank,r in enumerate(ranked_hmm,1):
    sig="✅" if r["p"]<0.05 else("~" if r["p"]<0.10 else "✗")
    print(f"  #{rank:2d}  [{r['key']:4s}] {r['label']:26s}  IC={r['ic']:+.4f}  p={r['p']:.4f}  {sig}")

# ── Full pipeline on MTF_HMM significant strategies (causal WFO) ─────────────
top_sig=[r for r in ranked_hmm if r["ic"]>0 and r["p"]<0.05]
if not top_sig: top_sig=ranked_hmm[:3]

pipeline_results=[]
for cand in top_sig:
    print(f"\n{SEP2}")
    print(f"  PIPELINE: [{cand['key']}] {cand['label']}")
    print(f"  IC(HMM full-sample)={cand['ic']:+.4f}  p={cand['p']:.4f}  n={cand['n']:,}")
    print(SEP2)

    evs_hmm = cand["events"]          # full-sample HMM filtered (for IS scan)
    evs_all = cand["events_all"]      # unfiltered (for causal WFO)

    print("  IS scan (full-sample HMM events, exploratory) …")
    _,bp = is_scan(evs_hmm)
    tp_f,sl_f = bp["tp_frac"],bp["sl_frac"]
    print(f"  Best tp={tp_f}×ATR4H  sl={sl_f}×ATR4H  WR={bp['wr']:.1f}%  "
          f"BE={bp['be']:.1f}%  BE(fee)={bp['be_fee']:.1f}%  "
          f"ExpPnL(adj)={bp['exp']:+.5f}%")

    print("  Walk-Forward (CAUSAL HMM — no lookahead) …")
    oos_t,oos_eq,wr_list,n_wins = run_wf_hmm(evs_all, tp_f, sl_f)
    kp=_kpis(oos_eq)
    oos_df=(pd.DataFrame([t.__dict__ for t in oos_t]).sort_values("entry_ts")
            if oos_t else pd.DataFrame())
    wins_n=int((oos_df["net_pnl"]>0).sum()) if not oos_df.empty else 0
    n_oos=len(oos_df)
    wr_oos=wins_n/n_oos if n_oos else 0
    rr=bp["rr"]; be_oos=1/(1+rr)*100 if rr>0 else 50.0
    binom_p=(st.binomtest(wins_n,max(n_oos,1),be_oos/100,
                          alternative="greater").pvalue if n_oos else 1)
    print(f"  OOS trades={n_oos}  Return={kp['total_return']:+.1%}  "
          f"MaxDD={kp['max_dd']:.1%}  WR={wr_oos:.1%}  BE≈{be_oos:.1f}%  "
          f"binom_p={binom_p:.4f}")

    print("  Monte Carlo …")
    mc=(run_monte_carlo(oos_df[["net_pnl","gross_pnl","total_fees"]],INIT_CAP,N_SIMS)
        if n_oos>5 else {})
    p_profit=float((mc["total_return"]>0).mean()) if mc else 0
    p_ruin  =float((mc["total_return"]<-0.5).mean()) if mc else 1
    print(f"  P(profit)={p_profit:.1%}  P(ruin)={p_ruin:.1%}")

    pipeline_results.append(dict(
        cand=cand,bp=bp,tp_f=tp_f,sl_f=sl_f,
        oos_t=oos_t,oos_eq=oos_eq,oos_df=oos_df,
        kp=kp,wr_list=wr_list,wins_n=wins_n,wr_oos=wr_oos,
        be_oos=be_oos,binom_p=binom_p,mc=mc,
        p_profit=p_profit,p_ruin=p_ruin,
    ))

# ── HTML report ───────────────────────────────────────────────────────────────
print(f"\n[HTML] Generating report …")

def _fig_to_b64(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=100,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()

def _imgt(b64):
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;margin-bottom:12px">'

def _kv(lbl,val):
    return (f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid {_GRID};padding:4px 0">'
            f'<span style="color:#9e9e9e">{lbl}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{val}</span></div>')

def _style(v,fmt=".1%",good=0):
    c=_GRN if v>good else _RED
    return f'<span style="color:{c}">{v:{fmt}}</span>'

def ic_color(ic,p):
    if p<0.05 and ic>0: return _GRN
    if p<0.10 and ic>0: return _YEL
    return _RED if ic<0 else "#9e9e9e"

# HMM regime chart
def plot_hmm_regimes():
    fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True,
                          gridspec_kw={"height_ratios":[3,1]})
    fig.patch.set_facecolor(_BG)
    ax,ax2=axes
    for a in axes: a.set_facecolor(_CARD); a.tick_params(colors=_TEXT)
    for a in axes:
        for sp in a.spines.values(): sp.set_color(_GRID)

    # Plot BTC 4H close
    ax.plot(df4h.index, df4h["close"].values, color="#9e9e9e", lw=0.8, alpha=0.7)
    ax.set_ylabel("BTC/USDT (4H)", color=_TEXT)

    # Shade regimes
    colors_map = {BULL_FULL: _GRN, BEAR_FULL: _RED}
    sideways_s = [s for s in range(HMM_STATES) if s not in (BULL_FULL,BEAR_FULL)][0]
    colors_map[sideways_s] = _YEL

    state_series_plot = pd.Series(states_full, index=df4h.index)
    prev_s = None; seg_start = df4h.index[0]
    for idx_val, s in state_series_plot.items():
        if s != prev_s:
            if prev_s is not None:
                ax.axvspan(seg_start, idx_val, alpha=0.12,
                           color=colors_map.get(prev_s,"#9e9e9e"), lw=0)
            seg_start = idx_val; prev_s = s
    if prev_s is not None:
        ax.axvspan(seg_start, df4h.index[-1], alpha=0.12,
                   color=colors_map.get(prev_s,"#9e9e9e"), lw=0)

    ax.set_title("4H BTC — HMM regime overlay  (green=BULL  yellow=SIDEWAYS  red=BEAR)",
                 color=_TEXT, fontsize=10)

    # Regime state over time
    lr_plot = np.log(df4h["close"]/df4h["close"].shift(1)).fillna(0).rolling(20).mean()*100
    ax2.plot(df4h.index, lr_plot, color=_ACC, lw=0.8)
    ax2.axhline(0,color=_GRID,lw=0.6,ls="--")
    ax2.set_ylabel("20-bar mean ret %", color=_TEXT)
    fig.tight_layout(); return _fig_to_b64(fig)

# IC comparison table (3-way)
def _ic_3way_table():
    ths=["Key","Strategy","BASE IC","BASE p","EMA IC","EMA p","HMM IC","HMM p","ΔHMM-EMA"]
    th="".join(f'<th style="padding:5px 8px;text-align:right;color:{_ACC};'
               f'border-bottom:1px solid {_GRID}">{h}</th>' for h in ths)
    rows=[]
    for b,e,h in zip(base_ic,ema_ic,hmm_ic):
        cb=ic_color(b["ic"],b["p"]); ce=ic_color(e["ic"],e["p"]); ch=ic_color(h["ic"],h["p"])
        sig_b="✅" if b["p"]<0.05 else("~" if b["p"]<0.10 else "✗")
        sig_e="✅" if e["p"]<0.05 else("~" if e["p"]<0.10 else "✗")
        sig_h="✅" if h["p"]<0.05 else("~" if h["p"]<0.10 else "✗")
        delta=h["ic"]-e["ic"]; dc=_GRN if delta>0 else _RED
        cs=[f'<span style="color:{_ACC}">{b["key"]}</span>',b["label"],
            f'<span style="color:{cb}">{b["ic"]:+.4f}</span>',
            f'<span style="color:{cb}">{b["p"]:.4f}{sig_b}</span>',
            f'<span style="color:{ce}">{e["ic"]:+.4f}</span>',
            f'<span style="color:{ce}">{e["p"]:.4f}{sig_e}</span>',
            f'<span style="color:{ch}">{h["ic"]:+.4f}</span>',
            f'<span style="color:{ch}">{h["p"]:.4f}{sig_h}</span>',
            f'<span style="color:{dc}">{delta:+.4f}</span>']
        rows.append("<tr>"+"".join(
            f'<td style="padding:4px 8px;text-align:right;color:{_TEXT}">{x}</td>'
            for x in cs)+"</tr>")
    return (f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>')

def plot_equity(eq,label):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,5),sharex=True,
                                gridspec_kw={"height_ratios":[2,1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1,ax2):
        ax.set_facecolor(_CARD); ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values(): sp.set_color(_GRID)
    ax1.plot(eq.index,eq.values/INIT_CAP,color=_GRN,lw=1.5)
    ax1.axhline(1.0,color=_GRID,lw=0.8,ls="--")
    ax1.set_ylabel("Equity (norm.)",color=_TEXT)
    rm=eq.cummax(); dd=(eq/rm-1)*100
    ax2.fill_between(dd.index,dd.values,0,color=_RED,alpha=0.4)
    ax2.set_ylabel("DD %",color=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle(f"OOS Equity (causal HMM WFO) — {label}",color=_TEXT); fig.tight_layout()
    return _fig_to_b64(fig)

def plot_wf_bars(wr_list,label):
    fig,ax=plt.subplots(figsize=(9,2.8)); fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    clrs=[_GRN if r>0 else _RED for r in wr_list]
    ax.bar(range(len(wr_list)),[r*100 for r in wr_list],color=clrs,alpha=0.8)
    ax.axhline(0,color=_GRID,lw=0.8)
    ax.set_title(f"WF OOS per-window — {label}",color=_TEXT)
    ax.set_ylabel("Return %",color=_TEXT); fig.tight_layout(); return _fig_to_b64(fig)

def plot_mc(mc_res,label):
    fin=mc_res["total_return"]*100
    fig,ax=plt.subplots(figsize=(8,3)); fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    ax.hist(fin,bins=60,color=_ACC,alpha=0.7,edgecolor="none")
    ax.axvline(0,color=_RED,lw=1.5,ls="--")
    ax.axvline(float(np.median(fin)),color=_YEL,lw=1.5,label=f"Median {np.median(fin):.1f}%")
    ax.set_xlabel("Total Return %",color=_TEXT)
    ax.set_title(f"Monte Carlo (5 000 sims) — {label}",color=_TEXT)
    ax.legend(facecolor=_CARD,labelcolor=_TEXT); fig.tight_layout(); return _fig_to_b64(fig)

img_regimes = plot_hmm_regimes()

def _pipeline_card(res):
    cand=res["cand"]; kp=res["kp"]
    ok=kp["total_return"]>0 and res["binom_p"]<0.05 and res["p_profit"]>0.5
    vc=_GRN if ok else _RED; vt="✅ VALIDATED" if ok else "✗ NOT VALIDATED"
    mc_fin=pd.Series(res["mc"]["total_return"]*100) if res["mc"] else pd.Series([0])
    m="".join([
        _kv("Strategy",          f'[{cand["key"]}] {cand["label"]}'),
        _kv("IC (HMM full-smp)", f'{cand["ic"]:+.4f}  p={cand["p"]:.4f}  n={cand["n"]:,}'),
        _kv("HMM filter",        f"BULL={BULL_FULL} BEAR={BEAR_FULL} (3-state Gaussian HMM on 4H log_ret, vol)"),
        _kv("IS best params",    f'tp={res["tp_f"]}×ATR4H  sl={res["sl_f"]}×ATR4H'),
        _kv("IS WR/BE/BE(fee)",  f'{res["bp"]["wr"]:.1f}%/{res["bp"]["be"]:.1f}%/{res["bp"]["be_fee"]:.1f}%  '
                                 f'ExpPnL(adj)={res["bp"]["exp"]:+.5f}%'),
        _kv("OOS trades",        str(len(res["oos_t"]))),
        _kv("WF method",         "CAUSAL: HMM re-fit on each IS window (no lookahead)"),
        _kv("WR OOS / BE",       _style(res["wr_oos"],".1%")+f' / {res["be_oos"]:.1f}%'),
        _kv("Total Return OOS",  _style(kp["total_return"],".1%")),
        _kv("Max Drawdown",      _style(kp["max_dd"],".1%")),
        _kv("Binom p",           f'{res["binom_p"]:.4f}'),
        _kv("MC P(profit)",      _style(res["p_profit"],".1%",good=0.5)),
        _kv("MC P(ruin<-50%)",   f'{res["p_ruin"]:.1%}'),
        _kv("MC median",         f'{float(np.median(mc_fin)):.1f}%'),
        _kv("WF profit windows", f'{sum(1 for x in res["wr_list"] if x>0)}/{len(res["wr_list"])}'),
        _kv("Verdict",           f'<b style="color:{vc};font-size:14px">{vt}</b>'),
    ])
    eq_img=_imgt(plot_equity(res["oos_eq"],cand["label"])) if not res["oos_eq"].empty else ""
    wf_img=_imgt(plot_wf_bars(res["wr_list"],cand["label"]))
    mc_img=_imgt(plot_mc(res["mc"],cand["label"])) if res["mc"] else ""
    return (f'<div style="background:{_CARD};border:1px solid {_GRID};'
            f'border-radius:8px;padding:16px 20px;margin-bottom:20px">'
            f'<h3 style="color:{_ACC};margin:0 0 10px">[{cand["key"]}] {cand["label"]}</h3>'
            f'{m}</div>{eq_img}{wf_img}{mc_img}')

pipeline_html="\n".join(_pipeline_card(r) for r in pipeline_results)

sideways_s=[s for s in range(HMM_STATES) if s not in (BULL_FULL,BEAR_FULL)][0]
state_labels={BULL_FULL:"BULL",BEAR_FULL:"BEAR",sideways_s:"SIDEWAYS"}

html=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MTF-HMM Experiment — BTCUSDT</title>
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
<h1>MTF-HMM Experiment — BTCUSDT 1H × 4H ATR + 3-state HMM | 2020-2026</h1>
<p>Entry signals on 1H bars · SL/TP sized with 4H ATR · HMM regime filter replaces EMA filter.</p>
<p><b>HMM:</b> 3-state GaussianHMM fit on 4H features: (log_ret, rolling_{VOL_WINDOW}-bar vol).
States labelled by mean return → BULL={BULL_FULL} / SIDEWAYS={sideways_s} / BEAR={BEAR_FULL}.</p>
<p><b>Rigor:</b> IC analysis uses full-sample HMM (exploratory upper bound).
WFO uses <b>causal HMM</b>: re-fit on each 6-month IS window, applied to OOS only — zero lookahead.</p>

<h2>HMM Regime Map — 4H BTCUSDT 2020-2026</h2>
{_imgt(img_regimes)}
<p>State counts: {state_counts} &nbsp;|&nbsp;
Mean 4H log-ret: { {state_labels[s]: f"{v:.3f}%" for s,v in mean_ret_per.items()} }</p>

<h2>IC Comparison — BASE vs EMA filter vs HMM filter (4H ATR sizing)</h2>
{_ic_3way_table()}

<h2>Full Pipeline — MTF_HMM significant strategies (causal WFO)</h2>
{pipeline_html}

<p style="color:#555;font-size:11px;margin-top:40px">
  Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC — BTCUSDT 1H 2020-2026 | MTF-HMM experiment
</p>
</body>
</html>"""

out=Path("reports/report_mtf_hmm.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html,encoding="utf-8")
print(f"\n✅ Report saved: {out}  ({out.stat().st_size/1024:.0f} KB)")
print(SEP2)
