#!/usr/bin/env python3
"""
create_intraday_strategy_report.py
====================================
Intraday BTC/USDT Perpetual Futures — 1H bars
Groups: Z-score MR · ORB · VWAP-MR · Session Momentum · RSI(4) · Funding · Volume · Combo

IC test : Spearman 8H forward log-return
WFO     : 6m IS / 2m OOS / 2m step-over
MC      : N=5,000 bootstrap
Criterion: OOS ret > 0 AND MC p_profit > 0.90 AND MC p_ruin < 0.05
"""
from __future__ import annotations

import base64, io, sys, warnings
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

from src.strategy.data_fetcher import (
    fetch_extended_data,
    fetch_binance_vision_funding,
)
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import run_monte_carlo

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP   = 100_000.0
RISK_PCT   = 0.01
FEE        = 0.0004
FEE_RT_PCT = FEE * 2 * 100
MAX_LEV    = 5.0
IC_HORIZON = 8        # 8H forward return for intraday IC test
START_YEAR = 2020
N_SIMS     = 5_000
COOLDOWN   = 4        # 4H minimum between trades
MAX_HOLD   = 24       # 24H max hold
WARMUP     = 200

WF_TRAIN_M = 6
WF_OOS_M   = 2
WF_STEP_M  = 2

TP_GRID = [1.0, 2.0, 3.0, 5.0]
SL_GRID = [0.25, 0.5, 0.75, 1.0]

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("Intraday BTC Strategy Report — 1H bars")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h  = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
print(f"  1H: {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

CL   = df1h["close"].values.astype(float)
HI   = df1h["high"].values.astype(float)
LO   = df1h["low"].values.astype(float)
VOL  = df1h["volume"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0,
                df1h["atr_14"].shift(1).values, 1.0)

CL_s  = pd.Series(CL, index=IDX1H)
VOL_s = pd.Series(VOL, index=IDX1H)

# ── UTC time features ─────────────────────────────────────────────────────────
IDX_DT   = pd.to_datetime(IDX1H)
HOUR     = IDX_DT.hour.values
DATE_STR = IDX_DT.strftime('%Y-%m-%d')   # UTC date key for groupby

# Session masks — applied to signal arrays (no lookahead: time is deterministic)
ASIA_MASK   = (HOUR >= 0)  & (HOUR < 8)
LONDON_MASK = (HOUR >= 7)  & (HOUR < 16)
NY_MASK     = (HOUR >= 13) & (HOUR < 22)
ACTIVE_MASK = (HOUR >= 7)  & (HOUR < 22)

# ── Intraday indicators — all shift(1) ────────────────────────────────────────

# Z-scores: 12H, 24H, 48H rolling windows
ZSCORE_ID = {}
for win in [12, 24, 48]:
    zm = CL_s.rolling(win).mean()
    zs = CL_s.rolling(win).std().replace(0, np.nan)
    ZSCORE_ID[win] = ((CL_s - zm) / zs).fillna(0).shift(1).values

# Momentum
MOM = {n: CL_s.pct_change(n).shift(1).fillna(0).values for n in [4, 8]}

# RSI(4) — very short, intraday oscillations
_d4   = CL_s.diff()
_ag4  = _d4.clip(lower=0).ewm(com=3, adjust=False).mean()
_al4  = (-_d4).clip(lower=0).ewm(com=3, adjust=False).mean()
RSI4  = (100 - 100 / (_ag4 / _al4.replace(0, np.nan))).fillna(50.0).shift(1).values

# Volume ratio: vs 24H rolling mean
VOL_RATIO = (VOL_s / VOL_s.rolling(24).mean().replace(0, np.nan)
             ).fillna(1.0).shift(1).values

# Daily VWAP (cumulative from UTC midnight, then shift(1))
_vw          = pd.DataFrame({'close': CL, 'volume': VOL, 'date': DATE_STR}, index=IDX1H)
_vw['pv']    = _vw['close'] * _vw['volume']
_vw['cpv']   = _vw.groupby('date')['pv'].cumsum()
_vw['cvol']  = _vw.groupby('date')['volume'].cumsum()
_vw['vwap']  = _vw['cpv'] / _vw['cvol'].replace(0, np.nan)
VWAP_S = pd.Series(_vw['vwap'].values, index=IDX1H).shift(1).values

# Previous bar close
CL_PREV = CL_s.shift(1).values

# Daily range helpers
def _daily_range(h_max: int):
    """Range high/low from bars with HOUR < h_max, broadcast to full index."""
    _df = pd.DataFrame({'high': HI, 'low': LO, 'hour': HOUR, 'date': DATE_STR},
                       index=IDX1H)
    rng = _df[_df['hour'] < h_max].groupby('date').agg(rh=('high','max'), rl=('low','min'))
    _df = _df.join(rng, on='date')
    return _df['rh'].values, _df['rl'].values   # NaN until range complete

ORB_H,  ORB_L  = _daily_range(4)   # UTC range: hours 0–3
ASIA_H, ASIA_L = _daily_range(8)   # Asia range: hours 0–7

# Funding rate (8H interval, forward-filled to 1H, shifted)
print("[DATA] Loading funding rate …")
try:
    _fund   = fetch_binance_vision_funding(start_year=START_YEAR)
    _f1h    = _fund.reindex(IDX1H, method='ffill').fillna(0)
    FUNDING = _f1h.shift(1).values
    HAS_FUND = True
    print(f"  Funding: {len(_fund):,} records loaded")
except Exception as exc:
    print(f"  Warning: funding failed ({exc}) — skipping Group F")
    FUNDING  = np.zeros(N1H)
    HAS_FUND = False

# ── WFO windows ───────────────────────────────────────────────────────────────
def wf_dates(IDX):
    t0 = IDX[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > IDX[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
print(f"[WFO] {len(WF_WINDOWS)} windows\n")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED PIPELINE HELPERS (same logic as create_github_opt_report.py)
# ══════════════════════════════════════════════════════════════════════════════

def _ev(i, direction, tp_f, sl_f):
    d = 1 if direction == "long" else -1
    a = ATR1[i]; ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep+d*tp_f*a, sl=ep-d*sl_f*a, a=a)

def run_bt(events):
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, exppnl=0.0, net_pnls=[], cap=INIT_CAP)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, MAX_HOLD + 1):
            if i+k >= N1H: break
            hk, lk = HI[i+k], LO[i+k]
            if d == 1:
                if hk >= tp: out = "tp"; break
                if lk <= sl: out = "sl"; break
            else:
                if lk <= tp: out = "tp"; break
                if hk >= sl: out = "sl"; break
        if out == "none": continue
        pnl_r = (abs(tp-ep)/a) if out == "tp" else -(abs(sl-ep)/a)
        risk  = cap * RISK_PCT
        lev   = min(max(abs(tp-ep)/ep, abs(sl-ep)/ep), MAX_LEV)
        dpnl  = pnl_r * risk * lev
        cap += dpnl; peak = max(peak, cap)
        mdd  = min(mdd, (cap-peak)/peak)
        wins += int(out == "tp"); net_pnls.append(dpnl)
    n = len(net_pnls); wr = wins/n if n else 0.0
    ret = (cap/INIT_CAP - 1) * 100
    avg_tp = np.mean([abs(ev["tp"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    avg_sl = np.mean([abs(ev["sl"]-ev["ep"])/ev["ep"] for ev in events]) * 100
    sln = avg_sl + FEE_RT_PCT/100; tpn = avg_tp - FEE_RT_PCT/100
    return dict(n=n, wr=wr, ret=ret, mdd=mdd*100, exppnl=wr*tpn-(1-wr)*sln,
                net_pnls=net_pnls, cap=cap)

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    arr   = np.array(pnls, dtype=float)
    df_mc = pd.DataFrame({"net_pnl": arr, "gross_pnl": arr,
                           "total_fees": np.zeros(len(arr))})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)),
                p_ruin=float(mc.get("p_ruin", 1.0)))

def ic_test(sig):
    fwd  = np.log(np.roll(CL, -IC_HORIZON) / CL)
    mask = sig != 0; mask[-IC_HORIZON:] = False
    x, y = sig[mask], fwd[mask]
    if len(x) < 30: return 0.0, 1.0, 0
    r, p = st.spearmanr(x, y)
    return float(r), float(p), int(mask.sum())

def make_events(sig, idx_arr, tp_f, sl_f):
    evs = []; last_s = -COOLDOWN
    for k in idx_arr:
        if k >= N1H or sig[k] == 0 or ATR1[k] <= 0: continue
        if k - last_s < COOLDOWN: continue
        evs.append(_ev(k, "long" if sig[k] > 0 else "short", tp_f, sl_f))
        last_s = k
    return evs

def is_scan(sig, idx_is):
    best = (TP_GRID[0], SL_GRID[0]); best_xp = -np.inf
    for tf, sf in product(TP_GRID, SL_GRID):
        res = run_bt(make_events(sig, idx_is, tf, sf))
        if res["exppnl"] > best_xp: best_xp = res["exppnl"]; best = (tf, sf)
    return best

def run_pipeline(variant_id: str, name: str, sig: np.ndarray) -> dict:
    print(f"  {SEP}")
    n_long = int((sig > 0).sum()); n_short = int((sig < 0).sum())
    n_sig  = n_long + n_short
    print(f"  [{variant_id}] {name}")
    print(f"    Signals: {n_sig:,}  (L={n_long:,} / S={n_short:,})")

    ic, p_ic, n_ic = ic_test(sig)
    ic_pass = (abs(ic) > 0) and (p_ic < 0.05)
    print(f"    IC: {ic:+.4f}  p={p_ic:.4f}  n={n_ic:,}  "
          f"→ {'PASS ✓' if ic_pass else 'FAIL ✗'}")

    rec = dict(
        id=variant_id, name=name,
        n_sig=n_sig, n_long=n_long, n_short=n_short,
        ic=ic, p_ic=p_ic, n_ic=n_ic, ic_pass=ic_pass,
        wfo_run=False, oos_n=0, oos_wr=0.0, oos_ret=0.0, oos_mdd=0.0,
        mc_p_profit=0.0, mc_p_ruin=1.0, validated=False,
        oos_pnls=[], oos_equity=[INIT_CAP],
    )
    if not ic_pass: return rec

    sig_used = sig if ic > 0 else -sig
    print("    WFO …", end="", flush=True)
    rec["wfo_run"] = True; all_evs = []
    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue
        tp_f, sl_f = is_scan(sig_used, idx_is)
        all_evs.extend(make_events(sig_used, idx_oos, tp_f, sl_f))
        print(".", end="", flush=True)
    print()

    if not all_evs:
        print("    No OOS events — skip"); return rec

    res   = run_bt(all_evs)
    mc    = mc_summary(res["net_pnls"])
    valid = (res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05)

    eq = [INIT_CAP]; cap_eq = INIT_CAP
    for p in res["net_pnls"]: cap_eq += p; eq.append(cap_eq)

    print(f"    OOS: n={res['n']}  wr={res['wr']:.1%}  "
          f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    print(f"    MC:  p_profit={mc['p_profit']:.3f}  p_ruin={mc['p_ruin']:.3f}")
    print(f"    {'VALIDATED ✅' if valid else 'NOT VALIDATED ❌'}")

    rec.update(oos_n=res["n"], oos_wr=res["wr"], oos_ret=res["ret"],
               oos_mdd=res["mdd"], mc_p_profit=mc["p_profit"],
               mc_p_ruin=mc["p_ruin"], validated=valid,
               oos_pnls=res["net_pnls"], oos_equity=eq)
    return rec

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def zscore_id_sig(win: int, thr: float, mask=None) -> np.ndarray:
    z = ZSCORE_ID[win]
    s = np.where(z < -thr, 1.0, np.where(z > thr, -1.0, 0.0))
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def orb_sig(rng_h, rng_l, trade_mask) -> np.ndarray:
    nan_m = np.isnan(rng_h) | np.isnan(rng_l)
    s = np.where(trade_mask & ~nan_m & (CL_PREV > rng_h),  1.0,
        np.where(trade_mask & ~nan_m & (CL_PREV < rng_l), -1.0, 0.0))
    s[:WARMUP] = 0; return s

def vwap_mr_sig(thr_pct: float, mask=None) -> np.ndarray:
    dev = np.where(VWAP_S > 0, (CL_PREV - VWAP_S) / VWAP_S * 100, 0.0)
    s   = np.where(dev < -thr_pct, 1.0, np.where(dev > thr_pct, -1.0, 0.0))
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

def mom_mr_sig(nper: int, thr: float, mask) -> np.ndarray:
    """Fade extreme intraday moves (mean reversion)."""
    m = MOM[nper]
    s = np.where(m < -thr, 1.0, np.where(m > thr, -1.0, 0.0))
    s[~mask] = 0; s[:WARMUP] = 0; return s

def rsi4_mr_sig(lo: float, hi: float, mask) -> np.ndarray:
    s = np.where(RSI4 < lo, 1.0, np.where(RSI4 > hi, -1.0, 0.0))
    s[~mask] = 0; s[:WARMUP] = 0; return s

def funding_contrarian_sig(thr: float) -> np.ndarray:
    """Contrarian 1–2H before 8H funding reset (00, 08, 16 UTC)."""
    pre = np.isin(HOUR, [22, 23, 6, 7, 14, 15])
    s   = np.where(pre & (FUNDING >  thr), -1.0,
          np.where(pre & (FUNDING < -thr),  1.0, 0.0))
    s[:WARMUP] = 0; return s

def vol_spike_mom_sig(vol_thr: float, nper: int, mask=None) -> np.ndarray:
    """Trend-following on volume spikes."""
    m = MOM[nper]; vspk = VOL_RATIO > vol_thr
    s = np.where(vspk & (m >  0.005),  1.0,
        np.where(vspk & (m < -0.005), -1.0, 0.0))
    if mask is not None: s[~mask] = 0
    s[:WARMUP] = 0; return s

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY GROUPS
# ══════════════════════════════════════════════════════════════════════════════
ALL_RESULTS: list[dict] = []
GRP: dict[str, list[dict]] = {}   # gid → records
GRP_META: dict[str, tuple[str, str]] = {}   # gid → (title, desc)

# ── Group A: Intraday Z-score MR ──────────────────────────────────────────────
print(f"\n{SEP2}\nGroup A — Intraday Z-score Mean Reversion\n{SEP2}")
A = []
for win, thr, sess, mask in [
    (12, 1.5, "ALL",    None),
    (12, 1.5, "ACTIVE", ACTIVE_MASK),
    (24, 1.5, "ALL",    None),
    (24, 1.5, "ACTIVE", ACTIVE_MASK),
    (24, 1.5, "NY",     NY_MASK),
    (48, 2.0, "ACTIVE", ACTIVE_MASK),
    (48, 2.0, "NY",     NY_MASK),
]:
    vid = f"A-Z{win}s{str(thr).replace('.','')}-{sess}"
    r   = run_pipeline(vid, f"Z-score MR  win={win}H thr={thr}σ sess={sess}",
                        zscore_id_sig(win, thr, mask))
    ALL_RESULTS.append(r); A.append(r)

GRP["A"] = A
GRP_META["A"] = ("Intraday Z-score MR",
    "Rolling z-score su finestre brevi (12–48H) con filtro sessione. "
    "Cattura oscillazioni intraday: prezzi estremi rispetto alla media recente "
    "tendono a tornare verso il centro entro 4–16H.")

# ── Group B: Opening Range Breakout ───────────────────────────────────────────
print(f"\n{SEP2}\nGroup B — Opening Range Breakout\n{SEP2}")
B = []

utc_trade  = (HOUR >= 4) & (HOUR <= 22)
asia_trade = (HOUR >= 8) & (HOUR <= 22)

r = run_pipeline("B-ORB-UTC",  "ORB UTC  (range 00–03, trade 04–22)",
                  orb_sig(ORB_H, ORB_L, utc_trade))
ALL_RESULTS.append(r); B.append(r)

r = run_pipeline("B-ORB-ASIA", "ORB Asia (range 00–07, trade 08–22)",
                  orb_sig(ASIA_H, ASIA_L, asia_trade))
ALL_RESULTS.append(r); B.append(r)

GRP["B"] = B
GRP_META["B"] = ("Opening Range Breakout",
    "Le prime 4 (o 8) candele UTC definiscono il range del giorno. "
    "Chiusura precedente sopra/sotto il range → segnale direzionale.")

# ── Group C: Daily VWAP Deviation MR ─────────────────────────────────────────
print(f"\n{SEP2}\nGroup C — Daily VWAP Deviation Mean Reversion\n{SEP2}")
C = []
for thr, sess, mask in [
    (0.3, "ACTIVE", ACTIVE_MASK),
    (0.5, "ACTIVE", ACTIVE_MASK),
    (1.0, "ACTIVE", ACTIVE_MASK),
    (0.5, "NY",     NY_MASK),
    (1.0, "NY",     NY_MASK),
]:
    vid = f"C-VWAP{str(thr).replace('.','p')}-{sess}"
    r   = run_pipeline(vid, f"VWAP MR  dev={thr}% sess={sess}",
                        vwap_mr_sig(thr, mask))
    ALL_RESULTS.append(r); C.append(r)

GRP["C"] = C
GRP_META["C"] = ("Daily VWAP Deviation — MR",
    "Deviazione dal VWAP giornaliero (reset a mezzanotte UTC). "
    "Prezzi lontani dal VWAP tendono a ritornarci entro la sessione.")

# ── Group D: Session Momentum Fade ────────────────────────────────────────────
print(f"\n{SEP2}\nGroup D — Session Momentum Fade\n{SEP2}")
D = []
for nper, thr, sess, mask in [
    (4, 0.005, "NY",     NY_MASK),
    (4, 0.010, "NY",     NY_MASK),
    (8, 0.010, "NY",     NY_MASK),
    (4, 0.005, "LONDON", LONDON_MASK),
    (4, 0.010, "LONDON", LONDON_MASK),
]:
    vid = f"D-MOM{nper}H-{int(thr*1000)}bp-{sess}"
    r   = run_pipeline(vid, f"Momentum Fade  {nper}H thr={thr*100:.1f}% sess={sess}",
                        mom_mr_sig(nper, thr, mask))
    ALL_RESULTS.append(r); D.append(r)

GRP["D"] = D
GRP_META["D"] = ("Session Momentum Fade",
    "Fade di movimenti estremi nelle ultime 4–8H durante una sessione specifica. "
    "Movimenti bruschi all'interno di una sessione spesso si correggono parzialmente.")

# ── Group E: Short RSI(4) ─────────────────────────────────────────────────────
print(f"\n{SEP2}\nGroup E — Short RSI(4) MR\n{SEP2}")
E = []
for lo, hi, sess, mask in [
    (35, 65, "NY",     NY_MASK),
    (30, 70, "NY",     NY_MASK),
    (25, 75, "NY",     NY_MASK),
    (30, 70, "ACTIVE", ACTIVE_MASK),
]:
    vid = f"E-RSI4-{lo}{hi}-{sess}"
    r   = run_pipeline(vid, f"RSI(4) MR  {lo}/{hi} sess={sess}",
                        rsi4_mr_sig(lo, hi, mask))
    ALL_RESULTS.append(r); E.append(r)

GRP["E"] = E
GRP_META["E"] = ("Short RSI(4) Mean Reversion",
    "RSI period=4 (ultra-veloce) cattura oscillazioni intraday. "
    "Filtro sessione concentra i segnali nelle ore più liquide.")

# ── Group F: Funding Rate Contrarian ─────────────────────────────────────────
F = []
if HAS_FUND:
    print(f"\n{SEP2}\nGroup F — Funding Rate Contrarian\n{SEP2}")
    for thr in [0.0003, 0.0005, 0.0010]:
        vid = f"F-FUND-{int(thr*10000)}bp"
        r   = run_pipeline(vid, f"Funding Contrarian  thr={thr*100:.2f}%",
                            funding_contrarian_sig(thr))
        ALL_RESULTS.append(r); F.append(r)

GRP["F"] = F
GRP_META["F"] = ("Funding Rate Contrarian",
    "Funding estremo implica posizioni sbilanciate. "
    "Andare contro la direzione dominante 1–2H prima del reset (00/08/16 UTC) "
    "per catturare la chiusura di posizioni che evitano il pagamento.")

# ── Group G: Volume Spike Momentum ────────────────────────────────────────────
print(f"\n{SEP2}\nGroup G — Volume Spike Momentum\n{SEP2}")
G = []
for vthr, nper, sess, mask in [
    (2.0, 4, "ALL",    None),
    (2.0, 4, "ACTIVE", ACTIVE_MASK),
    (3.0, 4, "ACTIVE", ACTIVE_MASK),
    (2.0, 8, "ACTIVE", ACTIVE_MASK),
]:
    vid = f"G-VSPK{str(vthr).replace('.','p')}x-MOM{nper}H-{sess}"
    r   = run_pipeline(vid, f"Vol Spike Momentum  vol>{vthr}× {nper}H sess={sess}",
                        vol_spike_mom_sig(vthr, nper, mask))
    ALL_RESULTS.append(r); G.append(r)

GRP["G"] = G
GRP_META["G"] = ("Volume Spike Momentum",
    "Volume > N× media rolling 24H segnala un breakout genuino. "
    "Direzione determinata dal momentum delle ultime 4–8H (trend-following).")

# ── Group H: Combo ────────────────────────────────────────────────────────────
print(f"\n{SEP2}\nGroup H — Combo Signals\n{SEP2}")
H = []

# H1: Z24 ACTIVE + VWAP 0.5% ACTIVE
_za = zscore_id_sig(24, 1.5, ACTIVE_MASK)
_vw = vwap_mr_sig(0.5, ACTIVE_MASK)
_h1 = np.where((_za > 0) & (_vw > 0), 1.0,
      np.where((_za < 0) & (_vw < 0), -1.0, 0.0))
_h1[:WARMUP] = 0
r = run_pipeline("H-Z24-VWAP05-ACT", "Combo Z24 + VWAP(0.5%) — ACTIVE", _h1)
ALL_RESULTS.append(r); H.append(r)

# H2: Z48 NY + RSI4 NY
_zn = zscore_id_sig(48, 2.0, NY_MASK)
_rn = rsi4_mr_sig(30, 70, NY_MASK)
_h2 = np.where((_zn > 0) & (_rn > 0), 1.0,
      np.where((_zn < 0) & (_rn < 0), -1.0, 0.0))
_h2[:WARMUP] = 0
r = run_pipeline("H-Z48-RSI4-NY", "Combo Z48 + RSI4(30/70) — NY", _h2)
ALL_RESULTS.append(r); H.append(r)

# H3: Z24 NY + Funding (if available)
if HAS_FUND:
    _z2 = zscore_id_sig(24, 1.5, NY_MASK)
    _fd = funding_contrarian_sig(0.0003)
    _h3 = np.where((_z2 > 0) & (_fd > 0), 1.0,
          np.where((_z2 < 0) & (_fd < 0), -1.0, 0.0))
    _h3[:WARMUP] = 0
    r = run_pipeline("H-Z24NY-FUND03", "Combo Z24-NY + Funding(0.03%)", _h3)
    ALL_RESULTS.append(r); H.append(r)

GRP["H"] = H
GRP_META["H"] = ("Combo Signals",
    "Due segnali indipendenti devono concordare per generare un trade. "
    "Riduce i falsi positivi a scapito della frequenza.")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
n_total = len(ALL_RESULTS)
n_ic    = sum(1 for r in ALL_RESULTS if r["ic_pass"])
n_wfo   = sum(1 for r in ALL_RESULTS if r["wfo_run"])
n_val   = sum(1 for r in ALL_RESULTS if r["validated"])

print(f"\n{SEP2}\nSUMMARY ({n_total} varianti)\n{SEP2}")
print(f"  IC pass   : {n_ic}/{n_total}")
print(f"  WFO run   : {n_wfo}/{n_total}")
print(f"  Validated : {n_val}/{n_total}\n")

wfo_ran = sorted([r for r in ALL_RESULTS if r["wfo_run"]],
                  key=lambda r: r["oos_ret"], reverse=True)
for r in wfo_ran:
    flag = "✅" if r["validated"] else ("⚠" if r["oos_ret"] > 0 else "✗")
    print(f"  {flag} [{r['id']:<26}]  ret={r['oos_ret']:+5.1f}%  "
          f"pp={r['mc_p_profit']:.3f}  pr={r['mc_p_ruin']:.3f}")
for r in ALL_RESULTS:
    if not r["wfo_run"]:
        print(f"  ✗ [{r['id']:<26}]  IC FAIL  ic={r['ic']:+.4f}  p={r['p_ic']:.3f}")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _equity_chart(rec: dict) -> str:
    eq = rec["oos_equity"]
    if len(eq) < 2: return ""
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=_BG)
    ax.set_facecolor(_BG)
    color = _GRN if eq[-1] >= INIT_CAP else _RED
    ax.plot(range(len(eq)), eq, color=color, lw=1.5)
    ax.axhline(INIT_CAP, color=_GRID, ls="--", lw=0.8)
    ax.set_ylabel("Equity ($)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(alpha=0.15, color=_GRID)
    ax.set_title(f"{rec['id']} — OOS Equity", color=_TEXT, fontsize=9)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:820px;">'

def _overview_chart() -> str:
    ran = sorted([r for r in ALL_RESULTS if r["wfo_run"]],
                  key=lambda r: r["oos_ret"], reverse=True)
    if not ran: return ""
    ids  = [r["id"] for r in ran]
    rets = [r["oos_ret"] for r in ran]
    pps  = [r["mc_p_profit"] for r in ran]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), facecolor=_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG); ax.tick_params(colors=_TEXT, labelsize=6.5)
        for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
        ax.grid(axis="y", alpha=0.2, color=_GRID)

    ax1.bar(range(len(ids)), rets,
            color=[_GRN if v > 0 else _RED for v in rets], edgecolor=_GRID, lw=0.4)
    ax1.axhline(0, color=_TEXT, lw=0.7)
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=55, ha="right", fontsize=6)
    ax1.set_ylabel("OOS Return (%)", color=_TEXT, fontsize=9)
    ax1.set_title("OOS Return per variante", color=_TEXT, fontsize=10)

    ax2.bar(range(len(ids)), pps,
            color=[_GRN if v > 0.90 else (_YEL if v > 0.60 else _RED) for v in pps],
            edgecolor=_GRID, lw=0.4)
    ax2.axhline(0.90, color=_YEL, ls="--", lw=0.9, label="threshold 0.90")
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids, rotation=55, ha="right", fontsize=6)
    ax2.set_ylabel("MC P(profit)", color=_TEXT, fontsize=9)
    ax2.set_title("MC P(profit) per variante", color=_TEXT, fontsize=10)
    ax2.legend(labelcolor=_TEXT, facecolor=_BG, fontsize=8)

    fig.tight_layout(pad=1.5)
    b64 = _fig_b64(fig); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1200px;">'

def badge(r):
    if r["validated"]:            return '<span class="badge green">VALIDATED ✅</span>'
    if r["wfo_run"] and r["oos_ret"] > 0:
                                   return '<span class="badge yellow">OOS+ ⚠</span>'
    if r["wfo_run"]:               return '<span class="badge orange">WFO FAIL ✗</span>'
    return                                '<span class="badge red">IC FAIL ✗</span>'

def build_section(gid, title, desc, variants):
    rows = ""
    for r in variants:
        if r["wfo_run"]:
            rc = "green" if r["oos_ret"] > 0 else "red"
            pc = "green" if r["mc_p_profit"] > 0.90 else ("yellow" if r["mc_p_profit"] > 0.60 else "red")
            qc = "green" if r["mc_p_ruin"] < 0.05 else "red"
            wc = (f"<td>{r['oos_n']}</td><td>{r['oos_wr']:.1%}</td>"
                  f"<td class='{rc}'>{r['oos_ret']:+.1f}%</td>"
                  f"<td class='red'>{r['oos_mdd']:.1f}%</td>"
                  f"<td class='{pc}'>{r['mc_p_profit']:.3f}</td>"
                  f"<td class='{qc}'>{r['mc_p_ruin']:.3f}</td>")
        else:
            wc = "<td colspan='6' style='color:#555'>—</td>"
        ic_c = "green" if r["ic"] > 0 else ("red" if r["ic_pass"] else "dim")
        rows += (f"<tr>"
                 f"<td><code>{r['id']}</code></td>"
                 f"<td>{r['name']}</td>"
                 f"<td>{r['n_sig']:,}</td>"
                 f"<td class='{ic_c}'>{r['ic']:+.4f}</td>"
                 f"<td class=\"{'green' if r['p_ic']<0.05 else 'red'}\">{r['p_ic']:.4f}</td>"
                 f"{wc}"
                 f"<td>{badge(r)}</td></tr>")
    eq_imgs = "".join(_equity_chart(r) for r in variants
                      if r["wfo_run"] and r["oos_n"] > 0)
    return f"""
    <div class="card section-card">
      <h2 style="margin-top:0">Group {gid} — {title}</h2>
      <p class="desc">{desc}</p>
      <div class="tbl-wrap"><table class="tbl">
        <thead><tr>
          <th>ID</th><th>Variante</th><th>N segnali</th>
          <th>IC</th><th>p-val</th>
          <th>OOS n</th><th>WR</th><th>OOS Ret</th><th>MDD</th>
          <th>P(profit)</th><th>P(ruin)</th><th>Status</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
      {eq_imgs}
    </div>"""

# Assemble report
overview_img   = _overview_chart()
sections_html  = ""
for gid in "ABCDEFGH":
    if GRP.get(gid):
        t, d = GRP_META.get(gid, ("", ""))
        sections_html += build_section(gid, t, d, GRP[gid])

best_ret = max((r["oos_ret"]       for r in ALL_RESULTS if r["wfo_run"]), default=0.0)
best_pp  = max((r["mc_p_profit"]   for r in ALL_RESULTS if r["wfo_run"]), default=0.0)

CSS = f"""
  :root{{--bg:{_BG};--card:{_CARD};--grid:{_GRID};--text:{_TEXT};
         --acc:{_ACC};--grn:{_GRN};--red:{_RED};--yel:{_YEL};--org:{_ORG};}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);
        font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;
        line-height:1.5;padding:24px;}}
  h1{{color:var(--acc);font-size:22px;margin-bottom:4px;}}
  h2{{color:var(--acc);font-size:15px;margin:0 0 8px;}}
  .sub{{color:#777;font-size:12px;margin-bottom:24px;}}
  .kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px;}}
  .kpi{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
         padding:12px 18px;min-width:130px;}}
  .kpi .val{{font-size:26px;font-weight:700;color:var(--acc);}}
  .kpi .lbl{{font-size:11px;color:#777;}}
  .card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;
          padding:18px;margin-bottom:20px;}}
  .section-card{{border-left:3px solid var(--acc);}}
  .desc{{color:#888;font-size:12px;margin-bottom:14px;}}
  .tbl-wrap{{overflow-x:auto;}}
  .tbl{{border-collapse:collapse;width:100%;min-width:900px;}}
  .tbl th{{background:var(--grid);color:var(--acc);text-align:left;
            padding:6px 10px;font-size:12px;white-space:nowrap;}}
  .tbl td{{padding:5px 10px;border-bottom:1px solid var(--grid);
            font-size:12px;white-space:nowrap;}}
  .tbl tr:hover td{{background:var(--grid);}}
  .green{{color:var(--grn)!important;}} .red{{color:var(--red)!important;}}
  .yellow{{color:var(--yel)!important;}} .dim{{color:#555!important;}}
  .badge{{display:inline-block;border-radius:4px;padding:2px 7px;
           font-size:11px;font-weight:600;}}
  .badge.green{{background:#1b3a1e;color:var(--grn);}}
  .badge.red{{background:#3a1a1a;color:var(--red);}}
  .badge.orange{{background:#3a2a00;color:var(--org);}}
  .badge.yellow{{background:#2d2a00;color:var(--yel);}}
  img{{display:block;margin-bottom:12px;border-radius:6px;max-width:100%;}}
  code{{font-size:11px;background:var(--grid);padding:1px 4px;border-radius:3px;}}
"""

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Intraday BTC Strategy Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>Intraday BTC/USDT Strategy Report — 1H Bars</h1>
<p class="sub">
  {n_total} varianti · {len(GRP)} gruppi strategici ·
  IC {IC_HORIZON}H · WFO 6m/2m/2m · MC N={N_SIMS:,} ·
  Cooldown {COOLDOWN}H · Max hold {MAX_HOLD}H &nbsp;|&nbsp;
  {IDX1H[0].date()} – {IDX1H[-1].date()} · {N1H:,} bar 1H
</p>

<div class="kpi-row">
  <div class="kpi"><div class="val">{n_total}</div>
    <div class="lbl">Varianti testate</div></div>
  <div class="kpi"><div class="val">{n_ic}</div>
    <div class="lbl">IC pass (p&lt;0.05)</div></div>
  <div class="kpi"><div class="val">{n_wfo}</div>
    <div class="lbl">WFO eseguito</div></div>
  <div class="kpi">
    <div class="val" style="color:{'var(--grn)' if n_val>0 else 'var(--red)'}">{n_val}</div>
    <div class="lbl">Validate</div></div>
  <div class="kpi"><div class="val" style="font-size:18px">{best_ret:+.1f}%</div>
    <div class="lbl">Miglior OOS ret</div></div>
  <div class="kpi"><div class="val" style="font-size:18px">{best_pp:.3f}</div>
    <div class="lbl">Miglior MC p_profit</div></div>
</div>

<div class="card">
  <h2>Overview — OOS Return e MC P(profit) per variante</h2>
  {overview_img}
</div>

{sections_html}

<hr style="border-color:var(--grid);margin:32px 0 16px">
<p style="color:#444;font-size:11px;text-align:center;">
  BTCUSDT perpetual futures · Binance Vision 2020–2026 · 1H bars ·
  IC {IC_HORIZON}H Spearman · WFO 6m IS / 2m OOS / 2m step ·
  MC N={N_SIMS:,} · Risk 1%/trade · Fee 0.08% RT ·
  Cooldown {COOLDOWN}H · Max hold {MAX_HOLD}H
</p>
</body>
</html>"""

out = Path("reports/report_intraday.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n[DONE] Report → {out}  ({out.stat().st_size//1024} KB)")
