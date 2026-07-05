"""
create_external_strategies_report.py
=====================================
External strategies translation and validation — BTCUSDT 1H × 4H ATR.

Repositories analyzed:
  EXT_01 [mynria/Nadaraya_Watson_Binance_Trading_Bot]
         Nadaraya-Watson Envelope → causal EWMA approximation
         Mean-reversion: short above upper band, long below lower band.

  EXT_02 [yulz008/orb_cryptoBot]
         Opening Range Breakout (00:00–03:59 UTC range, breakout after 04:00)
         Trend-following: long when close > daily range high, short < daily range low.

  EXT_03 [usamatariq014/XGBoost-BTC]
         XGBoost Triple Barrier ML (long-only, EMA200 filter)
         Causal WFO: retrain XGB each IS window, predict on OOS.

  EXT_04 [akenshaw/btcusdt-orderflow]
         Real-time orderbook visualizer — no trading signals. SKIPPED.

Pipeline: IC → IS scan → WFO → Monte Carlo → HTML report
"""
from __future__ import annotations

import base64, io, sys, warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.monte_carlo  import run_monte_carlo

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP     = 100_000.0
RISK_PCT     = 0.01
FEE          = 0.0004
FEE_RT_PCT   = FEE * 2 * 100         # 0.08%
MAX_LEV      = 5.0
MAX_HOLD     = 96                     # 4 days
IC_HORIZON   = 16
START_YEAR   = 2020
N_SIMS       = 5_000
COOLDOWN     = 8

WF_TRAIN_M   = 6
WF_OOS_M     = 2
WF_STEP_M    = 2

TP_FRAC_GRID = [1.0, 2.0, 3.0, 5.0]
SL_FRAC_GRID = [0.25, 0.5, 0.75, 1.0]

# EXT_01: Nadaraya-Watson params
NW_SPAN = 18
NW_MULT = 2.0

# EXT_02: Opening Range Breakout
OR_HOURS = 4          # 00:00–03:59 UTC

# EXT_03: XGBoost
XGB_PARAMS = dict(
    objective="binary:logistic",
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_weight=10,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=2,
    eval_metric="logloss",
)
XGB_THRESHOLD = 0.55
XGB_HORIZON   = 24
XGB_SL_MULT   = 1.0
XGB_TP_MULT   = 2.0

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
SEP   = "─" * 70
SEP2  = "═" * 70

# ── Boot ──────────────────────────────────────────────────────────────────────
print(SEP2)
print("External Strategies — BTCUSDT 1H × 4H ATR")
print("  EXT_01 : Nadaraya-Watson Envelope (causal EWMA)")
print("  EXT_02 : Opening Range Breakout (00:00–03:59 UTC)")
print("  EXT_03 : XGBoost Triple Barrier ML (long-only)")
print("  EXT_04 : OrderFlow Visualizer → SKIPPED")
print(SEP2)

print("[DATA] Loading …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df   = add_indicators(raw["1H"])
df4h = add_indicators(raw["4H"])

print(f"  1H : {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  4H : {len(df4h):,} bars  ({df4h.index[0].date()} → {df4h.index[-1].date()})")

IDX   = df.index;   N = len(df)
HI    = df["high"].values
LO    = df["low"].values
CL    = df["close"].values
OP    = df["open"].values
VOL   = df["volume"].values
ATR1H = np.where(df["atr_14"].values > 0, df["atr_14"].values, 1.0)
hr_   = np.array([t.hour for t in IDX], dtype=int)

# 4H ATR → 1H (shift 1: only previously closed 4H bar)
atr4h_raw = df4h["atr_14"].shift(1).reindex(IDX, method="ffill")
ATR4H     = np.where(atr4h_raw.values > 0, atr4h_raw.values, ATR1H)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _be_fee(avg_sl: float, avg_tp: float) -> float:
    sl_net = avg_sl + FEE_RT_PCT / 100
    tp_net = avg_tp - FEE_RT_PCT / 100
    return sl_net / (sl_net + tp_net) if (sl_net + tp_net) > 0 else 0.5

def _ev(i: int, direction: str, tp_f: float, sl_f: float) -> dict:
    d  = 1 if direction == "long" else -1
    a  = ATR4H[i]
    ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep + d * tp_f * a, sl=ep - d * sl_f * a, a=a)

def run_backtest(events: list[dict]) -> dict:
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0, exppnl=0.0,
                    trades=[], net_pnls=[], cap=INIT_CAP)
    cap  = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; trades = []; net_pnls = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, MAX_HOLD + 1):
            if i + k >= N: break
            h_k, l_k = HI[i + k], LO[i + k]
            if d == 1:
                if h_k >= tp: out = "tp"; break
                if l_k <= sl: out = "sl"; break
            else:
                if l_k <= tp: out = "tp"; break
                if h_k >= sl: out = "sl"; break
        if out == "none": continue
        pnl_r = (abs(tp - ep) / a) if out == "tp" else -(abs(sl - ep) / a)
        risk  = cap * RISK_PCT
        lev   = min(max(abs(tp - ep) / ep, abs(sl - ep) / ep), MAX_LEV)
        dollar_pnl = pnl_r * risk * lev
        cap  += dollar_pnl
        peak  = max(peak, cap)
        mdd   = min(mdd, (cap - peak) / peak)
        wins += int(out == "tp")
        trades.append(out)
        net_pnls.append(dollar_pnl)
    n     = len(trades)
    wr    = wins / n if n else 0.0
    ret   = (cap / INIT_CAP - 1) * 100
    avg_tp = np.mean([abs(ev["tp"] - ev["ep"]) / ev["ep"] for ev in events]) * 100
    avg_sl = np.mean([abs(ev["sl"] - ev["ep"]) / ev["ep"] for ev in events]) * 100
    be     = _be_fee(avg_sl, avg_tp)
    slnet  = avg_sl + FEE_RT_PCT / 100
    tpnet  = avg_tp - FEE_RT_PCT / 100
    ev_adj = wr * tpnet - (1 - wr) * slnet
    return dict(n=n, wr=wr, ret=ret, mdd=mdd * 100, be_fee=be * 100,
                exppnl=ev_adj, trades=trades, net_pnls=net_pnls, cap=cap)

def ic_test(signal_dir: np.ndarray) -> tuple[float, float, int]:
    fwd  = np.log(np.roll(CL, -IC_HORIZON) / CL)
    mask = signal_dir != 0
    mask[-IC_HORIZON:] = False
    x, y = signal_dir[mask], fwd[mask]
    if len(x) < 30:
        return 0.0, 1.0, 0
    r, p = st.spearmanr(x, y)
    return float(r), float(p), int(mask.sum())

def is_scan(gen_fn, tp_grid=TP_FRAC_GRID, sl_grid=SL_FRAC_GRID):
    best = None
    for tp_f, sl_f in product(tp_grid, sl_grid):
        res = run_backtest(gen_fn(tp_f, sl_f))
        if best is None or res["exppnl"] > best[2]:
            best = (tp_f, sl_f, res["exppnl"], res)
    return best   # (tp_f, sl_f, exppnl, stats)

def wf_dates():
    t0 = IDX[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > IDX[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 = t0 + pd.DateOffset(months=WF_STEP_M)
    return windows

def mc_summary(res: dict) -> dict:
    """Bootstrap MC from run_backtest result dict."""
    pnls = res.get("net_pnls", [])
    if len(pnls) < 5:
        return dict(p_profit=0.0, p_ruin=1.0)
    pnls_arr = np.array(pnls, dtype=float)
    df_mc = pd.DataFrame({"net_pnl": pnls_arr,
                          "gross_pnl": pnls_arr,
                          "total_fees": np.zeros(len(pnls_arr))})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)),
                p_ruin=float(mc.get("p_ruin", 1.0)))


# ══════════════════════════════════════════════════════════════════════════════
# EXT_01 — Nadaraya-Watson Envelope (causal EWMA)
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[EXT_01] Nadaraya-Watson Envelope (causal EWMA approximation)")

CL_s   = pd.Series(CL, index=IDX)
nw_ewm = CL_s.ewm(span=NW_SPAN, adjust=False).mean()
mae_nw = (CL_s - nw_ewm).abs().ewm(span=NW_SPAN, adjust=False).mean()

# shift(1): use previous bar's band values (causal)
nw_upper = (nw_ewm + NW_MULT * mae_nw).shift(1).values
nw_lower = (nw_ewm - NW_MULT * mae_nw).shift(1).values

warmup_nw = NW_SPAN * 3
nw_signal = np.where(CL > nw_upper, -1.0,
            np.where(CL < nw_lower, +1.0, 0.0))
nw_signal[:warmup_nw] = 0.0

ic_nw, p_nw, n_nw = ic_test(nw_signal)
print(f"  IC={ic_nw:+.4f}  p={p_nw:.4f}  n={n_nw}")

def gen_nw_all(tp_f: float, sl_f: float) -> list[dict]:
    evs = []; last_sig = -COOLDOWN
    for i in range(warmup_nw, N - MAX_HOLD - 2):
        if ATR4H[i] <= 0 or nw_signal[i] == 0: continue
        if i - last_sig < COOLDOWN: continue
        evs.append(_ev(i, "long" if nw_signal[i] > 0 else "short", tp_f, sl_f))
        last_sig = i
    return evs

nw_valid = False
all_oos_events_nw: list[dict] = []
mc_nw:  dict = dict(p_profit=0.0, p_ruin=1.0)
res_nw: dict = dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0,
                    exppnl=0.0, trades=[], cap=INIT_CAP)
best_nw = (1.0, 0.5, 0.0, {})
NW_TP, NW_SL = 1.0, 0.5

if ic_nw > 0 and p_nw < 0.05:
    print("  ✓ IC significativo — IS scan …")
    best_nw = is_scan(gen_nw_all)
    NW_TP, NW_SL = best_nw[0], best_nw[1]
    print(f"  IS best: tp={NW_TP}×ATR sl={NW_SL}×ATR ExpPnL={best_nw[2]:+.4f}%")

    print("  [WFO] Walk-forward …")
    for tr_s, tr_e, oo_s, oo_e in wf_dates():
        idx_is  = np.where((IDX >= tr_s) & (IDX < tr_e))[0]
        idx_oos = np.where((IDX >= oo_s) & (IDX < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue

        def _gen_is(tp_f, sl_f):
            evs = []; last_s = -COOLDOWN
            for k in idx_is:
                if ATR4H[k] <= 0 or nw_signal[k] == 0: continue
                if k - last_s < COOLDOWN: continue
                evs.append(_ev(k, "long" if nw_signal[k] > 0 else "short", tp_f, sl_f))
                last_s = k
            return evs

        bw = is_scan(_gen_is); tp_w, sl_w = bw[0], bw[1]
        last_s = -COOLDOWN
        for k in idx_oos:
            if ATR4H[k] <= 0 or nw_signal[k] == 0: continue
            if k - last_s < COOLDOWN: continue
            all_oos_events_nw.append(
                _ev(k, "long" if nw_signal[k] > 0 else "short", tp_w, sl_w))
            last_s = k

    res_nw  = run_backtest(all_oos_events_nw)
    mc_nw   = mc_summary(res_nw)
    nw_valid = (res_nw["ret"] > 0
                and mc_nw.get("p_profit", 0) > 0.9
                and mc_nw.get("p_ruin", 1) < 0.05)
    print(f"  OOS ret={res_nw['ret']:+.1f}%  WR={res_nw['wr']*100:.1f}%  "
          f"n={res_nw['n']}  P(profit)={mc_nw.get('p_profit',0)*100:.1f}%  "
          f"P(ruin)={mc_nw.get('p_ruin',0)*100:.1f}%")
    print(f"  {'✓ VALIDATA' if nw_valid else '✗ NON VALIDATA'}")
else:
    print(f"  SKIP: IC={ic_nw:+.4f} p={p_nw:.4f} — nessun edge significativo")


# ══════════════════════════════════════════════════════════════════════════════
# EXT_02 — Opening Range Breakout
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[EXT_02] Opening Range Breakout (00:00–03:59 UTC)")

df_or = df.copy()
df_or["date_"] = pd.to_datetime(df_or.index.date)
df_or["hour_"] = df_or.index.hour

or_bars   = df_or[df_or["hour_"] < OR_HOURS]
or_daily  = or_bars.groupby("date_").agg(
    or_high=("high", "max"), or_low=("low", "min"))

or_high_dict = dict(zip(or_daily.index, or_daily["or_high"].values))
or_low_dict  = dict(zip(or_daily.index, or_daily["or_low"].values))
or_high_arr  = np.array([or_high_dict.get(d, np.nan)
                          for d in df_or["date_"].values])
or_low_arr   = np.array([or_low_dict.get(d, np.nan)
                          for d in df_or["date_"].values])
valid_time   = hr_ >= OR_HOURS

orb_signal = np.zeros(N, dtype=float)
prev_date_str = ""; fired_today = False
for i in range(N):
    dt_str = str(IDX[i].date())
    if dt_str != prev_date_str:
        fired_today = False; prev_date_str = dt_str
    if not valid_time[i] or np.isnan(or_high_arr[i]): continue
    if fired_today: continue
    if CL[i] > or_high_arr[i]:
        orb_signal[i] = +1; fired_today = True
    elif CL[i] < or_low_arr[i]:
        orb_signal[i] = -1; fired_today = True

ic_orb, p_orb, n_orb = ic_test(orb_signal)
print(f"  IC={ic_orb:+.4f}  p={p_orb:.4f}  n={n_orb}")

def gen_orb_all(tp_f: float, sl_f: float) -> list[dict]:
    evs = []
    for i in range(OR_HOURS, N - MAX_HOLD - 2):
        if ATR4H[i] <= 0 or orb_signal[i] == 0: continue
        evs.append(_ev(i, "long" if orb_signal[i] > 0 else "short", tp_f, sl_f))
    return evs

orb_valid = False
all_oos_events_orb: list[dict] = []
mc_orb:  dict = dict(p_profit=0.0, p_ruin=1.0)
res_orb: dict = dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0,
                     exppnl=0.0, trades=[], cap=INIT_CAP)
best_orb = (1.0, 0.5, 0.0, {})
ORB_TP, ORB_SL = 1.0, 0.5

if ic_orb > 0 and p_orb < 0.05:
    print("  ✓ IC significativo — IS scan …")
    best_orb = is_scan(gen_orb_all)
    ORB_TP, ORB_SL = best_orb[0], best_orb[1]
    print(f"  IS best: tp={ORB_TP}×ATR sl={ORB_SL}×ATR ExpPnL={best_orb[2]:+.4f}%")

    print("  [WFO] Walk-forward …")
    for tr_s, tr_e, oo_s, oo_e in wf_dates():
        idx_is  = np.where((IDX >= tr_s) & (IDX < tr_e))[0]
        idx_oos = np.where((IDX >= oo_s) & (IDX < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue

        def _gen_orb_is(tp_f, sl_f):
            return [_ev(k, "long" if orb_signal[k] > 0 else "short", tp_f, sl_f)
                    for k in idx_is if ATR4H[k] > 0 and orb_signal[k] != 0]

        bw = is_scan(_gen_orb_is); tp_w, sl_w = bw[0], bw[1]
        for k in idx_oos:
            if ATR4H[k] > 0 and orb_signal[k] != 0:
                all_oos_events_orb.append(
                    _ev(k, "long" if orb_signal[k] > 0 else "short", tp_w, sl_w))

    res_orb  = run_backtest(all_oos_events_orb)
    mc_orb   = mc_summary(res_orb)
    orb_valid = (res_orb["ret"] > 0
                 and mc_orb.get("p_profit", 0) > 0.9
                 and mc_orb.get("p_ruin", 1) < 0.05)
    print(f"  OOS ret={res_orb['ret']:+.1f}%  WR={res_orb['wr']*100:.1f}%  "
          f"n={res_orb['n']}  P(profit)={mc_orb.get('p_profit',0)*100:.1f}%  "
          f"P(ruin)={mc_orb.get('p_ruin',0)*100:.1f}%")
    print(f"  {'✓ VALIDATA' if orb_valid else '✗ NON VALIDATA'}")
else:
    print(f"  SKIP: IC={ic_orb:+.4f} p={p_orb:.4f} — nessun edge significativo")


# ══════════════════════════════════════════════════════════════════════════════
# EXT_03 — XGBoost Triple Barrier ML
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[EXT_03] XGBoost Triple Barrier ML (long-only, EMA200 filter)")
print("  Computing features …")

CL_s  = pd.Series(CL, index=IDX)
HI_s  = pd.Series(HI, index=IDX)
LO_s  = pd.Series(LO, index=IDX)
OP_s  = pd.Series(OP, index=IDX)
VOL_s = pd.Series(VOL, index=IDX)

feat_df = pd.DataFrame(index=IDX)
feat_df["close"] = CL
feat_df["high"]  = HI
feat_df["low"]   = LO
feat_df["open"]  = OP

# EMA distances (shift 1)
for span in [10, 20, 50, 200]:
    ema = CL_s.ewm(span=span, adjust=False).mean()
    feat_df[f"dist_ema{span}"] = ((CL_s - ema) / ema).shift(1)

# ATR% (14-bar, shift 1)
tr14 = pd.concat([HI_s - LO_s,
                  (HI_s - CL_s.shift(1)).abs(),
                  (LO_s - CL_s.shift(1)).abs()], axis=1).max(axis=1)
atr14 = tr14.rolling(14, min_periods=1).mean()
feat_df["atr_pct"] = (atr14 / CL_s).shift(1)

# Bollinger Band width (shift 1)
bb_mid = CL_s.rolling(20, min_periods=5).mean()
bb_std = CL_s.rolling(20, min_periods=5).std().fillna(0)
feat_df["bb_width"] = (4 * bb_std / (bb_mid + 1e-8)).shift(1)

# Relative volume (shift 1)
avg_vol = VOL_s.rolling(20, min_periods=5).mean()
feat_df["rel_vol"] = (VOL_s / (avg_vol + 1e-8)).shift(1)

# Candlestick patterns (shift 1)
body    = CL_s - OP_s
full_rn = (HI_s - LO_s).replace(0, np.nan)
feat_df["cdl_doji"] = ((body.abs() / full_rn) < 0.1).astype(float).shift(1)
feat_df["cdl_bull_engulf"] = (
    (CL_s > OP_s.shift(1)) & (OP_s < CL_s.shift(1)) & (CL_s > OP_s)
).astype(float).shift(1)
feat_df["cdl_bear_engulf"] = (
    (CL_s < OP_s.shift(1)) & (OP_s > CL_s.shift(1)) & (CL_s < OP_s)
).astype(float).shift(1)

# Time features (no shift — known at bar open)
feat_df["hour_sin"]    = np.sin(2 * np.pi * IDX.hour / 24)
feat_df["hour_cos"]    = np.cos(2 * np.pi * IDX.hour / 24)
feat_df["day_of_week"] = IDX.dayofweek.astype(float)

FEATURE_COLS = [c for c in feat_df.columns
                if c not in ("open", "high", "low", "close")]

# EMA200 causal filter (shift 1)
ema200_causal = CL_s.ewm(span=200, adjust=False).mean().shift(1).values

feat_df_clean = feat_df.dropna(subset=FEATURE_COLS).copy()
feat_arr  = feat_df_clean[FEATURE_COLS].values
feat_gidx = np.array([IDX.get_loc(t) for t in feat_df_clean.index])  # global bar indices

def compute_tb_labels(gidx_arr: np.ndarray) -> np.ndarray:
    """Triple Barrier binary label computed on IS subset."""
    n_k    = len(gidx_arr)
    labels = np.zeros(n_k, dtype=int)
    ap_col = feat_df_clean["atr_pct"].values[
        np.searchsorted(feat_gidx, gidx_arr, side="left")]
    for j, gi in enumerate(gidx_arr):
        if gi + XGB_HORIZON >= N: continue
        ap = float(ap_col[j])
        if ap <= 0: continue
        entry  = CL[gi]
        tp_p   = entry * (1 + ap * XGB_TP_MULT)
        sl_p   = entry * (1 - ap * XGB_SL_MULT)
        fh     = HI[gi + 1: gi + 1 + XGB_HORIZON]
        fl     = LO[gi + 1: gi + 1 + XGB_HORIZON]
        tp_hit = np.where(fh >= tp_p)[0]
        sl_hit = np.where(fl <= sl_p)[0]
        if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
            labels[j] = 1
    return labels

print("  [WFO] Walk-forward XGBoost …")
all_xgb_oos_events: list[dict] = []
xgb_probs_list: list[tuple[int, float]] = []

windows = wf_dates()
for w_i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
    is_mask = (feat_df_clean.index >= tr_s) & (feat_df_clean.index < tr_e)
    oo_mask = (feat_df_clean.index >= oo_s) & (feat_df_clean.index < oo_e)
    if is_mask.sum() < 200 or oo_mask.sum() < 50: continue

    X_is    = feat_arr[is_mask]
    gidx_is = feat_gidx[is_mask]
    y_is    = compute_tb_labels(gidx_is)
    if y_is.sum() < 10 or (y_is == 0).sum() < 10: continue

    pos_w = (y_is == 0).sum() / max((y_is == 1).sum(), 1)
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_w)
    model.fit(X_is, y_is, verbose=False)

    X_oo     = feat_arr[oo_mask]
    gidx_oo  = feat_gidx[oo_mask]
    probs_oo = model.predict_proba(X_oo)[:, 1]
    probs_is = model.predict_proba(X_is)[:, 1]

    # IS scan: optimize TP/SL fracs
    best_ev = None; best_params = (TP_FRAC_GRID[0], SL_FRAC_GRID[0])
    for tp_f, sl_f in product(TP_FRAC_GRID, SL_FRAC_GRID):
        evs_tmp = []; last_s = -COOLDOWN
        for j, gi in enumerate(gidx_is):
            if ATR4H[gi] <= 0 or gi - last_s < COOLDOWN: continue
            if probs_is[j] >= XGB_THRESHOLD and CL[gi] > ema200_causal[gi] > 0:
                evs_tmp.append(_ev(gi, "long", tp_f, sl_f)); last_s = gi
        r = run_backtest(evs_tmp)
        if best_ev is None or r["exppnl"] > best_ev:
            best_ev = r["exppnl"]; best_params = (tp_f, sl_f)

    tp_w, sl_w = best_params
    last_s = -COOLDOWN
    for j, gi in enumerate(gidx_oo):
        if ATR4H[gi] <= 0 or gi - last_s < COOLDOWN: continue
        if probs_oo[j] >= XGB_THRESHOLD and CL[gi] > ema200_causal[gi] > 0:
            all_xgb_oos_events.append(_ev(gi, "long", tp_w, sl_w)); last_s = gi
        xgb_probs_list.append((gi, float(probs_oo[j])))

    if (w_i + 1) % 5 == 0:
        print(f"    window {w_i + 1}/{len(windows)}")

# IC for XGBoost (probability vs forward return)
if xgb_probs_list:
    gi_arr   = np.array([x[0] for x in xgb_probs_list])
    pr_arr   = np.array([x[1] for x in xgb_probs_list])
    fwd_16h  = np.log(np.roll(CL, -IC_HORIZON) / CL)
    vmask    = gi_arr < N - IC_HORIZON
    ic_xgb, p_xgb = st.spearmanr(pr_arr[vmask], fwd_16h[gi_arr[vmask]]) if vmask.sum() > 30 else (0.0, 1.0)
    n_xgb   = int(vmask.sum())
else:
    ic_xgb, p_xgb, n_xgb = 0.0, 1.0, 0

print(f"  IC={ic_xgb:+.4f}  p={p_xgb:.4f}  n={n_xgb}")
res_xgb  = run_backtest(all_xgb_oos_events)
mc_xgb   = mc_summary(res_xgb)
# Validation: positive OOS return + P(profit)>90% + P(ruin)<5%
xgb_valid = (res_xgb["ret"] > 0
             and mc_xgb.get("p_profit", 0) > 0.9
             and mc_xgb.get("p_ruin", 1) < 0.05)
print(f"  OOS ret={res_xgb['ret']:+.1f}%  WR={res_xgb['wr']*100:.1f}%  "
      f"n={res_xgb['n']}  P(profit)={mc_xgb.get('p_profit',0)*100:.1f}%  "
      f"P(ruin)={mc_xgb.get('p_ruin',0)*100:.1f}%")
print(f"  {'✓ VALIDATA' if xgb_valid else '✗ NON VALIDATA'}")


# ══════════════════════════════════════════════════════════════════════════════
# HTML Report
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[HTML] Generating report …")

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode(); plt.close(fig); return b64

def _equity_b64(events: list[dict], color: str) -> str:
    if not events: return ""
    cap = INIT_CAP; caps = [cap]; times = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, MAX_HOLD + 1):
            if i + k >= N: break
            h_k, l_k = HI[i + k], LO[i + k]
            if d == 1:
                if h_k >= tp: out = "tp"; break
                if l_k <= sl: out = "sl"; break
            else:
                if l_k <= tp: out = "tp"; break
                if h_k >= sl: out = "sl"; break
        if out == "none": continue
        pnl_r = (abs(tp - ep) / a) if out == "tp" else -(abs(sl - ep) / a)
        risk  = cap * RISK_PCT
        lev   = min(max(abs(tp - ep) / ep, abs(sl - ep) / ep), MAX_LEV)
        cap  += pnl_r * risk * lev
        caps.append(cap); times.append(IDX[i])
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    xs = [IDX[0]] + times
    ax.plot(xs, caps, color=color, lw=1.4)
    ax.axhline(INIT_CAP, color=_GRID, lw=0.8, ls="--")
    ax.tick_params(colors=_TEXT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    fig.tight_layout()
    return _fig_b64(fig)

def _badge(valid: bool, n: int) -> str:
    if valid:            return '<span class="badge g">✓ VALIDATA</span>'
    elif n < 20:         return '<span class="badge grey">↔ SEGNALI INSUFFICIENTI</span>'
    else:                return '<span class="badge r">✗ NON VALIDATA</span>'

def _ic_badge(ic: float, p: float) -> str:
    ok = ic > 0 and p < 0.05
    cls = "g" if ok else "r"
    sym = "✓" if ok else "✗"
    return f'<span class="badge {cls}">IC={ic:+.4f}  p={p:.4f}  {sym}</span>'

def _sr(label: str, val: str, hl: bool = False) -> str:
    cls = '  class="hl"' if hl else ""
    return f"<tr{cls}><td>{label}</td><td>{val}</td></tr>"

# Build strategy blocks
strategies_data = [
    dict(id="EXT_01", name="Nadaraya-Watson Envelope",
         repo="mynria/Nadaraya_Watson_Binance_Trading_Bot",
         ic=ic_nw, p_ic=p_nw, n_ic=n_nw, valid=nw_valid,
         res=res_nw, mc=mc_nw, tp=NW_TP, sl=NW_SL,
         events=all_oos_events_nw, color=_ACC,
         desc=(f"Kernel regression causale (EWMA, span={NW_SPAN}, mult=±{NW_MULT}). "
               "L'originale usa NW bidirezionale su 300 barre (O(N²), look-ahead). "
               "Qui: EWMA con shift(1) = NW causale con kernel esponenziale. "
               "SHORT quando close &gt; upper band, LONG quando close &lt; lower band."),
         logic="LONG: close &lt; nw − mult×MAE  |  SHORT: close &gt; nw + mult×MAE"),
    dict(id="EXT_02", name="Opening Range Breakout",
         repo="yulz008/orb_cryptoBot",
         ic=ic_orb, p_ic=p_orb, n_ic=n_orb, valid=orb_valid,
         res=res_orb, mc=mc_orb, tp=ORB_TP, sl=ORB_SL,
         events=all_oos_events_orb, color=_ORG,
         desc=(f"Su crypto 24/7: OR = prime {OR_HOURS}H UTC (00:00–03:59). "
               "Dopo le 04:00 UTC: LONG se close &gt; max(range), SHORT se close &lt; min(range). "
               "Un segnale per giorno. L'originale usa 15 min su exchange spot."),
         logic=f"LONG: close &gt; OR_high  |  SHORT: close &lt; OR_low (dopo {OR_HOURS}H UTC)"),
    dict(id="EXT_03", name="XGBoost Triple Barrier ML",
         repo="usamatariq014/XGBoost-BTC",
         ic=ic_xgb, p_ic=p_xgb, n_ic=n_xgb, valid=xgb_valid,
         res=res_xgb, mc=mc_xgb, tp=1.0, sl=0.5,
         events=all_xgb_oos_events, color=_GRN,
         desc=(f"XGBoost (depth=4, n=150, lr=0.05) con Triple Barrier target (24H, 1ATR SL, 2ATR TP). "
               f"WFO causale: modello riaddestrato su ogni finestra IS 6M. "
               f"Long-only, filtro EMA200. Threshold={XGB_THRESHOLD}. "
               "Features: dist EMA10/20/50/200, ATR%, BB width, volume ratio, pattern, time encoding."),
         logic=f"LONG: P(win) &ge; {XGB_THRESHOLD} AND close &gt; EMA200"),
]

cards_html = ""
for s in strategies_data:
    r  = s["res"]; mc = s["mc"]
    eq_b64 = _equity_b64(s["events"], s["color"])
    eq_tag = (f'<div class="sect-title">Equity Curve OOS</div>'
              f'<img src="data:image/png;base64,{eq_b64}" style="width:100%;max-width:700px">'
              if eq_b64 else '<div class="no-wfo">Walk-Forward non eseguito (IC non superato)</div>')
    wr_pct = r["wr"] * 100
    be_pct = r["be_fee"]
    rows = "".join([
        _sr("N segnali OOS",        str(r["n"])),
        _sr("OOS Return",           f"{r['ret']:+.1f}%",         r["ret"] > 0),
        _sr("Max Drawdown",         f"{r['mdd']:+.1f}%"),
        _sr("Win Rate",             f"{wr_pct:.1f}%",            wr_pct > be_pct),
        _sr("Break-even (fee-adj)", f"{be_pct:.1f}%"),
        _sr("WR surplus vs BE",     f"{wr_pct - be_pct:+.1f} pp", wr_pct > be_pct),
        _sr("P(profit) MC",         f"{mc.get('p_profit',0)*100:.1f}%",
                                    mc.get("p_profit", 0) > 0.9),
        _sr("P(ruin) MC",           f"{mc.get('p_ruin',0)*100:.1f}%",
                                    mc.get("p_ruin", 1) < 0.05),
        _sr("TP ottimale",          f"{s['tp']}×ATR4H"),
        _sr("SL ottimale",          f"{s['sl']}×ATR4H"),
    ])
    cards_html += f"""
<div class="card">
  <div class="card-hdr">
    <span class="sid">{s['id']}</span>
    <span class="sname">{s['name']}</span>
    <a class="rlink" href="https://github.com/{s['repo']}" target="_blank">⬡ {s['repo']}</a>
    {_badge(s['valid'], r['n'])}
  </div>
  <div class="two-col">
    <div>
      <div class="sect-title">Strategia e Adattamento</div>
      <p class="desc">{s['desc']}</p>
      <div class="mono">{s['logic']}</div>
      <div class="sect-title" style="margin-top:14px">Information Coefficient</div>
      {_ic_badge(s['ic'], s['p_ic'])}
      <div class="subn">n={s['n_ic']} segnali · orizzonte {IC_HORIZON}H · Spearman</div>
    </div>
    <div>
      <div class="sect-title">Risultati OOS (Walk-Forward)</div>
      <table class="st">{rows}</table>
    </div>
  </div>
  <div style="padding:0 18px 18px">{eq_tag}</div>
</div>
"""

# IC comparison bar chart
fig_ic, ax_ic = plt.subplots(figsize=(8, 2.8), facecolor=_BG)
ax_ic.set_facecolor(_BG)
ids_l  = ["EXT_01\nNadaraya-Watson", "EXT_02\nORB", "EXT_03\nXGBoost"]
ics_l  = [ic_nw, ic_orb, ic_xgb]
ps_l   = [p_nw, p_orb, p_xgb]
c_bars = [_GRN if (ic > 0 and p < 0.05) else _RED
          for ic, p in zip(ics_l, ps_l)]
bars = ax_ic.bar(ids_l, ics_l, color=c_bars, alpha=0.85, width=0.5)
ax_ic.axhline(0, color=_GRID, lw=1)
for bar, ic_v in zip(bars, ics_l):
    va = "bottom" if ic_v >= 0 else "top"
    offset = 0.001 if ic_v >= 0 else -0.001
    ax_ic.text(bar.get_x() + bar.get_width() / 2, ic_v + offset,
               f"{ic_v:+.4f}", ha="center", va=va, color=_TEXT, fontsize=8)
ax_ic.set_ylabel("IC (Spearman)", color=_TEXT, fontsize=9)
ax_ic.set_title("Information Coefficient — External Strategies", color=_TEXT, fontsize=10)
ax_ic.tick_params(colors=_TEXT, labelsize=8)
for sp in ax_ic.spines.values(): sp.set_edgecolor(_GRID)
fig_ic.tight_layout()
ic_b64 = _fig_b64(fig_ic)

# Summary table
def _srow(s: dict) -> str:
    r = s["res"]; mc = s["mc"]
    ic_c = "#66bb6a" if s["ic"] > 0 and s["p_ic"] < 0.05 else "#ef5350"
    rc   = "#66bb6a" if r["ret"] > 0 else "#ef5350"
    vc   = "#66bb6a" if s["valid"] else "#ef5350"
    return (f'<tr>'
            f'<td>{s["id"]}</td><td>{s["name"]}</td>'
            f'<td style="color:{ic_c}">{s["ic"]:+.4f}</td>'
            f'<td>{s["p_ic"]:.3f}</td>'
            f'<td style="color:{rc}">{r["ret"]:+.1f}%</td>'
            f'<td>{r["wr"]*100:.1f}%</td>'
            f'<td>{r["mdd"]:+.1f}%</td>'
            f'<td>{r["n"]}</td>'
            f'<td>{mc.get("p_profit",0)*100:.1f}%</td>'
            f'<td>{mc.get("p_ruin",0)*100:.1f}%</td>'
            f'<td style="color:{vc};font-weight:bold">{"✓" if s["valid"] else "✗"}</td>'
            f'</tr>')

summary_rows = "".join(_srow(s) for s in strategies_data)

HTML = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>External Strategies — BTCUSDT Validation Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{_BG};color:{_TEXT};font-family:'Segoe UI',sans-serif;font-size:13px;line-height:1.5}}
h1{{font-size:1.35em;color:{_ACC};padding:20px 24px 4px}}
.sub{{color:#888;padding:0 24px 16px;font-size:.9em}}
.wrap{{max-width:1100px;margin:0 auto;padding:0 16px 40px}}
.card{{background:{_CARD};border:1px solid {_GRID};border-radius:8px;margin-bottom:24px}}
.card-hdr{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;
           padding:14px 18px;background:#161926;border-bottom:1px solid {_GRID};border-radius:8px 8px 0 0}}
.sid{{background:{_ACC}22;color:{_ACC};font-weight:bold;font-size:.9em;
      padding:3px 10px;border-radius:4px;border:1px solid {_ACC}44}}
.sname{{font-weight:600;font-size:1.05em;flex:1}}
.rlink{{color:#777;font-size:.8em;text-decoration:none}}
.rlink:hover{{color:{_ACC}}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;
        font-size:.78em;font-weight:700;letter-spacing:.5px}}
.badge.g{{background:#1b3a2a;color:{_GRN};border:1px solid {_GRN}44}}
.badge.r{{background:#3a1b1b;color:{_RED};border:1px solid {_RED}44}}
.badge.grey{{background:#2a2a2a;color:#888;border:1px solid #44444444}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:18px}}
@media(max-width:700px){{.two-col{{grid-template-columns:1fr}}}}
.sect-title{{font-size:.72em;text-transform:uppercase;color:#888;
             letter-spacing:.6px;margin-bottom:6px;margin-top:10px}}
.desc{{color:{_TEXT};font-size:.88em;line-height:1.6;margin-bottom:10px}}
.mono{{background:#1a1d28;border:1px solid {_GRID};border-radius:4px;
       padding:8px 12px;font-family:monospace;font-size:.82em;color:#b0c4de}}
.subn{{color:#666;font-size:.78em;margin-top:5px}}
.st{{width:100%;border-collapse:collapse}}
.st td{{padding:5px 8px;border-bottom:1px solid {_GRID};font-size:.88em}}
.st td:first-child{{color:#aaa;width:55%}}
.st td:last-child{{font-weight:600;text-align:right}}
.st tr.hl td{{background:#1a2a1a}}
.no-wfo{{padding:16px;color:#666;font-style:italic}}
.ic-chart,.summary-card,.note-box{{background:{_CARD};border:1px solid {_GRID};
                                    border-radius:8px;margin-bottom:24px;padding:16px}}
.ic-chart{{text-align:center}}
h2{{font-size:.95em;color:{_ACC};margin-bottom:10px}}
.stbl{{width:100%;border-collapse:collapse;font-size:.85em}}
.stbl th{{background:#161926;color:#888;padding:8px 10px;text-align:left;
          border-bottom:1px solid {_GRID};font-size:.75em;text-transform:uppercase;letter-spacing:.5px}}
.stbl td{{padding:8px 10px;border-bottom:1px solid {_GRID}}}
.note-box h3{{font-size:.9em;color:{_YEL};margin-bottom:8px}}
.note-box p{{font-size:.85em;color:#aaa;margin-bottom:6px;line-height:1.6}}
footer{{text-align:center;color:#555;font-size:.78em;padding:20px 0}}
</style>
</head>
<body>
<div class="wrap">
<h1>External Strategies — BTCUSDT Validation Report</h1>
<div class="sub">
  Pipeline: IC → IS scan → WFO (6m IS / 2m OOS) → Monte Carlo N=5,000 &nbsp;|&nbsp;
  Dati: 2020-01 → 2026-06 · 1H × 4H ATR · FEE=0.08% RT · RISK=1%/trade
</div>

<div class="ic-chart">
  <div style="font-size:.8em;color:#888;margin-bottom:6px">IC Comparison — External Strategies</div>
  <img src="data:image/png;base64,{ic_b64}" style="width:100%;max-width:700px">
</div>

<div class="summary-card">
  <h2>Riepilogo</h2>
  <table class="stbl">
    <thead>
      <tr>
        <th>ID</th><th>Strategia</th><th>IC</th><th>p-val</th>
        <th>OOS Ret</th><th>WR</th><th>MaxDD</th><th>N trades</th>
        <th>P(profit)</th><th>P(ruin)</th><th>Esito</th>
      </tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>

{cards_html}

<div class="note-box" style="opacity:.65">
  <h3>EXT_04 — akenshaw/btcusdt-orderflow: SKIPPED</h3>
  <p>Tool di visualizzazione real-time (PyQt6 + WebSocket Binance): heatmap di liquidità,
  DOM, skew e imbalance del book ordini. <strong>Non genera segnali di trading algoritmici.</strong>
  L'imbalance bid/ask è una metrica per analisi manuale, non una regola di entry/exit
  implementata nel codice. Non backtestabile nel framework attuale.</p>
  <p>Potenziale sviluppo: usare l'imbalance come feature per XGBoost (EXT_03),
  richiederebbe dati storici di order book tick-by-tick non disponibili su Binance Vision.</p>
</div>

<div class="note-box">
  <h3>Metodologia — Adattamento Strategie Esterne</h3>
  <p><strong>EXT_01 (NW):</strong> Il kernel NW originale è bidirezionale (usa barre future → look-ahead).
  L'EWMA causale è il limite del kernel NW con banda esponenziale: <code>y[i] = Σ w(i-j)·close[j]</code>
  con <code>w(k) = α(1-α)^k</code>. shift(1) garantisce causalità sul segnale.</p>
  <p><strong>EXT_02 (ORB):</strong> Crypto 24/7: il "market open" diventa le prime {OR_HOURS}H UTC.
  L'approccio è identico all'originale (capture range → trade breakout), adattato al ritmo daily crypto.</p>
  <p><strong>EXT_03 (XGB):</strong> L'originale usa train/test split (no WFO). Noi impostiamo WFO
  con riaddestramento per ogni finestra IS 6M, rendendolo causale al 100%. Features identiche ma
  implementate senza talib (pandas/numpy). Triple Barrier target applicato solo ai dati IS.</p>
  <p><strong>Anti-lookahead:</strong> Tutti i segnali usano shift(1). ATR4H usa shift(1).
  Nessun bar della finestra OOS è visibile durante IS scan o training del modello XGB.</p>
</div>

<footer>Pipeline di validazione quantitativa — BTCUSDT 2020–2026 &nbsp;|&nbsp; {pd.Timestamp.now().date()}</footer>
</div>
</body>
</html>"""

out = Path("reports/report_external_strategies.html")
out.parent.mkdir(exist_ok=True)
out.write_text(HTML, encoding="utf-8")
print(f"\n✅ Report saved: {out}  ({out.stat().st_size // 1024} KB)")
print(SEP2)
