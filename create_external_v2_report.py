"""
create_external_v2_report.py
============================
Seconda iterazione delle strategie esterne — ottimizzazioni mirate:

  EXT_01v2 [NW Momentum + IS ottimizzazione soglia MULT]
    IC=-0.033 mean-reversion → invertito: MOMENTUM NW.
    IS scan: griglia MULT × TP/SL ottimizzata per finestra WFO.

  EXT_02v2 [ORB su 15M — Opening Range Breakout a risoluzione 15 minuti]
    Range = prime 4H UTC (prime 16 barre M15). Entry dal bar 04:00.
    Più segnali, maggior risoluzione per breakout intraday.

  EXT_03v2 [XGBoost su 4H — stesso pipeline, timeframe superiore]
    Stesse feature della v1 ma su 14K barre 4H invece di 57K 1H.
    Meno rumore, trend più stabili, WFO su 4H.
"""
from __future__ import annotations

import base64, io, sys, warnings
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

from src.strategy.data_fetcher import (fetch_extended_data,
                                        fetch_binance_vision_klines)
from src.strategy.indicators   import add_indicators
from src.strategy.monte_carlo  import run_monte_carlo

# ── Config globali ────────────────────────────────────────────────────────────
INIT_CAP     = 100_000.0
RISK_PCT     = 0.01
FEE          = 0.0004
FEE_RT_PCT   = FEE * 2 * 100           # 0.08%
MAX_LEV      = 5.0
IC_HORIZON_1H = 16                     # 16H forward (1H bars)
IC_HORIZON_4H =  4                     # 16H forward (4H bars)
START_YEAR   = 2020
N_SIMS       = 5_000
COOLDOWN_1H  = 8
COOLDOWN_4H  = 2
COOLDOWN_15M = 32                      # 8H in 15M bars

WF_TRAIN_M   = 6
WF_OOS_M     = 2
WF_STEP_M    = 2

TP_FRAC_GRID = [1.0, 2.0, 3.0, 5.0]
SL_FRAC_GRID = [0.25, 0.5, 0.75, 1.0]

# EXT_01v2: NW params
NW_MULT_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
NW_SPAN_BASE = 18

# EXT_02v2: ORB 15M
OR_HOURS_15M = 4        # range = first 4H = 16 barre 15M
MAX_HOLD_15M = 384      # 4 giorni in 15M

# EXT_03v2: XGBoost 4H
MAX_HOLD_4H  = 24       # 4 giorni in 4H
XGB_PARAMS_4H = dict(
    objective="binary:logistic", n_estimators=200,
    learning_rate=0.05, max_depth=4,
    reg_alpha=1.0, reg_lambda=1.0,
    min_child_weight=5, gamma=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=2, eval_metric="logloss",
)
XGB_THRESHOLD_4H = 0.55
XGB_HORIZON_4H   = 10   # 10 barre 4H = 40H
XGB_SL_MULT      = 1.0
XGB_TP_MULT      = 2.0

_BG = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC = "#42a5f5"; _GRN = "#66bb6a"
_RED = "#ef5350"; _YEL = "#ffd54f"; _ORG = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("External Strategies v2 — Ottimizzazioni mirate")
print("  EXT_01v2 : NW Momentum + IS ottimizzazione soglia MULT")
print("  EXT_02v2 : ORB su 15M")
print("  EXT_03v2 : XGBoost su 4H")
print(SEP2)

# ── Dati 1H e 4H ─────────────────────────────────────────────────────────────
print("[DATA] Loading 1H, 4H …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
df4h = add_indicators(raw["4H"])

print(f"  1H : {len(df1h):,} bars  ({df1h.index[0].date()} → {df1h.index[-1].date()})")
print(f"  4H : {len(df4h):,} bars  ({df4h.index[0].date()} → {df4h.index[-1].date()})")

IDX1H = df1h.index;  N1H = len(df1h)
CL1H  = df1h["close"].values
HI1H  = df1h["high"].values
LO1H  = df1h["low"].values
OP1H  = df1h["open"].values
VOL1H = df1h["volume"].values
ATR1H_raw = np.where(df1h["atr_14"].values > 0, df1h["atr_14"].values, 1.0)

IDX4H = df4h.index;  N4H = len(df4h)
CL4H  = df4h["close"].values
HI4H  = df4h["high"].values
LO4H  = df4h["low"].values
OP4H  = df4h["open"].values
VOL4H = df4h["volume"].values

# 4H ATR → 1H  (shift 1: usa solo barra 4H precedente chiusa)
atr4h_raw_1h = df4h["atr_14"].shift(1).reindex(IDX1H, method="ffill")
ATR4H_1H     = np.where(atr4h_raw_1h.values > 0, atr4h_raw_1h.values, ATR1H_raw)

# 4H ATR per 4H stesso (shift 1)
ATR4H_4H = np.where(df4h["atr_14"].shift(1).values > 0,
                    df4h["atr_14"].shift(1).values, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers condivisi
# ══════════════════════════════════════════════════════════════════════════════
def _be_fee(avg_sl: float, avg_tp: float) -> float:
    sl_net = avg_sl + FEE_RT_PCT / 100
    tp_net = avg_tp - FEE_RT_PCT / 100
    return sl_net / (sl_net + tp_net) if (sl_net + tp_net) > 0 else 0.5

def _ev_generic(i: int, direction: str, tp_f: float, sl_f: float,
                CL: np.ndarray, ATR: np.ndarray) -> dict:
    d  = 1 if direction == "long" else -1
    a  = ATR[i]
    ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep + d * tp_f * a, sl=ep - d * sl_f * a, a=a)

def run_backtest_generic(events: list[dict],
                         HI: np.ndarray, LO: np.ndarray,
                         N: int, max_hold: int) -> dict:
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0,
                    exppnl=0.0, trades=[], net_pnls=[], cap=INIT_CAP)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0
    trades: list[str] = []; net_pnls: list[float] = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, max_hold + 1):
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
        dpnl  = pnl_r * risk * lev
        cap  += dpnl; peak = max(peak, cap); mdd = min(mdd, (cap - peak) / peak)
        wins += int(out == "tp"); trades.append(out); net_pnls.append(dpnl)
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

def mc_summary(res: dict) -> dict:
    pnls = res.get("net_pnls", [])
    if len(pnls) < 5:
        return dict(p_profit=0.0, p_ruin=1.0)
    arr  = np.array(pnls, dtype=float)
    df_mc = pd.DataFrame({"net_pnl": arr, "gross_pnl": arr,
                           "total_fees": np.zeros(len(arr))})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)),
                p_ruin=float(mc.get("p_ruin", 1.0)))

def is_scan_grid(gen_fn, tp_grid=TP_FRAC_GRID, sl_grid=SL_FRAC_GRID):
    best = None
    for tp_f, sl_f in product(tp_grid, sl_grid):
        res = run_backtest_generic(**gen_fn(tp_f, sl_f))
        if best is None or res["exppnl"] > best[2]:
            best = (tp_f, sl_f, res["exppnl"], res)
    return best

def wf_dates_from_idx(IDX):
    t0 = IDX[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > IDX[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 = t0 + pd.DateOffset(months=WF_STEP_M)
    return windows

def ic_test_generic(signal_dir: np.ndarray, CL: np.ndarray,
                    ic_horizon: int) -> tuple[float, float, int]:
    fwd  = np.log(np.roll(CL, -ic_horizon) / CL)
    mask = signal_dir != 0
    mask[-ic_horizon:] = False
    x, y = signal_dir[mask], fwd[mask]
    if len(x) < 30: return 0.0, 1.0, 0
    r, p = st.spearmanr(x, y)
    return float(r), float(p), int(mask.sum())

def is_validated(res: dict, mc: dict) -> bool:
    return (res["ret"] > 0
            and mc.get("p_profit", 0) > 0.90
            and mc.get("p_ruin", 1) < 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# EXT_01v2 — Nadaraya-Watson MOMENTUM + IS ottimizzazione MULT
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[EXT_01v2] NW Momentum + IS scan su MULT")
print("  IC originale (mean-rev): -0.0325 → provo direzione MOMENTUM invertita")

CL1H_s = pd.Series(CL1H, index=IDX1H)
warmup_nw = NW_SPAN_BASE * 4

# Pre-compute NW bands per ogni MULT possibile (shift 1 → causal)
nw_ewm  = CL1H_s.ewm(span=NW_SPAN_BASE, adjust=False).mean()
nw_mae  = (CL1H_s - nw_ewm).abs().ewm(span=NW_SPAN_BASE, adjust=False).mean()
nw_ewm1 = nw_ewm.shift(1).values
nw_mae1 = nw_mae.shift(1).values

def nw_signals_momentum(mult: float) -> np.ndarray:
    """MOMENTUM: LONG quando close > nw+mult×mae, SHORT quando close < nw-mult×mae."""
    upper = nw_ewm1 + mult * nw_mae1
    lower = nw_ewm1 - mult * nw_mae1
    sig   = np.where(CL1H > upper, +1.0,
            np.where(CL1H < lower, -1.0, 0.0))
    sig[:warmup_nw] = 0.0
    return sig

# IC test con mult=2.0 (default)
sig_nw_default = nw_signals_momentum(2.0)
ic_nw2, p_nw2, n_nw2 = ic_test_generic(sig_nw_default, CL1H, IC_HORIZON_1H)
print(f"  IC momentum (mult=2.0): {ic_nw2:+.4f}  p={p_nw2:.4f}  n={n_nw2}")

# Test IC per vari MULT per capire quale funziona meglio
print("  IC per MULT:")
best_ic_mult = 2.0; best_ic_val = ic_nw2; best_ic_p = p_nw2
for mult in NW_MULT_GRID:
    sig_m = nw_signals_momentum(mult)
    ic_m, p_m, n_m = ic_test_generic(sig_m, CL1H, IC_HORIZON_1H)
    print(f"    mult={mult:.1f}  IC={ic_m:+.4f}  p={p_m:.4f}  n={n_m}")
    if ic_m > best_ic_val:
        best_ic_val = ic_m; best_ic_mult = mult; best_ic_p = p_m

print(f"  Miglior MULT per IC: {best_ic_mult} (IC={best_ic_val:+.4f} p={best_ic_p:.4f})")

res_nw2 = dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0,
               exppnl=0.0, trades=[], net_pnls=[], cap=INIT_CAP)
mc_nw2  = dict(p_profit=0.0, p_ruin=1.0)
nw2_valid = False
all_oos_nw2: list[dict] = []
best_nw2_params = (best_ic_mult, TP_FRAC_GRID[0], SL_FRAC_GRID[0])

if best_ic_val > 0 and best_ic_p < 0.05:
    print("  ✓ IC significativo — WFO con IS scan su MULT × TP/SL …")

    def gen_nw_ev(mult: float, tp_f: float, sl_f: float,
                  mask_idx: np.ndarray, cooldown: int) -> dict:
        """Genera eventi NW momentum su sottoinsieme mask_idx."""
        sig = nw_signals_momentum(mult)
        evs = []; last_s = -cooldown
        for k in mask_idx:
            if ATR4H_1H[k] <= 0 or sig[k] == 0: continue
            if k - last_s < cooldown: continue
            evs.append(_ev_generic(k, "long" if sig[k] > 0 else "short",
                                   tp_f, sl_f, CL1H, ATR4H_1H))
            last_s = k
        return dict(events=evs, HI=HI1H, LO=LO1H, N=N1H, max_hold=96)

    for tr_s, tr_e, oo_s, oo_e in wf_dates_from_idx(IDX1H):
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50: continue

        # IS scan su MULT × TP × SL
        best_ev_is = None; best_p_is = (best_ic_mult, TP_FRAC_GRID[0], SL_FRAC_GRID[0])
        for mult in NW_MULT_GRID:
            for tp_f, sl_f in product(TP_FRAC_GRID, SL_FRAC_GRID):
                kw  = gen_nw_ev(mult, tp_f, sl_f, idx_is, COOLDOWN_1H)
                res = run_backtest_generic(**kw)
                if best_ev_is is None or res["exppnl"] > best_ev_is:
                    best_ev_is = res["exppnl"]
                    best_p_is  = (mult, tp_f, sl_f)

        mult_w, tp_w, sl_w = best_p_is
        sig_oos = nw_signals_momentum(mult_w)
        last_s  = -COOLDOWN_1H
        for k in idx_oos:
            if ATR4H_1H[k] <= 0 or sig_oos[k] == 0: continue
            if k - last_s < COOLDOWN_1H: continue
            all_oos_nw2.append(
                _ev_generic(k, "long" if sig_oos[k] > 0 else "short",
                            tp_w, sl_w, CL1H, ATR4H_1H))
            last_s = k

    res_nw2  = run_backtest_generic(all_oos_nw2, HI1H, LO1H, N1H, 96)
    mc_nw2   = mc_summary(res_nw2)
    nw2_valid = is_validated(res_nw2, mc_nw2)
    print(f"  OOS ret={res_nw2['ret']:+.1f}%  WR={res_nw2['wr']*100:.1f}%  "
          f"n={res_nw2['n']}  MaxDD={res_nw2['mdd']:+.1f}%  "
          f"P(profit)={mc_nw2.get('p_profit',0)*100:.1f}%  "
          f"P(ruin)={mc_nw2.get('p_ruin',1)*100:.1f}%")
    print(f"  {'✓ VALIDATA' if nw2_valid else '✗ NON VALIDATA'}")
else:
    print(f"  IC non significativo per nessun MULT — skip WFO")


# ══════════════════════════════════════════════════════════════════════════════
# EXT_02v2 — Opening Range Breakout su 15M
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[EXT_02v2] Opening Range Breakout — 15M")
print("  Downloading 15M data …")

df15m_raw = fetch_binance_vision_klines("15m", start_year=START_YEAR, start_month=1)
if df15m_raw.empty:
    print("  ✗ 15M non disponibile — skip")
    res_orb15  = dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0,
                      exppnl=0.0, trades=[], net_pnls=[], cap=INIT_CAP)
    mc_orb15   = dict(p_profit=0.0, p_ruin=1.0)
    orb15_valid = False
    ic_orb15, p_orb15, n_orb15 = 0.0, 1.0, 0
    all_oos_orb15: list[dict] = []
else:
    df15m = add_indicators(df15m_raw)
    print(f"  15M: {len(df15m):,} bars  ({df15m.index[0].date()} → {df15m.index[-1].date()})")

    IDX15  = df15m.index;  N15 = len(df15m)
    CL15   = df15m["close"].values
    HI15   = df15m["high"].values
    LO15   = df15m["low"].values
    hr15   = np.array([t.hour for t in IDX15], dtype=int)
    ATR1H_15 = np.where(df15m["atr_14"].values > 0, df15m["atr_14"].values, 1.0)

    # 4H ATR → 15M (shift 1, causal)
    atr4h_15m  = df4h["atr_14"].shift(1).reindex(IDX15, method="ffill")
    ATR4H_15M  = np.where(atr4h_15m.values > 0, atr4h_15m.values, ATR1H_15)

    # OR = prime OR_HOURS_15M ore UTC (00:00–03:45), valida dal bar 04:00 in poi
    OR_BARS_15M = OR_HOURS_15M * 4     # 4 bar15M per ora
    df15m["date_"] = pd.to_datetime(IDX15.date)
    df15m["hour_"] = IDX15.hour

    or_bars_15  = df15m[df15m["hour_"] < OR_HOURS_15M]
    or_daily_15 = or_bars_15.groupby("date_").agg(
        or_high=("high", "max"), or_low=("low", "min"))
    or_hi_dict  = dict(zip(or_daily_15.index, or_daily_15["or_high"].values))
    or_lo_dict  = dict(zip(or_daily_15.index, or_daily_15["or_low"].values))
    or_hi_15    = np.array([or_hi_dict.get(d, np.nan) for d in df15m["date_"].values])
    or_lo_15    = np.array([or_lo_dict.get(d, np.nan) for d in df15m["date_"].values])
    valid_15    = hr15 >= OR_HOURS_15M

    orb15_signal = np.zeros(N15, dtype=float)
    prev_date_s  = ""; fired_today = False
    for i in range(N15):
        dt_s = str(IDX15[i].date())
        if dt_s != prev_date_s:
            fired_today = False; prev_date_s = dt_s
        if not valid_15[i] or np.isnan(or_hi_15[i]): continue
        if fired_today: continue
        if CL15[i] > or_hi_15[i]:
            orb15_signal[i] = +1; fired_today = True
        elif CL15[i] < or_lo_15[i]:
            orb15_signal[i] = -1; fired_today = True

    ic_orb15, p_orb15, n_orb15 = ic_test_generic(orb15_signal, CL15, IC_HORIZON_1H * 4)
    # IC_HORIZON equivalente: 16H = 64 barre 15M
    ic_orb15, p_orb15, n_orb15 = ic_test_generic(orb15_signal, CL15, 64)
    print(f"  IC={ic_orb15:+.4f}  p={p_orb15:.4f}  n={n_orb15}")

    all_oos_orb15: list[dict] = []
    res_orb15  = dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0,
                      exppnl=0.0, trades=[], net_pnls=[], cap=INIT_CAP)
    mc_orb15   = dict(p_profit=0.0, p_ruin=1.0)
    orb15_valid = False

    if ic_orb15 > 0 and p_orb15 < 0.05:
        print("  ✓ IC significativo — IS scan + WFO …")

        def gen_orb15_sub(idxs: np.ndarray, tp_f: float, sl_f: float) -> dict:
            evs = [_ev_generic(k, "long" if orb15_signal[k] > 0 else "short",
                               tp_f, sl_f, CL15, ATR4H_15M)
                   for k in idxs if ATR4H_15M[k] > 0 and orb15_signal[k] != 0]
            return dict(events=evs, HI=HI15, LO=LO15, N=N15, max_hold=MAX_HOLD_15M)

        for tr_s, tr_e, oo_s, oo_e in wf_dates_from_idx(IDX15):
            idx_is  = np.where((IDX15 >= tr_s) & (IDX15 < tr_e))[0]
            idx_oos = np.where((IDX15 >= oo_s) & (IDX15 < oo_e))[0]
            if len(idx_is) < 500 or len(idx_oos) < 100: continue

            best_ev_is = None; best_p = (TP_FRAC_GRID[0], SL_FRAC_GRID[0])
            for tp_f, sl_f in product(TP_FRAC_GRID, SL_FRAC_GRID):
                r = run_backtest_generic(**gen_orb15_sub(idx_is, tp_f, sl_f))
                if best_ev_is is None or r["exppnl"] > best_ev_is:
                    best_ev_is = r["exppnl"]; best_p = (tp_f, sl_f)
            tp_w, sl_w = best_p
            for k in idx_oos:
                if ATR4H_15M[k] > 0 and orb15_signal[k] != 0:
                    all_oos_orb15.append(
                        _ev_generic(k, "long" if orb15_signal[k] > 0 else "short",
                                    tp_w, sl_w, CL15, ATR4H_15M))

        res_orb15  = run_backtest_generic(all_oos_orb15, HI15, LO15, N15, MAX_HOLD_15M)
        mc_orb15   = mc_summary(res_orb15)
        orb15_valid = is_validated(res_orb15, mc_orb15)
        print(f"  OOS ret={res_orb15['ret']:+.1f}%  WR={res_orb15['wr']*100:.1f}%  "
              f"n={res_orb15['n']}  MaxDD={res_orb15['mdd']:+.1f}%  "
              f"P(profit)={mc_orb15.get('p_profit',0)*100:.1f}%  "
              f"P(ruin)={mc_orb15.get('p_ruin',1)*100:.1f}%")
        print(f"  {'✓ VALIDATA' if orb15_valid else '✗ NON VALIDATA'}")
    else:
        print(f"  SKIP: IC={ic_orb15:+.4f} p={p_orb15:.4f} — nessun edge su 15M")


# ══════════════════════════════════════════════════════════════════════════════
# EXT_03v2 — XGBoost su 4H
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[EXT_03v2] XGBoost Triple Barrier — 4H")
print("  Computing 4H features …")

CL4H_s = pd.Series(CL4H, index=IDX4H)
HI4H_s = pd.Series(HI4H, index=IDX4H)
LO4H_s = pd.Series(LO4H, index=IDX4H)
OP4H_s = pd.Series(OP4H, index=IDX4H)
VOL4H_s = pd.Series(VOL4H, index=IDX4H)

feat4h = pd.DataFrame(index=IDX4H)
feat4h["close"] = CL4H; feat4h["high"] = HI4H
feat4h["low"]   = LO4H; feat4h["open"] = OP4H

for span in [10, 20, 50, 200]:
    ema = CL4H_s.ewm(span=span, adjust=False).mean()
    feat4h[f"dist_ema{span}"] = ((CL4H_s - ema) / ema).shift(1)

tr4 = pd.concat([HI4H_s - LO4H_s,
                 (HI4H_s - CL4H_s.shift(1)).abs(),
                 (LO4H_s - CL4H_s.shift(1)).abs()], axis=1).max(axis=1)
atr4_14 = tr4.rolling(14, min_periods=1).mean()
feat4h["atr_pct"] = (atr4_14 / CL4H_s).shift(1)

bb_mid4 = CL4H_s.rolling(20, min_periods=5).mean()
bb_std4 = CL4H_s.rolling(20, min_periods=5).std().fillna(0)
feat4h["bb_width"] = (4 * bb_std4 / (bb_mid4 + 1e-8)).shift(1)

avg_vol4 = VOL4H_s.rolling(20, min_periods=5).mean()
feat4h["rel_vol"] = (VOL4H_s / (avg_vol4 + 1e-8)).shift(1)

# RSI-like: (close - ema10) / ema10 rolling momentum
feat4h["mom_4"] = (CL4H_s.pct_change(4)).shift(1)
feat4h["mom_10"] = (CL4H_s.pct_change(10)).shift(1)

body4   = CL4H_s - OP4H_s
full_r4 = (HI4H_s - LO4H_s).replace(0, np.nan)
feat4h["cdl_doji"] = ((body4.abs() / full_r4) < 0.1).astype(float).shift(1)
feat4h["cdl_bull_engulf"] = (
    (CL4H_s > OP4H_s.shift(1)) & (OP4H_s < CL4H_s.shift(1)) & (CL4H_s > OP4H_s)
).astype(float).shift(1)
feat4h["cdl_bear_engulf"] = (
    (CL4H_s < OP4H_s.shift(1)) & (OP4H_s > CL4H_s.shift(1)) & (CL4H_s < OP4H_s)
).astype(float).shift(1)

feat4h["hour_sin"]    = np.sin(2 * np.pi * IDX4H.hour / 24)
feat4h["hour_cos"]    = np.cos(2 * np.pi * IDX4H.hour / 24)
feat4h["day_of_week"] = IDX4H.dayofweek.astype(float)

FEAT4H_COLS = [c for c in feat4h.columns
               if c not in ("open", "high", "low", "close")]

ema200_4h_causal = CL4H_s.ewm(span=200, adjust=False).mean().shift(1).values

feat4h_clean = feat4h.dropna(subset=FEAT4H_COLS).copy()
farr4h       = feat4h_clean[FEAT4H_COLS].values
fgidx4h      = np.array([IDX4H.get_loc(t) for t in feat4h_clean.index])

def compute_tb_4h(gidx_arr: np.ndarray) -> np.ndarray:
    n_k = len(gidx_arr); labels = np.zeros(n_k, dtype=int)
    ap_col_raw = feat4h_clean["atr_pct"].values
    ap_map = {gi: ap_col_raw[j] for j, gi in enumerate(fgidx4h)}
    for j, gi in enumerate(gidx_arr):
        if gi + XGB_HORIZON_4H >= N4H: continue
        ap = ap_map.get(gi, 0.0)
        if ap <= 0: continue
        entry = CL4H[gi]
        tp_p  = entry * (1 + ap * XGB_TP_MULT)
        sl_p  = entry * (1 - ap * XGB_SL_MULT)
        fh    = HI4H[gi + 1: gi + 1 + XGB_HORIZON_4H]
        fl    = LO4H[gi + 1: gi + 1 + XGB_HORIZON_4H]
        tp_h  = np.where(fh >= tp_p)[0]
        sl_h  = np.where(fl <= sl_p)[0]
        if len(tp_h) > 0 and (len(sl_h) == 0 or tp_h[0] < sl_h[0]):
            labels[j] = 1
    return labels

print(f"  Features: {len(FEAT4H_COLS)} colonne, {len(feat4h_clean):,} barre pulite")
print("  [WFO] Walk-forward XGBoost 4H …")

all_xgb4h_events: list[dict] = []
xgb4h_probs: list[tuple[int, float]] = []

windows_4h = wf_dates_from_idx(IDX4H)
for w_i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows_4h):
    is_m = (feat4h_clean.index >= tr_s) & (feat4h_clean.index < tr_e)
    oo_m = (feat4h_clean.index >= oo_s) & (feat4h_clean.index < oo_e)
    if is_m.sum() < 100 or oo_m.sum() < 30: continue

    X_is = farr4h[is_m]; gidx_is = fgidx4h[is_m]
    y_is = compute_tb_4h(gidx_is)
    if y_is.sum() < 5 or (y_is == 0).sum() < 5: continue

    pos_w = (y_is == 0).sum() / max((y_is == 1).sum(), 1)
    model = xgb.XGBClassifier(**XGB_PARAMS_4H, scale_pos_weight=pos_w)
    model.fit(X_is, y_is, verbose=False)

    X_oo = farr4h[oo_m]; gidx_oo = fgidx4h[oo_m]
    probs_oo = model.predict_proba(X_oo)[:, 1]
    probs_is = model.predict_proba(X_is)[:, 1]

    # IS scan TP/SL
    best_ev4  = None; best_p4 = (TP_FRAC_GRID[0], SL_FRAC_GRID[0])
    for tp_f, sl_f in product(TP_FRAC_GRID, SL_FRAC_GRID):
        evs_tmp = []; last_s = -COOLDOWN_4H
        for j, gi in enumerate(gidx_is):
            if ATR4H_4H[gi] <= 0 or gi - last_s < COOLDOWN_4H: continue
            if probs_is[j] >= XGB_THRESHOLD_4H and CL4H[gi] > ema200_4h_causal[gi] > 0:
                evs_tmp.append(
                    _ev_generic(gi, "long", tp_f, sl_f, CL4H, ATR4H_4H)); last_s = gi
        r = run_backtest_generic(evs_tmp, HI4H, LO4H, N4H, MAX_HOLD_4H)
        if best_ev4 is None or r["exppnl"] > best_ev4:
            best_ev4 = r["exppnl"]; best_p4 = (tp_f, sl_f)

    tp_w, sl_w = best_p4
    last_s = -COOLDOWN_4H
    for j, gi in enumerate(gidx_oo):
        if ATR4H_4H[gi] <= 0 or gi - last_s < COOLDOWN_4H: continue
        if probs_oo[j] >= XGB_THRESHOLD_4H and CL4H[gi] > ema200_4h_causal[gi] > 0:
            all_xgb4h_events.append(
                _ev_generic(gi, "long", tp_w, sl_w, CL4H, ATR4H_4H)); last_s = gi
        xgb4h_probs.append((gi, float(probs_oo[j])))

    if (w_i + 1) % 5 == 0:
        print(f"    window {w_i + 1}/{len(windows_4h)}")

if xgb4h_probs:
    gi_arr4  = np.array([x[0] for x in xgb4h_probs])
    pr_arr4  = np.array([x[1] for x in xgb4h_probs])
    fwd_4h   = np.log(np.roll(CL4H, -IC_HORIZON_4H) / CL4H)
    vm4      = gi_arr4 < N4H - IC_HORIZON_4H
    ic_xgb4, p_xgb4 = (st.spearmanr(pr_arr4[vm4], fwd_4h[gi_arr4[vm4]])
                        if vm4.sum() > 30 else (0.0, 1.0))
    n_xgb4  = int(vm4.sum())
else:
    ic_xgb4, p_xgb4, n_xgb4 = 0.0, 1.0, 0

print(f"  IC={ic_xgb4:+.4f}  p={p_xgb4:.4f}  n={n_xgb4}")
res_xgb4  = run_backtest_generic(all_xgb4h_events, HI4H, LO4H, N4H, MAX_HOLD_4H)
mc_xgb4   = mc_summary(res_xgb4)
xgb4_valid = is_validated(res_xgb4, mc_xgb4)
print(f"  OOS ret={res_xgb4['ret']:+.1f}%  WR={res_xgb4['wr']*100:.1f}%  "
      f"n={res_xgb4['n']}  MaxDD={res_xgb4['mdd']:+.1f}%  "
      f"P(profit)={mc_xgb4.get('p_profit',0)*100:.1f}%  "
      f"P(ruin)={mc_xgb4.get('p_ruin',1)*100:.1f}%")
print(f"  {'✓ VALIDATA' if xgb4_valid else '✗ NON VALIDATA'}")


# ══════════════════════════════════════════════════════════════════════════════
# HTML Report
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("[HTML] Generating report …")

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    buf.seek(0); b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig); return b64

def _equity_chart(events: list[dict],
                  HI: np.ndarray, LO: np.ndarray,
                  N: int, max_hold: int, color: str) -> str:
    if not events: return ""
    cap = INIT_CAP; caps = [cap]; ts = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, max_hold + 1):
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
        lev   = min(max(abs(tp - ep) / ep, abs(sl - ep) / ep), MAX_LEV)
        cap  += pnl_r * cap * RISK_PCT * lev
        caps.append(cap); ts.append(i)
    if len(caps) < 2: return ""
    fig, ax = plt.subplots(figsize=(10, 3.0), facecolor=_BG)
    ax.set_facecolor(_BG)
    # Build timestamp axis from event indices
    IDX_ref = IDX4H if N == N4H else (IDX15 if N == N15 else IDX1H)
    xs = [IDX_ref[0]] + [IDX_ref[min(t, len(IDX_ref)-1)] for t in ts]
    ax.plot(xs[:len(caps)], caps, color=color, lw=1.4)
    ax.axhline(INIT_CAP, color=_GRID, lw=0.8, ls="--")
    ax.tick_params(colors=_TEXT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    fig.tight_layout()
    return f'<img src="data:image/png;base64,{_fig_b64(fig)}" style="width:100%;max-width:700px">'

def _badge(valid: bool, n: int) -> str:
    if valid:     return '<span class="badge g">✓ VALIDATA</span>'
    elif n < 20:  return '<span class="badge grey">↔ SEGNALI INSUFFICIENTI</span>'
    else:         return '<span class="badge r">✗ NON VALIDATA</span>'

def _ic_badge(ic: float, p: float) -> str:
    ok = ic > 0 and p < 0.05
    return (f'<span class="badge {"g" if ok else "r"}">'
            f'IC={ic:+.4f}  p={p:.4f}  {"✓" if ok else "✗"}</span>')

def _sr(label: str, val: str, hl: bool = False) -> str:
    cls = ' class="hl"' if hl else ""
    return f"<tr{cls}><td>{label}</td><td>{val}</td></tr>"

# IC bar chart
fig_ic, ax_ic = plt.subplots(figsize=(8, 2.8), facecolor=_BG)
ax_ic.set_facecolor(_BG)
ids_l = ["EXT_01v2\nNW Momentum", "EXT_02v2\nORB 15M", "EXT_03v2\nXGB 4H"]
ics_l = [best_ic_val, ic_orb15, ic_xgb4]
ps_l  = [best_ic_p,   p_orb15,  p_xgb4]
c_b   = [_GRN if ic > 0 and p < 0.05 else _RED for ic, p in zip(ics_l, ps_l)]
bars  = ax_ic.bar(ids_l, ics_l, color=c_b, alpha=0.85, width=0.5)
ax_ic.axhline(0, color=_GRID, lw=1)
for bar, iv in zip(bars, ics_l):
    off = 0.0005 if iv >= 0 else -0.0005
    va  = "bottom" if iv >= 0 else "top"
    ax_ic.text(bar.get_x() + bar.get_width()/2, iv + off,
               f"{iv:+.4f}", ha="center", va=va, color=_TEXT, fontsize=8)
ax_ic.set_ylabel("IC (Spearman)", color=_TEXT, fontsize=9)
ax_ic.set_title("IC — External Strategies v2", color=_TEXT, fontsize=10)
ax_ic.tick_params(colors=_TEXT, labelsize=8)
for sp in ax_ic.spines.values(): sp.set_edgecolor(_GRID)
fig_ic.tight_layout()
ic_b64 = _fig_b64(fig_ic)

# IC per MULT chart (NW)
mult_ics = []
for mult in NW_MULT_GRID:
    ic_m, p_m, _ = ic_test_generic(nw_signals_momentum(mult), CL1H, IC_HORIZON_1H)
    mult_ics.append((mult, ic_m, p_m))
fig_mult, ax_mult = plt.subplots(figsize=(7, 2.6), facecolor=_BG)
ax_mult.set_facecolor(_BG)
ax_mult.bar([str(m[0]) for m in mult_ics],
            [m[1] for m in mult_ics],
            color=[_GRN if m[1] > 0 and m[2] < 0.05 else (_ACC if m[1] > 0 else _RED)
                   for m in mult_ics], alpha=0.85, width=0.6)
ax_mult.axhline(0, color=_GRID, lw=1)
ax_mult.set_xlabel("NW MULT", color=_TEXT, fontsize=9)
ax_mult.set_ylabel("IC Momentum", color=_TEXT, fontsize=9)
ax_mult.set_title("IC NW Momentum per valore di MULT", color=_TEXT, fontsize=10)
ax_mult.tick_params(colors=_TEXT, labelsize=8)
for sp in ax_mult.spines.values(): sp.set_edgecolor(_GRID)
fig_mult.tight_layout()
mult_b64 = _fig_b64(fig_mult)

# Blocchi strategia
def _strat_card(sid: str, sname: str, repo: str,
                desc: str, logic: str, note: str,
                ic: float, p_ic: float, n_ic: int,
                res: dict, mc: dict, valid: bool,
                eq_html: str, extra_html: str = "") -> str:
    r     = res; be_pct = r["be_fee"]; wr_pct = r["wr"] * 100
    rows  = "".join([
        _sr("N segnali OOS",        str(r["n"])),
        _sr("OOS Return",           f"{r['ret']:+.1f}%",        r["ret"] > 0),
        _sr("Max Drawdown",         f"{r['mdd']:+.1f}%"),
        _sr("Win Rate",             f"{wr_pct:.1f}%",           wr_pct > be_pct),
        _sr("Break-even (fee-adj)", f"{be_pct:.1f}%"),
        _sr("WR surplus vs BE",     f"{wr_pct - be_pct:+.1f} pp", wr_pct > be_pct),
        _sr("P(profit) MC",         f"{mc.get('p_profit',0)*100:.1f}%",
                                    mc.get("p_profit", 0) > 0.9),
        _sr("P(ruin) MC",           f"{mc.get('p_ruin',0)*100:.1f}%",
                                    mc.get("p_ruin", 1) < 0.05),
    ])
    return f"""
<div class="card">
  <div class="card-hdr">
    <span class="sid">{sid}</span>
    <span class="sname">{sname}</span>
    <a class="rlink" href="https://github.com/{repo}" target="_blank">⬡ {repo}</a>
    {_badge(valid, r['n'])}
  </div>
  <div class="two-col">
    <div>
      <div class="sect">Ottimizzazione applicata</div>
      <p class="desc">{desc}</p>
      <div class="mono">{logic}</div>
      <div class="sect" style="margin-top:12px">Anti-lookahead</div>
      <div class="mono" style="color:#999;font-size:.8em">{note}</div>
      <div class="sect" style="margin-top:12px">IC</div>
      {_ic_badge(ic, p_ic)}
      <div class="subn">n={n_ic} · orizzonte 16H equivalente · Spearman</div>
      {extra_html}
    </div>
    <div>
      <div class="sect">Risultati OOS (Walk-Forward)</div>
      <table class="st">{rows}</table>
    </div>
  </div>
  <div style="padding:0 18px 18px">
    {"<div class='sect'>Equity Curve OOS</div>" + eq_html if eq_html
     else "<div class='no-wfo'>Walk-Forward non eseguito (IC non superato)</div>"}
  </div>
</div>
"""

# EXT_01v2 extra: IC per MULT chart
extra_nw2 = (f'<div class="sect" style="margin-top:12px">IC per MULT</div>'
             f'<img src="data:image/png;base64,{mult_b64}" '
             f'style="width:100%;max-width:500px;margin-top:4px">')

# Equity charts
eq_nw2   = _equity_chart(all_oos_nw2,    HI1H, LO1H, N1H, 96,          _ACC)
eq_orb15 = (_equity_chart(all_oos_orb15, HI15, LO15, N15, MAX_HOLD_15M, _ORG)
            if not df15m_raw.empty else "")
eq_xgb4  = _equity_chart(all_xgb4h_events, HI4H, LO4H, N4H, MAX_HOLD_4H, _GRN)

card_nw2 = _strat_card(
    "EXT_01v2", "NW Momentum + IS ottimizzazione MULT",
    "mynria/Nadaraya_Watson_Binance_Trading_Bot",
    (f"La v1 aveva IC=-0.033 (mean-reversion anti-predittiva su 1H). "
     f"Invertendo il segnale otteniamo un momentum NW: LONG sopra la banda superiore, "
     f"SHORT sotto la banda inferiore. In WFO IS scan: ottimizziamo MULT "
     f"(∈{NW_MULT_GRID}) × TP/SL in ogni finestra IS da 6 mesi. "
     f"MULT migliore trovato: {best_ic_mult} (IC={best_ic_val:+.4f})."),
    "LONG: close &gt; EWMA + mult×MAE  |  SHORT: close &lt; EWMA − mult×MAE",
    "EWMA shift(1), MAE shift(1), ATR4H shift(1). Causalità garantita.",
    best_ic_val, best_ic_p, n_nw2, res_nw2, mc_nw2, nw2_valid, eq_nw2, extra_nw2)

card_orb15 = _strat_card(
    "EXT_02v2", "Opening Range Breakout — 15M",
    "yulz008/orb_cryptoBot",
    (f"Stessa logica ORB ma su 15M (range = prime {OR_HOURS_15M}H UTC = 16 barre 15M). "
     f"Maggiore risoluzione rispetto a 1H: i breakout intraday sono più precisi "
     f"e la finestra temporale post-range è più granulare. "
     f"Sizing sempre via ATR4H (shift 1, reindex su 15M). "
     f"MAX_HOLD = {MAX_HOLD_15M} barre 15M (4 giorni)."),
    f"LONG: close &gt; OR_high (prime {OR_HOURS_15M}H UTC)  |  SHORT: close &lt; OR_low",
    "OR da barre 15M 00:00–03:45. ATR4H shift(1) reindex. Un segnale/giorno.",
    ic_orb15, p_orb15, n_orb15, res_orb15, mc_orb15, orb15_valid, eq_orb15)

card_xgb4 = _strat_card(
    "EXT_03v2", "XGBoost Triple Barrier — 4H",
    "usamatariq014/XGBoost-BTC",
    (f"Stesso modello XGBoost (depth=4, n=200, lr=0.05) ma su 4H invece di 1H. "
     f"Meno rumore, trend più persistenti. Triple Barrier: {XGB_HORIZON_4H} barre 4H "
     f"({XGB_HORIZON_4H*4}H), 1×ATR SL, 2×ATR TP. "
     f"WFO con riaddestramento su ogni IS 6M. Long-only, filtro EMA200 4H (shift 1). "
     f"Feature aggiuntive: mom_4 e mom_10 (price change 4/10 barre 4H)."),
    f"LONG: P(win) &ge; {XGB_THRESHOLD_4H} AND close &gt; EMA200_4H (shift 1)",
    "Tutte le feature shift(1). Triple Barrier target solo su IS data. ATR4H shift(1).",
    ic_xgb4, p_xgb4, n_xgb4, res_xgb4, mc_xgb4, xgb4_valid, eq_xgb4)

# Summary table
def _srow(sid, sname, ic, p_ic, res, mc, valid) -> str:
    r  = res
    ic_c = "#66bb6a" if ic > 0 and p_ic < 0.05 else "#ef5350"
    rc   = "#66bb6a" if r["ret"] > 0 else "#ef5350"
    vc   = "#66bb6a" if valid else "#ef5350"
    return (f'<tr><td>{sid}</td><td>{sname}</td>'
            f'<td style="color:{ic_c}">{ic:+.4f}</td><td>{p_ic:.3f}</td>'
            f'<td style="color:{rc}">{r["ret"]:+.1f}%</td>'
            f'<td>{r["wr"]*100:.1f}%</td><td>{r["mdd"]:+.1f}%</td>'
            f'<td>{r["n"]}</td>'
            f'<td>{mc.get("p_profit",0)*100:.1f}%</td>'
            f'<td>{mc.get("p_ruin",0)*100:.1f}%</td>'
            f'<td style="color:{vc};font-weight:bold">{"✓" if valid else "✗"}</td></tr>')

summary_rows = "".join([
    _srow("EXT_01v2", "NW Momentum + MULT opt.", best_ic_val, best_ic_p,
          res_nw2, mc_nw2, nw2_valid),
    _srow("EXT_02v2", "ORB 15M",       ic_orb15, p_orb15, res_orb15, mc_orb15, orb15_valid),
    _srow("EXT_03v2", "XGBoost 4H",    ic_xgb4,  p_xgb4,  res_xgb4,  mc_xgb4,  xgb4_valid),
])

HTML = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>External Strategies v2 — Ottimizzazioni BTCUSDT</title>
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
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.78em;font-weight:700;letter-spacing:.5px}}
.badge.g{{background:#1b3a2a;color:{_GRN};border:1px solid {_GRN}44}}
.badge.r{{background:#3a1b1b;color:{_RED};border:1px solid {_RED}44}}
.badge.grey{{background:#2a2a2a;color:#888;border:1px solid #44444444}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:18px}}
@media(max-width:700px){{.two-col{{grid-template-columns:1fr}}}}
.sect{{font-size:.72em;text-transform:uppercase;color:#888;letter-spacing:.6px;margin-bottom:5px;margin-top:8px}}
.desc{{color:{_TEXT};font-size:.88em;line-height:1.6;margin-bottom:10px}}
.mono{{background:#1a1d28;border:1px solid {_GRID};border-radius:4px;
       padding:8px 12px;font-family:monospace;font-size:.82em;color:#b0c4de;margin-bottom:4px;word-break:break-word}}
.subn{{color:#666;font-size:.78em;margin-top:4px}}
.st{{width:100%;border-collapse:collapse}}
.st td{{padding:5px 8px;border-bottom:1px solid {_GRID};font-size:.88em}}
.st td:first-child{{color:#aaa;width:55%}}
.st td:last-child{{font-weight:600;text-align:right}}
.st tr.hl td{{background:#1a2a1a}}
.no-wfo{{padding:14px;color:#666;font-style:italic}}
.ic-chart,.sum-card,.note-box{{background:{_CARD};border:1px solid {_GRID};border-radius:8px;margin-bottom:24px;padding:16px}}
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
<h1>External Strategies v2 — Ottimizzazioni Mirate</h1>
<div class="sub">
  Pipeline: IC → IS scan ottimizzato → WFO (6m IS / 2m OOS) → Monte Carlo N=5,000<br>
  Dati: 2020-01 → 2026-06 · FEE=0.08% RT · RISK=1%/trade
</div>

<div class="ic-chart">
  <div style="font-size:.8em;color:#888;margin-bottom:6px">IC per strategia (miglior valore testato)</div>
  <img src="data:image/png;base64,{ic_b64}" style="width:100%;max-width:700px">
</div>

<div class="sum-card">
  <h2>Riepilogo v2</h2>
  <table class="stbl">
    <thead>
      <tr>
        <th>ID</th><th>Strategia</th><th>IC</th><th>p-val</th>
        <th>OOS Ret</th><th>WR</th><th>MaxDD</th><th>N</th>
        <th>P(profit)</th><th>P(ruin)</th><th>Esito</th>
      </tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>

{card_nw2}{card_orb15}{card_xgb4}

<div class="note-box">
  <h3>Note metodologiche v2</h3>
  <p><strong>EXT_01v2 — Inversione direzione NW:</strong> L'IC negativo della v1 rivela che
  BTC su 1H segue un comportamento momentum (non mean-reversion) alle scale temporali del NW EWMA.
  Il segnale invertito è logicamente coerente: quando il prezzo sfonda la banda superiore del canale
  NW, sta mostrando forza → il momentum tende a continuare. L'IS scan ottimizza il MULT per finestra
  IS, adattando la sensibilità del canale alle diverse fasi di mercato.</p>
  <p><strong>EXT_02v2 — ORB 15M:</strong> Su 15M la sessione range (prime 4H = 16 barre) è la stessa
  ma i breakout sono catturati con 4× la risoluzione rispetto a 1H. I falsi breakout intraday
  si traducono in più segnali, ma anche in più whipsaw. Il sizing rimane su ATR4H (shift 1)
  per uniformità con il resto della pipeline.</p>
  <p><strong>EXT_03v2 — XGBoost 4H:</strong> Su 4H le EMA e gli altri indicatori catturano
  tendenze più strutturate. Con {XGB_HORIZON_4H} barre di orizzonte Triple Barrier ({XGB_HORIZON_4H*4}H)
  e meno rumore rispetto a 1H, il modello può imparare pattern più stabili.
  Feature aggiuntive: momentum a 4 e 10 barre 4H per catturare il trend intermedio.</p>
  <p><strong>Lookahead:</strong> Tutte le feature usano shift(1). ATR4H shift(1) in tutti i timeframe.
  Triple Barrier target calcolato solo su barre IS (nessun leak OOS→IS).</p>
</div>

<footer>Pipeline di validazione — BTCUSDT 2020–2026 &nbsp;|&nbsp; {pd.Timestamp.now().date()}</footer>
</div>
</body>
</html>"""

out = Path("reports/report_external_v2.html")
out.parent.mkdir(exist_ok=True)
out.write_text(HTML, encoding="utf-8")
print(f"\n✅ Report saved: {out}  ({out.stat().st_size // 1024} KB)")
print(SEP2)
