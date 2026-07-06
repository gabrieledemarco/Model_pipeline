"""
create_github_strategies_report.py
====================================
15 strategie da repository GitHub top-starred per BTC futures.
Pipeline: IC (Spearman, 16H forward) → WFO (6m IS / 2m OOS / 2m step) → MC (N=5000)
Criterio validazione: ret_oos > 0 AND p_profit > 0.90 AND p_ruin < 0.05

Strategie:
  H01–H10 : conor19w/Binance-Futures-Trading-Bot  (662 ⭐)
  H11      : Erfaniaa/binance-futures-trading-bot   (401 ⭐)
  H12      : enarjord/passivbot                     (2k  ⭐)
  H13      : nkaz001/algotrading-example            (320 ⭐)
  H14      : MHassangit/Futures-Trading-Strategy-Evaluation (~30 ⭐)
  H15      : nkaz001/hftbacktest                    (4.2k ⭐)
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
    fetch_binance_vision_taker_flow,
    fetch_binance_vision_funding,
)
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import run_monte_carlo

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP    = 100_000.0
RISK_PCT    = 0.01
FEE         = 0.0004
FEE_RT_PCT  = FEE * 2 * 100      # 0.08 %
MAX_LEV     = 5.0
IC_HORIZON  = 16                  # 16H forward log-return (1H bars)
START_YEAR  = 2020
N_SIMS      = 5_000
COOLDOWN    = 8                   # min bars between signals
MAX_HOLD    = 96                  # 4 days in 1H bars
WARMUP      = 200                 # bars discarded for indicator warm-up

WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2

TP_GRID  = [1.0, 2.0, 3.0, 5.0]
SL_GRID  = [0.25, 0.5, 0.75, 1.0]

PBOT_SPANS = [12, 17, 24]
PBOT_DIST  = 0.01

_BG   = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _ACC  = "#42a5f5"; _GRN  = "#66bb6a"
_RED  = "#ef5350"; _YEL  = "#ffd54f"; _ORG  = "#ffa726"
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("GitHub BTC Futures Strategies — Validation Report")
print(f"  15 strategie | IC → WFO ({WF_TRAIN_M}m/{WF_OOS_M}m/{WF_STEP_M}m) → MC (N={N_SIMS:,})")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw   = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h  = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
print(f"  1H: {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

CL  = df1h["close"].values.astype(float)
HI  = df1h["high"].values.astype(float)
LO  = df1h["low"].values.astype(float)
OP  = df1h["open"].values.astype(float)
VOL = df1h["volume"].values.astype(float)

ATR1 = np.where(df1h["atr_14"].shift(1).values > 0,
                df1h["atr_14"].shift(1).values, 1.0)

print("[DATA] Loading taker flow (1H) …")
try:
    df_flow = fetch_binance_vision_taker_flow("1h", start_year=START_YEAR)
    print(f"  Flow: {len(df_flow):,} bars")
except Exception as e:
    print(f"  Flow: FAIL ({e}) — H13 disabled")
    df_flow = pd.DataFrame()

print("[DATA] Loading funding rate …")
try:
    fr_series = fetch_binance_vision_funding(start_year=START_YEAR)
    print(f"  Funding: {len(fr_series):,} 8H samples")
except Exception as e:
    print(f"  Funding: FAIL ({e}) — H15 disabled")
    fr_series = pd.Series(dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE INDICATORS (all shift(1) → causal, no lookahead)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[INDICATORS] Pre-computing …")

CL_s   = pd.Series(CL, index=IDX1H)
HI_s   = pd.Series(HI, index=IDX1H)
LO_s   = pd.Series(LO, index=IDX1H)
OP_s   = pd.Series(OP, index=IDX1H)
VOL_s  = pd.Series(VOL, index=IDX1H)

# ── RSI (14) ─────────────────────────────────────────────────────────────────
_delta  = CL_s.diff()
_avg_g  = _delta.clip(lower=0).ewm(com=13, adjust=False).mean()
_avg_l  = (-_delta).clip(lower=0).ewm(com=13, adjust=False).mean()
_rs     = _avg_g / _avg_l.replace(0, np.nan)
_rsi_r  = (100 - 100 / (1 + _rs)).fillna(50.0)
RSI     = _rsi_r.shift(1).values          # RSI[i] = RSI computed on bars 0..i-1

# ── StochRSI (14, 14, 3, 3) ──────────────────────────────────────────────────
_rsi_lo = _rsi_r.rolling(14).min()
_rsi_hi = _rsi_r.rolling(14).max()
_srsi_r = ((_rsi_r - _rsi_lo) / (_rsi_hi - _rsi_lo + 1e-9)).fillna(0.5)
_sk_r   = _srsi_r.rolling(3).mean()
_sd_r   = _sk_r.rolling(3).mean()
SRSI_K  = _sk_r.shift(1).values
SRSI_D  = _sd_r.shift(1).values

# ── MACD (12, 26, 9) ─────────────────────────────────────────────────────────
_ema12_r    = CL_s.ewm(span=12, adjust=False).mean()
_ema26_r    = CL_s.ewm(span=26, adjust=False).mean()
_macd_l_r   = _ema12_r - _ema26_r
_macd_s_r   = _macd_l_r.ewm(span=9, adjust=False).mean()
_macd_h_r   = _macd_l_r - _macd_s_r
MACD_LINE   = _macd_l_r.shift(1).values
MACD_SIG    = _macd_s_r.shift(1).values
MACD_HIST   = _macd_h_r.shift(1).values

# ── EMA levels (causal) ───────────────────────────────────────────────────────
EMA8    = df1h["ema_8"].shift(1).values
EMA21   = df1h["ema_21"].shift(1).values
EMA50   = df1h["ema_50"].shift(1).values
EMA8_p  = df1h["ema_8"].shift(2).values    # bar i-2 (for crossover detection)
EMA21_p = df1h["ema_21"].shift(2).values
EMA50_p = df1h["ema_50"].shift(2).values

# ── Bollinger Bands (causal) ──────────────────────────────────────────────────
BB_UP   = df1h["bb_up"].shift(1).values
BB_LO   = df1h["bb_lo"].shift(1).values

# ── Heikin-Ashi (causal) ──────────────────────────────────────────────────────
_ha_cl_r  = (OP_s + HI_s + LO_s + CL_s) / 4
_ha_op_r  = _ha_cl_r.ewm(alpha=0.5, adjust=False).mean().shift(1)
HA_BULL   = (_ha_cl_r > _ha_op_r).astype(float).shift(1).values

# ── Passivbot EMA Band (spans 12, 17, 24) ────────────────────────────────────
_pemas       = [CL_s.ewm(span=sp, adjust=False).mean() for sp in PBOT_SPANS]
_pema_df     = pd.concat(_pemas, axis=1)
PBOT_LOWER   = _pema_df.min(axis=1).shift(1).values
PBOT_UPPER   = _pema_df.max(axis=1).shift(1).values

# ── Volume ratio (causal) ─────────────────────────────────────────────────────
VOL_RATIO    = df1h["vol_ratio"].shift(1).values

# ── Z-score mean reversion (causal) ──────────────────────────────────────────
_zmean   = CL_s.rolling(50).mean()
_zstd    = CL_s.rolling(50).std().replace(0, np.nan)
ZSCORE   = ((CL_s - _zmean) / _zstd).fillna(0).shift(1).values

# ── Funding rate (8H → 1H ffill, causal) ─────────────────────────────────────
if len(fr_series) > 0:
    _fr_1h  = fr_series.reindex(IDX1H, method="ffill").fillna(0)
    _fr_m   = _fr_1h.rolling(72).mean()
    _fr_s   = _fr_1h.rolling(72).std().replace(0, np.nan)
    FR_Z    = ((_fr_1h - _fr_m) / _fr_s).fillna(0).shift(1).values
else:
    FR_Z    = np.zeros(N1H)

# ── CVD taker imbalance (causal) ─────────────────────────────────────────────
if len(df_flow) > 0 and "taker_buy_base" in df_flow.columns:
    _tbuy   = df_flow["taker_buy_base"].reindex(IDX1H, method="ffill").fillna(0)
    _tvol   = df_flow["volume"].reindex(IDX1H, method="ffill").fillna(1).replace(0, 1)
    _tratio = _tbuy / _tvol
    _tr_m   = _tratio.rolling(24).mean()
    _tr_s   = _tratio.rolling(24).std().replace(0, np.nan)
    TAKER_Z = ((_tratio - _tr_m) / _tr_s).fillna(0).shift(1).values
else:
    TAKER_Z = np.zeros(N1H)

print("  All indicators ready.")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _be_fee(avg_sl: float, avg_tp: float) -> float:
    sl_net = avg_sl + FEE_RT_PCT / 100
    tp_net = avg_tp - FEE_RT_PCT / 100
    return sl_net / (sl_net + tp_net) if (sl_net + tp_net) > 0 else 0.5

def _ev(i: int, direction: str, tp_f: float, sl_f: float) -> dict:
    d  = 1 if direction == "long" else -1
    a  = ATR1[i]
    ep = CL[i]
    return dict(i=i, d=d, ep=ep, tp=ep + d * tp_f * a, sl=ep - d * sl_f * a, a=a)

def run_bt(events: list, max_hold: int = MAX_HOLD) -> dict:
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, be_fee=50.0,
                    exppnl=0.0, trades=[], net_pnls=[], cap=INIT_CAP)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0
    trades: list = []; net_pnls: list = []
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
        out = "none"
        for k in range(1, max_hold + 1):
            if i + k >= N1H: break
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
    n    = len(trades)
    wr   = wins / n if n else 0.0
    ret  = (cap / INIT_CAP - 1) * 100
    if events:
        avg_tp = np.mean([abs(ev["tp"] - ev["ep"]) / ev["ep"] for ev in events]) * 100
        avg_sl = np.mean([abs(ev["sl"] - ev["ep"]) / ev["ep"] for ev in events]) * 100
        be     = _be_fee(avg_sl, avg_tp)
        sl_net = avg_sl + FEE_RT_PCT / 100
        tp_net = avg_tp - FEE_RT_PCT / 100
        ev_adj = wr * tp_net - (1 - wr) * sl_net
    else:
        be = 50.0; ev_adj = 0.0
    return dict(n=n, wr=wr, ret=ret, mdd=mdd * 100, be_fee=be * 100,
                exppnl=ev_adj, trades=trades, net_pnls=net_pnls, cap=cap)

def mc_summary(net_pnls: list) -> dict:
    if len(net_pnls) < 5:
        return dict(p_profit=0.0, p_ruin=1.0)
    arr   = np.array(net_pnls, dtype=float)
    df_mc = pd.DataFrame({"net_pnl": arr, "gross_pnl": arr,
                           "total_fees": np.zeros(len(arr))})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)),
                p_ruin=float(mc.get("p_ruin", 1.0)))

def ic_test(sig: np.ndarray) -> tuple:
    fwd  = np.log(np.roll(CL, -IC_HORIZON) / CL)
    mask = sig != 0
    mask[-IC_HORIZON:] = False
    x, y = sig[mask], fwd[mask]
    if len(x) < 30: return 0.0, 1.0, 0
    r, p = st.spearmanr(x, y)
    return float(r), float(p), int(mask.sum())

def wf_dates(IDX):
    t0 = IDX[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > IDX[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 = t0 + pd.DateOffset(months=WF_STEP_M)
    return windows

def make_events(sig: np.ndarray, idx_arr: np.ndarray, tp_f: float, sl_f: float) -> list:
    evs = []; last_s = -COOLDOWN
    for k in idx_arr:
        if k >= N1H or sig[k] == 0 or ATR1[k] <= 0: continue
        if k - last_s < COOLDOWN: continue
        evs.append(_ev(k, "long" if sig[k] > 0 else "short", tp_f, sl_f))
        last_s = k
    return evs

def is_scan(sig: np.ndarray, idx_is: np.ndarray):
    best_exppnl = -np.inf; best = (TP_GRID[0], SL_GRID[0])
    for tp_f, sl_f in product(TP_GRID, SL_GRID):
        evs = make_events(sig, idx_is, tp_f, sl_f)
        res = run_bt(evs)
        if res["exppnl"] > best_exppnl:
            best_exppnl = res["exppnl"]; best = (tp_f, sl_f)
    return best[0], best[1]

def is_validated(ret: float, mc: dict) -> bool:
    return (ret > 0
            and mc.get("p_profit", 0) > 0.90
            and mc.get("p_ruin", 1) < 0.05)

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS — all shift(1) causal, shape (N1H,)
# +1 = long, -1 = short, 0 = no signal
# ══════════════════════════════════════════════════════════════════════════════

def sig_h01() -> np.ndarray:
    """RSI(14) Mean Reversion — oversold <30 / overbought >70."""
    s = np.where(RSI < 30, 1.0, np.where(RSI > 70, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h02() -> np.ndarray:
    """RSI(14) Momentum — RSI>55 = long, RSI<45 = short."""
    s = np.where(RSI > 55, 1.0, np.where(RSI < 45, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h03() -> np.ndarray:
    """StochRSI(14,14,3,3) K/D crossover from extreme zones."""
    sk_p = np.roll(SRSI_K, 1); sk_p[0] = 0.5
    sd_p = np.roll(SRSI_D, 1); sd_p[0] = 0.5
    cross_up = (SRSI_K > SRSI_D) & (sk_p <= sd_p) & (SRSI_D < 0.30)
    cross_dn = (SRSI_K < SRSI_D) & (sk_p >= sd_p) & (SRSI_D > 0.70)
    s = np.where(cross_up, 1.0, np.where(cross_dn, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h04() -> np.ndarray:
    """MACD(12,26,9) line/signal crossover."""
    ml_p = np.roll(MACD_LINE, 1); ml_p[0] = 0
    ms_p = np.roll(MACD_SIG,  1); ms_p[0] = 0
    cross_up = (MACD_LINE > MACD_SIG) & (ml_p <= ms_p)
    cross_dn = (MACD_LINE < MACD_SIG) & (ml_p >= ms_p)
    s = np.where(cross_up, 1.0, np.where(cross_dn, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h05() -> np.ndarray:
    """EMA21/EMA50 golden/death cross."""
    e21_p = np.roll(EMA21, 1); e21_p[0] = 0
    e50_p = np.roll(EMA50, 1); e50_p[0] = 0
    cross_up = (EMA21 > EMA50) & (e21_p <= e50_p) & (EMA50 > 0)
    cross_dn = (EMA21 < EMA50) & (e21_p >= e50_p) & (EMA50 > 0)
    s = np.where(cross_up, 1.0, np.where(cross_dn, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h06() -> np.ndarray:
    """Heikin-Ashi direction + EMA21/EMA50 trend filter."""
    up_trend = (EMA21 > EMA50) & (EMA50 > 0)
    dn_trend = (EMA21 < EMA50) & (EMA50 > 0)
    ha_bull  = HA_BULL > 0.5
    ha_bear  = HA_BULL < 0.5
    s = np.where(ha_bull & up_trend, 1.0,
         np.where(ha_bear & dn_trend, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h07() -> np.ndarray:
    """Bollinger Band(20,2σ) outer band bounce — mean reversion."""
    s = np.where(CL < BB_LO, 1.0, np.where(CL > BB_UP, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h08() -> np.ndarray:
    """RSI(14) + MACD histogram alignment — dual confirmation."""
    bull = (RSI > 50) & (MACD_HIST > 0)
    bear = (RSI < 50) & (MACD_HIST < 0)
    s = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h09() -> np.ndarray:
    """StochRSI extreme levels + EMA trend filter."""
    up_trend = EMA21 > EMA50
    dn_trend = EMA21 < EMA50
    bull = (SRSI_K < 0.20) & up_trend
    bear = (SRSI_K > 0.80) & dn_trend
    s = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h10() -> np.ndarray:
    """Heikin-Ashi direction + StochRSI K/D alignment."""
    ha_bull = HA_BULL > 0.5
    ha_bear = HA_BULL < 0.5
    k_up    = SRSI_K > SRSI_D
    k_dn    = SRSI_K < SRSI_D
    s = np.where(ha_bull & k_up, 1.0,
         np.where(ha_bear & k_dn, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h11() -> np.ndarray:
    """EMA8/EMA21 crossover + high-volume confirmation (≥1.5× avg)."""
    e8_p  = np.roll(EMA8,  1); e8_p[0]  = 0
    e21_p = np.roll(EMA21, 1); e21_p[0] = 0
    hi_vol    = VOL_RATIO > 1.5
    cross_up  = (EMA8 > EMA21) & (e8_p <= e21_p) & (EMA21 > 0)
    cross_dn  = (EMA8 < EMA21) & (e8_p >= e21_p) & (EMA21 > 0)
    s = np.where(cross_up & hi_vol, 1.0,
         np.where(cross_dn & hi_vol, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h12() -> np.ndarray:
    """Passivbot EMA Band (spans 12/17/24) — price 1% outside band."""
    lower = PBOT_LOWER * (1.0 - PBOT_DIST)
    upper = PBOT_UPPER * (1.0 + PBOT_DIST)
    s = np.where(CL < lower, 1.0, np.where(CL > upper, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h13() -> np.ndarray:
    """Taker buy ratio 24H z-score: >1.5σ=long, <-1.5σ=short."""
    s = np.where(TAKER_Z > 1.5, 1.0, np.where(TAKER_Z < -1.5, -1.0, 0.0))
    s[:WARMUP] = 0
    if TAKER_Z.max() == 0 and TAKER_Z.min() == 0:
        s[:] = 0
    return s

def sig_h14() -> np.ndarray:
    """50-bar rolling z-score MR: <-2σ=long, >+2σ=short."""
    s = np.where(ZSCORE < -2.0, 1.0, np.where(ZSCORE > 2.0, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_h15() -> np.ndarray:
    """Funding rate 72H z-score: extreme → mean reversion. >1.5σ=short, <-1.5σ=long."""
    s = np.where(FR_Z < -1.5, 1.0, np.where(FR_Z > 1.5, -1.0, 0.0))
    s[:WARMUP] = 0
    if FR_Z.max() == 0 and FR_Z.min() == 0:
        s[:] = 0
    return s

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
STRATEGIES = [
    dict(id="H01", fn=sig_h01, name="RSI Mean Reversion",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="Long RSI<30, Short RSI>70. Classic oversold/overbought."),
    dict(id="H02", fn=sig_h02, name="RSI Momentum",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="Long RSI>55, Short RSI<45. RSI as trend proxy."),
    dict(id="H03", fn=sig_h03, name="StochRSI Crossover",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="K crosses D from extreme zones (<0.30 / >0.70). Reversal timing."),
    dict(id="H04", fn=sig_h04, name="MACD Crossover",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="MACD(12,26,9) line crosses signal line. Classic trend entry."),
    dict(id="H05", fn=sig_h05, name="EMA 21/50 Cross",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="Golden/death cross EMA21 vs EMA50. Trend-following on confirmed breakout."),
    dict(id="H06", fn=sig_h06, name="Heikin-Ashi + EMA Trend",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="HA bullish/bearish filtered by EMA21>EMA50 trend alignment."),
    dict(id="H07", fn=sig_h07, name="Bollinger Band Bounce",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="Price touches outer BB(20,2σ). Mean reversion from extremes."),
    dict(id="H08", fn=sig_h08, name="RSI + MACD Combo",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="RSI>50 AND MACD hist>0 = long. Dual-indicator confirmation."),
    dict(id="H09", fn=sig_h09, name="StochRSI + EMA Filter",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="StochRSI extreme (<0.20/>0.80) filtered by EMA trend direction."),
    dict(id="H10", fn=sig_h10, name="Heikin-Ashi + StochRSI",
         repo="conor19w/Binance-Futures-Trading-Bot",
         desc="HA direction aligned with StochRSI K vs D momentum."),
    dict(id="H11", fn=sig_h11, name="EMA 8/21 + Volume",
         repo="Erfaniaa/binance-futures-trading-bot",
         desc="EMA8 crosses EMA21 with volume ≥1.5× average. Volume-confirmed trend."),
    dict(id="H12", fn=sig_h12, name="Passivbot EMA Band",
         repo="enarjord/passivbot",
         desc="Triple EMA band (12/17/24). Price 1% outside band triggers mean-reversion entry."),
    dict(id="H13", fn=sig_h13, name="CVD Taker Imbalance",
         repo="nkaz001/algotrading-example",
         desc="Taker buy ratio 24H z-score. >1.5σ = buy pressure → long. <-1.5σ → short."),
    dict(id="H14", fn=sig_h14, name="Z-score Mean Reversion",
         repo="MHassangit/Futures-Trading-Strategy-Evaluation",
         desc="50-bar rolling z-score. Price >2σ from mean → short, <-2σ → long."),
    dict(id="H15", fn=sig_h15, name="Funding Rate Reversion",
         repo="nkaz001/hftbacktest",
         desc="8H funding rate 72H z-score. Extreme funding predicts reversal. >1.5σ → short."),
]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
WF_WINDOWS = wf_dates(IDX1H)
print(f"\n[WFO] {len(WF_WINDOWS)} windows "
      f"({WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / {WF_STEP_M}m step)\n")

RESULTS = []

for strat in STRATEGIES:
    sid  = strat["id"]
    name = strat["name"]
    print(f"{SEP}\n[{sid}] {name}")
    print(f"  Source: {strat['repo']}")

    sig = strat["fn"]()
    n_long  = int((sig > 0).sum())
    n_short = int((sig < 0).sum())
    print(f"  Signals: {n_long+n_short:,}  (L={n_long:,} / S={n_short:,})")

    # ── IC test ─────────────────────────────────────────────────────────────
    ic, p_ic, n_ic = ic_test(sig)
    ic_pass = (abs(ic) > 0) and (p_ic < 0.05)
    dir_ok  = ic > 0     # True → use signal as-is; False → flip direction

    print(f"  IC: {ic:+.4f}  p={p_ic:.4f}  n={n_ic:,}  "
          f"→ {'PASS ✓' if ic_pass else 'FAIL ✗'}")

    rec = dict(
        id=sid, name=name, repo=strat["repo"], desc=strat["desc"],
        n_long=n_long, n_short=n_short,
        ic=ic, p_ic=p_ic, n_ic=n_ic, ic_pass=ic_pass,
        wfo_run=False, oos_n=0, oos_wr=0.0, oos_ret=0.0, oos_mdd=0.0,
        mc_p_profit=0.0, mc_p_ruin=1.0, validated=False,
        oos_pnls=[], oos_equity=[INIT_CAP],
    )

    if not ic_pass:
        RESULTS.append(rec)
        continue

    # Align direction: if IC<0, flip signal (mean-reversion of signal)
    sig_used = sig if dir_ok else (-sig)

    # ── WFO ─────────────────────────────────────────────────────────────────
    print(f"  WFO …", end="", flush=True)
    rec["wfo_run"] = True
    all_oos_evs: list[dict] = []

    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is  = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 200 or len(idx_oos) < 50:
            continue

        tp_f, sl_f = is_scan(sig_used, idx_is)
        oos_evs = make_events(sig_used, idx_oos, tp_f, sl_f)
        all_oos_evs.extend(oos_evs)
        print(".", end="", flush=True)

    print()

    if not all_oos_evs:
        print("  No OOS events — skip")
        RESULTS.append(rec)
        continue

    # Aggregate OOS across all windows
    res_oos = run_bt(all_oos_evs)
    oos_ret = res_oos["ret"]
    oos_wr  = res_oos["wr"]
    oos_n   = res_oos["n"]
    oos_mdd = res_oos["mdd"]

    # Rebuild equity curve from OOS pnls (cumulative)
    oos_eq  = [INIT_CAP] + list(np.cumsum([INIT_CAP] + res_oos["net_pnls"])[1:])

    print(f"  OOS: trades={oos_n}  wr={oos_wr:.1%}  "
          f"ret={oos_ret:+.1f}%  mdd={oos_mdd:.1f}%")

    # ── Monte Carlo ──────────────────────────────────────────────────────────
    mc = mc_summary(res_oos["net_pnls"])
    print(f"  MC:  p_profit={mc['p_profit']:.3f}  p_ruin={mc['p_ruin']:.3f}")

    valid = is_validated(oos_ret, mc)
    print(f"  {'VALIDATED ✅' if valid else 'NOT VALIDATED ❌'}")

    rec.update(
        oos_n=oos_n, oos_wr=oos_wr, oos_ret=oos_ret, oos_mdd=oos_mdd,
        mc_p_profit=mc["p_profit"], mc_p_ruin=mc["p_ruin"],
        validated=valid,
        oos_pnls=res_oos["net_pnls"],
        oos_equity=oos_eq,
    )
    RESULTS.append(rec)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("SUMMARY")
print(f"{SEP2}")
n_ic_pass  = sum(1 for r in RESULTS if r["ic_pass"])
n_wfo_run  = sum(1 for r in RESULTS if r["wfo_run"])
n_valid    = sum(1 for r in RESULTS if r["validated"])
print(f"  IC pass : {n_ic_pass}/15")
print(f"  WFO run : {n_wfo_run}/15")
print(f"  Validated: {n_valid}/15")
for r in RESULTS:
    status = ("✅ VALIDATED" if r["validated"]
              else ("⚠ IC PASS" if r["ic_pass"] else "✗ IC FAIL"))
    print(f"  [{r['id']}] {r['name']:<30s}  IC={r['ic']:+.4f} p={r['p_ic']:.3f}"
          + (f"  OOS ret={r['oos_ret']:+.1f}%  MC p_profit={r['mc_p_profit']:.2f}"
             if r["wfo_run"] else "")
          + f"  {status}")
print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# EQUITY CHART HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _equity_chart(rec: dict) -> str:
    eq = rec["oos_equity"]
    if len(eq) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(9, 3), facecolor=_BG)
    ax.set_facecolor(_BG)
    x = range(len(eq))
    color = _GRN if eq[-1] >= INIT_CAP else _RED
    ax.plot(x, eq, color=color, lw=1.5)
    ax.axhline(INIT_CAP, color=_GRID, ls="--", lw=0.8)
    ax.set_ylabel("Equity ($)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(alpha=0.15, color=_GRID)
    ax.set_title(f"{rec['id']} {rec['name']} — OOS equity", color=_TEXT, fontsize=9)
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:820px;">'


def _ic_bar_chart() -> str:
    ids   = [r["id"] for r in RESULTS]
    ics   = [r["ic"] for r in RESULTS]
    colors = [_GRN if (r["ic_pass"] and r["ic"] > 0)
               else (_YEL if r["ic_pass"] else _RED)
               for r in RESULTS]
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=_BG)
    ax.set_facecolor(_BG)
    bars = ax.bar(ids, ics, color=colors, edgecolor=_GRID, linewidth=0.5)
    ax.axhline(0, color=_TEXT, lw=0.8)
    ax.set_ylabel("IC (Spearman)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(axis="y", alpha=0.2, color=_GRID)
    ax.set_title("Information Coefficient — all 15 strategies", color=_TEXT, fontsize=10)
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:900px;">'


def _oos_ret_chart() -> str:
    ran = [r for r in RESULTS if r["wfo_run"]]
    if not ran:
        return ""
    ids    = [r["id"] for r in ran]
    rets   = [r["oos_ret"] for r in ran]
    colors = [_GRN if v > 0 else _RED for v in rets]
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.bar(ids, rets, color=colors, edgecolor=_GRID, linewidth=0.5)
    ax.axhline(0, color=_TEXT, lw=0.8)
    ax.set_ylabel("OOS Return (%)", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(axis="y", alpha=0.2, color=_GRID)
    ax.set_title("OOS Return — WFO strategies", color=_TEXT, fontsize=10)
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:900px;">'

# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def status_badge(r: dict) -> str:
    if r["validated"]:
        return '<span class="badge green">VALIDATED</span>'
    if r["wfo_run"]:
        return '<span class="badge orange">WFO FAIL</span>'
    if r["ic_pass"]:
        return '<span class="badge yellow">IC PASS / NO OOS</span>'
    return '<span class="badge red">IC FAIL</span>'

def fmt_pct(v, digits=1):
    return f"{v:+.{digits}f}%"

def fmt_f(v, digits=3):
    return f"{v:.{digits}f}"

ic_bar  = _ic_bar_chart()
oos_bar = _oos_ret_chart()

strategy_sections = ""
for r in RESULTS:
    eq_img = _equity_chart(r) if r["wfo_run"] and r["oos_n"] > 0 else ""
    wfo_row = ""
    if r["wfo_run"]:
        wfo_row = f"""
        <tr><td>OOS Trades</td><td>{r['oos_n']}</td></tr>
        <tr><td>OOS Win Rate</td><td>{r['oos_wr']:.1%}</td></tr>
        <tr><td>OOS Return</td><td class="{'green' if r['oos_ret']>0 else 'red'}">{fmt_pct(r['oos_ret'])}</td></tr>
        <tr><td>OOS Max DD</td><td class="red">{fmt_pct(r['oos_mdd'])}</td></tr>
        <tr><td>MC P(profit)</td><td class="{'green' if r['mc_p_profit']>0.9 else 'yellow'}">{fmt_f(r['mc_p_profit'])}</td></tr>
        <tr><td>MC P(ruin)</td><td class="{'green' if r['mc_p_ruin']<0.05 else 'red'}">{fmt_f(r['mc_p_ruin'])}</td></tr>
        """
    strategy_sections += f"""
    <div class="card strat-card">
      <div class="strat-header">
        <span class="strat-id">{r['id']}</span>
        <span class="strat-name">{r['name']}</span>
        {status_badge(r)}
      </div>
      <div class="strat-meta">Source: <a href="https://github.com/{r['repo']}" class="link">{r['repo']}</a></div>
      <div class="strat-desc">{r['desc']}</div>
      <table class="metrics-tbl">
        <tr><td>Signals (L/S)</td><td>{r['n_long']:,} / {r['n_short']:,}</td></tr>
        <tr><td>IC</td><td class="{'green' if r['ic']>0 else 'red'}">{r['ic']:+.4f}</td></tr>
        <tr><td>IC p-value</td><td class="{'green' if r['p_ic']<0.05 else 'red'}">{r['p_ic']:.4f}</td></tr>
        <tr><td>IC n</td><td>{r['n_ic']:,}</td></tr>
        {wfo_row}
      </table>
      {eq_img}
    </div>
"""

summary_rows = ""
for r in RESULTS:
    oos_ret_str = fmt_pct(r["oos_ret"]) if r["wfo_run"] else "—"
    oos_mdd_str = fmt_pct(r["oos_mdd"]) if r["wfo_run"] else "—"
    mc_pp_str   = fmt_f(r["mc_p_profit"]) if r["wfo_run"] else "—"
    mc_pr_str   = fmt_f(r["mc_p_ruin"])   if r["wfo_run"] else "—"
    oos_n_str   = str(r["oos_n"]) if r["wfo_run"] else "—"
    oos_wr_str  = f"{r['oos_wr']:.1%}" if r["wfo_run"] else "—"
    r_cls = "green" if r["ic"] > 0 else "red"
    summary_rows += f"""<tr>
      <td>{r['id']}</td>
      <td>{r['name']}</td>
      <td class="{r_cls}">{r['ic']:+.4f}</td>
      <td class="{'green' if r['p_ic']<0.05 else 'red'}">{r['p_ic']:.4f}</td>
      <td>{oos_n_str}</td>
      <td>{oos_wr_str}</td>
      <td class="{'green' if r['oos_ret']>0 else 'red'}">{oos_ret_str}</td>
      <td class="red">{oos_mdd_str}</td>
      <td class="{'green' if r['mc_p_profit']>0.9 else 'red'}">{mc_pp_str}</td>
      <td class="{'green' if r['mc_p_ruin']<0.05 else 'red'}">{mc_pr_str}</td>
      <td>{status_badge(r)}</td>
    </tr>
"""

n_valid = sum(1 for r in RESULTS if r["validated"])
n_ic_pass = sum(1 for r in RESULTS if r["ic_pass"])

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GitHub BTC Strategies — Validation Report</title>
<style>
  :root {{
    --bg: {_BG}; --card: {_CARD}; --grid: {_GRID};
    --text: {_TEXT}; --acc: {_ACC};
    --green: {_GRN}; --red: {_RED}; --yellow: {_YEL}; --orange: {_ORG};
  }}
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          font-size:14px; line-height:1.5; padding:24px; }}
  h1 {{ color:var(--acc); font-size:22px; margin-bottom:4px; }}
  h2 {{ color:var(--acc); font-size:16px; margin:24px 0 10px; }}
  .subtitle {{ color:#888; font-size:13px; margin-bottom:24px; }}
  .kpi-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
  .kpi {{ background:var(--card); border:1px solid var(--grid); border-radius:8px;
          padding:14px 20px; min-width:140px; }}
  .kpi .val {{ font-size:28px; font-weight:700; color:var(--acc); }}
  .kpi .lbl {{ font-size:11px; color:#888; margin-top:2px; }}
  .card {{ background:var(--card); border:1px solid var(--grid); border-radius:8px;
           padding:16px; margin-bottom:16px; }}
  .strat-card {{ border-left:3px solid var(--grid); }}
  .strat-header {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; flex-wrap:wrap; }}
  .strat-id {{ background:var(--acc); color:#000; font-weight:700; border-radius:4px;
               padding:2px 8px; font-size:13px; }}
  .strat-name {{ font-size:16px; font-weight:600; color:var(--text); }}
  .strat-meta {{ color:#666; font-size:11px; margin-bottom:4px; }}
  .strat-desc {{ color:#aaa; font-size:12px; margin-bottom:10px; }}
  .metrics-tbl {{ border-collapse:collapse; width:100%; max-width:420px; margin-bottom:12px; }}
  .metrics-tbl td {{ padding:3px 10px; border-bottom:1px solid var(--grid); font-size:13px; }}
  .metrics-tbl td:first-child {{ color:#888; width:45%; }}
  .green {{ color:var(--green) !important; }}
  .red   {{ color:var(--red) !important; }}
  .yellow{{ color:var(--yellow) !important; }}
  .badge {{ display:inline-block; border-radius:4px; padding:2px 8px;
            font-size:11px; font-weight:600; }}
  .badge.green  {{ background:#1b3a1e; color:var(--green); }}
  .badge.red    {{ background:#3a1a1a; color:var(--red); }}
  .badge.orange {{ background:#3a2a00; color:var(--orange); }}
  .badge.yellow {{ background:#2d2a00; color:var(--yellow); }}
  table.summary-tbl {{ width:100%; border-collapse:collapse; overflow-x:auto; display:block; }}
  table.summary-tbl th {{ background:var(--grid); color:var(--acc); text-align:left;
                           padding:6px 10px; font-size:12px; white-space:nowrap; }}
  table.summary-tbl td {{ padding:5px 10px; border-bottom:1px solid var(--grid);
                           font-size:12px; white-space:nowrap; }}
  table.summary-tbl tr:hover td {{ background:var(--grid); }}
  .link {{ color:var(--acc); text-decoration:none; }}
  .link:hover {{ text-decoration:underline; }}
  .section-desc {{ color:#888; font-size:12px; margin-bottom:14px; }}
  img {{ display:block; margin-bottom:12px; border-radius:6px; }}
  .repos-list {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .repo-tag {{ background:var(--grid); border-radius:4px; padding:4px 10px; font-size:12px; }}
</style>
</head>
<body>

<h1>GitHub BTC Futures — Validation Report</h1>
<p class="subtitle">
  15 strategie da 7 repository top-starred.
  Pipeline: IC (Spearman 16H) → WFO (6m/2m/2m) → Monte Carlo (N={N_SIMS:,}) &nbsp;|&nbsp;
  Data: {IDX1H[0].date()} – {IDX1H[-1].date()} · {N1H:,} bars 1H
</p>

<div class="kpi-row">
  <div class="kpi"><div class="val">15</div><div class="lbl">Strategie testate</div></div>
  <div class="kpi"><div class="val">{n_ic_pass}</div><div class="lbl">IC pass (p&lt;0.05)</div></div>
  <div class="kpi"><div class="val">{sum(1 for r in RESULTS if r['wfo_run'])}</div><div class="lbl">WFO eseguito</div></div>
  <div class="kpi"><div class="val" style="color:{'var(--green)' if n_valid>0 else 'var(--red)'}">{n_valid}</div><div class="lbl">Validate</div></div>
</div>

<div class="card">
  <h2 style="margin-top:0">Repository sorgente</h2>
  <div class="repos-list">
    <div class="repo-tag">conor19w/Binance-Futures-Trading-Bot (662⭐) — H01–H10</div>
    <div class="repo-tag">Erfaniaa/binance-futures-trading-bot (401⭐) — H11</div>
    <div class="repo-tag">enarjord/passivbot (2k⭐) — H12</div>
    <div class="repo-tag">nkaz001/algotrading-example (320⭐) — H13</div>
    <div class="repo-tag">MHassangit/Futures-Trading-Strategy-Evaluation (30⭐) — H14</div>
    <div class="repo-tag">nkaz001/hftbacktest (4.2k⭐) — H15</div>
  </div>
</div>

<h2>IC Overview</h2>
<div class="card">{ic_bar}</div>

{"<h2>OOS Return Overview</h2><div class='card'>" + oos_bar + "</div>" if oos_bar else ""}

<h2>Summary Table</h2>
<div class="card">
<table class="summary-tbl">
<thead>
  <tr>
    <th>ID</th><th>Strategy</th><th>IC</th><th>p-val</th>
    <th>OOS n</th><th>OOS WR</th><th>OOS Ret</th><th>OOS MDD</th>
    <th>P(profit)</th><th>P(ruin)</th><th>Status</th>
  </tr>
</thead>
<tbody>
{summary_rows}
</tbody>
</table>
</div>

<h2>Strategy Details</h2>
<p class="section-desc">
  Criteri di validazione: OOS ret &gt; 0% · MC P(profit) &gt; 90% · MC P(ruin) &lt; 5%
</p>
{strategy_sections}

<hr style="border-color:var(--grid);margin:32px 0 16px">
<p style="color:#555;font-size:11px;text-align:center;">
  BTCUSDT perpetual futures · Binance Vision CDN · 2020–2026 · 1H bars<br>
  WFO: 6m IS / 2m OOS / 2m step · MC bootstrap N=5,000 · Risk 1% per trade · Fee 0.08% RT
</p>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_path = Path("reports/report_github_strategies.html")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"\n[DONE] Report saved → {out_path}  ({out_path.stat().st_size//1024} KB)")
