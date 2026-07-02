"""
create_setup_ranking_report.py
===============================
25 BTCUSDT setup ideas built on volume confirmation, structure breaks,
clean retests and momentum shifts.

For each setup: trigger, invalidation level, exact no-trade condition.
All ranked by IC (Spearman signal vs 4H forward return).
Full validation pipeline (IS scan → Walk-Forward → Monte Carlo) applied
to the setup with the best statistical significance.
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
from src.strategy.indicators   import add_indicators, ema as _ema
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
IC_HORIZON   = 16        # 4H forward return
START_YEAR   = 2020
N_SIMS       = 5_000
LBK          = 20        # rolling lookback for most structural setups

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

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("25 Setup Ranking — BTCUSDT 15M | 2020-2026")
print(SEP2)
print("[DATA] Caricamento …")

raw   = fetch_extended_data(start_year=START_YEAR, start_month=1,
                             fetch_15m=True, fetch_1m=False, fetch_flow=False)
df    = add_indicators(raw["15M"])
df1h  = add_indicators(raw["1H"])

print(f"  15M: {len(df):,} bar  ({df.index[0].date()} → {df.index[-1].date()})")

# ── numpy arrays
IDX   = df.index;  N = len(df)
HI    = df["high"].values;    LO    = df["low"].values
CL    = df["close"].values;   OP    = df["open"].values
VOL   = df["volume"].values
VR    = df["vol_ratio"].values          # vol / sma20
ATR   = df["atr_14"].values
OBV_  = df["obv"].values
OBVT  = df["obv_trend"].values          # sign(obv - obv_ema21)
RSI14 = df["rsi_14"].values
MHIST = df["macd_hist"].values
EMA21 = df["ema_21"].values
EMA50 = df["ema_50"].values
ADX_  = df["adx"].values
BBSQ  = df["bb_squeeze"].values         # (bb_up - bb_lo) / bb_mid
yr_   = np.array([t.year for t in IDX], dtype=int)

prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
atr_1h_map = df1h["atr_14"].clip(lower=1.0).to_dict()
fb         = df["atr_14"].clip(lower=1.0).values
ATR1H      = np.array([atr_1h_map.get(t, np.nan) for t in prev_1h], dtype=float)
ATR1H      = np.where(np.isnan(ATR1H), fb, ATR1H)

# 4H EMA20 bias (for composite setups)
df4h      = df1h.resample("4h").agg({"close": "last"}).dropna()
ema4h     = df4h["close"].ewm(span=20, adjust=False).mean()
bias4h_d  = (df4h["close"] >= ema4h).to_dict()

def _4h_bias(ts: pd.Timestamp):
    p = ts.floor("4h") - pd.Timedelta("4h")
    return bias4h_d.get(p, None)

# Rolling 20-bar swing high/low (close-based)
swing_hi_cl = pd.Series(CL).rolling(LBK).max().values
swing_lo_cl = pd.Series(CL).rolling(LBK).min().values
# Rolling 20-bar swing high/low (wick-based)
swing_hi_wk = pd.Series(HI).rolling(LBK).max().values
swing_lo_wk = pd.Series(LO).rolling(LBK).min().values
# 20-bar avg ATR and BB squeeze
atr_ma20    = pd.Series(ATR).rolling(LBK).mean().values
bbsq_ma20   = pd.Series(BBSQ).rolling(LBK).mean().values

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
# BACKTEST ENGINE (for full validation of winning setup)
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
        entry = ev["entry_px"]; ref_size = max(ev["ref_size"], 1e-6)
        atr   = ev["atr_1h"];   d = 1 if ev["direction"] == "long" else -1
        tp_px = entry + d * tp_frac * ref_size
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
        wr = wins / n_v * 100; rr = (sum_tp/n_v) / (sum_sl/n_v) if sum_sl > 0 else 0
        be = 1/(1+rr)*100 if rr > 0 else 50.0
        exp = wr/100*(sum_tp/n_v) - (1-wr/100)*(sum_sl/n_v)
        p = st.binomtest(int(round(wr/100*n_v)), n_v, be/100, alternative="greater").pvalue
        results.append(dict(tp_frac=tp_f, sl_buf=sl_b, n=n_v, wr=round(wr,2),
                            rr=round(rr,2), be=round(be,2), exp=round(exp,5), p_val=round(p,4)))
    df = pd.DataFrame(results)
    if df.empty: return df, dict(tp_frac=2.0, sl_buf=0.5, n=0, wr=0, rr=0, be=50, exp=0, p_val=1)
    return df, df.sort_values("exp", ascending=False).iloc[0].to_dict()

def _kpis(eq, init=INIT_CAP):
    if eq.empty: return dict(total_return=0, calmar=0, sharpe=0, max_dd=0)
    full = pd.concat([pd.Series([init], index=[eq.index[0]-pd.Timedelta("1s")]), eq])
    dd = (full / full.cummax() - 1).min()
    ret = full.iloc[-1] / init - 1
    rets = full.pct_change().dropna(); vol = rets.std() * np.sqrt(365*24)
    ann = rets.mean() * 365*24
    return dict(total_return=ret, calmar=ret/abs(dd) if dd<0 else 0,
                sharpe=ann/vol if vol>0 else 0, max_dd=dd)

def run_wf(events, tp_frac, sl_buf):
    from dateutil.relativedelta import relativedelta
    start, end = IDX[0], IDX[-1]; wins = []; cur = start
    while True:
        tr_e = cur + relativedelta(months=WF_TRAIN_M)
        oo_e = tr_e + relativedelta(months=WF_OOS_M)
        if oo_e > end: break
        wins.append((cur, tr_e, oo_e)); cur = cur + relativedelta(months=WF_STEP_M)
    wr_list = []; all_oos = []
    for wid, (tr_s, tr_e, oo_e) in enumerate(wins):
        is_ev = [e for e in events if tr_s <= e["ts"] < tr_e]
        oo_ev = [e for e in events if tr_e <= e["ts"] < oo_e]
        if len(is_ev) < 5 or len(oo_ev) < 3: continue
        _, oo_eq = run_backtest(oo_ev, tp_frac, sl_buf, window_id=wid)
        if oo_eq.empty: continue
        wr_list.append(_kpis(oo_eq)["total_return"]); all_oos.extend(oo_ev)
    oos = sorted(all_oos, key=lambda e: e["ts"])
    oos_t, oos_eq = run_backtest(oos, tp_frac, sl_buf)
    return oos_t, oos_eq, wr_list, len(wins)


# ══════════════════════════════════════════════════════════════════════════════
# 25 SETUP DEFINITIONS
# Each returns list[dict] with at minimum:
#   direction, entry_i, entry_px, ref_lo, ref_hi, ref_size, atr_1h, year, ts
# ══════════════════════════════════════════════════════════════════════════════

# ── GROUP A: VOLUME ──────────────────────────────────────────────────────────

def A1_vol_spike_reversal():
    """
    Trigger   : vol > 3× avg20; candle body reverses prior 3-bar direction (body > 0.4× ATR)
    Invalidate: close beyond spike extreme in continuation direction
    No-trade  : vol < 3× avg OR trend direction same as current body (no reversal context)
    """
    evs = []
    for i in range(LBK + 3, N - MAX_HOLD - 2):
        if VR[i] < 3.0: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        prior_move = CL[i-1] - CL[i-3]   # 3-bar price move before spike
        body = CL[i] - OP[i]
        if prior_move > 0.3*atr and body < -0.4*atr:     # bullish run → spike reversal down
            evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                            ref_lo=CL[i], ref_hi=HI[i], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
        elif prior_move < -0.3*atr and body > 0.4*atr:   # bearish run → spike reversal up
            evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                            ref_lo=LO[i], ref_hi=CL[i], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def A2_vol_dry_retest():
    """
    Trigger   : BOS (CL > 20-bar swing high), then retest within 8 bars with vol < 0.7× avg
    Invalidate: close below the broken swing level
    No-trade  : retest volume ≥ 0.7× avg (not dry; sellers present)
    """
    evs = []
    bos_levels = {}   # idx → break_level
    for i in range(LBK + 1, N - MAX_HOLD - 10):
        if CL[i] > swing_hi_cl[i-1]:
            bos_levels[i] = swing_hi_cl[i-1]
    for bos_i, level in bos_levels.items():
        for j in range(bos_i + 1, min(bos_i + 9, N - MAX_HOLD)):
            if CL[j] < level - ATR1H[j] * 0.5: break   # fell too far
            if LO[j] <= level + ATR1H[j] * 0.3 and VR[j] < 0.7:
                evs.append(dict(direction="long", entry_i=j, entry_px=CL[j],
                                ref_lo=level - 0.3*ATR1H[j], ref_hi=level + 0.5*ATR1H[j],
                                ref_size=ATR1H[j], atr_1h=ATR1H[j], year=yr_[j], ts=IDX[bos_i]))
                break
    return evs

def A3_vol_bos_continuation():
    """
    Trigger   : CL > 20-bar swing high with vol > 1.5× avg → momentum continuation
    Invalidate: close back below the broken level within 3 bars
    No-trade  : RSI > 80 (overbought) or vol < 1.5× avg
    """
    evs = []
    for i in range(LBK + 1, N - MAX_HOLD - 2):
        if CL[i] <= swing_hi_cl[i-1]: continue
        if VR[i] < 1.5: continue
        if RSI14[i] > 80: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=swing_hi_cl[i-1], ref_hi=CL[i], ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def A4_obv_divergence():
    """
    Trigger   : Price at 20-bar extreme but OBV not confirming (divergence)
    Invalidate: price closes beyond divergence extreme
    No-trade  : OBV within 3% of its own extreme (no meaningful divergence)
    """
    evs = []
    obv_hi = pd.Series(OBV_).rolling(LBK).max().values
    obv_lo = pd.Series(OBV_).rolling(LBK).min().values
    for i in range(LBK + 1, N - MAX_HOLD - 2):
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        # Bearish div: price new high but OBV not at new high
        if HI[i] >= swing_hi_wk[i-1]:
            if OBV_[i] < obv_hi[i] * 0.97:
                evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                                ref_lo=CL[i] - atr, ref_hi=HI[i], ref_size=atr,
                                atr_1h=atr, year=yr_[i], ts=IDX[i]))
        # Bullish div: price new low but OBV not at new low
        elif LO[i] <= swing_lo_wk[i-1]:
            if OBV_[i] > obv_lo[i] * 1.03:
                evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                                ref_lo=LO[i], ref_hi=CL[i] + atr, ref_size=atr,
                                atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def A5_vol_climax_absorption():
    """
    Trigger   : vol > 4× avg; candle body < 30% of full range (absorption wick)
    Invalidate: next bar continues in wick direction (no absorption)
    No-trade  : body > 30% of range (not absorption) or vol < 4× avg
    """
    evs = []
    for i in range(LBK, N - MAX_HOLD - 2):
        if VR[i] < 4.0: continue
        full_range = HI[i] - LO[i]
        if full_range < 1e-6: continue
        body = abs(CL[i] - OP[i])
        if body / full_range > 0.30: continue   # not absorption
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        # Long wick below (buying pressure): absorption of sellers → long
        if LO[i] < min(OP[i], CL[i]) - 0.3*full_range:
            evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                            ref_lo=LO[i], ref_hi=CL[i]+atr, ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
        # Long wick above → absorption of buyers → short
        elif HI[i] > max(OP[i], CL[i]) + 0.3*full_range:
            evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                            ref_lo=CL[i]-atr, ref_hi=HI[i], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

# ── GROUP B: STRUCTURE BREAKS ────────────────────────────────────────────────

def B1_bos_close():
    """
    Trigger   : CL > 20-bar highest close (clean close-based structure break)
    Invalidate: close below the broken level
    No-trade  : ADX < 15 (no trend context)
    """
    evs = []
    for i in range(LBK + 1, N - MAX_HOLD - 2):
        if CL[i] <= swing_hi_cl[i-1]: continue
        if ADX_[i] < 15: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=swing_hi_cl[i-1], ref_hi=CL[i], ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def B2_bos_first_retest():
    """
    Trigger   : First bar touching broken swing level (within 0.4 ATR) after BOS, within 10 bars
    Invalidate: close below the broken level - 0.5 ATR
    No-trade  : retest happens before price moved ≥ 1 ATR above broken level (too tight)
    """
    evs = []
    for i in range(LBK + 1, N - MAX_HOLD - 12):
        if CL[i] <= swing_hi_cl[i-1]: continue
        level = swing_hi_cl[i-1]; atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        if CL[i] - level < atr: continue    # no-trade: moved less than 1 ATR above
        for j in range(i + 1, min(i + 11, N - MAX_HOLD)):
            if CL[j] < level - 0.5*ATR1H[j]: break
            if LO[j] <= level + 0.4*ATR1H[j] and CL[j] >= level - 0.1*ATR1H[j]:
                evs.append(dict(direction="long", entry_i=j, entry_px=CL[j],
                                ref_lo=level - 0.4*ATR1H[j], ref_hi=level + 0.4*ATR1H[j],
                                ref_size=ATR1H[j], atr_1h=ATR1H[j], year=yr_[j], ts=IDX[i]))
                break
    return evs

def B3_failed_bos_long():
    """
    Trigger   : Wick below 20-bar swing low but CLOSE back above it (fakeout squeeze → long)
    Invalidate: close below the original swing low
    No-trade  : prior 3-bar trend already bearish (wick in direction of trend = not a fakeout)
    """
    evs = []
    for i in range(LBK + 1, N - MAX_HOLD - 2):
        if LO[i] >= swing_lo_wk[i-1]: continue      # must wick below swing low
        if CL[i] <= swing_lo_wk[i-1]: continue      # must close back above
        prior = CL[i-1] - CL[i-3]
        if prior < 0: continue                       # no-trade: already in downtrend
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=LO[i], ref_hi=CL[i]+atr, ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def B4_choch_uptrend():
    """
    Trigger   : In downtrend (3 consecutive lower highs), first close above most recent swing high
    Invalidate: close below the CHoCH swing high level
    No-trade  : less than 2 lower highs (no clear downtrend to flip)
    """
    evs = []
    for i in range(LBK + 5, N - MAX_HOLD - 2):
        # Detect 3 lower highs
        if not (HI[i-3] > HI[i-2] > HI[i-1]): continue
        swing_ref = HI[i-1]       # most recent lower high
        if CL[i] <= swing_ref: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=swing_ref - 0.3*atr, ref_hi=swing_ref + 0.3*atr,
                        ref_size=atr, atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def B5_double_bottom_bos():
    """
    Trigger   : Two lows within 0.3% of each other (≥ 5 bars apart), then neckline break
    Invalidate: close below either bottom
    No-trade  : lows further than 50 bars apart (pattern too extended)
    """
    evs = []
    for i in range(LBK + 5, N - MAX_HOLD - 5):
        lo1 = LO[i]
        # Look back for a matching low
        for k in range(max(LBK, i-50), i-4):
            lo2 = LO[k]
            if abs(lo1 - lo2) / (lo2 + 1e-8) > 0.003: continue
            if (i - k) > 50: continue
            # Neckline: highest close between the two lows
            neckline = np.max(CL[k:i])
            if CL[i] > neckline:
                atr = ATR1H[i]
                if np.isnan(atr) or atr <= 0: break
                evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                                ref_lo=min(lo1, lo2), ref_hi=neckline,
                                ref_size=neckline - min(lo1, lo2), atr_1h=atr,
                                year=yr_[i], ts=IDX[i]))
            break
    return evs

# ── GROUP C: CLEAN RETESTS ───────────────────────────────────────────────────

def C1_ema21_rej_wick():
    """
    Trigger   : LO ≤ EMA21 but CL > EMA21; wick below EMA > 2× body (rejection)
    Invalidate: close below EMA21
    No-trade  : EMA21 slope ≤ 0 or price has been below EMA21 in last 3 bars
    """
    evs = []
    ema21_slope = np.diff(EMA21, prepend=EMA21[0])
    for i in range(LBK + 1, N - MAX_HOLD - 2):
        if LO[i] > EMA21[i]: continue
        if CL[i] <= EMA21[i]: continue
        if ema21_slope[i] <= 0: continue
        if any(CL[i-k] < EMA21[i-k] for k in range(1, 4)): continue   # below EMA recently
        body  = abs(CL[i] - OP[i])
        lower_wick = min(OP[i], CL[i]) - LO[i]
        if lower_wick < 2 * body: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=LO[i], ref_hi=CL[i]+atr, ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def C2_ema21_clean_retest():
    """
    Trigger   : 5 consecutive closes above EMA21, then first LO that touches EMA21 with CL > EMA21
    Invalidate: close below EMA21
    No-trade  : EMA21 slope < 0 (not in uptrend)
    """
    evs = []
    above_ema = (CL > EMA21).astype(int)
    consec = np.zeros(N, dtype=int)
    for i in range(1, N):
        consec[i] = consec[i-1] + 1 if above_ema[i] else 0
    ema21_slope = np.diff(EMA21, prepend=EMA21[0])
    for i in range(LBK + 6, N - MAX_HOLD - 2):
        if consec[i-1] < 5: continue           # need 5+ prior bars above EMA
        if LO[i] > EMA21[i]: continue          # must touch EMA
        if CL[i] <= EMA21[i]: continue         # must close above
        if ema21_slope[i] <= 0: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=EMA21[i] - 0.3*atr, ref_hi=EMA21[i] + 0.3*atr,
                        ref_size=atr, atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def C3_inside_bar_at_level():
    """
    Trigger   : Inside bar (HI < prev HI, LO > prev LO) within 0.5 ATR of 20-bar swing level
    Invalidate: close beyond opposite side of the mother bar
    No-trade  : inside bar body > 0.8× ATR (too large, not a real compression)
    """
    evs = []
    for i in range(LBK + 2, N - MAX_HOLD - 2):
        if HI[i] >= HI[i-1] or LO[i] <= LO[i-1]: continue   # not inside bar
        body = abs(CL[i] - OP[i]); atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        if body > 0.8*atr: continue
        dist_hi = abs(HI[i] - swing_hi_wk[i-1])
        dist_lo = abs(LO[i] - swing_lo_wk[i-1])
        if dist_hi <= 0.5*atr:
            evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                            ref_lo=HI[i]-atr, ref_hi=HI[i]+0.3*atr, ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
        elif dist_lo <= 0.5*atr:
            evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                            ref_lo=LO[i]-0.3*atr, ref_hi=LO[i]+atr, ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def C4_third_touch():
    """
    Trigger   : Third touch of a 20-bar swing level within 0.3 ATR (exhaustion bounce)
    Invalidate: level broken by close
    No-trade  : fewer than 8 bars between each touch (level too 'fresh')
    """
    evs = []
    for i in range(LBK*2, N - MAX_HOLD - 2):
        atr = ATR1H[i]; tol = 0.3 * atr
        if np.isnan(atr) or atr <= 0: continue
        level_lo = swing_lo_wk[i-1]; level_hi = swing_hi_wk[i-1]
        # Count touches of level_lo in lookback
        for level, direction in [(level_lo, "long"), (level_hi, "short")]:
            touches = [j for j in range(i-LBK*2, i)
                       if abs(LO[j] - level) <= tol or abs(HI[j] - level) <= tol]
            if len(touches) < 2: continue
            if (i - touches[-1]) < 3: continue   # too close to prior touch
            # Near current bar
            if direction == "long" and LO[i] <= level + tol and CL[i] >= level - tol:
                if len(touches) >= 2 and touches[-1] - touches[-2] >= 8:
                    evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                                    ref_lo=level-tol, ref_hi=level+tol, ref_size=atr,
                                    atr_1h=atr, year=yr_[i], ts=IDX[i]))
                    break
            elif direction == "short" and HI[i] >= level - tol and CL[i] <= level + tol:
                if len(touches) >= 2 and touches[-1] - touches[-2] >= 8:
                    evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                                    ref_lo=level-tol, ref_hi=level+tol, ref_size=atr,
                                    atr_1h=atr, year=yr_[i], ts=IDX[i]))
                    break
    return evs

def C5_bb_squeeze_breakout():
    """
    Trigger   : BB squeeze (bb_squeeze < 0.7× 20-bar avg) for 5+ bars → BOS on any vol
    Invalidate: close back inside the pre-breakout range within 3 bars
    No-trade  : no squeeze present (bb_squeeze ≥ 0.7× avg at breakout)
    """
    evs = []
    sq = BBSQ; sq_ma = bbsq_ma20
    for i in range(LBK + 6, N - MAX_HOLD - 2):
        # Check 5 bars of squeeze
        if not all(sq[i-k] < 0.7 * sq_ma[i-k] for k in range(1, 6)): continue
        # Current bar breaks out of squeeze
        if CL[i] > swing_hi_cl[i-1]:
            atr = ATR1H[i]
            if np.isnan(atr) or atr <= 0: continue
            evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                            ref_lo=swing_hi_cl[i-1], ref_hi=CL[i], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
        elif CL[i] < swing_lo_cl[i-1]:
            atr = ATR1H[i]
            if np.isnan(atr) or atr <= 0: continue
            evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                            ref_lo=CL[i], ref_hi=swing_lo_cl[i-1], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

# ── GROUP D: MOMENTUM SHIFTS ─────────────────────────────────────────────────

def D1_rsi_bull_divergence():
    """
    Trigger   : LO[i] < 5-bar minimum LO but RSI[i] > 5-bar minimum RSI (bullish div)
    Invalidate: close below the divergence low
    No-trade  : RSI[i] > 45 (not in oversold territory; divergence less meaningful)
    """
    evs = []
    for i in range(LBK + 5, N - MAX_HOLD - 2):
        if RSI14[i] > 45: continue
        lo5 = np.min(LO[i-5:i]); rsi5 = np.min(RSI14[i-5:i])
        if LO[i] >= lo5: continue     # price not at new low
        if RSI14[i] <= rsi5: continue  # RSI also lower → no divergence
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=LO[i], ref_hi=CL[i]+atr, ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def D2_rsi_bear_divergence():
    """
    Trigger   : HI[i] > 5-bar maximum HI but RSI[i] < 5-bar maximum RSI (bearish div)
    Invalidate: close above the divergence high
    No-trade  : RSI[i] < 55 (not in overbought territory)
    """
    evs = []
    for i in range(LBK + 5, N - MAX_HOLD - 2):
        if RSI14[i] < 55: continue
        hi5 = np.max(HI[i-5:i]); rsi5 = np.max(RSI14[i-5:i])
        if HI[i] <= hi5: continue
        if RSI14[i] >= rsi5: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                        ref_lo=CL[i]-atr, ref_hi=HI[i], ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def D3_macd_zero_cross():
    """
    Trigger   : MACD histogram crosses from negative to positive (zero-line cross)
    Invalidate: MACD histogram turns negative again within 3 bars
    No-trade  : MACD line still below zero (too early in cycle)
    """
    evs = []
    for i in range(LBK + 2, N - MAX_HOLD - 2):
        if MHIST[i] <= 0 or MHIST[i-1] >= 0: continue    # not a cross up
        if df["macd"].iloc[i] < 0: continue               # no-trade: macd below zero
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=CL[i]-atr, ref_hi=CL[i], ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def D4_engulfing_at_level():
    """
    Trigger   : Bullish/bearish engulfing candle within 0.5 ATR of 20-bar swing level
    Invalidate: close beyond opposite end of engulfing candle
    No-trade  : engulfing candle body < 0.5× ATR (weak engulf) or no nearby level
    """
    evs = []
    for i in range(LBK + 2, N - MAX_HOLD - 2):
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        body = abs(CL[i] - OP[i])
        if body < 0.5 * atr: continue
        # Bullish engulfing
        if CL[i] > OP[i] and OP[i] < CL[i-1] and CL[i] > OP[i-1]:
            if abs(LO[i] - swing_lo_wk[i-1]) <= 0.5*atr:
                evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                                ref_lo=LO[i], ref_hi=CL[i]+atr, ref_size=atr,
                                atr_1h=atr, year=yr_[i], ts=IDX[i]))
        # Bearish engulfing
        elif CL[i] < OP[i] and OP[i] > CL[i-1] and CL[i] < OP[i-1]:
            if abs(HI[i] - swing_hi_wk[i-1]) <= 0.5*atr:
                evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                                ref_lo=CL[i]-atr, ref_hi=HI[i], ref_size=atr,
                                atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def D5_momentum_burst():
    """
    Trigger   : Body > 1.5× ATR in trend direction after 3-bar pullback; vol > 1.1× avg
    Invalidate: close below candle open (long) or above candle open (short)
    No-trade  : vol < 1.1× avg (momentum without volume = unreliable)
    """
    evs = []
    for i in range(LBK + 4, N - MAX_HOLD - 2):
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        if VR[i] < 1.1: continue
        body = CL[i] - OP[i]
        if body > 1.5 * atr:      # bullish burst
            if CL[i-1] < CL[i-2] and CL[i-2] < CL[i-3]:  # 3-bar pullback
                evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                                ref_lo=OP[i], ref_hi=CL[i]+atr, ref_size=atr,
                                atr_1h=atr, year=yr_[i], ts=IDX[i]))
        elif body < -1.5 * atr:   # bearish burst
            if CL[i-1] > CL[i-2] and CL[i-2] > CL[i-3]:
                evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                                ref_lo=CL[i]-atr, ref_hi=OP[i], ref_size=atr,
                                atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

# ── GROUP E: COMPOSITE ───────────────────────────────────────────────────────

def E1_bos_vol_dry_obv():
    """
    Trigger   : BOS (high vol ≥ 1.5×) → dry retest (vol < 0.7×) → OBV trend aligned
    Invalidate: close below broken level
    No-trade  : OBV trend opposite to trade direction at retest
    """
    evs = []
    for i in range(LBK + 1, N - MAX_HOLD - 12):
        if CL[i] <= swing_hi_cl[i-1] or VR[i] < 1.5: continue
        level = swing_hi_cl[i-1]
        for j in range(i+1, min(i+9, N-MAX_HOLD)):
            if CL[j] < level - 0.5*ATR1H[j]: break
            if LO[j] <= level + 0.4*ATR1H[j] and VR[j] < 0.7 and OBVT[j] > 0:
                evs.append(dict(direction="long", entry_i=j, entry_px=CL[j],
                                ref_lo=level - 0.4*ATR1H[j], ref_hi=level + 0.4*ATR1H[j],
                                ref_size=ATR1H[j], atr_1h=ATR1H[j], year=yr_[j], ts=IDX[i]))
                break
    return evs

def E2_choch_obv_align():
    """
    Trigger   : CHoCH (3 lower highs, then close above swing high) + OBV > OBV_EMA21
    Invalidate: close below CHoCH level
    No-trade  : OBV trend ≤ 0 (OBV not confirming the flip)
    """
    evs = []
    for i in range(LBK + 5, N - MAX_HOLD - 2):
        if not (HI[i-3] > HI[i-2] > HI[i-1]): continue
        swing_ref = HI[i-1]
        if CL[i] <= swing_ref: continue
        if OBVT[i] <= 0: continue   # no-trade
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=swing_ref-0.3*atr, ref_hi=swing_ref+0.3*atr,
                        ref_size=atr, atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def E3_adx_trending_bos():
    """
    Trigger   : BOS (close > 20-bar high) when ADX > 25 (confirmed trending) + DI+ > DI-
    Invalidate: ADX drops below 20 or close back below level
    No-trade  : ADX < 25 or DI+ < DI- (no confirmed directional trend)
    """
    evs = []
    dip = df["di_plus"].values; dim = df["di_minus"].values
    for i in range(LBK + 1, N - MAX_HOLD - 2):
        if CL[i] <= swing_hi_cl[i-1]: continue
        if ADX_[i] < 25: continue
        if dip[i] < dim[i]: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                        ref_lo=swing_hi_cl[i-1], ref_hi=CL[i], ref_size=atr,
                        atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def E4_htf_ltf_align():
    """
    Trigger   : 4H EMA20 bullish + 15M BOS up (or bearish + 15M BOS down) with vol > 1.2×
    Invalidate: 4H trend flips or close breaks back below BOS level
    No-trade  : 4H and 15M bias in opposite directions
    """
    evs = []
    for i in range(LBK + 20, N - MAX_HOLD - 2):
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        bias = _4h_bias(IDX[i])
        if bias is None: continue
        if VR[i] < 1.2: continue
        if bias and CL[i] > swing_hi_cl[i-1]:    # 4H bull + 15M BOS up
            evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                            ref_lo=swing_hi_cl[i-1], ref_hi=CL[i], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
        elif not bias and CL[i] < swing_lo_cl[i-1]:  # 4H bear + 15M BOS down
            evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                            ref_lo=CL[i], ref_hi=swing_lo_cl[i-1], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs

def E5_vol_compression_break():
    """
    Trigger   : ATR < 0.6× 20-bar ATR avg for 5+ bars → BOS with vol > 1.5× avg
    Invalidate: close back inside the compression range (below BOS level)
    No-trade  : ATR not compressed (≥ 0.6× avg before breakout)
    """
    evs = []
    for i in range(LBK + 6, N - MAX_HOLD - 2):
        if not all(ATR[i-k] < 0.6 * atr_ma20[i-k] for k in range(1, 6)): continue
        if VR[i] < 1.5: continue
        atr = ATR1H[i]
        if np.isnan(atr) or atr <= 0: continue
        if CL[i] > swing_hi_cl[i-1]:
            evs.append(dict(direction="long", entry_i=i, entry_px=CL[i],
                            ref_lo=swing_hi_cl[i-1], ref_hi=CL[i], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
        elif CL[i] < swing_lo_cl[i-1]:
            evs.append(dict(direction="short", entry_i=i, entry_px=CL[i],
                            ref_lo=CL[i], ref_hi=swing_lo_cl[i-1], ref_size=atr,
                            atr_1h=atr, year=yr_[i], ts=IDX[i]))
    return evs


# ══════════════════════════════════════════════════════════════════════════════
# Metadata for each setup
# ══════════════════════════════════════════════════════════════════════════════
SETUPS = [
    # (key, label, collector, trigger, invalidation, no_trade)
    ("A1", "Vol Spike Reversal",        A1_vol_spike_reversal,
     "vol >3× avg + body reverses prior 3-bar direction by >0.4 ATR",
     "close beyond spike extreme in continuation direction",
     "trend direction same as current body (no reversal context)"),

    ("A2", "Vol Dry Retest",            A2_vol_dry_retest,
     "BOS → retest within 8 bars with vol <0.7× avg",
     "close below broken swing level",
     "retest volume ≥0.7× avg (sellers still present)"),

    ("A3", "Vol BOS Continuation",      A3_vol_bos_continuation,
     "CL > 20-bar highest close with vol >1.5× avg",
     "close back below broken level within 3 bars",
     "RSI >80 (overbought) OR vol <1.5× avg"),

    ("A4", "OBV Divergence",            A4_obv_divergence,
     "price at 20-bar extreme, OBV not confirming (div >3%)",
     "close beyond divergence extreme",
     "OBV within 3% of its own extreme (divergence too small)"),

    ("A5", "Vol Climax Absorption",     A5_vol_climax_absorption,
     "vol >4× avg, body <30% of range (rejection wick absorption)",
     "next bar continues in wick direction",
     "body >30% of range (not absorption) OR vol <4× avg"),

    ("B1", "BOS Close ADX",             B1_bos_close,
     "CL > 20-bar highest close with ADX >15",
     "close below the broken level",
     "ADX <15 (no trend context)"),

    ("B2", "BOS First Retest",          B2_bos_first_retest,
     "first bar touching broken swing level within 0.4 ATR, within 10 bars of BOS",
     "close below broken level –0.5 ATR",
     "price moved <1 ATR above broken level before retest (too tight)"),

    ("B3", "Failed BOS Long",           B3_failed_bos_long,
     "wick below 20-bar swing low, close back above (fakeout squeeze)",
     "close below original swing low",
     "prior 3-bar trend already bearish (wick follows trend, not a fakeout)"),

    ("B4", "CHoCH Uptrend",             B4_choch_uptrend,
     "after 3 consecutive lower highs, first close above most recent swing high",
     "close below CHoCH swing high level",
     "fewer than 3 lower highs (no confirmed downtrend to flip)"),

    ("B5", "Double Bottom BOS",         B5_double_bottom_bos,
     "two lows within 0.3% tolerance (≥5 bars apart), neckline break",
     "close below either bottom",
     "lows >50 bars apart (pattern too extended)"),

    ("C1", "EMA21 Rejection Wick",      C1_ema21_rej_wick,
     "LO ≤ EMA21, CL > EMA21, lower wick >2× body (rejection)",
     "close below EMA21",
     "EMA21 slope ≤0 OR price below EMA21 in prior 3 bars"),

    ("C2", "EMA21 Clean Retest",        C2_ema21_clean_retest,
     "5 consecutive closes above EMA21, then first LO touch with CL > EMA21",
     "close below EMA21",
     "EMA21 slope <0 (downtrend retest, not uptrend continuation)"),

    ("C3", "Inside Bar at Level",       C3_inside_bar_at_level,
     "inside bar (HI < prev HI, LO > prev LO) within 0.5 ATR of swing level",
     "close beyond opposite side of mother bar",
     "inside bar body >0.8× ATR (no real compression)"),

    ("C4", "Third Touch Level",         C4_third_touch,
     "third touch of 20-bar swing level within 0.3 ATR (exhaustion bounce)",
     "swing level broken by close",
     "fewer than 8 bars between touches (level too fresh)"),

    ("C5", "BB Squeeze Breakout",       C5_bb_squeeze_breakout,
     "BB squeeze for 5+ bars (<0.7× avg squeeze), then BOS bar",
     "close back inside pre-breakout range within 3 bars",
     "no squeeze present at breakout bar (bb_squeeze ≥0.7× avg)"),

    ("D1", "RSI Bull Divergence",       D1_rsi_bull_divergence,
     "LO < 5-bar minimum, RSI higher than 5-bar minimum (bullish div)",
     "close below divergence low",
     "RSI >45 (not in oversold territory)"),

    ("D2", "RSI Bear Divergence",       D2_rsi_bear_divergence,
     "HI > 5-bar maximum, RSI lower than 5-bar maximum (bearish div)",
     "close above divergence high",
     "RSI <55 (not in overbought territory)"),

    ("D3", "MACD Zero Cross",           D3_macd_zero_cross,
     "MACD histogram crosses from negative to positive",
     "MACD histogram turns negative again within 3 bars",
     "MACD line still below zero at crossover"),

    ("D4", "Engulfing at Level",        D4_engulfing_at_level,
     "bull/bear engulfing (body >0.5 ATR) within 0.5 ATR of swing level",
     "close beyond opposite end of engulfing candle",
     "engulfing body <0.5× ATR (weak signal) or no nearby level"),

    ("D5", "Momentum Burst",            D5_momentum_burst,
     "body >1.5× ATR after 3-bar pullback, vol >1.1× avg",
     "close below candle open (long) or above candle open (short)",
     "vol <1.1× avg (momentum without volume)"),

    ("E1", "BOS + Vol + Dry + OBV",    E1_bos_vol_dry_obv,
     "BOS (vol≥1.5×) → dry retest (vol<0.7×) → OBV trend aligned",
     "close below broken level",
     "OBV trend opposite to trade direction at retest bar"),

    ("E2", "CHoCH + OBV Align",        E2_choch_obv_align,
     "3 lower highs + close above swing high + OBV > OBV_EMA21",
     "close below CHoCH level",
     "OBV trend ≤0 at CHoCH bar (not confirming flip)"),

    ("E3", "ADX BOS Trending",          E3_adx_trending_bos,
     "BOS + ADX >25 + DI+ > DI- (confirmed directional trend)",
     "ADX drops below 20 or close back below BOS level",
     "ADX <25 OR DI+ < DI- (trend not confirmed)"),

    ("E4", "4H Align + 15M BOS",       E4_htf_ltf_align,
     "4H EMA20 bullish + 15M BOS up with vol >1.2× avg (or bearish + BOS down)",
     "4H trend flips or 15M close breaks back below BOS level",
     "4H and 15M biases in opposite directions"),

    ("E5", "Vol Compression Break",     E5_vol_compression_break,
     "ATR <0.6× 20-bar avg for 5+ bars → BOS with vol >1.5× avg",
     "close back inside compression range",
     "no ATR compression before breakout"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Compute IC for all 25 setups
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  Computing IC for all 25 setups …")
print(SEP)

ic_results = []
for key, label, collector, trigger, inv, notrade in SETUPS:
    events = collector()
    ic, p, n = compute_ic(events)
    ic_results.append(dict(key=key, label=label, n=n, ic=round(ic, 5),
                           p=round(p, 4), events=events,
                           trigger=trigger, inv=inv, notrade=notrade))
    sig = "✓" if p < 0.05 else ("~" if p < 0.10 else "✗")
    print(f"  [{key:2s}] {label:<30s}  n={n:6,}  IC={ic:+.4f}  p={p:.4f}  {sig}")

# Rank by IC significance (lowest p-value with IC > 0); guard against NaN
def _rank_key(x):
    ic = x["ic"] if not (x["ic"] != x["ic"]) else 0.0   # nan check
    p  = x["p"]  if not (x["p"]  != x["p"])  else 1.0
    return (p, -ic) if ic > 0 else (1.0 + abs(ic), -ic)
ranked = sorted(ic_results, key=_rank_key)

print(f"\n{SEP}")
print("  RANKING (by reliability = lowest p-value for positive IC)")
print(SEP)
for rank, r in enumerate(ranked, 1):
    sig = "✓" if r["p"] < 0.05 else ("~" if r["p"] < 0.10 else "✗")
    print(f"  #{rank:2d}  [{r['key']}] {r['label']:<30s}  IC={r['ic']:+.4f}  p={r['p']:.4f}  {sig}")

# Best setup for full pipeline
best = ranked[0]
print(f"\n{SEP2}")
print(f"  BEST SETUP: [{best['key']}] {best['label']}")
print(f"  IC={best['ic']:+.4f}  p={best['p']:.4f}  n={best['n']:,}")
print(SEP2)


# ══════════════════════════════════════════════════════════════════════════════
# Full validation pipeline on best setup
# ══════════════════════════════════════════════════════════════════════════════
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
oos_df = pd.DataFrame([t.__dict__ for t in oos_t]).sort_values("entry_ts") if oos_t else pd.DataFrame()
wins_n = int((oos_df["net_pnl"] > 0).sum()) if not oos_df.empty else 0
wr_oos = wins_n / len(oos_df) if len(oos_df) else 0
rr = best_params["rr"]
be_oos = 1/(1+rr)*100 if rr > 0 else 50.0
binom_p = st.binomtest(wins_n, max(len(oos_df),1), be_oos/100, alternative="greater").pvalue if len(oos_df) else 1
print(f"  Return={kp['total_return']:+.1%}  MaxDD={kp['max_dd']:.1%}  "
      f"WR={wr_oos:.1%}  BE≈{be_oos:.1f}%  binom_p={binom_p:.4f}")

print("  Monte Carlo …")
mc_res = run_monte_carlo(oos_df[["net_pnl","gross_pnl","total_fees"]], INIT_CAP, N_SIMS) if len(oos_df) > 5 else {}
p_profit = float((mc_res["total_return"] > 0).mean()) if mc_res else 0
p_ruin   = float((mc_res["total_return"] < -0.5).mean()) if mc_res else 1
print(f"  P(profit)={p_profit:.1%}  P(ruin)={p_ruin:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# HTML Report
# ══════════════════════════════════════════════════════════════════════════════
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
            f'border-bottom:1px solid {_GRID};padding:3px 0">'
            f'<span style="color:#9e9e9e">{lbl}</span>'
            f'<span style="color:{_TEXT};font-weight:600">{val}</span></div>')

def _style(v, fmt=".1%", good=0):
    c = _GRN if v > good else _RED
    return f'<span style="color:{c}">{v:{fmt}}</span>'

print(f"\n[HTML] Generazione report …")

# ── IC Ranking table
def ic_color(ic, p):
    if p < 0.05 and ic > 0: return _GRN
    if p < 0.10 and ic > 0: return _YEL
    return _RED

rank_rows = []
for rank, r in enumerate(ranked, 1):
    c = ic_color(r["ic"], r["p"])
    sig = "✅" if r["p"]<0.05 else ("~" if r["p"]<0.10 else "✗")
    rank_rows.append([
        f'<b style="color:{_ACC}">#{rank}</b>',
        f'<span style="color:{c}">[{r["key"]}] {r["label"]}</span>',
        f'{r["n"]:,}',
        f'<span style="color:{c}">{r["ic"]:+.4f}</span>',
        f'<span style="color:{c}">{r["p"]:.4f}</span>',
        sig,
    ])

th = "".join(f'<th style="padding:5px 8px;text-align:right;color:{_ACC};border-bottom:1px solid {_GRID}">{h}</th>'
             for h in ["Rank","Setup","N events","IC","p-value","Sig"])
td = "".join("<tr>"+"".join(f'<td style="padding:4px 8px;text-align:right;color:{_TEXT}">{c}</td>'
             for c in r)+"</tr>" for r in rank_rows)
rank_table = f'<table style="width:100%;border-collapse:collapse;font-size:12px"><thead><tr>{th}</tr></thead><tbody>{td}</tbody></table>'

# ── Full setup description table (25 rows)
desc_rows = []
for r in ic_results:  # original order
    c = ic_color(r["ic"], r["p"])
    desc_rows.append([
        f'<span style="color:{c}">[{r["key"]}]</span>',
        f'<b>{r["label"]}</b>',
        f'<span style="font-size:11px;color:#9e9e9e">{r["trigger"]}</span>',
        f'<span style="font-size:11px;color:{_RED}">{r["inv"]}</span>',
        f'<span style="font-size:11px;color:{_YEL}">{r["notrade"]}</span>',
        f'<span style="color:{c}">{r["ic"]:+.4f} (p={r["p"]:.3f})</span>',
    ])

th2 = "".join(f'<th style="padding:5px 8px;color:{_ACC};border-bottom:1px solid {_GRID};text-align:left">{h}</th>'
              for h in ["ID","Setup","Trigger","Invalidation","No-Trade","IC"])
td2 = "".join("<tr>"+"".join(f'<td style="padding:4px 8px;color:{_TEXT};vertical-align:top;font-size:11px">{c}</td>'
             for c in r)+"</tr>" for r in desc_rows)
desc_table = f'<table style="width:100%;border-collapse:collapse"><thead><tr>{th2}</tr></thead><tbody>{td2}</tbody></table>'

# ── Charts for best setup
def plot_ic_bar():
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    xs = range(len(ranked))
    ics = [r["ic"] for r in ranked]
    ps  = [r["p"]  for r in ranked]
    clrs = [_GRN if p<0.05 and ic>0 else (_YEL if p<0.10 and ic>0 else _RED)
            for ic, p in zip(ics, ps)]
    ax.bar(xs, ics, color=clrs, alpha=0.85)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["key"] for r in ranked], rotation=45, ha="right", fontsize=8, color=_TEXT)
    ax.set_ylabel("IC (Spearman)", color=_TEXT)
    ax.set_title("IC Ranking — 25 Setups (green=p<0.05, yellow=p<0.10)", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)

def plot_equity(eq, label):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5), sharex=True,
                                    gridspec_kw={"height_ratios":[2,1]})
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_CARD); ax.tick_params(colors=_TEXT)
        for sp in ax.spines.values(): sp.set_color(_GRID)
    ax1.plot(eq.index, eq.values/INIT_CAP, color=_GRN, lw=1.5)
    ax1.axhline(1.0, color=_GRID, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (norm.)", color=_TEXT)
    rm = eq.cummax(); dd = (eq/rm-1)*100
    ax2.fill_between(dd.index, dd.values, 0, color=_RED, alpha=0.4)
    ax2.set_ylabel("DD %", color=_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle(f"OOS Equity — {label}", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)

def plot_wf_bars(wr_list, label):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    clrs = [_GRN if r>0 else _RED for r in wr_list]
    ax.bar(range(len(wr_list)), [r*100 for r in wr_list], color=clrs, alpha=0.8)
    ax.axhline(0, color=_GRID, lw=0.8)
    ax.set_title(f"WF per-window OOS returns — {label}", color=_TEXT)
    ax.set_ylabel("Ret %", color=_TEXT)
    fig.tight_layout()
    return _fig_to_b64(fig)

def plot_mc(mc_res, label):
    fin = mc_res["total_return"]*100
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

img_rank = plot_ic_bar()
img_eq   = plot_equity(oos_eq, best["label"]) if not oos_eq.empty else ""
img_wf   = plot_wf_bars(wr_list, best["label"])
img_mc   = plot_mc(mc_res, best["label"]) if mc_res else ""

# ── Best setup metrics block
ok = kp["total_return"] > 0 and binom_p < 0.05 and p_profit > 0.5
vc = _GRN if ok else _RED
vt = "✅ VALIDATO" if ok else "✗ NON VALIDATO"

mc_fin = pd.Series(mc_res["total_return"]*100) if mc_res else pd.Series([0])

metrics_html = "".join([
    _kv("Events IS", f'{best["n"]:,}'),
    _kv("IC (Spearman)", f'{best["ic"]:+.4f}  p={best["p"]:.4f}'),
    _kv("Trigger", best["trigger"]),
    _kv("Invalidation", best["inv"]),
    _kv("No-trade", best["notrade"]),
    _kv("IS best tp/sl", f'tp={tp_f}  sl={sl_b}'),
    _kv("IS WR", f'{best_params["wr"]:.1f}%  BE={best_params["be"]:.1f}%  ExpPnL={best_params["exp"]:+.5f}%'),
    _kv("OOS trades", str(len(oos_t))),
    _kv("WR OOS", _style(wr_oos, ".1%")),
    _kv("Total Return OOS", _style(kp["total_return"], ".1%")),
    _kv("Max DD", _style(kp["max_dd"], ".1%")),
    _kv("Binom p", f'{binom_p:.4f}'),
    _kv("MC P(profit)", _style(p_profit, ".1%", good=0.5)),
    _kv("MC P(ruin)", f'{p_ruin:.1%}'),
    _kv("MC median", f'{float(np.median(mc_fin)):.1f}%'),
    _kv("WF windows", f'{n_wins} tot / {sum(1 for x in wr_list if x>0)} profit'),
    _kv("Verdict", f'<b style="color:{vc}">{vt}</b>'),
])

metrics_card = (f'<div style="background:{_CARD};border:1px solid {_GRID};'
                f'border-radius:8px;padding:16px 20px;margin-bottom:14px">'
                f'<h3 style="color:{_GRN};margin:0 0 10px">Best Setup: [{best["key"]}] {best["label"]}</h3>'
                f'{metrics_html}</div>')

html = f"""<!DOCTYPE html><html lang="it"><head>
<meta charset="UTF-8">
<title>25 Setup Ranking — BTCUSDT 15M</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{_BG};color:{_TEXT};font-family:monospace;padding:24px}}
  h1{{color:{_ACC};margin-bottom:6px}}
  h2{{color:{_ACC};margin-top:32px;margin-bottom:10px}}
  p{{margin-bottom:8px;line-height:1.5;color:#9e9e9e}}
  table{{width:100%;border-collapse:collapse;margin-bottom:16px}}
  th,td{{padding:4px 8px;vertical-align:top}}
  thead th{{color:{_ACC};border-bottom:1px solid {_GRID}}}
</style></head><body>

<h1>25 Setup Ideas — BTCUSDT 15M | 2020-2026</h1>
<p>Built on: volume confirmation · structure breaks · clean retests · momentum shifts</p>
<p>IC = Spearman corr(signal direction, 4H signed forward return) · ranked by p-value (reliability, not excitement)</p>

<h2>IC Ranking — Visual</h2>
{_imgt(img_rank)}

<h2>IC Ranking Table</h2>
{rank_table}

<h2>All 25 Setups — Complete Reference</h2>
<p>Trigger / Invalidation / No-Trade for each setup, with IC result</p>
{desc_table}

<h2>Full Validation: [{best["key"]}] {best["label"]}</h2>
<p>IS scan (4×4 TP/SL grid) → Walk-Forward (35 windows, 6m/2m) → Monte Carlo (5000 sims)</p>
{metrics_card}
{"" if not img_eq else _imgt(img_eq)}
{_imgt(img_wf)}
{"" if not img_mc else _imgt(img_mc)}

<p style="color:#555;font-size:11px;margin-top:40px">
  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC — BTCUSDT 15M 2020-2026
</p>
</body></html>"""

out = Path("reports/report_setup_ranking.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n✅ Report salvato: {out}  ({out.stat().st_size/1024:.0f} KB)")
print(SEP2)
