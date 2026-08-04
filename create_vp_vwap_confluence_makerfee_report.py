#!/usr/bin/env python3
"""
create_vp_vwap_confluence_makerfee_report.py
==============================================
Verifica di sensibilità alle fee della strategia Volume Profile + VWAP
Confluence (RR=3.0, il migliore per DSR nel report base). L'entry è
LEVEL-BASED (zona di confluenza POC/VWAP nota in anticipo, tocco +
rejection già confermati sulla barra precedente) — candidata naturale per
ordini limit/maker (0.02%/lato) invece di market/taker (0.055%/lato),
stessa distinzione già applicata alla VWAP Mean-Reversion in questa
sessione. L'uscita (SL o TP) resta invece tipicamente reattiva
(attraversamento di un livello in corso) → più realisticamente taker.

Scenari fee (round-trip):
  A) Taker reale Bybit derivatives (baseline, 0.055%+0.055%=0.110%)
  B) Full maker ottimistico (0.02%+0.02%=0.04%) — assume fill limit al 100%
  C) Mix: entry maker / exit taker (0.02%+0.055%=0.075%) — più realistico
     per un entry pianificato ma un'uscita reattiva
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

LOOKBACK = 5
MIN_SEP_ATR = 0.5
LOOKBACK_DAYS = 20
N_BINS = 50
CONF_ATR_MULT = 0.5
SWING_LOOKBACK = 10
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 48
BEST_RR = 3.0

FEE_SCENARIOS = [
    ("A) Taker reale (0.055%+0.055%)", 0.00055, 0.00055),
    ("B) Full maker (0.02%+0.02%)",     0.00020, 0.00020),
    ("C) Entry maker / exit taker",     0.00020, 0.00055),
]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Volume Profile + VWAP Confluence (RR=3.0) — Sensibilità alle fee")
w(SEP)

t0 = time.time()
print("\n[DATA] Loading 1H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H = len(df1h)

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
OP = df1h["open"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
ATR = np.where(df1h["atr_14"].values > 0, df1h["atr_14"].values, 1.0)

dates = IDX1H.normalize().values
day_change = np.r_[True, dates[1:] != dates[:-1]]
day_id = np.cumsum(day_change) - 1
day_start_idx = np.where(day_change)[0]
n_days = len(day_start_idx)

tp = (HI + LO + CL) / 3.0
tmp = pd.DataFrame({"day_id": day_id, "pv": tp * VOL, "vol": VOL})
g = tmp.groupby("day_id")
cum_pv = g["pv"].cumsum().values
cum_v = g["vol"].cumsum().values
bar_in_day = g.cumcount().values
VWAP = np.where(cum_v > 0, cum_pv / cum_v, np.nan)

daily_poc = np.full(n_days, np.nan)
for d in range(n_days):
    win_start_day = d - LOOKBACK_DAYS
    if win_start_day < 0:
        continue
    lo_bar = day_start_idx[win_start_day]
    hi_bar = day_start_idx[d]
    if hi_bar - lo_bar < 100:
        continue
    seg_tp = tp[lo_bar:hi_bar]
    seg_vol = VOL[lo_bar:hi_bar]
    seg_lo = LO[lo_bar:hi_bar].min()
    seg_hi = HI[lo_bar:hi_bar].max()
    if seg_hi <= seg_lo:
        continue
    hist, edges = np.histogram(seg_tp, bins=N_BINS, range=(seg_lo, seg_hi), weights=seg_vol)
    poc_idx = int(np.argmax(hist))
    daily_poc[d] = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0
POC = daily_poc[day_id]
print(f"[DATA+VP+VWAP] ready in {time.time()-t0:.0f}s")


def build_events():
    evs = []
    last_exit = -1
    start_i = max(LOOKBACK, SWING_LOOKBACK) + 2
    for i in range(start_i, N1H - 1):
        if i <= last_exit:
            continue
        if not np.isfinite(VWAP[i]) or not np.isfinite(POC[i]) or bar_in_day[i] < 2 or ATR[i] <= 0:
            continue
        ref = i - LOOKBACK
        if not np.isfinite(VWAP[ref]) or ATR[ref] <= 0:
            continue
        sep = CL[ref] - VWAP[ref]
        if abs(sep) < MIN_SEP_ATR * ATR[ref]:
            continue
        bias = 1 if sep > 0 else -1
        if abs(POC[i] - VWAP[i]) > CONF_ATR_MULT * ATR[i]:
            continue
        zone_lo = min(POC[i], VWAP[i]); zone_hi = max(POC[i], VWAP[i])
        touch = LO[i] <= zone_hi and HI[i] >= zone_lo
        if not touch:
            continue
        zone_mid = (POC[i] + VWAP[i]) / 2.0
        rejection = (bias == 1 and CL[i] > zone_mid) or (bias == -1 and CL[i] < zone_mid)
        if not rejection:
            continue
        if bias == 1:
            swing = LO[i - SWING_LOOKBACK:i + 1].min()
            sl = swing - STOP_BUFFER_ATR * ATR[i]
        else:
            swing = HI[i - SWING_LOOKBACK:i + 1].max()
            sl = swing + STOP_BUFFER_ATR * ATR[i]
        entry_i = i + 1
        ep = OP[entry_i]
        risk = abs(ep - sl)
        if risk <= 0:
            continue
        hold = min(MAX_HOLD_BARS, N1H - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(i=i, entry_i=entry_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold))
        last_exit = entry_i + hold
    return evs


ALL_EVENTS = build_events()
holdout_events = [e for e in ALL_EVENTS if IDX1H[e["entry_i"]] >= CUTOFF]
w(f"\n[EVENTS] {len(ALL_EVENTS)} trade totali  |  holdout 2025-2026: {len(holdout_events)}")


def run_bt(evs, rr, fee_entry_pct, fee_exit_pct):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[])
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    for ev in evs:
        d, ep, sl, risk, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"]
        target = ep + d * rr * risk
        exit_price = None
        for k in range(hold):
            j = entry_i + k
            if j >= N1H: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl:
                exit_price = sl; break
            if hit_tp:
                exit_price = target; break
        if exit_price is None:
            j = min(entry_i + hold, N1H - 1)
            exit_price = CL[j]
        if risk <= 0: continue
        r = INIT_CAP * RISK_PCT
        units = min(r / risk, MAX_LEV * INIT_CAP / ep)
        notional_entry = units * ep
        notional_exit = units * exit_price
        pnl = units * (exit_price - ep) * d - fee_entry_pct * notional_entry - fee_exit_pct * notional_exit
        cap += pnl
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl > 0)
        net_pnls.append(pnl)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls)


def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=10):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


for name, fee_e, fee_x in FEE_SCENARIOS:
    w(f"\n{SEP}")
    w(f"{name}  (RT={100*(fee_e+fee_x):.3f}%)")
    w(SEP)

    res = run_bt(ALL_EVENTS, BEST_RR, fee_e, fee_x)
    mc = mc_summary(res["net_pnls"]); mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL-SAMPLE: n={res['n']}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    hres = run_bt(holdout_events, BEST_RR, fee_e, fee_x)
    hmc = mc_summary(hres["net_pnls"]); hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vp_vwap_confluence_makerfee.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Volume Profile + VWAP Confluence (RR=3.0) — Sensibilità alle fee\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
