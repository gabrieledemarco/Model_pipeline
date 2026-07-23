#!/usr/bin/env python3
"""
analyze_fvg_sniper_holdout_losses.py
=======================================
Autopsy bottom-up (complementare alla diagnosi top-down per anno/chop già
fatta): tra TUTTI i trade IFVG Flip Only (1H, v2 anti-leva + posizioni
concorrenti, SENZA filtro di regime — il segnale base) con entry nel
holdout 2025-2026, quali sono quelli che causano la perdita maggiore?
Che caratteristiche condividono (direzione, ER al momento del segnale,
grade del gap, clustering temporale — un singolo evento che ne spazza
via molti insieme vs. tante piccole perdite indipendenti)?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
MAX_TOTAL_RISK_PCT = 0.05
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")

ATR_LEN = 14
MIN_GAP_ATR = 0.25
MIN_BODY_RATIO = 0.45
MIN_GRADE = 4.0
HTF_EMA_LEN = 50
SL_BUF_ATR = 0.20
MIN_RISK_ATR = 0.40
MAX_RISK_ATR = 4.00
TP3_R = 3.0
MAX_TRADE_BARS = 120
GAP_MAX_AGE_BARS = 220
ER_WINDOWS = [96, 168, 336]

print("[DATA] Loading 4H, 1H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
d = add_indicators(raw["1H"])
IDX = d.index; N = len(d)
print(f"  4H: {len(df4h):,} bars   1H: {N:,} bars")

ema50_4h = df4h["close"].ewm(span=HTF_EMA_LEN, adjust=False).mean().values
htf_bias_4h = np.where(df4h["close"].values > ema50_4h, 1, -1)
IDX4H_vals = df4h.index.values


def map_htf_bias(idx_ltf):
    pos = np.searchsorted(IDX4H_vals, idx_ltf.values, side="right") - 1
    pos = np.clip(pos, 0, len(htf_bias_4h) - 1)
    bias = htf_bias_4h[pos]
    bias[pos < 0] = 0
    return bias


HI = d["high"].values.astype(float); LO = d["low"].values.astype(float)
OP = d["open"].values.astype(float); CL = d["close"].values.astype(float)
VOL = d["volume"].values.astype(float)
ATR = np.where(d["atr_14"].values > 0, d["atr_14"].values, np.nan)
htf_bias = map_htf_bias(d.index)


def detect_graded_fvgs(HI, LO, OP, CL, VOL, ATR):
    n = len(CL)
    vol_avg = pd.Series(VOL).rolling(20, min_periods=1).mean().values
    events = []
    for i in range(2, n):
        atr_i = ATR[i]
        safe_atr = atr_i if (not np.isnan(atr_i) and atr_i > 0) else 1e-9
        body_ratio = abs(CL[i - 1] - OP[i - 1]) / max(HI[i - 1] - LO[i - 1], 1e-9)
        disp3 = (HI[i - 1] - LO[i - 1]) / safe_atr
        vol_score = min(VOL[i - 1] / vol_avg[i - 1], 2.0) / 2.0 if (vol_avg[i - 1] > 0 and not np.isnan(vol_avg[i - 1])) else 0.5
        if LO[i] > HI[i - 2]:
            g_top, g_bot = LO[i], HI[i - 2]
            gap_atr = (g_top - g_bot) / safe_atr
            if gap_atr >= MIN_GAP_ATR and body_ratio >= MIN_BODY_RATIO:
                size_score = min(gap_atr / 1.0, 1.0); disp_score = min(disp3 / 2.0, 1.0)
                grade = min(10.0, (size_score * 0.40 + disp_score * 0.35 + vol_score * 0.25) * 10.0)
                events.append(dict(idx=i, dir=1, top=g_top, bot=g_bot, grade=grade))
        elif HI[i] < LO[i - 2]:
            g_top, g_bot = LO[i - 2], HI[i]
            gap_atr = (g_top - g_bot) / safe_atr
            if gap_atr >= MIN_GAP_ATR and body_ratio >= MIN_BODY_RATIO:
                size_score = min(gap_atr / 1.0, 1.0); disp_score = min(disp3 / 2.0, 1.0)
                grade = min(10.0, (size_score * 0.40 + disp_score * 0.35 + vol_score * 0.25) * 10.0)
                events.append(dict(idx=i, dir=-1, top=g_top, bot=g_bot, grade=grade))
    return events


def run_engine_ifvg_only(fvg_births, HI, LO, OP, CL, N, htf_bias):
    births_by_idx = {}
    for f in fvg_births:
        births_by_idx.setdefault(f["idx"], []).append(f)
    active = []
    signals = []
    for i in range(N):
        still_active = []
        for f in active:
            age = i - f["born"]
            if not f["inverted"]:
                if f["dir"] == 1:
                    if LO[i] <= f["bot"]:
                        f["mitigated"] = True
                    if CL[i] < f["bot"]:
                        f["inverted"] = True
                        if htf_bias[i] <= 0 and f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=-1, top=f["top"], bot=f["bot"], grade=f["grade"]))
                else:
                    if HI[i] >= f["top"]:
                        f["mitigated"] = True
                    if CL[i] > f["top"]:
                        f["inverted"] = True
                        if htf_bias[i] >= 0 and f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=1, top=f["top"], bot=f["bot"], grade=f["grade"]))
            if age <= GAP_MAX_AGE_BARS:
                still_active.append(f)
        active = still_active
        for f in births_by_idx.get(i, []):
            active.append(dict(born=i, dir=f["dir"], top=f["top"], bot=f["bot"], grade=f["grade"],
                                mitigated=False, inverted=False))
    return signals


def build_candidates(signals, HI, LO, OP, CL, ATR, N):
    evs = []
    for s in sorted(signals, key=lambda s: s["idx"]):
        i = s["idx"]; entry_i = i + 1
        if entry_i >= N - 1: continue
        atr_i = ATR[i]
        if np.isnan(atr_i) or atr_i <= 0: continue
        dd = s["dir"]; ep = OP[entry_i]
        raw_stop = (s["bot"] - atr_i * SL_BUF_ATR) if dd == 1 else (s["top"] + atr_i * SL_BUF_ATR)
        risk0 = abs(ep - raw_stop)
        risk = min(max(risk0, atr_i * MIN_RISK_ATR), atr_i * MAX_RISK_ATR)
        if risk <= 0: continue
        if risk < RISK_PCT * ep / MAX_LEV: continue
        sl = ep - dd * risk; target = ep + dd * TP3_R * risk
        hold = min(MAX_TRADE_BARS, N - 1 - entry_i)
        if hold < 1: continue
        evs.append(dict(idx=i, entry_i=entry_i, d=dd, ep=ep, sl=sl, target=target, risk=risk,
                         hold=hold, grade=s["grade"]))
    return evs


def run_bt_portfolio_detailed(evs, HI, LO, CL, N, max_total_risk_pct=MAX_TOTAL_RISK_PCT):
    slip_pct = SLIPPAGE_BASE
    risk_budget_dollar = INIT_CAP * RISK_PCT
    max_total_risk_dollar = INIT_CAP * max_total_risk_pct
    by_entry: dict[int, list] = {}
    for ev in evs:
        by_entry.setdefault(ev["entry_i"], []).append(ev)
    if not evs:
        return []
    min_i = min(by_entry); max_i = max(ev["entry_i"] + ev["hold"] for ev in evs)
    open_positions = []; total_open_risk = 0.0
    trades_out = []

    def close_position(pos, exit_price, exit_reason, exit_i):
        nonlocal total_open_risk
        dd, ep, risk = pos["d"], pos["ep"], pos["risk"]
        units = risk_budget_dollar / risk
        fill_ep = ep * (1 + dd * slip_pct); fill_xp = exit_price * (1 - dd * slip_pct)
        pnl = units * (fill_xp - fill_ep) * dd - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        r_mult = pnl / risk_budget_dollar
        trades_out.append(dict(entry_i=pos["entry_i"], exit_i=exit_i, d=dd, pnl=pnl, r_mult=r_mult,
                                exit_reason=exit_reason, hold_bars=exit_i - pos["entry_i"],
                                grade=pos["grade"], idx=pos["idx"]))
        total_open_risk -= risk_budget_dollar

    for i in range(min_i, max_i + 1):
        still_open = []
        for pos in open_positions:
            dd, sl, target, deadline_i = pos["d"], pos["sl"], pos["target"], pos["deadline_i"]
            hk, lk = HI[i], LO[i]
            hit_sl = (lk <= sl) if dd == 1 else (hk >= sl)
            hit_tp = (hk >= target) if dd == 1 else (lk <= target)
            if hit_sl: close_position(pos, sl, "SL", i); continue
            if hit_tp: close_position(pos, target, "TP", i); continue
            if i >= deadline_i: close_position(pos, CL[min(i, N - 1)], "TIME", i); continue
            still_open.append(pos)
        open_positions = still_open
        for ev in by_entry.get(i, []):
            if total_open_risk + risk_budget_dollar > max_total_risk_dollar + 1e-9:
                continue
            open_positions.append(dict(entry_i=ev["entry_i"], d=ev["d"], ep=ev["ep"], sl=ev["sl"],
                                        target=ev["target"], risk=ev["risk"], deadline_i=ev["entry_i"] + ev["hold"],
                                        grade=ev["grade"], idx=ev["idx"]))
            total_open_risk += risk_budget_dollar
    for pos in open_positions:
        close_position(pos, CL[min(max_i, N - 1)], "TIME", max_i)
    return trades_out


print("\n[ENGINE] Reproducing IFVG Flip Only trades …")
fvgs = detect_graded_fvgs(HI, LO, OP, CL, VOL, ATR)
signals = run_engine_ifvg_only(fvgs, HI, LO, OP, CL, N, htf_bias)
cands = build_candidates(signals, HI, LO, OP, CL, ATR, N)
trades = run_bt_portfolio_detailed(cands, HI, LO, CL, N)
tdf = pd.DataFrame(trades)
tdf["entry_ts"] = d.index[tdf["entry_i"].values]
tdf["exit_ts"] = d.index[tdf["exit_i"].clip(upper=N - 1).values]
tdf["dir_label"] = np.where(tdf["d"] == 1, "LONG", "SHORT")

abs_diff = np.abs(np.diff(CL, prepend=CL[0]))
for W in ER_WINDOWS:
    path_sum = pd.Series(abs_diff).rolling(W, min_periods=W).sum().values
    net_move = np.abs(CL - np.roll(CL, W)); net_move[:W] = np.nan
    er = np.where(path_sum > 0, net_move / path_sum, np.nan)
    tdf[f"er_{W}"] = er[tdf["idx"].values]

hdf = tdf[tdf["entry_ts"] >= CUTOFF].copy().sort_values("pnl")
print(f"  {len(tdf):,} trade totali -> {len(hdf):,} in holdout 2025-2026")
print(f"  Perdita totale holdout: {hdf['pnl'].sum():,.0f}$  ({hdf['pnl'].sum()/INIT_CAP*100:+.1f}%)")

print(f"\n{SEP}\nI 25 PEGGIORI TRADE DEL 2025-2026\n{SEP}")
worst = hdf.head(25)
print(f"\n  {'Entry':<17}{'Dir':<7}{'Exit':<6}{'Hold':>6}{'R':>7}{'PnL':>10}{'Grade':>7}{'ER96':>7}{'ER168':>7}{'ER336':>7}")
for _, t in worst.iterrows():
    print(f"  {str(t['entry_ts'])[:16]:<17}{t['dir_label']:<7}{t['exit_reason']:<6}{t['hold_bars']:>6}"
          f"{t['r_mult']:>+7.2f}{t['pnl']:>+10,.0f}{t['grade']:>7.1f}"
          f"{t['er_96']:>7.3f}{t['er_168']:>7.3f}{t['er_336']:>7.3f}")

worst_pnl_sum = worst["pnl"].sum()
print(f"\n  Somma perdita dei 25 peggiori: {worst_pnl_sum:,.0f}$ "
      f"({worst_pnl_sum/hdf['pnl'].sum()*100:.1f}% della perdita totale holdout, "
      f"su {len(worst)}/{len(hdf)} trade = {len(worst)/len(hdf)*100:.1f}% del conteggio)")

print(f"\n{SEP}\nCARATTERISTICHE: 25 PEGGIORI vs RESTO DEL HOLDOUT\n{SEP}")
rest = hdf.iloc[25:]
for col, label in [("er_96", "ER 96h"), ("er_168", "ER 168h"), ("er_336", "ER 336h"), ("grade", "Grade FVG")]:
    print(f"  {label:<12}: peggiori={worst[col].mean():.3f}   resto={rest[col].mean():.3f}")
print(f"  Direzione   : peggiori LONG={int((worst['d']==1).sum())}/{len(worst)}   "
      f"resto LONG={int((rest['d']==1).sum())}/{len(rest)}")
print(f"  Exit reason peggiori: {worst['exit_reason'].value_counts().to_dict()}")
print(f"  Exit reason resto   : {rest['exit_reason'].value_counts().to_dict()}")

print(f"\n{SEP}\nCLUSTERING TEMPORALE — i 25 peggiori raggruppati per finestre di 72h\n{SEP}")
worst_sorted = worst.sort_values("entry_ts")
clusters = []
cur_cluster = [worst_sorted.iloc[0]]
for i in range(1, len(worst_sorted)):
    row = worst_sorted.iloc[i]
    if (row["entry_ts"] - cur_cluster[-1]["entry_ts"]) <= pd.Timedelta(hours=72):
        cur_cluster.append(row)
    else:
        clusters.append(cur_cluster)
        cur_cluster = [row]
clusters.append(cur_cluster)
for c in clusters:
    total = sum(r["pnl"] for r in c)
    print(f"  {str(c[0]['entry_ts'])[:16]} -> {str(c[-1]['entry_ts'])[:16]}  "
          f"n={len(c)}  pnl_totale={total:>+10,.0f}$  dirs={[r['dir_label'] for r in c]}")
print(f"\n  -> {len(clusters)} cluster distinti spiegano i 25 peggiori trade "
      f"({len(worst)} trade in {len(clusters)} eventi)")

print(f"\n{SEP}\nDONE\n{SEP}")
