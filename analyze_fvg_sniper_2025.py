#!/usr/bin/env python3
"""
analyze_fvg_sniper_2025.py
=============================
Diagnostica: perché il 2025 è stato l'unico anno negativo (-56%) per
"IFVG Flip Only" su BTCUSDT 1H (v2, filtro anti-leva + posizioni
concorrenti)? Riproduce esattamente i trade di quella configurazione e li
scompone per: direzione (long/short), motivo di uscita (TP/SL/tempo),
R medio realizzato — confrontando 2025 con gli altri 6 anni. Aggiunge
statistiche di regime di mercato per anno (rendimento buy&hold, volatilità
annualizzata, frequenza di whipsaw del bias 4H EMA50) per capire se il
2025 è stato strutturalmente diverso (trend forte vs chop) rispetto agli
anni in cui il segnale ha funzionato.
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

print("[DATA] Loading 4H, 1H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
d = add_indicators(raw["1H"])
print(f"  4H: {len(df4h):,} bars   1H: {len(d):,} bars")

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
N = len(d)
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
                            signals.append(dict(idx=i, dir=-1, top=f["top"], bot=f["bot"]))
                else:
                    if HI[i] >= f["top"]:
                        f["mitigated"] = True
                    if CL[i] > f["top"]:
                        f["inverted"] = True
                        if htf_bias[i] >= 0 and f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=1, top=f["top"], bot=f["bot"]))
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
        evs.append(dict(idx=i, entry_i=entry_i, d=dd, ep=ep, sl=sl, target=target, risk=risk, hold=hold))
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
    open_positions = []
    total_open_risk = 0.0
    trades_out = []

    def close_position(pos, exit_price, exit_reason, exit_i):
        nonlocal total_open_risk
        dd, ep, risk = pos["d"], pos["ep"], pos["risk"]
        units = risk_budget_dollar / risk
        fill_ep = ep * (1 + dd * slip_pct); fill_xp = exit_price * (1 - dd * slip_pct)
        pnl = units * (fill_xp - fill_ep) * dd - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        r_mult = pnl / risk_budget_dollar
        trades_out.append(dict(entry_i=pos["entry_i"], exit_i=exit_i, d=dd, pnl=pnl, r_mult=r_mult,
                                exit_reason=exit_reason, hold_bars=exit_i - pos["entry_i"]))
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
                                        target=ev["target"], risk=ev["risk"], deadline_i=ev["entry_i"] + ev["hold"]))
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
tdf["year"] = d.index[tdf["entry_i"].values].year
tdf["dir_label"] = np.where(tdf["d"] == 1, "LONG", "SHORT")
print(f"  {len(tdf):,} trade totali")

print(f"\n{SEP}\nBREAKDOWN PER ANNO — n, ret%, WR, R medio, per direzione ed exit reason\n{SEP}")
for yr, g in tdf.groupby("year"):
    ret_pct = g["pnl"].sum() / INIT_CAP * 100
    wr = (g["pnl"] > 0).mean() * 100
    avg_r = g["r_mult"].mean()
    n_long = (g["d"] == 1).sum(); n_short = (g["d"] == -1).sum()
    long_pnl = g.loc[g["d"] == 1, "pnl"].sum(); short_pnl = g.loc[g["d"] == -1, "pnl"].sum()
    long_wr = (g.loc[g["d"] == 1, "pnl"] > 0).mean() * 100 if n_long else float("nan")
    short_wr = (g.loc[g["d"] == -1, "pnl"] > 0).mean() * 100 if n_short else float("nan")
    exit_counts = g["exit_reason"].value_counts().to_dict()
    print(f"\n  {yr}: n={len(g)}  ret={ret_pct:+.1f}%  wr={wr:.1f}%  avgR={avg_r:+.3f}")
    print(f"    LONG : n={n_long:>4}  pnl={long_pnl:>+12,.0f}  wr={long_wr:>5.1f}%")
    print(f"    SHORT: n={n_short:>4}  pnl={short_pnl:>+12,.0f}  wr={short_wr:>5.1f}%")
    print(f"    exits: {exit_counts}")

print(f"\n{SEP}\nREGIME DI MERCATO PER ANNO (BTC 1H)\n{SEP}")
d["ret1h"] = d["close"].pct_change()
for yr, g in d.groupby(d.index.year):
    if yr < START_YEAR or len(g) < 100:
        continue
    bh_ret = (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100
    ann_vol = g["ret1h"].std() * np.sqrt(24 * 365) * 100
    atr_pct = (g["atr_14"] / g["close"]).mean() * 100
    # frequenza di whipsaw del bias 4H (quante volte cambia segno per anno)
    print(f"  {yr}: buy&hold={bh_ret:>+7.1f}%   vol_ann={ann_vol:>6.1f}%   ATR%_medio={atr_pct:>5.2f}%")

# whipsaw del bias 4H per anno (quante volte l'EMA50 4H cambia segno)
bias_series = pd.Series(htf_bias_4h, index=df4h.index)
flips = (bias_series != bias_series.shift(1)).astype(int)
flips.iloc[0] = 0
print(f"\n{SEP}\nWHIPSAW BIAS 4H EMA50 — cambi di segno per anno\n{SEP}")
for yr, g in flips.groupby(flips.index.year):
    if yr < START_YEAR:
        continue
    n_bars = len(g)
    print(f"  {yr}: {int(g.sum())} cambi di bias su {n_bars} barre 4H "
          f"({int(g.sum())/(n_bars/ (365*6)):.1f} cambi/anno equiv.)" if n_bars else "")

print(f"\n{SEP}\nDETTAGLIO 2025: trade mese per mese\n{SEP}")
t2025 = tdf[tdf["year"] == 2025].copy()
t2025["month"] = d.index[t2025["entry_i"].values].month
for mo, g in t2025.groupby("month"):
    ret_pct = g["pnl"].sum() / INIT_CAP * 100
    wr = (g["pnl"] > 0).mean() * 100
    print(f"  2025-{mo:02d}: n={len(g):>3}  ret={ret_pct:>+7.2f}%  wr={wr:>5.1f}%  "
          f"long={int((g['d']==1).sum())}  short={int((g['d']==-1).sum())}")

print(f"\n{SEP}\nDONE\n{SEP}")
