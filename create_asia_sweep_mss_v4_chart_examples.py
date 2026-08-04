#!/usr/bin/env python3
"""
create_asia_sweep_mss_v3_chart_examples.py
=============================================
Grafici di esempio per la strategia Asia Range Sweep + MSS + FVG/OB v3
(create_asia_sweep_mss_v3_report.py): candlestick M15 con marker per range
Asia, sweep (con rigetto), market structure shift, equilibrium (50% ICT),
zona FVG/OB in discount/premium, entry, target, stop, uscita.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from src.strategy.data_fetcher import fetch_binance_vision_klines
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import causal_trend_state

START_YEAR = 2020
BAR_MIN = 15
SCALE = 60 // BAR_MIN

ASIA_START_H, ASIA_END_H = 0, 7
MIN_ASIA_BARS = 4 * SCALE
PIVOT_LR = 3
MSS_MAX_BARS = 24 * SCALE
FVG_OB_SEARCH_EXT = 6 * SCALE
FVG_OB_MAX_AGE = 24 * SCALE
MIN_ZONE_ATR_FRAC = 0.05
SWING_ATR_OB = 1.5
OB_SWING_BARS = 8 * SCALE
MAX_HOLD = 48 * SCALE
RR = 2.0
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0004
MAX_LEV = 10.0

_BG = "#0f1117"; _CARD = "#12151f"; _GRID = "#1e2130"
_TEXT = "#e0e0e0"; _GRN = "#66bb6a"; _RED = "#ef5350"
_YEL = "#ffd54f"; _BLU = "#42a5f5"; _ORG = "#ffa726"; _PUR = "#ab47bc"

print(f"[DATA] Loading {BAR_MIN}m …")
df15 = add_indicators(fetch_binance_vision_klines(
    f"{BAR_MIN}m", start_year=START_YEAR, start_month=1, workers=6, verbose=False))
IDX1H = df15.index
N1H = len(df15)

OP = df15["open"].values.astype(float)
CL = df15["close"].values.astype(float)
HI = df15["high"].values.astype(float)
LO = df15["low"].values.astype(float)
ATR = np.where(df15["atr_14"].values > 0, df15["atr_14"].values, 1.0)
hour_arr = IDX1H.hour.values
date_arr = IDX1H.normalize()

trend = causal_trend_state(CL, HI, LO, PIVOT_LR, PIVOT_LR)
LAST_PH = trend["last_pivot_high"].values
LAST_PL = trend["last_pivot_low"].values
TREND_STATE = trend["trend_state"].values
_range = HI - LO
AVG_RANGE_3 = pd.Series(_range).rolling(3).mean().shift(1).values
DISPLACEMENT_MULT = 2.0


def find_fvg_entry(direction, leg_start, leg_end, mss_i, cutoff_i, equilibrium):
    best = None
    for i in range(max(leg_start, 1), min(leg_end, N1H - 2)):
        atr = ATR[i]
        if atr <= 0: continue
        if direction == 1 and HI[i - 1] < LO[i + 1]:
            fbot, ftop = HI[i - 1], LO[i + 1]
            fsz = ftop - fbot
            if fsz < MIN_ZONE_ATR_FRAC * atr: continue
            start_j = max(i + 2, mss_i + 1)
            for j in range(start_j, min(i + 2 + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] < fbot: break
                if LO[j] <= ftop:
                    fill = min(CL[j], ftop)
                    if fill < equilibrium:
                        trig = dict(j=j, price=fill, kind="FVG", zone_lo=fbot, zone_hi=ftop, form_i=i)
                        if best is None or trig["j"] < best["j"]: best = trig
                    break
        if direction == -1 and LO[i - 1] > HI[i + 1]:
            ftop, fbot = LO[i - 1], HI[i + 1]
            fsz = ftop - fbot
            if fsz < MIN_ZONE_ATR_FRAC * atr: continue
            start_j = max(i + 2, mss_i + 1)
            for j in range(start_j, min(i + 2 + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] > ftop: break
                if HI[j] >= fbot:
                    fill = max(CL[j], fbot)
                    if fill > equilibrium:
                        trig = dict(j=j, price=fill, kind="FVG", zone_lo=fbot, zone_hi=ftop, form_i=i)
                        if best is None or trig["j"] < best["j"]: best = trig
                    break
    return best


def find_ob_entry(direction, leg_start, leg_end, mss_i, cutoff_i, equilibrium):
    best = None
    for i in range(max(leg_start, 0), min(leg_end, N1H - OB_SWING_BARS)):
        atr = ATR[i]
        if atr <= 0: continue
        if direction == 1 and OP[i] > CL[i]:
            ob_hi, ob_lo = HI[i], LO[i]
            if ob_hi - ob_lo < 1e-9: continue
            swing_end = -1; swing_start = CL[i]
            for s in range(i + 1, min(i + OB_SWING_BARS, N1H)):
                if CL[s] - swing_start >= SWING_ATR_OB * atr: swing_end = s; break
            if swing_end < 0: continue
            start_j = max(swing_end + 1, mss_i + 1)
            for j in range(start_j, min(i + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] < ob_lo: break
                if LO[j] <= ob_hi and HI[j] >= ob_lo:
                    fill = max(CL[j], ob_lo)
                    if fill < equilibrium:
                        trig = dict(j=j, price=fill, kind="OB", zone_lo=ob_lo, zone_hi=ob_hi, form_i=i)
                        if best is None or trig["j"] < best["j"]: best = trig
                    break
        if direction == -1 and OP[i] < CL[i]:
            ob_hi, ob_lo = HI[i], LO[i]
            if ob_hi - ob_lo < 1e-9: continue
            swing_end = -1; swing_start = CL[i]
            for s in range(i + 1, min(i + OB_SWING_BARS, N1H)):
                if swing_start - CL[s] >= SWING_ATR_OB * atr: swing_end = s; break
            if swing_end < 0: continue
            start_j = max(swing_end + 1, mss_i + 1)
            for j in range(start_j, min(i + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] > ob_hi: break
                if HI[j] >= ob_lo and LO[j] <= ob_hi:
                    fill = min(CL[j], ob_hi)
                    if fill > equilibrium:
                        trig = dict(j=j, price=fill, kind="OB", zone_lo=ob_lo, zone_hi=ob_hi, form_i=i)
                        if best is None or trig["j"] < best["j"]: best = trig
                    break
    return best


print("[EVENTS] Scanning full sequences …")
all_days = pd.date_range(IDX1H[0].normalize(), IDX1H[-1].normalize(), freq="D")
events = []

for day in all_days:
    asia_mask = (date_arr == day) & (hour_arr >= ASIA_START_H) & (hour_arr < ASIA_END_H)
    asia_idx = np.where(asia_mask)[0]
    if len(asia_idx) < MIN_ASIA_BARS: continue
    asia_hi = float(HI[asia_idx].max()); asia_lo = float(LO[asia_idx].min())
    if asia_hi <= asia_lo: continue

    post_mask = (date_arr == day) & (hour_arr >= ASIA_END_H)
    post_idx = np.where(post_mask)[0]
    if len(post_idx) == 0: continue

    sweep_i = -1; direction = 0
    for k in post_idx:
        body = abs(CL[k] - OP[k])
        lower_wick = min(OP[k], CL[k]) - LO[k]
        upper_wick = HI[k] - max(OP[k], CL[k])
        bull = (LO[k] < asia_lo) and (CL[k] > asia_lo) and (lower_wick > body)
        bear = (HI[k] > asia_hi) and (CL[k] < asia_hi) and (upper_wick > body)
        if bull and bear: break
        if bull: sweep_i = k; direction = 1; break
        if bear: sweep_i = k; direction = -1; break
    if sweep_i < 0: continue

    ctx_state = TREND_STATE[sweep_i]
    if direction == 1 and ctx_state != -1: continue
    if direction == -1 and ctx_state != 1: continue

    ref_level = LAST_PH[sweep_i] if direction == 1 else LAST_PL[sweep_i]
    if np.isnan(ref_level): continue

    mss_i = -1
    for j in range(sweep_i + 1, min(sweep_i + 1 + MSS_MAX_BARS, N1H)):
        broke = (direction == 1 and CL[j] > ref_level) or (direction == -1 and CL[j] < ref_level)
        if not broke: continue
        avg_r = AVG_RANGE_3[j]
        if np.isnan(avg_r) or avg_r <= 0: continue
        if (HI[j] - LO[j]) >= DISPLACEMENT_MULT * avg_r:
            mss_i = j; break
    if mss_i < 0: continue

    if direction == 1:
        leg_lo, leg_hi = LO[sweep_i], ref_level
    else:
        leg_lo, leg_hi = ref_level, HI[sweep_i]
    equilibrium = (leg_lo + leg_hi) / 2.0

    leg_start = sweep_i
    leg_end = min(mss_i + FVG_OB_SEARCH_EXT, N1H - 2)
    cutoff_i = min(mss_i + 1 + FVG_OB_MAX_AGE, N1H)

    trig_fvg = find_fvg_entry(direction, leg_start, leg_end, mss_i, cutoff_i, equilibrium)
    trig_ob = find_ob_entry(direction, leg_start, leg_end, mss_i, cutoff_i, equilibrium)
    candidates = [t for t in (trig_fvg, trig_ob) if t is not None]
    if not candidates: continue
    trig = min(candidates, key=lambda t: t["j"])
    entry_i, entry_price = trig["j"], trig["price"]

    target = asia_hi if direction == 1 else asia_lo
    if (direction == 1 and target <= entry_price) or (direction == -1 and target >= entry_price):
        continue
    target_dist = abs(target - entry_price)
    if target_dist < 0.05 * ATR[entry_i]: continue

    events.append(dict(day=day, direction=direction, asia_hi=asia_hi, asia_lo=asia_lo,
                        sweep_i=sweep_i, ref_level=ref_level, mss_i=mss_i,
                        equilibrium=equilibrium, leg_lo=leg_lo, leg_hi=leg_hi,
                        entry_i=entry_i, entry_price=entry_price, target=target,
                        target_dist=target_dist, zone=trig))

print(f"  {len(events)} trade candidati")


def simulate(ev):
    d = ev["direction"]; ep = ev["entry_price"]; target = ev["target"]
    stop_dist = ev["target_dist"] / RR
    sl = ep - d * stop_dist
    i0 = ev["entry_i"]
    out = "time"; exit_i = min(i0 + MAX_HOLD, N1H - 1); exit_price = CL[exit_i]
    for k in range(1, MAX_HOLD + 1):
        j = i0 + k
        if j >= N1H: break
        hk, lk = HI[j], LO[j]
        if d == 1:
            hit_sl = lk <= sl; hit_tp = hk >= target
        else:
            hit_sl = hk >= sl; hit_tp = lk <= target
        if hit_sl: out = "sl"; exit_price = sl; exit_i = j; break
        if hit_tp: out = "tp"; exit_price = target; exit_i = j; break
    risk = INIT_CAP * RISK_PCT
    units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep) if stop_dist > 0 else 0
    notional = units * ep
    pnl = units * (exit_price - ep) * d - FEE * 2 * notional
    return dict(sl=sl, out=out, exit_i=exit_i, exit_price=exit_price, pnl=pnl)


for ev in events:
    ev["sim"] = simulate(ev)

wins = [e for e in events if e["sim"]["pnl"] > 0]
losses = [e for e in events if e["sim"]["pnl"] <= 0]
print(f"  wins={len(wins)}  losses={len(losses)}")

def pick_examples(pool, n):
    pool_sorted = sorted(pool, key=lambda e: e["day"])
    chosen = []; seen_years = set()
    for e in pool_sorted:
        y = e["day"].year
        if y in seen_years: continue
        chosen.append(e); seen_years.add(y)
        if len(chosen) == n: break
    if len(chosen) < n:
        for e in pool_sorted:
            if e not in chosen: chosen.append(e)
            if len(chosen) == n: break
    return chosen

examples = pick_examples(wins, 2) + pick_examples(losses, 2)
print(f"  Esempi selezionati: {[(e['day'].date(), e['sim']['out'], round(e['sim']['pnl'])) for e in examples]}")


def plot_candles(ax, idx_range):
    for i in idx_range:
        t = IDX1H[i]
        color = _GRN if CL[i] >= OP[i] else _RED
        ax.plot([t, t], [LO[i], HI[i]], color=color, linewidth=0.8, zorder=2)
        body_lo, body_hi = min(OP[i], CL[i]), max(OP[i], CL[i])
        width = pd.Timedelta(minutes=10)
        ax.add_patch(Rectangle((mdates.date2num(t) - width.total_seconds() / 86400 / 2, body_lo),
                                width.total_seconds() / 86400, max(body_hi - body_lo, 1e-6),
                                facecolor=color, edgecolor=color, zorder=3))


fig, axes = plt.subplots(2, 2, figsize=(20, 12), facecolor=_BG)
fig.suptitle("Asia Range Sweep + MSS + FVG/OB v4 (trend context + displacement MSS) "
              "— Trade Examples (RR=2:1)", color=_TEXT, fontsize=14, y=0.98)

for ax, ev in zip(axes.flat, examples):
    ax.set_facecolor(_CARD)
    asia_start_i = int(np.where((date_arr == ev["day"]) & (hour_arr == ASIA_START_H))[0][0])
    plot_start = max(min(ev["sweep_i"] - 12, asia_start_i - 2), 0)
    plot_end = min(ev["sim"]["exit_i"] + 8, N1H - 1)
    idx_range = range(plot_start, plot_end + 1)
    plot_candles(ax, idx_range)

    t0, t1 = IDX1H[plot_start], IDX1H[plot_end]

    asia_day_start = pd.Timestamp(ev["day"]) + pd.Timedelta(hours=ASIA_START_H)
    asia_day_end = pd.Timestamp(ev["day"]) + pd.Timedelta(hours=ASIA_END_H)
    ax.add_patch(Rectangle((mdates.date2num(asia_day_start), ev["asia_lo"]),
                            mdates.date2num(asia_day_end) - mdates.date2num(asia_day_start),
                            ev["asia_hi"] - ev["asia_lo"],
                            facecolor=_BLU, alpha=0.15, edgecolor=_BLU, linewidth=1, zorder=1,
                            label="Asia range"))

    zone = ev["zone"]
    form_t = IDX1H[zone["form_i"]]
    entry_t = IDX1H[ev["entry_i"]]
    zcolor = _PUR if zone["kind"] == "FVG" else _ORG
    ax.add_patch(Rectangle((mdates.date2num(form_t), zone["zone_lo"]),
                            mdates.date2num(entry_t) - mdates.date2num(form_t) + 0.005,
                            zone["zone_hi"] - zone["zone_lo"],
                            facecolor=zcolor, alpha=0.3, edgecolor=zcolor, linewidth=1, zorder=1,
                            label=f"{zone['kind']} zone"))

    # equilibrium (50% ICT) del trading range sweep -> ref_level
    ax.axhline(ev["equilibrium"], color="white", linestyle="-.", linewidth=1, alpha=0.5,
               label="Equilibrium 50%")

    sweep_t = IDX1H[ev["sweep_i"]]
    sweep_y = LO[ev["sweep_i"]] if ev["direction"] == 1 else HI[ev["sweep_i"]]
    ax.scatter([sweep_t], [sweep_y], marker="*", s=220, color=_YEL, zorder=5,
               edgecolor="black", linewidth=0.5, label="Sweep")

    mss_t = IDX1H[ev["mss_i"]]
    ax.axhline(ev["ref_level"], color=_YEL, linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(mss_t, color=_YEL, linestyle="--", linewidth=1, alpha=0.6)
    ax.scatter([mss_t], [CL[ev["mss_i"]]], marker="D", s=70, color=_YEL, zorder=5,
               edgecolor="black", linewidth=0.5, label="MSS")

    ax.scatter([entry_t], [ev["entry_price"]], marker="^" if ev["direction"] == 1 else "v",
               s=160, color="white", zorder=6, edgecolor="black", linewidth=0.8, label="Entry")

    ax.axhline(ev["target"], color=_GRN, linestyle="--", linewidth=1.3, label="Target")
    ax.axhline(ev["sim"]["sl"], color=_RED, linestyle="--", linewidth=1.3, label="Stop")

    exit_t = IDX1H[ev["sim"]["exit_i"]]
    out_color = _GRN if ev["sim"]["out"] == "tp" else (_RED if ev["sim"]["out"] == "sl" else _YEL)
    ax.scatter([exit_t], [ev["sim"]["exit_price"]], marker="X", s=140, color=out_color,
               zorder=6, edgecolor="black", linewidth=0.8, label="Exit")

    dirlabel = "LONG" if ev["direction"] == 1 else "SHORT"
    zonelabel = "discount" if ev["direction"] == 1 else "premium"
    ax.set_title(f"{ev['day'].date()}  {dirlabel} ({zonelabel})  |  exit={ev['sim']['out'].upper()}  "
                 f"pnl=${ev['sim']['pnl']:,.0f}", color=_TEXT, fontsize=11)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    for spine in ax.spines.values(): spine.set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=7, facecolor=_CARD, edgecolor=_GRID,
              labelcolor=_TEXT, ncol=2)
    ax.set_xlim(t0, t1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = Path("reports/asia_sweep_mss_v4_examples.png")
plt.savefig(out_path, dpi=130, facecolor=_BG)
print(f"[DONE] {out_path}")
