#!/usr/bin/env python3
"""
create_ict_objects_chart_examples.py
=======================================
Esempi visivi reali (non sintetici) degli oggetti rilevati algoritmicamente
in `ict_objects_predictive.md` e `htf_ltf_confluence.md`: prende il primo
evento "pulito" trovato dai detector sui dati reali BTCUSDT 15M (+ un
esempio di confluenza 4H/15M) e lo disegna su un grafico a candele, per
verifica visiva diretta di cosa il codice sta effettivamente riconoscendo.
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

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import find_pivots

_BG = "#0f1117"; _CARD = "#12151f"
_TEXT = "#e0e0e0"; _GRN = "#66bb6a"; _RED = "#ef5350"
_YEL = "#ffd54f"; _BLU = "#42a5f5"; _ORG = "#ffa726"; _PUR = "#ab47bc"; _WHT = "#ffffff"

START_YEAR = 2023
FVG_MIN_ATR_FRAC = 0.05
OB_LOCAL_WIN = 3
OB_MAX_CONFIRM_BARS = 20
SWEEP_MAX_REJECT_BARS = 3
BREAKER_MAX_WAIT_BARS = 60
ACC_BARS = 8
ACC_RANGE_ATR_MULT = 1.5
MANIP_MAX_BARS = 10

print("[DATA] Loading 4H + 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
df15 = add_indicators(raw["15M"])
IDX4H = df4h.index; N4H = len(df4h)
IDX = df15.index; N = len(df15)
CL4, HI4, LO4, OP4 = (df4h[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR4 = np.where(df4h["atr_14"].values > 0, df4h["atr_14"].values, np.nan)
CL, HI, LO, OP = (df15[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR = np.where(df15["atr_14"].values > 0, df15["atr_14"].values, np.nan)
print(f"  4H: {N4H:,}  15M: {N:,}")


def plot_candles(ax, i0, i1, idxarr, o, h, l, c):
    for i in range(i0, i1 + 1):
        t = idxarr[i]
        color = _GRN if c[i] >= o[i] else _RED
        ax.plot([t, t], [l[i], h[i]], color=color, linewidth=0.9, zorder=2)
        body_lo, body_hi = min(o[i], c[i]), max(o[i], c[i])
        width = (idxarr[1] - idxarr[0]) * 0.6
        ax.add_patch(Rectangle((mdates.date2num(t) - width.total_seconds() / 86400 / 2, body_lo),
                                width.total_seconds() / 86400, max(body_hi - body_lo, 1e-6),
                                facecolor=color, edgecolor=color, zorder=3))


def style_ax(ax, title):
    ax.set_facecolor(_CARD)
    ax.set_title(title, color=_TEXT, fontsize=11)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values(): spine.set_color("#333")
    ax.legend(loc="upper left", fontsize=7, facecolor=_CARD, edgecolor="#333", labelcolor=_TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))


fig, axes = plt.subplots(3, 2, figsize=(20, 16), facecolor=_BG)
fig.suptitle("Esempi reali degli oggetti ICT/SMC rilevati (BTCUSDT 15M)", color=_TEXT, fontsize=15, y=0.995)

# ── 1) Swing High/Low + Liquidity Sweep ────────────────────────────────
pivots = find_pivots(HI, LO, 1, 1)
confirmed_highs, confirmed_lows = [], []
p_idx = 0
sweep_example = None
for i in range(N):
    while p_idx < len(pivots) and pivots[p_idx]["confirm_idx"] == i:
        piv = pivots[p_idx]
        (confirmed_highs if piv["kind"] == 1 else confirmed_lows).append((i, piv["bar_idx"], piv["price"]))
        p_idx += 1
    if confirmed_lows and LO[i] < confirmed_lows[-1][2]:
        level = confirmed_lows[-1][2]; swing_bar = confirmed_lows[-1][1]
        for k in range(0, min(SWEEP_MAX_REJECT_BARS, N - i)):
            if CL[i + k] > level:
                sweep_example = dict(swing_bar=swing_bar, sweep_i=i, confirm_i=i + k, level=level, dir=1)
                break
    if sweep_example and sweep_example["swing_bar"] >= 20: break
    sweep_example = None

ax = axes[0, 0]
ev = sweep_example
i0, i1 = ev["swing_bar"] - 10, ev["confirm_i"] + 15
plot_candles(ax, i0, i1, IDX, OP, HI, LO, CL)
ax.axhline(ev["level"], color=_YEL, linestyle=":", linewidth=1.3, label="Swing Low (liquidity)")
ax.scatter([IDX[ev["sweep_i"]]], [LO[ev["sweep_i"]]], marker="*", s=220, color=_YEL, zorder=5,
           edgecolor="black", label="Sweep (wick)")
ax.scatter([IDX[ev["confirm_i"]]], [CL[ev["confirm_i"]]], marker="^", s=140, color=_GRN, zorder=5,
           edgecolor="black", label="Rejection close")
style_ax(ax, "1) Swing Low + Liquidity Sweep (SSL grab -> rialzista)")

# ── 2) FVG (bullish BISI) ───────────────────────────────────────────────
fvg_example = None
for i in range(1, N - 1):
    atr = ATR[i]
    if atr <= 0 or np.isnan(atr): continue
    if HI[i - 1] < LO[i + 1]:
        fbot, ftop = HI[i - 1], LO[i + 1]
        if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
        fvg_example = dict(i=i, fbot=fbot, ftop=ftop)
        if i > 300: break
ax = axes[0, 1]
ev = fvg_example
i0, i1 = ev["i"] - 12, ev["i"] + 15
plot_candles(ax, i0, i1, IDX, OP, HI, LO, CL)
t0, t1 = IDX[ev["i"] - 1], IDX[ev["i"] + 15]
ax.add_patch(Rectangle((mdates.date2num(IDX[ev["i"]]), ev["fbot"]),
                        mdates.date2num(t1) - mdates.date2num(IDX[ev["i"]]), ev["ftop"] - ev["fbot"],
                        facecolor=_PUR, alpha=0.30, edgecolor=_PUR, linewidth=1, label="FVG (BISI) zone"))
style_ax(ax, "2) Fair Value Gap rialzista (BISI, 3 candele)")

# ── 3) Order Block ───────────────────────────────────────────────────────
ob_example = None
for i in range(OB_LOCAL_WIN, N - OB_MAX_CONFIRM_BARS - 1):
    if CL[i] < OP[i] and CL[i] == CL[i - OB_LOCAL_WIN + 1:i + 1].min():
        for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
            if CL[j] > OP[i]:
                ob_example = dict(origin_i=i, confirm_i=j, lo=min(OP[i], CL[i]), hi=max(OP[i], CL[i]))
                break
    if ob_example and ob_example["origin_i"] > 400: break
ax = axes[1, 0]
ev = ob_example
i0, i1 = ev["origin_i"] - 8, ev["confirm_i"] + 12
plot_candles(ax, i0, i1, IDX, OP, HI, LO, CL)
t1 = IDX[ev["confirm_i"] + 12]
ax.add_patch(Rectangle((mdates.date2num(IDX[ev["origin_i"]]), ev["lo"]),
                        mdates.date2num(t1) - mdates.date2num(IDX[ev["origin_i"]]), ev["hi"] - ev["lo"],
                        facecolor=_ORG, alpha=0.30, edgecolor=_ORG, linewidth=1, label="Order Block zone"))
ax.scatter([IDX[ev["confirm_i"]]], [CL[ev["confirm_i"]]], marker="^", s=140, color=_GRN, zorder=5,
           edgecolor="black", label="Validazione (close > open origine)")
style_ax(ax, "3) Order Block rialzista (validato)")

# ── 4) Breaker Block ──────────────────────────────────────────────────────
breaker_example = None
for i in range(OB_LOCAL_WIN, N - OB_MAX_CONFIRM_BARS - 1):
    if CL[i] < OP[i] and CL[i] == CL[i - OB_LOCAL_WIN + 1:i + 1].min():
        j0 = None
        for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
            if CL[j] > OP[i]: j0 = j; break
        if j0 is None: continue
        lo, hi = min(OP[i], CL[i]), max(OP[i], CL[i])
        invalid_i = None
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, N)):
            if CL[k] < lo: invalid_i = k; break
        if invalid_i is None: continue
        for k in range(invalid_i + 1, min(invalid_i + 1 + BREAKER_MAX_WAIT_BARS, N)):
            if LO[k] <= hi and HI[k] >= lo and CL[k] < lo:
                breaker_example = dict(origin_i=i, confirm_i=j0, invalid_i=invalid_i, retest_i=k, lo=lo, hi=hi)
                break
    if breaker_example: break
ax = axes[1, 1]
ev = breaker_example
i0, i1 = ev["origin_i"] - 5, ev["retest_i"] + 10
plot_candles(ax, i0, i1, IDX, OP, HI, LO, CL)
ax.add_patch(Rectangle((mdates.date2num(IDX[ev["origin_i"]]), ev["lo"]),
                        mdates.date2num(IDX[i1]) - mdates.date2num(IDX[ev["origin_i"]]), ev["hi"] - ev["lo"],
                        facecolor=_RED, alpha=0.22, edgecolor=_RED, linewidth=1, label="OB fallito -> Breaker"))
ax.axvline(IDX[ev["invalid_i"]], color=_RED, linestyle="--", linewidth=1, alpha=0.7, label="Invalidazione")
ax.scatter([IDX[ev["retest_i"]]], [CL[ev["retest_i"]]], marker="v", s=140, color=_RED, zorder=5,
           edgecolor="black", label="Retest + rigetto (breaker ribassista)")
style_ax(ax, "4) Breaker Block (OB rialzista fallito -> resistenza)")

# ── 5) Inversion FVG ──────────────────────────────────────────────────────
ifvg_example = None
for i in range(1, N - 1 - BREAKER_MAX_WAIT_BARS):
    atr = ATR[i]
    if atr <= 0 or np.isnan(atr): continue
    if LO[i - 1] > HI[i + 1]:   # bearish FVG
        ftop, fbot = LO[i - 1], HI[i + 1]
        if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
        j0 = i + 1
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, N)):
            if CL[k] > ftop:
                ifvg_example = dict(form_i=i, confirm_i=j0, break_i=k, fbot=fbot, ftop=ftop)
                break
    if ifvg_example: break
ax = axes[2, 0]
ev = ifvg_example
i0, i1 = ev["form_i"] - 8, ev["break_i"] + 10
plot_candles(ax, i0, i1, IDX, OP, HI, LO, CL)
ax.add_patch(Rectangle((mdates.date2num(IDX[ev["confirm_i"]]), ev["fbot"]),
                        mdates.date2num(IDX[i1]) - mdates.date2num(IDX[ev["confirm_i"]]), ev["ftop"] - ev["fbot"],
                        facecolor=_BLU, alpha=0.25, edgecolor=_BLU, linewidth=1, label="FVG ribassista originale"))
ax.scatter([IDX[ev["break_i"]]], [CL[ev["break_i"]]], marker="^", s=140, color=_GRN, zorder=5,
           edgecolor="black", label="Rottura sopra -> IFVG rialzista")
style_ax(ax, "5) Inversion FVG (bearish FVG rotta -> flip rialzista)")

# ── 6) Power of 3 (AMD) ────────────────────────────────────────────────────
amd_example = None
for i in range(ACC_BARS, N - MANIP_MAX_BARS - 1):
    atr = ATR[i]
    if atr <= 0 or np.isnan(atr): continue
    seg_hi = HI[i - ACC_BARS:i].max(); seg_lo = LO[i - ACC_BARS:i].min()
    if (seg_hi - seg_lo) > ACC_RANGE_ATR_MULT * atr: continue
    for k in range(i, min(i + MANIP_MAX_BARS, N)):
        if LO[k] < seg_lo:
            for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, N)):
                if CL[m] > seg_lo:
                    amd_example = dict(acc_start=i - ACC_BARS, acc_end=i, manip_i=k, dist_i=m,
                                        seg_hi=seg_hi, seg_lo=seg_lo)
                    break
            break
    if amd_example: break
ax = axes[2, 1]
ev = amd_example
i0, i1 = ev["acc_start"] - 3, ev["dist_i"] + 15
plot_candles(ax, i0, i1, IDX, OP, HI, LO, CL)
ax.add_patch(Rectangle((mdates.date2num(IDX[ev["acc_start"]]), ev["seg_lo"]),
                        mdates.date2num(IDX[ev["acc_end"]]) - mdates.date2num(IDX[ev["acc_start"]]),
                        ev["seg_hi"] - ev["seg_lo"],
                        facecolor=_BLU, alpha=0.25, edgecolor=_BLU, linewidth=1, label="Accumulation (range compresso)"))
ax.scatter([IDX[ev["manip_i"]]], [LO[ev["manip_i"]]], marker="*", s=220, color=_RED, zorder=5,
           edgecolor="black", label="Manipulation (sweep sotto range)")
ax.scatter([IDX[ev["dist_i"]]], [CL[ev["dist_i"]]], marker="^", s=140, color=_GRN, zorder=5,
           edgecolor="black", label="Distribution (rigetto -> rialzista)")
style_ax(ax, "6) Power of 3 / AMD (accumulazione -> manipolazione -> distribuzione)")

plt.tight_layout(rect=[0, 0, 1, 0.98])
out1 = Path("reports/ict_objects_examples.png")
plt.savefig(out1, dpi=110, facecolor=_BG)
print(f"[SAVED] {out1}")


# ── 7) Confluenza multi-timeframe: zona demand 4H + trigger 15M ──────────
def detect_order_blocks_full(OPx, CLx, local_win, max_confirm):
    n = len(CLx); zones = []
    for i in range(local_win, n - max_confirm - 1):
        if CLx[i] < OPx[i] and CLx[i] == CLx[i - local_win + 1:i + 1].min():
            for j in range(i + 1, i + 1 + max_confirm):
                if CLx[j] > OPx[i]:
                    zones.append(dict(idx=j, dir=1, lo=min(OPx[i], CLx[i]), hi=max(OPx[i], CLx[i]))); break
    return zones

ob4h = detect_order_blocks_full(OP4, CL4, OB_LOCAL_WIN, 20)
combo_example = None
for z in ob4h:
    j0 = z["idx"]
    invalid_4h = None
    for k in range(j0, min(j0 + 60, N4H)):
        if CL4[k] < z["lo"]: invalid_4h = k; break
    end_4h = invalid_4h if invalid_4h is not None else min(N4H - 1, j0 + 60)
    start_ts, end_ts = IDX4H[j0], IDX4H[end_4h]
    start15 = np.searchsorted(IDX.values, np.datetime64(start_ts), side="left")
    end15 = np.searchsorted(IDX.values, np.datetime64(end_ts), side="left")
    if end15 - start15 < 20 or end15 >= N - 5: continue
    # cerca un semplice sweep rialzista 15m dentro la zona
    seg_lo15 = LO[start15:end15]
    found = None
    for ii in range(start15 + 5, end15):
        if LO[ii] <= z["lo"] + (z["hi"] - z["lo"]) * 0.5 and CL[ii] > z["lo"]:
            found = ii; break
    if found is None: continue
    combo_example = dict(zone_4h=z, start15=start15, end15=end15, trigger_i=found)
    if j0 > 300: break

if combo_example:
    fig2, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor=_BG)
    ev = combo_example; z = ev["zone_4h"]
    i0, i1 = max(ev["start15"] - 10, 0), min(ev["trigger_i"] + 20, N - 1)
    plot_candles(ax, i0, i1, IDX, OP, HI, LO, CL)
    ax.add_patch(Rectangle((mdates.date2num(IDX[ev["start15"]]), z["lo"]),
                            mdates.date2num(IDX[i1]) - mdates.date2num(IDX[ev["start15"]]), z["hi"] - z["lo"],
                            facecolor=_ORG, alpha=0.22, edgecolor=_ORG, linewidth=1.2,
                            label="Zona Demand 4H attiva (Order Block)"))
    ax.scatter([IDX[ev["trigger_i"]]], [CL[ev["trigger_i"]]], marker="^", s=180, color=_WHT, zorder=6,
               edgecolor="black", label="Trigger 15M dentro la zona (confluenza)")
    style_ax(ax, "7) Confluenza multi-timeframe: zona Demand 4H + trigger 15M")
    plt.tight_layout()
    out2 = Path("reports/ict_confluence_example.png")
    plt.savefig(out2, dpi=110, facecolor=_BG)
    print(f"[SAVED] {out2}")
else:
    print("[WARN] Nessun esempio di confluenza trovato entro i primi 300 OB 4H")
