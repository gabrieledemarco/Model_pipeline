#!/usr/bin/env python3
"""
create_htf_ltf_confluence_report.py
======================================
Confluenza multi-timeframe GENUINA (diversa dal "COMBO" di
`ict_objects_predictive.md`, che incatenava oggetti sullo STESSO
timeframe): una ZONA 4H (demand = Order Block o FVG rialzista; supply =
ribassista) è ATTIVA (non ancora invalidata) mentre un TRIGGER si forma
sul 15M — esattamente l'esempio dell'utente ("demand 4h, po3, breaker
block etc su 15min").

Per ogni tipo di trigger 15M (Sweep, FVG, Order Block, Breaker Block,
Inversion FVG, Power of 3), gli eventi vengono divisi in due gruppi:
  CONFLUENZA : il trigger si forma mentre il prezzo è dentro una zona 4H
               attiva della STESSA direzione (demand per un trigger
               rialzista, supply per uno ribassista)
  STANDALONE : il trigger si forma senza alcuna zona 4H attiva coerente

Stessa metodologia di event-study di `ict_objects_predictive.md`:
forward return nella direzione implicita, Welch t-test vs controllo,
4 orizzonti (1/5/10/20 barre 15M). Se la narrativa ICT "il trigger vale
di più dentro una zona HTF" è corretta, CONFLUENZA dovrebbe mostrare
effetti più forti/più significativi di STANDALONE — testato direttamente,
non assunto.

Zone 4H attive: Order Block e FVG (gli unici due oggetti con un'AREA di
prezzo persistente, non un evento puntuale) — stessa logica di
invalidazione già usata in `detect_breakers`/`detect_ifvgs`.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import find_pivots

SEP = "═" * 78
START_YEAR = 2020
HORIZONS = [1, 5, 10, 20]
FVG_MIN_ATR_FRAC = 0.05
OB_LOCAL_WIN = 3
OB_MAX_CONFIRM_BARS = 20
SWEEP_MAX_REJECT_BARS = 3
ZONE_MAX_AGE_BARS_4H = 60          # staleness massima di una zona 4H mai invalidata (~10gg)
ACC_BARS = 8
ACC_RANGE_ATR_MULT = 1.5
MANIP_MAX_BARS = 10
BREAKER_MAX_WAIT_BARS = 60

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Confluenza multi-timeframe: zona 4H (demand/supply) attiva + trigger 15M")
w(SEP)
w("\nDemand 4H (OB/FVG rialzista) o Supply 4H (OB/FVG ribassista) ATTIVA")
w("mentre un trigger (sweep/FVG/OB/breaker/IFVG/PO3) si forma sul 15M —")
w("confronto diretto: la confluenza migliora il potere predittivo o no?")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 4H + 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
df15 = add_indicators(raw["15M"])
IDX4H = df4h.index; N4H = len(df4h)
IDX15 = df15.index; N15 = len(df15)
print(f"  4H:  {N4H:,} bars   15M: {N15:,} bars  (loaded in {time.time()-t0:.0f}s)")

CL4, HI4, LO4, OP4 = (df4h[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR4 = np.where(df4h["atr_14"].values > 0, df4h["atr_14"].values, np.nan)
CL, HI, LO, OP = (df15[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR = np.where(df15["atr_14"].values > 0, df15["atr_14"].values, np.nan)

IDX15_vals = IDX15.values
def bar15_at_or_after(ts):
    pos = np.searchsorted(IDX15_vals, np.datetime64(ts), side="left")
    return pos if pos < N15 else None


# ── Detectors (identici a create_ict_objects_predictive_report.py) ───────
def detect_fvgs(HIx, LOx, ATRx):
    n = len(HIx); events = []; zones = []
    for i in range(1, n - 1):
        atr = ATRx[i]
        if atr <= 0 or np.isnan(atr): continue
        if HIx[i - 1] < LOx[i + 1]:
            fbot, ftop = HIx[i - 1], LOx[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=1)); zones.append(dict(idx=i + 1, dir=1, lo=fbot, hi=ftop))
        elif LOx[i - 1] > HIx[i + 1]:
            ftop, fbot = LOx[i - 1], HIx[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=-1)); zones.append(dict(idx=i + 1, dir=-1, lo=fbot, hi=ftop))
    return events, zones


def detect_order_blocks(OPx, CLx):
    n = len(CLx); events = []; zones = []
    for i in range(OB_LOCAL_WIN, n - OB_MAX_CONFIRM_BARS - 1):
        if CLx[i] < OPx[i]:
            if CLx[i] == CLx[i - OB_LOCAL_WIN + 1:i + 1].min():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CLx[j] > OPx[i]:
                        events.append(dict(idx=j, dir=1))
                        zones.append(dict(idx=j, dir=1, lo=min(OPx[i], CLx[i]), hi=max(OPx[i], CLx[i])))
                        break
        elif CLx[i] > OPx[i]:
            if CLx[i] == CLx[i - OB_LOCAL_WIN + 1:i + 1].max():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CLx[j] < OPx[i]:
                        events.append(dict(idx=j, dir=-1))
                        zones.append(dict(idx=j, dir=-1, lo=min(OPx[i], CLx[i]), hi=max(OPx[i], CLx[i])))
                        break
    return events, zones


def detect_sweeps(HIx, LOx, CLx, pivots):
    n = len(CLx); events = []
    confirmed_highs, confirmed_lows = [], []; p_idx = 0
    for i in range(n):
        while p_idx < len(pivots) and pivots[p_idx]["confirm_idx"] == i:
            piv = pivots[p_idx]
            (confirmed_highs if piv["kind"] == 1 else confirmed_lows).append(piv["price"])
            p_idx += 1
        if confirmed_highs and HIx[i] > confirmed_highs[-1]:
            level = confirmed_highs[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CLx[i + k] < level: events.append(dict(idx=i + k, dir=-1)); break
        if confirmed_lows and LOx[i] < confirmed_lows[-1]:
            level = confirmed_lows[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CLx[i + k] > level: events.append(dict(idx=i + k, dir=1)); break
    return events


def detect_breakers(zones_ob, HIx, LOx, CLx):
    n = len(CLx); events = []
    for z in zones_ob:
        j0 = z["idx"]; lo, hi = z["lo"], z["hi"]
        invalid_i = None
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1 and CLx[k] < lo: invalid_i = k; break
            if z["dir"] == -1 and CLx[k] > hi: invalid_i = k; break
        if invalid_i is None: continue
        for k in range(invalid_i + 1, min(invalid_i + 1 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1 and LOx[k] <= hi and HIx[k] >= lo and CLx[k] < lo:
                events.append(dict(idx=k, dir=-1)); break
            if z["dir"] == -1 and HIx[k] >= lo and LOx[k] <= hi and CLx[k] > hi:
                events.append(dict(idx=k, dir=1)); break
    return events


def detect_ifvgs(zones_fvg, CLx):
    n = len(CLx); events = []
    for z in zones_fvg:
        j0 = z["idx"]; lo, hi, d = z["lo"], z["hi"], z["dir"]
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if d == 1 and CLx[k] < lo: events.append(dict(idx=k, dir=-1)); break
            if d == -1 and CLx[k] > hi: events.append(dict(idx=k, dir=1)); break
    return events


def detect_amd(HIx, LOx, CLx, ATRx):
    n = len(CLx); events = []
    for i in range(ACC_BARS, n - MANIP_MAX_BARS - 1):
        atr = ATRx[i]
        if atr <= 0 or np.isnan(atr): continue
        seg_hi = HIx[i - ACC_BARS:i].max(); seg_lo = LOx[i - ACC_BARS:i].min()
        if (seg_hi - seg_lo) > ACC_RANGE_ATR_MULT * atr: continue
        for k in range(i, min(i + MANIP_MAX_BARS, n)):
            if HIx[k] > seg_hi:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CLx[m] < seg_hi: events.append(dict(idx=m, dir=-1)); break
                break
            if LOx[k] < seg_lo:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CLx[m] > seg_lo: events.append(dict(idx=m, dir=1)); break
                break
    return events


# ── Zone 4H (demand/supply) con finestra attiva, mappata su barre 15M ────
print("[HTF] Detecting 4H demand/supply zones (OB + FVG) …")
_, ob_zones_4h = detect_order_blocks(OP4, CL4)
_, fvg_zones_4h = detect_fvgs(HI4, LO4, ATR4)
raw_zones_4h = ob_zones_4h + fvg_zones_4h

htf_zones_15 = []   # dict(dir, start15, end15)
for z in raw_zones_4h:
    j0 = z["idx"]
    if j0 >= N4H: continue
    invalid_i = None
    for k in range(j0, min(j0 + ZONE_MAX_AGE_BARS_4H, N4H)):
        if z["dir"] == 1 and CL4[k] < z["lo"]: invalid_i = k; break
        if z["dir"] == -1 and CL4[k] > z["hi"]: invalid_i = k; break
    end_4h = invalid_i if invalid_i is not None else min(N4H - 1, j0 + ZONE_MAX_AGE_BARS_4H)
    start15 = bar15_at_or_after(IDX4H[j0])
    end15 = bar15_at_or_after(IDX4H[end_4h])
    if start15 is None: continue
    if end15 is None: end15 = N15
    if end15 <= start15: continue
    htf_zones_15.append(dict(dir=z["dir"], start=start15, end=end15))
print(f"  {len(htf_zones_15)} zone 4H (demand+supply) mappate su 15M")

htf_by_dir = {1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == 1]),
              -1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == -1])}
starts_1 = np.array([s for s, e in htf_by_dir[1]]); ends_1 = np.array([e for s, e in htf_by_dir[1]])
starts_m1 = np.array([s for s, e in htf_by_dir[-1]]); ends_m1 = np.array([e for s, e in htf_by_dir[-1]])


def in_active_zone(idx, d):
    starts, ends = (starts_1, ends_1) if d == 1 else (starts_m1, ends_m1)
    if len(starts) == 0: return False
    pos = np.searchsorted(starts, idx, side="right") - 1
    while pos >= 0:
        if starts[pos] <= idx <= ends[pos]:
            return True
        if idx - starts[pos] > 4 * ZONE_MAX_AGE_BARS_4H:
            break
        pos -= 1
    return False


# ── Trigger 15M ────────────────────────────────────────────────────────
print("[LTF] Detecting 15M triggers …")
pivots15 = find_pivots(HI, LO, 1, 1)
sweeps = detect_sweeps(HI, LO, CL, pivots15)
fvgs, _ = detect_fvgs(HI, LO, ATR)
obs, ob_zones15 = detect_order_blocks(OP, CL)
_, fvg_zones15 = detect_fvgs(HI, LO, ATR)
breakers = detect_breakers(ob_zones15, HI, LO, CL)
ifvgs = detect_ifvgs(fvg_zones15, CL)
amd = detect_amd(HI, LO, CL, ATR)

TRIGGERS = {"Sweep": sweeps, "FVG": fvgs, "Order Block": obs,
            "Breaker Block": breakers, "Inversion FVG": ifvgs, "Power of 3 (AMD)": amd}

all_idx_pool = np.arange(50, N15 - max(HORIZONS) - 1)


def eval_group(events, label_extra=""):
    if len(events) < 20: return None
    dirs = np.array([e["dir"] for e in events]); idxs = np.array([e["idx"] for e in events])
    rows = []
    for h in HORIZONS:
        valid = idxs + h < N15
        ev_ret = dirs[valid] * (CL[idxs[valid] + h] - CL[idxs[valid]]) / CL[idxs[valid]]
        ctrl_idx = all_idx_pool[all_idx_pool + h < N15]
        rng = np.random.default_rng(42 + h)
        ctrl_dir = rng.choice([-1, 1], size=len(ctrl_idx))
        ctrl_ret = ctrl_dir * (CL[ctrl_idx + h] - CL[ctrl_idx]) / CL[ctrl_idx]
        if len(ev_ret) < 10: continue
        tstat, pval = st.ttest_ind(ev_ret, ctrl_ret, equal_var=False)
        rows.append(dict(h=h, n=len(ev_ret), mean=ev_ret.mean() * 100, pval=pval))
    return rows


results = {}
for label, evs in TRIGGERS.items():
    conf = [e for e in evs if in_active_zone(e["idx"], e["dir"])]
    stand = [e for e in evs if not in_active_zone(e["idx"], e["dir"])]
    results[label] = dict(n_all=len(evs), n_conf=len(conf), n_stand=len(stand),
                           all_stats=eval_group(evs), conf_stats=eval_group(conf), stand_stats=eval_group(stand))
    print(f"  {label:>20}: totale={len(evs)}  confluenza={len(conf)}  standalone={len(stand)}")

# ── Report ──────────────────────────────────────────────────────────────
w(f"\n{SEP}")
w("Frequenza: confluenza (dentro zona 4H attiva) vs standalone")
w(SEP)
w(f"\n  {'Trigger 15M':<20}  {'n totale':>10}  {'n confluenza':>13}  {'n standalone':>13}  {'% confluenza':>13}")
for label, r in results.items():
    pct = r["n_conf"] / r["n_all"] * 100 if r["n_all"] else 0
    w(f"  {label:<20}  {r['n_all']:>10,}  {r['n_conf']:>13,}  {r['n_stand']:>13,}  {pct:>12.1f}%")

for h_show in HORIZONS:
    w(f"\n{SEP}")
    w(f"Potere predittivo a {h_show} barre — CONFLUENZA vs STANDALONE vs TUTTI (mean %, * = p<0.05)")
    w(SEP)
    w(f"\n  {'Trigger 15M':<20}  {'TUTTI':>16}  {'CONFLUENZA':>16}  {'STANDALONE':>16}")
    for label, r in results.items():
        def cell(stats):
            if not stats: return "n/a"
            m = [s for s in stats if s["h"] == h_show]
            if not m: return "n/a"
            s = m[0]
            return f"{s['mean']:+.3f}%{'*' if s['pval']<0.05 else ' '} (n={s['n']})"
        w(f"  {label:<20}  {cell(r['all_stats']):>16}  {cell(r['conf_stats']):>16}  {cell(r['stand_stats']):>16}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/htf_ltf_confluence.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Confluenza multi-timeframe: zona 4H (demand/supply) attiva + trigger 15M\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
