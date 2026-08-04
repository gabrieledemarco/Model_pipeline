#!/usr/bin/env python3
"""
create_ict_objects_predictive_report.py
==========================================
Riconoscimento + potere predittivo dei 7 oggetti ICT/SMC condivisi
dall'utente (Swing High/Low, Liquidity BSL/SSL, Fair Value Gap BISI/SIBI,
Order Block, Breaker Block, Inversion FVG, Power of 3/AMD), su 4
timeframe (1D, 4H, 1H, 15M — 1W escluso per frequenza troppo bassa per
statistiche robuste, 1M escluso per costo computazionale visto lo studio
precedente su 6 timeframe), + test del segnale COMBINATO
(sweep -> Order Block -> FVG in sequenza) vs ciascun oggetto da solo.

Metodologia (event study diagnostico, non backtest — nessuna fee/sizing,
stessa metodologia già usata per `large_order_impact.md`): ogni oggetto
ha una direzione implicita (rialzista/ribassista); si misura il forward
return NELLA DIREZIONE IMPLICITA a 4 orizzonti (1,5,10,20 barre), con
Welch t-test contro un gruppo di controllo (tutte le altre barre) per
verificare se l'effetto è statisticamente reale o rumore.

── Definizioni (causali, replicano esattamente le slide condivise) ──────
  SWING HIGH/LOW: fractal a 3 candele (find_pivots, left=right=1) —
    "il punto medio di 3 barre più alto/basso delle barre adiacenti".
  LIQUIDITY SWEEP (BSL/SSL grab): la barra supera (wick) un precedente
    swing high/low confermato MA chiude di nuovo dentro il range
    precedente (rigetto, non breakout) — direzione implicita: CONTRARIAN
    al lato spazzato (sweep di un high -> ribassista; sweep di un low ->
    rialzista), per l'ipotesi ICT "grab liquidità poi inversione".
  FVG (BISI/SIBI): gap standard a 3 candele (HI[i-1]<LO[i+1] rialzista,
    LO[i-1]>HI[i+1] ribassista), filtro dimensione minima 0.05xATR.
  ORDER BLOCK: la candela down-close con chiusura minima locale (tra le
    ultime 3) è l'origine; OB validato quando una barra successiva
    CHIUDE SOPRA L'OPEN di quella candela (regola esatta della slide);
    speculare per l'OB ribassista (candela up-close con chiusura massima
    locale, validato da una chiusura sotto il suo open).
  BREAKER BLOCK: un OB rialzista invalidato (chiusura sotto il suo
    minimo) che viene poi ri-testato dall'alto e RIGETTATO (chiusura di
    nuovo sotto la zona) -> breaker ribassista; speculare per l'OB
    ribassista fallito -> breaker rialzista.
  INVERSION FVG: un FVG rotto per intero (chiusura oltre il bordo
    lontano, non solo mitigato) -> IFVG con direzione OPPOSTA
    all'originale (bearish FVG rotto sopra -> IFVG rialzista; bullish
    FVG rotto sotto -> IFVG ribassista) — esattamente la regola della
    slide.
  POWER OF 3 (AMD): accumulazione = range compresso (<=1.5xATR) su
    ACC_BARS barre; manipolazione = sweep oltre quel range entro
    MANIP_MAX_BARS barre; distribuzione = si assume iniziare al momento
    della manipolazione/sweep stesso (il punto di segnale ICT), direzione
    = contrarian al lato del sweep (stessa logica del liquidity sweep,
    ma condizionata a un'accumulazione precedente genuina).

── Segnale COMBINATO ────────────────────────────────────────────────────
  Liquidity Sweep -> (entro COMBO_WINDOW barre) Order Block nella stessa
  direzione -> (entro COMBO_WINDOW barre) FVG nella stessa direzione.
  Confronta il potere predittivo del punto di conferma finale (l'FVG)
  contro ciascun componente preso da solo.
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
TIMEFRAMES = ["1D", "4H", "1H", "15M"]
HORIZONS = [1, 5, 10, 20]
FVG_MIN_ATR_FRAC = 0.05
OB_LOCAL_WIN = 3
OB_MAX_CONFIRM_BARS = 20
SWEEP_MAX_REJECT_BARS = 3         # barre entro cui il prezzo deve richiudere dentro il range
BREAKER_MAX_WAIT_BARS = 60
ACC_BARS = 8
ACC_RANGE_ATR_MULT = 1.5
MANIP_MAX_BARS = 10
COMBO_WINDOW = 15

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("ICT/SMC Objects — Riconoscimento, chiarezza per timeframe, potere predittivo")
w(SEP)
w("\n7 oggetti dalle slide condivise + segnale combinato (sweep->OB->FVG).")
w("Event study: forward return nella direzione implicita, Welch t-test vs")
w("controllo, 4 orizzonti (1/5/10/20 barre), 4 timeframe (1D/4H/1H/15M).")

t0 = time.time()
print("\n[DATA] Loading 1D, 4H, 1H, 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
dfs = {tf: add_indicators(raw[tf]) for tf in TIMEFRAMES}
for tf in TIMEFRAMES:
    print(f"  {tf:>4}: {len(dfs[tf]):,} bars")
print(f"  (loaded in {time.time()-t0:.0f}s)")


# ── Detectors ───────────────────────────────────────────────────────────
def get_swings(HI, LO):
    return find_pivots(HI, LO, 1, 1)


def detect_sweeps(HI, LO, CL, pivots):
    """Sweep di uno swing high/low confermato: wick oltre, chiusura rientra entro N barre."""
    n = len(CL)
    events = []
    confirmed_highs, confirmed_lows = [], []
    p_idx = 0
    pivots_sorted = pivots
    for i in range(n):
        while p_idx < len(pivots_sorted) and pivots_sorted[p_idx]["confirm_idx"] == i:
            piv = pivots_sorted[p_idx]
            if piv["kind"] == 1: confirmed_highs.append(piv["price"])
            else: confirmed_lows.append(piv["price"])
            p_idx += 1
        if confirmed_highs and HI[i] > confirmed_highs[-1]:
            level = confirmed_highs[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CL[i + k] < level:
                    events.append(dict(idx=i + k, dir=-1)); break
        if confirmed_lows and LO[i] < confirmed_lows[-1]:
            level = confirmed_lows[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CL[i + k] > level:
                    events.append(dict(idx=i + k, dir=1)); break
    return events


def detect_fvgs(HI, LO, ATR):
    n = len(HI)
    events = []
    zones = []  # (idx, dir, top, bot) per riuso in IFVG
    for i in range(1, n - 1):
        atr = ATR[i]
        if atr <= 0 or np.isnan(atr): continue
        if HI[i - 1] < LO[i + 1]:
            fbot, ftop = HI[i - 1], LO[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=1))
            zones.append(dict(idx=i + 1, dir=1, top=ftop, bot=fbot))
        elif LO[i - 1] > HI[i + 1]:
            ftop, fbot = LO[i - 1], HI[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=-1))
            zones.append(dict(idx=i + 1, dir=-1, top=ftop, bot=fbot))
    return events, zones


def detect_order_blocks(OP, CL):
    n = len(CL)
    events = []
    zones = []
    for i in range(OB_LOCAL_WIN, n - OB_MAX_CONFIRM_BARS - 1):
        if CL[i] < OP[i]:
            recent = CL[i - OB_LOCAL_WIN + 1:i + 1]
            if CL[i] == recent.min():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CL[j] > OP[i]:
                        events.append(dict(idx=j, dir=1))
                        zones.append(dict(idx=j, dir=1, lo=min(OP[i], CL[i]), hi=max(OP[i], CL[i])))
                        break
        elif CL[i] > OP[i]:
            recent = CL[i - OB_LOCAL_WIN + 1:i + 1]
            if CL[i] == recent.max():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CL[j] < OP[i]:
                        events.append(dict(idx=j, dir=-1))
                        zones.append(dict(idx=j, dir=-1, lo=min(OP[i], CL[i]), hi=max(OP[i], CL[i])))
                        break
    return events, zones


def detect_breakers(zones_ob, HI, LO, CL):
    n = len(CL)
    events = []
    for z in zones_ob:
        j0 = z["idx"]; lo, hi = z["lo"], z["hi"]
        invalid_i = None
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1 and CL[k] < lo:
                invalid_i = k; break
            if z["dir"] == -1 and CL[k] > hi:
                invalid_i = k; break
        if invalid_i is None: continue
        for k in range(invalid_i + 1, min(invalid_i + 1 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1:
                if LO[k] <= hi and HI[k] >= lo and CL[k] < lo:
                    events.append(dict(idx=k, dir=-1)); break
            else:
                if HI[k] >= lo and LO[k] <= hi and CL[k] > hi:
                    events.append(dict(idx=k, dir=1)); break
    return events


def detect_ifvgs(zones_fvg, CL):
    n = len(CL)
    events = []
    for z in zones_fvg:
        j0 = z["idx"]; top, bot, d = z["top"], z["bot"], z["dir"]
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if d == 1 and CL[k] < bot:
                events.append(dict(idx=k, dir=-1)); break
            if d == -1 and CL[k] > top:
                events.append(dict(idx=k, dir=1)); break
    return events


def detect_amd(HI, LO, CL, ATR):
    n = len(CL)
    events = []
    for i in range(ACC_BARS, n - MANIP_MAX_BARS - 1):
        atr = ATR[i]
        if atr <= 0 or np.isnan(atr): continue
        seg_hi = HI[i - ACC_BARS:i].max(); seg_lo = LO[i - ACC_BARS:i].min()
        if (seg_hi - seg_lo) > ACC_RANGE_ATR_MULT * atr: continue
        for k in range(i, min(i + MANIP_MAX_BARS, n)):
            if HI[k] > seg_hi:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CL[m] < seg_hi:
                        events.append(dict(idx=m, dir=-1)); break
                break
            if LO[k] < seg_lo:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CL[m] > seg_lo:
                        events.append(dict(idx=m, dir=1)); break
                break
    return events


def detect_combo(sweep_events, ob_events, fvg_events):
    ob_by_dir = {1: sorted([e["idx"] for e in ob_events if e["dir"] == 1]),
                 -1: sorted([e["idx"] for e in ob_events if e["dir"] == -1])}
    fvg_by_dir = {1: sorted([e["idx"] for e in fvg_events if e["dir"] == 1]),
                  -1: sorted([e["idx"] for e in fvg_events if e["dir"] == -1])}
    events = []
    for sw in sweep_events:
        i0, d = sw["idx"], sw["dir"]
        obs = [x for x in ob_by_dir[d] if i0 < x <= i0 + COMBO_WINDOW]
        if not obs: continue
        i1 = obs[0]
        fvgs = [x for x in fvg_by_dir[d] if i1 < x <= i1 + COMBO_WINDOW]
        if not fvgs: continue
        events.append(dict(idx=fvgs[0], dir=d))
    return events


# ── Predictive power event study ───────────────────────────────────────
def eval_predictive_power(events, CL, all_idx_pool):
    if len(events) < 20:
        return None
    n = len(CL)
    dirs = np.array([e["dir"] for e in events])
    idxs = np.array([e["idx"] for e in events])
    rows = []
    for h in HORIZONS:
        valid = idxs + h < n
        ev_ret = dirs[valid] * (CL[idxs[valid] + h] - CL[idxs[valid]]) / CL[idxs[valid]]
        ctrl_idx = all_idx_pool[all_idx_pool + h < n]
        ctrl_dir = np.random.default_rng(42 + h).choice([-1, 1], size=len(ctrl_idx))
        ctrl_ret = ctrl_dir * (CL[ctrl_idx + h] - CL[ctrl_idx]) / CL[ctrl_idx]
        if len(ev_ret) < 10: continue
        tstat, pval = st.ttest_ind(ev_ret, ctrl_ret, equal_var=False)
        rows.append(dict(h=h, n=len(ev_ret), mean=ev_ret.mean() * 100, tstat=tstat, pval=pval))
    return rows


OBJECT_LABELS = ["Swing H/L (sweep)", "FVG (BISI/SIBI)", "Order Block", "Breaker Block",
                  "Inversion FVG", "Power of 3 (AMD)", "COMBO sweep->OB->FVG"]

all_results = {}
for tf in TIMEFRAMES:
    d = dfs[tf]
    CL = d["close"].values.astype(float); HI = d["high"].values.astype(float)
    LO = d["low"].values.astype(float); OP = d["open"].values.astype(float)
    ATR = np.where(d["atr_14"].values > 0, d["atr_14"].values, np.nan)
    n = len(d)
    all_idx_pool = np.arange(50, n - max(HORIZONS) - 1)

    pivots = get_swings(HI, LO)
    sweeps = detect_sweeps(HI, LO, CL, pivots)
    fvgs, fvg_zones = detect_fvgs(HI, LO, ATR)
    obs, ob_zones = detect_order_blocks(OP, CL)
    breakers = detect_breakers(ob_zones, HI, LO, CL)
    ifvgs = detect_ifvgs(fvg_zones, CL)
    amd = detect_amd(HI, LO, CL, ATR)
    combo = detect_combo(sweeps, obs, fvgs)

    objs = {"Swing H/L (sweep)": sweeps, "FVG (BISI/SIBI)": fvgs, "Order Block": obs,
            "Breaker Block": breakers, "Inversion FVG": ifvgs, "Power of 3 (AMD)": amd,
            "COMBO sweep->OB->FVG": combo}
    print(f"  {tf:>4}: " + "  ".join(f"{k}={len(v)}" for k, v in objs.items()))

    tf_results = {}
    for label, evs in objs.items():
        tf_results[label] = dict(n=len(evs), stats=eval_predictive_power(evs, CL, all_idx_pool))
    all_results[tf] = tf_results

# ── Report: frequenza ──────────────────────────────────────────────────
w(f"\n{SEP}")
w("FREQUENZA (eventi totali nel campione, per timeframe)")
w(SEP)
w(f"\n  {'Oggetto':<22}" + "".join(f"{tf:>10}" for tf in TIMEFRAMES))
for label in OBJECT_LABELS:
    row = f"  {label:<22}"
    for tf in TIMEFRAMES:
        row += f"{all_results[tf][label]['n']:>10,}"
    w(row)

# ── Report: potere predittivo, orizzonte breve (1 barra) e medio (20 barre) ──
for h_show in [1, 5, 10, 20]:
    w(f"\n{SEP}")
    w(f"POTERE PREDITTIVO — orizzonte {h_show} barre (mean return % nella direzione implicita, * = p<0.05)")
    w(SEP)
    w(f"\n  {'Oggetto':<22}" + "".join(f"{tf:>16}" for tf in TIMEFRAMES))
    for label in OBJECT_LABELS:
        row = f"  {label:<22}"
        for tf in TIMEFRAMES:
            stats = all_results[tf][label]["stats"]
            cell = "n/a"
            if stats:
                match = [s for s in stats if s["h"] == h_show]
                if match:
                    s = match[0]
                    flag = "*" if s["pval"] < 0.05 else " "
                    cell = f"{s['mean']:+.3f}%{flag}"
            row += f"{cell:>16}"
        w(row)

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/ict_objects_predictive.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# ICT/SMC Objects — Riconoscimento, chiarezza per timeframe, potere predittivo\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
