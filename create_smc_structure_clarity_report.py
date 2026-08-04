#!/usr/bin/env python3
"""
create_smc_structure_clarity_report.py
=========================================
Studio empirico (diagnostico, non backtest): su quale timeframe le
strutture SMC (Order Block, Fair Value Gap, Break of Structure, Change of
Character, Equal Highs/Lows = liquidità) sono identificabili più
facilmente e con meno rumore? Confronto quantitativo su tutti i 6
timeframe disponibili in pipeline: 1W, 1D, 4H, 1H, 15M, 1M.

Metriche di "facilità di identificazione" per ciascuna struttura (nessuna
è soggettiva/estetica — tutte causali, calcolabili algoritmicamente):

  BOS / CHoCH (rottura di struttura, via pivot fractali left=right=3):
    - overshoot medio (ATR) alla rottura: quanto DECISAMENTE il prezzo
      chiude oltre il pivot — una rottura che chiude appena oltre (basso
      overshoot) è ambigua/rumorosa, una che chiude ben oltre è netta.
    - failure rate: frazione di rotture che vengono invalidate (il
      prezzo richiude dall'altra parte) entro FAIL_CHECK_BARS barre —
      il proxy più diretto di "quanto ti puoi fidare della rottura".

  FVG (gap a 3 candele, definizione ICT standard):
    - frequenza/anno
    - size media (ATR): gap più larghi relativo al rumore locale sono
      più facili da individuare visivamente e algoritmicamente
    - frazione mitigata entro MITIGATION_CHECK_BARS barre (gap che si
      richiudono subito sono più "rumorosi"/meno significativi)

  Order Block (ultima candela opposta prima di un impulso >= 1.5×ATR):
    - frequenza/anno
    - dimensione media dell'impulso che lo definisce (ATR) — impulsi più
      forti = OB più netti, meno ambigui

  Equal Highs/Lows (pool di liquidità, $$$ nell'immagine): coppie di
  pivot dello stesso tipo entro TOLERANCE_ATR di prezzo l'uno dall'altro
  — frequenza/anno.

1M limitato agli ultimi 6 mesi (per tempo di calcolo; a 1 minuto un
campione di 6 mesi è già ~260k barre, sufficiente per il confronto).
Tutte le altre serie usano la storia completa 2020-2026.
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
from src.strategy.mtf_swing import find_pivots

SEP = "═" * 78
START_YEAR = 2020
LEFT = RIGHT = 3                  # finestra fractale pivot, fissa per confronto equo
FAIL_CHECK_BARS = 10
FVG_MIN_ATR_FRAC = 0.05
MITIGATION_CHECK_BARS = 20
OB_SWING_ATR_MULT = 1.5
OB_MAX_SWING_BARS = 8
EQ_TOLERANCE_ATR = 0.15
EQ_LOOKBACK_PIVOTS = 20
M1_MONTHS_BACK = 6

TIMEFRAMES = ["1W", "1D", "4H", "1H", "15M", "1M"]
BARS_PER_YEAR = {"1W": 52, "1D": 365, "4H": 6 * 365, "1H": 24 * 365, "15M": 96 * 365, "1M": 1440 * 365}

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("SMC Structure Clarity — confronto multi-timeframe (1W/1D/4H/1H/15M/1M)")
w(SEP)
w("\nDove le strutture OB/FVG/BOS-CHoCH/liquidità (EQH-EQL) sono più facili")
w("da identificare: overshoot/dimensione relativa all'ATR, failure/mitigation")
w("rate, frequenza — tutte causali, nessuna metrica soggettiva.")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading all timeframes …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=True, fetch_flow=False)
dfs = {}
for tf in TIMEFRAMES:
    d = add_indicators(raw[tf])
    if tf == "1M":
        cutoff = d.index[-1] - pd.DateOffset(months=M1_MONTHS_BACK)
        d = d[d.index >= cutoff]
    dfs[tf] = d
    print(f"  {tf:>4}: {len(d):,} bars  ({d.index[0].date()} → {d.index[-1].date()})")
print(f"  (loaded in {time.time()-t0:.0f}s)")


# ── Detector helpers (self-contenuti, riusano find_pivots) ────────────────
def detect_structure_breaks(CL, HI, LO, ATR):
    pivots = find_pivots(HI, LO, LEFT, RIGHT)
    n = len(CL)
    events = []
    highs_seen, lows_seen = [], []
    state = 0
    p_idx = 0
    n_piv = len(pivots)
    last_ph = last_pl = None
    ph_broken = pl_broken = True
    for i in range(n):
        while p_idx < n_piv and pivots[p_idx]["confirm_idx"] == i:
            piv = pivots[p_idx]
            if piv["kind"] == 1:
                highs_seen.append(piv["price"]); last_ph = piv["price"]; ph_broken = False
            else:
                lows_seen.append(piv["price"]); last_pl = piv["price"]; pl_broken = False
            p_idx += 1
            if len(highs_seen) >= 2 and len(lows_seen) >= 2:
                hh = highs_seen[-1] > highs_seen[-2]; hl = lows_seen[-1] > lows_seen[-2]
                lh = highs_seen[-1] < highs_seen[-2]; ll = lows_seen[-1] < lows_seen[-2]
                if hh and hl: state = 1
                elif lh and ll: state = -1
        atr = ATR[i]
        if atr <= 0 or np.isnan(atr):
            continue
        if last_ph is not None and not ph_broken and CL[i] > last_ph:
            overshoot = (CL[i] - last_ph) / atr
            kind = "BOS" if state == 1 else "CHoCH"
            failed = False
            for k in range(1, FAIL_CHECK_BARS + 1):
                if i + k >= n: break
                if CL[i + k] < last_ph: failed = True; break
            events.append(dict(idx=i, kind=kind, overshoot=overshoot, failed=failed))
            ph_broken = True
        if last_pl is not None and not pl_broken and CL[i] < last_pl:
            overshoot = (last_pl - CL[i]) / atr
            kind = "BOS" if state == -1 else "CHoCH"
            failed = False
            for k in range(1, FAIL_CHECK_BARS + 1):
                if i + k >= n: break
                if CL[i + k] > last_pl: failed = True; break
            events.append(dict(idx=i, kind=kind, overshoot=overshoot, failed=failed))
            pl_broken = True
    return events


def detect_fvgs(HI, LO, ATR):
    n = len(HI)
    events = []
    for i in range(1, n - 1):
        atr = ATR[i]
        if atr <= 0 or np.isnan(atr): continue
        if HI[i - 1] < LO[i + 1]:
            fbot, ftop = HI[i - 1], LO[i + 1]
            sz = ftop - fbot
            if sz < FVG_MIN_ATR_FRAC * atr: continue
            mitigated = False
            for j in range(i + 2, min(i + 2 + MITIGATION_CHECK_BARS, n)):
                if LO[j] <= ftop: mitigated = True; break
            events.append(dict(idx=i, size_atr=sz / atr, mitigated=mitigated))
        elif LO[i - 1] > HI[i + 1]:
            ftop, fbot = LO[i - 1], HI[i + 1]
            sz = ftop - fbot
            if sz < FVG_MIN_ATR_FRAC * atr: continue
            mitigated = False
            for j in range(i + 2, min(i + 2 + MITIGATION_CHECK_BARS, n)):
                if HI[j] >= fbot: mitigated = True; break
            events.append(dict(idx=i, size_atr=sz / atr, mitigated=mitigated))
    return events


def detect_obs(OP, CL, ATR):
    n = len(CL)
    events = []
    for i in range(1, n - OB_MAX_SWING_BARS - 1):
        atr = ATR[i]
        if atr <= 0 or np.isnan(atr): continue
        if CL[i] < OP[i]:
            base = CL[i]
            for s in range(i + 1, min(i + 1 + OB_MAX_SWING_BARS, n)):
                if CL[s] - base >= OB_SWING_ATR_MULT * atr:
                    events.append(dict(idx=i, impulse_atr=(CL[s] - base) / atr)); break
        elif CL[i] > OP[i]:
            base = CL[i]
            for s in range(i + 1, min(i + 1 + OB_MAX_SWING_BARS, n)):
                if base - CL[s] >= OB_SWING_ATR_MULT * atr:
                    events.append(dict(idx=i, impulse_atr=(base - CL[s]) / atr)); break
    return events


def detect_equal_levels(HI, LO, ATR):
    pivots = find_pivots(HI, LO, LEFT, RIGHT)
    highs = [p for p in pivots if p["kind"] == 1]
    lows = [p for p in pivots if p["kind"] == -1]
    n_eq = 0
    for group in (highs, lows):
        for k in range(1, len(group)):
            p = group[k]
            atr = ATR[p["bar_idx"]] if p["bar_idx"] < len(ATR) else np.nan
            if np.isnan(atr) or atr <= 0: continue
            for j in range(max(0, k - EQ_LOOKBACK_PIVOTS), k):
                if abs(group[j]["price"] - p["price"]) <= EQ_TOLERANCE_ATR * atr:
                    n_eq += 1
                    break
    return n_eq


# ── Loop timeframe, calcolo metriche ───────────────────────────────────────
summary = []
for tf in TIMEFRAMES:
    d = dfs[tf]
    n_bars = len(d)
    years = n_bars / BARS_PER_YEAR[tf]
    CL = d["close"].values.astype(float); HI = d["high"].values.astype(float)
    LO = d["low"].values.astype(float); OP = d["open"].values.astype(float)
    ATR = np.where(d["atr_14"].values > 0, d["atr_14"].values, np.nan)

    t1 = time.time()
    breaks = detect_structure_breaks(CL, HI, LO, ATR)
    fvgs = detect_fvgs(HI, LO, ATR)
    obs = detect_obs(OP, CL, ATR)
    n_eq = detect_equal_levels(HI, LO, ATR)
    dt = time.time() - t1

    bos = [e for e in breaks if e["kind"] == "BOS"]
    choch = [e for e in breaks if e["kind"] == "CHoCH"]

    row = dict(tf=tf, n_bars=n_bars, years=years, calc_s=dt,
               n_bos=len(bos), bos_per_yr=len(bos) / years if years > 0 else 0,
               bos_overshoot=np.mean([e["overshoot"] for e in bos]) if bos else np.nan,
               bos_fail=np.mean([e["failed"] for e in bos]) if bos else np.nan,
               n_choch=len(choch), choch_per_yr=len(choch) / years if years > 0 else 0,
               choch_overshoot=np.mean([e["overshoot"] for e in choch]) if choch else np.nan,
               choch_fail=np.mean([e["failed"] for e in choch]) if choch else np.nan,
               n_fvg=len(fvgs), fvg_per_yr=len(fvgs) / years if years > 0 else 0,
               fvg_size=np.mean([e["size_atr"] for e in fvgs]) if fvgs else np.nan,
               fvg_mit_frac=np.mean([e["mitigated"] for e in fvgs]) if fvgs else np.nan,
               n_ob=len(obs), ob_per_yr=len(obs) / years if years > 0 else 0,
               ob_impulse=np.mean([e["impulse_atr"] for e in obs]) if obs else np.nan,
               n_eq=n_eq, eq_per_yr=n_eq / years if years > 0 else 0)
    summary.append(row)
    print(f"  {tf:>4}: BOS={len(bos)} CHoCH={len(choch)} FVG={len(fvgs)} OB={len(obs)} EQH/L={n_eq}  "
          f"(calc {dt:.1f}s)")

# ── Report tables ──────────────────────────────────────────────────────────
w(f"\n{SEP}")
w("BOS (Break of Structure) — continuazione di trend")
w(SEP)
w(f"\n  {'TF':>5}  {'n/anno':>7}  {'overshoot(ATR)':>15}  {'failure rate':>13}")
for r in summary:
    w(f"  {r['tf']:>5}  {r['bos_per_yr']:>7.1f}  {r['bos_overshoot']:>14.3f}x  {r['bos_fail']:>12.1%}")

w(f"\n{SEP}")
w("CHoCH (Change of Character) — segnale di inversione")
w(SEP)
w(f"\n  {'TF':>5}  {'n/anno':>7}  {'overshoot(ATR)':>15}  {'failure rate':>13}")
for r in summary:
    w(f"  {r['tf']:>5}  {r['choch_per_yr']:>7.1f}  {r['choch_overshoot']:>14.3f}x  {r['choch_fail']:>12.1%}")

w(f"\n{SEP}")
w("FVG (Fair Value Gap)")
w(SEP)
w(f"\n  {'TF':>5}  {'n/anno':>7}  {'size media(ATR)':>16}  {'% mitigato entro {}barre'.format(MITIGATION_CHECK_BARS):>26}")
for r in summary:
    w(f"  {r['tf']:>5}  {r['fvg_per_yr']:>7.1f}  {r['fvg_size']:>15.3f}x  {r['fvg_mit_frac']:>25.1%}")

w(f"\n{SEP}")
w("Order Block (ultima candela opposta prima di un impulso >=1.5xATR)")
w(SEP)
w(f"\n  {'TF':>5}  {'n/anno':>7}  {'impulso medio(ATR)':>19}")
for r in summary:
    w(f"  {r['tf']:>5}  {r['ob_per_yr']:>7.1f}  {r['ob_impulse']:>18.3f}x")

w(f"\n{SEP}")
w("Equal Highs/Lows (pool di liquidità, tolleranza 0.15xATR)")
w(SEP)
w(f"\n  {'TF':>5}  {'n/anno':>7}")
for r in summary:
    w(f"  {r['tf']:>5}  {r['eq_per_yr']:>7.1f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/smc_structure_clarity.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# SMC Structure Clarity — confronto multi-timeframe\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
