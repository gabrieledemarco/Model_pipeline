#!/usr/bin/env python3
"""
create_ewcore_elliott_predictive_report.py
==========================================
Analisi degli "oggetti" usati dall'indicatore TradingView "EWCore v0.2.6.4"
(Elliott Wave auto-counter condiviso dall'utente come Pine Script v6) e
valutazione del loro potere predittivo su BTCUSDT, con la stessa
metodologia (event study, forward return nella direzione implicita,
Welch t-test vs controllo random-direction) già usata in questa sessione
per `create_ict_objects_predictive_report.py`.

── Oggetti dell'indicatore, replicati fedelmente ────────────────────────
  1. PIVOT ATR-ADATTIVO (spike-robust): la costruzione di base di TUTTI
     gli altri oggetti. Soglia = min(ATR(14), medianTR(14)*1.3) * mult,
     dove mult è dato dalla legge log dello script:
       mult = clip(0.87 + 0.159*ln(minuti_TF), 0.1, 5.0)
     con floor 1.7 per TF intraday <180 minuti (H1, M15) — replica esatta
     dei valori di default dello script (D1=2.03, H4=1.74, H1=1.70[floor],
     M15=1.70[floor]).
  2. IMPULSE (5-3-5-3-5): pattern motivo a 6 punti (P0..P5) con le 3
     regole hard classiche verificate ESATTAMENTE come nello script
     (tolleranza=0, comportamento di default):
       - onda 2 non ritraccia oltre l'inizio di onda 1
       - onda 3 non è la più corta tra 1/3/5 E supera la fine di onda 1
       - onda 4 non entra nel territorio di onda 1 (no overlap)
     Direzione implicita del segnale = OPPOSTA alla direzione dell'impulso
     (teoria EW: al completamento di onda 5 ci si aspetta un'inversione
     in una fase correttiva — è l'uso pratico che lo script stesso
     implica disegnando target/invalidation per la fase successiva).
  3. ZIGZAG/FLAT (3 onde correttive, A-B-C): pattern a 4 punti (P0..P3)
     con l'unica regola hard condivisa dallo script per entrambi i tipi:
     l'onda B non ritraccia oltre 1.5x della gamba che corregge. Direzione
     implicita = OPPOSTA alla gamba C (teoria EW: al completamento della
     correzione il prezzo dovrebbe riprendere il trend precedente — stessa
     identica logica "fade the completed move" già validata in questa
     sessione per la strategia ICT Fade Standalone).
  4. INVALIDATION BREACH: rottura, in chiusura, del livello di
     invalidazione di un Impulse già completato (fine di onda 1 — la
     regola di non-overlap onda4/onda1) — l'esatto livello disegnato
     dallo script (`topInvalidationLevel`, linea rossa + dot). Direzione
     implicita = CONTINUAZIONE nella direzione della rottura (il conteggio
     è invalidato, la struttura "motiva" si nega da sola).

Oggetti NON replicati (fuori scope, spiegato nel report): Diagonal
(variante dell'Impulse con overlap ammesso — stesso carattere
direzionale, avrebbe raddoppiato l'implementazione per un segnale non
distinguibile nell'event study), Triangle e Combination WXY/WXYXZ
(pattern rari, "possono esistere solo in teoria" per WXYXZ secondo il
tooltip dello script stesso — campioni troppo piccoli per un test
robusto), i livelli di compressione di grado superiore (skeleton
multi-grado — feature di visualizzazione/ricerca, non un oggetto di
segnale distinto), la verifica di sotto-struttura (sub-wave check) e il
Parent-Timeframe Context Gate (filtro di score, non un oggetto).
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

SEP = "═" * 78
START_YEAR = 2020
TIMEFRAMES = ["1D", "4H", "1H", "15M"]
TF_MINUTES = {"1D": 1440, "4H": 240, "1H": 60, "15M": 15}
HORIZONS = [1, 5, 10, 20]

# EWCore defaults (replicati esattamente)
ATR_LEN = 14
AUTO_BASE = 0.87
AUTO_SLOPE = 0.159
SUB_H1_NOISE_FLOOR = 1.7
MED_TR_SCALE = 1.3
B_RATIO_CEILING = 1.5          # tolleranza 0 = comportamento esatto di default
INVALIDATION_MAX_WAIT_BARS = 200

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w('EWCore v0.2.6.4 (Elliott Wave, TradingView) — Oggetti e potere predittivo')
w(SEP)
w("\nReplica fedele dei costrutti causali dello script Pine v6 condiviso")
w("(pivot ATR-adattivo spike-robust, Impulse, Zigzag/Flat, Invalidation")
w("breach) + stessa metodologia event-study già usata per gli oggetti ICT/SMC:")
w("forward return nella direzione implicita, Welch t-test vs controllo")
w("random-direction, 4 orizzonti (1/5/10/20 barre), 4 timeframe.")

t0 = time.time()
print("\n[DATA] Loading 1D, 4H, 1H, 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
dfs = {tf: add_indicators(raw[tf]) for tf in TIMEFRAMES}
for tf in TIMEFRAMES:
    print(f"  {tf:>4}: {len(dfs[tf]):,} bars")
print(f"  (loaded in {time.time()-t0:.0f}s)")


# ── 1) Pivot ATR-adattivo, spike-robust (identico allo script) ──────────
def get_atr_mult(tf_minutes):
    mult = np.clip(AUTO_BASE + AUTO_SLOPE * np.log(tf_minutes), 0.1, 5.0)
    if tf_minutes < 180:
        mult = max(mult, SUB_H1_NOISE_FLOOR)
    return mult


def compute_pivots(df, tf_minutes):
    """Zigzag ATR-adattivo causale con clamp spike-robust (median-TR).
    Replica bar-per-bar la logica isconfirmed dello script: estremo
    corrente vs soglia dinamica; nessun lookahead (ogni pivot usa solo
    dati fino alla barra corrente)."""
    HI = df["high"].values.astype(float)
    LO = df["low"].values.astype(float)
    CL = df["close"].values.astype(float)
    prev_c = np.roll(CL, 1); prev_c[0] = CL[0]
    tr = np.maximum.reduce([HI - LO, np.abs(HI - prev_c), np.abs(LO - prev_c)])
    atr = pd.Series(tr).ewm(com=ATR_LEN - 1, adjust=False).mean().values
    med_tr = pd.Series(tr).rolling(ATR_LEN, min_periods=ATR_LEN).median().values
    basis = np.where(np.isnan(med_tr), atr, np.minimum(atr, med_tr * MED_TR_SCALE))
    mult = get_atr_mult(tf_minutes)
    threshold = basis * mult

    n = len(CL)
    pivots = []  # dict(idx, price, kind=+1 high/-1 low)
    pivot_dir = 0
    extreme_price = CL[0]
    extreme_idx = 0
    for i in range(1, n):
        if np.isnan(threshold[i]):
            continue
        if pivot_dir >= 0 and HI[i] > extreme_price:
            extreme_price, extreme_idx = HI[i], i
        if pivot_dir <= 0 and LO[i] < extreme_price:
            extreme_price, extreme_idx = LO[i], i
        if pivot_dir <= 0 and (HI[i] - extreme_price) > threshold[i] and pivot_dir != 1:
            pivots.append(dict(idx=extreme_idx, price=extreme_price, kind=-1))
            pivot_dir = 1
            extreme_price, extreme_idx = HI[i], i
        elif pivot_dir >= 0 and (extreme_price - LO[i]) > threshold[i] and pivot_dir != -1:
            pivots.append(dict(idx=extreme_idx, price=extreme_price, kind=1))
            pivot_dir = -1
            extreme_price, extreme_idx = LO[i], i
    return pivots


# ── 2) Impulse (5-3-5-3-5), regole hard esatte (tolleranza 0) ───────────
def detect_impulses(pivots):
    """Finestra di 6 pivot consecutivi P0..P5. Ritorna eventi al pivot P5
    (completamento onda 5), direzione implicita = OPPOSTA all'impulso
    (aspettativa di inversione post-completamento, uso pratico EW)."""
    events = []
    P = pivots
    for i in range(len(P) - 5):
        p0, p1, p2, p3, p4, p5 = (P[i + k]["price"] for k in range(6))
        up = p1 > p0
        if up:
            len1, len3, len5 = p1 - p0, p3 - p2, p5 - p4
            ok = (p2 > p0) and (p3 > p1) and (len3 >= min(len1, len5)) and (p4 > p1)
        else:
            len1, len3, len5 = p0 - p1, p2 - p3, p4 - p5
            ok = (p2 < p0) and (p3 < p1) and (len3 >= min(len1, len5)) and (p4 < p1)
        if ok:
            impulse_dir = 1 if up else -1
            events.append(dict(idx=P[i + 5]["idx"], dir=-impulse_dir,
                                inval_idx=P[i + 1]["idx"], inval_price=p1,
                                inval_dir=-impulse_dir))
    return events


# ── 3) Zigzag/Flat (A-B-C correttivo), B <= 1.5x A ───────────────────────
def detect_zigzags(pivots):
    """Finestra di 4 pivot P0..P3. Evento al pivot P3 (fine onda C),
    direzione implicita = OPPOSTA alla gamba C (ripresa del trend
    pre-correzione, stessa logica 'fade' della strategia ICT validata)."""
    events = []
    P = pivots
    for i in range(len(P) - 3):
        p0, p1, p2, p3 = (P[i + k]["price"] for k in range(4))
        len_a = abs(p1 - p0); len_b = abs(p2 - p1); len_c = abs(p3 - p2)
        if len_a <= 0:
            continue
        if len_b > B_RATIO_CEILING * len_a:
            continue
        c_dir = 1 if p3 > p2 else -1
        events.append(dict(idx=P[i + 3]["idx"], dir=-c_dir))
    return events


# ── 4) Invalidation breach (solo Impulse: rottura fine onda 1) ──────────
def detect_invalidation_breaches(impulse_events, CL):
    """Cerca, a partire dalla barra SUCCESSIVA al completamento (j0+1), la
    prima chiusura che rompe il livello di invalidazione (fine onda 1) —
    d=-1 (impulso up invalidato) richiede una rottura AL RIBASSO
    (CL<lvl), d=+1 (impulso down invalidato) una rottura AL RIALZO
    (CL>lvl). Un evento distinto dal completamento stesso, non lo stesso
    bar (bug precedente: condizioni invertite facevano scattare il breach
    sulla stessa barra del completamento, producendo numeri quasi
    identici all'Impulse — corretto qui)."""
    n = len(CL)
    events = []
    for e in impulse_events:
        j0, lvl, d = e["idx"], e["inval_price"], e["inval_dir"]
        for k in range(j0 + 1, min(j0 + 1 + INVALIDATION_MAX_WAIT_BARS, n)):
            if d == -1 and CL[k] < lvl:
                events.append(dict(idx=k, dir=d)); break
            if d == 1 and CL[k] > lvl:
                events.append(dict(idx=k, dir=d)); break
    return events


# ── Predictive power event study (identica metodologia sessione) ────────
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
        if len(ev_ret) < 10:
            continue
        tstat, pval = st.ttest_ind(ev_ret, ctrl_ret, equal_var=False)
        rows.append(dict(h=h, n=len(ev_ret), mean=ev_ret.mean() * 100, tstat=tstat, pval=pval))
    return rows


OBJECT_LABELS = ["Impulse (rev. attesa)", "Zigzag/Flat (fade C)", "Invalidation breach"]

all_results = {}
pivot_counts = {}
for tf in TIMEFRAMES:
    d = dfs[tf]
    CL = d["close"].values.astype(float)
    n = len(d)
    all_idx_pool = np.arange(50, n - max(HORIZONS) - 1)

    pivots = compute_pivots(d, TF_MINUTES[tf])
    pivot_counts[tf] = len(pivots)
    impulses = detect_impulses(pivots)
    zigzags = detect_zigzags(pivots)
    invalidations = detect_invalidation_breaches(impulses, CL)

    objs = {"Impulse (rev. attesa)": impulses, "Zigzag/Flat (fade C)": zigzags,
            "Invalidation breach": invalidations}
    print(f"  {tf:>4}: pivots={len(pivots)}  " + "  ".join(f"{k}={len(v)}" for k, v in objs.items()))

    tf_results = {}
    for label, evs in objs.items():
        tf_results[label] = dict(n=len(evs), stats=eval_predictive_power(evs, CL, all_idx_pool))
    all_results[tf] = tf_results

# ── Report: pivot count + mult effettivo per TF ──────────────────────────
w(f"\n{SEP}")
w("PIVOT ATR-ADATTIVI — conteggio e moltiplicatore effettivo per timeframe")
w(SEP)
w(f"\n  {'TF':<6}{'ATR mult (auto)':>18}{'Pivot totali':>16}")
for tf in TIMEFRAMES:
    w(f"  {tf:<6}{get_atr_mult(TF_MINUTES[tf]):>18.3f}{pivot_counts[tf]:>16,}")

# ── Report: frequenza oggetti ─────────────────────────────────────────────
w(f"\n{SEP}")
w("FREQUENZA (eventi totali nel campione, per timeframe)")
w(SEP)
w(f"\n  {'Oggetto':<24}" + "".join(f"{tf:>10}" for tf in TIMEFRAMES))
for label in OBJECT_LABELS:
    row = f"  {label:<24}"
    for tf in TIMEFRAMES:
        row += f"{all_results[tf][label]['n']:>10,}"
    w(row)

# ── Report: potere predittivo per orizzonte ───────────────────────────────
for h_show in HORIZONS:
    w(f"\n{SEP}")
    w(f"POTERE PREDITTIVO — orizzonte {h_show} barre (mean return % nella direzione implicita, * = p<0.05)")
    w(SEP)
    w(f"\n  {'Oggetto':<24}" + "".join(f"{tf:>16}" for tf in TIMEFRAMES))
    for label in OBJECT_LABELS:
        row = f"  {label:<24}"
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
out_path = Path("reports/ewcore_elliott_predictive.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# EWCore v0.2.6.4 (Elliott Wave) — Oggetti e potere predittivo\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
