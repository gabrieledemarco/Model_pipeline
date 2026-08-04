#!/usr/bin/env python3
"""
create_harmonic_patterns_report.py
===================================
Replica della metodologia di ricerca "harmonic patterns" descritta
dall'utente (post stile forum: ricerca generica su tutte le 15x15x15=3375
combinazioni di rapporti di Carney, invece dei soli pattern con nome
tipo Gartley/Bat/Crab), applicata a BTCUSDT al posto delle 93 ETF
dell'originale.

── Costruzione degli swing point (identica alla specifica dell'utente) ──
  A ogni barra: previousLow/previousHigh = low/high della barra
  immediatamente PRIMA dell'ultimo swing confermato (o della barra
  precedente se non c'è ancora uno swing). Se low corrente <=
  previousLow -> direzione "down"; se high corrente >= previousHigh ->
  direzione "up". Quando la direzione passa a "up" la barra del
  cambiamento è uno swing LOW (trough); quando passa a "down" è uno
  swing HIGH (peak). Un metodo "trend-line non convenzionale" più
  permissivo di uno zigzag a soglia fissa (ATR ecc.) — produce più swing,
  cattura pattern che uno zigzag standard salterebbe.

── Ricerca pattern generica (5 punti, 3 rapporti gamba-a-gamba) ────────
  15 rapporti armonici di Carney: 0.382 0.500 0.618 0.707 0.786 0.886
  1.130 1.270 1.414 1.618 2.000 2.240 2.618 3.140 3.618.
  Per ogni nuovo swing point (candidato punto 4, il più recente), si
  enumerano tutte le combinazioni di punti 0..3 tra gli ultimi 26 swing
  (finestra di 27 swing totale) con gap di indice DISPARI tra punti
  consecutivi (necessario per l'alternanza picco/valle — la sequenza di
  swing alterna già per costruzione, quindi un gap pari darebbe due
  punti dello stesso tipo). Vincolo di contenimento (rilassato a livello
  di SWING, non di ogni singola barra OHLC — la lettura letterale
  "nessuna barra interna" produce ZERO pattern con span>=9 su BTCUSDT,
  perché uno swing intermedio saltato è quasi sempre un estremo locale
  che eccede il range stretto della gamba, vedi commento nel codice):
  nessuno SWING interno a una gamba può uscire dal range
  [min(inizio,fine), max(inizio,fine)] di quella gamba. I 3 rapporti gamba-a-gamba
  (leg2/leg1, leg3/leg2, leg4/leg3) devono essere entro il 5% di UNO dei
  15 valori canonici ciascuno per essere un'istanza valida del pattern
  "p_XXXX_YYYY_ZZZZ".

── Fitness e simulazione di trade (identica alla specifica) ────────────
  Fitness = (numero di swing nello span, 5-27) + frazione di vicinanza
  media ai rapporti ideali (0-1). Solo pattern con fitness intero >= 9.
  Direzione: punto4 = trough -> long, punto4 = peak -> short (convenzione
  standard "reversal" del trading armonico). Target = altezza del
  pattern (max-min dei 5 punti) dal PUNTO 4 nella direzione del trade;
  stop = 25% dell'altezza dal punto 4 in direzione opposta. Entry =
  apertura della barra successiva al completamento; trade scartato se
  l'apertura non cade tra stop e target. Performance = risultato del
  trade (in prezzo) / altezza del pattern — NESSUNA fee, esattamente
  come lo studio originale (i costi reali vengono aggiunti a parte più
  sotto, per lo standard di questa sessione).

Ambiguità nella fonte originale (segnalate esplicitamente dall'utente
stesso per i rapporti Carney, qui segnalate per la simulazione trade):
il testo dice "if ... the risk to reward ratio would be greater than
1:4, no simulated trade is taken" senza specificare la direzione della
disuguaglianza in modo univoco (dato che stop/target sono fissati da
punto4 non da entry, il R:R realizzato da entry può variare). Questo
filtro è stato OMESSO qui (si mantiene solo il vincolo "entry dentro
[stop,target]") — segnalato esplicitamente nel report.
"""
from __future__ import annotations

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators

SEP = "═" * 78
START_YEAR = 2020
TIMEFRAMES = ["1D", "4H"]

CARNEY_RATIOS = [0.382, 0.500, 0.618, 0.707, 0.786, 0.886, 1.130, 1.270,
                 1.414, 1.618, 2.000, 2.240, 2.618, 3.140, 3.618]
RATIO_TOL = 0.05
WINDOW_SWINGS = 27          # span massimo in numero di swing (punto0..punto4)
MIN_FITNESS = 9
STOP_FRAC = 0.25
MAX_HOLD_BARS = 150
FEE_ROUNDTRIP = 0.0014      # standard di sessione: taker 0.055%*2 + slippage 0.015%*2

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Harmonic Patterns (ricerca generica sui 15 rapporti di Carney) — BTCUSDT")
w(SEP)
w("\nReplica della metodologia condivisa dall'utente: 3375 pattern generici")
w("p_XXXX_YYYY_ZZZZ (invece dei soli Gartley/Bat/Crab con nome), swing")
w("point via trend-line 'non convenzionale', tolleranza 5% sui rapporti,")
w("simulazione trade con target=altezza pattern, stop=25% altezza.")

t0 = time.time()
print("\n[DATA] Loading 1D, 4H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
dfs = {tf: add_indicators(raw[tf]) for tf in TIMEFRAMES}
for tf in TIMEFRAMES:
    print(f"  {tf:>4}: {len(dfs[tf]):,} bars")
print(f"  (loaded in {time.time()-t0:.0f}s)")


# ── Swing points (metodo "trend-line non convenzionale", come da spec) ──
def compute_swings(HI, LO):
    """previousLow/previousHigh = low/high della barra i-1 (1 bar back);
    SE la barra i-1 è proprio la barra dell'ultimo swing confermato
    (self-reference), si usa invece la barra subito precedente a quello
    swing ('just before the previous swing'). Riferimento MOLTO locale
    (non un estremo accumulato): due letture alternative testate prima
    (singolo bar fisso pre-swing; estremo rolling da last-swing) si
    bloccavano dopo pochi swing su un trend persistente come BTC 2020-26
    (il riferimento non veniva più violato) — questa lettura, coerente
    con la dicitura 'two-or-more-bar swings' del testo originale (swing
    molto ravvicinati, non major turn), produce ~31% di barre come
    swing point su BTC 1D, distribuiti su tutto lo storico."""
    n = len(HI)
    swings = []  # dict(idx, price, kind: 1=peak, -1=trough)
    direction = 0
    last_swing_idx = None
    for i in range(1, n):
        if last_swing_idx is not None and (i - 1) == last_swing_idx:
            ref_idx = last_swing_idx - 1
        else:
            ref_idx = i - 1
        ref_idx = max(ref_idx, 0)
        prev_low, prev_high = LO[ref_idx], HI[ref_idx]
        new_dir = direction
        if LO[i] <= prev_low:
            new_dir = -1
        elif HI[i] >= prev_high:
            new_dir = 1
        if new_dir != direction and new_dir != 0:
            if new_dir == 1:
                swings.append(dict(idx=i, price=LO[i], kind=-1))
            else:
                swings.append(dict(idx=i, price=HI[i], kind=1))
            last_swing_idx = i
            direction = new_dir
    return swings


def nearest_ratio_ok(r):
    diffs = [abs(r - c) / c for c in CARNEY_RATIOS]
    j = int(np.argmin(diffs))
    return (j, diffs[j]) if diffs[j] <= RATIO_TOL else (None, None)


def swings_contained(swings, i_a, i_b, p_a, p_b):
    """Vincolo di contenimento a livello di SWING (non di ogni singola
    barra OHLC): nessuno swing point tra i_a e i_b (indici nell'array
    swings, non bar-index) può eccedere [min(p_a,p_b), max(p_a,p_b)].
    NOTA: la lettura letterale ('nessuna barra OHLC interna può uscire
    dal range della gamba') è stata provata per prima e produce ZERO
    pattern con span>=9 su BTCUSDT — per costruzione, uno swing
    intermedio saltato è quasi sempre un estremo locale che eccede il
    range stretto della gamba, quindi qualunque combinazione non
    adiacente (l'intero punto della ricerca 'oltre il semplice zigzag'
    descritta dall'utente) viene scartata. Qui si rilassa il controllo
    ai soli SWING intermedi (comunque estremi locali reali, non barre
    qualsiasi) — 410 istanze trovate su BTCUSDT 1D con questa lettura."""
    lo_b, hi_b = min(p_a, p_b), max(p_a, p_b)
    for k in range(i_a + 1, i_b):
        pk = swings[k]["price"]
        if pk < lo_b - 1e-9 or pk > hi_b + 1e-9:
            return False
    return True


def search_patterns(swings):
    """Per ogni nuovo swing (candidato punto4), enumera le combinazioni
    di punti 0..3 con gap dispari nella finestra di WINDOW_SWINGS."""
    found = []
    odd_gaps = list(range(1, WINDOW_SWINGS, 2))
    n_sw = len(swings)
    for i4 in range(4, n_sw):
        p4 = swings[i4]
        for g4 in odd_gaps:
            i3 = i4 - g4
            if i3 < 0:
                break
            for g3 in odd_gaps:
                i2 = i3 - g3
                if i2 < 0:
                    break
                if (i4 - i2) > WINDOW_SWINGS - 1:
                    break
                for g2 in odd_gaps:
                    i1 = i2 - g2
                    if i1 < 0:
                        break
                    if (i4 - i1) > WINDOW_SWINGS - 1:
                        break
                    for g1 in odd_gaps:
                        i0 = i1 - g1
                        if i0 < 0:
                            break
                        span = i4 - i0 + 1
                        if span > WINDOW_SWINGS:
                            break
                        p0, p1, p2, p3 = swings[i0], swings[i1], swings[i2], swings[i3]
                        leg1 = abs(p1["price"] - p0["price"])
                        leg2 = abs(p2["price"] - p1["price"])
                        leg3 = abs(p3["price"] - p2["price"])
                        leg4 = abs(p4["price"] - p3["price"])
                        if leg1 <= 0 or leg2 <= 0 or leg3 <= 0:
                            continue
                        r1, r2, r3 = leg2 / leg1, leg3 / leg2, leg4 / leg3
                        j1, e1 = nearest_ratio_ok(r1)
                        if j1 is None:
                            continue
                        j2, e2 = nearest_ratio_ok(r2)
                        if j2 is None:
                            continue
                        j3, e3 = nearest_ratio_ok(r3)
                        if j3 is None:
                            continue
                        if not (swings_contained(swings, i0, i1, p0["price"], p1["price"]) and
                                swings_contained(swings, i1, i2, p1["price"], p2["price"]) and
                                swings_contained(swings, i2, i3, p2["price"], p3["price"]) and
                                swings_contained(swings, i3, i4, p3["price"], p4["price"])):
                            continue
                        closeness = 1.0 - (e1 + e2 + e3) / (3 * RATIO_TOL)
                        fitness = span + max(0.0, min(0.999, closeness))
                        if int(fitness) < MIN_FITNESS:
                            continue
                        name = f"p_{CARNEY_RATIOS[j1]*1000:04.0f}_{CARNEY_RATIOS[j2]*1000:04.0f}_{CARNEY_RATIOS[j3]*1000:04.0f}"
                        found.append(dict(name=name, p0=p0, p1=p1, p2=p2, p3=p3, p4=p4,
                                           fitness=fitness))
    return found


def simulate_trade(pattern, OP, HI, LO, CL, n):
    p0, p4 = pattern["p0"], pattern["p4"]
    prices = [pattern["p0"]["price"], pattern["p1"]["price"], pattern["p2"]["price"],
              pattern["p3"]["price"], pattern["p4"]["price"]]
    height = max(prices) - min(prices)
    if height <= 0:
        return None
    direction = 1 if p4["kind"] == -1 else -1   # trough->long, peak->short
    d_price = p4["price"]
    if direction == 1:
        target, stop = d_price + height, d_price - STOP_FRAC * height
    else:
        target, stop = d_price - height, d_price + STOP_FRAC * height

    entry_idx = p4["idx"] + 1
    if entry_idx >= n:
        return None
    entry = OP[entry_idx]
    if direction == 1 and not (stop < entry < target):
        return None
    if direction == -1 and not (target < entry < stop):
        return None

    exit_price, exit_reason = None, "time"
    for k in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, n)):
        if direction == 1:
            if LO[k] <= stop:
                exit_price, exit_reason = stop, "stop"; break
            if HI[k] >= target:
                exit_price, exit_reason = target, "target"; break
        else:
            if HI[k] >= stop:
                exit_price, exit_reason = stop, "stop"; break
            if LO[k] <= target:
                exit_price, exit_reason = target, "target"; break
    if exit_price is None:
        last_k = min(entry_idx + MAX_HOLD_BARS - 1, n - 1)
        exit_price, exit_reason = CL[last_k], "time"

    raw_pnl = direction * (exit_price - entry)
    perf = raw_pnl / height
    ret_pct = direction * (exit_price - entry) / entry
    ret_pct_fee = ret_pct - FEE_ROUNDTRIP
    return dict(name=pattern["name"], dir=direction, entry_idx=entry_idx, perf=perf,
                ret_pct=ret_pct, ret_pct_fee=ret_pct_fee, exit_reason=exit_reason,
                fitness=pattern["fitness"])


# ── Esecuzione per timeframe ─────────────────────────────────────────────
all_trades = {}
for tf in TIMEFRAMES:
    d = dfs[tf]
    HI = d["high"].values.astype(float); LO = d["low"].values.astype(float)
    OP = d["open"].values.astype(float); CL = d["close"].values.astype(float)
    n = len(d)

    print(f"\n[{tf}] Computing swings …")
    swings = compute_swings(HI, LO)
    print(f"  {len(swings):,} swing points su {n:,} barre "
          f"({100*len(swings)/n:.1f}% delle barre)")

    print(f"[{tf}] Searching patterns (finestra {WINDOW_SWINGS} swing, "
          f"tolleranza {RATIO_TOL*100:.0f}%) …")
    t1 = time.time()
    patterns = search_patterns(swings)
    print(f"  {len(patterns):,} istanze pattern trovate (fitness>={MIN_FITNESS}) "
          f"in {time.time()-t1:.0f}s")

    trades = [simulate_trade(p, OP, HI, LO, CL, n) for p in patterns]
    trades = [t for t in trades if t is not None]
    print(f"  {len(trades):,} trade simulati (entry dentro [stop,target])")
    all_trades[tf] = trades

# ── Aggregazione per nome pattern (come studio originale: n>=30, perf medio>=0.1) ──
MIN_OCC = 30
MIN_MEAN_PERF = 0.10

for tf in TIMEFRAMES:
    trades = all_trades[tf]
    if not trades:
        w(f"\n{SEP}\n[{tf}] Nessun trade simulato.\n{SEP}")
        continue
    df_t = pd.DataFrame(trades)
    agg = df_t.groupby("name").agg(n=("perf", "size"),
                                    mean_perf=("perf", "mean"),
                                    mean_ret=("ret_pct", "mean"),
                                    mean_ret_fee=("ret_pct_fee", "mean"),
                                    win_rate=("ret_pct", lambda s: (s > 0).mean())).reset_index()
    good = agg[(agg["n"] >= MIN_OCC) & (agg["mean_perf"] >= MIN_MEAN_PERF)].sort_values(
        "mean_perf", ascending=False)

    w(f"\n{SEP}")
    w(f"[{tf}] RISULTATI — {len(df_t):,} trade totali, {agg.shape[0]} pattern distinti con >=1 occorrenza")
    w(SEP)
    w(f"\n  Pattern 'buoni' (n>={MIN_OCC}, perf medio>={MIN_MEAN_PERF}, SENZA fee): {len(good)}")
    if len(good):
        w(f"\n  {'Pattern':<20}{'n':>6}{'perf_medio':>12}{'ret%':>10}{'ret%_fee':>12}{'win%':>8}")
        for _, row in good.head(20).iterrows():
            w(f"  {row['name']:<20}{int(row['n']):>6}{row['mean_perf']:>12.3f}"
              f"{row['mean_ret']*100:>9.2f}%{row['mean_ret_fee']*100:>11.2f}%{row['win_rate']*100:>7.1f}%")
        n_survive_fee = (good["mean_ret_fee"] > 0).sum()
        w(f"\n  Di questi, {n_survive_fee}/{len(good)} restano PROFITTEVOLI dopo fee reali "
          f"({FEE_ROUNDTRIP*100:.2f}% round-trip Bybit).")
    else:
        w("  Nessun pattern raggiunge la soglia (n>=30, perf medio>=0.10).")

    # aggregato: TUTTI i trade insieme (tutti i pattern), con e senza fee
    w(f"\n  TUTTI i trade insieme (ignorando nome pattern): n={len(df_t):,}  "
      f"mean_perf={df_t['perf'].mean():.3f}  mean_ret={df_t['ret_pct'].mean()*100:+.3f}%  "
      f"mean_ret_fee={df_t['ret_pct_fee'].mean()*100:+.3f}%  win_rate={100*(df_t['ret_pct']>0).mean():.1f}%")

    # significatività (t-test one-sample vs 0, sul pool DOPO fee) + breakdown annuale
    import scipy.stats as st
    tstat, pval = st.ttest_1samp(df_t["ret_pct_fee"].values, 0.0)
    w(f"  t-test one-sample (ret_fee vs 0): t={tstat:.2f}  p={pval:.4f}"
      f"  {'(significativo, p<0.05)' if pval < 0.05 else '(NON significativo)'}")

    df_t["year"] = dfs[tf].index[df_t["entry_idx"].values].year
    yr = df_t.groupby("year").agg(n=("ret_pct_fee", "size"),
                                   ret_fee=("ret_pct_fee", "mean"),
                                   win_rate=("ret_pct", lambda s: (s > 0).mean()))
    w(f"\n  Breakdown annuale (ret_fee medio per trade, dopo fee reali):")
    w(f"  {'Year':<6}{'n':>6}{'ret_fee_medio':>16}{'win%':>8}")
    for yy, row in yr.iterrows():
        w(f"  {yy:<6}{int(row['n']):>6}{row['ret_fee']*100:>15.3f}%{row['win_rate']*100:>7.1f}%")
    n_pos_years = (yr["ret_fee"] > 0).sum()
    w(f"  Anni con ret_fee medio positivo: {n_pos_years}/{len(yr)}")

w(f"\n{SEP}")
w("AVVERTENZE METODOLOGICHE IMPORTANTI")
w(SEP)
w("""
Questo è un event study diagnostico che replica la ricetta descritta
dall'utente, NON una strategia validata secondo lo standard di questa
sessione (DSR family, Monte Carlo, holdout 2025-2026 mai toccato in
design, walk-forward). In particolare:

1. SPAZIO DI RICERCA ENORME: 15x15x15=3375 nomi di pattern possibili,
   moltiplicati per tutte le combinazioni di gap tra swing (fino a
   ~1365 per punto4) e 2 timeframe. Il pool "tutti i trade insieme" che
   mostra un edge positivo su 1D nasconde un multiple-testing implicito
   enorme: la extraction dei 15 rapporti Carney stessi è stata scelta
   a posteriori da decenni di trader come "quelli che funzionano" — un
   caso da manuale del problema Deflated Sharpe Ratio discusso nel post
   Rulyfi ("100 Million Bitcoin Backtests") letto in questa sessione.
   NESSUNA correzione DSR è stata applicata qui.
2. Interpretazioni scelte per ambiguità nel testo originale (segnalate
   inline nel codice): la definizione degli swing point ("previousLow/
   High") e il vincolo di contenimento sono state entrambe reinterpretate
   dopo che le letture letterali producevano risultati degeneri (0-4
   swing totali, o 0 pattern con span>=9). Risultati diversi sono
   possibili con altre letture plausibili dello stesso testo.
3. Nessun controllo di overlap tra trade (più pattern possono aprire
   posizioni sovrapposte sullo stesso periodo, il pool tratta ogni
   trade come indipendente ai fini del t-test, sovrastimando il
   sample size effettivo).
4. Nessuna simulazione di sizing/leva/margine — solo return per-trade.
""")
w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/harmonic_patterns.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Harmonic Patterns (ricerca generica Carney) — BTCUSDT\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
