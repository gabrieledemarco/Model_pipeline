# Harmonic Patterns (ricerca generica Carney) — BTCUSDT

```
══════════════════════════════════════════════════════════════════════════════
Harmonic Patterns (ricerca generica sui 15 rapporti di Carney) — BTCUSDT
══════════════════════════════════════════════════════════════════════════════

Replica della metodologia condivisa dall'utente: 3375 pattern generici
p_XXXX_YYYY_ZZZZ (invece dei soli Gartley/Bat/Crab con nome), swing
point via trend-line 'non convenzionale', tolleranza 5% sui rapporti,
simulazione trade con target=altezza pattern, stop=25% altezza.

══════════════════════════════════════════════════════════════════════════════
[1D] RISULTATI — 410 trade totali, 328 pattern distinti con >=1 occorrenza
══════════════════════════════════════════════════════════════════════════════

  Pattern 'buoni' (n>=30, perf medio>=0.1, SENZA fee): 0
  Nessun pattern raggiunge la soglia (n>=30, perf medio>=0.10).

  TUTTI i trade insieme (ignorando nome pattern): n=410  mean_perf=0.050  mean_ret=+1.762%  mean_ret_fee=+1.622%  win_rate=34.9%
  t-test one-sample (ret_fee vs 0): t=1.86  p=0.0634  (NON significativo)

  Breakdown annuale (ret_fee medio per trade, dopo fee reali):
  Year       n   ret_fee_medio    win%
  2020      53          8.084%   45.3%
  2021      51          5.514%   31.4%
  2022      58         -0.085%   32.8%
  2023      22         10.601%   72.7%
  2024      81         -0.823%   29.6%
  2025      99         -1.732%   30.3%
  2026      46         -0.752%   30.4%
  Anni con ret_fee medio positivo: 3/7

══════════════════════════════════════════════════════════════════════════════
[4H] RISULTATI — 2,745 trade totali, 1609 pattern distinti con >=1 occorrenza
══════════════════════════════════════════════════════════════════════════════

  Pattern 'buoni' (n>=30, perf medio>=0.1, SENZA fee): 0
  Nessun pattern raggiunge la soglia (n>=30, perf medio>=0.10).

  TUTTI i trade insieme (ignorando nome pattern): n=2,745  mean_perf=0.003  mean_ret=+0.070%  mean_ret_fee=-0.070%  win_rate=31.6%
  t-test one-sample (ret_fee vs 0): t=-0.63  p=0.5292  (NON significativo)

  Breakdown annuale (ret_fee medio per trade, dopo fee reali):
  Year       n   ret_fee_medio    win%
  2020     363         -0.348%   27.8%
  2021     449          0.466%   33.6%
  2022     459         -0.043%   31.8%
  2023     409         -0.019%   32.3%
  2024     480         -0.435%   29.8%
  2025     352          0.200%   36.9%
  2026     233         -0.464%   27.5%
  Anni con ret_fee medio positivo: 2/7

══════════════════════════════════════════════════════════════════════════════
AVVERTENZE METODOLOGICHE IMPORTANTI
══════════════════════════════════════════════════════════════════════════════

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


══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
