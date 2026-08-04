# Confluenza multi-timeframe: zona 4H (demand/supply) attiva + trigger 15M

```
══════════════════════════════════════════════════════════════════════════════
Confluenza multi-timeframe: zona 4H (demand/supply) attiva + trigger 15M
══════════════════════════════════════════════════════════════════════════════

Demand 4H (OB/FVG rialzista) o Supply 4H (OB/FVG ribassista) ATTIVA
mentre un trigger (sweep/FVG/OB/breaker/IFVG/PO3) si forma sul 15M —
confronto diretto: la confluenza migliora il potere predittivo o no?

══════════════════════════════════════════════════════════════════════════════
Frequenza: confluenza (dentro zona 4H attiva) vs standalone
══════════════════════════════════════════════════════════════════════════════

  Trigger 15M             n totale   n confluenza   n standalone   % confluenza
  Sweep                     54,081         42,545         11,536          78.7%
  FVG                       37,130         31,748          5,382          85.5%
  Order Block              123,958        100,861         23,097          81.4%
  Breaker Block             85,612         74,294         11,318          86.8%
  Inversion FVG             28,489         23,432          5,057          82.2%
  Power of 3 (AMD)           5,122          3,859          1,263          75.3%

══════════════════════════════════════════════════════════════════════════════
Potere predittivo a 1 barre — CONFLUENZA vs STANDALONE vs TUTTI (mean %, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Trigger 15M                      TUTTI        CONFLUENZA        STANDALONE
  Sweep                 -0.002%  (n=54080)  +0.008%* (n=42545)  -0.037%* (n=11535)
  FVG                   -0.002%  (n=37130)  +0.005%* (n=31748)  -0.043%* (n=5382)
  Order Block           -0.002%  (n=123958)  +0.007%* (n=100861)  -0.044%* (n=23097)
  Breaker Block         +0.000%  (n=85612)  +0.008%* (n=74294)  -0.052%* (n=11318)
  Inversion FVG         -0.008%* (n=28489)  +0.005%* (n=23432)  -0.068%* (n=5057)
  Power of 3 (AMD)      -0.005%  (n=5122)  +0.003%  (n=3859)  -0.030%* (n=1263)

══════════════════════════════════════════════════════════════════════════════
Potere predittivo a 5 barre — CONFLUENZA vs STANDALONE vs TUTTI (mean %, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Trigger 15M                      TUTTI        CONFLUENZA        STANDALONE
  Sweep                 +0.002%  (n=54080)  +0.044%* (n=42545)  -0.153%* (n=11535)
  FVG                   +0.002%  (n=37129)  +0.031%* (n=31748)  -0.168%* (n=5381)
  Order Block           -0.004%  (n=123958)  +0.034%* (n=100861)  -0.170%* (n=23097)
  Breaker Block         -0.001%  (n=85612)  +0.032%* (n=74294)  -0.224%* (n=11318)
  Inversion FVG         -0.010%  (n=28489)  +0.033%* (n=23432)  -0.212%* (n=5057)
  Power of 3 (AMD)      -0.002%  (n=5122)  +0.023%* (n=3859)  -0.079%* (n=1263)

══════════════════════════════════════════════════════════════════════════════
Potere predittivo a 10 barre — CONFLUENZA vs STANDALONE vs TUTTI (mean %, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Trigger 15M                      TUTTI        CONFLUENZA        STANDALONE
  Sweep                 +0.003%  (n=54077)  +0.072%* (n=42545)  -0.249%* (n=11532)
  FVG                   +0.006%  (n=37129)  +0.055%* (n=31748)  -0.287%* (n=5381)
  Order Block           -0.007%* (n=123958)  +0.056%* (n=100861)  -0.286%* (n=23097)
  Breaker Block         -0.006%  (n=85612)  +0.051%* (n=74294)  -0.378%* (n=11318)
  Inversion FVG         -0.007%  (n=28489)  +0.065%* (n=23432)  -0.339%* (n=5057)
  Power of 3 (AMD)      +0.008%  (n=5122)  +0.054%* (n=3859)  -0.135%* (n=1263)

══════════════════════════════════════════════════════════════════════════════
Potere predittivo a 20 barre — CONFLUENZA vs STANDALONE vs TUTTI (mean %, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Trigger 15M                      TUTTI        CONFLUENZA        STANDALONE
  Sweep                 +0.000%  (n=54074)  +0.076%* (n=42544)  -0.281%* (n=11530)
  FVG                   +0.005%  (n=37127)  +0.060%* (n=31748)  -0.318%* (n=5379)
  Order Block           -0.013%* (n=123958)  +0.060%* (n=100861)  -0.333%* (n=23097)
  Breaker Block         -0.010%  (n=85607)  +0.053%* (n=74293)  -0.429%* (n=11314)
  Inversion FVG         -0.011%  (n=28488)  +0.069%* (n=23432)  -0.377%* (n=5056)
  Power of 3 (AMD)      -0.002%  (n=5121)  +0.056%* (n=3859)  -0.181%* (n=1262)

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: la scoperta più netta e sistematica di tutta questa sessione

```
A 20 barre (5h), tutti e 6 i trigger 15M:
                     CONFLUENZA (4H attivo)     STANDALONE (nessuna zona 4H)
Sweep                     +0.076%*                    -0.281%*
FVG                       +0.060%*                    -0.318%*
Order Block               +0.060%*                    -0.333%*
Breaker Block             +0.053%*                    -0.429%*
Inversion FVG             +0.069%*                    -0.377%*
Power of 3 (AMD)          +0.056%*                    -0.181%*
```

**Il pattern è totale e senza eccezioni**: su TUTTI i 6 tipi di trigger,
TUTTI i 4 orizzonti (1/5/10/20 barre), la CONFLUENZA è sempre positiva e
significativa (p<0.05, quasi ogni singola cella), lo STANDALONE è sempre
negativo e significativo. Non è un risultato isolato o fragile come le
scoperte precedenti (1D con 29 eventi che invertiva segno) — qui parliamo
di decine di migliaia di eventi per gruppo, con una direzione
perfettamente coerente su 24 combinazioni trigger×orizzonte. È la
scoperta più netta e sistematica di tutta questa sessione di validazione.

**Ma la lettura corretta non è "le zone 4H funzionano come dice la
teoria ICT" — è più probabilmente un filtro di ALLINEAMENTO DI TREND
travestito da confluenza**: una zona demand 4H attiva esiste perché nei
giorni precedenti il 4H ha avuto un impulso rialzista (è così che un
Order Block o una FVG si formano). "Trigger rialzista dentro zona demand
attiva" equivale quindi, in buona parte, a "trigger rialzista mentre il
4H è in un contesto recentemente rialzista" — lo stesso principio di
allineamento di trend che era l'ingrediente chiave dell'UNICA strategia
validata in questa sessione (`vp_vwap_confluence.md`). Le zone 4H qui
potrebbero essere un proxy per il bias direzionale del timeframe
superiore, non un meccanismo causale "smart money" specifico.

**Il numero più interessante per un possibile sfruttamento pratico non è
la confluenza — è lo STANDALONE**: -0.18% a -0.43% a 20 barre è un
ordine di grandezza SOPRA la frizione round-trip minima Bybit (0.14%)
stabilita in questa sessione. Un trigger che si forma SENZA alcuna zona
4H a supporto tende, in modo sistematico e ben misurato, a muoversi
CONTRO la propria direzione implicita — il che significa che FADARE
(prendere il lato opposto di) i trigger standalone potrebbe avere un
edge lordo economicamente rilevante, a differenza di qualunque altro
segnale trovato finora in questa famiglia di test (dove gli effetti,
quando statisticamente reali, restavano sempre 10-50× sotto i costi di
transazione).

**Non ancora un risultato tradabile**: questo resta un event-study
diagnostico (nessuna fee, nessun sizing, nessuno stop, nessun holdout
separato, nessuna correzione DSR per i 6×4=24 confronti testati). Prima
di trarre conclusioni operative servirebbe: (a) backtest completo della
regola "fade dei trigger standalone" con frizioni reali e sizing, (b)
holdout 2025-2026 genuino, (c) Monte Carlo, (d) verificare se l'effetto
regge quando isolato dal semplice bias di trend (es. controllando per lo
stato HMM o il trend-context già usati altrove in sessione) per capire
se la confluenza aggiunge qualcosa oltre al trend-alignment puro.
**Candidato naturale per il prossimo test**, dato che è il numero più
promettente emerso finora nell'intera esplorazione ICT/SMC di questa
sessione.
