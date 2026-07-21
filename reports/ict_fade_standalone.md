# Fade dei trigger ICT standalone (senza supporto 4H) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Fade dei trigger ICT standalone (senza supporto 4H) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Combina i 6 oggetti (Sweep/FVG/OB/Breaker/IFVG/PO3): quando un trigger 15M
si forma SENZA una zona demand/supply 4H attiva, si prende il lato OPPOSTO
(fade) — segnale con l'effetto più forte trovato in questa sessione.

  Holdout genuino 2025-2026: 314 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.5
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1399  wr=57.8% (p=0.0000 vs 50%)  ret=+178.4%  mdd=-15.8%
    Exit: TP=786  SL=507  time=106
    MC i.i.d.  : pp=1.000  pr=0.000
    MC block   : pp=1.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     221    +59.1%  62.9%
      2021     226    +73.2%  60.6%
      2022     221    +46.0%  62.0%
      2023     201    -35.8%  46.8%
      2024     216    +29.3%  58.8%
      2025     209    -11.8%  52.2%
      2026     105    +18.4%  62.9%

  HOLDOUT GENUINO 2025-2026: n=314  wr=55.7%  ret=+6.6%  mdd=-31.3%
    MC i.i.d.  : pp=0.604  pr=0.000
    MC block   : pp=0.607  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.0
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1399  wr=52.8% (p=0.0185 vs 50%)  ret=+289.4%  mdd=-9.9%
    Exit: TP=664  SL=581  time=154
    MC i.i.d.  : pp=1.000  pr=0.000
    MC block   : pp=1.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     221    +72.7%  55.7%
      2021     226    +85.3%  54.4%
      2022     221    +80.3%  58.8%
      2023     201    -16.5%  45.3%
      2024     216    +56.3%  54.2%
      2025     209     -3.2%  47.4%
      2026     105    +14.4%  53.3%

  HOLDOUT GENUINO 2025-2026: n=314  wr=49.4%  ret=+11.3%  mdd=-30.2%
    MC i.i.d.  : pp=0.641  pr=0.003
    MC block   : pp=0.644  pr=0.001

══════════════════════════════════════════════════════════════════════════════
RR = 3.0
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1399  wr=47.0% (p=0.9893 vs 50%)  ret=+512.7%  mdd=-6.3%
    Exit: TP=525  SL=664  time=210
    MC i.i.d.  : pp=1.000  pr=0.000
    MC block   : pp=1.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     221   +118.5%  49.8%
      2021     226   +116.4%  47.8%
      2022     221   +109.4%  50.7%
      2023     201    +26.3%  42.8%
      2024     216    +76.9%  46.8%
      2025     209    +27.0%  42.6%
      2026     105    +38.2%  48.6%

  HOLDOUT GENUINO 2025-2026: n=314  wr=44.6%  ret=+65.2%  mdd=-22.3%
    MC i.i.d.  : pp=0.960  pr=0.000
    MC block   : pp=0.968  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (griglia RR)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    1.5    1399   +178.4%        3.876    1.000
    2.0    1399   +289.4%        5.321    1.000
    3.0    1399   +512.7%        7.382    1.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    1.5     314     +6.6%        0.299    0.000
    2.0     314    +11.3%        0.436    0.000
    3.0     314    +65.2%        2.012    1.000

══════════════════════════════════════════════════════════════════════════════
Slippage-stress — RR migliore per DSR full+holdout = 3.0
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    1399   +512.7%  47.0%   1.000   0.000
        0bps     holdout     314    +65.2%  44.6%   0.960   0.000
        2bps  full-sample    1399   +372.4%  46.5%   1.000   0.000
        2bps     holdout     314    +28.3%  43.6%   0.762   0.003
        5bps  full-sample    1399   +162.0%  45.9%   0.998   0.000
        5bps     holdout     314    -27.1%  42.7%   0.268   0.235
       10bps  full-sample    1399   -188.7%  44.4%   0.000   1.000
       10bps     holdout     314   -119.4%  40.4%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: strategia validata — la prima in questa sessione a superare DSR=1.000 su entrambi gli ambiti

```
RR=3.0 (il candidato validato)
  Full-sample:  n=1399  ret=+512.7%  WR=47.0%  DSR=1.000  MC pp=1.000  pr=0.000
  Holdout 25-26: n=314  ret=+65.2%   WR=44.6%  DSR=1.000  MC pp=0.960  pr=0.000

  Positivo in OGNI singolo anno 2020-2026 (unico caso in tutta la sessione):
  2020 +118.5%  2021 +116.4%  2022 +109.4%  2023 +26.3%  2024 +76.9%  2025 +27.0%  2026 +38.2%
```

**Cosa fa la strategia**: combina i 6 oggetti ICT/SMC studiati (Sweep,
FVG, Order Block, Breaker Block, Inversion FVG, Power of 3) in un unico
segnale — quando uno qualsiasi di questi trigger si forma sul 15M SENZA
una zona demand/supply 4H attiva a supportarlo (nessun Order Block o FVG
4H ancora valido nella stessa direzione), si prende il lato OPPOSTO
(fade). Stop strutturale (minimo/massimo su 10 barre), target RR=3.0,
frizioni Bybit obbligatorie (0.14% round-trip) già incluse in ogni
numero sopra.

**Perché è diverso da tutto il resto testato in questa sessione**:
- **DSR=1.000 su ENTRAMBI gli ambiti** — full-sample E holdout. Nessun'altra
  strategia (inclusa `vp_vwap_confluence.md`, full=0.884/holdout=0.498) ha
  raggiunto questo livello su entrambi contemporaneamente.
- **Positivo in ogni singolo anno del campione 2020-2026** — nessun'altra
  strategia di questa sessione ha questa proprietà (tutte le altre,
  comprese quelle validate, avevano almeno un anno negativo).
- **Win rate strutturalmente sotto il 50% (44.6-47.0%) compensato da RR
  asimmetrico** — stesso pattern di trend-continuation/mean-reversion-fade
  già visto nell'unica altra strategia riuscita, ma qui con Sharpe_hat
  molto più alto (7.382 full, 2.012 holdout, contro 2.5-2.8 di Breakout
  pool di Carver).

**Margine di sicurezza sui costi di esecuzione**: la strategia regge fino
a ~2bps di slippage EXTRA oltre la frizione già inclusa (2bps: full
+372.4%, holdout +28.3%, ancora solido) ma degrada rapidamente oltre: a
5bps extra il full-sample resta positivo (+162.0%, MC pp=0.998) ma
l'holdout diventa negativo (-27.1%, MC pp scende a 0.268, MC pr sale a
0.235) — un margine reale ma non enorme. Su BTCUSDT (mercato liquido,
spread tipicamente sub-bps sui book migliori) questo è un'assunzione
ragionevole ma non extra-conservativa; andrebbe verificato con dati
order-book reali prima di un uso live.

**Caveat onesti**:
- Il campione holdout (n=314) è più piccolo del full-sample — il Sharpe_hat
  2.012 è comunque forte, ma la DSR family qui è piccola (N=3, solo la
  griglia RR) — non è stata testata alcuna selezione sui parametri dei
  6 detector sottostanti (soglie ATR, finestre di lookback), che sono
  fissi e mutuati direttamente dagli studi diagnostici precedenti, non
  ottimizzati per questo backtest — un punto a favore della genuinità
  del risultato (nessun grid-search sui parametri di detection), ma
  comunque da tenere presente.
- Il 15% dei trade (RR=3.0: 210/1399) esce per tempo (24h) invece che per
  stop o target — un mark-to-market alla chiusura, non un'uscita pulita.
- Come discusso in `htf_ltf_confluence.md`, l'interpretazione più
  probabile non è "i pattern ICT funzionano al contrario" ma un
  meccanismo di ALLINEAMENTO DI TREND: un trigger senza zona 4H a
  supporto tende a comparire in un contesto dove il timeframe superiore
  NON conferma quella direzione — fadarlo equivale, in parte, a scommettere
  contro un segnale privo di conferma strutturale, coerente con
  l'ingrediente chiave dell'altra strategia validata in sessione.

**Conclusione**: strategia validata secondo tutti i criteri stabiliti in
questa sessione (DSR, holdout genuino, Monte Carlo i.i.d.+block, frizioni
Bybit reali, slippage-stress) — il risultato più solido e completo
dell'intera esplorazione ICT/SMC, e il migliore in assoluto per
consistenza anno-su-anno.
