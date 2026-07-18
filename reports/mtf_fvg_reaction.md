# MTF FVG Reaction (4H FVG -> reazione 15m -> nuovo FVG 15m) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
MTF FVG Reaction (4H FVG -> reazione 15m -> nuovo FVG 15m -> entry) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Replica letterale di una regola pubblicata sui social (non attribuita,
nessun track record allegato): 4H FVG come zona, reazione aggressiva +
nuovo FVG sul 15m come timing di ingresso, target 2-3 RR.

[EVENTS] 2136 FVG 4H -> 403 pattern completi (1 trade/FVG 4H max)

  Holdout genuino 2025-2026: 98 trade

══════════════════════════════════════════════════════════════════════════════
RR = 2.0
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=403  wr=37.5% (p=1.0000 vs 50%)  ret=-63.4%  mdd=-72.6%
    Exit: TP=111  SL=223  time=69
    MC i.i.d.  : pp=0.021  pr=0.753
    MC block   : pp=0.014  pr=0.753

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      53     -8.7%  35.8%
      2021      74     -6.3%  37.8%
      2022      68    -11.4%  41.2%
      2023      51    -10.5%  35.3%
      2024      59     -9.0%  39.0%
      2025      63    -19.5%  31.7%
      2026      35     +1.9%  42.9%

  HOLDOUT GENUINO 2025-2026: n=98  wr=35.7%  ret=-17.6%  mdd=-26.2%
    MC i.i.d.  : pp=0.121  pr=0.001
    MC block   : pp=0.125  pr=0.002

══════════════════════════════════════════════════════════════════════════════
RR = 3.0
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=403  wr=32.8% (p=1.0000 vs 50%)  ret=-55.4%  mdd=-74.8%
    Exit: TP=73  SL=241  time=89
    MC i.i.d.  : pp=0.077  pr=0.582
    MC block   : pp=0.084  pr=0.573

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      53     +3.5%  35.8%
      2021      74    -13.9%  29.7%
      2022      68    -18.7%  33.8%
      2023      51     -6.1%  33.3%
      2024      59     -8.0%  33.9%
      2025      63    -24.4%  25.4%
      2026      35    +12.2%  42.9%

  HOLDOUT GENUINO 2025-2026: n=98  wr=31.6%  ret=-12.2%  mdd=-30.0%
    MC i.i.d.  : pp=0.267  pr=0.003
    MC block   : pp=0.280  pr=0.008

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=2 (griglia RR)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    2.0     403    -63.4%       -2.443    0.000
    3.0     403    -55.4%       -1.803    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    2.0      98    -17.6%       -1.342    0.000
    3.0      98    -12.2%       -0.773    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage-stress — RR migliore per DSR = 3.0
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample     403    -55.4%  32.8%   0.077   0.582
        0bps     holdout      98    -12.2%  31.6%   0.267   0.003
        2bps  full-sample     403    -79.9%  32.5%   0.141   0.739
        2bps     holdout      98    -20.3%  31.6%   0.152   0.015
        5bps  full-sample     403   -116.7%  32.3%   0.000   1.000
        5bps     holdout      98    -32.3%  31.6%   0.059   0.103
       10bps  full-sample     403   -178.0%  31.5%   0.000   1.000
       10bps     holdout      98    -52.5%  29.6%   0.009   0.567

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: fallisce, e con un profilo di rischio peggiore delle altre bocciature

```
RR    n(full)  WR full   Ret full   MC p_ruin full   n(hold)  Ret hold   MC p_ruin hold
2.0     403     37.5%     -63.4%        0.753           98      -17.6%       0.001
3.0     403     32.8%     -55.4%        0.582           98      -12.2%       0.003
```

**DSR=0.000 in entrambi gli ambiti, entrambi gli RR** — nessuna
sopravvivenza alla correzione. Il win rate (32.8-37.5%) è ben sotto il
50% e, a differenza della VP+VWAP Confluence (dove un WR basso ma
strutturalmente stabile era compensato da un RR asimmetrico ben calibrato
alla reale estensione del movimento), qui **allargare RR da 2 a 3 non
inverte il segno** — passa da -63.4% a -55.4% (full-sample), un
miglioramento marginale, non un'inversione: il pattern non ha
un'estensione naturale che un RR più largo riesce a catturare meglio.

**Nota distintiva**: il MC p_ruin full-sample è **notevolmente più alto**
di quasi tutte le altre strategie bocciate in questa sessione (0.582-0.753
qui, contro valori tipicamente vicini a 0.000-0.05 per altri fallimenti
come Asia-sweep, VWAP bounce, order-flow absorption). Questo non è solo
"non funziona" — è "funziona male E con un profilo di rischio
concretamente pericoloso" nelle simulazioni Monte Carlo sull'intero
campione (il quadro migliora molto sull'holdout più recente, p_ruin
0.001-0.003, ma il pattern è comunque in perdita anche lì).

**Perché fallisce, verosimilmente**: la regola impone una catena di 4
condizioni tecniche sequenziali (tocco FVG 4H → reazione aggressiva →
nuovo FVG 15m → retest) — un filtro molto selettivo (2.136 FVG 4H
rilevati, solo 403 pattern completi, ~19% tasso di completamento) che
however elaborate non produce di per sé un'informazione direzionale
reale. Coerente con **tutte** le altre strategie di stampo ICT/SMC
testate in questa sessione (Asia-sweep MSS v1-v4, VWAP bounce, order-flow
absorption): l'elaborazione tecnica di un pattern grafico multi-timeframe,
per quanto elegante nella sua logica narrativa, non equivale a un edge
statistico su BTCUSDT — l'unica eccezione trovata finora (VP+VWAP
Confluence) ha funzionato non per la sofisticazione del pattern, ma per
un allineamento fortuito/strutturale tra il target RR scelto e l'effettiva
distribuzione dei movimenti dopo l'evento.

**Nota sulla fonte**: la regola è una claim social-media non attribuita,
senza track record verificabile allegato (il post termina con un invito a
un "programma di mentorship" — un modello di business che non richiede
che la regola funzioni per generare ricavi, un conflitto di interessi
strutturale comune a questo genere di contenuti). Il test qui condotto
non dipende dalla fonte per la sua validità — è stato valutato
esclusivamente sui suoi meriti quantitativi, che risultano negativi.
