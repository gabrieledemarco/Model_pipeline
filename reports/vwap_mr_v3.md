# VWAP Mean-Reversion v3 — SL ATR-multiple IS-scan — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VWAP Mean-Reversion v3 — SL ATR-multiple IS-scan — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[IS-SCAN] Parametri selezionati per finestra (35 finestre valide):
  Z_ENTRY più scelto: {2.0: 30, 1.5: 3, 1.0: 2}
  Filtro più scelto  : {'SIDEWAYS_ONLY': 23, 'NO_FILTER': 12}
  SL_ATR_MULT più scelto: {2.0: 21, 1.5: 6, 1.0: 3, 0.75: 2, 0.5: 2, 1.25: 1}

══════════════════════════════════════════════════════════════════════════════
RISULTATI — walk-forward OOS aggregato
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=3665  wr=60.4%  ret=-232.2%  mdd=-231.5%
    Exit: TP=2026  SL=859  time=780
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370    -29.1%  60.8%
      2021    1255    -54.2%  62.2%
      2022     766    -46.3%  60.7%
      2023     362    -26.1%  62.2%
      2024     419    -38.5%  51.8%
      2025     356    -25.5%  61.0%
      2026     137    -12.5%  61.3%

  HOLDOUT GENUINO 2025-2026: n=493  wr=61.1%  ret=-38.0%  mdd=-40.3%
    MC i.i.d.  : pp=0.004  pr=0.123
    MC block   : pp=0.002  pr=0.110

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample    3665   -232.2%  60.4%  -231.5%   0.000   1.000
        0bps     holdout     493    -38.0%  61.1%   -40.3%   0.004   0.123
        2bps  full-sample    3665   -348.6%  57.8%  -347.2%   0.000   1.000
        2bps     holdout     493    -58.8%  58.0%   -59.7%   0.000   0.802
        5bps  full-sample    3665   -523.2%  53.0%  -523.1%   0.000   1.000
        5bps     holdout     493    -90.1%  50.3%   -90.5%   0.000   1.000
       10bps  full-sample    3665   -814.2%  45.8%  -814.4%   0.000   1.000
       10bps     holdout     493   -142.1%  39.8%  -142.2%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: aggiungere un asse di scan ha peggiorato il risultato — overfitting, non soluzione

```
                    v1 (SL fisso 2.0xATR)   v3 (SL scansionato, grid 0.5-2.0)
Full-sample ret          -182.7%                    -232.2%
Holdout ret                -27.1%                     -38.0%
```

L'IS-scan sceglie SL_ATR_MULT=2.0 (il valore MASSIMO della griglia testata)
in 21/35 finestre — la maggioranza — confermando ancora una volta che lo
stop largo è ciò che preserva il win rate reale. Ma il risultato aggregato
OOS è **peggiore** di v1 (che fissava semplicemente 2.0 a priori), non
migliore: aggiungere un terzo asse libero alla selezione (36 combinazioni
per finestra invece di 6) ha introdotto overfitting sulle finestre IS più
piccole — alcune finestre hanno solo poche centinaia di eventi IS, e
scegliere il migliore tra 36 combinazioni su un campione così piccolo
premia il rumore, non il segnale. Lezione: più gradi di libertà nella
selezione causale non è automaticamente meglio, specialmente quando i dati
IS per finestra sono limitati — lo stesso principio di parsimonia già
osservato nel resto della sessione (grid piccole > grid grandi) vale anche
per un IS-scan "onesto" (causale, non un pick a posteriori).

**Prossimo tentativo (v4, non ancora eseguito)**: invece di uno stop a
prezzo come uscita primaria (che soffre comunque dell'asimmetria di
first-passage-time in qualche forma, largo o stretto), applicare la
ricetta che ha funzionato per la strategia ML RandomForest 8h — uscita A
TEMPO FISSO come meccanismo primario, con un solo stop di sicurezza molto
largo (fisso, non scansionato) per la protezione da eventi estremi. Questo
scorpora completamente l'uscita dal problema del "quale distanza di prezzo
usare", sostituendolo con "quante ore aspettare" — un asse diverso, mai
provato su questa strategia.
