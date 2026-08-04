# VWAP Mean-Reversion v4 — fixed time-based exit (ML8h recipe) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VWAP Mean-Reversion v4 — fixed time-based exit (ML8h recipe) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[IS-SCAN] Parametri selezionati per finestra (35 finestre valide):
  Z_ENTRY più scelto: {2.0: 32, 1.5: 3}
  Filtro più scelto  : {'SIDEWAYS_ONLY': 24, 'NO_FILTER': 11}
  hold_hours più scelto: {8: 19, 4: 10, 2: 6}

══════════════════════════════════════════════════════════════════════════════
RISULTATI — walk-forward OOS aggregato
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=3017  wr=48.2%  ret=-138.5%  mdd=-138.9%
    Exit: safety-SL=286  time=2731
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370    -36.8%  43.5%
      2021     712    -31.8%  48.2%
      2022     710    -31.4%  48.0%
      2023     313    -13.5%  49.5%
      2024     419     -8.6%  52.0%
      2025     356    -14.1%  47.2%
      2026     137     -2.3%  50.4%

  HOLDOUT GENUINO 2025-2026: n=493  wr=48.1%  ret=-16.4%  mdd=-23.2%
    MC i.i.d.  : pp=0.104  pr=0.000
    MC block   : pp=0.095  pr=0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample    3017   -138.5%  48.2%  -138.9%   0.000   1.000
        0bps     holdout     493    -16.4%  48.1%   -23.2%   0.104   0.000
        2bps  full-sample    3017   -193.2%  46.7%  -192.0%   0.000   1.000
        2bps     holdout     493    -28.7%  45.8%   -33.6%   0.017   0.013
        5bps  full-sample    3017   -275.4%  43.1%  -273.0%   0.000   1.000
        5bps     holdout     493    -47.2%  40.8%   -49.6%   0.001   0.395
       10bps  full-sample    3017   -412.2%  38.1%  -410.0%   0.000   1.000
       10bps     holdout     493    -77.9%  34.9%   -78.3%   0.000   0.996

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: la ricetta ML8h qui distrugge l'edge invece di salvarlo

```
                    v1 (touch VWAP, SL=2xATR)   v4 (tempo fisso, SL=3xATR safety)
Win rate full-sample       62.9%                        48.2%
Ret full-sample            -182.7%                       -138.5%
Ret holdout                  -27.1%                        -16.4%
```

Il win rate crolla da 62.9% a 48.2% — sotto il 50%, un coin-flip. Questo
rivela qualcosa di importante sulla NATURA dell'edge originale: il 61-66%
di v1 non era un vantaggio "la direzione è giusta il più delle volte nelle
prossime N ore" (un edge di TIMING/classificazione, come quello della
strategia ML RandomForest 8h) — era specificamente "il prezzo tende a
ritoccare il VWAP prima o poi" (un edge di LIVELLO). Rimuovendo il tocco
del VWAP come condizione di vittoria e sostituendolo con un'uscita a tempo
fisso, si è ridefinito cosa significa "vincere" — e la nuova definizione
(prezzo oltre l'entry dopo N ore) non è ciò che il segnale VWAP realmente
predice. La ricetta che ha funzionato per ML8h (un edge di classificazione
diretta) non si trasferisce a un edge di mean-reversion verso un livello:
sono due tipi diversi di segnale e richiedono uscite concettualmente
diverse.

## Sintesi complessiva dei 4 tentativi

```
v1  stop largo fisso (2xATR), target=tocco VWAP        -> WR reale (62.9%) ma R:R cattivo
v2  stop stretto proporzionale al target (RR 1/1.5/2)   -> R:R sistemato ma WR crolla (rumore)
v3  scan causale del moltiplicatore ATR (0.5-2.0)       -> conferma "largo è meglio" ma overfit, peggio di v1
v4  uscita a tempo fisso (ricetta ML8h) + safety-stop   -> WR crolla a coin-flip: cambia la natura dell'edge
```

Quattro costruzioni diverse del trade, tutte fallite, ciascuna per un
motivo diverso e informativo. Il pattern che emerge è strutturale, non un
problema di calibrazione: l'edge (win rate 61-66% stabile) esiste
SOLTANTO quando la vittoria è definita esattamente come "il prezzo tocca
di nuovo il VWAP" — qualunque tentativo di renderlo tradeable (stop più
stretto per il R:R, o un'uscita diversa dal tocco) altera la definizione
di vittoria e fa sparire l'edge. Un'unica leva non ancora tentata: invece
di modificare lo STOP o l'USCITA, allargare il TARGET oltre il semplice
tocco del VWAP (es. puntare a un livello oltre il VWAP, come la banda
opposta), lasciando correre i vincitori più a lungo per migliorare la
reward mantenendo intatta la condizione di vittoria originale — non
ancora testato, ma richiede un'altra iterazione prima di sapere se
funziona.
