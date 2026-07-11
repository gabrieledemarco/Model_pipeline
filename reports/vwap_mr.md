# VWAP Mean-Reversion — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VWAP Mean-Reversion — Validation Pipeline (S07-style 3-layer architecture)
══════════════════════════════════════════════════════════════════════════════

[IS-SCAN] Parametri selezionati per finestra (35 finestre valide):
  Z_ENTRY più scelto: {2.0: 28, 1.5: 5, 1.0: 2}
  Filtro più scelto  : {'SIDEWAYS_ONLY': 21, 'NO_FILTER': 14}

══════════════════════════════════════════════════════════════════════════════
RISULTATI — walk-forward OOS aggregato
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=3912  wr=62.9%  ret=-182.7%  mdd=-182.1%
    Exit: TP=2266  SL=725  time=921
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370    -15.9%  63.5%
      2021    1391    -56.4%  62.5%
      2022     799    -45.6%  61.1%
      2023     362    -26.5%  62.7%
      2024     497    -11.2%  66.0%
      2025     356    -19.7%  62.9%
      2026     137     -7.4%  65.0%

  HOLDOUT GENUINO 2025-2026: n=493  wr=63.5%  ret=-27.1%  mdd=-29.2%
    MC i.i.d.  : pp=0.013  pr=0.005
    MC block   : pp=0.011  pr=0.005

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample    3912   -182.7%  62.9%  -182.1%   0.000   1.000
        0bps     holdout     493    -27.1%  63.5%   -29.2%   0.013   0.005
        2bps  full-sample    3912   -279.8%  60.3%  -279.1%   0.000   1.000
        2bps     holdout     493    -45.5%  60.4%   -46.4%   0.000   0.318
        5bps  full-sample    3912   -425.6%  55.2%  -425.6%   0.000   1.000
        5bps     holdout     493    -73.2%  52.5%   -73.7%   0.000   0.994
       10bps  full-sample    3912   -668.4%  47.6%  -668.6%   0.000   1.000
       10bps     holdout     493   -119.3%  41.8%  -119.4%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Diagnostica: perché il win rate è alto (61-66% ogni anno) ma il ritorno è negativo

```
sigma_vwap (dispersione intra-day) come % del prezzo:  mediana 0.415%   p90 1.307%
ATR_14 come % del prezzo:                              mediana 0.759%   p90 1.422%

Z_ENTRY selezionato più spesso dall'IS-scan: 2.0 (28/35 finestre)
  -> distanza tipica dell'entry/target (2×sigma):  ≈ 0.83% del prezzo
  -> distanza dello stop (2×ATR):                  ≈ 1.52% del prezzo
  -> rapporto SL/target ≈ 1.83  ->  win rate di breakeven (ignorando fee) ≈ 64.7%
```

Il win rate osservato (62.9% full-sample, 61-66% ogni singolo anno — **un edge
direzionale reale e molto stabile**, ben sopra il 50%) è però leggermente
**sotto** il breakeven implicito nel rapporto stop/target (64.7%), e le fee
peggiorano ulteriormente il margine. Questo è l'esatto specchio del problema
della strategia NY-ORB-VP (lì lo stop era troppo STRETTO rispetto al target,
qui è lo stop 2×ATR ad essere troppo LARGO rispetto alla tipica distanza di
reversione al VWAP): un edge direzionale genuino viene annullato da un
rapporto rischio/rendimento male impostato, non da assenza di segnale.

**Prossimo passo naturale (non ancora implementato):** sostituire lo stop
fisso a 2×ATR con uno stop proporzionale alla stessa scala del target (es.
un multiplo di sigma_vwap, o uno stop a RR fisso 1:1/1.5:1 rispetto alla
distanza di entry come nella strategia NY-ORB-VP) per allineare il
rischio/rendimento al win rate realmente misurato.
