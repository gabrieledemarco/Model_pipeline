# VWAP Bounce (rimbalzo in direzione del trend) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VWAP Bounce (rimbalzo in direzione del trend) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Diagnosi del fallimento della replica letterale (flip ad ogni cross):
overtrading, ~4.6 trade/giorno, negativo ogni anno. Questa variante entra
solo quando il prezzo RITORNA sul VWAP e RIMBALZA nella direzione del
trend già stabilito, non ad ogni attraversamento.

══════════════════════════════════════════════════════════════════════════════
RISULTATI
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1979  wr=35.6% (p=1.0000 vs 50%)  ret=-120.9%  mdd=-113.4%
    Exit: SL=999  time(fine giornata)=980
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     298    +42.6%  37.9%
      2021     320     -1.9%  40.0%
      2022     297    -33.5%  35.7%
      2023     300    -66.2%  31.0%
      2024     305    -13.3%  36.4%
      2025     304    -18.9%  34.9%
      2026     155    -29.6%  30.3%

  HOLDOUT GENUINO 2025-2026: n=459  wr=33.3%  ret=-48.5%  mdd=-57.2%
    MC i.i.d.  : pp=0.044  pr=0.488
    MC block   : pp=0.032  pr=0.470

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    1979   -120.9%  35.6%   0.000   1.000
        0bps     holdout     459    -48.5%  33.3%   0.044   0.488
        2bps  full-sample    1979   -198.6%  34.3%   0.000   1.000
        2bps     holdout     459    -70.4%  31.2%   0.008   0.858
        5bps  full-sample    1979   -315.1%  33.0%   0.000   1.000
        5bps     holdout     459   -103.2%  30.7%   0.007   0.993
       10bps  full-sample    1979   -509.3%  31.0%   0.000   1.000
       10bps     holdout     459   -157.8%  30.1%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: il filtro anti-whipsaw funziona, ma il segnale non c'è

```
                      Flip letterale (replica paper)   Bounce (questa variante)
N trade                    10.938                            1.979
Trade/giorno                  4,6                             0,83
Win rate full-sample         44,3%                            35,6%
Ret full-sample            -1008,7%                          -120,9%
Ret holdout                  -274,2%                           -48,5%
```

Il filtro anti-whipsaw ha funzionato esattamente come previsto: la
frequenza scende da 4,6 a 0,83 trade/giorno (-82%), un livello di
selettività ragionevole per un vero setup "di qualità". **Ma il win rate
peggiora ulteriormente** (35,6% vs 44,3%) invece di migliorare — il
rimbalzo sul VWAP, con questa definizione, non ha edge direzionale
positivo su BTCUSDT 1H, anzi è più sbagliato della semplice regola
"sopra/sotto".

**Diagnostica**: 999 uscite su 999+980=1979 (50,5%) sono via stop-loss.
Con uno stop fisso a 1,5×ATR e un'uscita a tempo che può restare aperta
fino a fine giornata (anche molte ore), lo stop ha molto tempo per essere
toccato dal rumore indipendentemente dalla bontà della tesi di
continuazione — la stessa asimmetria di first-passage-time diagnosticata
più volte in questa sessione (stop troppo vicino rispetto alla durata
della posizione, tocca il rumore prima che il vero movimento si esprima).
Non è possibile distinguere da questi soli dati se il segnale di
rimbalzo sia realmente inesistente, o se sia annegato dallo stesso
problema strutturale già visto altrove.

**Nota positiva parziale**: il 2020 è isolatamente positivo (+42,6%, pur
con WR 37,9% sotto il 50%, quindi trainato da pochi vincitori grandi) —
un pattern coerente con un vero edge di continuazione occasionalmente
presente ma raro, non con puro rumore.

**Prossimo passo naturale (non eseguito)**: allargare lo stop (es. 3×ATR,
come nel safety-stop validato altrove in questa famiglia di strategie) per
isolare se il problema sia la tesi del rimbalzo o la costruzione dello
stop — replicando lo stesso approccio diagnostico già usato con successo
sulla VWAP mean-reversion.
