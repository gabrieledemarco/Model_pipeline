# NY ORB + Volume Profile Absorption — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
NY ORB + Volume Profile Absorption — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[EVENTS] 2373 giorni scansionati -> 1735 trade candidati
  scartati: no-range=0  no-breakout=45  no-absorption=557  no-target=36

  Holdout genuino 2025-2026: 402 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.0 : 1   (stop = target_dist / 1.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1735  wr=47.9% (BE_teorico=50.0%, p=0.9622)  ret=-178.2%  mdd=-177.8%
    Exit: TP=414  SL=469  time=852
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     261    -26.0%  47.9%
      2021     279     -7.1%  52.0%
      2022     264    -34.1%  47.0%
      2023     270    -47.0%  45.9%
      2024     259    -10.2%  51.4%
      2025     262    -36.8%  44.3%
      2026     140    -16.9%  45.7%

  HOLDOUT 2025-2026: n=402  wr=44.8%  ret=-53.7%  mdd=-54.2%
    MC i.i.d.  : pp=0.001  pr=0.629
    MC block   : pp=0.000  pr=0.652

══════════════════════════════════════════════════════════════════════════════
RR = 2.0 : 1   (stop = target_dist / 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1735  wr=40.2% (BE_teorico=33.3%, p=0.0000)  ret=-262.9%  mdd=-262.9%
    Exit: TP=323  SL=798  time=614
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     261    -32.0%  42.5%
      2021     279    -10.2%  44.8%
      2022     264    -44.1%  38.6%
      2023     270    -67.9%  38.1%
      2024     259    -26.7%  42.1%
      2025     262    -62.8%  35.1%
      2026     140    -19.1%  39.3%

  HOLDOUT 2025-2026: n=402  wr=36.6%  ret=-81.9%  mdd=-82.2%
    MC i.i.d.  : pp=0.003  pr=0.954
    MC block   : pp=0.000  pr=0.990

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=2 (RR 1:1 vs RR 2:1)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0    1735   -178.2%       -5.592    0.000
    2.0    1735   -262.9%       -5.446    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0     402    -53.7%       -3.433    0.000
    2.0     402    -81.9%       -3.499    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 1.0:1
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample    1735   -178.2%  47.9%  -177.8%   0.000   1.000
        0bps     holdout     402    -53.7%  44.8%   -54.2%   0.001   0.629
        2bps  full-sample    1735   -256.6%  46.3%  -256.2%   0.000   1.000
        2bps     holdout     402    -76.0%  43.3%   -76.2%   0.000   0.992
        5bps  full-sample    1735   -374.3%  44.6%  -374.3%   0.000   1.000
        5bps     holdout     402   -109.5%  41.5%  -109.4%   0.000   1.000
       10bps  full-sample    1735   -570.3%  41.0%  -570.3%   0.000   1.000
       10bps     holdout     402   -165.3%  38.3%  -164.9%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Diagnostica post-hoc: la perdita è da fee/leva degenerata o da assenza di edge?

Dato il precedente della sessione (stop troppo stretti → fee che divorano il rischio, come
nel caso ADP/ICT Silver Bullet prima del fix), è stata verificata la distribuzione di
`target_dist` (distanza dal pivot 1H usato come target) e del rapporto fee/rischio per
escludere un artefatto di sizing prima di concludere "nessun edge":

```
target_dist (in unità di ATR_1H):  mediana=2.19×ATR   p5=0.49×ATR   p95=6.78×ATR
target_dist (% del prezzo):        mediana=1.64%       p5=0.27%      p95=7.26%

RR=1.0 — fee/rischio: mediana=4.9%   p90=18.7%   p99=80.0%   leva mediana=0.61×  (1.6% dei trade al cap 10×)
RR=2.0 — fee/rischio: mediana=9.8%   p90=37.4%   p99=80.0%   leva mediana=1.22×  (3.6% dei trade al cap 10×)
```

Le distanze target sono normali (non degeneri) e il fee consuma solo il 5-10% del rischio
per trade nella mediana — molto lontano dal 80-132% osservato nei casi di vera distruzione
da fee di questa sessione. **Il risultato negativo non è un artefatto di sizing: la logica
"breakout + ritorno al POC = continuazione" non ha edge direzionale reale su BTCUSDT 1H/5m
2020-2026** — fallisce in tutti e 7 gli anni, su entrambi gli RR testati, con o senza
slippage aggiuntivo.
