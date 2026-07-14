# Asia Range Sweep + MSS + FVG/OB Entry v2 (rejection-filtered sweep) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Asia Range Sweep + MSS + FVG/OB Entry v2 (rejection-filtered sweep) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[EVENTS] 2373 giorni scansionati -> 281 trade candidati
  scartati: no-range=0  no-sweep=817  no-mss=919  no-entry(fvg/ob)=85  bad-target=271

  Holdout genuino 2025-2026: 69 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.0 : 1   (stop = target_dist / 1.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=281  wr=48.8% (BE_teorico=50.0%, p=0.6834)  ret=-69.5%  mdd=-72.2%
    Exit: TP=137  SL=142  time=2
    MC i.i.d.  : pp=0.000  pr=0.941
    MC block   : pp=0.000  pr=0.964

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      33     -5.7%  51.5%
      2021      40     -9.7%  45.0%
      2022      38     -9.9%  44.7%
      2023      49    -14.1%  49.0%
      2024      52    -12.3%  50.0%
      2025      51    -17.1%  47.1%
      2026      18     -0.7%  61.1%

  HOLDOUT 2025-2026: n=69  wr=50.7%  ret=-17.8%  mdd=-20.2%
    MC i.i.d.  : pp=0.023  pr=0.000
    MC block   : pp=0.001  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.0 : 1   (stop = target_dist / 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=281  wr=36.7% (BE_teorico=33.3%, p=0.1322)  ret=-73.0%  mdd=-78.9%
    Exit: TP=104  SL=176  time=1
    MC i.i.d.  : pp=0.003  pr=0.901
    MC block   : pp=0.003  pr=0.912

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      33    -10.9%  33.3%
      2021      40     -5.6%  35.0%
      2022      38    -14.3%  31.6%
      2023      49    -20.5%  34.7%
      2024      52     +1.0%  48.1%
      2025      51    -25.7%  29.4%
      2026      18     +3.1%  50.0%

  HOLDOUT 2025-2026: n=69  wr=34.8%  ret=-22.6%  mdd=-28.1%
    MC i.i.d.  : pp=0.036  pr=0.001
    MC block   : pp=0.019  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 3.0 : 1   (stop = target_dist / 3.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=281  wr=31.0% (BE_teorico=25.0%, p=0.0140)  ret=-65.7%  mdd=-70.0%
    Exit: TP=88  SL=192  time=1
    MC i.i.d.  : pp=0.013  pr=0.787
    MC block   : pp=0.010  pr=0.800

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      33     -8.1%  30.3%
      2021      40     +1.8%  32.5%
      2022      38    -17.7%  23.7%
      2023      49    -27.1%  26.5%
      2024      52    +10.1%  44.2%
      2025      51    -23.3%  25.5%
      2026      18     -1.4%  33.3%

  HOLDOUT 2025-2026: n=69  wr=27.5%  ret=-24.7%  mdd=-27.2%
    MC i.i.d.  : pp=0.044  pr=0.003
    MC block   : pp=0.005  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (RR 1:1, 2:1, 3:1)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0     281    -69.5%       -4.078    0.000
    2.0     281    -73.0%       -3.145    0.000
    3.0     281    -65.7%       -2.309    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0      69    -17.8%       -2.034    0.000
    2.0      69    -22.6%       -1.936    0.000
    3.0      69    -24.7%       -1.815    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 3.0:1
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample     281    -65.7%  31.0%   -70.0%   0.013   0.787
        0bps     holdout      69    -24.7%  27.5%   -27.2%   0.044   0.003
        2bps  full-sample     281   -135.1%  28.8%  -136.5%   0.000   1.000
        2bps     holdout      69    -44.2%  24.6%   -46.5%   0.003   0.290
        5bps  full-sample     281   -239.3%  25.6%  -241.0%   0.000   1.000
        5bps     holdout      69    -73.4%  24.6%   -75.4%   0.000   0.977
       10bps  full-sample     281   -412.8%  23.5%  -414.3%   0.000   1.000
       10bps     holdout      69   -122.2%  21.7%  -123.6%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Confronto v1 (qualsiasi close-back) vs v2 (rigetto: wick > body)

```
                    v1 (n=362)      v2 (n=281, -22% eventi)
RR=1  wr            46.7%           48.8%
RR=1  ret            -104.7%          -69.5%
RR=1  MC p_profit     0.004            0.000
RR=1  DSR              0.000            0.000
```

Il filtro di rigetto migliora leggermente il win rate e riduce la perdita
(-104.7% → -69.5%), ma **il segnale resta chiaramente sotto il breakeven in
tutti e 7 gli anni e su tutti e 3 gli RR testati**, e la correzione DSR
resta 0.000 per l'intera famiglia. La qualità del rigetto (coda più lunga
del corpo) rende lo sweep più "pulito" come pattern candlestick, ma non
introduce un edge direzionale sufficiente a rendere profittevole la
sequenza sweep→MSS→FVG/OB con target fisso all'estremo opposto del range.
