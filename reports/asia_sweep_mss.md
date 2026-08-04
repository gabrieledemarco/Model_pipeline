# Asia Range Sweep + MSS + FVG/OB Entry — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Asia Range Sweep + MSS + FVG/OB Entry — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[EVENTS] 2373 giorni scansionati -> 362 trade candidati
  scartati: no-range=0  no-sweep=425  no-mss=1123  no-entry(fvg/ob)=97  bad-target=366

  Holdout genuino 2025-2026: 85 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.0 : 1   (stop = target_dist / 1.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=362  wr=46.7% (BE_teorico=50.0%, p=0.9056)  ret=-104.7%  mdd=-106.6%
    Exit: TP=170  SL=190  time=2
    MC i.i.d.  : pp=0.004  pr=0.996
    MC block   : pp=0.021  pr=0.976

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      48    -11.5%  47.9%
      2021      57     -7.2%  50.9%
      2022      48    -17.1%  41.7%
      2023      62    -33.6%  37.1%
      2024      62    -11.0%  53.2%
      2025      61    -19.6%  45.9%
      2026      24     -4.8%  54.2%

  HOLDOUT 2025-2026: n=85  wr=48.2%  ret=-24.4%  mdd=-26.5%
    MC i.i.d.  : pp=0.006  pr=0.000
    MC block   : pp=0.004  pr=0.001

══════════════════════════════════════════════════════════════════════════════
RR = 2.0 : 1   (stop = target_dist / 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=362  wr=35.6% (BE_teorico=33.3%, p=0.1908)  ret=-105.8%  mdd=-109.4%
    Exit: TP=130  SL=231  time=1
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.036  pr=0.954

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      48    -20.1%  31.2%
      2021      57     -4.7%  36.8%
      2022      48    -24.2%  27.1%
      2023      62    -33.5%  30.6%
      2024      62     +0.3%  48.4%
      2025      61    -20.6%  34.4%
      2026      24     -3.0%  41.7%

  HOLDOUT 2025-2026: n=85  wr=36.5%  ret=-23.7%  mdd=-27.5%
    MC i.i.d.  : pp=0.044  pr=0.003
    MC block   : pp=0.042  pr=0.004

══════════════════════════════════════════════════════════════════════════════
RR = 3.0 : 1   (stop = target_dist / 3.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=362  wr=30.7% (BE_teorico=25.0%, p=0.0086)  ret=-92.5%  mdd=-95.1%
    Exit: TP=112  SL=249  time=1
    MC i.i.d.  : pp=0.026  pr=0.924
    MC block   : pp=0.020  pr=0.932

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      48    -18.3%  27.1%
      2021      57     +0.3%  31.6%
      2022      48    -29.5%  20.8%
      2023      62    -31.2%  27.4%
      2024      62     +7.7%  43.5%
      2025      61    -13.7%  31.1%
      2026      24     -7.9%  29.2%

  HOLDOUT 2025-2026: n=85  wr=30.6%  ret=-21.6%  mdd=-24.1%
    MC i.i.d.  : pp=0.086  pr=0.003
    MC block   : pp=0.078  pr=0.003

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (RR 1:1, 2:1, 3:1)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0     362   -104.7%       -5.384    0.000
    2.0     362   -105.8%       -4.024    0.000
    3.0     362    -92.5%       -2.894    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0      85    -24.4%       -2.525    0.000
    2.0      85    -23.7%       -1.791    0.000
    3.0      85    -21.6%       -1.366    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 3.0:1
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample     362    -92.5%  30.7%   -95.1%   0.026   0.924
        0bps     holdout      85    -21.6%  30.6%   -24.1%   0.086   0.003
        2bps  full-sample     362   -182.1%  28.2%  -184.2%   0.000   1.000
        2bps     holdout      85    -44.8%  28.2%   -47.2%   0.004   0.330
        5bps  full-sample     362   -316.5%  25.1%  -318.5%   0.000   1.000
        5bps     holdout      85    -79.7%  27.1%   -81.7%   0.000   0.993
       10bps  full-sample     362   -540.5%  22.7%  -541.9%   0.000   1.000
       10bps     holdout      85   -137.9%  24.7%  -139.3%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Diagnostica: quanto pesa la distruzione da fee/leva rispetto alla mancanza di edge

L'entry avviene DOPO che sweep + MSS + gamba d'impulso hanno già consumato
una parte del movimento verso il target — quindi `target_dist` (distanza
residua entry→target) è spesso piccola rispetto all'ATR:

```
target_dist (in unità di ATR_1H):  mediana 0.72×ATR   p25 0.30×ATR   p5 0.09×ATR
% trade su FVG vs Order Block:     72% FVG / 28% OB

RR=1.0 — fee/rischio: mediana 15.8%   p90 80.0%   leva mediana 1.98×  (10.5% dei trade al cap 10×)
RR=2.0 — fee/rischio: mediana 31.7%   p90 80.0%   leva mediana 3.96×  (24.0% dei trade al cap 10×)
RR=3.0 — fee/rischio: mediana 47.5%   p90 80.0%   leva mediana 5.94×  (32.9% dei trade al cap 10×)
```

A differenza delle due strategie precedenti (NY-ORB-VP, VWAP-MR), qui la
distruzione da fee/leva **non è trascurabile**, specialmente a RR=2 e RR=3
dove mediana 32-48% del rischio nominale viene eroso dalle fee e un
quarto/un terzo dei trade tocca il cap di leva 10× (il che significa che il
rischio reale supera l'1% nominale per quei trade). Questo è un effetto
collaterale strutturale della scelta di entry tardiva (dopo la conferma
MSS): la finestra di edge residua fino al target si è già ristretta.

Tuttavia **anche a RR=1:1 (il meno contaminato)**, dove fee/rischio mediano
è un più contenuto 15.8% e solo il 10.5% dei trade tocca il cap di leva, il
risultato resta chiaramente negativo: wr=46.7% contro un breakeven teorico
del 50% (p=0.9056, cioè nessuna evidenza statistica di edge sopra il caso),
ret=-104.7%, MC p_profit=0.004. La componente di fee/leva peggiora
ulteriormente RR=2 e RR=3, ma non è la causa primaria del fallimento: la
sequenza sweep→MSS→FVG/OB, con questo target fisso, non produce un edge
direzionale sufficiente nemmeno nella sua variante meno penalizzata dal
sizing.
