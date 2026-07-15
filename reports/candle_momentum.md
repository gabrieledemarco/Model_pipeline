# Candle Momentum (green/red, volume filter, 1-15min hold) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Candle Momentum (green/red, volume filter, 1-15min hold) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════════════
HOLD = 1 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1133402  wr=9.7% (p=1.0000 vs 50%)  ret=-90753.5%  mdd=-90753.5%  mean/trade=-0.0801%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020    175236  -14123.3%  10.4%
      2021    175137  -14082.8%  17.9%
      2022    174488  -13861.0%  10.7%
      2023    173152  -13792.1%   5.1%
      2024    175109  -14095.9%   8.7%
      2025    173899  -13884.5%   6.4%
      2026     86381  -6914.0%   7.4%

  HOLDOUT 2025-2026: n=260280  wr=6.7%  ret=-20798.5%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 1 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=599602  wr=10.5% (p=1.0000 vs 50%)  ret=-48038.2%  mdd=-48038.2%  mean/trade=-0.0801%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     95481  -7680.0%  11.0%
      2021     92660  -7409.1%  19.3%
      2022     90510  -7118.2%  12.0%
      2023     90893  -7220.4%   5.8%
      2024     92099  -7485.2%   9.3%
      2025     92875  -7479.9%   6.8%
      2026     45084  -3645.5%   8.0%

  HOLDOUT 2025-2026: n=137959  wr=7.2%  ret=-11125.4%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 5 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=487059  wr=22.1% (p=1.0000 vs 50%)  ret=-40072.2%  mdd=-40072.2%  mean/trade=-0.0823%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     75205  -6357.4%  23.2%
      2021     75071  -6318.1%  31.3%
      2022     74957  -6104.3%  23.1%
      2023     74670  -6086.6%  15.3%
      2024     75176  -6118.2%  22.4%
      2025     74837  -6038.2%  18.9%
      2026     37143  -3049.4%  19.2%

  HOLDOUT 2025-2026: n=111980  wr=19.0%  ret=-9087.7%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 5 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=339422  wr=21.7% (p=1.0000 vs 50%)  ret=-28263.0%  mdd=-28263.0%  mean/trade=-0.0833%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     53685  -4634.3%  22.4%
      2021     52445  -4518.2%  30.7%
      2022     51543  -4154.4%  23.1%
      2023     51532  -4231.7%  14.9%
      2024     52055  -4331.9%  21.8%
      2025     52439  -4266.2%  18.4%
      2026     25723  -2126.3%  19.2%

  HOLDOUT 2025-2026: n=78162  wr=18.6%  ret=-6392.5%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 10 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=284372  wr=27.8% (p=1.0000 vs 50%)  ret=-23623.5%  mdd=-23623.5%  mean/trade=-0.0831%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     43891  -3847.8%  28.8%
      2021     43793  -3628.9%  35.7%
      2022     43753  -3636.7%  28.8%
      2023     43653  -3533.0%  21.4%
      2024     43878  -3639.5%  28.5%
      2025     43716  -3608.6%  24.8%
      2026     21688  -1729.0%  26.0%

  HOLDOUT 2025-2026: n=65404  wr=25.2%  ret=-5337.6%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 10 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=224600  wr=27.1% (p=1.0000 vs 50%)  ret=-19004.8%  mdd=-19004.8%  mean/trade=-0.0846%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     35270  -3078.2%  27.7%
      2021     34652  -3019.0%  34.7%
      2022     34267  -2910.2%  28.0%
      2023     34177  -2773.5%  20.8%
      2024     34599  -2915.3%  27.9%
      2025     34629  -2907.4%  24.3%
      2026     17006  -1401.1%  25.4%

  HOLDOUT 2025-2026: n=51635  wr=24.7%  ret=-4308.6%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 15 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=200817  wr=31.1% (p=1.0000 vs 50%)  ret=-16598.7%  mdd=-16587.1%  mean/trade=-0.0827%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     30986  -2677.4%  32.3%
      2021     30913  -2711.3%  37.8%
      2022     30894  -2503.2%  32.2%
      2023     30848  -2522.4%  24.8%
      2024     30983  -2434.2%  32.0%
      2025     30878  -2486.7%  28.6%
      2026     15315  -1263.4%  29.2%

  HOLDOUT 2025-2026: n=46193  wr=28.8%  ret=-3750.1%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 15 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=167590  wr=30.0% (p=1.0000 vs 50%)  ret=-14502.4%  mdd=-14502.4%  mean/trade=-0.0865%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     26172  -2399.3%  30.7%
      2021     25915  -2338.6%  36.9%
      2022     25615  -2171.6%  30.8%
      2023     25555  -2138.5%  23.6%
      2024     25772  -2223.7%  30.9%
      2025     25823  -2162.9%  27.8%
      2026     12738  -1067.8%  28.1%

  HOLDOUT 2025-2026: n=38561  wr=27.9%  ret=-3230.8%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=8 (4 hold x 2 filtro volume)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
              Variante         n      Ret%   Sharpe_hat      DSR
            1min_novol   1133402  -90753.5%     -926.317    0.000
          1min_volfilt    599602  -48038.2%     -602.566    0.000
            5min_novol    487059  -40072.2%     -289.449    0.000
          5min_volfilt    339422  -28263.0%     -231.028    0.000
           10min_novol    284372  -23623.5%     -159.071    0.000
         10min_volfilt    224600  -19004.8%     -139.472    0.000
           15min_novol    200817  -16598.7%     -109.172    0.000
         15min_volfilt    167590  -14502.4%     -103.766    0.000

  Holdout 2025-2026:
              Variante         n      Ret%   Sharpe_hat      DSR
            1min_novol    260280  -20798.5%     -620.051    0.000
          1min_volfilt    137959  -11125.4%     -414.153    0.000
            5min_novol    111980  -9087.7%     -186.336    0.000
          5min_volfilt     78162  -6392.5%     -153.563    0.000
           10min_novol     65404  -5337.6%     -104.196    0.000
         10min_volfilt     51635  -4308.6%      -91.666    0.000
           15min_novol     46193  -3750.1%      -71.598    0.000
         15min_volfilt     38561  -3230.8%      -66.478    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — variante migliore per DSR = hold=1min, vol_filter=False
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope         n      Ret%      WR    MC pp    MC pr
        0bps  full-sample   1133402  -90753.5%   9.7%   0.000   1.000
        0bps     holdout    260280  -20798.5%   6.7%   0.000   1.000
        2bps  full-sample   1133402  -136089.6%   5.0%   0.000   1.000
        2bps     holdout    260280  -31209.7%   3.1%   0.000   1.000
        5bps  full-sample   1133402  -204093.8%   2.2%   0.000   1.000
        5bps     holdout    260280  -46826.5%   1.1%   0.000   1.000
       10bps  full-sample   1133402  -317434.0%   0.8%   0.000   1.000
       10bps     holdout    260280  -72854.5%   0.3%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi

Fallimento netto e uniforme su tutte e 8 le varianti — non un fallimento
marginale come le altre strategie di questa sessione, ma un'inversione di
segno pesante:

```
Hold     Win rate (no filtro)   Win rate (con filtro volume)
1 min          9.7%                    10.5%
5 min         22.1%                    21.7%
10 min        27.8%                    27.1%
15 min        31.1%                    30.0%
```

Nota: la colonna "Ret%" nel report sopra usa un notional fisso non
ricompostato (nessun position sizing a rischio, dato che la strategia non
ha uno stop) — i numeri assoluti (es. -90.753%) non vanno letti come una
curva di equity realistica, ma come somma cumulata di P&L su milioni di
trade indipendenti. La metrica corretta da guardare è **mean/trade**
(circa -0,08% per trade, sorprendentemente stabile su tutti gli hold) e
il **win rate**.

**Interpretazione**: il segnale non è debole, è **invertito**. Dopo una
candela verde, il prezzo tende a MUOVERSI IN SENSO CONTRARIO (mean-reversion
di brevissimo termine / bid-ask bounce) molto più spesso di quanto continui
— specialmente a 1 minuto, dove solo il 9,7% dei trade momentum è vincente.
Il filtro volume non cambia la sostanza (win rate quasi identico con o
senza). All'aumentare dell'hold (5→15 min) il win rate sale (9,7%→31,1%)
perché il rumore di brevissimo termine viene progressivamente diluito da
movimento più genuinamente direzionale, ma resta comunque ben sotto il 50%.

**Implicazione interessante (non testata qui)**: se il momentum a 1 minuto
ha un win rate del 9,7%, la strategia SPECULARE — FADARE la candela (short
dopo una candela verde, long dopo una candela rossa) — avrebbe un win rate
implicito del ~90,3% a 1 minuto. Questo è un classico effetto di
microstruttura (bid-ask bounce) ben documentato in letteratura, ma non
dice ancora nulla sulla profittabilità netta: serve verificare la
dimensione media di vincita/perdita e il netto delle fee prima di
concludere che sia sfruttabile — un test naturale da fare come prossimo
passo se interessa.
