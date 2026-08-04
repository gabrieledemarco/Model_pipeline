# Candle Fade (short green / long red, volume filter, 1-15min hold) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Candle Fade (short green / long red, volume filter, 1-15min hold) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════════════
HOLD = 1 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1133402  wr=9.9% (p=1.0000 vs 50%)  ret=-90590.8%  mdd=-90590.8%  mean/trade=-0.0799%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020    175236  -13914.5%  11.0%
      2021    175137  -13939.2%  18.7%
      2022    174488  -14057.1%  10.5%
      2023    173152  -13912.2%   5.1%
      2024    175109  -13921.5%   9.0%
      2025    173899  -13939.3%   6.3%
      2026     86381  -6907.0%   7.0%

  HOLDOUT 2025-2026: n=260280  wr=6.5%  ret=-20846.3%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 1 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=599602  wr=10.9% (p=1.0000 vs 50%)  ret=-47898.1%  mdd=-47898.1%  mean/trade=-0.0799%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     95481  -7597.0%  11.8%
      2021     92660  -7416.5%  20.6%
      2022     90510  -7363.4%  11.7%
      2023     90893  -7322.5%   5.8%
      2024     92099  -7250.6%  10.0%
      2025     92875  -7380.1%   6.8%
      2026     45084  -3567.9%   8.0%

  HOLDOUT 2025-2026: n=137959  wr=7.2%  ret=-10948.0%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 5 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=487059  wr=23.7% (p=1.0000 vs 50%)  ret=-37857.2%  mdd=-37857.2%  mean/trade=-0.0777%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     75205  -5675.4%  25.8%
      2021     75071  -5693.3%  33.6%
      2022     74957  -5888.8%  24.7%
      2023     74670  -5860.6%  17.0%
      2024     75176  -5910.0%  23.4%
      2025     74837  -5935.7%  19.4%
      2026     37143  -2893.5%  20.4%

  HOLDOUT 2025-2026: n=111980  wr=19.8%  ret=-8829.1%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 5 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=339422  wr=24.7% (p=1.0000 vs 50%)  ret=-26044.6%  mdd=-26044.6%  mean/trade=-0.0767%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.003  pr=0.997

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     53685  -3955.3%  27.0%
      2021     52445  -3873.0%  35.4%
      2022     51543  -4092.5%  25.8%
      2023     51532  -4013.4%  17.4%
      2024     52055  -3996.9%  24.3%
      2025     52439  -4124.0%  19.8%
      2026     25723  -1989.4%  21.7%

  HOLDOUT 2025-2026: n=78162  wr=20.4%  ret=-6113.5%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 10 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=284372  wr=29.6% (p=1.0000 vs 50%)  ret=-21876.0%  mdd=-21876.0%  mean/trade=-0.0769%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     43891  -3174.7%  31.7%
      2021     43793  -3377.9%  38.0%
      2022     43753  -3363.8%  30.6%
      2023     43653  -3451.5%  22.6%
      2024     43878  -3380.9%  30.1%
      2025     43716  -3386.0%  26.2%
      2026     21688  -1741.1%  26.7%

  HOLDOUT 2025-2026: n=65404  wr=26.4%  ret=-5127.1%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 10 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=224600  wr=30.9% (p=1.0000 vs 50%)  ret=-16931.2%  mdd=-16900.0%  mean/trade=-0.0754%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     35270  -2565.0%  33.1%
      2021     34652  -2525.3%  39.9%
      2022     34267  -2572.6%  32.0%
      2023     34177  -2694.8%  23.5%
      2024     34599  -2620.5%  30.9%
      2025     34629  -2633.2%  27.1%
      2026     17006  -1319.8%  28.1%

  HOLDOUT 2025-2026: n=51635  wr=27.4%  ret=-3953.0%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 15 min   (senza filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=200817  wr=32.7% (p=1.0000 vs 50%)  ret=-15532.1%  mdd=-15532.1%  mean/trade=-0.0773%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     30986  -2280.4%  34.6%
      2021     30913  -2234.8%  40.5%
      2022     30894  -2439.8%  33.6%
      2023     30848  -2413.2%  26.4%
      2024     30983  -2523.1%  32.6%
      2025     30878  -2453.7%  29.7%
      2026     15315  -1187.0%  30.8%

  HOLDOUT 2025-2026: n=46193  wr=30.1%  ret=-3640.8%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
HOLD = 15 min   (con filtro volume)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=167590  wr=34.0% (p=1.0000 vs 50%)  ret=-12312.0%  mdd=-12313.1%  mean/trade=-0.0735%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year         n      Ret%      WR
      2020     26172  -1788.2%  35.9%
      2021     25915  -1807.8%  42.1%
      2022     25615  -1926.8%  35.0%
      2023     25555  -1950.3%  27.8%
      2024     25772  -1899.8%  34.3%
      2025     25823  -1968.8%  30.2%
      2026     12738   -970.2%  31.2%

  HOLDOUT 2025-2026: n=38561  wr=30.5%  ret=-2939.0%
    MC i.i.d.  : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=8 (4 hold x 2 filtro volume)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
              Variante         n      Ret%   Sharpe_hat      DSR
            1min_novol   1133402  -90590.8%     -924.656    0.000
          1min_volfilt    599602  -47898.1%     -600.808    0.000
            5min_novol    487059  -37857.2%     -273.449    0.000
          5min_volfilt    339422  -26044.6%     -212.895    0.000
           10min_novol    284372  -21876.0%     -147.304    0.000
         10min_volfilt    224600  -16931.2%     -124.254    0.000
           15min_novol    200817  -15532.1%     -102.157    0.000
         15min_volfilt    167590  -12312.0%      -88.093    0.000

  Holdout 2025-2026:
              Variante         n      Ret%   Sharpe_hat      DSR
            1min_novol    260280  -20846.3%     -621.477    0.000
          1min_volfilt    137959  -10948.0%     -407.550    0.000
            5min_novol    111980  -8829.1%     -181.034    0.000
          5min_volfilt     78162  -6113.5%     -146.861    0.000
           10min_novol     65404  -5127.1%     -100.087    0.000
         10min_volfilt     51635  -3953.0%      -84.102    0.000
           15min_novol     46193  -3640.8%      -69.511    0.000
         15min_volfilt     38561  -2939.0%      -60.475    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — variante migliore per DSR = hold=1min, vol_filter=False
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope         n      Ret%      WR    MC pp    MC pr
        0bps  full-sample   1133402  -90590.8%   9.9%   0.000   1.000
        0bps     holdout    260280  -20846.3%   6.5%   0.000   1.000
        2bps  full-sample   1133402  -135926.9%   4.9%   0.000   1.000
        2bps     holdout    260280  -31257.5%   2.8%   0.000   1.000
        5bps  full-sample   1133402  -203931.1%   2.0%   0.000   1.000
        5bps     holdout    260280  -46874.3%   1.0%   0.000   1.000
       10bps  full-sample   1133402  -317271.3%   0.7%   0.000   1.000
       10bps     holdout    260280  -72902.3%   0.3%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Scoperta: la fade fallisce QUANTO il momentum — la spiegazione "reversal" era sbagliata

Confronto diretto win rate momentum vs fade (stesse barre segnale, stesso hold):

```
Hold     WR momentum   WR fade
1 min        9.7%        9.9%
5 min       22.1%       23.7%
10 min      27.8%       29.6%
15 min      31.1%       32.7%
```

Se il problema fosse davvero un'inversione di segnale (mean-reversion reale
dopo una candela direzionale), il win rate della fade dovrebbe essere
vicino a **1 - WR_momentum** (es. ~90% a 1 minuto, non ~10%). Il fatto che
siano quasi IDENTICI smentisce l'ipotesi di "reversal" scritta nel report
precedente (`candle_momentum.md`) — quella lettura era sbagliata.

**Causa reale, verificata con un diagnostico dedicato** (return lordo,
senza fee, calcolato per ogni barra verde/rossa indipendentemente dalla
direzione presa):

```
Hold     Win rate LORDO (nessuna fee)   Mean return lordo
1 min           48.8%                        +0.00001%
5 min           48.8%                        -0.00208%
10 min          48.9%                        -0.00274%
15 min          49.1%                        -0.00244%

Frazione di trade con |move lordo| < fee round-trip (0,08%):
1 min: 80.3%   5 min: 53.7%   10 min: 42.1%   15 min: 35.9%
```

Il colore della candela (verde/rosso) su barre da 1 minuto **non ha alcun
potere predittivo, né di continuazione né di inversione** — il win rate
lordo è un coin-flip quasi perfetto (48.8-49.1%) in entrambe le direzioni.
Il crollo del win rate netto (9,7-34%) osservato sia in momentum sia in
fade non è un segnale di mercato: è la fee (0,08% round-trip) che, a
questi orizzonti, è quasi sempre più grande del movimento di prezzo reale
(80% dei trade a 1 minuto hanno un movimento lordo inferiore alla fee
stessa). Qualunque direzione si scelga, la fee decide il segno della
maggior parte dei trade.

## Tentativo di soluzione: filtrare per dimensione della candela (body/ATR)

Ipotesi: forse solo le candele "forti" (body molto sopra la media) hanno
un vero segnale, annegato nel rumore delle candele piccole. Testato
filtrando per `body >= {1x, 2x, 3x, 5x, 8x}` la media mobile 20 barre del
body:

```
                    momentum (continuazione)          fade equivalente (per costruzione = -momentum)
Hold=1   body>=1x   wr=47.9%  mean=-0.0005%     ->     mean=+0.0005%
Hold=1   body>=8x   wr=42.8%  mean=-0.0126%     ->     mean=+0.0126%
Hold=15  body>=1x   wr=48.4%  mean=-0.0032%     ->     mean=+0.0032%
Hold=15  body>=8x   wr=43.9%  mean=-0.0121%     ->     mean=+0.0121%
```

Risultato interessante ma non risolutivo: le candele con body grande
mostrano una tendenza a **invertirsi** più delle candele piccole (win rate
di continuazione che scende da ~48% a ~43% man mano che il body cresce,
un pattern statisticamente rilevabile e coerente su tutti gli hold) — quindi
fadare le candele più estreme ha un effetto lordo positivo, non casuale.
Ma la dimensione dell'effetto (max ~0,013% = 1,3 bps anche nel caso più
estremo) resta **un ordine di grandezza sotto la fee round-trip di 0,08%
(8 bps)** — non sopravvive nemmeno lontanamente ai costi di transazione.

## Conclusione della ricerca di soluzioni

Ho testato: momentum, fade, con/senza filtro volume, 4 durate di hold, e
un filtro di dimensione candela a 5 soglie — 18 configurazioni in totale.
**Nessuna supera la fee.** Il limite non è la direzione o il filtro
scelto, è strutturale: a 1 minuto su BTCUSDT, il movimento di prezzo
tipico dopo una singola candela è dell'ordine di qualche bps, mentre il
costo di transazione realistico (fee, ignorando anche lo slippage) è di
8 bps round-trip. Nessuna combinazione di questi ingredienti (colore,
volume, dimensione candela, direzione, durata fino a 15 minuti) genera un
segnale abbastanza grande da coprire quel costo. Per operare in modo
sostenibile a questa granularità servirebbe un exchange/account con fee
strutturalmente più basse (maker rebate, VIP tier) oppure un orizzonte di
holding più lungo — la strada "segnale su singola candela 1m" così com'è
richiesta non ha soluzione entro i costi di questa pipeline.
