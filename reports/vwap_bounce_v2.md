# VWAP Bounce v2 — ottimizzazione dello stop ATR — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VWAP Bounce v2 — ottimizzazione dello stop ATR — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

v1 (stop fisso 1.5xATR) aveva WR=35.6%, peggio della regola letterale,
con 50.5% delle uscite via stop-loss — sospetto di first-passage-time.
v2 scansiona il moltiplicatore ATR dello stop [1.0-4.0] per isolare se
il problema sia la tesi del rimbalzo o la costruzione dello stop.

  Holdout genuino 2025-2026: 459 trade

══════════════════════════════════════════════════════════════════════════════
SL = 1.00 × ATR
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1979  wr=28.3% (p=1.0000 vs 50%)  ret=-235.8%  mdd=-182.4%
    Exit: SL=1290  time(fine giornata)=689
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     298    +41.9%  32.6%
      2021     320    -19.2%  32.2%
      2022     297    -53.1%  26.6%
      2023     300    -85.4%  25.3%
      2024     305    -32.9%  28.5%
      2025     304    -49.1%  27.3%
      2026     155    -38.0%  22.6%

  HOLDOUT GENUINO 2025-2026: n=459  wr=25.7%  ret=-87.1%  mdd=-89.0%
    MC i.i.d.  : pp=0.013  pr=0.938
    MC block   : pp=0.005  pr=0.959

══════════════════════════════════════════════════════════════════════════════
SL = 1.50 × ATR
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
SL = 2.00 × ATR
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1979  wr=39.8% (p=1.0000 vs 50%)  ret=-85.0%  mdd=-93.8%
    Exit: SL=792  time(fine giornata)=1187
    MC i.i.d.  : pp=0.029  pr=0.893
    MC block   : pp=0.019  pr=0.901

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     298    +31.6%  40.9%
      2021     320     +8.7%  45.6%
      2022     297    -11.1%  41.4%
      2023     300    -49.4%  34.0%
      2024     305    -20.8%  39.3%
      2025     304    -17.1%  39.1%
      2026     155    -26.9%  35.5%

  HOLDOUT GENUINO 2025-2026: n=459  wr=37.9%  ret=-44.0%  mdd=-55.5%
    MC i.i.d.  : pp=0.033  pr=0.376
    MC block   : pp=0.031  pr=0.348

══════════════════════════════════════════════════════════════════════════════
SL = 2.50 × ATR
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1979  wr=42.1% (p=1.0000 vs 50%)  ret=-50.4%  mdd=-67.5%
    Exit: SL=604  time(fine giornata)=1375
    MC i.i.d.  : pp=0.077  pr=0.516
    MC block   : pp=0.062  pr=0.510

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     298    +26.7%  43.3%
      2021     320     +6.8%  47.5%
      2022     297    -11.1%  43.1%
      2023     300    -24.3%  36.7%
      2024     305    -16.9%  42.0%
      2025     304    -14.7%  41.4%
      2026     155    -16.9%  38.7%

  HOLDOUT GENUINO 2025-2026: n=459  wr=40.5%  ret=-31.6%  mdd=-43.3%
    MC i.i.d.  : pp=0.061  pr=0.113
    MC block   : pp=0.047  pr=0.085

══════════════════════════════════════════════════════════════════════════════
SL = 3.00 × ATR
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1979  wr=43.8% (p=1.0000 vs 50%)  ret=-38.3%  mdd=-55.5%
    Exit: SL=474  time(fine giornata)=1505
    MC i.i.d.  : pp=0.121  pr=0.316
    MC block   : pp=0.099  pr=0.293

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     298    +24.1%  45.0%
      2021     320     +2.8%  47.8%
      2022     297    -10.1%  44.8%
      2023     300    -14.3%  39.0%
      2024     305    -16.4%  43.3%
      2025     304    -10.8%  44.4%
      2026     155    -13.6%  40.6%

  HOLDOUT GENUINO 2025-2026: n=459  wr=43.1%  ret=-24.4%  mdd=-33.9%
    MC i.i.d.  : pp=0.093  pr=0.025
    MC block   : pp=0.070  pr=0.013

══════════════════════════════════════════════════════════════════════════════
SL = 4.00 × ATR
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=1979  wr=45.3% (p=1.0000 vs 50%)  ret=-31.8%  mdd=-50.9%
    Exit: SL=297  time(fine giornata)=1682
    MC i.i.d.  : pp=0.125  pr=0.179
    MC block   : pp=0.106  pr=0.165

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     298    +19.3%  46.6%
      2021     320     +0.2%  48.8%
      2022     297     -8.8%  47.1%
      2023     300    -12.2%  40.0%
      2024     305    -17.0%  44.6%
      2025     304     -6.4%  45.7%
      2026     155     -6.8%  43.2%

  HOLDOUT GENUINO 2025-2026: n=459  wr=44.9%  ret=-13.2%  mdd=-25.1%
    MC i.i.d.  : pp=0.185  pr=0.000
    MC block   : pp=0.173  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=6 (moltiplicatori ATR dello stop)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
   SL xATR       n      Ret%   Sharpe_hat      DSR
     1.00    1979   -235.8%       -2.617    0.000
     1.50    1979   -120.9%       -1.750    0.000
     2.00    1979    -85.0%       -1.508    0.000
     2.50    1979    -50.4%       -1.061    0.000
     3.00    1979    -38.3%       -0.933    0.000
     4.00    1979    -31.8%       -0.992    0.000

  Holdout 2025-2026:
   SL xATR       n      Ret%   Sharpe_hat      DSR
     1.00     459    -87.1%       -2.055    0.000
     1.50     459    -48.5%       -1.466    0.000
     2.00     459    -44.0%       -1.653    0.000
     2.50     459    -31.6%       -1.399    0.000
     3.00     459    -24.4%       -1.226    0.000
     4.00     459    -13.2%       -0.854    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — SL migliore per DSR = 1.00×ATR
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    1979   -235.8%  28.3%   0.000   1.000
        0bps     holdout     459    -87.1%  25.7%   0.013   0.938
        2bps  full-sample    1979   -352.2%  27.3%   0.000   1.000
        2bps     holdout     459   -119.9%  24.4%   0.000   1.000
        5bps  full-sample    1979   -526.9%  26.4%   0.000   1.000
        5bps     holdout     459   -169.1%  24.0%   0.000   1.000
       10bps  full-sample    1979   -818.1%  24.9%   0.000   1.000
       10bps     holdout     459   -251.1%  23.5%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: la diagnosi era corretta, ma non basta a salvare la strategia

```
SL xATR   WR full   Ret full   WR holdout   Ret holdout   SL-hit rate   DSR (full/holdout)
  1.00     28.3%    -235.8%       25.7%        -87.1%         65.2%       0.000 / 0.000
  1.50     35.6%    -120.9%       33.3%         -48.5%         50.5%       0.000 / 0.000
  2.00     39.8%     -85.0%       37.9%         -44.0%         40.0%       0.000 / 0.000
  2.50     42.1%     -50.4%       40.5%         -31.6%         30.5%       0.000 / 0.000
  3.00     43.8%     -38.3%       43.1%         -24.4%         24.0%       0.000 / 0.000
  4.00     45.3%     -31.8%       44.9%         -13.2%         15.0%       0.000 / 0.000
```

**La diagnosi era corretta**: allargando lo stop da 1.0x a 4.0x ATR, il
win rate migliora in modo monotono e marcato (28.3% → 45.3%) e la perdita
si riduce costantemente (-235.8% → -31.8% full-sample; -87.1% → -13.2%
holdout), esattamente il pattern atteso dall'asimmetria di
first-passage-time — uno stop più stretto viene toccato dal rumore molto
più spesso, indipendentemente dalla bontà della tesi di continuazione (il
tasso di attivazione dello stop scende da 65,2% a 15,0% man mano che si
allarga).

**Ma anche lo stop più largo testato (4.0×ATR) resta negativo** — win
rate 45.3% (ancora sotto il 50%), ret -31.8% full-sample e -13.2%
holdout, Sharpe_hat -0.992. **DSR = 0.000 su tutta la griglia**, in
entrambi gli ambiti (full-sample e holdout): nessuno dei 6 moltiplicatori
testati supera la correzione per selection bias.

Il trend è chiaramente asintotico (i miglioramenti si riducono man mano
che lo stop si allarga: da 1.0→1.5 il WR guadagna +7,3pp, da 3.0→4.0 solo
+1,5pp) — allargare ulteriormente lo stop oltre 4×ATR porterebbe
verosimilmente a miglioramenti sempre più piccoli, non a un'inversione di
segno. **Conclusione**: la costruzione dello stop spiega una parte
importante del problema (come sospettato), ma non tutto — la tesi
"rimbalzo sul VWAP in direzione del trend, tenuto fino a fine giornata"
non ha, di per sé, un edge diretto positivo su BTCUSDT 1H, indipendentemente
da come si costruisce lo stop.
