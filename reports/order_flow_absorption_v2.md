# Order-Flow: Large-Order Absorption v2 — stop ATR scan — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Order-Flow: Large-Order Absorption v2 — stop ATR scan — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

v1 aveva SL-hit rate crescente (52.0%->65.6%) con RR, sintomo di uno
stop troppo vicino perché ancorato all'estremo di una barra selezionata
apposta per range basso. v2 scansiona lo stop ATR con RR fisso=2.0.

  Holdout genuino 2025-2026: 26 trade

══════════════════════════════════════════════════════════════════════════════
SL = 0.50 × ATR  (RR fisso = 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=33.6% (p=0.9999 vs 50%)  ret=-66.7%  mdd=-68.3%
    Exit: TP=42  SL=83  time=0
    MC i.i.d.  : pp=0.000  pr=0.947
    MC block   : pp=0.000  pr=0.932

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     -5.2%  40.0%
      2021      22     -7.9%  31.8%
      2022      21     -8.7%  38.1%
      2023      20     -7.8%  45.0%
      2024      21    -16.2%  23.8%
      2025      19    -12.8%  31.6%
      2026       7     -8.1%  14.3%

  HOLDOUT GENUINO 2025-2026: n=26  wr=26.9%  ret=-20.8%  mdd=-22.6%
    MC i.i.d.  : pp=0.002  pr=0.000
    MC block   : pp=0.000  pr=0.000

══════════════════════════════════════════════════════════════════════════════
SL = 0.75 × ATR  (RR fisso = 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=32.0% (p=1.0000 vs 50%)  ret=-50.6%  mdd=-54.7%
    Exit: TP=40  SL=85  time=0
    MC i.i.d.  : pp=0.001  pr=0.545
    MC block   : pp=0.002  pr=0.500

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     -5.8%  33.3%
      2021      22     -8.6%  27.3%
      2022      21     -4.8%  38.1%
      2023      20     -3.1%  45.0%
      2024      21     -3.8%  38.1%
      2025      19    -20.8%  10.5%
      2026       7     -3.7%  28.6%

  HOLDOUT GENUINO 2025-2026: n=26  wr=15.4%  ret=-24.6%  mdd=-26.2%
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

══════════════════════════════════════════════════════════════════════════════
SL = 1.00 × ATR  (RR fisso = 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=31.2% (p=1.0000 vs 50%)  ret=-42.2%  mdd=-46.9%
    Exit: TP=39  SL=86  time=0
    MC i.i.d.  : pp=0.004  pr=0.252
    MC block   : pp=0.006  pr=0.235

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     -4.3%  33.3%
      2021      22    -13.5%  18.2%
      2022      21     -8.8%  28.6%
      2023      20     -0.5%  45.0%
      2024      21     +0.9%  42.9%
      2025      19    -12.9%  21.1%
      2026       7     -3.0%  28.6%

  HOLDOUT GENUINO 2025-2026: n=26  wr=23.1%  ret=-15.9%  mdd=-17.8%
    MC i.i.d.  : pp=0.013  pr=0.000
    MC block   : pp=0.000  pr=0.000

══════════════════════════════════════════════════════════════════════════════
SL = 1.50 × ATR  (RR fisso = 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=34.4% (p=0.9998 vs 50%)  ret=-18.1%  mdd=-24.8%
    Exit: TP=43  SL=81  time=1
    MC i.i.d.  : pp=0.125  pr=0.002
    MC block   : pp=0.132  pr=0.002

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     +0.1%  40.0%
      2021      22     -3.3%  31.8%
      2022      21     -3.9%  33.3%
      2023      20     -1.0%  40.0%
      2024      21     +2.6%  42.9%
      2025      19    -10.2%  21.1%
      2026       7     -2.4%  28.6%

  HOLDOUT GENUINO 2025-2026: n=26  wr=23.1%  ret=-12.6%  mdd=-16.4%
    MC i.i.d.  : pp=0.030  pr=0.000
    MC block   : pp=0.004  pr=0.000

══════════════════════════════════════════════════════════════════════════════
SL = 2.00 × ATR  (RR fisso = 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=36.0% (p=0.9994 vs 50%)  ret=-9.1%  mdd=-21.0%
    Exit: TP=43  SL=79  time=3
    MC i.i.d.  : pp=0.291  pr=0.000
    MC block   : pp=0.278  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     +0.8%  40.0%
      2021      22     -5.7%  27.3%
      2022      21     -5.9%  28.6%
      2023      20     -4.5%  35.0%
      2024      21     +6.4%  47.6%
      2025      19     +2.8%  42.1%
      2026       7     -3.0%  28.6%

  HOLDOUT GENUINO 2025-2026: n=26  wr=38.5%  ret=-0.2%  mdd=-5.7%
    MC i.i.d.  : pp=0.489  pr=0.000
    MC block   : pp=0.447  pr=0.000

══════════════════════════════════════════════════════════════════════════════
SL = 3.00 × ATR  (RR fisso = 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=40.0% (p=0.9902 vs 50%)  ret=-3.5%  mdd=-14.9%
    Exit: TP=35  SL=70  time=20
    MC i.i.d.  : pp=0.394  pr=0.000
    MC block   : pp=0.414  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     -3.1%  33.3%
      2021      22     +1.0%  40.9%
      2022      21     +3.2%  42.9%
      2023      20     -6.7%  35.0%
      2024      21     -1.2%  38.1%
      2025      19     +4.9%  47.4%
      2026       7     -1.6%  42.9%

  HOLDOUT GENUINO 2025-2026: n=26  wr=46.2%  ret=+3.3%  mdd=-4.5%
    MC i.i.d.  : pp=0.694  pr=0.000
    MC block   : pp=0.837  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=6 (moltiplicatori ATR dello stop)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
   SL xATR       n      Ret%   Sharpe_hat      DSR
     0.50     125    -66.7%       -4.156    0.000
     0.75     125    -50.6%       -3.196    0.000
     1.00     125    -42.2%       -2.690    0.000
     1.50     125    -18.1%       -1.128    0.000
     2.00     125     -9.1%       -0.569    0.000
     3.00     125     -3.5%       -0.239    0.000

  Holdout 2025-2026:
   SL xATR       n      Ret%   Sharpe_hat      DSR
     0.50      26    -20.8%       -3.095    0.000
     0.75      26    -24.6%       -4.468    0.000
     1.00      26    -15.9%       -2.413    0.000
     1.50      26    -12.6%       -1.918    0.000
     2.00      26     -0.2%       -0.032    0.000
     3.00      26     +3.3%        0.485    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — SL migliore per DSR = 3.00×ATR
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample     125     -3.5%  40.0%   0.394   0.000
        0bps     holdout      26     +3.3%  46.2%   0.694   0.000
        2bps  full-sample     125     -7.7%  40.0%   0.290   0.000
        2bps     holdout      26     +2.4%  46.2%   0.643   0.000
        5bps  full-sample     125    -13.9%  40.0%   0.169   0.000
        5bps     holdout      26     +0.9%  46.2%   0.562   0.000
       10bps  full-sample     125    -24.2%  38.4%   0.051   0.005
       10bps     holdout      26     -1.5%  42.3%   0.417   0.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: la diagnosi era corretta, ma il segnale sottostante non ha edge

```
SL xATR   Ret full   Ret holdout   DSR full   DSR holdout
  0.50     -66.7%       -20.8%       0.000       0.000
  0.75     -50.6%       -24.6%       0.000       0.000
  1.00     -42.2%       -15.9%       0.000       0.000
  1.50     -18.1%       -12.6%       0.000       0.000
  2.00      -9.1%        -0.2%       0.000       0.000
  3.00      -3.5%        +3.3%       0.000       0.000
```

**La diagnosi era corretta**: allargando lo stop da 0.5× a 3.0× ATR, il
ret migliora in modo monotono e marcato in entrambi gli ambiti (full-sample
-66.7%→-3.5%; holdout -20.8%→+3.3%) — esattamente il pattern di
first-passage-time già visto ripetutamente in sessione. Il problema NON
era (solo) la tesi di assorbimento/CVD, ma proprio l'ancoraggio dello stop
all'estremo di una barra selezionata per avere range strutturalmente
piccolo.

**Ma anche lo stop più largo testato (3.0×ATR) non stacca un vero edge**:
full-sample resta negativo (-3.5%, Sharpe_hat -0.239), l'holdout diventa
marginalmente positivo (+3.3%) ma con Sharpe_hat 0.485 — troppo debole
per sopravvivere alla correzione DSR anche su una famiglia piccola (N=6):
**DSR=0.000 su tutta la griglia, in entrambi gli ambiti**, senza eccezioni.

**Confronto diretto con Volume Profile + VWAP Confluence** (la strategia
sorella di questa sessione, stessa infrastruttura POC/VWAP/trend-context,
RR=3.0): quella raggiungeva DSR full-sample 0.884 e holdout 0.498 con uno
stop strutturale a finestra più ampia (minimo/massimo su 10-40 barre, non
la sola barra di segnale). Qui, anche dopo aver rimosso l'artefatto dello
stop-strettissimo, il segnale aggiuntivo di questa strategia — tape
footprint (volume/dimensione trade in percentile alto) + CVD locale sopra
il tocco POC — non produce un edge comparabile. Ipotesi più probabile:
il proxy di "large print" costruito da barre aggregate 15m (non da tick
individuali) è troppo rumoroso per isolare un vero comportamento di un
grande player, e la richiesta di CVD locale concorde aggiunge un filtro
in più senza aggiungere segnale reale.

**Conclusione onesta**: la strategia di order-flow richiesta
("riconoscere dove sono piazzati ordini di grandi dimensioni e tradare
quei livelli") non produce un edge validato con i dati disponibili
(OHLCV + taker_buy_base + n_trades aggregati a barra). Il Volume Profile
POC da solo (componente "dove") ha già mostrato un edge reale in
`vp_vwap_confluence.md`; l'aggiunta del proxy di tape/CVD (componente
"chi/quando") in questo report non lo migliora — anzi il campione più
selettivo (125 vs 160 trade) e il segnale extra non aggiungono
robustezza. Un vero miglioramento richiederebbe dati a livello di singolo
trade (aggTrades) o order book L2, non disponibili in questa pipeline.
