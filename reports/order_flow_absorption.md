# Order-Flow: Large-Order Absorption @ Volume Profile POC — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Order-Flow: Large-Order Absorption @ Volume Profile POC — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

DOVE: POC rolling del Volume Profile. QUANDO/CHI: barra con volume e
dimensione media trade in percentile alto (proxy di 'tape' da klines,
nessun vero order book L2/tick-by-tick disponibile) E range sotto
mediana (assorbimento). CONFERMA: delta/CVD locale concorde col trend.

  Holdout genuino 2025-2026: 26 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=44.0% (p=0.9239 vs 50%)  ret=-40.7%  mdd=-43.4%
    Exit: TP=58  SL=65  time=2
    MC i.i.d.  : pp=0.001  pr=0.123
    MC block   : pp=0.000  pr=0.128

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     -4.6%  46.7%
      2021      22     -4.1%  45.5%
      2022      21    -10.1%  33.3%
      2023      20     -9.3%  45.0%
      2024      21     -1.3%  57.1%
      2025      19    -10.3%  31.6%
      2026       7     -0.9%  57.1%

  HOLDOUT GENUINO 2025-2026: n=26  wr=38.5%  ret=-11.2%  mdd=-13.1%
    MC i.i.d.  : pp=0.010  pr=0.000
    MC block   : pp=0.021  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 1.50
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=44.0% (p=0.9239 vs 50%)  ret=-23.1%  mdd=-28.4%
    Exit: TP=53  SL=69  time=3
    MC i.i.d.  : pp=0.045  pr=0.002
    MC block   : pp=0.044  pr=0.003

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     -1.1%  46.7%
      2021      22     +1.3%  50.0%
      2022      21     -8.6%  33.3%
      2023      20     -6.1%  45.0%
      2024      21     +2.2%  52.4%
      2025      19     -8.9%  36.8%
      2026       7     -1.9%  42.9%

  HOLDOUT GENUINO 2025-2026: n=26  wr=38.5%  ret=-10.8%  mdd=-13.0%
    MC i.i.d.  : pp=0.030  pr=0.000
    MC block   : pp=0.009  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=40.0% (p=0.9902 vs 50%)  ret=-12.7%  mdd=-20.9%
    Exit: TP=47  SL=74  time=4
    MC i.i.d.  : pp=0.209  pr=0.000
    MC block   : pp=0.212  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     -1.4%  40.0%
      2021      22     +3.8%  45.5%
      2022      21     -5.2%  33.3%
      2023      20     -5.1%  40.0%
      2024      21     +4.7%  47.6%
      2025      19     -8.5%  31.6%
      2026       7     -0.9%  42.9%

  HOLDOUT GENUINO 2025-2026: n=26  wr=34.6%  ret=-9.4%  mdd=-13.7%
    MC i.i.d.  : pp=0.078  pr=0.000
    MC block   : pp=0.061  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 3.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=125  wr=32.8% (p=1.0000 vs 50%)  ret=-7.3%  mdd=-20.6%
    Exit: TP=35  SL=82  time=8
    MC i.i.d.  : pp=0.331  pr=0.000
    MC block   : pp=0.340  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      15     +3.6%  40.0%
      2021      22     +4.7%  40.9%
      2022      21     +1.8%  33.3%
      2023      20     -6.1%  30.0%
      2024      21     +2.4%  33.3%
      2025      19    -10.7%  21.1%
      2026       7     -2.9%  28.6%

  HOLDOUT GENUINO 2025-2026: n=26  wr=23.1%  ret=-13.6%  mdd=-15.7%
    MC i.i.d.  : pp=0.034  pr=0.000
    MC block   : pp=0.005  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=4 (griglia RR)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
   1.00     125    -40.7%       -3.603    0.000
   1.50     125    -23.1%       -1.681    0.000
   2.00     125    -12.7%       -0.784    0.000
   3.00     125     -7.3%       -0.371    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
   1.00      26    -11.2%       -2.306    0.000
   1.50      26    -10.8%       -1.892    0.000
   2.00      26     -9.4%       -1.417    0.000
   3.00      26    -13.6%       -1.837    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 3.00
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample     125     -7.3%  32.8%   0.331   0.000
        0bps     holdout      26    -13.6%  23.1%   0.034   0.000
        2bps  full-sample     125    -19.8%  32.8%   0.137   0.009
        2bps     holdout      26    -16.9%  23.1%   0.015   0.000
        5bps  full-sample     125    -38.6%  32.8%   0.021   0.185
        5bps     holdout      26    -21.8%  23.1%   0.003   0.000
       10bps  full-sample     125    -69.8%  28.8%   0.000   0.938
       10bps     holdout      26    -30.0%  15.4%   0.000   0.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: fallisce, con un difetto strutturale specifico nello stop

```
RR      n(full)   Ret full   WR full   n(hold)   Ret hold   WR hold   SL-hit rate
1.00      125       -40.7%    44.0%      26       -11.2%     38.5%       52.0%
1.50      125       -23.1%    44.0%      26       -10.8%     38.5%       55.2%
2.00      125       -12.7%    40.0%      26        -9.4%     34.6%       59.2%
3.00      125        -7.3%    32.8%      26       -13.6%     23.1%       65.6%
```

**Negativo su tutta la griglia RR, in entrambi gli ambiti, DSR=0.000
ovunque** — nessuna combinazione sopravvive alla correzione per selection
bias. A differenza della Volume Profile + VWAP Confluence (dove allargare
il RR migliorava progressivamente il risultato fino a superare la soglia
di break-even), qui il win rate PEGGIORA man mano che RR sale (44.0% →
32.8% full-sample) e il tasso di stop-loss SALE (52.0% → 65.6%) — il
pattern opposto a quello desiderabile per un vero setup di trend-following
asimmetrico.

**Diagnosi — difetto strutturale auto-inflitto nello stop**: lo stop è
posizionato all'estremo della barra di "assorbimento" stessa. Ma la barra
di assorbimento è STATA SELEZIONATA per avere un range basso rispetto al
suo volume (score = volume/range ATR-normalizzato in top percentile) — di
conseguenza il suo estremo è, per costruzione, insolitamente VICINO al
prezzo di entrata. Questo crea esattamente l'asimmetria di
first-passage-time diagnosticata più volte in questa sessione (NY-ORB-VP,
VWAP MR v2, VWAP Bounce v1/v2): uno stop strutturalmente troppo vicino
viene toccato dal rumore molto più spesso di quanto la tesi di
continuazione possa compensare, indipendentemente dalla qualità del
segnale di assorbimento/CVD a monte.

**Nota sui limiti dei dati usati**: questa strategia usa `taker_buy_base`
e `n_trades` per barra (i soli campi "order-flow" disponibili nei klines
di Binance Vision) come proxy di delta/CVD/dimensione-trade — NON è tape
reading vero (nessun accesso a singoli print o all'order book L2/depth).
Il segnale "large print" è quindi un'approssimazione a livello di barra
aggregata, non un'identificazione di un singolo ordine di grandi
dimensioni. Questo limite dei dati non spiega da solo il fallimento (il
problema dominante osservato è lo stop, non il segnale di ingresso), ma
resta un limite di fedeltà rispetto a un vero sistema di order-flow
professionale basato su footprint/orderbook.

**Prossimo passo naturale (non eseguito)**: sostituire lo stop
"estremo della barra di assorbimento" con uno stop ATR-multiplo (come
nelle altre strategie VWAP di questa sessione) o con lo swing
strutturale più vicino su una finestra più ampia (come in
`vp_vwap_confluence`, che usa un lookback di 10-40 barre invece della sola
barra di segnale) — per isolare se il problema sia la tesi di
assorbimento in sé o, come sospettato, solo la costruzione dello stop.
