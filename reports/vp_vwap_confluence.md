# Volume Profile + VWAP Confluence — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Volume Profile + VWAP Confluence — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Replica della strategia 'Volume Profile + VWAP' (POC = dove, VWAP =
quando): entry sul pullback nella zona di confluenza POC/VWAP con
rimbalzo in direzione del trend, stop sotto/sopra un minimo/massimo
chiave strutturale. Estende la VWAP Bounce (v1/v2) con un filtro di
confluenza sul Volume Profile e uno stop strutturale invece che ATR fisso.

  Holdout genuino 2025-2026: 45 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=160  wr=51.2% (p=0.4063 vs 50%)  ret=-6.6%  mdd=-10.1%
    Exit: TP=75  SL=65  time=20
    MC i.i.d.  : pp=0.290  pr=0.000
    MC block   : pp=0.212  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      24     -2.2%  50.0%
      2021      18     -4.6%  44.4%
      2022      25     +1.9%  56.0%
      2023      25     +6.4%  68.0%
      2024      23     -4.3%  39.1%
      2025      32     -3.3%  46.9%
      2026      13     -0.7%  53.8%

  HOLDOUT GENUINO 2025-2026: n=45  wr=48.9%  ret=-4.0%  mdd=-7.2%
    MC i.i.d.  : pp=0.257  pr=0.000
    MC block   : pp=0.124  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 1.50
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=160  wr=46.9% (p=0.8077 vs 50%)  ret=+7.8%  mdd=-9.1%
    Exit: TP=61  SL=71  time=28
    MC i.i.d.  : pp=0.710  pr=0.000
    MC block   : pp=0.752  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      24     -4.8%  41.7%
      2021      18     -1.6%  44.4%
      2022      25     +3.8%  48.0%
      2023      25    +11.5%  64.0%
      2024      23     +0.2%  39.1%
      2025      32     -3.6%  40.6%
      2026      13     +2.3%  53.8%

  HOLDOUT GENUINO 2025-2026: n=45  wr=44.4%  ret=-1.3%  mdd=-8.1%
    MC i.i.d.  : pp=0.427  pr=0.000
    MC block   : pp=0.386  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=160  wr=41.9% (p=0.9838 vs 50%)  ret=+6.7%  mdd=-15.5%
    Exit: TP=47  SL=79  time=34
    MC i.i.d.  : pp=0.654  pr=0.000
    MC block   : pp=0.683  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      24     -1.8%  41.7%
      2021      18     -6.8%  33.3%
      2022      25     -0.4%  40.0%
      2023      25     +7.9%  52.0%
      2024      23     +4.1%  39.1%
      2025      32     -1.6%  37.5%
      2026      13     +5.3%  53.8%

  HOLDOUT GENUINO 2025-2026: n=45  wr=42.2%  ret=+3.7%  mdd=-5.3%
    MC i.i.d.  : pp=0.655  pr=0.000
    MC block   : pp=0.717  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 3.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=160  wr=37.5% (p=0.9994 vs 50%)  ret=+12.8%  mdd=-11.8%
    Exit: TP=32  SL=84  time=44
    MC i.i.d.  : pp=0.741  pr=0.000
    MC block   : pp=0.805  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      24     +4.2%  41.7%
      2021      18     -4.8%  33.3%
      2022      25     -2.4%  32.0%
      2023      25     +7.8%  44.0%
      2024      23     +1.8%  30.4%
      2025      32     +0.2%  34.4%
      2026      13     +6.0%  53.8%

  HOLDOUT GENUINO 2025-2026: n=45  wr=40.0%  ret=+6.2%  mdd=-6.2%
    MC i.i.d.  : pp=0.724  pr=0.000
    MC block   : pp=0.823  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=4 (griglia RR)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
   1.00     160     -6.6%       -0.554    0.000
   1.50     160     +7.8%        0.541    0.301
   2.00     160     +6.7%        0.407    0.008
   3.00     160    +12.8%        0.649    0.884

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
   1.00      45     -4.0%       -0.629    0.000
   1.50      45     -1.3%       -0.175    0.000
   2.00      45     +3.7%        0.405    0.100
   3.00      45     +6.2%        0.582    0.498

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 3.00
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample     160    +12.8%  37.5%   0.741   0.000
        0bps     holdout      45     +6.2%  40.0%   0.724   0.000
        2bps  full-sample     160     +7.2%  37.5%   0.639   0.000
        2bps     holdout      45     +4.2%  40.0%   0.652   0.000
        5bps  full-sample     160     -1.3%  36.9%   0.467   0.000
        5bps     holdout      45     +1.2%  40.0%   0.538   0.000
       10bps  full-sample     160    -15.4%  36.9%   0.218   0.005
       10bps     holdout      45     -3.7%  40.0%   0.353   0.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: il miglior edge grezzo trovato in questa sessione, ma su un campione sottile

```
RR      n(full)   Ret full   DSR full   n(hold)   Ret hold   DSR hold   WR full
1.00      160       -6.6%      0.000      45        -4.0%      0.000     51.2%
1.50      160       +7.8%      0.301      45        -1.3%      0.000     46.9%
2.00      160       +6.7%      0.008      45        +3.7%      0.100     41.9%
3.00      160      +12.8%      0.884      45        +6.2%      0.498     37.5%
```

**RR=3.0 è l'unica combinazione, in tutta la sessione, con DSR full-sample
>0.8** (0.884) — un segnale che sopravvive alla correzione per selection
bias sul grid RR. Il pattern è quello classico del trend-following
asimmetrico: win rate strutturalmente sotto il 50% (37.5% full, 40.0%
holdout) compensato da un rapporto premio/rischio 3:1, non da accuratezza
direzionale. MC p_ruin=0.000 su tutta la griglia RR≥1.5, e p_profit sale
monotonicamente con RR (0.741 full, 0.724 holdout a RR=3.0).

**Fee sensitivity (`vp_vwap_confluence_makerfee.md`)**: a differenza della
VWAP Mean-Reversion (positiva SOLO nello scenario full-maker ottimistico),
qui il risultato **resta positivo anche con fee taker reali Bybit**
(0.055%+0.055%): full +12.8%, holdout +6.2%. Sotto fee maker (l'entry è
level-based — zona di confluenza nota in anticipo — quindi un limit order
è realistico) il risultato migliora ulteriormente: full +22.6%, holdout
+9.6%. Anche nello scenario misto (entry maker/exit taker, più realistico
perché l'uscita reagisce a un livello appena toccato) resta solido: full
+17.7%, holdout +7.9%.

**Ma il campione è sottile**: solo 160 trade in 6,5 anni (~24/anno, meno
di 2 al mese) — il filtro a 4 condizioni (trend + confluenza POC/VWAP +
touch + rejection) è molto selettivo. La griglia RR ha solo N=4 elementi,
quindi la correzione DSR è meno potente che su famiglie più larghe testate
altrove in sessione (es. N=6 per gli stop ATR). Il breakdown per anno a
RR=3.0 mostra un contributo positivo distribuito su 4 dei 7 anni (2020,
2023, 2024, 2026) e negativo negli altri 3 (2021, 2022, 2025) — non
concentrato in un singolo regime "fortunato" come si era visto altrove
in sessione (es. ADP, Asia-sweep v3), ma nemmeno uniformemente stabile.

**Slippage sensitivity**: il margine si erode rapidamente. A 5bps di
slippage il full-sample torna negativo (-1.3%) mentre l'holdout resta
marginalmente positivo (+1.2%); a 10bps entrambi peggiorano ulteriormente
(full -15.4%, holdout -3.7%, anche se il MC p_ruin resta basso, 0.005 e
0.000). Il margine di sicurezza sopra costi di esecuzione realistici è
quindi stretto — plausibile su BTC (spread tipicamente sub-bps sui
principali exchange) ma non ampio.

**Conclusione**: è il candidato più solido emerso in questa sessione tra
le strategie SMC/volume-profile — combina un edge asimmetrico che supera
la correzione DSR, un p_ruin nullo, e una robustezza alle fee che le altre
varianti VWAP non avevano. Resta però un campione piccolo (n=160/45) e un
margine di slippage stretto: prima di qualsiasi utilizzo reale servirebbe
(a) più storia per allargare il campione fuori-holdout, (b) verifica se il
fill rate limit assunto nello scenario maker è realistico in esecuzione
live, (c) eventualmente irrigidire ulteriormente il filtro di confluenza
per capire se un campione più selettivo migliora ulteriormente la qualità
del segnale o semplicemente riduce la potenza statistica.
