# VWAP Mean-Reversion v2 — RR-proportional stop

```
══════════════════════════════════════════════════════════════════════════════
VWAP Mean-Reversion v2 — stop a RR proporzionale al target (fix v1)
══════════════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════════════
RR = 1.0 : 1   (stop = target_dist / 1.0)
══════════════════════════════════════════════════════════════════════════════
  Z_ENTRY più scelto: {2.0: 33, 1.5: 2}
  Filtro più scelto  : {'SIDEWAYS_ONLY': 29, 'NO_FILTER': 6}

  FULL-SAMPLE: n=2451  wr=49.4% (BE_teorico=50.0%)  ret=-473.2%  mdd=-473.1%
    Exit: TP=986  SL=972  time=493
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     343    -85.5%  46.6%
      2021     589    -96.8%  47.9%
      2022     430    -85.1%  46.7%
      2023     313    -67.5%  52.4%
      2024     283    -51.9%  51.6%
      2025     356    -68.4%  52.0%
      2026     137    -18.1%  52.6%

  HOLDOUT GENUINO 2025-2026: n=493  wr=52.1%  ret=-86.5%  mdd=-87.5%
    MC i.i.d.  : pp=0.001  pr=0.986
    MC block   : pp=0.000  pr=0.999

══════════════════════════════════════════════════════════════════════════════
RR = 1.5 : 1   (stop = target_dist / 1.5)
══════════════════════════════════════════════════════════════════════════════
  Z_ENTRY più scelto: {2.0: 33, 1.5: 2}
  Filtro più scelto  : {'SIDEWAYS_ONLY': 31, 'NO_FILTER': 4}

  FULL-SAMPLE: n=2327  wr=40.6% (BE_teorico=40.0%)  ret=-649.5%  mdd=-648.5%
    Exit: TP=749  SL=1209  time=369
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     289    -97.1%  37.0%
      2021     589   -152.7%  38.4%
      2022     360   -108.7%  37.2%
      2023     313    -87.0%  44.4%
      2024     283    -75.2%  43.1%
      2025     356    -95.9%  43.5%
      2026     137    -32.9%  44.5%

  HOLDOUT GENUINO 2025-2026: n=493  wr=43.8%  ret=-128.8%  mdd=-129.4%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.0 : 1   (stop = target_dist / 2.0)
══════════════════════════════════════════════════════════════════════════════
  Z_ENTRY più scelto: {2.0: 34, 1.5: 1}
  Filtro più scelto  : {'SIDEWAYS_ONLY': 31, 'NO_FILTER': 4}

  FULL-SAMPLE: n=2230  wr=34.6% (BE_teorico=33.3%)  ret=-797.1%  mdd=-793.3%
    Exit: TP=595  SL=1340  time=295
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     165    -87.8%  30.3%
      2021     589   -222.6%  30.6%
      2022     291    -92.1%  34.4%
      2023     313   -124.5%  36.4%
      2024     283    -83.5%  38.2%
      2025     452   -141.2%  37.6%
      2026     137    -45.5%  36.5%

  HOLDOUT GENUINO 2025-2026: n=589  wr=37.4%  ret=-186.7%  mdd=-186.8%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (RR 1:1, 1.5:1, 2:1)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0    2451   -473.2%      -10.959    0.000
    1.5    2327   -649.5%      -12.535    0.000
    2.0    2230   -797.1%      -13.555    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0     493    -86.5%       -4.309    0.000
    1.5     493   -128.8%       -5.309    0.000
    2.0     589   -186.7%       -6.069    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 1.0:1
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample    2451   -473.2%  49.4%  -473.1%   0.000   1.000
        0bps     holdout     493    -86.5%  52.1%   -87.5%   0.001   0.986
        2bps  full-sample    2451   -656.5%  48.1%  -656.0%   0.000   1.000
        2bps     holdout     493   -130.1%  50.3%  -130.4%   0.000   1.000
        5bps  full-sample    2451   -931.4%  44.6%  -930.7%   0.000   1.000
        5bps     holdout     493   -195.6%  45.6%  -195.1%   0.000   1.000
       10bps  full-sample    2451  -1389.6%  39.2%  -1389.4%   0.000   1.000
       10bps     holdout     493   -304.7%  37.9%  -303.8%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Diagnostica: il fix ha peggiorato tutto — perché

Il fix ha ridotto sensibilmente la distanza dello stop (da 2×ATR a
target_dist/RR), ma il win rate è CROLLATO invece di restare stabile:

```
                v1 (stop=2xATR)   v2 RR=1.0   v2 RR=1.5   v2 RR=2.0
WR full-sample       62.9%          49.4%       40.6%       34.6%
SL hit rate          18.5%          39.7%       ~52%        ~60%
Ret full-sample     -182.7%        -473.2%     -649.5%     -797.1%
```

Più lo stop si stringe (RR crescente), più il win rate crolla verso — e
sotto — il breakeven teorico, e il P&L peggiora monotonicamente. Questo
conferma, in un contesto diverso, la stessa lezione già emersa con la
strategia ML RF a 8h in questa sessione: **entro una finestra intraday
breve, una barriera più vicina viene toccata dal rumore molto più spesso
di quanto suggerisca un ipotetico edge direzionale**. Il win rate del 62.9%
osservato in v1 non era quindi la vera accuratezza direzionale del segnale
VWAP — era in parte un ARTEFATTO dello stop largo, che semplicemente veniva
toccato di rado, lasciando molte più operazioni libere di "arrivare"
eventualmente al target. Stringere lo stop per correggere il rapporto
rischio/rendimento fa esattamente l'errore opposto: il target diventa
proporzionalmente più vicino ma lo stop MOLTO più vicino ancora, e la
stessa asimmetria di first-passage-time lavora ora contro la strategia in
modo anche più severo.

**Conclusione:** né lo stop largo (v1, R:R sfavorevole) né lo stop stretto
e proporzionale (v2, win rate eroso dal rumore) producono un edge
sfruttabile per il mean-reversion su VWAP intraday con questa costruzione
del trade. Il problema non è il filtro di regime o la soglia di ingresso
(l'IS-scan converge sistematicamente su Z_ENTRY=2.0 e filtro SIDEWAYS_ONLY
in quasi tutte le finestre, in entrambe le versioni) — è la meccanica
stessa di TP/SL entro una finestra intraday breve.
