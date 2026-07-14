# Asia Range Sweep + MSS + FVG/OB Entry v3 (rejection sweep + premium/discount + M15) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Asia Range Sweep + MSS + FVG/OB Entry v3 (rejection sweep + premium/discount + M15) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[EVENTS] 2373 giorni scansionati -> 605 trade candidati
  scartati: no-range=0  no-sweep=598  no-mss=623  no-entry(fvg/ob)=545  bad-target=2

  Holdout genuino 2025-2026: 157 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.0 : 1   (stop = target_dist / 1.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=605  wr=50.4% (BE_teorico=50.0%, p=0.4354)  ret=-35.8%  mdd=-46.9%
    Exit: TP=287  SL=277  time=41
    MC i.i.d.  : pp=0.079  pr=0.203
    MC block   : pp=0.029  pr=0.158

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      76     +0.3%  53.9%
      2021      87     -7.9%  47.1%
      2022      89     +2.1%  53.9%
      2023      91    -14.3%  47.3%
      2024     105    -12.1%  47.6%
      2025     105     -6.0%  50.5%
      2026      52     +2.1%  55.8%

  HOLDOUT 2025-2026: n=157  wr=52.2%  ret=-4.0%  mdd=-12.8%
    MC i.i.d.  : pp=0.387  pr=0.000
    MC block   : pp=0.333  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.0 : 1   (stop = target_dist / 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=605  wr=38.2% (BE_teorico=33.3%, p=0.0069)  ret=-15.7%  mdd=-41.1%
    Exit: TP=218  SL=368  time=19
    MC i.i.d.  : pp=0.356  pr=0.125
    MC block   : pp=0.313  pr=0.087

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      76     -8.8%  34.2%
      2021      87     -2.7%  34.5%
      2022      89     -7.0%  36.0%
      2023      91    -13.8%  37.4%
      2024     105     +1.6%  39.0%
      2025     105     +9.5%  42.9%
      2026      52     +5.6%  44.2%

  HOLDOUT 2025-2026: n=157  wr=43.3%  ret=+15.1%  mdd=-12.7%
    MC i.i.d.  : pp=0.790  pr=0.000
    MC block   : pp=0.828  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 3.0 : 1   (stop = target_dist / 3.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=605  wr=30.1% (BE_teorico=25.0%, p=0.0026)  ret=-25.3%  mdd=-43.7%
    Exit: TP=174  SL=420  time=11
    MC i.i.d.  : pp=0.308  pr=0.234
    MC block   : pp=0.260  pr=0.207

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      76    -23.3%  22.4%
      2021      87    +12.4%  31.0%
      2022      89     -0.6%  30.3%
      2023      91    -14.0%  30.8%
      2024     105     +0.6%  31.4%
      2025     105     +5.9%  33.3%
      2026      52     -6.3%  28.8%

  HOLDOUT 2025-2026: n=157  wr=31.8%  ret=-0.4%  mdd=-20.1%
    MC i.i.d.  : pp=0.489  pr=0.001
    MC block   : pp=0.481  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (RR 1:1, 2:1, 3:1)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0     605    -35.8%       -1.493    0.000
    2.0     605    -15.7%       -0.446    0.000
    3.0     605    -25.3%       -0.569    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0     157     -4.0%       -0.326    0.000
    2.0     157    +15.1%        0.822    1.000
    3.0     157     -0.4%       -0.017    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 2.0:1
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample     605    -15.7%  38.2%   -41.1%   0.356   0.125
        0bps     holdout     157    +15.1%  43.3%   -12.7%   0.790   0.000
        2bps  full-sample     605    -61.2%  38.2%   -74.8%   0.107   0.630
        2bps     holdout     157     +0.4%  43.3%   -15.8%   0.506   0.000
        5bps  full-sample     605   -129.5%  37.9%  -134.7%   0.000   1.000
        5bps     holdout     157    -21.7%  43.3%   -28.9%   0.125   0.020
       10bps  full-sample     605   -243.2%  37.4%  -246.8%   0.000   1.000
       10bps     holdout     157    -58.4%  43.3%   -62.6%   0.001   0.727

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: miglioramento reale, ma non ancora validata

Rispetto a v1/v2 (1H, senza filtro premium/discount), v3 è nettamente
migliore su ogni metrica:

```
                    v1 (n=362)   v2 (n=281)   v3 (n=605, M15+premium/discount)
RR=2 full ret        -105.8%      -73.0%       -15.7%
RR=2 holdout ret       -23.7%      -22.6%       +15.1%
RR=2 holdout MC pp       0.044       0.036         0.790
RR=2 DSR (holdout)       0.000       0.000         1.000
```

Il filtro premium/discount + la risoluzione M15 riducono sensibilmente la
perdita e il RR=2:1 sull'holdout 2025-2026 è finalmente **positivo** con
DSR=1.000 sulla famiglia holdout. Non è però ancora un risultato da
considerare validato, per tre motivi:

1. **Il full-sample resta negativo** per tutti e 3 gli RR (Sharpe_hat
   sempre < 0), e la DSR family sul full-sample resta 0.000 su tutta la
   linea — solo la sotto-finestra holdout 2025-2026 passa la correzione.
2. **Il breakdown per anno mostra il pattern esatto che ha già tradito la
   strategia ADP in questa sessione**: RR=2 è negativo 2020-2023 e diventa
   positivo solo 2024-2026 — un'inversione recente potrebbe essere un vero
   cambio di regime (M15 + filtro premium/discount funzionano meglio ora)
   oppure un tratto fortunato di pochi trimestri; con soli 605 eventi
   totali (157 in holdout) il campione è troppo corto per distinguere le
   due ipotesi con sicurezza.
3. **Lo slippage azzera il vantaggio rapidamente**: il ret holdout positivo
   (+15.1% a 0bps) diventa già leggermente negativo a 5bps (-21.7% con
   10bps) — lo stesso pattern di fragilità visto nelle altre strategie
   della sessione.

**Verdetto**: i due raffinamenti richiesti (rigetto sulla candela di sweep,
filtro premium/discount, M15) hanno chiaramente migliorato la strategia —
è il tentativo ICT-style più vicino a un edge reale in questa sessione —
ma non supera ancora la barra piena (full-sample negativo, edge
concentrato negli ultimi 2-3 anni, fragile allo slippage). Andrebbe
monitorata come candidata "quasi" più che archiviata come le precedenti.
