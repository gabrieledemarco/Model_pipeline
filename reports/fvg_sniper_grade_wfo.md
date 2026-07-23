# FVG Sniper — Filtro sul grade del gap, via WFO

```
══════════════════════════════════════════════════════════════════════════════
FVG Sniper — Filtro sul grade del gap, via Walk-Forward Optimization
══════════════════════════════════════════════════════════════════════════════

Segue l'autopsy dei trade peggiori del 2025-2026: grade medio 5.67 nei
25 peggiori vs 7.26 nel resto. Soglia MIN_GRADE selezionata causalmente
per finestra (WFO 6m IS/2m OOS), non scelta a posteriori sull'intera storia.

══════════════════════════════════════════════════════════════════════════════
1) STATICO — soglia fissa (riferimento, NON walk-forward)
══════════════════════════════════════════════════════════════════════════════

   MIN_GRADE    n_full   ret_full%    n_hold   ret_hold%   wr_hold
         4.0      1995     +200.9%       486       -9.8%     30.5%
         5.0      1725     +225.3%       420      +15.2%     31.7%
         6.0      1401     +219.5%       323      +15.4%     31.9%
         7.0      1116     +142.7%       251      +10.9%     31.5%
         8.0       778      +81.9%       167      +13.4%     32.3%

══════════════════════════════════════════════════════════════════════════════
2) WALK-FORWARD — soglia MIN_GRADE selezionata per finestra (6m IS / 2m OOS)
══════════════════════════════════════════════════════════════════════════════

[WFO] 35 finestre, griglia MIN_GRADE [4.0, 5.0, 6.0, 7.0, 8.0]

[WFO] Selezione per finestra (35 finestre valide):
  MIN_GRADE più scelto: {5.0: 12, 4.0: 9, 6.0: 6, 8.0: 6, 7.0: 2}

  FULL WFO-OOS: n=1463  wr=32.3%  ret=+103.8%  mdd=-24.5%
    MC i.i.d.  : pp=0.954  pr=0.000
    MC block   : pp=0.923  pr=0.003

  Breakdown per anno:
        Year       n      Ret%      WR
        2020      82     -4.0%  30.5%
        2021     288    +57.6%  34.0%
        2022     252    +28.5%  33.3%
        2023     220     -4.3%  31.4%
        2024     296    +23.7%  32.8%
        2025     225    -30.4%  27.1%
        2026     100    +32.6%  39.0%

  HOLDOUT GENUINO 2025-2026 (sub-slice del WFO-OOS): n=325  wr=30.8%  ret=+2.2%
    MC i.i.d.  : pp=0.530  pr=0.024
    MC block   : pp=0.511  pr=0.025

══════════════════════════════════════════════════════════════════════════════
3) Slippage-stress — MIN_GRADE=5.0 statico (full-sample + holdout)
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    1725   +225.3%   33.9%   1.000   0.000
        0bps     holdout     420    +15.2%   31.7%   0.644   0.015
        2bps  full-sample    1725   +147.4%   33.8%   0.989   0.000
        2bps     holdout     420     -8.3%   31.7%   0.414   0.089
        5bps  full-sample    1725    +30.5%   33.7%   0.688   0.031
        5bps     holdout     420    -43.5%   31.7%   0.178   0.433
       10bps  full-sample    1725   -164.3%   33.6%   0.000   1.000
       10bps     holdout     420   -102.3%   31.7%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
