# FVG Sniper FADE — direzione invertita + stop strutturale + retest

```
══════════════════════════════════════════════════════════════════════════════
FVG Sniper FADE — direzione invertita + stop strutturale + entry su retest
══════════════════════════════════════════════════════════════════════════════

Tre cambi strutturali (non filtri da ottimizzare): (1) fade invece di
continuazione, (2) stop strutturale 10-barre (param. riusati da ICT Fade),
(3) entry su reclaim del livello invece che a mercato immediato.

══════════════════════════════════════════════════════════════════════════════
FULL-SAMPLE 2020-2026
══════════════════════════════════════════════════════════════════════════════

  n=822  wr=32.7% (p=1.0000 vs 50%)  ret=+66.8%  mdd=-24.5%  Sharpe_trade=1.365
    Exit: TP=201  SL=519  time=102
    MC i.i.d.  : pp=0.925  pr=0.000
    MC block   : pp=0.904  pr=0.001

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     101    +16.1%  33.7%
      2021     149    +16.8%  31.5%
      2022     140     +4.0%  31.4%
      2023      94     -2.1%  31.9%
      2024     121    +41.3%  39.7%
      2025     155     -6.5%  31.0%
      2026      62     -2.9%  29.0%

══════════════════════════════════════════════════════════════════════════════
HOLDOUT GENUINO 2025-2026
══════════════════════════════════════════════════════════════════════════════

  n=217  wr=30.4%  ret=-9.3%  mdd=-23.9%  Sharpe_trade=-0.382
    Exit: TP=47  SL=145  time=25
    MC i.i.d.  : pp=0.342  pr=0.006
    MC block   : pp=0.334  pr=0.007

══════════════════════════════════════════════════════════════════════════════
Slippage-stress
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample     822    +66.8%  32.7%   0.925   0.000
        0bps     holdout     217     -9.3%  30.4%   0.342   0.006
        2bps  full-sample     822    +45.4%  32.7%   0.841   0.002
        2bps     holdout     217    -16.2%  30.4%   0.245   0.018
        5bps  full-sample     822    +13.4%  32.4%   0.613   0.024
        5bps     holdout     217    -26.6%  30.4%   0.134   0.074
       10bps  full-sample     822    -40.0%  32.0%   0.162   0.368
       10bps     holdout     217    -43.9%  29.5%   0.039   0.352

══════════════════════════════════════════════════════════════════════════════
CONFRONTO — FADE (qui) vs continuazione (baseline/filtro grade)
══════════════════════════════════════════════════════════════════════════════

                              n      Ret%      WR    MC pp(full)   ret_hold%   MC pp(hold)
  FADE (qui)                  822    +66.8%   32.7%       0.925         -9.3%        0.342
  Continuazione, grade=5.0   1725   +225.3%   33.9%      1.000        +15.2%        0.644
  Continuazione, baseline    1995   +200.9%   33.2%      0.999         -9.8%        0.414

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
