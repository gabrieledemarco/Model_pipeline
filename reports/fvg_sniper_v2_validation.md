# FVG Sniper [JOAT] v2 — anti-leverage-cap + concurrent positions

```
══════════════════════════════════════════════════════════════════════════════
FVG Sniper [JOAT] v2 — anti-leverage-cap filter + concurrent positions
══════════════════════════════════════════════════════════════════════════════

v2: (1) scarta i trade il cui rischio in prezzo è troppo stretto per
il sizing 1% entro MAX_LEV=10x (invece di troncare la size pagando fee
sproporzionate); (2) trade CONCORRENTI con tetto sul rischio aggregato
aperto (5% del capitale) invece di un trade alla volta.

══════════════════════════════════════════════════════════════════════════════
[15M] Detection
══════════════════════════════════════════════════════════════════════════════

  19,135 FVG (grade>=4.0, size>=0.25xATR, body>=0.45)

══════════════════════════════════════════════════════════════════════════════
[15M] Mode = Rejection Only
══════════════════════════════════════════════════════════════════════════════

  4,407 segnali -> 4,339 candidati (post filtro anti-leva; holdout 2025-2026: 1,187)

  FULL-SAMPLE: n=4339 (scartati per budget rischio: 0)  wr=30.7% (p=1.0000 vs 50%)  ret=-658.2%  mdd=-565.0%
    Exit: TP=1212  SL=2964  time=163
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     584    -60.3%  31.7%
      2021     675   -123.5%  26.4%
      2022     635   -132.9%  28.3%
      2023     570    -83.9%  33.7%
      2024     688    -56.1%  31.7%
      2025     770   -217.6%  29.5%
      2026     417    +16.0%  36.0%

  HOLDOUT GENUINO 2025-2026: n=1187  wr=31.8%  ret=-201.5%  mdd=-211.1%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
[15M] Mode = IFVG Flip Only
══════════════════════════════════════════════════════════════════════════════

  7,268 segnali -> 7,060 candidati (post filtro anti-leva; holdout 2025-2026: 1,926)

  FULL-SAMPLE: n=7056 (scartati per budget rischio: 4)  wr=30.4% (p=1.0000 vs 50%)  ret=-1461.9%  mdd=-1387.5%
    Exit: TP=2002  SL=4849  time=205
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     894   -212.9%  29.2%
      2021    1028    -30.8%  30.8%
      2022    1039   -183.2%  29.5%
      2023    1025   -304.2%  30.8%
      2024    1146   -248.3%  30.5%
      2025    1279   -446.7%  28.9%
      2026     645    -35.7%  35.3%

  HOLDOUT GENUINO 2025-2026: n=1924  wr=31.1%  ret=-482.4%  mdd=-467.3%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
[15M] Mode = Rejection + IFVG
══════════════════════════════════════════════════════════════════════════════

  11,675 segnali -> 11,399 candidati (post filtro anti-leva; holdout 2025-2026: 3,113)

  FULL-SAMPLE: n=11360 (scartati per budget rischio: 39)  wr=30.5% (p=1.0000 vs 50%)  ret=-2097.7%  mdd=-1860.4%
    Exit: TP=3207  SL=7786  time=367
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020    1475   -269.5%  30.2%
      2021    1702   -153.1%  29.1%
      2022    1671   -311.9%  29.1%
      2023    1594   -386.6%  31.9%
      2024    1830   -299.0%  31.0%
      2025    2037   -654.2%  29.2%
      2026    1051    -23.3%  35.5%

  HOLDOUT GENUINO 2025-2026: n=3088  wr=31.3%  ret=-677.6%  mdd=-615.6%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

══════════════════════════════════════════════════════════════════════════════
[15M] DSR — famiglia N=3 (griglia: modalità segnale)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
  Mode                     n      Ret%   Sharpe_hat      DSR
  Rejection Only        4339   -658.2%       -5.554    0.000
  IFVG Flip Only        7056  -1461.9%       -9.546    0.000
  Rejection + IFVG     11360  -2097.7%      -10.847    0.000

  Holdout 2025-2026:
  Mode                     n      Ret%   Sharpe_hat      DSR
  Rejection Only        1187   -201.5%       -3.197    0.000
  IFVG Flip Only        1924   -482.4%       -5.992    0.000
  Rejection + IFVG      3088   -677.6%       -6.654    0.000

══════════════════════════════════════════════════════════════════════════════
[15M] Slippage-stress — modalità migliore per DSR full+holdout = Rejection Only
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    4339   -658.2%  30.7%   0.000   1.000
        0bps     holdout    1187   -201.5%  31.8%   0.000   1.000
        2bps  full-sample    4339  -1067.2%  30.6%   0.000   1.000
        2bps     holdout    1187   -339.3%  31.8%   0.000   1.000
        5bps  full-sample    4339  -1680.7%  30.5%   0.000   1.000
        5bps     holdout    1187   -545.9%  31.6%   0.000   1.000
       10bps  full-sample    4339  -2703.2%  30.1%   0.000   1.000
       10bps     holdout    1187   -890.2%  31.1%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[1H] Detection
══════════════════════════════════════════════════════════════════════════════

  4,403 FVG (grade>=4.0, size>=0.25xATR, body>=0.45)

══════════════════════════════════════════════════════════════════════════════
[1H] Mode = Rejection Only
══════════════════════════════════════════════════════════════════════════════

  1,252 segnali -> 1,252 candidati (post filtro anti-leva; holdout 2025-2026: 298)

  FULL-SAMPLE: n=1252 (scartati per budget rischio: 0)  wr=28.4% (p=1.0000 vs 50%)  ret=-74.4%  mdd=-109.2%
    Exit: TP=318  SL=883  time=51
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.026  pr=0.971

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     173     +0.6%  29.5%
      2021     208    -51.6%  22.1%
      2022     208    -47.4%  23.1%
      2023     168    +25.7%  34.5%
      2024     197    +34.3%  35.0%
      2025     191    -51.0%  25.1%
      2026     107    +15.1%  33.6%

  HOLDOUT GENUINO 2025-2026: n=298  wr=28.2%  ret=-35.9%  mdd=-54.6%
    MC i.i.d.  : pp=0.178  pr=0.296
    MC block   : pp=0.160  pr=0.285

══════════════════════════════════════════════════════════════════════════════
[1H] Mode = IFVG Flip Only
══════════════════════════════════════════════════════════════════════════════

  1,996 segnali -> 1,996 candidati (post filtro anti-leva; holdout 2025-2026: 486)

  FULL-SAMPLE: n=1995 (scartati per budget rischio: 1)  wr=33.2% (p=1.0000 vs 50%)  ret=+200.9%  mdd=-24.2%
    Exit: TP=597  SL=1316  time=82
    MC i.i.d.  : pp=0.999  pr=0.000
    MC block   : pp=0.997  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     250    +61.8%  36.8%
      2021     346    +68.5%  34.1%
      2022     309    +28.9%  33.3%
      2023     290    +19.4%  33.4%
      2024     314    +32.1%  33.4%
      2025     314    -56.0%  26.4%
      2026     172    +46.1%  37.8%

  HOLDOUT GENUINO 2025-2026: n=486  wr=30.5%  ret=-9.8%  mdd=-62.3%
    MC i.i.d.  : pp=0.414  pr=0.133
    MC block   : pp=0.422  pr=0.180

══════════════════════════════════════════════════════════════════════════════
[1H] Mode = Rejection + IFVG
══════════════════════════════════════════════════════════════════════════════

  3,248 segnali -> 3,248 candidati (post filtro anti-leva; holdout 2025-2026: 784)

  FULL-SAMPLE: n=3230 (scartati per budget rischio: 18)  wr=31.5% (p=1.0000 vs 50%)  ret=+137.3%  mdd=-43.8%
    Exit: TP=913  SL=2185  time=132
    MC i.i.d.  : pp=0.931  pr=0.003
    MC block   : pp=0.882  pr=0.015

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     418    +66.8%  34.0%
      2021     553    +18.0%  29.7%
      2022     514    -19.2%  29.2%
      2023     458    +45.0%  33.8%
      2024     509    +68.7%  34.2%
      2025     502   -106.9%  25.9%
      2026     276    +64.8%  36.6%

  HOLDOUT GENUINO 2025-2026: n=778  wr=29.7%  ret=-42.0%  mdd=-113.0%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.022  pr=0.975

══════════════════════════════════════════════════════════════════════════════
[1H] DSR — famiglia N=3 (griglia: modalità segnale)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
  Mode                     n      Ret%   Sharpe_hat      DSR
  Rejection Only        1252    -74.4%       -1.205    0.000
  IFVG Flip Only        1995   +200.9%        2.468    1.000
  Rejection + IFVG      3230   +137.3%        1.345    0.000

  Holdout 2025-2026:
  Mode                     n      Ret%   Sharpe_hat      DSR
  Rejection Only         298    -35.9%       -1.196    0.000
  IFVG Flip Only         486     -9.8%       -0.247    0.000
  Rejection + IFVG       778    -42.0%       -0.844    0.000

══════════════════════════════════════════════════════════════════════════════
[1H] Slippage-stress — modalità migliore per DSR full+holdout = IFVG Flip Only
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    1995   +200.9%  33.2%   0.999   0.000
        0bps     holdout     486     -9.8%  30.5%   0.414   0.133
        2bps  full-sample    1995   +105.8%  33.1%   0.963   0.000
        2bps     holdout     486    -38.8%  30.5%   0.252   0.404
        5bps  full-sample    1995    -36.9%  33.0%   0.276   0.376
        5bps     holdout     486    -82.2%  30.5%   0.000   1.000
       10bps  full-sample    1995   -274.8%  33.0%   0.000   1.000
       10bps     holdout     486   -154.5%  30.5%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
