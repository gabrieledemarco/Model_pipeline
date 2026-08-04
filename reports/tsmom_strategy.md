# Time-Series Momentum (TSMOM) su BTCUSDT — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Time-Series Momentum (Moskowitz/Ooi/Pedersen 2012) su BTCUSDT — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Costruzione classica: posizione = segno del rendimento passato su
lookback L, vol-targeting separato. Pool su 5 lookback (30-252 giorni).
Stesso framework infrastrutturale (forecast continui, mark-to-market,
frizioni sul turnover) già validato per Carver Breakout pool.

══════════════════════════════════════════════════════════════════════════════
Diagnostica — singoli lookback TSMOM-sign e TSMOM-magnitude (mai tradati da soli)
══════════════════════════════════════════════════════════════════════════════

                  Rule   Ret full%      WR     MDD%
        TSMOM-sign(30)     +126.2%  48.9%   -13.8%
        TSMOM-sign(60)     +100.8%  49.0%   -18.1%
        TSMOM-sign(90)      +88.8%  48.2%   -28.7%
       TSMOM-sign(120)     +131.3%  47.5%   -18.7%
       TSMOM-sign(252)     +123.2%  45.1%   -19.9%
         TSMOM-mag(30)     +160.5%  48.1%   -11.7%
         TSMOM-mag(60)      +94.9%  48.5%   -22.8%
         TSMOM-mag(90)     +110.3%  47.5%   -23.8%
        TSMOM-mag(120)     +144.6%  46.9%   -19.5%
        TSMOM-mag(252)     +100.1%  44.9%   -32.9%

══════════════════════════════════════════════════════════════════════════════
TSMOM-sign pool
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=2403  wr=49.4%  ret=+126.3%  mdd=-7.6%
    MC i.i.d.  : pp=1.000  pr=0.000
    MC block   : pp=0.999  pr=0.000

  Breakdown per anno:
        Year      Ret%      WR
        2020    +58.1%  40.5%
        2021    +16.1%  51.8%
        2022     +5.3%  54.0%
        2023    +15.1%  46.8%
        2024    +24.9%  51.4%
        2025     +5.1%  50.7%
        2026     +1.7%  50.9%

  HOLDOUT GENUINO 2025-2026: n=577  wr=50.8%  ret=+6.9%
    MC i.i.d.  : pp=0.650  pr=0.000
    MC block   : pp=0.631  pr=0.000

══════════════════════════════════════════════════════════════════════════════
TSMOM-magnitude pool
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=2403  wr=49.8%  ret=+137.0%  mdd=-13.2%
    MC i.i.d.  : pp=0.997  pr=0.000
    MC block   : pp=0.991  pr=0.000

  Breakdown per anno:
        Year      Ret%      WR
        2020    +81.9%  43.8%
        2021    +19.4%  49.9%
        2022     +7.4%  54.8%
        2023     -2.7%  46.8%
        2024    +31.6%  51.9%
        2025     -0.2%  49.6%
        2026     -0.4%  52.8%

  HOLDOUT GENUINO 2025-2026: n=577  wr=50.8%  ret=-0.6%
    MC i.i.d.  : pp=0.493  pr=0.001
    MC block   : pp=0.479  pr=0.000

══════════════════════════════════════════════════════════════════════════════
Combined (sign+mag)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=2403  wr=49.9%  ret=+132.0%  mdd=-9.1%
    MC i.i.d.  : pp=0.999  pr=0.000
    MC block   : pp=0.996  pr=0.000

  Breakdown per anno:
        Year      Ret%      WR
        2020    +70.1%  44.4%
        2021    +17.8%  51.2%
        2022     +6.3%  54.5%
        2023     +6.3%  46.8%
        2024    +28.3%  50.8%
        2025     +2.5%  50.7%
        2026     +0.7%  51.4%

  HOLDOUT GENUINO 2025-2026: n=577  wr=51.0%  ret=+3.3%
    MC i.i.d.  : pp=0.570  pr=0.000
    MC block   : pp=0.548  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (3 candidati finali)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
                   Candidate       n      Ret%   Sharpe_hat      DSR
             TSMOM-sign pool    2403   +126.3%        2.931    1.000
        TSMOM-magnitude pool    2403   +137.0%        2.373    1.000
         Combined (sign+mag)    2403   +132.0%        2.653    1.000

  Holdout 2025-2026:
                   Candidate       n      Ret%   Sharpe_hat      DSR
             TSMOM-sign pool     577     +6.9%        0.352    1.000
        TSMOM-magnitude pool     577     -0.6%       -0.026    0.000
         Combined (sign+mag)     577     +3.3%        0.154    0.433

══════════════════════════════════════════════════════════════════════════════
Slippage-stress — migliore per DSR = TSMOM-sign pool
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    2403   +126.3%  49.4%   1.000   0.000
        0bps     holdout     577     +6.9%  50.8%   0.650   0.000
        2bps  full-sample    2403   +124.3%  49.2%   1.000   0.000
        2bps     holdout     577     +6.3%  50.8%   0.639   0.000
        5bps  full-sample    2403   +121.2%  49.2%   0.999   0.000
        5bps     holdout     577     +5.4%  50.8%   0.622   0.000
       10bps  full-sample    2403   +116.1%  48.9%   0.999   0.000
       10bps     holdout     577     +4.0%  50.6%   0.594   0.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
