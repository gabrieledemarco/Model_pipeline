# Volume Profile + VWAP Confluence — M15 — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Volume Profile + VWAP Confluence — M15 — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Stessa logica del report 1H base (RR=3.0 migliore per DSR: full DSR=0.884,
holdout DSR=0.498). Qui su barre M15, soglie in-barra riscalate ×4 per
conservare lo stesso significato in tempo reale (LOOKBACK=20b=5h, SWING_LOOKBACK=40b=10h, MAX_HOLD=192b=2gg).

  Holdout genuino 2025-2026: 30 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=123  wr=48.0% (p=0.7057 vs 50%)  ret=-18.0%  mdd=-21.1%
    Exit: TP=56  SL=58  time=9
    MC i.i.d.  : pp=0.050  pr=0.000
    MC block   : pp=0.043  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      13     +0.3%  53.8%
      2021      20     -4.0%  45.0%
      2022      21     -4.8%  47.6%
      2023      17     +2.7%  64.7%
      2024      22     -8.3%  36.4%
      2025      21     -2.0%  47.6%
      2026       9     -1.9%  44.4%

  HOLDOUT GENUINO 2025-2026: n=30  wr=46.7%  ret=-3.9%  mdd=-5.4%
    MC i.i.d.  : pp=0.220  pr=0.000
    MC block   : pp=0.067  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 1.50
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=123  wr=39.8% (p=0.9907 vs 50%)  ret=-18.3%  mdd=-25.1%
    Exit: TP=42  SL=67  time=14
    MC i.i.d.  : pp=0.089  pr=0.000
    MC block   : pp=0.057  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      13     +1.3%  46.2%
      2021      20     -5.1%  35.0%
      2022      21     -7.0%  38.1%
      2023      17     -0.4%  47.1%
      2024      22     -7.3%  31.8%
      2025      21     +0.1%  42.9%
      2026       9     +0.1%  44.4%

  HOLDOUT GENUINO 2025-2026: n=30  wr=43.3%  ret=+0.2%  mdd=-4.6%
    MC i.i.d.  : pp=0.503  pr=0.000
    MC block   : pp=0.539  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=123  wr=36.6% (p=0.9990 vs 50%)  ret=-16.2%  mdd=-28.0%
    Exit: TP=32  SL=69  time=22
    MC i.i.d.  : pp=0.145  pr=0.001
    MC block   : pp=0.128  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      13     -0.1%  38.5%
      2021      20     -6.1%  30.0%
      2022      21     -7.5%  33.3%
      2023      17     -2.8%  41.2%
      2024      22     -4.3%  31.8%
      2025      21     +2.6%  42.9%
      2026       9     +2.1%  44.4%

  HOLDOUT GENUINO 2025-2026: n=30  wr=43.3%  ret=+4.7%  mdd=-4.5%
    MC i.i.d.  : pp=0.735  pr=0.000
    MC block   : pp=0.889  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 3.00
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=123  wr=36.6% (p=0.9990 vs 50%)  ret=+7.6%  mdd=-17.1%
    Exit: TP=27  SL=69  time=27
    MC i.i.d.  : pp=0.661  pr=0.000
    MC block   : pp=0.666  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      13     +1.7%  38.5%
      2021      20     -2.1%  30.0%
      2022      21     -4.2%  33.3%
      2023      17     +1.2%  41.2%
      2024      22     +0.2%  31.8%
      2025      21     +8.6%  42.9%
      2026       9     +2.2%  44.4%

  HOLDOUT GENUINO 2025-2026: n=30  wr=43.3%  ret=+10.8%  mdd=-4.2%
    MC i.i.d.  : pp=0.884  pr=0.000
    MC block   : pp=0.977  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=4 (griglia RR)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
   1.00     123    -18.0%       -1.659    0.000
   1.50     123    -18.3%       -1.417    0.000
   2.00     123    -16.2%       -1.130    0.000
   3.00     123     +7.6%        0.418    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
   1.00      30     -3.9%       -0.739    0.000
   1.50      30     +0.2%        0.034    0.000
   2.00      30     +4.7%        0.633    0.088
   3.00      30    +10.8%        1.174    0.988

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 3.00
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample     123     +7.6%  36.6%   0.661   0.000
        0bps     holdout      30    +10.8%  43.3%   0.884   0.000
        2bps  full-sample     123     +2.2%  35.8%   0.550   0.000
        2bps     holdout      30     +9.4%  43.3%   0.855   0.000
        5bps  full-sample     123     -6.0%  35.8%   0.388   0.000
        5bps     holdout      30     +7.3%  43.3%   0.797   0.000
       10bps  full-sample     123    -19.5%  35.8%   0.160   0.010
       10bps     holdout      30     +3.7%  43.3%   0.659   0.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: M15 indebolisce il segnale full-sample, l'holdout è un'illusione da campione piccolo

```
                    1H (baseline)          M15 (questo report)
n full / holdout      160 / 45                123 / 30
RR=3.0 Ret full       +12.8%                   +7.6%
RR=3.0 Ret holdout     +6.2%                  +10.8%
DSR full               0.884                   0.000
DSR holdout             0.498                   0.988
```

**Il quadro è invertito rispetto al 1H, non migliorato**: su M15 il
DSR full-sample CROLLA a 0.000 (Sharpe_hat 0.418, troppo basso per
sopravvivere alla correzione N=4), mentre il DSR holdout SALE a 0.988 —
esattamente l'opposto di quanto un vero edge dovrebbe mostrare (un segnale
genuino dovrebbe reggere sia sull'intero campione sia nell'holdout, non
uno sì e uno no). Con soli n=30 trade nell'holdout, un DSR alto è
compatibile con rumore favorevole quanto con un vero edge — non c'è modo
di distinguerli con un campione così piccolo.

**Meno eventi, non di più**: nonostante M15 offra 4× più barre da
scansionare, gli eventi scendono da 160 a 123 (full) e da 45 a 30
(holdout) — le condizioni di trend-context/confluenza/rejection,
riscalate per avere lo stesso significato in tempo reale, filtrano in
modo più severo a grana fine (l'ATR a 15 minuti è più rumoroso barra per
barra, rendendo più raro il pattern "separazione pulita + tocco pulito +
rejection pulita" rispetto a barre 1H che aggregano il rumore intrabar).

**Slippage sensitivity**: il full-sample diventa negativo già a 5bps
(-6.0%) e peggiora nettamente a 10bps (-19.5%) — più fragile del 1H, che
reggeva fino a ~2bps sul full-sample. L'holdout resta positivo fino a
10bps (+3.7%) ma con un campione di 30 trade il margine statistico è
comunque sottile.

**Conclusione**: M15 NON conferma né rafforza l'edge — lo indebolisce sul
campione più ampio e affidabile (full-sample) mentre gonfia il numero
sull'holdout più piccolo e meno affidabile. Il timeframe 1H resta la
versione da preferire di questa strategia: campione più ampio, DSR robusto
su ENTRAMBI gli ambiti (full e holdout), e margine di slippage più ampio.
