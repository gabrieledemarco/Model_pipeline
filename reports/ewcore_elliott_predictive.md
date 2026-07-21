# EWCore v0.2.6.4 (Elliott Wave) — Oggetti e potere predittivo

```
══════════════════════════════════════════════════════════════════════════════
EWCore v0.2.6.4 (Elliott Wave, TradingView) — Oggetti e potere predittivo
══════════════════════════════════════════════════════════════════════════════

Replica fedele dei costrutti causali dello script Pine v6 condiviso
(pivot ATR-adattivo spike-robust, Impulse, Zigzag/Flat, Invalidation
breach) + stessa metodologia event-study già usata per gli oggetti ICT/SMC:
forward return nella direzione implicita, Welch t-test vs controllo
random-direction, 4 orizzonti (1/5/10/20 barre), 4 timeframe.

══════════════════════════════════════════════════════════════════════════════
PIVOT ATR-ADATTIVI — conteggio e moltiplicatore effettivo per timeframe
══════════════════════════════════════════════════════════════════════════════

  TF       ATR mult (auto)    Pivot totali
  1D                 2.026             358
  4H                 1.741           3,332
  1H                 1.700          13,229
  15M                1.700          50,298

══════════════════════════════════════════════════════════════════════════════
FREQUENZA (eventi totali nel campione, per timeframe)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                         1D        4H        1H       15M
  Impulse (rev. attesa)           23       154       581     2,207
  Zigzag/Flat (fade C)           264     2,566    10,284    38,815
  Invalidation breach             10        87       397     1,536

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 1 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                               1D              4H              1H             15M
  Impulse (rev. attesa)           +1.161%         +0.510%*        +0.235%*        +0.160%*
  Zigzag/Flat (fade C)            +1.351%*        +0.400%*        +0.220%*        +0.120%*
  Invalidation breach                  n/a        +0.215%         +0.036%         -0.017% 

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 5 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                               1D              4H              1H             15M
  Impulse (rev. attesa)           -1.884%         +0.759%*        +0.367%*        +0.239%*
  Zigzag/Flat (fade C)            +2.889%*        +0.860%*        +0.455%*        +0.237%*
  Invalidation breach                  n/a        +0.349%         -0.021%         -0.044% 

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 10 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                               1D              4H              1H             15M
  Impulse (rev. attesa)           +0.167%         +0.437%         +0.265%*        +0.212%*
  Zigzag/Flat (fade C)            +3.003%*        +0.882%*        +0.453%*        +0.256%*
  Invalidation breach                  n/a        +0.750%         -0.072%         -0.039% 

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 20 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                               1D              4H              1H             15M
  Impulse (rev. attesa)           +0.427%         +0.059%         +0.291%*        +0.209%*
  Zigzag/Flat (fade C)            +2.784%*        +0.846%*        +0.461%*        +0.257%*
  Invalidation breach                  n/a        +0.370%         -0.007%         -0.099%*

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
