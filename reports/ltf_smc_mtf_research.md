# BTCUSDT LTF SMC/MTF Strategy Research

```
══════════════════════════════════════════════════════════════════════════════
BTCUSDT — LTF Market Structure / SMC / Liquidity — MTF Strategy Research
══════════════════════════════════════════════════════════════════════════════

[WFO] 23 windows (train=6m/oos=2m/step=2m)  |  8 holdout windows (OOS start >= 2025-01-01)
  V1 Structure Baseline         n= 3610  ret= +485.6%  mdd=  -7.0%  wr= 57.5%  (full-history, no fitted params)
  V2 + Regime + Sweep           n=  774  ret=  +79.8%  mdd=  -8.3%  wr= 52.8%  (WFO-OOS chained)
  V3 + Expected-Return Gate     n=  236  ret=   +9.8%  mdd=  -8.4%  wr= 49.6%  (WFO-OOS chained)

  Deflated Sharpe Ratio (family N=3, threshold=0.95):
    V1 Structure Baseline         sharpe_hat=+7.394  DSR=1.000  PASS
    V2 + Regime + Sweep           sharpe_hat=+3.895  DSR=1.000  PASS
    V3 + Expected-Return Gate     sharpe_hat=+0.994  DSR=0.000  FAIL
  V1 Structure Baseline         n>=30:True  OOS>0:True  P(ruin)<10%:True  DSR>=0.95:True  holdout>0:True  -> VALIDATA
  V2 + Regime + Sweep           n>=30:True  OOS>0:True  P(ruin)<10%:True  DSR>=0.95:True  holdout>0:True  -> VALIDATA
  V3 + Expected-Return Gate     n>=30:True  OOS>0:True  P(ruin)<10%:True  DSR>=0.95:False  holdout>0:False  -> NON VALIDATA

  Breakdown per anno (OOS chained):

    V1 Structure Baseline
        Year      n      Ret%      WR
        2022    827   +143.5%   57.1%
        2023    626    +77.9%   56.7%
        2024    854    +67.7%   55.7%
        2025    933   +159.8%   59.5%
        2026    370    +36.8%   58.4%

    V2 + Regime + Sweep
        Year      n      Ret%      WR
        2022    103     +9.1%   52.4%
        2023    126    +16.1%   55.6%
        2024    247    +13.1%   46.6%
        2025    216    +26.6%   58.3%
        2026     82    +14.8%   53.7%

    V3 + Expected-Return Gate
        Year      n      Ret%      WR
        2022     33     +0.6%   54.5%
        2023     73     +9.1%   54.8%
        2024     85     +5.4%   45.9%
        2025     33     -5.0%   48.5%
        2026     12     -0.4%   33.3%
```
