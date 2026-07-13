══════════════════════════════════════════════════════════════════════════════
BTCUSDT — Volatility-Regime-Gated Dual-Mode — Strategy Research
══════════════════════════════════════════════════════════════════════════════
  V1 Dual-Mode Gated      n=  585  ret=  +35.1%  mdd= -17.0%  wr= 55.0%
  V2 OU Always-On         n=  972  ret=  -59.2%  mdd= -61.5%  wr= 51.4%
  V3 SMC Always-On        n=  629  ret= +145.8%  mdd=  -6.1%  wr= 53.3%

  Deflated Sharpe Ratio (family N=3, threshold=0.95):
    V1 Dual-Mode Gated      sharpe_hat=+1.413  DSR=0.000  FAIL
    V2 OU Always-On         sharpe_hat=-3.221  DSR=0.000  FAIL
    V3 SMC Always-On        sharpe_hat=+4.297  DSR=0.998  PASS
  V1 Dual-Mode Gated      n>=30:True  ret>0:True  P(ruin)<10%:True  DSR>=0.95:False  holdout>0:False  -> NON VALIDATA
  V2 OU Always-On         n>=30:True  ret>0:False  P(ruin)<10%:False  DSR>=0.95:False  holdout>0:False  -> NON VALIDATA
  V3 SMC Always-On        n>=30:True  ret>0:True  P(ruin)<10%:True  DSR>=0.95:True  holdout>0:True  -> VALIDATA

  Breakdown per anno:

    V1 Dual-Mode Gated
        2022  n=  124  ret=  -11.1%  wr= 45.2%
        2023  n=  105  ret=  +18.5%  wr= 61.9%
        2024  n=  141  ret=  +31.9%  wr= 61.7%
        2025  n=  139  ret=  -12.7%  wr= 50.4%
        2026  n=   76  ret=   +8.6%  wr= 57.9%

    V2 OU Always-On
        2022  n=  208  ret=  -10.9%  wr= 51.4%
        2023  n=  202  ret=  -12.6%  wr= 52.0%
        2024  n=  229  ret=  -14.9%  wr= 53.3%
        2025  n=  218  ret=  -13.6%  wr= 49.1%
        2026  n=  115  ret=   -7.2%  wr= 51.3%

    V3 SMC Always-On
        2022  n=  144  ret=   +8.8%  wr= 47.9%
        2023  n=  132  ret=  +45.8%  wr= 58.3%
        2024  n=  139  ret=  +59.0%  wr= 56.8%
        2025  n=  153  ret=   +3.5%  wr= 47.7%
        2026  n=   61  ret=  +28.6%  wr= 60.7%

  V1 Dual-Mode — breakdown by leg:
            n=  585  ret=  +35.1%  wr= 55.0%