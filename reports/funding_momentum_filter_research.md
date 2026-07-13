══════════════════════════════════════════════════════════════════════════════
BTCUSDT — Funding-Rate Momentum Filter — Strategy Research
══════════════════════════════════════════════════════════════════════════════
  C1 Baseline (no funding filter)     n=  629  ret= +145.8%  mdd=  -6.1%  wr= 53.3%
  C2 Avoid Fighting Extreme Funding   n=  572  ret= +119.7%  mdd=  -6.1%  wr= 53.8%
  C3 Require Funding Alignment        n=  216  ret=  +17.2%  mdd=  -9.9%  wr= 47.7%

  Deflated Sharpe Ratio (family N=3, threshold=0.95):
    C1 Baseline (no funding filter)     sharpe_hat=+4.297  DSR=1.000  PASS
    C2 Avoid Fighting Extreme Funding   sharpe_hat=+3.891  DSR=1.000  PASS
    C3 Require Funding Alignment        sharpe_hat=+1.097  DSR=0.000  FAIL
  C1 Baseline (no funding filter)     n>=30:True  ret>0:True  P(ruin)<10%:True  DSR>=0.95:True  holdout>0:True  -> VALIDATA
  C2 Avoid Fighting Extreme Funding   n>=30:True  ret>0:True  P(ruin)<10%:True  DSR>=0.95:True  holdout>0:True  -> VALIDATA
  C3 Require Funding Alignment        n>=30:True  ret>0:True  P(ruin)<10%:True  DSR>=0.95:False  holdout>0:True  -> NON VALIDATA