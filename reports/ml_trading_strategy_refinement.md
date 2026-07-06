# ML Trading Strategy — Refinement Checks

```
══════════════════════════════════════════════════════════════════════════════
ML Trading Strategy — Refinement Checks (slippage / DSR / leverage)
══════════════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════════════
CHECK 1 — Slippage sensitivity (horizon=8h, full-sample + holdout)
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample    8312   +213.5%  52.7%   -12.4%   1.000   0.000
        0bps     holdout    1770    +47.4%  51.6%   -12.5%   0.989   0.000
        2bps  full-sample    8312    +86.8%  51.1%   -30.3%   0.986   0.000
        2bps     holdout    1770    +15.3%  49.6%   -21.2%   0.787   0.000
        5bps  full-sample    8312   -103.1%  48.3%  -102.8%   0.000   1.000
        5bps     holdout    1770    -32.8%  46.9%   -46.7%   0.037   0.103
       10bps  full-sample    8312   -419.8%  44.0%  -419.0%   0.000   1.000
       10bps     holdout    1770   -113.0%  42.8%  -113.5%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
CHECK 2 — Deflated Sharpe Ratio, family = {2h, 4h, 8h} (N=3 trials)
══════════════════════════════════════════════════════════════════════════════

  Full-sample family:
   Horizon       n      Ret%   Sharpe_hat      DSR
       2h    7202   -109.9%       -4.996    0.000
       4h    7686     +2.4%        0.075    0.000
       8h    8312   +213.5%        4.481    1.000

  Holdout 2025-2026 family:
   Horizon       n      Ret%   Sharpe_hat      DSR
       2h    1147    -24.8%       -2.679    0.000
       4h    1528     -4.1%       -0.291    0.000
       8h    1770    +47.4%        2.146    0.937

  Nota: DSR qui e' calcolato su una famiglia STRETTA e genuinamente
  comparabile (stessa pipeline/modello, solo l'orizzonte cambia, N=3),
  non sull'intero spazio di ricerca storico della sessione — la lettura
  corretta per 'l'orizzonte 8h sopravvive alla scelta tra 3 alternative?'

══════════════════════════════════════════════════════════════════════════════
CHECK 3 — Position sizing, implied leverage & safety-stop diagnostic (8h)
══════════════════════════════════════════════════════════════════════════════

  Sizing (rischio fisso = 1% di INIT_CAP = $1,000/trade, MAX_LEV cap=10.0x):
    Notional/trade:  median=$33,131  p95=$78,932  max=$279,379
    Leverage/trade:  median=0.33x  p95=0.79x  max=2.79x
    Trades against MAX_LEV cap (leverage >= 9.99x): 0 / 8312 (0.0%)

  Safety-stop (4xATR) activation:
    Exits by safety-SL: 565 / 8312  (6.8%)
    Exits by time (@8h): 7747 / 8312  (93.2%)

  P&L per trade by exit type (dollars, on $100,000 capital):
    Safety-SL exits (565): mean=$-1,036  median=$-1,030  worst=$-1,223  as %% of cap: worst=-1.22%
    Time exits     (7747): mean=$103  median=$50  worst=$-1,004  as %% of cap: worst=-1.00%

  Interpretazione: lo stop di sicurezza attiva solo su una piccola minoranza
  dei trade e la sua perdita per-trade (in % di capitale, a rischio fisso)
  resta contenuta — non produce code di coda catastrofiche isolate rispetto
  alla normale variabilita' delle uscite a tempo.

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
