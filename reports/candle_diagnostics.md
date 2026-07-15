# Candle Color Diagnostics — gross predictive power check

```
==============================================================================
Candle color diagnostics — gross (no-fee) predictive power check
==============================================================================

[1] Return lordo (nessuna fee) seguendo il colore della candela — direzione = +1 su verde, -1 su rosso, indipendentemente da cosa poi si tradi:

    Hold           n    gross WR   gross mean%   frac|move|<fee_rt
     1m     3371846      48.8%      0.00001%              80.3%
     5m     3371842      48.8%     -0.00208%              53.7%
    10m     3371837      48.9%     -0.00274%              42.1%
    15m     3371832      49.1%     -0.00244%              35.9%

[2] Stesso controllo, filtrato per dimensione del body (>= Nx la media mobile 20 barre):

    Hold    body>=          n    gross WR   gross mean%
     1m     1.0x    1383221      47.9%     -0.00050%
     1m     2.0x     449357      47.1%     -0.00037%
     1m     3.0x     152706      46.2%     -0.00045%
     1m     5.0x      28579      44.4%     -0.00187%
     1m     8.0x       6062      42.8%     -0.00126%

     5m     1.0x    1383219      48.2%     -0.00256%
     5m     2.0x     449357      47.6%     -0.00202%
     5m     3.0x     152706      46.8%     -0.00227%
     5m     5.0x      28579      45.3%     -0.00521%
     5m     8.0x       6062      44.0%     -0.01395%

    10m     1.0x    1383219      48.2%     -0.00372%
    10m     2.0x     449357      47.4%     -0.00435%
    10m     3.0x     152706      46.5%     -0.00512%
    10m     5.0x      28579      44.7%     -0.00939%
    10m     8.0x       6062      43.3%     -0.01955%

    15m     1.0x    1383215      48.4%     -0.00318%
    15m     2.0x     449356      47.5%     -0.00380%
    15m     3.0x     152706      46.6%     -0.00405%
    15m     5.0x      28579      45.0%     -0.00718%
    15m     8.0x       6062      43.9%     -0.01210%

==============================================================================
[CONCLUSIONE] Il colore della candela 1m non ha potere predittivo lordo (win rate ~48.8-49.1%, un coin-flip). Filtrare per body grande rivela una leggera tendenza a INVERTIRSI (win rate di continuazione scende con la dimensione del body), ma l'effetto (<2bps anche nei casi estremi) resta un ordine di grandezza sotto la fee round-trip (8bps) — non sfruttabile.
==============================================================================
```
