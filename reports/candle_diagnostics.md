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
[CONCLUSIONE parziale] Il colore della candela 1m non ha potere predittivo lordo (win rate ~48.8-49.1%, un coin-flip). Filtrare per body grande rivela una leggera tendenza a INVERTIRSI (win rate di continuazione scende con la dimensione del body), ma l'effetto (<2bps anche nei casi estremi) resta un ordine di grandezza sotto la fee round-trip (8bps) — non sfruttabile.
==============================================================================

[3] Claim specifica: 'big trades hold information for the next 20 minutes' (volume = top X% della finestra scorrevole 24h, causale)

  Soglia: volume nel top 1.00% della finestra 24h — direzione FADE (short big-green, long big-red):
    Hold         n    fade WR    fade gross%    fade net%
     1m     40490     53.6%       0.00039%    -0.07961%
     2m     40490     54.5%       0.00634%    -0.07366%
     5m     40490     54.4%       0.01064%    -0.06936%
    10m     40490     54.7%       0.01770%    -0.06230%
    15m     40490     54.4%       0.01664%    -0.06336%
    20m     40490     54.7%       0.01685%    -0.06315%
    30m     40490     54.4%       0.02178%    -0.05822%
    60m     40490     53.9%       0.01970%    -0.06030%
   120m     40490     53.4%       0.01600%    -0.06400%
   240m     40490     52.5%       0.01481%    -0.06519%
   480m     40490     51.8%       0.00007%    -0.07993%

  Soglia: volume nel top 0.10% della finestra 24h — direzione FADE (short big-green, long big-red):
    Hold         n    fade WR    fade gross%    fade net%
     1m      5992     54.1%       0.00050%    -0.07950%
     2m      5992     55.1%       0.01011%    -0.06989%
     5m      5992     54.0%       0.01775%    -0.06225%
    10m      5992     54.0%       0.01291%    -0.06709%
    15m      5992     53.0%      -0.00116%    -0.08116%
    20m      5992     53.9%      -0.00357%    -0.08357%
    30m      5992     53.2%      -0.00346%    -0.08346%
    60m      5992     53.3%      -0.00942%    -0.08942%
   120m      5992     52.4%      -0.01106%    -0.09106%
   240m      5992     52.6%       0.00677%    -0.07323%
   480m      5992     51.9%      -0.02054%    -0.10054%

==============================================================================
[CONCLUSIONE FINALE] Anche isolando le candele con volume genuinamente estremo (top 1% e top 0.1% di una finestra di 24h, non solo 'sopra la media 20 barre') e testando orizzonti fino a 8 ore, l'effetto lordo massimo osservato è ~2.2 bps (fade, top 1%, hold 30min) — un quarto della fee round-trip (8bps) — e decade verso il rumore su orizzonti più lunghi (240-480min). La claim 'big trades hold information for the next 20 minutes' non è verificabile con dati OHLCV aggregati a 1 minuto: servirebbero dati order-flow/Level 2 (aggressore reale, bid/ask imbalance) per isolare i 'big trade' veri, di cui il volume di barra 1m è solo un proxy molto rumoroso.
==============================================================================
```
