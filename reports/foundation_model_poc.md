# Foundation Model POC — Chronos-Bolt zero-shot su BTC/USDT

```

══════════════════════════════════════════════════════════════════════════════
RISULTATI AGGREGATI (tutto il periodo)
══════════════════════════════════════════════════════════════════════════════
   Horizon        n   Hit rate   Corr(fc,real)   Coverage q10-90 (atteso ~80%)
       4h    14098     50.2%         +0.004                          81.9%
      16h    14098     51.0%         +0.015                          75.2%
      24h    14098     50.9%         +0.010                          74.0%
      48h    14098     51.2%         +0.016                          72.7%

══════════════════════════════════════════════════════════════════════════════
RISULTATI PER ANNO (verifica di stabilità — lezione dalla validazione ADP)
══════════════════════════════════════════════════════════════════════════════

  Horizon 4h:
      Year       n   Hit rate      Corr   Coverage
      2020    2069     53.2%   +0.007     83.2%
      2021    2190     48.9%   +0.018     80.4%
      2022    2190     49.5%   -0.008     81.6%
      2023    2190     50.7%   -0.009     83.3%
      2024    2196     50.8%   -0.021     82.1%
      2025    2190     49.7%   -0.030     81.0%
      2026    1073     47.0%   +0.013     81.8%

  Horizon 16h:
      Year       n   Hit rate      Corr   Coverage
      2020    2069     54.0%   +0.027     76.9%
      2021    2190     51.4%   +0.012     75.0%
      2022    2190     49.8%   -0.037     74.3%
      2023    2190     49.9%   +0.045     75.6%
      2024    2196     53.5%   +0.042     75.8%
      2025    2190     49.4%   -0.004     74.0%
      2026    1073     47.7%   -0.072     74.7%

  Horizon 24h:
      Year       n   Hit rate      Corr   Coverage
      2020    2069     55.8%   +0.013     75.8%
      2021    2190     50.1%   -0.016     74.7%
      2022    2190     48.1%   -0.016     72.7%
      2023    2190     50.0%   +0.031     74.5%
      2024    2196     53.8%   +0.051     74.1%
      2025    2190     49.5%   -0.009     73.8%
      2026    1073     46.9%   -0.093     71.2%

  Horizon 48h:
      Year       n   Hit rate      Corr   Coverage
      2020    2069     57.3%   +0.040     75.2%
      2021    2190     48.8%   -0.036     72.7%
      2022    2190     50.0%   +0.029     72.1%
      2023    2190     48.9%   -0.008     72.9%
      2024    2196     53.2%   +0.031     71.9%
      2025    2190     51.6%   -0.031     72.9%
      2026    1073     47.2%   -0.116     69.4%

══════════════════════════════════════════════════════════════════════════════
INTERPRETAZIONE
══════════════════════════════════════════════════════════════════════════════
  Hit rate: 50% = random walk (nessun potere predittivo). Valori consistentemente
  >52-53% su TUTTI gli anni (non solo alcuni) sarebbero un segnale genuino;
  valori vicini al 50% o instabili anno per anno indicano nessun edge reale.
  Coverage q10-90: se il modello è ben calibrato, l'80% degli outcome reali
  dovrebbe cadere nella banda [q10,q90]. Coverage molto diverso da 80% indica
  intervalli di incertezza mal calibrati per questo asset/orizzonte.
```
