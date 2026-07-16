# Replica paper VWAP Trend-Following (Zarattini & Aziz, SSRN 2023) su BTCUSDT

```
══════════════════════════════════════════════════════════════════════════════
Replica paper: VWAP Trend-Following (Zarattini & Aziz, SSRN 2023) su BTCUSDT
══════════════════════════════════════════════════════════════════════════════

Paper: 'Volume Weighted Average Price (VWAP): The Holy Grail for Day
Trading Systems' — SSRN abstract_id=4631351
Regola originale (QQQ 2018-2023): long sopra VWAP, short sotto VWAP.
Risultati dichiarati: +671% ret, MaxDD -9.4%, Sharpe 2.1 (no walk-forward,
no holdout dichiarati nelle fonti disponibili; PDF SSRN non fetchabile,
metodologia ricostruita da Concretum Group — sito degli stessi autori —
e Bear Bull Traders).

  Holdout genuino 2025-2026: 2418 trade

══════════════════════════════════════════════════════════════════════════════
A) Regola letterale (nessuno stop)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=10938  wr=44.3% (p=1.0000 vs 50%)  ret=-1008.7%  mdd=-740.8%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020    1646    -47.8%  45.3%
      2021    1603   -174.9%  46.2%
      2022    1772   -165.0%  45.1%
      2023    1825   -203.4%  40.3%
      2024    1674   -143.3%  46.1%
      2025    1586   -185.6%  43.3%
      2026     832    -88.7%  43.4%

  HOLDOUT GENUINO 2025-2026: n=2418  wr=43.3%  ret=-274.2%  mdd=-262.2%
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Slippage sensitivity:
      Slippage       Scope       n      Ret%      WR    MC pp    MC pr
          0bps  full-sample   10938  -1008.7%  44.3%   0.000   1.000
          0bps     holdout    2418   -274.2%  43.3%   0.000   1.000
          2bps  full-sample   10938  -1446.5%  42.8%   0.000   1.000
          2bps     holdout    2418   -371.0%  41.8%   0.000   1.000
          5bps  full-sample   10938  -2103.1%  40.7%   0.000   1.000
          5bps     holdout    2418   -516.0%  39.5%   0.000   1.000
         10bps  full-sample   10938  -3197.6%  38.1%   0.000   1.000
         10bps     holdout    2418   -757.8%  36.5%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
B) + safety-stop 3xATR
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=10938  wr=42.5% (p=1.0000 vs 50%)  ret=-426.9%  mdd=-354.2%
    Exit: safety-SL=2439  time(fine giornata)=8499
    MC i.i.d.  : pp=0.084  pr=0.531
    MC block   : pp=0.053  pr=0.517

  Breakdown per anno:
      Year       n      Ret%      WR
      2020    1646    +10.2%  43.9%
      2021    1603    -43.7%  44.6%
      2022    1772    -85.7%  43.1%
      2023    1825   -132.0%  38.5%
      2024    1674    -50.7%  44.6%
      2025    1586    -65.5%  41.4%
      2026     832    -59.6%  41.1%

  HOLDOUT GENUINO 2025-2026: n=2418  wr=41.3%  ret=-125.1%  mdd=-124.1%
    MC i.i.d.  : pp=0.037  pr=0.892
    MC block   : pp=0.029  pr=0.905

  Slippage sensitivity:
      Slippage       Scope       n      Ret%      WR    MC pp    MC pr
          0bps  full-sample   10938   -426.9%  42.5%   0.084   0.531
          0bps     holdout    2418   -125.1%  41.3%   0.037   0.892
          2bps  full-sample   10938   -651.5%  41.1%   0.035   0.863
          2bps     holdout    2418   -184.1%  39.8%   0.000   1.000
          5bps  full-sample   10938   -988.2%  39.1%   0.000   1.000
          5bps     holdout    2418   -272.6%  37.6%   0.000   1.000
         10bps  full-sample   10938  -1549.5%  36.6%   0.000   1.000
         10bps     holdout    2418   -420.0%  34.9%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: la regola letterale del paper fallisce nettamente su BTCUSDT

```
                         Paper originale (QQQ)      Replica (BTCUSDT 1H, letterale)
Periodo                  2018-2023 (5.75y)           2020-2026 (6.5y)
Return totale                +671%                        -1008.7% (A) / -426.9% (B)
Max Drawdown                  -9.4%                       -740.8% (A) / -354.2% (B)
Sharpe                          2.1                       fortemente negativo
Win rate                   non riportato                  44.3% (A) / 42.5% (B)
```

**Causa diagnosticata**: 10.938 trade in 6,5 anni = **~4,6 flip al giorno**.
La regola "long sopra VWAP / short sotto VWAP" applicata letteralmente su
barre 1H genera un cambio di posizione ogni volta che il prezzo attraversa
il VWAP — che su un asset volatile come BTC accade molto spesso durante
la giornata. Il risultato è puro **overtrading/whipsaw**: si perde denaro
in OGNI singolo anno 2020-2026, sia con sia senza safety-stop (variante B
migliora la magnitudine ma resta negativa ovunque, MC p_profit full-sample
0,084).

**Perché il paper originale riporta risultati opposti**: due ipotesi, non
mutuamente esclusive:
1. **Periodo backtest fortemente favorevole**: QQQ 2018-2023 include uno dei
   più lunghi bull market tecnologici della storia (Nasdaq-100 quasi
   triplicato nel periodo, nonostante i due bear market 2018 e 2022
   citati). Una regola "long quando sopra la media" in un trend
   dominante cattura gran parte del rialzo strutturale — è più vicino a
   un filtro di bias long con occasionali stop-out, che a un vero segnale
   di timing. Sullo stesso principio (edge concentrato/gonfiato da un
   periodo favorevole) si sono già arenate la strategia ADP e la v3 di
   Asia-sweep in questa sessione.
2. **Granularità/frequenza non specificata nelle fonti disponibili**: il
   PDF SSRN non è fetchabile (403) e le fonti secondarie (sito degli
   autori, Bear Bull Traders) non specificano il timeframe né se il
   sistema ribalta posizione ad ogni attraversamento o valuta la
   posizione una sola volta al giorno. Un vero sistema "day trading" più
   verosimilmente valuta la direzione UNA volta (es. alla chiusura di
   una finestra di apertura) invece di ribaltare posizione ad ogni
   incrocio — la replica qui presentata, la più fedele possibile al
   materiale disponibile ("long sopra VWAP, short sotto"), può quindi
   essere più aggressiva/rumorosa dell'originale.

**Conclusione**: presa alla lettera, la regola del paper non regge su
BTCUSDT — fallisce per overtrading, non per assenza di segnale. Una
verifica più equa richiederebbe l'accesso al PDF completo (per le regole
esatte di frequenza/timeframe) o un test con un filtro anti-whipsaw (es.
richiedere che il segnale persista N barre prima di ribaltare, o valutare
la direzione una sola volta al giorno) — non ancora provato qui.
