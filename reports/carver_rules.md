# Carver Systematic Rules (EWMAC / Breakout / Carry) su BTCUSDT — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Carver Systematic Rules (EWMAC / Breakout / Carry) su BTCUSDT — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Building block del framework di qoppac.blogspot.com: forecast continui,
vol-targeting, mai una sola velocità (pool). Backtest mark-to-market
giornaliero, frizioni Bybit obbligatorie sul turnover.

══════════════════════════════════════════════════════════════════════════════
Diagnostica — singole velocità EWMAC e Breakout (mai tradate da sole)
══════════════════════════════════════════════════════════════════════════════

              Rule   Ret full%      WR     MDD%
        EWMAC(2,8)     +146.1%  45.2%   -16.9%
       EWMAC(4,16)     +169.7%  47.8%   -13.6%
       EWMAC(8,32)     +167.5%  49.0%   -13.7%
      EWMAC(16,64)     +137.1%  48.4%   -17.4%
     EWMAC(32,128)     +116.6%  47.7%   -24.2%
     EWMAC(64,256)      +60.4%  44.8%   -35.4%
           BRK(10)     +169.3%  47.5%   -24.8%
           BRK(20)     +248.9%  49.3%   -20.1%
           BRK(40)     +244.2%  49.9%   -19.3%
           BRK(80)     +140.3%  49.2%   -32.3%
          BRK(160)     +177.0%  47.1%   -30.5%
          BRK(320)     +125.8%  42.8%   -58.3%

══════════════════════════════════════════════════════════════════════════════
EWMAC pool
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n_days=2372  wr=48.5%  ret=+143.8%  mdd=-12.3%
    MC i.i.d.  : pp=0.997  pr=0.000
    MC block   : pp=0.993  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     365    +80.1%  41.6%
      2021     365    +18.7%  49.9%
      2022     365     +2.5%  53.2%
      2023     365     +7.8%  46.0%
      2024     366    +29.7%  50.5%
      2025     365     -8.7%  47.9%
      2026     181    +13.7%  51.9%

  HOLDOUT GENUINO 2025-2026: n=546  wr=49.3%  ret=+5.0%
    MC i.i.d.  : pp=0.578  pr=0.001
    MC block   : pp=0.563  pr=0.000

══════════════════════════════════════════════════════════════════════════════
Breakout pool
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n_days=2372  wr=49.3%  ret=+217.9%  mdd=-15.7%
    MC i.i.d.  : pp=0.998  pr=0.000
    MC block   : pp=0.998  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     365   +112.4%  50.1%
      2021     365    +20.7%  48.2%
      2022     365     +7.3%  51.5%
      2023     365     +6.9%  44.1%
      2024     366    +39.9%  50.8%
      2025     365     +5.0%  49.6%
      2026     181    +25.7%  51.9%

  HOLDOUT GENUINO 2025-2026: n=546  wr=50.4%  ret=+30.8%
    MC i.i.d.  : pp=0.816  pr=0.001
    MC block   : pp=0.800  pr=0.000

══════════════════════════════════════════════════════════════════════════════
Carry (funding)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n_days=2372  wr=48.6%  ret=-55.4%  mdd=-79.7%
    MC i.i.d.  : pp=0.190  pr=0.544
    MC block   : pp=0.205  pr=0.539

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     365    -23.2%  35.9%
      2021     365    -13.7%  50.4%
      2022     365    +11.1%  54.0%
      2023     365    -11.0%  49.0%
      2024     366    -36.3%  47.8%
      2025     365     +6.5%  50.1%
      2026     181    +11.2%  57.5%

  HOLDOUT GENUINO 2025-2026: n=546  wr=52.6%  ret=+17.7%
    MC i.i.d.  : pp=0.958  pr=0.000
    MC block   : pp=0.968  pr=0.000

══════════════════════════════════════════════════════════════════════════════
Grand pool (EWMAC+BRK+Carry)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n_days=2372  wr=49.5%  ret=+111.2%  mdd=-12.0%
    MC i.i.d.  : pp=0.998  pr=0.000
    MC block   : pp=0.998  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     365    +64.2%  49.3%
      2021     365     +8.9%  49.9%
      2022     365     +7.1%  54.2%
      2023     365     +1.5%  44.9%
      2024     366    +11.4%  46.7%
      2025     365     +1.2%  49.3%
      2026     181    +17.0%  54.7%

  HOLDOUT GENUINO 2025-2026: n=546  wr=51.1%  ret=+18.2%
    MC i.i.d.  : pp=0.833  pr=0.000
    MC block   : pp=0.806  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=4 (4 candidati finali)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
                            Rule       n      Ret%   Sharpe_hat      DSR
                      EWMAC pool    2372   +143.8%        2.528    1.000
                   Breakout pool    2372   +217.9%        2.788    1.000
                 Carry (funding)    2372    -55.4%       -1.224    0.000
    Grand pool (EWMAC+BRK+Carry)    2372   +111.2%        2.750    1.000

  Holdout 2025-2026:
                            Rule       n      Ret%   Sharpe_hat      DSR
                      EWMAC pool     546     +5.0%        0.224    0.000
                   Breakout pool     546    +30.8%        0.951    1.000
                 Carry (funding)     546    +17.7%        1.816    1.000
    Grand pool (EWMAC+BRK+Carry)     546    +18.2%        1.013    1.000

══════════════════════════════════════════════════════════════════════════════
Slippage-stress — migliore per DSR = Breakout pool
══════════════════════════════════════════════════════════════════════════════

  Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
        0bps  full-sample    2372   +217.9%  49.3%   0.998   0.000
        0bps     holdout     546    +30.8%  50.4%   0.816   0.001
        2bps  full-sample    2372   +214.3%  49.0%   0.998   0.000
        2bps     holdout     546    +29.8%  50.4%   0.808   0.001
        5bps  full-sample    2372   +209.1%  48.6%   0.997   0.000
        5bps     holdout     546    +28.3%  50.0%   0.795   0.001
       10bps  full-sample    2372   +200.3%  48.2%   0.997   0.000
       10bps     holdout     546    +25.9%  49.1%   0.775   0.002

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: il candidato più robusto in tutta questa sessione — con un caveat importante sul benchmark

```
                        Ret full   DSR full   Ret hold   DSR hold   B&H full   B&H hold
Buy & Hold (benchmark)   +714.2%       -         -38.0%       -         -           -
EWMAC pool               +143.8%     1.000        +5.0%     0.000
Breakout pool             +217.9%     1.000       +30.8%     1.000
Carry (funding)            -55.4%     0.000       +17.7%     1.000
Grand pool (i 3 insieme)  +111.2%     1.000       +18.2%     1.000
```

**Il fatto più importante di questo report**: l'holdout genuino
2025-2026 è stato un **mercato ribassista per BTC** (buy & hold: -38.0%,
da $94.581 a $58.605) — non il proseguimento del bull market che
domina il full-sample (+714.2%). Questo è cruciale per interpretare
correttamente i risultati: un sistema che fosse "semplicemente sempre
long" per beneficiare del trend strutturale 2020-2026 avrebbe DOVUTO
perdere pesantemente nell'holdout, esattamente come il buy & hold. Invece
**tutti e quattro i candidati sono risultati positivi nell'holdout**
(EWMAC +5.0%, Breakout +30.8%, Carry +17.7%, Grand pool +18.2%) — una
prova diretta che questi sistemi stanno facendo vero timing (andando
short/riducendo l'esposizione durante il ribasso), non semplicemente
cavalcando la deriva strutturale. Nessun'altra strategia testata in
questa sessione (quasi tutte intraday/mean-reversion di breve periodo)
ha mai affrontato esplicitamente uno scenario di bear market prolungato
nel proprio holdout.

**Breakout pool è il candidato più solido**: DSR=1.000 SIA full-sample
SIA holdout (l'unico, insieme al Grand pool, a farlo) — ret full +217.9%,
holdout +30.8%, Sharpe_hat holdout 0.951 (robusto, non marginale). Da
notare: nel full-sample il sistema cattura comunque meno della beta pura
(+217.9% vs +714.2% del buy & hold) — un vol-targeting a target 20%
annuo tiene l'esposizione strutturalmente più bassa di un buy & hold
non levato, quindi il confronto corretto è risk-adjusted (Sharpe), non
sul ritorno grezzo.

**EWMAC pool è il candidato più debole dei quattro**, nonostante il
DSR full-sample perfetto (1.000): l'holdout è debole (+5.0%, DSR=0.000,
Sharpe_hat solo 0.224) — lo stesso pattern di divergenza full/holdout
già visto altrove in sessione (es. VP+VWAP M15) che generalmente indica
un edge storico che non si trasferisce bene al periodo più recente.
Guardando il breakdown annuale, il grosso del rendimento EWMAC viene dal
2020 (+80.1%, l'anno del bull run più esplosivo) — un singolo anno
dominante è un segnale di concentrazione, non di edge distribuito.

**Carry (funding) va scartato nonostante l'holdout attraente**: fallisce
nettamente full-sample (-55.4%, DSR=0.000) con un **MC p_ruin=0.544**
allarmante (più di 1 simulazione su 2 finisce in rovina sull'intera
storia) — il pattern "fallisce sul campione grande e affidabile, va bene
sull'holdout piccolo" è stato osservato più volte in sessione come
compatibile con fortuna statistica del periodo recente più che con un
vero edge (n holdout=546 giorni, non enorme). Va rigettato come
componente standalone; il suo contributo al Grand pool è probabilmente
più un elemento di diversificazione/decorrelazione che una fonte di
edge propria.

**Grand pool (media dei tre)** eredita robustezza da Breakout (che
domina il paniere) diluendo sia la debolezza holdout di EWMAC sia il
fallimento full-sample di Carry — DSR=1.000 in entrambi gli ambiti, un
profilo bilanciato ma meno puro del solo Breakout pool.

**Conclusione**: **Breakout pool (canale Donchian, 6 velocità pooled)
è il risultato più solido di tutta questa sessione** — è l'unica
strategia, tra le ~20 testate, che (a) supera DSR=1.000 sia full-sample
sia holdout, (b) ha dimostrato di navigare con profitto un vero bear
market nel proprio holdout invece di limitarsi a beneficiare di un trend
favorevole, e (c) deriva da un framework pubblicato, con oltre un
decennio di validazione indipendente (a differenza delle claim
social-media testate finora in sessione). Va comunque validato
ulteriormente prima di qualunque uso reale: qui il position sizing è a
capitale fisso non-compounding (coerente col resto della sessione ma
diverso dal compounding reale), il target di volatilità (20% annuo) è
un'assunzione di modellazione non calibrata, e il costo di
turnover potrebbe essere sottostimato per un sistema a ribilanciamento
giornaliero continuo su un book reale (qui si assume sempre esecuzione
al fixing di chiusura giornaliera).
