# VWAP Mean-Reversion v1 — re-check con fee reali Bybit derivatives

```
══════════════════════════════════════════════════════════════════════════════
VWAP Mean-Reversion v1 — re-check con fee reali Bybit derivatives
══════════════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════════════
Scenario: Taker reale Bybit (0.055%+0.055%)  (round-trip = 0.110%)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=2916  wr=60.5%  ret=-221.4%  mdd=-220.9%
    Exit: TP=1713  SL=554  time=649
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370    -24.1%  60.3%
      2021     637    -51.4%  59.8%
      2022     635    -49.9%  58.4%
      2023     362    -35.9%  59.1%
      2024     419    -19.2%  64.9%
      2025     356    -29.8%  61.0%
      2026     137    -11.2%  62.0%

  HOLDOUT GENUINO 2025-2026: n=493  wr=61.3%  ret=-40.9%  mdd=-41.9%
    MC i.i.d.  : pp=0.001  pr=0.169
    MC block   : pp=0.000  pr=0.142

══════════════════════════════════════════════════════════════════════════════
Scenario: Assunzione originale sessione (0.04%+0.04%)  (round-trip = 0.080%)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=3912  wr=62.9%  ret=-182.7%  mdd=-182.1%
    Exit: TP=2266  SL=725  time=921
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370    -15.9%  63.5%
      2021    1391    -56.4%  62.5%
      2022     799    -45.6%  61.1%
      2023     362    -26.5%  62.7%
      2024     497    -11.2%  66.0%
      2025     356    -19.7%  62.9%
      2026     137     -7.4%  65.0%

  HOLDOUT GENUINO 2025-2026: n=493  wr=63.5%  ret=-27.1%  mdd=-29.2%
    MC i.i.d.  : pp=0.013  pr=0.005
    MC block   : pp=0.011  pr=0.005

══════════════════════════════════════════════════════════════════════════════
Scenario: Mista: entry taker / exit maker (0.055%+0.02%)  (round-trip = 0.075%)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=3963  wr=63.4%  ret=-175.1%  mdd=-173.3%
    Exit: TP=2298  SL=734  time=931
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370    -14.6%  64.3%
      2021    1391    -53.3%  62.7%
      2022     799    -43.4%  61.7%
      2023     362    -24.9%  64.6%
      2024     548    -14.2%  65.7%
      2025     356    -18.0%  62.9%
      2026     137     -6.8%  65.7%

  HOLDOUT GENUINO 2025-2026: n=493  wr=63.7%  ret=-24.8%  mdd=-27.2%
    MC i.i.d.  : pp=0.022  pr=0.002
    MC block   : pp=0.019  pr=0.003

══════════════════════════════════════════════════════════════════════════════
Scenario: Piena maker: limit su entrambi i lati (0.02%+0.02%)  (round-trip = 0.040%)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=4975  wr=65.8%  ret=-82.8%  mdd=-114.1%
    Exit: TP=2918  SL=906  time=1151
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370     -5.1%  67.6%
      2021    1391    -31.3%  63.9%
      2022     853    -38.6%  62.6%
      2023     389    -15.4%  66.8%
      2024     753     -6.2%  66.9%
      2025     421     -5.0%  65.8%
      2026     798    +18.7%  70.1%

  HOLDOUT GENUINO 2025-2026: n=1219  wr=68.6%  ret=+13.7%  mdd=-16.4%
    MC i.i.d.  : pp=0.750  pr=0.000
    MC block   : pp=0.692  pr=0.001

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: la fee reale cambia il quadro, ma non chiude la partita

```
Scenario                         RT fee   Full ret   Holdout ret   Holdout MC pp
Taker reale Bybit                 0.110%   -221.4%      -40.9%         0.001
Assunzione originale sessione     0.080%   -182.7%      -27.1%         0.013
Mista (entry taker/exit maker)    0.075%   -175.1%      -24.8%         0.022
Piena maker (limit su 2 lati)     0.040%    -82.8%      +13.7%         0.750
```

**Primo risultato positivo genuino su questa strategia in tutta la sessione**:
con fee piena maker (0,02%+0,02%, possibile perché sia l'entry — soglia
Z-score nota in anticipo — sia l'uscita — tocco VWAP, anch'esso noto in
anticipo — sono segnali "a livello di prezzo" candidati naturali per
ordini limit) l'holdout 2025-2026 diventa **+13.7%**, con MC i.i.d.
p_profit=0.750 e MC block p_profit=0.692 — la prima volta che il Monte
Carlo dà un giudizio favorevole su questa strategia.

**Tre avvertenze prima di considerarla pronta:**

1. **Il full-sample resta negativo** (-82.8% su 6.5 anni) anche nello
   scenario più favorevole. Il breakdown per anno mostra perdite pesanti
   2020-2023 (-5.1% a -38.6%) che si affievoliscono e diventano positive
   solo nel 2024-2026 (-6.2%, -5.0%, **+18.7%**) — lo stesso pattern
   "edge concentrato negli anni recenti" che ha già tradito la strategia
   ADP in questa sessione. Con un solo scenario a fee piena maker che
   mostra miglioramento, non è ancora possibile distinguere un vero
   cambio di regime da un tratto favorevole recente.
2. **100% di fill rate sugli ordini limit è un'assunzione ottimistica.**
   Il backtest assume che ogni segnale (sia in entry che in uscita) venga
   eseguito a limit price con certezza. Nella realtà un ordine limit
   piazzato esattamente alla soglia Z-score o al livello VWAP potrebbe
   non riempirsi (il prezzo tocca il livello e rimbalza via senza
   eseguire l'ordine) — specialmente per l'entry, dove per definizione si
   piazza il limit al prezzo più estremo del movimento in quel momento.
   Un fill rate reale <100% ridurrebbe sia il numero di trade eseguiti
   sia, probabilmente, il win rate (i trade "mancati" per fill parziale
   tendono a essere quelli con reversal più netto e veloce).
3. **Lo scenario "misto" (entry a mercato, uscita a limit) non aiuta**:
   -24.8% holdout, quasi identico all'assunzione originale. Il beneficio
   arriva solo se ENTRAMBI i lati sono maker — un'esecuzione a mercato
   anche solo in entrata (spesso necessaria per non perdere il segnale se
   il prezzo non ripassa esattamente dalla soglia) vanifica gran parte
   del guadagno.

**Conclusione**: la fee reale Bybit rende questa l'unica strategia di
tutta la fase di "sfruttamento dell'edge VWAP" (5 tentativi precedenti,
tutti falliti) a mostrare un risultato holdout genuinamente positivo con
supporto Monte Carlo forte — ma resta condizionata a (a) esecuzione
full-maker su entrambi i lati, verificabile solo con dati reali di fill
rate, e (b) un edge full-sample ancora negativo con miglioramento
concentrato negli ultimi 2-3 anni, da confermare con più tempo prima di
allocare capitale.
