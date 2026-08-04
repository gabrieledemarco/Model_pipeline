# Fade ICT standalone — Walk-Forward Optimization stop × target

```
══════════════════════════════════════════════════════════════════════════════
Fade ICT standalone — Walk-Forward Optimization stop (SWING_LOOKBACK) × target (RR)
══════════════════════════════════════════════════════════════════════════════

[WFO] 35 finestre (6m IS / 2m OOS / step 2m)
[WFO] Griglia: SWING_LOOKBACK [5, 10, 15] × RR [1.5, 2.0, 2.5, 3.0] = 12 combinazioni/finestra

[WFO] Selezione per finestra (35 finestre valide):
  SWING_LOOKBACK più scelto: {15: 30, 10: 5}
  RR più scelto             : {2.5: 14, 3.0: 13, 2.0: 6, 1.5: 2}

══════════════════════════════════════════════════════════════════════════════
RISULTATI AGGREGATI — walk-forward OOS (stop+target ottimizzati per finestra)
══════════════════════════════════════════════════════════════════════════════

  FULL WFO-OOS: n=1255  wr=51.0%  ret=+426.5%  mdd=-10.2%
    MC i.i.d.  : pp=1.000  pr=0.000
    MC block   : pp=1.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     113    +32.5%  48.7%
      2021     226    +76.0%  51.3%
      2022     221   +110.9%  54.8%
      2023     201    +46.8%  46.3%
      2024     216    +95.2%  52.8%
      2025     209    +37.6%  49.8%
      2026      69    +27.5%  53.6%

  HOLDOUT GENUINO 2025-2026 (sub-slice del WFO-OOS): n=278  wr=50.7%  ret=+65.1%
    MC i.i.d.  : pp=0.990  pr=0.000
    MC block   : pp=0.993  pr=0.000

══════════════════════════════════════════════════════════════════════════════
CONFRONTO — WFO (stop/target dinamici) vs RR=3.0 fisso (ict_fade_standalone.md)
══════════════════════════════════════════════════════════════════════════════

                             n      Ret%      WR    MC pp
          WFO dinamico    1255   +426.5%  51.0%   1.000
   RR=3.0 fisso (rif.)    1399   +512.7%   47.0%   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: la selezione causale per finestra CONFERMA il risultato a RR fisso

```
                             n      Ret%      WR    MC p(profit)   MC p(ruin)
WFO dinamico (causale)    1255   +426.5%   51.0%       1.000          0.000
RR=3.0 fisso (rif.)       1399   +512.7%   47.0%       1.000          0.000
```

**Nessuna correzione DSR è stata necessaria qui**: a differenza del report base (dove RR
era scelto a posteriori su un grid dell'intera storia, richiedendo la correzione DSR per
il selection bias), qui ogni finestra seleziona SWING_LOOKBACK/RR usando SOLO dati IS,
poi applica quella scelta a barre OOS mai viste durante la selezione — il processo è
causale per costruzione, lo stesso principio già usato per VWAP MR e il regime-filter HMM
in questa sessione.

**Il risultato converge**: ordine di grandezza comparabile (+426.5% vs +512.7%), win rate
addirittura più alto sotto WFO (51.0% vs 47.0%, sopra la soglia del 50%), **positivo in
ogni singolo anno 2020-2026** in entrambi i casi, MC p(profit)=1.000 e p(ruin)=0.000
identici. Due metodologie di selezione indipendenti (grid+DSR sull'intera storia vs
IS-scan causale per finestra) che convergono su un risultato simile è la prova di
robustezza più solida raccolta in questa sessione — riduce sensibilmente la probabilità
che il risultato a RR fisso fosse un artefatto della selezione a posteriori.

**Parametri selezionati più spesso**: `SWING_LOOKBACK=15` (30/35 finestre — più largo del
valore base 10), `RR` diviso quasi equamente tra 2.5 (14 finestre) e 3.0 (13 finestre).
Raccomandazione operativa per il paper trading: `SWING_LOOKBACK=15`, `RR=3.0` (dettagli
completi in `docs/ICT_FADE_STANDALONE_STRATEGY_SPEC.md`, Sezioni 5 e 8; report HTML con
grafici in `reports/ict_fade_wfo_report.html`).
