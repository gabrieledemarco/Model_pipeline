# Pre-Paper-Trading Validation Checklist

```

══════════════════════════════════════════════════════════════════════════════
SECTION A — Time stability per finestra WFO (5 config validati)
══════════════════════════════════════════════════════════════════════════════

  Config                          2020          2021          2022          2023          2024          2025          2026
  MR24-t2p0-PB100x5       +0.9%(n=259)  +6.2%(n=639)  -0.2%(n=548)  -0.3%(n=450)  +0.5%(n=549)  -1.9%(n=527)  +0.0%(n=174)
  MR18-t2p5-PB100x5       +0.9%(n=183)  +6.2%(n=495)  -0.1%(n=399)  -0.1%(n=329)  +0.8%(n=411)  -2.3%(n=316)  -0.2%(n=147)
  MR48-t2p5-PB100x5       +0.8%(n=181)  +6.2%(n=461)  -0.4%(n=371)  -0.2%(n=333)  +0.7%(n=412)  -1.8%(n=336)  -0.1%(n=141)
  MR24-t2p5-PB100x5       +0.8%(n=191)  +6.4%(n=492)  -0.4%(n=392)  -0.3%(n=337)  +0.8%(n=408)  -2.1%(n=331)  -0.1%(n=149)
  MR36-t2p5-PB100x5       +0.7%(n=172)  +6.1%(n=457)  -0.6%(n=367)  -0.1%(n=327)  +0.7%(n=416)  -1.9%(n=332)  -0.1%(n=143)

  Nota: ogni cella è il return% OOS aggregato per quell'anno solare (somma dei trade le cui finestre OOS iniziano in quell'anno), non ricapitalizzato in modo indipendente — utile per vedere in QUALI anni si concentra l'edge.

══════════════════════════════════════════════════════════════════════════════
SECTION B — Holdout temporale genuino  (selezione su 27 finestre pre-2025, eval su 8 finestre 2025-2026)
══════════════════════════════════════════════════════════════════════════════
  NOTA METODOLOGICA: la config Pullback-in-Trend (T100-E5-d003) resta quella scelta nella Phase 1 originale (full-sample) — rifare da zero anche quella selezione sotto vincolo temporale rigoroso richiederebbe rieseguire l'intera Phase 1 (60 varianti) ristretta a pre-2025, un altro ordine di grandezza di calcolo. Qui isoliamo la domanda più rilevante: la scelta dei parametri MR (soglia/finestra) regge se selezionata SOLO su dati fino al 2024?

  Selezione ristretta a pre-2025 (N=27 finestre):
  ID                            n      Ret     pp    DSR  Val
  ✅ MR18-t2p5-PB100x5        1817    +7.8%  0.986  1.000
  ✅ MR24-t2p5-PB100x5        1820    +7.3%  0.981  1.000
  ✅ MR48-t2p5-PB100x5        1758    +7.2%  0.981  1.000
  ✅ MR24-t2p0-PB100x5        2445    +7.1%  0.982  1.000
  ✅ MR36-t2p5-PB100x5        1739    +6.8%  0.972  1.000
  ✅ MR18-t2p0-PB100x5        2529    +6.1%  0.960  1.000

  → Winner selezionato SOLO su pre-2025: MR18-t2p5-PB100x5  (ret=+7.8%, pp=0.986)
  Coincide con il winner full-sample (MR24-t2p0)? NO — vedi sotto

  MR18-t2p5-PB100x5 valutato SUL VERO HOLDOUT 2025-2026 (N=8 finestre, MAI viste in selezione):
    n=463  ret=-2.48%  mdd=-3.35%  pp=0.005  pr=0.000

  Per confronto, MR24-t2p0-PB100x5 (winner full-sample) sullo stesso holdout 2025-2026:
    n=701  ret=-1.90%  mdd=-2.96%  pp=0.019

══════════════════════════════════════════════════════════════════════════════
Baseline MR24-t2p0-PB100x5 (full-sample, per riferimento nelle sezioni C-F):
  n=3146  ret=+5.12%  mdd=-5.01%  n_dropped_none=94

══════════════════════════════════════════════════════════════════════════════
SECTION C — Slippage sensitivity (bps aggiuntivi su fill TP/SL)
══════════════════════════════════════════════════════════════════════════════
    Slippage       n      Ret%     MDD%      pp
        0bps    3146    +5.12%   -5.01%   0.922
        2bps    3146    +3.87%   -5.86%   0.858
        5bps    3146    +2.03%   -7.12%   0.715
       10bps    3146    -0.96%   -9.18%   0.385

══════════════════════════════════════════════════════════════════════════════
SECTION D — Gestione trade 'no-touch' (né TP né SL entro max_hold)
══════════════════════════════════════════════════════════════════════════════
  Modalità                           n      Ret%     MDD%      pp
  Scarta (comportamento attuale)    3146    +5.12%   -5.01%   0.922
  Forza chiusura a mercato        3240    +7.17%   -4.66%   0.975
  Trade coinvolti: 94 (2.9% del totale)

══════════════════════════════════════════════════════════════════════════════
SECTION E — Verifica realistica del position sizing
══════════════════════════════════════════════════════════════════════════════
  Notional implicito come % del capitale corrente (N=3240 trade):
      p50: 2.0%
      p75: 3.0%
      p90: 3.0%
      p95: 3.0%
      p99: 5.0%
      max: 5.0%

  Interpretazione: un notional-implicito mediano di 2% del capitale corrente equivale a leva reale ~0.02x; il 3° percentile (p95) implica leva ~0.03x sui trade più aggressivi (tp_f/sl_f più ampi). Verificare che questi livelli siano compatibili col margine disponibile sull'exchange target prima di tradare live.

══════════════════════════════════════════════════════════════════════════════
SECTION F — Monte Carlo: i.i.d. bootstrap vs block-bootstrap
══════════════════════════════════════════════════════════════════════════════
  Metodo                 p_profit    p_ruin    p5 MDD%   p50 MDD%
  i.i.d. bootstrap          0.922    0.0000     -4.63%     -2.46%
  block (n=10)              0.818    0.0000     -7.69%     -4.01%
  block (n=25)              0.782    0.0000     -8.54%     -4.44%
  block (n=50)              0.775    0.0000     -8.42%     -4.43%

══════════════════════════════════════════════════════════════════════════════
[DONE]
```
