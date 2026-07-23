# FVG Sniper — Filtro di regime (efficiency ratio) via WFO

```
══════════════════════════════════════════════════════════════════════════════
FVG Sniper — Filtro di regime (efficiency ratio) via Walk-Forward Optimization
══════════════════════════════════════════════════════════════════════════════

Segue la diagnosi del 2025 (chop estremo, ER=0.0025 vs 0.007-0.053 negli
altri anni): filtro ER selezionato causalmente per finestra (WFO 6m IS/2m
OOS), non scelto a posteriori sull'intera storia (hindsight bias).

[WFO] 35 finestre (6m IS / 2m OOS / step 2m)
[WFO] Griglia: ER lookback [96, 168, 336] x ER minima [0.0, 0.03, 0.05, 0.08, 0.12] = 15 combinazioni/finestra

[WFO] Selezione per finestra (35 finestre valide):
  ER lookback più scelto : {336: 16, 96: 13, 168: 6}
  ER minima più scelta   : {0.03: 17, 0.12: 6, 0.0: 5, 0.08: 4, 0.05: 3}

══════════════════════════════════════════════════════════════════════════════
RISULTATI AGGREGATI — walk-forward OOS (filtro ER selezionato per finestra)
══════════════════════════════════════════════════════════════════════════════

  FULL WFO-OOS: n=1100  wr=30.4%  ret=+14.5%  mdd=-46.2%
    MC i.i.d.  : pp=0.629  pr=0.019
    MC block   : pp=0.613  pr=0.038

  Breakdown per anno:
        Year       n      Ret%      WR
        2020      64    +20.8%  37.5%
        2021     265    +20.5%  30.6%
        2022     241    +32.9%  34.0%
        2023     137     -8.1%  29.2%
        2024     205     -1.2%  30.7%
        2025     164    -60.1%  20.7%
        2026      24     +9.7%  41.7%

  HOLDOUT GENUINO 2025-2026 (sub-slice del WFO-OOS): n=188  wr=23.4%  ret=-50.4%
    MC i.i.d.  : pp=0.016  pr=0.517
    MC block   : pp=0.046  pr=0.516

══════════════════════════════════════════════════════════════════════════════
CONFRONTO — WFO con filtro ER vs v2 senza filtro (create_fvg_sniper_v2_report.py)
══════════════════════════════════════════════════════════════════════════════

                          n      Ret%      WR
  WFO + filtro ER        1100    +14.5%   30.4%
  v2 nessun filtro (rif.) 1995   +200.9%   33.2%   (full-sample, non-WFO, per riferimento)

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```
