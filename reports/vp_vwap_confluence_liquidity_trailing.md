# VP + VWAP Confluence — Liquidity filter & ATR Trailing Stop — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VP + VWAP Confluence — Liquidity filter & ATR Trailing Stop — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Due direzioni strutturali dopo il fallimento del regime-filter HMM:
A) LIQUIDITY: filtro di prominence del POC (volume distribution, non prezzo)
B) TRAILING: stop ATR dinamico al posto del target RR fisso
C) COMBINED. Confronto diretto vs BASELINE, stesso pool di eventi causali.

══════════════════════════════════════════════════════════════════════════════
BASELINE (RR=3.0 fisso, no filtro)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=160  wr=37.5%  ret=+8.6%  mdd=-12.5%
    MC i.i.d.  : pp=0.667  pr=0.000
    MC block   : pp=0.720  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      24     +3.6%  41.7%
      2021      18     -5.2%  33.3%
      2022      25     -2.9%  32.0%
      2023      25     +6.9%  44.0%
      2024      23     +1.3%  30.4%
      2025      32     -0.9%  34.4%
      2026      13     +5.5%  53.8%

  HOLDOUT GENUINO 2025-2026: n=45  wr=40.0%  ret=+4.7%  mdd=-6.8%
    MC i.i.d.  : pp=0.673  pr=0.000
    MC block   : pp=0.751  pr=0.000

  Slippage-stress (sopra frizione base 0.14% RT già inclusa):
    Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
          0bps  full-sample     160     +8.6%  37.5%   0.667   0.000
          0bps     holdout      45     +4.7%  40.0%   0.673   0.000
          2bps  full-sample     160     +2.9%  37.5%   0.552   0.000
          2bps     holdout      45     +2.7%  40.0%   0.598   0.000
          5bps  full-sample     160     -5.5%  36.9%   0.386   0.001
          5bps     holdout      45     -0.3%  40.0%   0.479   0.000
         10bps  full-sample     160    -19.6%  36.9%   0.158   0.013
         10bps     holdout      45     -5.2%  40.0%   0.301   0.000

══════════════════════════════════════════════════════════════════════════════
LIQUIDITY (prominence POC >= mediana)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=87  wr=32.2%  ret=-6.4%  mdd=-14.5%
    MC i.i.d.  : pp=0.330  pr=0.000
    MC block   : pp=0.221  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      11     +1.6%  36.4%
      2021      15     -1.6%  33.3%
      2022      14     -5.4%  21.4%
      2023      14     -2.9%  35.7%
      2024      10     +0.8%  30.0%
      2025      14     -0.2%  28.6%
      2026       9     +1.4%  44.4%

  HOLDOUT GENUINO 2025-2026: n=23  wr=34.8%  ret=+1.1%  mdd=-4.4%
    MC i.i.d.  : pp=0.544  pr=0.000
    MC block   : pp=0.605  pr=0.000

  Slippage-stress (sopra frizione base 0.14% RT già inclusa):
    Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
          0bps  full-sample      87     -6.4%  32.2%   0.330   0.000
          0bps     holdout      23     +1.1%  34.8%   0.544   0.000
          2bps  full-sample      87     -9.7%  32.2%   0.250   0.000
          2bps     holdout      23     +0.2%  34.8%   0.493   0.000
          5bps  full-sample      87    -14.8%  32.2%   0.159   0.000
          5bps     holdout      23     -1.1%  34.8%   0.425   0.000
         10bps  full-sample      87    -23.3%  32.2%   0.060   0.004
         10bps     holdout      23     -3.3%  34.8%   0.313   0.000

══════════════════════════════════════════════════════════════════════════════
TRAILING (ATR chandelier, no filtro)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=160  wr=38.8%  ret=-23.0%  mdd=-25.1%
    MC i.i.d.  : pp=0.036  pr=0.002
    MC block   : pp=0.008  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      24     +0.8%  37.5%
      2021      18     -7.4%  44.4%
      2022      25     -3.5%  28.0%
      2023      25     +0.5%  60.0%
      2024      23     -9.0%  21.7%
      2025      32     -3.7%  37.5%
      2026      13     -0.8%  46.2%

  HOLDOUT GENUINO 2025-2026: n=45  wr=40.0%  ret=-4.4%  mdd=-6.5%
    MC i.i.d.  : pp=0.259  pr=0.000
    MC block   : pp=0.182  pr=0.000

  Slippage-stress (sopra frizione base 0.14% RT già inclusa):
    Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
          0bps  full-sample     160    -23.0%  38.8%   0.036   0.002
          0bps     holdout      45     -4.4%  40.0%   0.259   0.000
          2bps  full-sample     160    -28.6%  38.1%   0.015   0.009
          2bps     holdout      45     -6.4%  40.0%   0.183   0.000
          5bps  full-sample     160    -37.1%  36.2%   0.004   0.073
          5bps     holdout      45     -9.4%  40.0%   0.100   0.000
         10bps  full-sample     160    -51.2%  33.8%   0.000   0.554
         10bps     holdout      45    -14.3%  37.8%   0.027   0.000

══════════════════════════════════════════════════════════════════════════════
COMBINED (liquidity + trailing)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=87  wr=35.6%  ret=-20.2%  mdd=-23.2%
    MC i.i.d.  : pp=0.022  pr=0.000
    MC block   : pp=0.001  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      11     +2.6%  36.4%
      2021      15     -5.5%  40.0%
      2022      14     -2.3%  42.9%
      2023      14     -4.2%  42.9%
      2024      10     -5.5%  20.0%
      2025      14     -2.9%  28.6%
      2026       9     -2.4%  33.3%

  HOLDOUT GENUINO 2025-2026: n=23  wr=30.4%  ret=-5.3%  mdd=-6.3%
    MC i.i.d.  : pp=0.128  pr=0.000
    MC block   : pp=0.030  pr=0.000

  Slippage-stress (sopra frizione base 0.14% RT già inclusa):
    Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
          0bps  full-sample      87    -20.2%  35.6%   0.022   0.000
          0bps     holdout      23     -5.3%  30.4%   0.128   0.000
          2bps  full-sample      87    -23.6%  34.5%   0.009   0.000
          2bps     holdout      23     -6.2%  30.4%   0.095   0.000
          5bps  full-sample      87    -28.7%  33.3%   0.002   0.001
          5bps     holdout      23     -7.5%  30.4%   0.056   0.000
         10bps  full-sample      87    -37.2%  31.0%   0.000   0.031
         10bps     holdout      23     -9.7%  30.4%   0.021   0.000

══════════════════════════════════════════════════════════════════════════════
RIEPILOGO
══════════════════════════════════════════════════════════════════════════════

  Variante                                        n      Ret%      WR    MCpp   n(h)   Ret%(h)  MCpp(h)
  BASELINE (RR=3.0 fisso, no filtro)            160     +8.6%  37.5%   0.667     45     +4.7%    0.673
  LIQUIDITY (prominence POC >= mediana)          87     -6.4%  32.2%   0.330     23     +1.1%    0.544
  TRAILING (ATR chandelier, no filtro)          160    -23.0%  38.8%   0.036     45     -4.4%    0.259
  COMBINED (liquidity + trailing)                87    -20.2%  35.6%   0.022     23     -5.3%    0.128

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: entrambe le direzioni strutturali falliscono — la baseline resta la migliore

```
Variante                              n    Ret full   MCpp full   n(h)   Ret hold   MCpp hold
BASELINE (RR=3.0 fisso)              160     +8.6%      0.667      45     +4.7%       0.673
LIQUIDITY (prominence filter)         87     -6.4%      0.330      23     +1.1%       0.544
TRAILING (ATR chandelier)            160    -23.0%      0.036      45     -4.4%       0.259
COMBINED                              87    -20.2%      0.022      23     -5.3%       0.128
```

**Nota preliminare**: la BASELINE qui (+8.6% full / +4.7% holdout) è più
debole del report originale `vp_vwap_confluence.md` (+12.8%/+6.2%) per un
motivo noto e atteso, non un regresso nascosto: qui è in vigore la
frizione Bybit obbligatoria completa (0.14% round-trip: taker 0.055% +
slippage 0.015% per lato), contro lo 0.11% round-trip taker-only del
report originale. Stesso pool di eventi (n=160/45), solo costo di
esecuzione più realistico.

### A) LIQUIDITY (prominence del POC) — bocciata

Il filtro "richiedi che il POC sia un vero livello di consenso (volume
concentrato, non disperso)" **peggiora il risultato invece di
migliorarlo**: ret full-sample -6.4% (da +8.6%), MC p_profit crolla da
0.667 a 0.330. Il campione si dimezza (160→87) ma, come nel caso del
regime-filter HMM, il taglio non è selettivo verso trade di qualità
peggiore — rimuove trade buoni e cattivi indiscriminatamente rispetto
alla loro profittabilità reale. Diagnosi: la "prominence" del POC misura
quanto un livello sia un vero consenso storico di *lungo periodo* (20
giorni), ma l'edge di questa strategia è un rimbalzo di *breve periodo*
(poche ore) su quel livello — un profilo relativamente piatto/disperso
non impedisce che, IN QUEL MOMENTO, il livello funga comunque da
supporto/resistenza efficace per un pullback di trend a breve termine.
L'unica nota parzialmente positiva: sull'holdout il ret è comunque
positivo (+1.1%) anche se peggiore del baseline (+4.7%) — non abbastanza
per giustificare il filtro.

### B) TRAILING (ATR chandelier) — bocciata nettamente

Risultato molto peggiore: ret full-sample -23.0% (da +8.6%), MC p_profit
crolla a 0.036 (quasi nessuna possibilità di profitto nelle simulazioni),
compare anche un p_ruin non-nullo (0.002 iid) mai visto nella baseline.
Interessante: il **win rate sale leggermente** (38.8% vs 37.5%) — più
trade escono almeno in pareggio/piccolo utile grazie al trailing — ma il
**payoff medio crolla**, perché il meccanismo dà via 2.0×ATR di
guadagno già maturato prima di confermare l'uscita, e molti trade che
con il target fisso RR=3.0 avrebbero chiuso al massimo profitto vengono
invece "restituiti" quasi interamente al mercato prima che il trailing
si attivi/segua abbastanza stretto. **Diagnosi**: l'edge di questa
strategia NON è un vero trend che corre lontano — è un rimbalzo
delimitato che tipicamente raggiunge un'estensione equivalente a ~3×il
rischio iniziale prima di esaurirsi (esattamente il valore RR=3.0 già
selezionato nel report base) e poi inverte. Un target fisso a quella
distanza CATTURA l'edge; un trailing stop progettato per "lasciar
correre i vincitori" presume un tipo di edge diverso (trend persistente)
che qui non esiste, e quindi restituisce sistematicamente il profitto
già maturato.

### C) COMBINED — nessuna sorpresa

Combina i due fallimenti (n=87, ret -20.2% full-sample, MC p_profit
0.022): il filtro di liquidità non salva il trailing, e viceversa.

**Conclusione dell'iterazione**: dopo tre tentativi di ottimizzazione
strutturale (regime HMM, filtro di liquidità/prominence, trailing stop
ATR), **nessuno migliora la baseline originale** — anzi tutti la
peggiorano, in modi diversi ma diagnosticabili in ciascun caso. Questo è
un segnale che la configurazione base (POC/VWAP confluence + trend-context
+ stop strutturale + target fisso RR=3.0) è già relativamente ben
calibrata rispetto alla vera natura dell'edge (un rimbalzo di trend a
breve termine con estensione tipica ~3× il rischio), e che ulteriori
raffinamenti strutturali in queste direzioni specifiche non aggiungono
valore. Non proseguo con altre varianti di queste due direzioni (rischio
di scivolare in tuning mascherato) — la baseline (ora ri-validata sotto
la frizione Bybit rigorosa: full +8.6%/holdout +4.7%, MC p_profit
0.667/0.673, p_ruin sempre 0.000) resta la configurazione raccomandata.
