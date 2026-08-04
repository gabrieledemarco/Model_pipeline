# Grid Trading su range settimana/giorno precedente — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Grid Trading su range settimana/giorno precedente — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Grid long-only classico: N livelli nel range del periodo precedente,
buy-low/sell-high per cella indipendente. Stop di sicurezza sotto il
range, chiusura forzata a fine periodo. Fee maker su fill griglia,
taker sulle chiusure forzate.

══════════════════════════════════════════════════════════════════════════════
VARIANTE: WEEKLY (barre 1H, range = settimana precedente)
══════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────
N_GRID = 5
──────────────────────────────────────────────────────────────────────────────

  FULL-SAMPLE: n=1539  wr=56.5% (p=0.0000 vs 50%)  ret=-14.7%  mdd=-15.6%
    buy=1539  chiusure-stop=483  chiusure-fine-periodo=237
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     227     +0.4%  67.4%
      2021     232     -2.3%  59.9%
      2022     255     -5.7%  46.3%
      2023     244     -1.5%  56.1%
      2024     217     -1.5%  57.6%
      2025     233     -2.6%  53.6%
      2026     131     -1.4%  55.0%

  HOLDOUT GENUINO 2025-2026: n=364  wr=54.1%  ret=-4.1%
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

──────────────────────────────────────────────────────────────────────────────
N_GRID = 10
──────────────────────────────────────────────────────────────────────────────

  FULL-SAMPLE: n=5339  wr=73.1% (p=0.0000 vs 50%)  ret=-33.4%  mdd=-34.0%
    buy=5339  chiusure-stop=1006  chiusure-fine-periodo=466
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     753     -0.5%  78.8%
      2021     820     -5.4%  75.9%
      2022     844    -12.2%  65.6%
      2023     839     -3.4%  72.3%
      2024     795     -2.9%  75.6%
      2025     826     -5.7%  72.0%
      2026     462     -3.2%  71.2%

  HOLDOUT GENUINO 2025-2026: n=1288  wr=71.7%  ret=-8.9%
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

──────────────────────────────────────────────────────────────────────────────
N_GRID = 20
──────────────────────────────────────────────────────────────────────────────

  FULL-SAMPLE: n=18369  wr=84.4% (p=0.0000 vs 50%)  ret=-75.9%  mdd=-76.0%
    buy=18369  chiusure-stop=1985  chiusure-fine-periodo=913
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020    2595     -2.9%  87.8%
      2021    3032    -11.5%  86.9%
      2022    2734    -26.4%  78.8%
      2023    2830     -8.1%  83.6%
      2024    2707     -7.5%  85.8%
      2025    2876    -12.5%  83.9%
      2026    1595     -6.9%  83.2%

  HOLDOUT GENUINO 2025-2026: n=4471  wr=83.7%  ret=-19.5%
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (griglia N_GRID) — WEEKLY (barre 1H, range = settimana precedente)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
   N_GRID       n      Ret%   Sharpe_hat      DSR
        5    1539    -14.7%       -8.399    0.000
       10    5339    -33.4%      -12.992    0.000
       20   18369    -75.9%      -20.504    0.000

  Holdout 2025-2026:
   N_GRID       n      Ret%   Sharpe_hat      DSR
        5     364     -4.1%       -5.947    0.000
       10    1288     -8.9%       -8.624    0.000
       20    4471    -19.5%      -12.868    0.000

══════════════════════════════════════════════════════════════════════════════
Fee/slippage sensitivity — N_GRID migliore per DSR = 5 — WEEKLY (barre 1H, range = settimana precedente)
══════════════════════════════════════════════════════════════════════════════

                      Scenario       Scope       n      Ret%      WR    MC pp    MC pr
    Mix maker/taker, 0bps slip  full-sample    1539    -14.7%  56.5%   0.000   0.000
    Mix maker/taker, 0bps slip     holdout     364     -4.1%  54.1%   0.000   0.000
        Tutto taker, 0bps slip  full-sample    1539    -15.3%  56.5%   0.000   0.000
        Tutto taker, 0bps slip     holdout     364     -4.2%  54.1%   0.000   0.000
    Mix maker/taker, 2bps slip  full-sample    1539    -15.3%  56.5%   0.000   0.000
    Mix maker/taker, 2bps slip     holdout     364     -4.2%  54.1%   0.000   0.000
    Mix maker/taker, 5bps slip  full-sample    1539    -16.2%  56.3%   0.000   0.000
    Mix maker/taker, 5bps slip     holdout     364     -4.4%  53.6%   0.000   0.000
   Mix maker/taker, 10bps slip  full-sample    1539    -17.8%  56.0%   0.000   0.000
   Mix maker/taker, 10bps slip     holdout     364     -4.8%  53.3%   0.000   0.000

══════════════════════════════════════════════════════════════════════════════
VARIANTE: DAILY (barre 15m, range = giorno precedente)
══════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────
N_GRID = 5
──────────────────────────────────────────────────────────────────────────────

  FULL-SAMPLE: n=11594  wr=57.4% (p=0.0000 vs 50%)  ret=-50.0%  mdd=-50.0%
    buy=11594  chiusure-stop=3475  chiusure-fine-periodo=1791
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020    1614     -4.5%  62.2%
      2021    1678    -11.6%  55.8%
      2022    1832    -11.1%  53.7%
      2023    1888     -4.7%  59.5%
      2024    1815     -6.9%  57.5%
      2025    1847     -7.0%  56.5%
      2026     920     -4.2%  56.2%

  HOLDOUT GENUINO 2025-2026: n=2767  wr=56.4%  ret=-11.2%
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

──────────────────────────────────────────────────────────────────────────────
N_GRID = 10
──────────────────────────────────────────────────────────────────────────────

  FULL-SAMPLE: n=40117  wr=74.1% (p=0.0000 vs 50%)  ret=-112.6%  mdd=-112.6%
    buy=40117  chiusure-stop=7089  chiusure-fine-periodo=3524
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020    5769    -10.4%  77.9%
      2021    5948    -24.5%  74.0%
      2022    6300    -24.4%  71.7%
      2023    6333    -11.6%  74.6%
      2024    6301    -16.0%  74.3%
      2025    6354    -16.2%  73.3%
      2026    3112     -9.4%  72.3%

  HOLDOUT GENUINO 2025-2026: n=9466  wr=73.0%  ret=-25.6%
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

──────────────────────────────────────────────────────────────────────────────
N_GRID = 20
──────────────────────────────────────────────────────────────────────────────

  FULL-SAMPLE: n=132406  wr=82.9% (p=0.0000 vs 50%)  ret=-269.4%  mdd=-269.4%
    buy=132406  chiusure-stop=13873  chiusure-fine-periodo=6903
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020   19920    -27.7%  87.3%
      2021   20101    -55.9%  84.8%
      2022   20743    -56.1%  82.5%
      2023   20365    -29.9%  79.3%
      2024   20707    -38.6%  83.9%
      2025   20564    -38.8%  80.5%
      2026   10006    -22.3%  81.5%

  HOLDOUT GENUINO 2025-2026: n=30570  wr=80.9%  ret=-61.1%
    MC i.i.d.  : pp=0.000  pr=0.000
    MC block   : pp=0.000  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (griglia N_GRID) — DAILY (barre 15m, range = giorno precedente)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
   N_GRID       n      Ret%   Sharpe_hat      DSR
        5   11594    -50.0%      -25.571    0.000
       10   40117   -112.6%      -38.258    0.000
       20  132406   -269.4%      -63.234    0.000

  Holdout 2025-2026:
   N_GRID       n      Ret%   Sharpe_hat      DSR
        5    2767    -11.2%      -16.055    0.000
       10    9466    -25.6%      -24.169    0.000
       20   30570    -61.1%      -39.678    0.000

══════════════════════════════════════════════════════════════════════════════
Fee/slippage sensitivity — N_GRID migliore per DSR = 5 — DAILY (barre 15m, range = giorno precedente)
══════════════════════════════════════════════════════════════════════════════

                      Scenario       Scope       n      Ret%      WR    MC pp    MC pr
    Mix maker/taker, 0bps slip  full-sample   11594    -50.0%  57.4%   0.000   0.000
    Mix maker/taker, 0bps slip     holdout    2767    -11.2%  56.4%   0.000   0.000
        Tutto taker, 0bps slip  full-sample   11594    -54.4%  57.0%   0.000   0.000
        Tutto taker, 0bps slip     holdout    2767    -12.3%  56.0%   0.000   0.000
    Mix maker/taker, 2bps slip  full-sample   11594    -54.6%  56.9%   0.000   0.000
    Mix maker/taker, 2bps slip     holdout    2767    -12.3%  56.0%   0.000   0.000
    Mix maker/taker, 5bps slip  full-sample   11594    -61.6%  55.6%   0.000   0.000
    Mix maker/taker, 5bps slip     holdout    2767    -14.0%  53.6%   0.000   0.000
   Mix maker/taker, 10bps slip  full-sample   11594    -73.1%  52.0%   0.000   0.000
   Mix maker/taker, 10bps slip     holdout    2767    -16.7%  47.6%   0.000   0.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: fallimento netto e istruttivo — tanti piccoli vinti, pochi grandi persi

```
                WEEKLY (1H)                        DAILY (15m)
N_GRID    n       WR      Ret full   Ret hold    n       WR      Ret full   Ret hold
   5    1,539   56.5%     -14.7%      -4.1%    11,594   57.4%     -50.0%    -11.2%
  10    5,339   73.1%     -33.4%      -8.9%    40,117   74.1%    -112.6%    -25.6%
  20   18,369   84.4%     -75.9%     -19.5%   132,406   82.9%    -269.4%    -61.1%
```

**Fallisce su tutta la linea**: DSR=0.000 su ENTRAMBE le varianti, TUTTI
gli N_GRID, full-sample E holdout — nessuna eccezione. MC p_profit=0.000
ovunque: nelle simulazioni Monte Carlo, letteralmente 0% dei path
risultano profittevoli. È uno dei risultati più nettamente negativi di
tutta la sessione.

**Il pattern è il classico "tante piccole vincite, poche grandi perdite"**:
il win rate SALE vistosamente con la finezza del grid (56%→84% weekly,
57%→83% daily) — più celle, più oscillazioni catturate, più round-trip
vincenti — ma il ritorno PEGGIORA drammaticamente nello stesso verso
(-14.7%→-75.9% weekly; -50.0%→-269.4% daily). Un win rate dell'84% con un
ritorno di -75.9% è la firma inequivocabile di una asimmetria di payoff
negativa: molte vincite piccole (un salto di livello) contro poche
perdite enormi (lo stop di sicurezza chiude TUTTE le celle aperte in un
colpo solo quando il range viene rotto al ribasso).

**Meccanismo**: il grid assume implicitamente che il range della
settimana/giorno precedente sia un buon stimatore del range del periodo
corrente (mean-reversion locale). Su BTCUSDT questo è sistematicamente
falso: i range NON sono stazionari da un periodo all'altro — BTC tende
(trend, non oscillazione pura), quindi il prezzo rompe regolarmente sotto
il minimo del periodo precedente. Quando questo accade, TUTTE le celle
ancora aperte (potenzialmente fino a N_GRID posizioni simultanee)
vengono chiuse in perdita nello stesso momento — un evento raro ma il
cui impatto aggregato domina su centinaia di piccoli guadagni. Coerente
con l'osservazione che gli anni più negativi (2021 -55.9%, 2022 -56.1%
su daily N_GRID=20) sono quelli con i trend/drawdown più marcati.

**DAILY è sistematicamente peggiore di WEEKLY** a parità di N_GRID
(es. N_GRID=20: -269.4% daily vs -75.9% weekly) — il range giornaliero è
più stretto e più rumoroso rispetto alla volatilità intra-day di BTC, per
cui il breakout dello stop di sicurezza è ancora più frequente
relativamente al ciclo di vita di ogni griglia.

**Fee/slippage non è la causa**: anche a 0bps di slippage e fee maker
ottimistiche sui fill di griglia, il risultato è già catastroficamente
negativo — il problema è strutturale (l'assunzione di range stazionario),
non un artefatto di costi di esecuzione. Il confronto mix-maker/tutto-taker
mostra una differenza minima (-14.7% vs -15.3% weekly N_GRID=5) — irrilevante
di fronte alla magnitudine del fallimento.

**Conclusione**: il grid trading "puro" (range storico, nessun filtro di
trend/regime) non è adatto a BTCUSDT proprio perché BTC è un asset
fortemente direzionale — esattamente l'opposto dell'ipotesi implicita del
grid trading (range-bound/mean-reverting). Un'estensione naturale (non
eseguita) sarebbe attivare il grid SOLO quando un filtro di regime (es. HMM
SIDEWAYS, già usato nella VWAP MR di questa sessione) indica un mercato
laterale, disattivandolo nei regimi di trend — ma dato che qui anche il
caso "migliore" (N_GRID=5, il meno esposto) resta solidamente negativo in
OGNI anno del campione (incluso 2020, l'unico prossimo al breakeven), il
problema sembra strutturale al prodotto BTC più che risolvibile con un
filtro di attivazione.
