# VP + VWAP Confluence v3 — Regime Filter strutturale (HMM walk-forward) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VP + VWAP Confluence v3 — Regime Filter strutturale (HMM walk-forward) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

Stessa entry logic del report base (RR=3.0, DSR full=0.884/holdout=0.498).
Aggiunta: HMM 3-stati rifittato per finestra WFO (6m IS/2m OOS), causale,
gate 'regime coerente col bias' vs baseline NO_FILTER sullo stesso pool OOS.
Frizioni Bybit: taker 0.055% + slippage 0.015%/lato = 0.14% round-trip (default).

[HMM] Distribuzione stati OOS (tutte le finestre): bear=33.3%  sideways=36.4%  bull=30.4%

══════════════════════════════════════════════════════════════════════════════
NO_FILTER (baseline)
══════════════════════════════════════════════════════════════════════════════

  FULL OOS: n=144  wr=36.1%  ret=+4.4%  mdd=-13.8%
    Exit: TP=29  SL=77  time=38
    MC i.i.d.  : pp=0.575  pr=0.000
    MC block   : pp=0.614  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      10     +2.1%  40.0%
      2021      19     -6.2%  31.6%
      2022      25     -2.9%  32.0%
      2023      25     +6.9%  44.0%
      2024      24     +0.2%  29.2%
      2025      32     -0.9%  34.4%
      2026       9     +5.2%  55.6%

  HOLDOUT GENUINO 2025-2026: n=41  wr=39.0%  ret=+4.3%  mdd=-6.8%
    MC i.i.d.  : pp=0.642  pr=0.000
    MC block   : pp=0.728  pr=0.000

  Slippage-stress (sopra la frizione base 0.14% round-trip già inclusa):
    Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
          0bps    full-OOS     144     +4.4%  36.1%   0.575   0.000
          0bps     holdout      41     +4.3%  39.0%   0.642   0.000
          2bps    full-OOS     144     -0.8%  36.1%   0.473   0.000
          2bps     holdout      41     +2.4%  39.0%   0.573   0.000
          5bps    full-OOS     144     -8.7%  35.4%   0.316   0.001
          5bps     holdout      41     -0.4%  39.0%   0.463   0.000
         10bps    full-OOS     144    -21.8%  35.4%   0.117   0.018
         10bps     holdout      41     -5.1%  39.0%   0.293   0.000

══════════════════════════════════════════════════════════════════════════════
REGIME_MATCH (HMM gate esatto)
══════════════════════════════════════════════════════════════════════════════

  FULL OOS: n=75  wr=34.7%  ret=-5.3%  mdd=-10.0%
    Exit: TP=13  SL=41  time=21
    MC i.i.d.  : pp=0.344  pr=0.000
    MC block   : pp=0.233  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2021       7     -2.5%  28.6%
      2022       9     -1.8%  33.3%
      2023      14     -1.0%  42.9%
      2024      17     +5.1%  35.3%
      2025      20     -8.3%  25.0%
      2026       5     +1.1%  40.0%

  HOLDOUT GENUINO 2025-2026: n=25  wr=28.0%  ret=-7.2%  mdd=-10.2%
    MC i.i.d.  : pp=0.161  pr=0.000
    MC block   : pp=0.013  pr=0.000

  Slippage-stress (sopra la frizione base 0.14% round-trip già inclusa):
    Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
          0bps    full-OOS      75     -5.3%  34.7%   0.344   0.000
          0bps     holdout      25     -7.2%  28.0%   0.161   0.000
          2bps    full-OOS      75     -8.2%  34.7%   0.263   0.000
          2bps     holdout      25     -8.4%  28.0%   0.126   0.000
          5bps    full-OOS      75    -12.4%  34.7%   0.170   0.000
          5bps     holdout      25    -10.1%  28.0%   0.088   0.000
         10bps    full-OOS      75    -19.6%  33.3%   0.065   0.000
         10bps     holdout      25    -13.0%  28.0%   0.046   0.000

══════════════════════════════════════════════════════════════════════════════
REGIME_SOFT (esclude solo regime opposto)
══════════════════════════════════════════════════════════════════════════════

  FULL OOS: n=116  wr=35.3%  ret=+1.3%  mdd=-14.9%
    Exit: TP=23  SL=63  time=30
    MC i.i.d.  : pp=0.514  pr=0.000
    MC block   : pp=0.526  pr=0.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020       8     +4.2%  50.0%
      2021      13     -4.9%  38.5%
      2022      20     -4.5%  25.0%
      2023      22     +2.5%  40.9%
      2024      20     +2.1%  30.0%
      2025      26     -3.1%  30.8%
      2026       7     +5.0%  57.1%

  HOLDOUT GENUINO 2025-2026: n=33  wr=36.4%  ret=+1.9%  mdd=-8.5%
    MC i.i.d.  : pp=0.566  pr=0.000
    MC block   : pp=0.598  pr=0.000

  Slippage-stress (sopra la frizione base 0.14% round-trip già inclusa):
    Extra slip       Scope       n      Ret%      WR    MC pp    MC pr
          0bps    full-OOS     116     +1.3%  35.3%   0.514   0.000
          0bps     holdout      33     +1.9%  36.4%   0.566   0.000
          2bps    full-OOS     116     -3.1%  35.3%   0.418   0.000
          2bps     holdout      33     +0.3%  36.4%   0.505   0.000
          5bps    full-OOS     116     -9.7%  35.3%   0.277   0.000
          5bps     holdout      33     -2.0%  36.4%   0.401   0.000
         10bps    full-OOS     116    -20.7%  35.3%   0.105   0.006
         10bps     holdout      33     -6.0%  36.4%   0.252   0.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: il regime filter strutturale PEGGIORA il risultato — ipotesi respinta

```
Variante            n(OOS)   Ret OOS   n(hold)   Ret hold   MC pp OOS   MC pp hold
NO_FILTER             144     +4.4%       41       +4.3%      0.575       0.642
REGIME_SOFT            116     +1.3%       33       +1.9%      0.514       0.566
REGIME_MATCH            75     -5.3%       25       -7.2%      0.344       0.161
```

**L'ipotesi era: il filtro di trend-context economico (separazione
close/VWAP a 5h) genera falsi positivi in mercati choppy, e una conferma
HMM indipendente li filtrerebbe.** I dati la respingono in modo netto e
monotono: più stringente è il vincolo di regime (nessuno → esclude solo
l'opposto → richiede match esatto), più il campione si riduce (144→116→75)
E più il risultato peggiora (+4.4%→+1.3%→-5.3% OOS; +4.3%→+1.9%→-7.2%
holdout). Se il filtro stesse davvero rimuovendo falsi positivi, ci
aspetteremmo il pattern opposto (campione più piccolo ma qualità più
alta). Invece rimuove trade a caso rispetto alla loro profittabilità, o
peggio, rimuove sistematicamente trade BUONI insieme a quelli cattivi.

**Perché**: il trend-context bias (5 barre, ~5h) e lo stato HMM (fittato
su return log e volatilità a finestra 24h) operano su ORIZZONTI TEMPORALI
diversi. Un vero setup di continuazione può verificarsi su un pullback di
poche ore anche dentro un regime HMM "sideways" più ampio (es. un
mini-trend all'interno di un range settimanale) — il gate HMM esatto
scarta esattamente questi casi legittimi insieme al rumore, senza
guadagno netto di qualità.

**Effetto collaterale onesto da segnalare**: anche il baseline NO_FILTER
qui (walk-forward OOS-only, frizione Bybit rigorosa 0.14% round-trip) è
più DEBOLE del report originale full-history (`vp_vwap_confluence.md`,
RR=3.0: full +12.8%/DSR=0.884, holdout +6.2%/DSR=0.498): qui MC
p_profit scende a 0.575/0.642 (comunque >0.5, MC p_ruin sempre 0.000, ma
il margine è più sottile). Due fattori concorrono, non scomponibili con
questo solo run: (1) la frizione più severa richiesta esplicitamente in
questo report (0.14% RT — taker+slippage su entrambi i lati — contro il
~0.11% RT taker-only del report base), (2) il pool WFO-OOS-only è per
costruzione più piccolo e diverso dal campione full-history (esclude i
primi 6 mesi di ogni finestra IS, usati solo per fittare l'HMM, non per
generare trade).

**Conclusione della iterazione strutturale**: il filtro di regime HMM,
in ENTRAMBE le forme testate (match esatto e versione permissiva), non
migliora la strategia — la peggiora, in modo monotono con la sua
severità. La baseline NO_FILTER (già validata nel report originale)
resta la configurazione migliore trovata finora. Non proseguo con
ulteriori varianti del filtro HMM (rischierebbe di diventare tuning di
parametri sotto mentite spoglie, in contrasto con la direttiva
anti-overfitting) — la prossima direzione strutturale sensata sarebbe un
filtro di natura diversa (es. volume/liquidità, non regime di
prezzo/volatilità) oppure accettare che il trend-context a 5h è già,
di per sé, il filtro di regime più efficace per questo setup.
