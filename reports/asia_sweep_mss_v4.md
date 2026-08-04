# Asia Range Sweep + MSS + FVG/OB Entry v4 (trend context + displacement MSS) — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
Asia Range Sweep + MSS + FVG/OB Entry v4 (trend context + displacement MSS) — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[EVENTS] 2373 giorni scansionati -> 286 trade candidati
  scartati: no-range=0  no-sweep=598  no-trend-context=627  no-mss(displacement)=502  no-entry(fvg/ob)=360  bad-target=0

  Holdout genuino 2025-2026: 73 trade

══════════════════════════════════════════════════════════════════════════════
RR = 1.0 : 1   (stop = target_dist / 1.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=286  wr=48.3% (BE_teorico=50.0%, p=0.7423)  ret=-28.5%  mdd=-38.7%
    Exit: TP=132  SL=139  time=15
    MC i.i.d.  : pp=0.044  pr=0.030
    MC block   : pp=0.036  pr=0.029

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      30     +3.7%  60.0%
      2021      37     -7.4%  40.5%
      2022      44     +2.2%  54.5%
      2023      45     -7.6%  46.7%
      2024      57    -10.2%  43.9%
      2025      43    -12.6%  39.5%
      2026      30     +3.4%  60.0%

  HOLDOUT 2025-2026: n=73  wr=47.9%  ret=-9.1%  mdd=-15.4%
    MC i.i.d.  : pp=0.149  pr=0.000
    MC block   : pp=0.142  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 2.0 : 1   (stop = target_dist / 2.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=286  wr=37.4% (BE_teorico=33.3%, p=0.0816)  ret=-9.2%  mdd=-33.9%
    Exit: TP=104  SL=176  time=6
    MC i.i.d.  : pp=0.357  pr=0.014
    MC block   : pp=0.329  pr=0.004

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      30     -6.4%  30.0%
      2021      37     +2.4%  37.8%
      2022      44     +6.6%  43.2%
      2023      45    -13.7%  31.1%
      2024      57     -4.5%  35.1%
      2025      43     -3.6%  37.2%
      2026      30     +9.9%  50.0%

  HOLDOUT 2025-2026: n=73  wr=42.5%  ret=+6.3%  mdd=-11.4%
    MC i.i.d.  : pp=0.675  pr=0.000
    MC block   : pp=0.707  pr=0.000

══════════════════════════════════════════════════════════════════════════════
RR = 3.0 : 1   (stop = target_dist / 3.0)
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=286  wr=29.0% (BE_teorico=25.0%, p=0.0682)  ret=-17.8%  mdd=-31.0%
    Exit: TP=81  SL=200  time=5
    MC i.i.d.  : pp=0.285  pr=0.085
    MC block   : pp=0.230  pr=0.029

  Breakdown per anno:
      Year       n      Ret%      WR
      2020      30    -11.0%  20.0%
      2021      37     +3.2%  29.7%
      2022      44     +4.8%  31.8%
      2023      45    -16.7%  24.4%
      2024      57     +3.3%  31.6%
      2025      43     -3.9%  30.2%
      2026      30     +2.6%  33.3%

  HOLDOUT 2025-2026: n=73  wr=31.5%  ret=-1.4%  mdd=-9.8%
    MC i.i.d.  : pp=0.449  pr=0.000
    MC block   : pp=0.438  pr=0.000

══════════════════════════════════════════════════════════════════════════════
DSR — famiglia N=3 (RR 1:1, 2:1, 3:1)
══════════════════════════════════════════════════════════════════════════════

  Full-sample:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0     286    -28.5%       -1.715    0.000
    2.0     286     -9.2%       -0.378    0.000
    3.0     286    -17.8%       -0.581    0.000

  Holdout 2025-2026:
      RR       n      Ret%   Sharpe_hat      DSR
    1.0      73     -9.1%       -1.077    0.000
    2.0      73     +6.3%        0.494    0.047
    3.0      73     -1.4%       -0.084    0.000

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity — RR migliore per DSR = 2.0:1
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample     286     -9.2%  37.4%   -33.9%   0.357   0.014
        0bps     holdout      73     +6.3%  42.5%   -11.4%   0.675   0.000
        2bps  full-sample     286    -30.3%  37.4%   -46.7%   0.125   0.147
        2bps     holdout      73     -0.5%  42.5%   -13.6%   0.474   0.000
        5bps  full-sample     286    -61.8%  37.1%   -70.3%   0.017   0.721
        5bps     holdout      73    -10.8%  42.5%   -19.3%   0.223   0.000
       10bps  full-sample     286   -114.5%  36.7%  -121.6%   0.000   1.000
       10bps     holdout      73    -27.9%  42.5%   -35.0%   0.022   0.014

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: MSS più rigoroso, risultato onesto (non è una cura miracolosa)

```
                       v3 (n=605)   v4 (n=286, -53% eventi)
RR=2 full ret            -15.7%       -9.2%   (migliore full-sample di tutta l'esplorazione)
RR=2 full MC p_ruin        0.125       0.014   (nettamente meno rischio di rovina)
RR=2 holdout ret          +15.1%       +6.3%   (più debole)
RR=2 holdout DSR            1.000       0.047   (crollato quasi a zero)
```

Il filtro di contesto di trend (627 sweep scartati perché il pivot rotto
non apparteneva a una struttura realmente opposta) e il filtro di
displacement (502 ulteriori scarti per rotture senza un vero movimento
impulsivo) eliminano più della metà degli eventi di v3. Il risultato è
**onesto ma non risolutivo**:

- Il full-sample migliora in modo pulito (perdita quasi dimezzata, P(ruin)
  crolla da 12.5% a 1.4%) — il filtro elimina effettivamente molto rumore,
  confermando il sospetto dell'utente che "alcuni MSS erano sbagliati".
- Il breakdown per anno è ora molto più equidistribuito (RR=2: 2020 -6.4%,
  2021 +2.4%, 2022 +6.6%, 2023 -13.7%, 2024 -4.5%, 2025 -3.6%, 2026 +9.9%)
  — non più concentrato negli ultimi 2-3 anni come in v3, un segnale
  metodologicamente più sano.
- **Ma il segnale sull'holdout, dopo la correzione DSR, si è indebolito
  drasticamente** (da 1.000 a 0.047): il risultato positivo di v3
  sull'holdout (+15.1%) era in parte gonfiato da MSS meno rigorosi che
  probabilmente inglobavano più rumore fortuito in quella finestra
  recente — con la definizione corretta l'apparente edge quasi scompare.
- Il full-sample resta comunque netto negativo per tutti e 3 gli RR, DSR
  family piena a 0.000.

**Conclusione**: la correzione della definizione di MSS era necessaria e
ben fondata (ricerca ICT/SMC: contesto di trend + displacement), ha
chiaramente ripulito la strategia da falsi segnali strutturali, ma non ha
rivelato un edge nascosto — anzi, il segnale più promettente di v3 si è
ridimensionato sotto un esame più rigoroso. Questo è tipicamente un buon
segno per la QUALITÀ della metodologia (meno probabile che fosse
overfitting) mentre resta un cattivo segno per la strategia in sé.
