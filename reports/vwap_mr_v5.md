# VWAP Mean-Reversion v5 — target allargato oltre il tocco VWAP — Validation Pipeline

```
══════════════════════════════════════════════════════════════════════════════
VWAP Mean-Reversion v5 — target allargato oltre il tocco VWAP — Validation Pipeline
══════════════════════════════════════════════════════════════════════════════

[IS-SCAN] Parametri selezionati per finestra (35 finestre valide):
  Z_ENTRY più scelto: {2.0: 28, 1.5: 5, 1.0: 2}
  Filtro più scelto  : {'SIDEWAYS_ONLY': 22, 'NO_FILTER': 13}
  Overshoot più scelto: {0.0: 18, 1.0: 15, 0.5: 2}

══════════════════════════════════════════════════════════════════════════════
RISULTATI — walk-forward OOS aggregato
══════════════════════════════════════════════════════════════════════════════

  FULL-SAMPLE: n=3833  wr=61.4%  ret=-195.0%  mdd=-197.8%
    Exit: TP=2045  SL=806  time=982
    MC i.i.d.  : pp=0.000  pr=1.000
    MC block   : pp=0.000  pr=1.000

  Breakdown per anno:
      Year       n      Ret%      WR
      2020     370    -30.1%  59.5%
      2021    1391    -56.4%  62.5%
      2022     720    -45.6%  57.9%
      2023     362    -25.5%  63.0%
      2024     497    -13.5%  64.0%
      2025     356    -22.8%  59.0%
      2026     137     -1.1%  65.0%

  HOLDOUT GENUINO 2025-2026: n=493  wr=60.6%  ret=-23.9%  mdd=-29.0%
    MC i.i.d.  : pp=0.070  pr=0.009
    MC block   : pp=0.041  pr=0.005

══════════════════════════════════════════════════════════════════════════════
Slippage sensitivity
══════════════════════════════════════════════════════════════════════════════

    Slippage       Scope       n      Ret%      WR     MDD%    MC pp    MC pr
        0bps  full-sample    3833   -195.0%  61.4%  -197.8%   0.000   1.000
        0bps     holdout     493    -23.9%  60.6%   -29.0%   0.070   0.009
        2bps  full-sample    3833   -289.7%  59.5%  -290.2%   0.000   1.000
        2bps     holdout     493    -42.3%  58.4%   -43.6%   0.004   0.260
        5bps  full-sample    3833   -431.8%  55.7%  -432.2%   0.000   1.000
        5bps     holdout     493    -70.0%  53.5%   -70.7%   0.000   0.950
       10bps  full-sample    3833   -668.5%  49.6%  -668.9%   0.000   1.000
       10bps     holdout     493   -116.1%  46.7%  -116.4%   0.000   1.000

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: nemmeno allargare il target salva la strategia

```
                    v1 (target=tocco VWAP)   v5 (target allargato, overshoot scansionato)
Win rate full-sample      62.9%                       61.4%
Ret full-sample          -182.7%                      -195.0%
Ret holdout                -27.1%                        -23.9%
MC p_profit full          0.000                         0.000
```

L'IS-scan sceglie overshoot=1.0 in 15/35 finestre (una frazione
consistente — non rumore, il target allargato ha davvero un valore su
alcune finestre) ma overshoot=0.0 (tocco esatto, comportamento di v1)
resta la scelta più comune (18/35). Il risultato aggregato è
sostanzialmente invariato rispetto a v1: leggerissimo miglioramento
sull'holdout (-23.9% vs -27.1%), leggero peggioramento sul full-sample
(-195.0% vs -182.7%). Nessun miglioramento economicamente significativo.

Spiegazione: allargare il target aumenta il guadagno potenziale per
trade vincente, ma riduce la probabilità di raggiungerlo (il prezzo deve
percorrere più strada, oltre il VWAP, il che accade meno spesso del
semplice tocco) — i due effetti si compensano quasi esattamente, senza
guadagno netto.

## Sintesi finale — 5 tentativi, 5 fallimenti coerenti

```
v1  stop largo fisso, target=tocco VWAP          -> WR reale (62.9%) ma R:R cattivo
v2  stop stretto proporzionale al target           -> R:R sistemato ma WR crolla (rumore)
v3  scan causale del moltiplicatore ATR dello stop -> conferma "largo è meglio" ma overfit
v4  uscita a tempo fisso (ricetta ML8h)            -> WR crolla a coin-flip (cambia l'edge)
v5  target allargato oltre il tocco VWAP           -> nessun guadagno netto (WR-reward si compensano)
```

Sono state tentate tutte le leve ragionevoli sulla costruzione del trade
(stop stretto/largo/scansionato, uscita a tempo, target allargato) senza
trovare una configurazione profittevole. Il pattern che emerge con
sicurezza crescente ad ogni tentativo: l'edge (win rate 61-66%, stabile e
statisticamente reale) esiste, ma la sua **dimensione economica è troppo
piccola rispetto al costo di transazione (fee 0,08% round-trip)** per
qualunque costruzione del trade provata — lo stesso tipo di limite
strutturale già trovato con il segnale su candela 1 minuto in
`candle_diagnostics.md`. Ulteriori iterazioni sulla costruzione del trade
(stop/target/uscita) hanno rendimenti marginali decrescenti già visibili
da v3 in poi; un progresso reale richiederebbe probabilmente costi di
transazione strutturalmente più bassi (maker fee, VIP tier) o un modo di
aggregare molti di questi segnali a basso edge in un portafoglio più
ampio, non un'ulteriore modifica a stop/target/uscita di un singolo trade.
