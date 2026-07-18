# Large-Order Impact Study — potere predittivo del piazzamento di grandi ordini

```
══════════════════════════════════════════════════════════════════════════════
Large-Order Impact Study — potere predittivo del piazzamento di grandi ordini
══════════════════════════════════════════════════════════════════════════════

Analisi diagnostica (event study), non backtest: nessuna fee/sizing.
Proxy 'grande ordine' = avg_trade_size (volume/n_trades) in percentile alto
di finestra rolling 24h causale. Direzione = segno del delta (flusso
aggressivo netto). Forward return + Welch t-test per 8 orizzonti, 2 soglie.

══════════════════════════════════════════════════════════════════════════════
SOGLIA: avg_trade_size nel top 1.00% (finestra 24h)  —  n eventi totali = 37,085 (1.085% delle barre)
══════════════════════════════════════════════════════════════════════════════

  Grande BUY (delta>0)  —  n=18,142
      Hold          n    mean fwd%                  95% CI    t-stat    p-value   ctrl mean%
       1m     18,142     +0.0030%  [+0.0014%,+0.0047%]      3.55     0.0004     +0.0001%
       5m     18,142     +0.0031%  [+0.0001%,+0.0062%]      1.70     0.0894     +0.0005%
      15m     18,142     +0.0064%  [+0.0012%,+0.0115%]      1.85     0.0637     +0.0014%
      30m     18,142     +0.0075%  [+0.0004%,+0.0146%]      1.28     0.2005     +0.0029%
      60m     18,142     +0.0111%  [+0.0015%,+0.0207%]      1.08     0.2812     +0.0058%
     120m     18,142     +0.0176%  [+0.0043%,+0.0309%]      0.88     0.3766     +0.0116%
     240m     18,142     +0.0220%  [+0.0035%,+0.0404%]     -0.13     0.8979     +0.0232%
     480m     18,142     +0.0457%  [+0.0203%,+0.0711%]     -0.00     0.9992     +0.0457%

  Grande SELL (delta<0)  —  n=18,943
      Hold          n    mean fwd%                  95% CI    t-stat    p-value   ctrl mean%
       1m     18,943     +0.0006%  [-0.0016%,+0.0028%]      0.46     0.6487     +0.0001%
       5m     18,943     +0.0062%  [+0.0022%,+0.0102%]      2.81     0.0050     +0.0005%
      15m     18,943     +0.0083%  [+0.0026%,+0.0140%]      2.35     0.0188     +0.0014%
      30m     18,943     +0.0133%  [+0.0058%,+0.0209%]      2.70     0.0068     +0.0029%
      60m     18,943     +0.0147%  [+0.0048%,+0.0245%]      1.75     0.0795     +0.0058%
     120m     18,943     +0.0103%  [-0.0032%,+0.0238%]     -0.19     0.8526     +0.0116%
     240m     18,943     +0.0159%  [-0.0024%,+0.0342%]     -0.78     0.4374     +0.0232%
     480m     18,943     +0.0611%  [+0.0356%,+0.0865%]      1.18     0.2389     +0.0457%

══════════════════════════════════════════════════════════════════════════════
SOGLIA: avg_trade_size nel top 0.10% (finestra 24h)  —  n eventi totali = 5,178 (0.152% delle barre)
══════════════════════════════════════════════════════════════════════════════

  Grande BUY (delta>0)  —  n=2,566
      Hold          n    mean fwd%                  95% CI    t-stat    p-value   ctrl mean%
       1m      2,566     +0.0060%  [+0.0025%,+0.0096%]      3.29     0.0010     +0.0001%
       5m      2,566     +0.0050%  [-0.0030%,+0.0130%]      1.11     0.2677     +0.0005%
      15m      2,566     +0.0042%  [-0.0073%,+0.0157%]      0.46     0.6436     +0.0015%
      30m      2,566     -0.0018%  [-0.0187%,+0.0152%]     -0.54     0.5864     +0.0030%
      60m      2,566     -0.0032%  [-0.0269%,+0.0205%]     -0.75     0.4549     +0.0059%
     120m      2,566     +0.0168%  [-0.0164%,+0.0501%]      0.31     0.7581     +0.0116%
     240m      2,566     +0.0052%  [-0.0407%,+0.0511%]     -0.77     0.4441     +0.0232%
     480m      2,566     +0.0422%  [-0.0235%,+0.1078%]     -0.11     0.9132     +0.0458%

  Grande SELL (delta<0)  —  n=2,612
      Hold          n    mean fwd%                  95% CI    t-stat    p-value   ctrl mean%
       1m      2,612     -0.0012%  [-0.0065%,+0.0041%]     -0.50     0.6200     +0.0001%
       5m      2,612     +0.0065%  [-0.0020%,+0.0150%]      1.39     0.1652     +0.0005%
      15m      2,612     +0.0053%  [-0.0079%,+0.0185%]      0.56     0.5747     +0.0015%
      30m      2,612     +0.0142%  [-0.0029%,+0.0313%]      1.29     0.1977     +0.0030%
      60m      2,612     +0.0235%  [-0.0007%,+0.0477%]      1.43     0.1535     +0.0059%
     120m      2,612     +0.0135%  [-0.0200%,+0.0470%]      0.11     0.9134     +0.0116%
     240m      2,612     -0.0005%  [-0.0459%,+0.0450%]     -1.02     0.3087     +0.0232%
     480m      2,612     +0.0335%  [-0.0299%,+0.0969%]     -0.38     0.7041     +0.0458%

══════════════════════════════════════════════════════════════════════════════
RIEPILOGO — traiettoria del forward return per orizzonte (temporaneo vs permanente)
══════════════════════════════════════════════════════════════════════════════

  Soglia top 1.00%:
    Direzione                        1m        5m       15m       30m       60m      120m      240m      480m
    Grande BUY (delta>0)       +0.0030  +0.0031  +0.0064  +0.0075  +0.0111  +0.0176  +0.0220  +0.0457
    Grande SELL (delta<0)      +0.0006  +0.0062  +0.0083  +0.0133  +0.0147  +0.0103  +0.0159  +0.0611
    -> significativo (p<0.05) 
      Grande BUY (delta>0)      sig*     n.s.     n.s.     n.s.     n.s.     n.s.     n.s.     n.s.  
      Grande SELL (delta<0)     n.s.     sig*     sig*     sig*     n.s.     n.s.     n.s.     n.s.  

  Soglia top 0.10%:
    Direzione                        1m        5m       15m       30m       60m      120m      240m      480m
    Grande BUY (delta>0)       +0.0060  +0.0050  +0.0042  -0.0018  -0.0032  +0.0168  +0.0052  +0.0422
    Grande SELL (delta<0)      -0.0012  +0.0065  +0.0053  +0.0142  +0.0235  +0.0135  -0.0005  +0.0335
    -> significativo (p<0.05) 
      Grande BUY (delta>0)      sig*     n.s.     n.s.     n.s.     n.s.     n.s.     n.s.     n.s.  
      Grande SELL (delta<0)     n.s.     n.s.     n.s.     n.s.     n.s.     n.s.     n.s.     n.s.  

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: un effetto genuino esiste, è statisticamente reale, ma è troppo piccolo per essere tradato

```
Soglia top 1% (n~18-19k per direzione)
                       1m         5m        15m        30m
Grande BUY          +0.003%*     n.s.       n.s.       n.s.      (continuazione, decade subito)
Grande SELL           n.s.     +0.006%*   +0.008%*   +0.013%*    (rimbalzo/assorbimento, 5-30min)
```
`*` = p<0.05 vs gruppo di controllo (Welch t-test)

**Sì, c'è potere predittivo statisticamente reale — non solo rumore.**
Due effetti distinti, entrambi coerenti con la teoria classica di market
microstructure (Kyle 1985, Almgren-Chriss):

1. **Un grande BUY produce un piccolissimo impatto di CONTINUAZIONE
   immediato**: +0.0030% (0.30bps) nel minuto successivo, p=0.0004 —
   altamente significativo statisticamente. Ma decade a insignificanza
   già al 5° minuto (p=0.09) — impatto quasi interamente **temporaneo**,
   coerente con "consumo istantaneo di liquidità" (l'ordine spinge il
   prezzo, poi il libro si ricompone) più che con informazione duratura.

2. **Un grande SELL produce un piccolo effetto di ASSORBIMENTO/RIMBALZO**,
   opposto all'intuizione ingenua "grande vendita → il prezzo scende":
   nei 5-30 minuti successivi il prezzo tende a salire PIÙ della norma
   (+0.006% a 5min fino a +0.013% a 30min, tutti p<0.05) — coerente con
   l'ipotesi che una vendita aggressiva di grandi dimensioni venga spesso
   assorbita da liquidità passiva, marcando un minimo locale di breve
   termine piuttosto che avviare un trend ribassista.

**Ma la magnitudine è un ordine di grandezza sotto qualunque costo di
esecuzione reale.** L'effetto più forte misurato (SELL, 30min, +0.013%)
resta a **circa 1/10 della frizione round-trip Bybit minima (0.14% =
14bps)** stabilita in questa sessione — e questo è il numero LORDO, prima
di sottrarre il rendimento del gruppo di controllo (+0.0029%), che
ridurrebbe l'effetto NETTO attribuibile all'evento a ~0.010% (1bps),
ancora più piccolo.

**Il "growth" nei numeri grezzi a orizzonti lunghi è quasi tutto drift,
non l'ordine**: a 480 minuti, il forward return medio dopo un grande BUY
(+0.0457%) è **praticamente identico** al rendimento del gruppo di
controllo nello stesso orizzonte (+0.0457%) — la crescita "impressionante"
visibile scorrendo la tabella da 1m a 480m riflette semplicemente il
drift rialzista strutturale di BTC nel campione 2020-2026, presente
ugualmente nelle barre evento e non-evento, non un effetto informativo
del grande ordine che si accumula nel tempo. Solo il confronto esplicito
col gruppo di controllo (fatto qui via Welch t-test) separa correttamente
i due fenomeni — motivo per cui un semplice "guarda il rendimento medio
dopo l'evento" (senza controllo) sovrastimerebbe grossolanamente l'edge.

**Dose-response inconcludente**: ci si aspetterebbe che la soglia più
estrema (top 0.1%, eventi "ancora più grandi") mostri un effetto più
forte. Non è quello che si osserva — al contrario, quasi nessun orizzonte
resta significativo a quella soglia (solo BUY@1min). La spiegazione più
probabile è perdita di potenza statistica (n crolla da ~18-19k a ~2,6k
per direzione, un fattore ~7×), non un'inversione genuina dell'effetto —
ma non è possibile escludere con certezza che il proxy "avg_trade_size"
diventi più rumoroso ai percentili estremi (un singolo trade enorme in
mezzo a un minuto altrimenti tranquillo produce un avg_trade_size molto
alto ma un campione di un solo evento, comportamento diverso da un
minuto con molti trade tutti relativamente grandi).

**Risposta diretta alla domanda**: sì, l'inserimento di un grande ordine
ha un'influenza misurabile e statisticamente reale sui movimenti di
prezzo a breve termine — un buy spinge il prezzo (temporaneamente), un
sell viene tipicamente assorbito e seguito da un piccolo rimbalzo. Ma
l'effetto (0.3-1.3bps) è troppo piccolo per essere sfruttato da qualunque
strategia con costi di esecuzione reali (14bps round-trip) — coerente con
tutti i test di order-flow/tape di questa sessione (`order_flow_absorption.md`,
`candle_diagnostics.md`): il segnale grezzo esiste ed è misurabile con
rigore statistico, ma è ordini di grandezza sotto la soglia di
sfruttabilità economica su BTCUSDT con i dati OHLCV aggregati disponibili
(nessun vero tape trade-by-trade o order book L2).
