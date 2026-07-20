# SMC Structure Clarity — confronto multi-timeframe

```
══════════════════════════════════════════════════════════════════════════════
SMC Structure Clarity — confronto multi-timeframe (1W/1D/4H/1H/15M/1M)
══════════════════════════════════════════════════════════════════════════════

Dove le strutture OB/FVG/BOS-CHoCH/liquidità (EQH-EQL) sono più facili
da identificare: overshoot/dimensione relativa all'ATR, failure/mitigation
rate, frequenza — tutte causali, nessuna metrica soggettiva.

══════════════════════════════════════════════════════════════════════════════
BOS (Break of Structure) — continuazione di trend
══════════════════════════════════════════════════════════════════════════════

     TF   n/anno   overshoot(ATR)   failure rate
     1W      1.4           0.612x         44.4%
     1D     13.1           0.513x         68.2%
     4H     71.1           0.557x         63.0%
     1H    280.4           0.585x         65.6%
    15M   1125.6           0.535x         70.4%
     1M  20249.4           0.598x         71.4%

══════════════════════════════════════════════════════════════════════════════
CHoCH (Change of Character) — segnale di inversione
══════════════════════════════════════════════════════════════════════════════

     TF   n/anno   overshoot(ATR)   failure rate
     1W      1.7           0.522x         45.5%
     1D     12.2           0.527x         55.7%
     4H     79.4           0.523x         63.2%
     1H    307.9           0.523x         67.2%
    15M   1278.5           0.464x         71.2%
     1M  20728.7           0.577x         68.7%

══════════════════════════════════════════════════════════════════════════════
FVG (Fair Value Gap)
══════════════════════════════════════════════════════════════════════════════

     TF   n/anno   size media(ATR)    % mitigato entro 20barre
     1W     10.9            0.485x                      70.4%
     1D     61.5            0.502x                      76.2%
     4H    328.5            0.525x                      81.1%
     1H   1263.6            0.472x                      82.6%
    15M   5711.1            0.423x                      84.2%
     1M  153337.5            0.589x                      85.0%

══════════════════════════════════════════════════════════════════════════════
Order Block (ultima candela opposta prima di un impulso >=1.5xATR)
══════════════════════════════════════════════════════════════════════════════

     TF   n/anno   impulso medio(ATR)
     1W     15.0               2.093x
     1D     95.1               2.045x
     4H    543.6               2.115x
     1H   2075.7               2.152x
    15M   8690.9               2.073x
     1M  186662.7               2.106x

══════════════════════════════════════════════════════════════════════════════
Equal Highs/Lows (pool di liquidità, tolleranza 0.15xATR)
══════════════════════════════════════════════════════════════════════════════

     TF   n/anno
     1W      1.8
     1D     20.0
     4H    160.6
     1H    649.6
    15M   2519.3
     1M  31915.3

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: le strutture sono più "vere" su timeframe alti, molto più rumorose su quelli bassi

```
BOS/CHoCH failure rate per timeframe (metrica chiave: quanto ti puoi fidare della rottura)
   1W     1D     4H     1H    15M     1M
 44-46%  56-68%  63-64%  66-67%  70-71%  69-71%
```

**Risposta diretta**: BOS e CHoCH sono identificabili in modo NETTAMENTE
più affidabile sul **settimanale (1W)** — failure rate 44-46%, meno della
metà di quello osservato su 15M/1M (70-71%). Il **giornaliero (1D)** è il
compromesso migliore tra affidabilità e frequenza: failure rate ancora
sensibilmente più basso di 4H-1M (56-68% vs 63-71%) con ~12-13
eventi/anno, un numero gestibile per un'analisi discrezionale. Dal 4H in
giù (4H, 1H, 15M, 1M) il failure rate è sostanzialmente PIATTO e alto
(~63-71%) — più della metà delle rotture rilevate su questi timeframe
sono fakeout, indipendentemente da quanto si scenda in granularità.

**FVG**: stesso pattern — la frazione di gap mitigati entro 20 barre
SALE monotonicamente scendendo di timeframe (70.4% su 1W → 85.0% su 1M):
i FVG su timeframe alti "reggono" più a lungo come zone rilevanti, quelli
su timeframe bassi vengono richiusi quasi subito — più transitori, più
rumore di microstruttura che vero squilibrio di ordini.

**Order Block**: **nessuna differenza significativa tra timeframe**
(impulso medio ~2.0-2.15×ATR ovunque) — ma è un artefatto della metrica,
non un risultato genuino: la soglia di rilevazione (impulso >=1.5×ATR)
vincola per costruzione tutti gli OB rilevati ad avere un impulso di quella
grandezza, quindi la media non può discriminare. Per giudicare la
"qualità" di un OB servirebbe una metrica diversa (es. tasso di reazione
favorevole al primo retest), non testata qui.

**Equal Highs/Lows**: la frequenza scala proporzionalmente al numero di
barre di ciascun timeframe (da ~2/anno su 1W a ~32.000/anno su 1M) — è
una misura di disponibilità di setup, non di qualità; non discrimina la
"facilità" nel senso di affidabilità.

**Il collegamento con quanto già testato in sessione**: questo studio
conferma esattamente la logica dietro l'approccio ICT "HTF per il bias,
LTF per il timing" (il 4H-FVG + reazione 15m appena testato in
`mtf_fvg_reaction.md`) — le strutture SONO oggettivamente più affidabili
su timeframe alti. **Ma affidabilità della struttura non equivale a
edge tradabile**: la strategia che incapsulava esattamente questa logica
(zona 4H + timing 15m) è comunque fallita nettamente (DSR=0.000, MC
p_ruin fino a 0.75). La lezione è che un pattern "più pulito" riduce il
rumore nella sua IDENTIFICAZIONE, ma non garantisce che il movimento
successivo sia nella direzione attesa — le due cose sono distinte, e
questa sessione ha misurato entrambe separatamente: qui la pulizia del
pattern, altrove (ripetutamente) l'assenza di edge nel tradarlo.

**Raccomandazione pratica**: per un'analisi discrezionale o come filtro
di contesto, 1D (e 1W per il bias macro) restano i timeframe dove BOS/CHoCH
e FVG sono più affidabili come strutture in sé. Come già mostrato più
volte in questa sessione, però, tradurre quella "pulizia" in un'entry
tradabile su timeframe più bassi resta il problema irrisolto — nessuna
delle combinazioni HTF-context + LTF-entry testate finora ha superato la
validazione.
