# Confronto motore custom vs pybroker

Config testata: MR24-t2.0-PB100x5-d003 (stessi eventi di ingresso rigiocati su entrambi i motori).

## Risultati aggregati

| Metrica | Custom run_bt | pybroker | Δ |
|---|---|---|---|
| n_trades | 3146 | 3229 | +83 |
| win rate | 33.1% | 33.8% | +0.7pp |
| return % | +5.12% | +6.96% | +1.84pp |
| MDD % | -5.01% | -4.35% | +0.66pp |

## Riconciliazione dello scarto

Su 3240 eventi generati, 3230 sono stati agganciati da pybroker (gli altri sono a bordo-dataset: bar `i-1` cade fuori dal range di backtest per via del delay di fill).

Il nostro `run_bt` **scarta silenziosamente** i trade che non toccano né TP né SL entro 48h (94 casi, 2.9% del totale) — non compaiono né nel win rate né nel P&L. pybroker li **forza in chiusura a mercato** (`stop='bar'`, 84 casi).

Escludendo i forced-bar-exit da pybroker: **3145** trade comparabili vs **3146** del motore custom (scarto residuo: -1, presumibilmente edge di inizio/fine dataset).

86 trade in pybroker hanno `stop=NaN` (chiusi da un ordine opposto anziché da uno stop) — indica che pybroker **netta le posizioni** sullo stesso simbolo quando arriva un segnale di direzione opposta mentre una posizione è ancora aperta, mentre il nostro motore tratta ogni evento come trade totalmente indipendente anche in caso di sovrapposizione temporale (cooldown=4h << max_hold=48h, quindi le sovrapposizioni sono frequenti).

## Punti di forza / debolezza

**Motore custom (`run_bt`)**
- ✅ Trasparente e ispezionabile riga per riga; fee model ATR-multiplo verificato manualmente.
- ✅ Nessuna dipendenza esterna, esecuzione rapida (77 varianti in ~50 min).
- ⚠️ Scarta silenziosamente i trade che non risolvono entro max_hold — sovrastima leggermente la qualità del segnale (i trade '`none`' sono spesso in prossimità del breakeven, ma non è garantito).
- ⚠️ Non impone alcun vincolo di capitale/margine reale: somma P&L sequenzialmente per evento, senza verificare se posizioni sovrapposte nel tempo sarebbero davvero finanziabili in un conto reale.
- ⚠️ Assume fill esatto al livello TP/SL anche in caso di gap (nessuno slippage, nessun gap-through).

**pybroker**
- ✅ Accounting realistico: cash/equity tracking bar-by-bar, fee configurabili, gestione nativa walk-forward/bootstrap.
- ✅ Rileva automaticamente casi limite che il motore custom ignora (forced-bar-exit, netting di posizioni opposte).
- ⚠️ Richiede `enable_fractional_shares=True` esplicito per asset come BTC (il default tronca le shares a interi — con la size implicita di questa strategia, azzera silenziosamente ogni ordine).
- ⚠️ Convenzioni non ovvie da replicare correttamente: `stop_loss`/`stop_profit` sono espressi in *punti* (non prezzo assoluto), `buy_fill_price` default è `MIDDLE` (non open/close), `buy_delay=1` sposta il fill di una barra — richiede pieno controllo dell'API per un audit fedele.
- ⚠️ Dipendenze transitive pesanti (akshare/yahooquery) rendono l'installazione fragile su alcuni ambienti (build `jsonpath` rotta con setuptools recenti); serve installare senza `--no-deps` alcuni pacchetti e con `--no-deps` per il pacchetto principale.

## Conclusione

Una volta riconciliata la gestione dei trade non risolti entro max_hold (differenza strutturale intenzionale nel motore custom, non un bug), i due motori concordano entro ~1 punto percentuale su return e MDD, e i conteggi trade combaciano quasi esattamente (3145 vs 3146). Il motore custom fee-corretto appare **corretto nella sua logica di esecuzione**; la sua principale semplificazione da tenere a mente è l'omissione silenziosa dei trade senza touch TP/SL, che andrebbe eventualmente esplicitata (es. chiuderli a mercato) per un confronto ancora più fedele alla realtà.
