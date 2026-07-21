# BTCUSDT Perpetual — Fade ICT Standalone Strategy (Candidata a Paper Trading)

**Simbolo:** BTCUSDT Perpetual (Bybit)
**Timeframe base:** 15M (segnale/esecuzione) + 4H (contesto/zone)
**Periodo di validazione (backtest):** Gennaio 2020 – Giugno 2026
**Stato:** Validata su RR fisso (DSR=1.000 full-sample e holdout) + walk-forward stop/target dinamico + Monte Carlo + holdout genuino 2025-2026.
**⚠️ Non ancora validata su book/tick reali — vedi "Avvertenze critiche" prima di allocare capitale reale.**
**Prossimo passo raccomandato:** Paper trading live per misurare lo slippage reale sui trigger di ingresso.

---

## 0. Perché questa strategia

Nella stessa sessione di ricerca sono state testate e **scartate** numerose varianti SMC/ICT: Asia Range Sweep + MSS (v1-v4), NY-ORB + Volume Profile absorption, Order-flow "large order" absorption, VWAP Mean-Reversion (5 varianti), VWAP trend-following (replica paper accademico), VWAP Bounce (2 varianti), un filtro di regime HMM su VP+VWAP Confluence, una strategia MTF FVG (4H→15M) letta da un post social-media, e uno studio predittivo puro sui 7 oggetti ICT/SMC individuali (nessuno con potere predittivo economicamente sfruttabile in isolamento).

**Questa è la prima strategia della sessione a superare DSR=1.000 sia sul full-sample sia sull'holdout genuino 2025-2026**, ed è l'unica con rendimento positivo in ogni singolo anno del campione (2020-2026). Nasce da un'osservazione empirica (non dalla narrativa ICT presa alla lettera): i trigger tecnici SMC che si formano SENZA una zona di domanda/offerta di timeframe superiore a supporto tendono a **fallire** nella loro direzione implicita in modo statisticamente forte — fadarli (prendere il lato opposto) produce l'edge validato in questo documento.

---

## 1. Architettura del Sistema

```
[Dati]        4H OHLCV (contesto/zone) + 15M OHLCV (segnale/esecuzione), storico continuo
                 │
[Livello 1]   Rilevamento zone Demand/Supply 4H (Order Block + FVG), tracciate come
              "attive" da conferma a invalidazione
                 │
[Livello 2]   Rilevamento di 6 tipi di trigger tecnici sul 15M (Sweep, FVG, Order Block,
              Breaker Block, Inversion FVG, Power of 3/AMD)
                 │
[Filtro]      STANDALONE = il trigger si forma SENZA una zona 4H attiva della stessa
              direzione a supporto
                 │
[Segnale]     FADE — si entra nella direzione OPPOSTA a quella implicita dal trigger
                 │
[Trade]       Entry causale all'apertura della barra 15M successiva alla conferma,
              stop strutturale (minimo/massimo locale), target = RR × rischio
                 │
[Sizing]      Rischio fisso in $ (1% capitale), cap di leva 10×
```

**Perché funziona (interpretazione più probabile, non certezza)**: una zona 4H attiva esiste perché il timeframe superiore ha avuto un impulso recente in quella direzione. Un trigger 15M "standalone" (senza quella conferma) tende a comparire in un contesto dove il timeframe superiore NON supporta la direzione implicita del trigger — fadarlo equivale in parte a scommettere contro un segnale privo di conferma strutturale, un meccanismo di allineamento di trend più che una vera dinamica "smart money" nel senso letterale della narrativa ICT.

---

## 2. Dati richiesti

| Timeframe | Uso | Storico minimo |
|-----------|-----|-----------------|
| 4H | Rilevamento zone Demand/Supply (Order Block + FVG) | 60+ barre 4H (~10 giorni) prima di poter tracciare una zona attiva; per calibrare ATR e operare in produzione, consigliati 6+ mesi di storico continuo |
| 15M | Rilevamento dei 6 trigger, entry/exit, ATR | 20+ giorni di storico continuo prima di iniziare (per popolare pivot, ATR_14, finestre di rilevamento) |

Nessun dato di order-flow, tick, o order-book è richiesto — tutta la logica opera su OHLC standard. Fonte usata in validazione: Binance Vision klines (perpetual futures reali, non spot). In produzione, sostituire con il feed OHLCV live di Bybit per BTCUSDT Perpetual (stesso simbolo tradato).

---

## 3. Costruzione degli oggetti (algoritmi esatti, causali)

Tutte le definizioni seguenti sono **causali per costruzione**: ogni oggetto è utilizzabile solo a partire dalla barra in cui la sua condizione è osservabile con dati già chiusi, mai prima.

### 3.1 Swing High / Swing Low (fractal a 3 candele)

Su barre HI/LO, un punto `i` è:
- **Swing High** se `HI[i]` è il massimo tra `HI[i-1], HI[i], HI[i+1]`.
- **Swing Low** se `LO[i]` è il minimo tra `LO[i-1], LO[i], LO[i+1]`.

Confermato/utilizzabile solo alla barra `i+1` (serve la barra successiva per sapere che è un estremo locale). Base per Liquidity Sweep e per lo stop strutturale.

### 3.2 Liquidity Sweep (BSL/SSL grab)

Per ogni swing high/low CONFERMATO, si tiene traccia del suo prezzo. Un **sweep ribassista** avviene alla barra `i` se `HI[i]` supera l'ultimo swing high confermato, MA entro le successive 3 barre il prezzo **richiude sotto** quel livello (rigetto, non breakout). Direzione implicita del trigger: **ribassista** (contrarian al lato spazzato). Speculare per il sweep rialzista (sotto uno swing low, poi richiusura sopra).

### 3.3 Fair Value Gap — FVG (BISI rialzista / SIBI ribassista)

Gap a 3 candele: **FVG rialzista** se `HI[i-1] < LO[i+1]` (gap tra il massimo della 1ª candela e il minimo della 3ª). **FVG ribassista** se `LO[i-1] > HI[i+1]`. Filtro dimensione minima: `(top - bottom) >= 0.05 × ATR_14` alla barra `i`. Zona di prezzo: `[bottom, top]`. Confermato alla barra `i+1` (chiusura della 3ª candela).

### 3.4 Order Block

Cerca la candela **down-close** (`close < open`) la cui chiusura è il minimo locale tra le ultime 3 candele down-close consecutive. È un **Order Block rialzista** validato alla prima barra successiva la cui **chiusura supera l'OPEN** di quella candela d'origine (non il massimo — l'open). Zona di prezzo: `[min(open,close), max(open,close)]` della candela d'origine. Speculare per l'Order Block ribassista (candela up-close con chiusura massima locale, validato da una chiusura sotto il suo open).

### 3.5 Breaker Block

Un Order Block la cui zona viene **invalidata** (chiusura oltre l'estremo opposto della zona, es. per un OB rialzista: chiusura sotto il minimo della zona) e che successivamente viene **ri-testata dall'altro lato e rigettata** (il prezzo rientra nella zona ma richiude di nuovo dalla parte dell'invalidazione) diventa un Breaker Block, con direzione **opposta** all'Order Block originale (OB rialzista fallito → breaker ribassista/resistenza).

### 3.6 Inversion FVG (IFVG)

Una FVG la cui zona viene **rotta per intero** (chiusura oltre il bordo LONTANO, non solo un tocco/mitigazione parziale) diventa una Inversion FVG con direzione **opposta** all'originale: FVG ribassista rotta sopra (chiusura > top) → IFVG rialzista; FVG rialzista rotta sotto (chiusura < bottom) → IFVG ribassista.

### 3.7 Power of 3 / AMD (Accumulation-Manipulation-Distribution)

1. **Accumulation**: un range di 8 barre consecutive con `(max_HI - min_LO) <= 1.5 × ATR_14` (range compresso).
2. **Manipulation**: entro le 10 barre successive, il prezzo rompe (wick) sopra il massimo o sotto il minimo del range di accumulazione.
3. **Distribution**: entro 3 barre dalla rottura, il prezzo richiude di nuovo dentro il range — il trigger è confermato a quella barra, direzione **opposta** al lato rotto (rottura sopra → ribassista; rottura sotto → rialzista).

### 3.8 Zone Demand/Supply 4H (contesto HTF)

Applicando i rilevatori **Order Block** e **FVG** (sezioni 3.3-3.4) alle barre **4H** invece che 15M, si ottiene una lista di zone con:
- `direzione` (demand=rialzista, supply=ribassista)
- `inizio attività` = barra 4H di conferma
- `fine attività` = prima barra 4H la cui chiusura invalida la zona (o, se mai invalidata, un tetto di 60 barre 4H ≈ 10 giorni)

La finestra di attività viene poi mappata sulle barre 15M corrispondenti (allineamento per timestamp) per il controllo di confluenza.

---

## 4. Condizioni del segnale

Un trade FADE viene generato quando, su una barra 15M `i`:

1. Uno qualunque dei 6 trigger (Sezioni 3.2-3.7) è confermato alla barra `i`, con direzione implicita `dir` (+1 rialzista, -1 ribassista).
2. **STANDALONE**: NESSUNA zona 4H (Sezione 3.8) della stessa direzione `dir` è attiva sulla barra `i` (usando la mappatura 4H→15M).
3. Se più trigger si formano sulla stessa barra o in barre molto vicine, si prende solo il **primo** cronologicamente (gating sequenziale: nessun nuovo trade finché il precedente non è chiuso — niente posizioni sovrapposte).

Se le condizioni 1-2 sono soddisfatte: **direzione del trade = -dir** (opposta al trigger).

---

## 5. Entry, Stop, Target (esecuzione)

- **Entry**: causale, all'**apertura della barra 15M successiva** (`i+1`) alla conferma del trigger — mai sulla barra `i` stessa (che è già "in corso" quando il trigger si conferma sulla sua chiusura).
- **Stop (strutturale)**: 
  - Long (fade di un trigger ribassista): `stop = min(LO[i-SWING_LOOKBACK : i+1]) - 0.1 × ATR_14[i]`
  - Short (fade di un trigger rialzista): `stop = max(HI[i-SWING_LOOKBACK : i+1]) + 0.1 × ATR_14[i]`
  - `SWING_LOOKBACK = 15` barre (~3,75 ore) — valore modale scelto dal walk-forward causale su 35 finestre (Sezione 8); il report base (senza WFO) usava 10 barre con risultati comparabili.
- **Target**: `entry ± RR × |entry - stop|`, con **RR = 3.0** — confermato sia dal grid+DSR sull'intera storia (`ict_fade_standalone.md`) sia come uno dei due valori più selezionati dal walk-forward causale (RR=2.5/3.0 sostanzialmente equivalenti in frequenza, Sezione 8).
- **Uscita a tempo**: se né stop né target vengono toccati entro **96 barre 15M (24 ore)**, chiusura forzata al prezzo di chiusura della barra 96.
- **Priorità in caso di ambiguità intrabar**: se sia stop sia target sono teoricamente raggiungibili nella stessa barra, si assume lo STOP come toccato per primo (convenzione conservativa, coerente con tutta la pipeline di validazione di questa sessione).

---

## 6. Risk management e position sizing

```python
RISK_PCT = 0.01          # 1% del capitale iniziale, rischiato per trade
MAX_LEV  = 10.0           # leva massima consentita

risk_per_trade = INIT_CAP * RISK_PCT
stop_distance  = abs(entry_price - stop_price)
units          = min(risk_per_trade / stop_distance, MAX_LEV * INIT_CAP / entry_price)
```

**Nota sul sizing usato in validazione**: `INIT_CAP` è il capitale INIZIALE fisso (non ricalcolato dinamicamente sul capitale corrente — sizing non-compounding), la stessa convenzione usata in tutta questa sessione di validazione. In produzione, un trader può scegliere di ricalcolare `INIT_CAP` periodicamente (es. mensilmente) sul capitale corrente per un effetto di compounding controllato — non testato in questo backtest, da validare separatamente se adottato.

**Un solo trade alla volta** (nessuna posizione sovrapposta): il prossimo segnale viene ignorato finché la posizione corrente non è chiusa (stop, target, o timeout 24h).

---

## 7. Fee e frizioni (assunzioni di validazione)

| Voce | Valore | Note |
|------|--------|------|
| Fee taker (Bybit derivatives) | 0.055% per lato | Esecuzione a mercato assunta per tutti gli entry/exit |
| Slippage | 0.015% per lato | Stima conservativa per simulare execution lag / profondità book |
| **Frizione round-trip totale** | **0.14%** | (0.055%+0.015%)×2, applicata su ogni trade nel backtest |

**Nota**: la strategia regge fino a ~2bps di slippage AGGIUNTIVO oltre questa baseline (vedi `ict_fade_standalone.md`); oltre i ~5bps aggiuntivi, l'holdout diventa negativo. In paper trading, monitorare attentamente lo slippage reale sui trigger di ingresso (specialmente Sweep e Breaker Block, che sono per natura eventi di rottura/volatilità) — è la metrica più critica da validare prima di allocare capitale reale.

---

## 8. Ottimizzazione walk-forward stop/target

Il risultato base (Sezione 5, RR=3.0 fisso, SWING_LOOKBACK=10 fisso) è stato selezionato su un grid scan con correzione DSR sull'intera storia. Per verificarne la robustezza con una selezione **causale per finestra** (mai guardando avanti), è stato eseguito un walk-forward: 6 mesi in-sample / 2 mesi out-of-sample / step 2 mesi, con IS-scan su una griglia `SWING_LOOKBACK ∈ {5,10,15} × RR ∈ {1.5,2.0,2.5,3.0}` (12 combinazioni/finestra), selezione per Sharpe IS, applicazione SOLO su barre OOS mai viste durante quella selezione.

**Risultati**: vedi `reports/ict_fade_wfo.md` e il report HTML `reports/ict_fade_wfo_report.html` per il dettaglio completo (selezione per finestra, equity curve OOS, breakdown per anno, confronto diretto con la versione a RR fisso).

```
                             n      Ret%      WR    MC p(profit)    MC p(ruin)
WFO dinamico (causale)    1255   +426.5%   51.0%        1.000          0.000
RR=3.0 fisso (rif.)       1399   +512.7%   47.0%        1.000          0.000
```

**Il WFO conferma il risultato base**: selezionando stop/target in modo interamente causale
(mai guardando le barre OOS durante la scelta, nessuna correzione DSR necessaria per
costruzione) su 35 finestre, il risultato è dello stesso ordine di grandezza del RR
fisso — ret. inferiore (+426.5% vs +512.7%) ma win rate **più alto** (51.0% vs 47.0%,
sopra il 50%), ancora positivo in ogni singolo anno 2020-2026, MC p(profit)=1.000,
p(ruin)=0.000. Questa convergenza tra due metodologie di selezione indipendenti (grid+DSR
sull'intera storia vs IS-scan causale per finestra) è la prova di robustezza più solida
raccolta in questa sessione.

**Parametri selezionati più di frequente (raccomandazione operativa)**:
- `SWING_LOOKBACK = 15` barre (scelto in 30 delle 35 finestre) — più largo del valore
  base 10 usato nel report a RR fisso: lo stop strutturale beneficia di una finestra
  leggermente più ampia per il calcolo del minimo/massimo locale.
- `RR = 2.5` (14 finestre) o `RR = 3.0` (13 finestre) — i due valori sostanzialmente
  equivalenti in frequenza di selezione; **RR = 2.75 come compromesso**, oppure RR=3.0
  per coerenza con il report base già ampiamente documentato.

**Raccomandazione finale per il paper trading**: `SWING_LOOKBACK = 15`, `RR = 3.0`
(preferito a 2.5 per il track record più esteso già documentato in `ict_fade_standalone.md`
e per la maggiore semplicità di replica) — aggiornare Sezione 5 con questi valori prima
di iniziare il paper trading.

---

## 9. Protocollo di Paper Trading

### 9.1 Setup

1. Feed OHLCV live Bybit per BTCUSDT Perpetual, timeframe 4H e 15M, con almeno 20 giorni di storico caricato all'avvio (per popolare ATR_14, pivot, zone attive).
2. Ricalcolare **ad ogni chiusura di barra 15M**: pivot/swing, i 6 detector di trigger, lo stato delle zone 4H attive (aggiornato ad ogni chiusura di barra 4H).
3. Nessun re-training o re-fitting di modelli: tutta la logica è basata su regole fisse, deterministiche, causali — nessun parametro stimato staticamente dal passato che richieda aggiornamento periodico (a differenza di strategie ML/HMM usate altrove in questa sessione).

### 9.2 Loop di segnale live (pseudocodice)

```
ad ogni chiusura di barra 15M:
    aggiorna ATR_14, pivot, swing confermati
    aggiorna lo stato delle zone 4H attive (se è appena chiusa anche una barra 4H)
    per ciascuno dei 6 detector di trigger:
        controlla se un trigger si conferma su questa barra
        se sì:
            calcola la direzione implicita `dir`
            controlla se esiste una zona 4H attiva della direzione `dir` su questa barra
            se NESSUNA zona attiva (standalone) E nessuna posizione aperta:
                direzione_trade = -dir
                calcola stop strutturale (min/max ultime SWING_LOOKBACK barre + buffer ATR)
                calcola target = entry ± RR × rischio
                invia ordine di mercato all'apertura della PROSSIMA barra 15M
                registra: timestamp segnale, tipo di trigger, prezzo entry pianificato,
                          stop, target, direzione
                interrompi la scansione degli altri trigger per questa barra (un solo trade)
    se posizione aperta:
        controlla stop/target/timeout 24h ad ogni nuova barra 15M
```

### 9.3 Campi di log per trade (obbligatori per la validazione paper trading)

| Campo | Descrizione |
|-------|-------------|
| `signal_ts` | Timestamp di conferma del trigger |
| `trigger_type` | Uno tra: Sweep, FVG, OrderBlock, BreakerBlock, InversionFVG, PowerOf3 |
| `trigger_dir` | Direzione implicita del trigger (prima del fade) |
| `entry_ts_planned` / `entry_ts_actual` | Timestamp pianificato vs eseguito |
| `entry_price_planned` / `entry_price_actual` | Prezzo teorico (open barra) vs fill reale |
| `slippage_bps` | `(entry_actual - entry_planned)/entry_planned × 10000`, con segno |
| `stop_price`, `target_price` | Come da Sezione 5 |
| `exit_ts`, `exit_price`, `exit_reason` | stop / target / timeout24h |
| `units`, `notional`, `leverage_used` | Come da Sezione 6 |
| `fee_paid`, `net_pnl` | Costi reali vs assunti |
| `htf_zone_state` | Nessuna zona attiva della direzione opposta (per verificare che il filtro standalone stia escludendo correttamente le zone concorrenti) |

### 9.4 Cadenza di refresh e monitoraggio

- Nessun ri-addestramento: la logica è statica. Rivalidare l'intera pipeline (rieseguire il backtest completo) con cadenza **trimestrale**, per verificare che il comportamento storico non sia cambiato in modo strutturale (es. nuova regolamentazione, cambio di regime di volatilità).
- Monitorare **mensilmente**: win rate realizzato vs atteso (~44-47% a RR=3.0), slippage medio realizzato vs assunto (0.015%), frequenza di trade realizzata vs attesa (~0,59/giorno storico).

### 9.5 Criteri go/no-go per il passaggio a capitale reale

| Criterio | Soglia minima |
|----------|-----------------|
| Durata paper trading | Almeno 60-90 giorni (per accumulare ~35-53 trade, coerente con l'orizzonte 60gg testato nel Monte Carlo forward) |
| Slippage medio realizzato | ≤ 2bps aggiuntivi oltre l'assunzione di validazione (0,015%) — oltre questa soglia il margine di sicurezza si erode rapidamente (vedi Sezione 7) |
| Win rate realizzato | Non significativamente sotto il range storico (~42-50% a seconda dell'anno) |
| Nessun errore sistematico di esecuzione | Fill price coerente con l'apertura di barra pianificata, nessun trigger "fantasma" (bug di rilevamento) |
| Drawdown realizzato | Coerente con l'ordine di grandezza storico (MDD full-sample ~-6 / -16% a seconda della configurazione stop/target) |

---

## 10. Avvertenze critiche

1. **Nessuna validazione su dati tick/order-book reali** — tutta la logica opera su OHLC di barra chiusa; l'esecuzione reale (specialmente su trigger di rottura come Sweep/Breaker) potrebbe subire slippage superiore alle ipotesi di backtest durante momenti di alta volatilità.
2. **Interpretazione causale non certa**: l'ipotesi più probabile (Sezione 0) è un effetto di allineamento di trend, non una vera dinamica "smart money" — se il meccanismo reale è diverso, la robustezza futura non è garantita.
3. **Margine di sicurezza sui costi non enorme**: la strategia degrada tra 2 e 5bps di slippage aggiuntivo (vedi Sezione 7) — un peggioramento delle condizioni di mercato (spread più larghi, minore liquidità) potrebbe erodere l'edge più rapidamente di altre strategie con margini più ampi.
4. **Nessun filtro di regime esplicito**: a differenza di altre strategie testate in sessione, questa non usa un filtro HMM o di volatilità — è stato verificato che il rendimento è positivo in ogni anno del campione, ma non è stato testato un filtro di regime esplicito che potrebbe migliorare ulteriormente la robustezza (o rivelarsi superfluo, come accaduto per VP+VWAP Confluence in questa sessione).
