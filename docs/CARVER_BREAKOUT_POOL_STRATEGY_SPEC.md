# BTCUSDT Perpetual — Carver Breakout Pool Strategy (ARCHIVIATA — non idonea al live)

> **⛔ ESCLUSA dal set di strategie live (2026-08-04).** Nuovo requisito di sessione:
> le strategie candidate al live devono operare su timeframe **intraday**. Questa
> strategia usa timeframe **1D**, quindi non è idonea a prescindere dalla qualità
> statistica del risultato (che resta valida e documentata qui sotto come
> **archivio storico di ricerca**). Non validarla per il live. Vedi
> `docs/CARVER_BREAKOUT_INTRADAY_4H_STRATEGY_SPEC.md` (o il report più recente
> `reports/carver_intraday_4h.md`) per il tentativo di porting a timeframe
> intraday e il relativo esito.

**Simbolo:** BTCUSDT Perpetual (Bybit)
**Timeframe base:** 1D (segnale, sizing ed esecuzione — nessun altro timeframe richiesto)
**Periodo di validazione (backtest):** Gennaio 2020 – Luglio 2026
**Stato:** Validata su 1D — **unica strategia, tra le ~20 testate in questa sessione, con DSR=1.000 sia full-sample sia holdout genuino 2025-2026**, e l'unica ad aver dimostrato profitto durante un vero bear market (-38.0% buy&hold) nel proprio holdout. **Esclusa dal live per vincolo di timeframe (Sezione di apertura).**
**⚠️ Non ancora validata su book/tick reali — vedi "Avvertenze critiche" prima di allocare capitale reale.**
**Prossimo passo raccomandato:** Nessuno — strategia archiviata. Vedi il porting intraday.

---

## 0. Perché questa strategia

A differenza di tutte le altre strategie testate in sessione (quasi tutte intraday, trade-based, entry/exit discreti su timeframe 15M-4H), questa nasce dal framework sistematico pubblicato di Rob Carver (qoppac.blogspot.com, "Systematic Trading"/"Leveraged Trading"): **non un'entry singola con stop/target, ma un forecast continuo che ridimensiona la posizione ogni giorno** (vol-targeting), esattamente come nel framework originale — mai testato prima su crypto in questa sessione con l'infrastruttura corretta di calibrazione causale del forecast.

Sono stati confrontati 4 candidati con lo stesso framework (EWMAC/trend, Breakout/canale Donchian, Carry/funding rate, Grand pool = media dei tre): **solo Breakout pool e Grand pool superano DSR=1.000 su entrambi gli ambiti**, ma Breakout pool è il più puro (Grand pool ne eredita la robustezza diluendola con componenti più deboli — vedi `reports/carver_rules.md`). L'holdout 2025-2026 è stato un **vero bear market per BTC** (buy&hold -38.0%): che Breakout pool resti profittevole (+30.8%) in quel periodo è la prova più forte raccolta in sessione che il sistema fa **timing reale** (va short/riduce esposizione), non semplicemente cavalca il trend strutturale rialzista 2020-2026.

---

## 1. Architettura del Sistema

```
[Dati]        1D OHLCV close, storico continuo
                 │
[Livello 1]   6 canali Donchian paralleli (N = 10, 20, 40, 80, 160, 320 giorni),
              ciascuno smussato con EWMA(span=N/4) → forecast continuo per velocità
                 │
[Livello 2]   POOL = media delle 6 velocità (mai una singola velocità tradata da sola)
                 │
[Sizing]      Vol-targeting: unità = (forecast/10) × capitale × target_vol_annuo(20%)
              / volatilità annualizzata del prezzo, clip a ±10× leva
                 │
[Esecuzione]  Ribilanciamento a OGNI chiusura di barra 1D — nessun concetto di
              "trade" discreto: la size cambia quasi ogni giorno
```

**Perché funziona (interpretazione)**: un canale Donchian misura quanto il prezzo corrente è vicino agli estremi recenti — pooling di 6 velocità (da 10 a 320 giorni) evita di legare l'edge a un singolo orizzonte temporale, filosofia esplicita del framework Carver ("mai tradare una sola velocità"). Il vol-targeting riduce automaticamente l'esposizione nei periodi di alta volatilità (spesso correlati a inversioni improvvise), un meccanismo di risk management incorporato nel sizing stesso, non aggiunto sopra come filtro separato.

---

## 2. Dati richiesti

| Timeframe | Uso | Storico minimo |
|-----------|-----|-----------------|
| 1D | Canale Donchian (max/min rolling), EWMA di volatilità (span 25), forecast scalar (expanding mean) | Almeno 320 giorni continui prima di poter calcolare il canale più lento (N=320); per il forecast scalar e la vol EWMA, minimo 60 giorni prima che il sistema inizi a generare size non-nulla |

Nessun dato di funding rate richiesto per il solo Breakout pool (necessario solo se si volesse aggiungere il componente Carry, escluso da questa spec perché fallisce la validazione full-sample — vedi Sezione 8). Fonte usata in validazione: Binance Vision klines giornaliere (perpetual futures). In produzione, sostituire con il feed OHLCV 1D live di Bybit per BTCUSDT Perpetual.

---

## 3. Costruzione del forecast (algoritmo esatto, causale)

Per ciascuna delle 6 finestre `N ∈ {10, 20, 40, 80, 160, 320}` giorni:

```python
roll_max = price.rolling(N, min_periods=N).max()      # SOLO barre passate incluse t
roll_min = price.rolling(N, min_periods=N).min()
mid      = (roll_max + roll_min) / 2
half_rng = (roll_max - roll_min) / 2

raw[t]   = 40 * (price[t] - mid[t]) / half_rng[t]      # in [-40, +40] circa
forecast_N[t] = EWMA(raw, span=max(N/4, 2))[t]         # smussamento causale
forecast_N[t] = clip(forecast_N[t], -20, +20)
```

`forecast_N = +20` (o vicino) significa "prezzo vicino al massimo del canale N-giorni" (bias rialzista massimo per quella velocità); `-20` il simmetrico ribassista. La costante `40` è la scala standard del framework Carver per il breakout rule (calibrata affinché il forecast medio assoluto atteso sia ~10, lo stesso target usato per EWMAC/Carry — qui non serve ulteriore calibrazione a posteriori perché la formula è già scale-corretta per costruzione).

```python
BREAKOUT_POOL[t] = mean(forecast_10[t], forecast_20[t], forecast_40[t],
                         forecast_80[t], forecast_160[t], forecast_320[t])   # ignora i NaN nel warmup
```

**Nessun parametro è ottimizzato sui dati** — le 6 velocità e i cap sono le costanti standard pubblicate dal framework, non frutto di grid-search su BTCUSDT (a differenza di quasi ogni altra strategia testata in sessione).

---

## 4. Position sizing (vol-targeting)

```python
price_vol[t]            = EWMA_stdev(daily_return, span=25)[t]     # solo barre passate
annualized_price_vol[t] = price_vol[t] * sqrt(365)                  # $/unità/anno

TARGET_ANNUAL_VOL = 0.20      # 20% annuo, capitale non a leva
INIT_CAP          = capitale corrente (non-compounding in validazione, vedi nota)
MAX_LEV           = 10.0

raw_units[t] = (BREAKOUT_POOL[t] / 10.0) * INIT_CAP * TARGET_ANNUAL_VOL / annualized_price_vol[t]
units[t]     = clip(raw_units[t], -MAX_LEV * INIT_CAP / price[t], +MAX_LEV * INIT_CAP / price[t])
```

**Nota sul sizing usato in validazione**: `INIT_CAP` è il capitale iniziale fisso (non ricalcolato dinamicamente — stessa convenzione non-compounding di tutta la sessione). In produzione si può ricalcolare periodicamente sul capitale corrente per compounding controllato — non testato qui.

A differenza delle strategie trade-based (es. ICT Fade Standalone), **non esiste un concetto di "rischio per trade" o stop-loss discreto**: il controllo del rischio è interamente nel vol-targeting (size più piccola quando la volatilità realizzata sale) e nel cap di leva.

---

## 5. Esecuzione — ribilanciamento giornaliero

- **Cadenza**: ricalcolo del forecast e della size target a **ogni chiusura di barra 1D** (convenzionalmente 00:00 UTC).
- **Ordine**: la differenza `Δunits = units[t] - units[t-1]` viene eseguita all'apertura della barra successiva (coerente con l'assunzione di backtest: nessun lookahead, esecuzione al fixing di chiusura/apertura giornaliera).
- **Non è un sistema "entry/exit"**: non esistono stop o target — l'esposizione si adatta continuamente. La posizione può restare dello stesso segno per mesi (trend lungo) o flippare rapidamente in mercati whipsaw.

---

## 6. Fee e frizioni (assunzioni di validazione)

| Voce | Valore | Note |
|------|--------|------|
| Fee taker (Bybit derivatives) | 0.055% per lato | |
| Slippage | 0.015% per lato | |
| **Frizione totale** | **0.07%/lato** | Applicata sul **turnover** (`|Δunits[t]| × price[t]`), non per-trade — coerente con un sistema a posizione continua |

**Robustezza allo slippage — la più alta delle 3 strategie**: a +10bps di slippage extra oltre la baseline, l'holdout resta positivo (+25.9% vs +30.8% baseline, DSR invariato) — degrado minimo (vedi Sezione 8).

---

## 7. Frequenza di trading attesa

**Non è una strategia a trade discreti** — è più corretto parlare di *cadenza di ribilanciamento* e *turnover*:

| Metrica (calcolata su 2.404 giorni, 2020-2026) | Valore |
|---|---|
| Giorni con variazione di size (ribilanciamento) | 98.8% dei giorni → **~6.9 ribilanciamenti/settimana** (praticamente ogni giorno) |
| Cambi di segno posizione (long↔short, flip di direzione) | 126 totali → **~19/anno, ~1 ogni 19 giorni (~0.37/settimana)** |
| Turnover medio giornaliero | **~7.4% del capitale** (base del calcolo delle fee) |

**Implicazione operativa**: un ribilanciamento giornaliero letterale genera un ordine quasi ogni giorno (size piccola, spesso incrementale). Se l'infrastruttura di esecuzione reale ha costi fissi per ordine (oltre al puro spread/fee %), va valutato un **buffer "no-trade"** (tecnica standard nel framework Carver: non ribilanciare se `|Δunits|` è sotto una soglia, es. 10% della size corrente) per ridurre la frequenza di esecuzione senza alterare sostanzialmente l'esposizione — **non testato in questo backtest**, da validare separatamente se adottato in produzione.

---

## 8. Risultati di validazione (sintesi da `reports/carver_rules.md`)

```
                        Ret full   DSR full   Ret hold   DSR hold   Sharpe hold
Buy & Hold (benchmark)   +714.2%       -         -38.0%       -           -
Breakout pool            +217.9%     1.000       +30.8%     1.000       0.951
```

Breakdown per anno (full-sample, n=2.372 giorni):

| Anno | Ret% | WR |
|---|---|---|
| 2020 | +112.4% | 50.1% |
| 2021 | +20.7% | 48.2% |
| 2022 | +7.3% | 51.5% |
| 2023 | +6.9% | 44.1% |
| 2024 | +39.9% | 50.8% |
| 2025 | +5.0% | 49.6% |
| 2026 (parziale) | +25.7% | 51.9% |

**Holdout genuino 2025-2026** (n=546 giorni, durante un bear market -38.0% buy&hold): ret=+30.8%, wr=50.4%, MC p(profit)=0.816, p(ruin)=0.001, Sharpe_hat=0.951, **DSR=1.000**.

Slippage-stress (holdout): 0bps→+30.8%, 2bps→+29.8%, 5bps→+28.3%, **10bps→+25.9%** — degrado minimo, il più robusto delle 3 strategie validate.

---

## 9. Protocollo di Paper Trading

### 9.1 Setup

1. Feed OHLCV live Bybit per BTCUSDT Perpetual, timeframe 1D, con almeno 320 giorni di storico caricato all'avvio (per popolare il canale Donchian a 320 giorni).
2. Ricalcolare **ad ogni chiusura di barra 1D**: 6 canali Donchian + EWMA smoothing, forecast pool, volatilità annualizzata (EWMA span 25), size target.
3. Nessun re-training/re-fitting: logica interamente deterministica su regole fisse — nessuna calibrazione statica da aggiornare (a differenza di strategie ML/HMM).

### 9.2 Loop di segnale live (pseudocodice)

```
ad ogni chiusura di barra 1D:
    aggiorna price_vol (EWMA span 25) e annualized_price_vol
    per ciascuna delle 6 finestre N:
        aggiorna roll_max, roll_min, mid, half_range
        calcola raw_N, EWMA-smussa, clip ±20 → forecast_N
    BREAKOUT_POOL = media dei 6 forecast_N (ignora NaN warmup)
    raw_units = (BREAKOUT_POOL/10) * capitale_corrente * 0.20 / annualized_price_vol
    target_units = clip(raw_units, -10*capitale/price, +10*capitale/price)
    delta = target_units - units_correnti
    se |delta| > soglia_no_trade (opzionale, da calibrare in paper trading):
        invia ordine per portare la posizione a target_units, all'apertura
        della prossima barra 1D
    registra: timestamp, forecast_pool, target_units, delta, prezzo fill, fee pagata
```

### 9.3 Campi di log per il ribilanciamento (obbligatori per la validazione paper trading)

| Campo | Descrizione |
|-------|-------------|
| `rebalance_ts` | Timestamp di chiusura barra 1D |
| `forecast_pool` | Valore del pool Breakout (-20..+20) |
| `annualized_price_vol` | Volatilità annualizzata usata per il sizing |
| `target_units` / `prev_units` | Size target vs size precedente |
| `delta_units`, `delta_pct_cap` | Variazione assoluta e come % del capitale (turnover) |
| `fill_price_planned` / `fill_price_actual` | Prezzo teorico (fixing di chiusura) vs fill reale |
| `slippage_bps` | Scostamento fill reale vs teorico |
| `fee_paid`, `net_pnl_daily` | Costi reali vs assunti |
| `leverage_used` | `|target_units| × price / capitale` |

### 9.4 Cadenza di refresh e monitoraggio

- Nessun ri-addestramento: rivalidare l'intera pipeline **trimestralmente** per verificare che il comportamento storico non sia cambiato strutturalmente.
- Monitorare **settimanalmente**: turnover realizzato vs atteso (~7.4%/giorno), slippage medio realizzato vs assunto (0.015%/lato), frequenza di flip di direzione vs attesa (~19/anno).

### 9.5 Criteri go/no-go per il passaggio a capitale reale

| Criterio | Soglia minima |
|----------|-----------------|
| Durata paper trading | Almeno 90 giorni (per osservare più cicli di ribilanciamento e almeno 1-2 potenziali flip di direzione) |
| Slippage/costo di esecuzione realizzato | ≤ 2bps aggiuntivi oltre l'assunzione di validazione (0,015%/lato) — la strategia degrada linearmente ma lentamente con lo slippage (vedi Sezione 8) |
| Turnover realizzato | Coerente con ~7%/giorno; se sistematicamente più alto, valutare un buffer no-trade prima di scalare capitale |
| Nessun errore sistematico di esecuzione | Fill coerente col fixing di chiusura/apertura pianificato |
| Drawdown realizzato | Coerente con l'ordine di grandezza storico (MDD full-sample ~-15.7%) |

---

## 10. Avvertenze critiche

1. **Nessun parametro ottimizzato su BTCUSDT** — punto di forza (bassissimo rischio di overfitting, costanti standard del framework Carver pubblicato) ma anche limite: non è stato verificato se una ri-calibrazione specifica per crypto migliorerebbe i risultati (deliberatamente non tentato per preservare l'assenza di data-snooping).
2. **Sizing non-compounding in validazione** — in produzione il compounding cambierà il profilo di rischio (leva effettiva cresce/scende con l'equity) rispetto al backtest; da validare separatamente se adottato.
3. **Turnover giornaliero non banale (~7.4%/giorno)** — su book reale con liquidità inferiore a quella assunta (fee 0.055%+slippage 0.015%), il costo di esecuzione reale potrebbe superare l'assunzione; è la metrica più critica da validare in paper trading.
4. **Il vol-target del 20% annuo è un'assunzione di modellazione**, non calibrata specificamente su BTCUSDT — un valore diverso cambierebbe leva effettiva e drawdown attesi, non testato in questa sessione.
5. **Il confronto con buy&hold va fatto risk-adjusted**: il ritorno grezzo full-sample (+217.9%) è inferiore al buy&hold (+714.2%) perché il vol-targeting mantiene l'esposizione strutturalmente più bassa — la metrica corretta di confronto è lo Sharpe/DSR, non il rendimento assoluto.
