# BTCUSDT Perpetual — Time-Series Momentum (TSMOM-sign pool) Strategy (Candidata a Paper Trading)

**Simbolo:** BTCUSDT Perpetual (Bybit)
**Timeframe base:** 1D (segnale, sizing ed esecuzione — nessun altro timeframe richiesto)
**Periodo di validazione (backtest):** Gennaio 2020 – Luglio 2026
**Stato:** Validata — DSR=1.000 sia full-sample sia holdout genuino 2025-2026. **L'unica strategia della sessione positiva in OGNI singolo anno del campione 2020-2026 e la più robusta allo slippage** (holdout resta positivo fino a +10bps extra).
**⚠️ Non ancora validata su book/tick reali — vedi "Avvertenze critiche" prima di allocare capitale reale.**
**Prossimo passo raccomandato:** Paper trading live in parallelo a Carver Breakout pool (bassa correlazione attesa: costruzione basata sul segno del rendimento, non sull'ampiezza).

---

## 0. Perché questa strategia

Costruita riutilizzando **verbatim** l'infrastruttura di calibrazione forecast e vol-targeting già validata per Carver Breakout pool (stessa scalar calibration causale, stesso sizing, stesso motore di backtest a costi di turnover) — l'obiettivo era testare una terza fonte di edge **strutturalmente indipendente**: il Time-Series Momentum classico (Moskowitz/Ooi/Pedersen 2012), non un canale di prezzo (Breakout) né un incrocio di medie (EWMAC), ma il **segno del rendimento passato** su un paniere di lookback, pooled per lo stesso motivo filosofico ("mai una sola velocità").

Due costruzioni sono state confrontate: **TSMOM-sign** (segno del rendimento) e **TSMOM-magnitude** (rendimento normalizzato per volatilità). Solo **TSMOM-sign pool** supera DSR=1.000 su entrambi gli ambiti — TSMOM-magnitude fallisce l'holdout (DSR=0.000), e la combinazione dei due è debole (DSR=0.433). Il segnale binario (solo direzione, non ampiezza) si è dimostrato più robusto della versione continua.

---

## 1. Architettura del Sistema

```
[Dati]        1D OHLCV close, storico continuo
                 │
[Livello 1]   5 lookback paralleli L = 30, 60, 90, 120, 252 giorni:
              raw_sign_L[t] = sign( price[t] - price[t-L] )     (+1 / -1 / 0)
                 │
[Livello 2]   Calibrazione causale dello scalar (expanding mean di |raw|,
              target=10, cap ±20) per ciascuna velocità
                 │
[Livello 3]   POOL = media delle 5 velocità calibrate (mai una sola velocità)
                 │
[Sizing]      Identico a Breakout pool: vol-targeting 20% annuo, clip ±10× leva
                 │
[Esecuzione]  Ribilanciamento a ogni chiusura di barra 1D
```

**Perché funziona (interpretazione)**: TSMOM è l'anomalia di momentum più replicata nella letteratura accademica (azionario, futures, FX, commodities, oltre 25 anni fuori-campione nel paper originale) — qui applicata a BTCUSDT con la stessa filosofia "mai una sola velocità" del resto della sessione. Il fatto che sia positiva in **ogni** anno 2020-2026 (incluso un bear market nell'holdout) è coerente con la letteratura: il momentum cattura sia trend rialzisti sia ribassisti, per costruzione.

---

## 2. Dati richiesti

| Timeframe | Uso | Storico minimo |
|-----------|-----|-----------------|
| 1D | Rendimento su 5 lookback (fino a 252 giorni), EWMA di volatilità (span 25), forecast scalar (expanding mean, min 60 giorni) | Almeno 252 giorni continui prima di poter calcolare il lookback più lento |

Nessun altro dato richiesto. Fonte usata in validazione: Binance Vision klines giornaliere. In produzione, feed OHLCV 1D live Bybit per BTCUSDT Perpetual.

---

## 3. Costruzione del forecast (algoritmo esatto, causale)

Per ciascun lookback `L ∈ {30, 60, 90, 120, 252}` giorni:

```python
past_price   = price.shift(L)                          # prezzo di L giorni fa (già chiuso)
raw_sign_L[t] = sign( price[t] - past_price[t] )         # +1, -1, o 0

# Calibrazione causale (identica a EWMAC/Carry — vedi Sezione 3 di
# CARVER_BREAKOUT_POOL_STRATEGY_SPEC.md):
abs_expanding_mean[t] = expanding_mean( |raw_sign_L| )[0:t]     # min_periods=60
scalar[t]             = 10.0 / abs_expanding_mean[t]
forecast_L[t]          = clip( raw_sign_L[t] * scalar[t], -20, +20 )
```

Poiché `|raw_sign_L|` è sempre 0 o 1, dopo il warmup lo scalar converge a un valore prossimo a 10 (se il segnale è quasi sempre definito) — `forecast_L` oscilla quindi essenzialmente tra **+10 e -10** (raramente ai cap ±20).

```python
TSMOM_SIGN_POOL[t] = mean(forecast_30[t], forecast_60[t], forecast_90[t],
                           forecast_120[t], forecast_252[t])    # ignora NaN nel warmup
```

Essendo ciascun `forecast_L` sostanzialmente binario (±10), il pool assume valori discreti a "gradini" (es. tutte e 5 le velocità concordi → pool ≈ ±10; 4 su 5 → ±6; 3 su 5 → ±2; ecc.) — una misura diretta di **quante velocità concordano sulla direzione**.

---

## 4. Position sizing (vol-targeting) — identico a Breakout pool

```python
price_vol[t]            = EWMA_stdev(daily_return, span=25)[t]
annualized_price_vol[t] = price_vol[t] * sqrt(365)

raw_units[t] = (TSMOM_SIGN_POOL[t] / 10.0) * capitale_corrente * 0.20 / annualized_price_vol[t]
units[t]     = clip(raw_units[t], -10 * capitale/price[t], +10 * capitale/price[t])
```

Stessa convenzione non-compounding, stesso `MAX_LEV = 10.0`, stesso `TARGET_ANNUAL_VOL = 0.20` di Breakout pool. Nessun concetto di stop-loss/target discreto — il controllo del rischio è nel vol-targeting.

---

## 5. Esecuzione — ribilanciamento giornaliero

- **Cadenza**: ricalcolo a ogni chiusura di barra 1D (00:00 UTC).
- **Ordine**: `Δunits` eseguito all'apertura della barra successiva.
- A differenza di Breakout pool (che si muove con continuità mentre il prezzo attraversa il canale), il forecast TSMOM-sign cambia **solo quando uno dei 5 segni cambia** (un lookback che "scade" un rendimento da positivo a negativo, o viceversa) — variazioni più discrete, meno rumorose giorno per giorno.

---

## 6. Fee e frizioni (assunzioni di validazione)

| Voce | Valore | Note |
|------|--------|------|
| Fee taker (Bybit derivatives) | 0.055% per lato | |
| Slippage | 0.015% per lato | |
| **Frizione totale** | **0.07%/lato** | Sul turnover (`|Δunits[t]| × price[t]`) |

**La strategia più robusta allo slippage delle 3 validate**: holdout 0bps→+6.9%, 2bps→+6.3%, 5bps→+5.4%, **10bps→+4.0%** — resta positiva anche a stress massimo (unica delle 3 a farlo, DSR pieno anche a slippage aggiuntivo).

---

## 7. Frequenza di trading attesa

**Non trade discreti — ribilanciamento giornaliero continuo**, ma con turnover **più basso** di Breakout pool grazie alla natura binaria del segnale:

| Metrica (calcolata su 2.404 giorni, 2020-2026) | Valore |
|---|---|
| Giorni con variazione di size (ribilanciamento) | 95.5% dei giorni → **~6.7 ribilanciamenti/settimana** |
| Cambi di segno posizione (long↔short, flip di direzione) | 52 totali → **~7.9/anno, ~1 ogni 46 giorni (~0.15/settimana)** |
| Turnover medio giornaliero | **~4.2% del capitale** (~43% in meno di Breakout pool) |

**Implicazione operativa**: turnover più contenuto rende TSMOM-sign pool meno sensibile a un buffer no-trade rispetto a Breakout pool, ma lo stesso principio si applica — variazioni giornaliere piccole quando i lookback sono quasi tutti concordi, salti più ampi (fino a ~±4 in valore di forecast) quando un lookback cambia segno.

---

## 8. Risultati di validazione (sintesi da `reports/tsmom_strategy.md`)

Breakdown per anno (full-sample, n=2.403 giorni) — **positiva in OGNI anno**:

| Anno | Ret% | WR |
|---|---|---|
| 2020 | +58.1% | 40.5% |
| 2021 | +16.1% | 51.8% |
| 2022 | +5.3% | 54.0% |
| 2023 | +15.1% | 46.8% |
| 2024 | +24.9% | 51.4% |
| 2025 | +5.1% | 50.7% |
| 2026 (parziale) | +1.7% | 50.9% |

Full-sample: ret=+126.3%, Sharpe_hat=2.931, MC p(profit)=1.000/p(ruin)=0.000, **DSR=1.000**.
**Holdout genuino 2025-2026** (n=577 giorni): ret=+6.9%, wr=50.8%, Sharpe_hat=0.352, MC p(profit)=0.650, **DSR=1.000**.

Confronto famiglia (DSR N=3): TSMOM-sign pool DSR=1.000/1.000 (full/holdout) — TSMOM-magnitude pool DSR=1.000/**0.000** (fallisce holdout) — Combined DSR=1.000/0.433 (debole holdout). **Solo TSMOM-sign pool è validato.**

---

## 9. Protocollo di Paper Trading

### 9.1 Setup

1. Feed OHLCV live Bybit per BTCUSDT Perpetual, timeframe 1D, con almeno 252 giorni di storico caricato all'avvio (lookback più lento).
2. Ricalcolare **ad ogni chiusura di barra 1D**: 5 raw_sign_L, scalar di calibrazione (expanding mean causale), pool, volatilità annualizzata, size target.
3. Nessun re-training/re-fitting: logica deterministica su regole fisse.

### 9.2 Loop di segnale live (pseudocodice)

```
ad ogni chiusura di barra 1D:
    aggiorna price_vol (EWMA span 25) e annualized_price_vol
    per ciascun lookback L in [30, 60, 90, 120, 252]:
        raw_sign_L = sign(price[t] - price[t-L])
        aggiorna abs_expanding_mean_L (media di |raw_sign_L| fino a t)
        scalar_L = 10 / abs_expanding_mean_L
        forecast_L = clip(raw_sign_L * scalar_L, -20, +20)
    TSMOM_POOL = media dei 5 forecast_L (ignora NaN warmup)
    raw_units = (TSMOM_POOL/10) * capitale_corrente * 0.20 / annualized_price_vol
    target_units = clip(raw_units, -10*capitale/price, +10*capitale/price)
    delta = target_units - units_correnti
    se |delta| > soglia_no_trade (opzionale, da calibrare in paper trading):
        invia ordine per portare la posizione a target_units, all'apertura
        della prossima barra 1D
    registra: timestamp, 5 raw_sign_L, forecast_pool, target_units, delta, fill, fee
```

### 9.3 Campi di log per il ribilanciamento (obbligatori per la validazione paper trading)

| Campo | Descrizione |
|-------|-------------|
| `rebalance_ts` | Timestamp di chiusura barra 1D |
| `raw_sign_30/60/90/120/252` | Segno di ciascun lookback (per capire quante velocità concordano) |
| `forecast_pool` | Valore del pool TSMOM-sign |
| `target_units` / `prev_units` / `delta_units` | Come in Breakout pool |
| `fill_price_planned` / `fill_price_actual`, `slippage_bps` | Come in Breakout pool |
| `fee_paid`, `net_pnl_daily`, `leverage_used` | Come in Breakout pool |

### 9.4 Cadenza di refresh e monitoraggio

- Rivalidare l'intera pipeline **trimestralmente**.
- Monitorare **settimanalmente**: turnover realizzato vs atteso (~4.2%/giorno), numero di lookback concordi (diagnostica di quanto il segnale sia "forte" in un dato momento), frequenza di flip vs attesa (~7.9/anno).

### 9.5 Criteri go/no-go per il passaggio a capitale reale

| Criterio | Soglia minima |
|----------|-----------------|
| Durata paper trading | Almeno 90 giorni |
| Slippage/costo di esecuzione realizzato | ≤ 5bps aggiuntivi oltre l'assunzione di validazione — margine di sicurezza più ampio delle altre 2 strategie (resta DSR-valida fino a 10bps in backtest) |
| Turnover realizzato | Coerente con ~4.2%/giorno |
| Nessun errore sistematico di esecuzione | Fill coerente col fixing pianificato |
| Drawdown realizzato | Coerente con l'ordine di grandezza storico (MDD full-sample ~-7.6%, il più basso delle 3 strategie) |

---

## 10. Avvertenze critiche

1. **Nessun parametro ottimizzato su BTCUSDT** — i 5 lookback (30/60/90/120/252 giorni) sono scelte standard di letteratura (mesi/trimestri/anno), non frutto di grid-search.
2. **Holdout positivo ma con Sharpe modesto (0.352)** — più debole del full-sample (2.931), pattern comune a tutte le strategie di questa sessione (l'holdout 2025-2026 è un periodo oggettivamente più difficile). DSR=1.000 conferma che il segnale è statisticamente genuino anche a Sharpe più basso, ma il margine è più sottile che nel full-sample.
3. **TSMOM-magnitude (variante scartata) fallisce l'holdout** — un promemoria diretto che la costruzione "giusta" del segnale (sign vs magnitude) non è ovvia a priori; non assumere che varianti apparentemente simili si comportino allo stesso modo senza ri-validarle.
4. **Correlazione con Breakout pool da misurare in paper trading**: entrambe le strategie sono trend-following su 1D — pur essendo costruite in modo strutturalmente diverso (canale di prezzo vs segno del rendimento), potrebbero muoversi in modo più correlato del previsto in mercati fortemente direzionali, riducendo il beneficio di diversificazione nel portafoglio combinato.
5. **Sizing non-compounding in validazione** — stessa nota di Breakout pool, Sezione 10.
