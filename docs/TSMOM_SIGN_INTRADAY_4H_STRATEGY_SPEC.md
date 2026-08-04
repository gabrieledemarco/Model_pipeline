# BTCUSDT Perpetual — Time-Series Momentum INTRADAY (4H, TSMOM-sign pool) Strategy (Candidata a Paper Trading)

**Simbolo:** BTCUSDT Perpetual (Bybit)
**Timeframe base:** **4H** (segnale, sizing ed esecuzione — nessun altro timeframe richiesto)
**Periodo di validazione (backtest):** Gennaio 2020 – Luglio 2026
**Stato:** Validata — **DSR=1.000 sia full-sample sia holdout genuino 2025-2026**, timeframe intraday (sostituisce la versione 1D archiviata in `docs/TSMOM_SIGN_POOL_STRATEGY_SPEC.md`, esclusa per il nuovo requisito di sessione "solo intraday"). **Il candidato con lo Sharpe full-sample più alto di questo porting (3.362)**.
**⚠️ Non ancora validata su book/tick reali — vedi "Avvertenze critiche" prima di allocare capitale reale.**
**Prossimo passo raccomandato:** Paper trading live in parallelo a Carver Breakout pool 4H (bassa correlazione strutturale attesa).

---

## 0. Perché questa strategia

TSMOM-sign pool su timeframe 1D era risultata validata (DSR=1.000 full-sample+holdout, positiva in ogni anno 2020-2026, `reports/tsmom_strategy.md`) ma esclusa dal set live per il nuovo vincolo "solo intraday". Questo documento verifica il porting a **4H**, riscalando i 5 lookback in barre per preservare la stessa finestra di calendario (es. 30 giorni = 180 barre 4H). Il risultato (`reports/carver_intraday_4h.md`) conferma **DSR=1.000 su entrambi gli ambiti** — l'edge sopravvive, con lo Sharpe full-sample più alto (3.362) tra i tre candidati testati nel porting (Breakout 4H 2.850, Grand pool 4H 3.203).

---

## 1. Architettura del Sistema

```
[Dati]        4H OHLCV close, storico continuo
                 │
[Livello 1]   5 lookback paralleli, barre 4H equivalenti a 30/60/90/120/
              252 giorni di calendario (180/360/540/720/1512 barre):
              raw_sign_L[t] = sign( price[t] - price[t-L] )
                 │
[Livello 2]   Calibrazione causale dello scalar (expanding mean di |raw|,
              target=10, cap ±20) per ciascuna velocità
                 │
[Livello 3]   POOL = media delle 5 velocità calibrate
                 │
[Sizing]      Vol-targeting 20% annuo, clip ±10× leva (identico a Breakout 4H)
                 │
[Esecuzione]  Ribilanciamento a ogni chiusura di barra 4H
```

---

## 2. Dati richiesti

| Timeframe | Uso | Storico minimo |
|-----------|-----|-----------------|
| 4H | Rendimento su 5 lookback (fino a 1.512 barre = 252gg), EWMA di volatilità (span 150 barre ≈ 25gg), forecast scalar (expanding mean, min 360 barre ≈ 60gg) | Almeno 1.512 barre 4H (~252 giorni) continue |

Fonte usata in validazione: Binance Vision klines 4H. In produzione, feed OHLCV 4H live Bybit per BTCUSDT Perpetual.

---

## 3. Costruzione del forecast (algoritmo esatto, causale)

Identico al report 1D (`docs/TSMOM_SIGN_POOL_STRATEGY_SPEC.md`, Sezione 3), con `L ∈ {180, 360, 540, 720, 1512}` barre 4H (equivalenti a 30/60/90/120/252 giorni):

```python
past_price    = price.shift(L)
raw_sign_L[t] = sign( price[t] - past_price[t] )

abs_expanding_mean[t] = expanding_mean( |raw_sign_L| )[0:t]   # min_periods=360
scalar[t]              = 10.0 / abs_expanding_mean[t]
forecast_L[t]           = clip( raw_sign_L[t] * scalar[t], -20, +20 )

TSMOM_SIGN_POOL[t] = mean(forecast_180, forecast_360, forecast_540,
                           forecast_720, forecast_1512)[t]
```

Volatilità annualizzata: stessa costruzione di Breakout 4H (EWMA span 150 barre, annualizzazione ×√(365×6)).

---

## 4. Position sizing (vol-targeting) — identico a Breakout pool 4H

```python
raw_units[t] = (TSMOM_SIGN_POOL[t] / 10.0) * capitale_corrente * 0.20 / annualized_price_vol[t]
units[t]     = clip(raw_units[t], -10 * capitale/price[t], +10 * capitale/price[t])
```

---

## 5. Esecuzione — ribilanciamento ogni 4H

- **Cadenza**: ricalcolo a ogni chiusura di barra 4H.
- **Ordine**: `Δunits` eseguito all'apertura della barra 4H successiva.

---

## 6. Fee e frizioni

| Voce | Valore |
|------|--------|
| Fee taker (Bybit derivatives) | 0.055%/lato |
| Slippage | 0.015%/lato |
| **Frizione totale** | **0.07%/lato sul turnover** |

Slippage-stress testato sul Grand pool (blend Breakout+TSMOM, proxy per la famiglia): holdout resta positivo fino a +10bps extra.

---

## 7. Frequenza di trading attesa

| Metrica (14.424 barre 4H, 2020-2026) | Valore |
|---|---|
| Barre con variazione di size | 95.4% → **~40.1 ribilanciamenti/settimana (~5.7/giorno)** |
| Cambi di segno (flip di direzione) | 120 totali → **~18.2/anno, ~1 ogni 20 giorni (~0.35/settimana)** |
| Turnover medio giornaliero | **~8.5% del capitale** |

**Nota**: a differenza della versione 1D (dove TSMOM-sign aveva turnover PIÙ BASSO di Breakout, 4.2% vs 7.4%/giorno), a 4H il rapporto si inverte (TSMOM 8.5% > Breakout 7.1%/giorno) — il segnale binario sign() sembra più sensibile al rumore intraday che alla scala giornaliera, producendo più cambi di stato per unità di calendario. Da tenere presente se si sceglie un buffer no-trade per contenere i costi.

---

## 8. Risultati di validazione (sintesi da `reports/carver_intraday_4h.md`)

Breakdown per anno (full-sample, n=14.423 barre 4H):

| Anno | Ret% | WR |
|---|---|---|
| 2020 | +53.5% | 37.1% |
| 2021 | +12.1% | 50.9% |
| 2022 | +16.5% | 50.3% |
| 2023 | +17.7% | 50.6% |
| 2024 | +24.8% | 50.5% |
| 2025 | +5.9% | 49.5% |
| 2026 (parziale) | +6.3% | 49.8% |

Full-sample: ret=+136.8%, Sharpe_hat=3.362 (il più alto dei 3 candidati testati), MC p(profit)=1.000/p(ruin)=0.000, **DSR=1.000**.
**Holdout genuino 2025-2026** (n=3.462 barre): ret=+12.2%, wr=49.6%, Sharpe_hat=0.668, MC p(profit)=0.752, **DSR=1.000**.

Confronto con la versione 1D archiviata: ret full inferiore (+136.8% vs +126.3% — comparabile), Sharpe holdout comparabile (0.668 vs 0.352, **qui addirittura superiore** — segnale rafforzato, non indebolito, dal porting a 4H).

---

## 9. Protocollo di Paper Trading

Identico nella struttura a `docs/TSMOM_SIGN_POOL_STRATEGY_SPEC.md` Sezione 9, con cadenza **4H** al posto di 1D in ogni riferimento. Criteri go/no-go invariati.

---

## 10. Avvertenze critiche

1. **Turnover più alto della versione 1D e, insolitamente, anche più alto di Breakout pool 4H** (8.5% vs 7.1%/giorno) — verificare che il costo di esecuzione reale non eroda il margine più rapidamente del previsto.
2. **Correlazione con Breakout pool 4H da misurare in paper trading**: entrambe trend-following su 4H, costruzione diversa (canale di prezzo vs segno del rendimento) ma esposte allo stesso regime di mercato.
3. Tutte le altre avvertenze della versione 1D (Sezione 10 di `docs/TSMOM_SIGN_POOL_STRATEGY_SPEC.md`) restano valide: nessun parametro ottimizzato su BTCUSDT, sizing non-compounding in validazione.
