# BTCUSDT Perpetual — Carver Breakout Pool INTRADAY (4H) Strategy (Candidata a Paper Trading)

**Simbolo:** BTCUSDT Perpetual (Bybit)
**Timeframe base:** **4H** (segnale, sizing ed esecuzione — nessun altro timeframe richiesto)
**Periodo di validazione (backtest):** Gennaio 2020 – Luglio 2026
**Stato:** Validata — **DSR=1.000 sia full-sample sia holdout genuino 2025-2026**, timeframe intraday (sostituisce la versione 1D archiviata in `docs/CARVER_BREAKOUT_POOL_STRATEGY_SPEC.md`, esclusa per il nuovo requisito di sessione "solo intraday").
**⚠️ Non ancora validata su book/tick reali — vedi "Avvertenze critiche" prima di allocare capitale reale.**
**Prossimo passo raccomandato:** Paper trading live per misurare turnover/slippage reale sul ribilanciamento 4H.

---

## 0. Perché questa strategia

Carver Breakout pool su timeframe **1D** era risultata l'unica strategia (tra ~20 testate) con DSR=1.000 su full-sample E holdout genuino, profittevole anche durante il bear market 2025-2026 (`reports/carver_rules.md`) — ma il nuovo requisito di sessione impone timeframe esclusivamente intraday, escludendola dal set live.

Questo documento verifica se lo stesso edge (canale Donchian pooled su 6 velocità, vol-targeting) **sopravvive** portando l'intera infrastruttura a **4H** — stessi lookback riscalati per preservare la finestra di calendario (es. 10 giorni = 60 barre 4H invece di 10 barre 1D), ma con il ribilanciamento (e quindi il costo di turnover) **6× più frequente**. Il risultato (`reports/carver_intraday_4h.md`) conferma DSR=1.000 su entrambi gli ambiti — **l'edge sopravvive al porting intraday**, con un costo in termini di Sharpe holdout leggermente inferiore alla versione 1D (0.689 vs 0.951).

---

## 1. Architettura del Sistema

```
[Dati]        4H OHLCV close, storico continuo
                 │
[Livello 1]   6 canali Donchian paralleli, lookback in barre 4H equivalenti
              a 10/20/40/80/160/320 giorni di calendario (60/120/240/480/
              960/1920 barre), ciascuno smussato con EWMA(span=N/4)
                 │
[Livello 2]   POOL = media delle 6 velocità
                 │
[Sizing]      Vol-targeting: unità = (forecast/10) × capitale × 20% annuo
              / volatilità annualizzata (EWMA su barre 4H, span=150 barre
              ≈ 25 giorni di calendario), clip a ±10× leva
                 │
[Esecuzione]  Ribilanciamento a OGNI chiusura di barra 4H
```

---

## 2. Dati richiesti

| Timeframe | Uso | Storico minimo |
|-----------|-----|-----------------|
| 4H | Canale Donchian (fino a 1920 barre = 320gg), EWMA di volatilità (span 150 barre ≈ 25gg), forecast scalar (expanding mean, min 360 barre ≈ 60gg) | Almeno 1920 barre 4H (~320 giorni) continue prima di poter calcolare il canale più lento |

Fonte usata in validazione: Binance Vision klines 4H (perpetual futures). In produzione, feed OHLCV 4H live Bybit per BTCUSDT Perpetual.

---

## 3. Costruzione del forecast (algoritmo esatto, causale)

Identico al report 1D (`docs/CARVER_BREAKOUT_POOL_STRATEGY_SPEC.md`, Sezione 3), con `N ∈ {60, 120, 240, 480, 960, 1920}` barre 4H (equivalenti a 10/20/40/80/160/320 giorni):

```python
roll_max = price.rolling(N, min_periods=N).max()
roll_min = price.rolling(N, min_periods=N).min()
mid      = (roll_max + roll_min) / 2
half_rng = (roll_max - roll_min) / 2

raw[t]        = 40 * (price[t] - mid[t]) / half_rng[t]
forecast_N[t] = EWMA(raw, span=max(N/4, 2))[t]
forecast_N[t] = clip(forecast_N[t], -20, +20)

BREAKOUT_POOL[t] = mean(forecast_60, forecast_120, forecast_240,
                         forecast_480, forecast_960, forecast_1920)[t]
```

Volatilità annualizzata: `price_vol = EWMA_stdev(bar_return, span=150)` (150 barre 4H ≈ 25 giorni di calendario, stessa finestra della versione 1D), `annualized_price_vol = price_vol * sqrt(365*6)` (2.190 barre 4H/anno).

---

## 4. Position sizing (vol-targeting) — identico alla versione 1D

```python
raw_units[t] = (BREAKOUT_POOL[t] / 10.0) * capitale_corrente * 0.20 / annualized_price_vol[t]
units[t]     = clip(raw_units[t], -10 * capitale/price[t], +10 * capitale/price[t])
```

Nessun concetto di stop-loss/target discreto — controllo del rischio nel vol-targeting.

---

## 5. Esecuzione — ribilanciamento ogni 4H

- **Cadenza**: ricalcolo del forecast e size target a ogni chiusura di barra 4H (00/04/08/12/16/20 UTC).
- **Ordine**: `Δunits` eseguito all'apertura della barra 4H successiva.

---

## 6. Fee e frizioni

| Voce | Valore |
|------|--------|
| Fee taker (Bybit derivatives) | 0.055%/lato |
| Slippage | 0.015%/lato |
| **Frizione totale** | **0.07%/lato sul turnover** |

Slippage-stress (sul Grand pool, proxy per l'intera famiglia — vedi Sezione 8): holdout resta positivo fino a +10bps extra (ret +12.2% vs +16.9% baseline).

---

## 7. Frequenza di trading attesa

| Metrica (14.424 barre 4H, 2020-2026) | Valore |
|---|---|
| Barre con variazione di size | 99.0% → **~41.6 ribilanciamenti/settimana (~5.9/giorno)** |
| Cambi di segno (flip di direzione) | 107 totali → **~16.2/anno, ~1 ogni 22.5 giorni (~0.31/settimana)** |
| Turnover medio giornaliero | **~7.1% del capitale** (sostanzialmente invariato rispetto alla versione 1D, ~7.4%/giorno — il turnover totale per calendario non cambia molto, solo la sua granularità: più operazioni più piccole invece di una grande operazione giornaliera) |

**Implicazione operativa**: con 6 ribilanciamenti/giorno invece di 1, un buffer no-trade (soglia minima di `|Δunits|` prima di eseguire) diventa più rilevante che nella versione 1D per contenere il numero di ordini reali — non testato in questo backtest.

---

## 8. Risultati di validazione (sintesi da `reports/carver_intraday_4h.md`)

Breakdown per anno (full-sample, n=14.423 barre 4H):

| Anno | Ret% | WR |
|---|---|---|
| 2020 | +104.6% | 47.2% |
| 2021 | +10.2% | 49.7% |
| 2022 | +14.9% | 50.2% |
| 2023 | +5.5% | 49.5% |
| 2024 | +45.7% | 51.4% |
| 2025 | +0.7% | 48.4% |
| 2026 (parziale) | +20.5% | 49.7% |

Full-sample: ret=+202.1%, Sharpe_hat=2.850, MC p(profit)=1.000/p(ruin)=0.000, **DSR=1.000**.
**Holdout genuino 2025-2026** (n=3.462 barre): ret=+21.2%, wr=48.9%, Sharpe_hat=0.689, MC p(profit)=0.756, **DSR=1.000**.

Confronto con la versione 1D archiviata: ret full comparabile (+202.1% vs +217.9%), Sharpe holdout leggermente inferiore (0.689 vs 0.951) — il costo del ribilanciamento più frequente si vede, ma **l'edge resta statisticamente valido a DSR=1.000 su entrambi gli ambiti**.

---

## 9. Protocollo di Paper Trading

Identico nella struttura a `docs/CARVER_BREAKOUT_POOL_STRATEGY_SPEC.md` Sezione 9, con cadenza **4H** al posto di 1D in ogni riferimento (loop di ribilanciamento, log dei trade, refresh). Criteri go/no-go invariati; durata minima consigliata 60-90 giorni di paper trading data la maggiore frequenza di ribilanciamento (più osservazioni accumulate più rapidamente).

---

## 10. Avvertenze critiche

1. **Turnover leggermente più concentrato in operazioni piccole e frequenti** — verificare in paper trading che il costo di esecuzione reale (spread, latenza) non ecceda l'assunzione 0.07%/lato su ordini più piccoli e più ravvicinati nel tempo.
2. **Sharpe holdout più basso della versione 1D** (0.689 vs 0.951) — il margine di sicurezza è più sottile; monitorare con particolare attenzione nei primi mesi di paper trading.
3. Tutte le altre avvertenze della versione 1D (Sezione 10 di `docs/CARVER_BREAKOUT_POOL_STRATEGY_SPEC.md`) restano valide: nessun parametro ottimizzato su BTCUSDT, sizing non-compounding in validazione, vol-target 20% annuo non calibrato specificamente.
