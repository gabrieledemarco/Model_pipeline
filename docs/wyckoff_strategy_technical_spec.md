# Wyckoff Spring/Upthrust Strategy — Specifiche Tecniche
## BTCUSDT Perpetual Futures | Implementazione Live

> **⚠️ NOTA DI CORREZIONE (2026-08-04):** questo documento (e il motore di
> backtest `src/strategy/engine.py` su cui si basa) usa `FEE=0.04%` per lato
> (round-trip 0.08%), stile Binance spot/taker light — **non** le frizioni
> Bybit derivatives reali (taker 0.055% + slippage 0.015%/lato, round-trip
> 0.14%) usate come standard obbligatorio nel resto di questa sessione di
> validazione. Una ri-validazione indipendente con frizioni Bybit corrette
> (`create_wyckoff_bybit_validation_report.py`, stesso segnale causale
> `src/strategy/wyckoff.py`, stop ATR fisso + griglia RR con DSR family al
> posto della WFO Calmar-based di questo documento) **non supera la
> correzione DSR sul full-sample** (DSR max = 0.099 a RR=2.0, mai 1.000) —
> vedi `reports/wyckoff_bybit_validation.md`. L'edge documentato qui sotto
> (+12.7/+18.3% su 6.5 anni) era già sottile con frizioni più leggere;
> **non risulta validato sotto lo standard di costo reale Bybit di questa
> sessione.** Trattare questo documento come archivio storico, non come
> spec pronta per il live.

**Documento:** Specifiche Tecniche v1.0  
**Data:** 2026-06-25  
**Asset:** BTCUSDT Perpetual Futures (Binance)  
**Timeframe primario:** 1H  
**Periodo di test:** 2020-01 → 2026-05 (56.232 barre)

---

## 1. Sommario Esecutivo

La strategia implementa la metodologia Wyckoff classica in forma algoritmica:
individua fasi di **accumulation/distribution** in mercati laterali, cattura
gli eventi di **manipolazione** (Spring e Upthrust), ed entra sulla barra
successiva all'evento con SL/TP calibrati via walk-forward.

### Risultati chiave (OOS stitched, 2020-2026)

| Variante | Return | Max DD | Calmar | Sharpe | Trades | Win Rate |
|---|---|---|---|---|---|---|
| Wyckoff WF-ottimizzato | **+12.7%** | 6.3% | **2.005** | 4.8 | 177 | 56.5% |
| Wyckoff WF-opt + Vol 0.20 | **+18.3%** | 10.3% | **1.778** | 5.2 | 177 | 56.5% |
| Baseline composite | -30.2% | 36.1% | -0.836 | — | 2.636 | — |

### Monte Carlo (10.000 permutazioni, variante WF-opt)

| Metrica | Actual | p5 | p50 | p95 |
|---|---|---|---|---|
| Return | +12.7% | +7.1% | +12.7% | +18.4% |
| Max Drawdown | 6.3% | 3.8% | 6.3% | 9.9% |
| Calmar | 2.005 | 0.97 | 2.005 | 4.61 |
| **P(Return > 0)** | — | **100%** | — | — |
| **P(DD > 30%)** | — | **0%** | — | — |

Il vantaggio è robusto all'ordine delle operazioni: tutte le 10.000 permutazioni
producono rendimento positivo, confermando un edge strutturale e non sequencing luck.

---

## 2. Dati e Universe

### Sorgente
- **Binance Vision CDN** — dati OHLCV storici pubblici, timeframe 1H
- **Endpoint REST Binance** — top-up real-time per dati recenti
- **Cache locale** — directory `data/` con file Parquet per riuso

### Indicatori computati da `add_indicators()`

Tutti calcolati da zero su numpy/pandas senza librerie TA esterne:

| Indicatore | Descrizione | Colonna output |
|---|---|---|
| ATR(14) | Average True Range su 14 periodi (EWM) | `atr_14` |
| ADX(14) | Average Directional Index | `adx` |
| OBV | On-Balance Volume cumulativo | `obv` |
| OBV EMA(21) | EMA a 21 periodi dell'OBV | `obv_ema21` |
| OBV Trend | `sign(OBV - OBV_EMA21)` → {-1, 0, +1} | `obv_trend` |
| Vol Ratio | Volume corrente / rolling mean(20) | `vol_ratio` |
| RVOL(20) | Realized volatility annualizzata su 20 barre | `rvol_20` |
| EMA(200) | Trend filter | `ema_200` |

---

## 3. Logica del Segnale Wyckoff

### Modulo: `src/strategy/wyckoff.py`
### Funzione: `build_wyckoff_signals(df_1h, n_range, adx_max, range_width_max, vol_threshold, session_hours)`

Il pipeline di rilevamento si compone di 5 fasi sequenziali.

---

### Fase 1 — Range Detection

Il mercato è classificato come **ranging** quando sono verificate simultaneamente
tre condizioni:

```
is_ranging = (ADX < adx_max) AND (range_width < range_width_max) AND range_high.notna()
```

dove:

```
range_high  = rolling_max(high.shift(1), n_range)   # massimo delle ultime n barre (no look-ahead)
range_low   = rolling_min(low.shift(1),  n_range)   # minimo delle ultime n barre
range_width = (range_high - range_low) / range_low  # ampiezza relativa
```

**Parametri produzione (1H):**
- `n_range = 24` barre (= 1 giorno)
- `adx_max = 25.0`
- `range_width_max = 0.10` (banda massima 10%)
- `min_periods = max(n_range // 2, 5) = 12`

Il `shift(1)` sull'input è critico: garantisce che i boundary siano calcolati
esclusivamente su barre **chiuse** precedenti, eliminando qualsiasi look-ahead bias.

---

### Fase 2 — Volume Bias: Accumulation vs Distribution

All'interno di un mercato ranging, il bias direzionale smart-money è determinato
dalla posizione dell'OBV rispetto alla sua EMA a 21 periodi:

```
obv_trend = sign(OBV - OBV_EMA21)    # da add_indicators()

accum_bias   = is_ranging AND (obv_trend > 0)   # OBV sale: smart money compra
distrib_bias = is_ranging AND (obv_trend < 0)   # OBV scende: smart money vende
```

**Razionale:** In fase di accumulation, i grandi operatori assorbono offerta
senza fare muovere il prezzo. Il volume cresce senza breakout del range →
l'OBV salirà mentre il prezzo resta laterale. In distribution è il contrario.

---

### Fase 3 — Manipulation Events

#### Spring (setup long)

```
is_spring = is_ranging
         AND accum_bias
         AND (low  < range_low)    # punta sotto il supporto
         AND (close >= range_low)  # chiude DENTRO il range
         AND (vol_ratio >= vol_threshold)  # su volume elevato (≥ 1.3× media)
```

Il **Spring** è una rottura falsa sotto il supporto: i grandi operatori abbassano
il prezzo sotto range_low per triggerare gli stop dei long deboli, poi riacquistano
tutto assorbendo la liquidità. La chiusura dentro il range con volume elevato
è la firma dell'evento.

#### Upthrust (setup short)

```
is_upthrust = is_ranging
           AND distrib_bias
           AND (high > range_high)   # punta sopra la resistenza
           AND (close <= range_high) # chiude DENTRO il range
           AND (vol_ratio >= vol_threshold)
```

Speculare allo Spring: rottura falsa sopra la resistenza per attrarre late longs,
poi distribuzione aggressiva.

---

### Fase 4 — Entry: Next-Bar Execution

Il segnale viene traslato di **una barra** in avanti:

```python
long_trigger  = is_spring.shift(1).fillna(False)
short_trigger = is_upthrust.shift(1).fillna(False)
```

L'ingresso avviene all'**open della barra successiva** all'evento. Questo simula
il flusso live: l'evento si verifica sulla barra T (vista alla chiusura), l'ordine
viene inviato all'open di T+1. Zero look-ahead bias.

**Conflitto stesso istante (raro):** se Spring e Upthrust si verificano sulla
stessa barra T, il segnale viene azzerato (`signal = 0, composite = 0`).

---

### Fase 5 — Composite Score

```
score_long  = (spring_depth / ATR).clip(0.2, 4.0)  ×  vol_ratio.clip(1.0, 4.0)  ×  2
score_short = (upthrust_height / ATR).clip(0.2, 4.0) × vol_ratio.clip(1.0, 4.0) × 2
```

dove:
- `spring_depth    = range_low - low` (quanto ha penetrato il supporto)
- `upthrust_height = high - range_high` (quanto ha penetrato la resistenza)

Score traslato sull'event bar → signal bar (stesso shift della Fase 4).

Il composite score è positivo per long, negativo per short. Usato per analytics
e filtraggio opzionale; nella configurazione di produzione tutti i segnali
validi vengono tradati (no `min_score` threshold).

---

### Filtro sessione

```python
session_hours = (8, 21)   # UTC
not_session = ~((index.hour >= 8) & (index.hour < 21))
signal[not_session]    = 0
composite[not_session] = 0.0
```

Solo operazioni avviate tra le 08:00 e le 20:59 UTC. Esclude la sessione asiatica
notturna che storicamente produce più falsi segnali su BTCUSDT.

---

## 4. Modello di Esecuzione

### Modulo: `src/strategy/engine.py`
### Funzione: `run_backtest(df_1h, signals, ...)`

```
Segnale → end-of-bar T
Entry   → open di T+1
```

**Parametri di rischio:**

| Parametro | Valore | Note |
|---|---|---|
| `RISK_PCT` | 1% | Rischio per trade come frazione dell'equity corrente |
| `ATR_SL` | 2.0× | Stop loss = entry ± 2 × ATR(14) |
| `ATR_TP1` | 2.0× | TP1: chiude 50% della posizione |
| `ATR_TP2` | 4.0× | TP2: chiude 25% della posizione |
| `ATR_TP3` | 6.0× | TP3: chiude 25% restante (runner) |
| `FEE` | 0.04% | Taker fee per side (Binance USDT perp) |
| `leverage` | 1.0× | No leva nel modello di produzione |

**Position sizing (fixed risk):**
```
notional_at_risk = equity × RISK_PCT
position_size    = notional_at_risk / (ATR_SL × atr_14)   # in BTC
```

**Trailing stop dopo TP1:** Quando TP1 viene colpito, lo stop della posizione
residua si muove al break-even (entry price). Questo elimina il rischio di
perdere trades che avevano raggiunto TP1.

**Sizing con volatility target (variante vol 0.20):**
```
vol_scalar = vol_target / rvol_20          # rvol_20 = realized vol annualizzata
vol_scalar = clip(vol_scalar, 0.25, 3.0)  # bounds di sicurezza
effective_risk = RISK_PCT × vol_scalar
```
In periodi di bassa volatilità la size aumenta; in periodi ad alta vol si riduce,
mantenendo il rischio portafoglio stabile attorno al 20% annuo.

---

## 5. Walk-Forward Validation

### Schema temporale

```
Train (IS): 6 mesi  →  Ottimizzazione SL/TP
OOS:        2 mesi  →  Esecuzione con params IS-ottimizzati
Step:       2 mesi  →  Rolling forward
```

**Numero di finestre:** 35 su 2020-01 → 2026-05

Ogni finestra OOS è contigua e non-overlapping con le altre OOS.
La serie di trade OOS viene concatenata in ordine temporale per costruire
la **synthetic equity curve** aggregata.

### In-Sample Optimization (IS)

Grid search su 20 combinazioni di SL × TP1:

```
SL_GRID  = [0.5, 1.0, 1.5, 2.0, 2.5]   × ATR
TP1_GRID = [1.0, 1.5, 2.0, 3.0]         × ATR  (TP2=2×TP1, TP3=3×TP1)
```

**Criterio di selezione:**
- Se `Calmar > 0` → massimizza Calmar
- Se `Calmar ≤ 0` → massimizza Return
- Minimo 5 trade IS per validare il combo
- Fallback su (SL=2.0×, TP1=2.0×) se nessun combo supera min_trades

Il Calmar come metrica IS riduce il rischio di selezionare param combo
con returns alti ma drawdown eccessivi.

---

## 6. Risultati Quantitativi

### 6.1 Frequenza parametri selezionati (35 finestre)

I parametri più frequentemente selezionati nelle finestre IS indicano
la preferenza strutturale del mercato:

- **SL = 0.5×ATR, TP1 = 1.0×ATR** — 8 finestre (23%)  
  Profilo aggressivo: stop stretto, obiettivo rapido
- **SL = 2.0×ATR, TP1 = 2.0×ATR** — 6 finestre (17%)  
  Default classico Wyckoff
- **SL = 1.0×ATR, TP1 = 1.5×ATR** — 5 finestre (14%)

### 6.2 OOS Window Breakdown (Wyckoff WF-opt)

| Metrica | Valore |
|---|---|
| Finestre OOS totali | 35 |
| Finestre con Return > 0 | 21 (60%) |
| OOS t-statistic (vs 0) | +2.42 |
| OOS p-value (one-sided) | 0.0112 |

Il t-test one-sided (H0: return medio OOS = 0) restituisce p = 0.011, significativo
al livello del 5%. Il risultato non è spiegabile dal solo caso.

### 6.3 Multi-Timeframe Comparison

| Timeframe | n_range | Return | Max DD | Calmar | Trades | Note |
|---|---|---|---|---|---|---|
| 15M | 192 | +2.0% | 4.0% | 0.491 | 51 | Pochi segnali |
| **1H** | **24** | **+12.7%** | **6.3%** | **2.005** | **177** | **Ottimale** |
| 4H | 42 | +0.2% | 2.8% | 0.068 | 20 | Troppo sparso |
| 1D | 30 | +4.5% | 1.6% | 2.882 | 10 | N insuff. (p=0.12) |

Il 1H con `n_range=24` è il timeframe ottimale: massimizza il numero di
segnali mantenendo la qualità dei pattern Wyckoff.

---

## 7. Monte Carlo Analysis

### Procedura

```
N_SIM   = 10.000 permutazioni
BATCH   = 1.000 sims per volta (controllo memoria)
Seed    = 42 (riproducibilità)
Capital = 100.000 USDT iniziali
```

Per ogni simulazione:
1. Permuta casualmente l'ordine dei trade OOS reali
2. Costruisce la equity curve: `equity[i+1] = equity[i] + trade_pnl[perm[i]]`
3. Registra: total_return%, max_drawdown%, Calmar ratio

Questa tecnica isola l'effetto del sequencing: se la distribuzione delle
simulazioni è centrata sul risultato actual, l'edge è strutturale.

### Risultati

**Wyckoff WF-ottimizzato (177 trade OOS)**

| Statistica | Valore |
|---|---|
| P(Return > 0) | **100%** |
| P(Drawdown > 30%) — ruin | **0%** |
| P(Drawdown < 20%) | 100% |
| Expected Return (media sim) | +12.7% |
| Expected Max DD | 6.3% |
| Return p1 / p99 | +6.3% / +19.6% |

**Wyckoff WF-opt + Vol 0.20 (177 trade OOS)**

| Statistica | Valore |
|---|---|
| P(Return > 0) | **100%** |
| P(Drawdown > 30%) | **0%** |
| Expected Return | +18.3% |
| Expected Max DD | 10.3% |

**Baseline composite + Vol 0.20 (2.669 trade)**

| Statistica | Valore |
|---|---|
| P(Return > 0) | 0% |
| P(Drawdown > 30%) | 100% |

**Conclusione:** Il Wyckoff presenta edge statisticamente positivo, robusto
all'ordine delle operazioni. Il Baseline non ha edge.

---

## 8. Parametri di Produzione Raccomandati

```python
# src/strategy/wyckoff.py
build_wyckoff_signals(
    df_1h            = df_1h,
    n_range          = 24,          # 1 giorno look-back
    adx_max          = 25.0,        # soglia ranging
    range_width_max  = 0.10,        # banda max 10%
    vol_threshold    = 1.3,         # volume confirmation
    session_hours    = (8, 21),     # UTC, solo sessione EU+US
)

# src/strategy/engine.py
run_backtest(
    atr_sl_override   = None,       # da WF: varia per finestra, tipicamente 0.5-2.0
    atr_tp1_override  = None,       # da WF: tipicamente 1.0-2.0
    vol_target        = 0.20,       # volatility targeting 20% annuo (variante raccomandata)
    dd_halt_pct       = None,       # opzionale: 0.15 per circuit breaker a 15% DD
    min_score         = None,       # tutti i segnali validi (no filtro score in prod)
    leverage          = 1.0,        # no leva
)
```

**SL/TP per implementazione live senza WF retraining:**
Usare i parametri dell'ultima finestra IS disponibile, o come fallback sicuro:
`atr_sl = 1.5×, atr_tp1 = 2.0×` (media ponderata delle selezioni storiche).

---

## 9. Checklist Implementazione Live

### Pre-deployment

- [ ] **Feed dati real-time:** Websocket Binance BTCUSDT 1H kline + volume tick
- [ ] **Indicatori:** Calcolo rolling di ATR(14), ADX(14), OBV, OBV_EMA21, vol_ratio, rvol_20
- [ ] **Latency:** Il segnale si forma alla chiusura della candela 1H. Ordine inviato entro i primi secondi della candela T+1
- [ ] **Tipo ordine:** Market order all'open oppure limit a ±0.05% dallo spot (slippage minimo su BTCUSDT perp)
- [ ] **SL/TP:** Ordini OCO (One-Cancels-Other) piazzati immediatamente post-entry
- [ ] **TP1 → Break-even:** Dopo fill TP1, aggiornare SL al prezzo di entry

### Gestione posizione

- [ ] **Una posizione alla volta:** No pyramiding — `IN_POS` flag globale
- [ ] **Position size:** `risk_usdt = equity × 0.01`; `btc_size = risk_usdt / (atr_sl × atr_14)`
- [ ] **Notional cap:** Limitare la posizione a max 50% dell'equity per sicurezza
- [ ] **Vol target scaling:** Se `rvol_20 > 0.80` (alta vol), ridurre size del 50%

### Risk management

- [ ] **Circuit breaker:** Sospendere trading se DD from peak > 15% (`dd_halt_pct = 0.15`)
- [ ] **Max loss giornaliero:** Stop trading per la giornata se perdita > 3% equity
- [ ] **WF Retraining:** Ogni 2 mesi rieseguire IS optimization sugli ultimi 6 mesi per aggiornare SL/TP

### Monitoring

- [ ] Log ogni segnale con: timestamp, direction, `composite_score`, `range_width`, `vol_ratio`, ADX
- [ ] Dashboard equity curve real-time vs benchmark
- [ ] Alert se `n_trades_live / expected_rate < 0.3` (anomalia segnali)
- [ ] Alert se win_rate live < 40% su ultimi 30 trade (degradation check)

### Condizioni di mercato da monitorare

Il Wyckoff perde efficacia in:
- Mercati fortemente trending (ADX > 35 sostenuto): già filtrato dal `adx_max`
- Crash improvvisi (flash crash): il vol_threshold limita l'esposizione ma non elimina il rischio
- Liquidity crises: spread allargati aumentano lo slippage effettivo vs modello

---

## 10. Struttura Codice

```
Model_pipeline/
├── src/strategy/
│   ├── wyckoff.py          # build_wyckoff_signals() — segnali
│   ├── engine.py           # run_backtest() — esecuzione e P&L
│   ├── indicators.py       # add_indicators() — tutti gli indicatori
│   ├── data_fetcher.py     # fetch_extended_data() — download + cache
│   ├── signals.py          # build_signal_matrix() — baseline composite
│   └── optimizer.py        # apply_filters(), ScenarioConfig
├── create_wyckoff_report.py        # WF validation 1H
├── create_wyckoff_multitf_report.py# Multi-TF comparison
├── create_montecarlo_report.py     # Monte Carlo simulation
└── reports/
    ├── report_wyckoff.html         # Risultati WF + parametri per finestra
    ├── report_wyckoff_multitf.html # Confronto 15M/1H/4H/1D
    └── report_montecarlo.html      # Distribuzione 10.000 simulazioni
```

---

## 11. Limitazioni e Rischi

1. **Sample size OOS:** 177 trade in 6 anni sono statisticamente sufficienti
   per Calmar/Sharpe ma pochi per stimare tail risk precisi. Il Monte Carlo
   assume stazionarietà dei trade — violata in periodi di regime change.

2. **Sopravvivenza del pattern:** Il pattern Wyckoff è noto pubblicamente.
   Con la diffusione di bot algoritmici, le fasi di manipulation potrebbero
   diventare più brevi o meno affidabili. Monitorare la win rate live vs OOS.

3. **Slippage non modellato:** Il modello usa open price della barra T+1.
   In live, market order su 1H può avere slippage di 0.02-0.10% rispetto
   all'open teorico, specialmente in momenti di alta volatilità.

4. **Funding rate:** Il modello non include il funding rate (tipicamente
   ±0.01% ogni 8h). Su posizioni overnight multi-giorno può incidere
   significativamente. Monitorare l'esposizione netta nelle fasi di
   funding rate anomalo (> ±0.10%).

5. **Overfitting IS:** La grid search su 20 combos per finestra è
   relativamente contenuta. Tuttavia, in finestre con pochi segnali IS
   (< 5), il fallback a SL=2.0×/TP1=2.0× introduce discontinuità.
   In live, considerare di usare parametri fissi se la finestra IS
   corrente ha < 10 segnali.

---

*Branch: `claude/btcusdt-quant-strategy-y6hft2`*  
*Repository: `gabrieledemarco/Model_pipeline`*
