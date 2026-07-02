# BTCUSDT Perpetual — Validated Quantitative Strategies

**Simbolo:** BTCUSDT Perpetual (Binance / Bybit)  
**Timeframe entry:** 1H  
**Periodo di validazione:** Gennaio 2020 – Maggio 2026 (6.4 anni)  
**Stato:** Validato via Walk-Forward Out-of-Sample + Monte Carlo

---

## Strategia Validata

| ID | Nome | OOS Return | MaxDD | Win Rate | BE_fee | P(profit) | P(ruin) |
|----|------|-----------|-------|----------|--------|-----------|---------|
| **S07** | OU Mean Reversion | **+129.2%** | -25.7% | 56.6% | 51.9% | 99.4% | **0.0%** |

> Unica strategia che supera la pipeline completa WFO + Monte Carlo senza lookahead bias.  
> S08 Quantile Channel è stata esclusa (OOS -2.4%, P(ruin)=14.1% dopo correzione del bias — vedi nota).

---

## Nota metodologica: correzione lookahead bias 4H

Una versione precedente di questa analisi riportava S08 a +214.3% OOS e S07 a +78.1% OOS. Tali numeri erano gonfiati da un lookahead bias sottile: i valori ATR4H e il regime HMM erano mappati sulla timeline 1H tramite `reindex(method="ffill")` senza `shift(1)`, il che assegnava a ogni 1H bar i valori calcolati sulla barra 4H **corrente non ancora chiusa** (anziché sulla barra 4H precedente già chiusa).

**Fix applicato:**
```python
# ERRATO — usa la barra 4H corrente (non ancora chiusa):
df4h["atr_14"].reindex(IDX, method="ffill")

# CORRETTO — usa la barra 4H precedente già chiusa:
df4h["atr_14"].shift(1).reindex(IDX, method="ffill")
```

Lo stesso shift è stato applicato alla serie dei regimi HMM (full-sample e causal WFO).

Dopo la correzione S07 rimane validata (e migliora: +129.2%), mentre S08 risulta non validata (-2.4% OOS). La robustezza di S07 conferma che la sua edge non dipendeva dal leak.

---

## Architettura del Sistema (3 Layer)

```
[Layer 1] Segnale entry 1H           →  S07 emette evento (long/short)
[Layer 2] HMM Regime Gate 4H         →  accetta solo BULL (long) o BEAR (short)
[Layer 3] Sizing 4H ATR + fee-aware  →  TP = tp_frac × ATR4H,  SL = sl_frac × ATR4H
```

---

## Layer 1 — Strategia di Entry: S07 OU Mean Reversion (1H)

**Logica:** Mean-reversion calibrata su processo Ornstein-Uhlenbeck. Entra quando il prezzo è statisticamente "tirato" rispetto al suo equilibrio stimato, ma nella direzione del trend intermedio (SMA30).

**Indicatori:**
- `ou_z` = z-score OU su finestra rolling 30 barre di `log(close)`
- `sma30` = Simple Moving Average 30 barre del close 1H

**Calcolo OU z-score (per ogni barra `i`):**
```python
# Su log-prezzi x[i-30 : i]
# Regressione OLS: dx_t = alpha + beta * x_{t-1}  (deve avere beta < 0)
# Velocità di mean-reversion: kappa = -beta
# Livello di equilibrio: mu = -alpha / beta
# Std della deviazione: sigma_eq = sigma_eps / sqrt(2*kappa)
# Z-score: (x[i] - mu) / sigma_eq
```

**Regole di Entry:**
```
LONG  : ou_z[i] < -1.0  AND  close[i] > sma30[i]
SHORT : ou_z[i] > +1.0  AND  close[i] < sma30[i]
```

- Soglia `|z| > 1.0`: il prezzo è ≥ 1 deviazione standard dall'equilibrio OU
- Condizione SMA30: filtra le trade in direzione del trend a medio termine (evita mean-reversion contro trend dominante)

**Requisiti tecnici:**
- Warmup minimo: 35 barre 1H
- `beta < 0` nel fit OLS (altrimenti il processo non è stazionario → skip)
- `ATR4H[i] > 0` (dalla barra 4H precedente già chiusa)

**Razionale:** L'OU z-score modella la tendenza del prezzo a tornare verso un valore di equilibrio stimato localmente. La condizione SMA30 fa sì che si entri solo in dip all'interno di un trend, non contro-trend puri. Il filtro HMM (Layer 2) elimina le trade in regime SIDEWAYS dove il mean-reversion non ha direzionalità affidabile.

---

## Layer 2 — HMM Regime Gate (4H)

### Modello

**Tipo:** `GaussianHMM` — 3 stati nascosti, matrice di covarianza piena  
**Libreria:** `hmmlearn 0.3.3`

**Feature input (4H):**
```python
log_ret[t] = log(close[t] / close[t-1])
rolling_vol[t] = std(log_ret[t-9 : t+1])   # 10-bar rolling std
X = column_stack([log_ret, rolling_vol])
```

**Parametri modello:**
```
n_components    = 3        # BULL, SIDEWAYS, BEAR
covariance_type = "full"
n_iter          = 500
random_state    = 42
```

**Labeling automatico degli stati:**
- Calcola il rendimento medio (`log_ret` medio) per ogni stato
- Stato con media più alta → **BULL**
- Stato con media più bassa → **BEAR**
- Stato intermedio → **SIDEWAYS**

**Distribuzione osservata (full-sample 2020-2026):**
| Stato | N barre 4H | Mean log-ret 4H |
|-------|-----------|----------------|
| BULL | ~5,300 (37.7%) | +0.039% |
| BEAR | ~1,520 (10.8%) | -0.070% |
| SIDEWAYS | ~7,230 (51.5%) | +0.018% |

**Regola di filtro:**
```
BULL regime  → accetta solo segnali LONG
BEAR regime  → accetta solo segnali SHORT
SIDEWAYS     → scarta tutti i segnali
```

### Protocollo Causale (zero lookahead bias)

> **CRITICO:** Sia in walk-forward che in live, il regime 4H deve sempre essere letto dalla barra 4H **precedente già chiusa**, non da quella corrente.

```python
# Mapping corretto: shift(1) prima del reindex
hmm_regime_4h_series = pd.Series(states, index=df4h.index)
hmm_regime_1h = hmm_regime_4h_series.shift(1).reindex(df1h.index, method="ffill")
```

**In walk-forward:**
```
IS window (6 mesi 4H) ──→ fit_hmm(X_is) ──→ hmm_w
                                                │
OOS tail (14d prepend + 2 mesi):               │
  hmm_w.predict(X_tail) ──→ states_tail        │
  shift(1) su serie 4H  ──→ regime[t-1]  ──────┤
                                                │
Signal at 1H bar t ──→ usa regime[t-1]   ──────┘
```

---

## Layer 3 — Sizing e TP/SL (4H ATR)

### ATR a 4H — con shift corretto

L'ATR4H deve essere letto sempre dalla barra 4H **precedente già chiusa**:

```python
atr4h = df4h["atr_14"].shift(1).reindex(df1h.index, method="ffill")
ATR4H = np.where(atr4h.values > 0, atr4h.values, ATR1H)   # fallback su ATR1H
```

**Motivazione:** ATR4H ≈ 4× ATR1H (~1.6% vs ~0.4%), che abbassa il break-even fee-adjusted da ~47-68% (con ATR1H) a ~51-52% (con ATR4H).

### Formula TP/SL

```
TP_px = entry_price + direction × tp_frac × ATR4H[prev_bar]
SL_px = entry_price − direction × sl_frac × ATR4H[prev_bar]
```

dove `direction = +1` per long, `-1` per short.

### Parametri ottimizzati per S07

| tp_frac | sl_frac | R:R | WR IS | BE_fee | ExpPnL(adj)/trade |
|---------|---------|-----|-------|--------|-------------------|
| **1.0** | **1.0** | 1:1 | 55.0% | 51.9% | **+0.130%** |

> R:R simmetrico 1:1 è l'ottimale per S07 con HMM filter. WR IS al 55% è stabile — la soglia BE_fee al 51.9% lascia un margine di 3.1pp.

### Risk per trade

```python
sl_distance = sl_frac × ATR4H_prev_bar
qty = min(
    equity × RISK_PCT / sl_distance,     # sizing basato su rischio 1% del capitale
    equity × MAX_LEV  / entry_price      # cap leva massima 5×
)
notional = qty × entry_price
```

**Parametri:**
```
RISK_PCT = 0.01   # 1% del capitale per trade
MAX_LEV  = 5.0    # leva massima 5×
FEE      = 0.04%  # per side (taker fee Binance/Bybit)
FEE_RT   = 0.08%  # round-trip
MAX_HOLD = 96 barre 1H  # = 4 giorni (time-stop se né TP né SL colpiti)
```

---

## Metodologia di Validazione

### Information Coefficient (IC)

```
IC = Spearman(signal_direction, signed_forward_return)
horizon = 16 barre 1H = 16 ore
```

| Strategia | IC BASE | IC EMA_FILT | IC HMM_FILT | Δ HMM vs BASE |
|-----------|---------|------------|------------|----------------|
| S07 OU Mean Rev. | -0.001 | +0.001 | **+0.095** | +0.096 |
| S08 Quantile Ch. | +0.078 | +0.083 | +0.173 | +0.095 |

> L'IC elevato di S08 dopo HMM filter (0.17) non si trasla in WR tradeable perché il IS scan con la griglia 16-combinazioni seleziona parametri leggermente negativi in OOS (WR=45.9% < BE_fee=44.7% con margine troppo sottile).

### Walk-Forward Out-of-Sample

```
IS window  : 6 mesi
OOS window : 2 mesi
Step       : 2 mesi
Periodo    : 2020-01 → 2026-05  (~35 finestre)
```

**Sequenza per finestra WFO:**
1. Fit HMM su IS 4H bars (dati strettamente passati)
2. IS scan: griglia 4×4 (tp_frac × sl_frac) → seleziona best ExpPnL(adj)
3. Predict regime su OOS con HMM riaddestrato + `shift(1)` (no lookahead)
4. Run backtest OOS con parametri IS migliori
5. Concatena tutti i periodi OOS → equity curve totale

### Monte Carlo (N=5,000 simulazioni)

- **P(profit)**: % simulazioni con return finale > 0
- **P(ruin)**: % simulazioni con equity < 5% del capitale iniziale

---

## Protocollo Live Trading

### Setup iniziale

1. Scarica 4H OHLCV degli ultimi 12+ mesi
2. Fit HMM su tutti i 4H bars disponibili → ottieni `BULL_state`, `BEAR_state`
3. Scarica dati 1H degli ultimi 6 mesi
4. Esegui IS scan con HMM filter + `shift(1)` → ottieni `(tp_frac, sl_frac)` ottimali
5. Configura bot con i parametri risultanti

### Regime in tempo reale

```python
# Ad ogni nuova barra 1H chiusa:
current_4h_bar_open = floor(now, "4H")           # apertura della barra 4H corrente
prev_4h_bar_open    = current_4h_bar_open - 4h   # barra 4H PRECEDENTE (già chiusa)

regime = hmm_model.predict(features_up_to(prev_4h_bar_open))[-1]
atr4h  = df4h.loc[prev_4h_bar_open, "atr_14"]
```

> Non usare mai i valori della barra 4H corrente — è aperta e non ha ancora il suo ATR finale.

### Ottimizzazione TP/SL semestrale (obbligatoria)

> ⚠️ **Ogni 6 mesi** va eseguita una nuova ottimizzazione IS scan per aggiornare `tp_frac` e `sl_frac`.

**Procedura:**
```
1. Raccogli 1H bars degli ultimi 6 mesi in produzione
2. Applica HMM filter con shift(1) corretto
3. Esegui is_scan() sulla griglia TP_FRAC × SL_FRAC
4. Seleziona la coppia con ExpPnL(adj) massimo
5. Aggiorna i parametri del bot
6. Logga: data, n_trade IS, parametri scelti, ExpPnL(adj) IS, WR IS
```

**Calendario:** Gennaio e Luglio.

**Trigger anticipato:** Se nel periodo di 30 giorni consecutivi il WR live scende sotto `BE_fee - 3pp` (cioè < 48.9%), eseguire ottimizzazione fuori ciclo.

### Riaddestramento HMM

```
Frequenza : mensile (ogni 4 settimane)
Input     : ultimi 12 mesi di 4H bars
Procedura : fit_hmm(X_12m) → verifica BULL mean > 0, BEAR mean < 0
```

> Verificare sempre che il labeling automatico sia coerente: BULL = stato con mean log_ret più alto, BEAR = più basso.

### Monitoraggio continuo

| Metrica | Soglia allerta | Azione |
|---------|---------------|--------|
| WR rolling 30gg | < BE_fee − 3pp (< 48.9%) | Ottimizzazione TP/SL anticipata |
| WR rolling 30gg | < BE_fee − 6pp (< 45.9%) | Stop trading, revisione strategia |
| % trade SIDEWAYS filtrati | > 80% in 30gg | Verifica parametri HMM |
| MaxDD live | > 30% | Riduzione size del 50% |
| MaxDD live | > 40% | Stop trading |

---

## Parametri di Configurazione

```python
# ─── Simbolo e timeframe ────────────────────────────────────────────────────
SYMBOL         = "BTCUSDT"
TF_ENTRY       = "1h"
TF_REGIME      = "4h"

# ─── Capitale e rischio ──────────────────────────────────────────────────────
INIT_CAP       = 100_000
RISK_PCT       = 0.01           # 1% per trade
MAX_LEV        = 5.0
MAX_HOLD       = 96             # barre 1H = 4 giorni

# ─── Fee ─────────────────────────────────────────────────────────────────────
FEE_PER_SIDE   = 0.0004         # 0.04% taker
FEE_RT         = 0.0008         # 0.08% round-trip

# ─── HMM ─────────────────────────────────────────────────────────────────────
HMM_STATES     = 3
HMM_COVARIANCE = "full"
HMM_ITER       = 500
HMM_SEED       = 42
HMM_VOL_WINDOW = 10             # barre 4H per rolling vol features
HMM_SHIFT      = 1              # OBBLIGATORIO: shift(1) prima di reindex su 1H
HMM_RETRAIN_M  = 1              # riaddestramento mensile
HMM_HISTORY_M  = 12            # 12 mesi di dati 4H per fit

# ─── S07 TP/SL (aggiornati ogni 6 mesi via IS scan) ─────────────────────────
S07_TP_FRAC    = 1.0            # × ATR4H dalla barra precedente
S07_SL_FRAC    = 1.0            # × ATR4H dalla barra precedente

# ─── Griglia IS scan ─────────────────────────────────────────────────────────
TP_FRAC_GRID   = [1.0, 2.0, 3.0, 5.0]
SL_FRAC_GRID   = [0.25, 0.5, 0.75, 1.0]

# ─── IC validation ───────────────────────────────────────────────────────────
IC_HORIZON     = 16             # barre 1H
IC_MIN         = 0.0
IC_P_MAX       = 0.05

# ─── WFO ─────────────────────────────────────────────────────────────────────
WF_TRAIN_M     = 6
WF_OOS_M       = 2
WF_STEP_M      = 2

# ─── Monte Carlo ─────────────────────────────────────────────────────────────
MC_SIMS        = 5_000
MC_RUIN_THRESH = 0.05
```

---

## Performance Summary — S07 OU Mean Reversion (WFO OOS, corretto)

| Metrica | Valore |
|---------|--------|
| OOS Return (6.4 anni) | **+129.2%** |
| Max Drawdown | **-25.7%** |
| Win Rate OOS | 56.6% |
| Break-even (fee-adj) | 51.9% |
| WR surplus vs BE_fee | **+4.7 pp** |
| N trade OOS | 1,134 |
| Binomial p-value | < 0.0001 |
| P(profit) MC | **99.4%** |
| P(ruin) MC | **0.0%** |
| IC (HMM filtered, full-sample) | +0.095 (p=0.0007) |
| ExpPnL(adj) IS | +0.130%/trade |
| Parametri TP/SL | 1.0×ATR4H / 1.0×ATR4H |

---

## Dipendenze Software

```
python      >= 3.10
pandas      >= 2.0
numpy       >= 1.24
hmmlearn    == 0.3.3
scipy       >= 1.10
matplotlib  >= 3.7
```

---

## Avvertenze e Limiti

1. **Campione trade OOS**: 1,134 trade in 6.4 anni = ~177/anno. L'intervallo di confidenza del WR è più ampio rispetto a strategie ad alta frequenza di segnale. Monitorare con attenzione i primi 6 mesi live.

2. **Regime shift**: L'HMM è addestrato su dati storici. Un cambiamento strutturale (ETF spot, nuova regolamentazione) potrebbe alterare la stazionarietà dei regimi.

3. **Slippage**: Il backtest assume esecuzione al close della barra 1H di entry. In live usare limit order o accettare 0.01-0.02% di slippage addizionale che riduce l'edge netto.

4. **Barra 4H corrente**: In live NON usare mai ATR o regime HMM della barra 4H aperta. Usare solo l'ultima barra 4H **chiusa**.

5. **Lookahead bias eliminato**: Il bug `shift(1)` è stato identificato e corretto. I numeri in questo documento riflettono un backtest senza lookahead.

---

*Documento generato da pipeline di validazione quantitativa — BTCUSDT 2020-2026*  
*Bug fix lookahead 4H applicato: 2026-07-02*  
*Ultimo aggiornamento: 2026-07-02*
