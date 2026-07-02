# BTCUSDT Perpetual — Validated Quantitative Strategies

**Simbolo:** BTCUSDT Perpetual (Binance / Bybit)  
**Timeframe entry:** 1H  
**Periodo di validazione:** Gennaio 2020 – Maggio 2026 (6.4 anni)  
**Stato:** Validato via Walk-Forward Out-of-Sample + Monte Carlo

---

## Strategie Validate

| ID | Nome | OOS Return | MaxDD | Win Rate | BE_fee | P(profit) | P(ruin) |
|----|------|-----------|-------|----------|--------|-----------|---------|
| **S08** | Quantile Channel | **+214.3%** | -63.9% | 54.5% | 51.6% | 98.5% | **0.0%** |
| **S07** | OU Mean Reversion | **+78.1%** | -30.2% | 55.5% | ~51.8% | 95.5% | **0.0%** |

> I ritorni OOS sono cumulativi su 6 anni senza leva aggiuntiva oltre il risk-sizing interno.  
> Le strategie S01, S02, S03, S04 non sono state validate (P(ruin) > 10% o OOS negativo).

---

## Architettura del Sistema (3 Layer)

```
[Layer 1] Segnale entry 1H           →  S08 o S07 emette evento (long/short)
[Layer 2] HMM Regime Gate 4H         →  accetta solo BULL (long) o BEAR (short)
[Layer 3] Sizing 4H ATR + fee-aware  →  TP = tp_frac × ATR4H,  SL = sl_frac × ATR4H
```

---

## Layer 1 — Strategie di Entry (1H)

### S08 — Quantile Channel Breakout

**Logica:** Breakout del canale statistico di prezzo a 30 barre. Entra in direzione del breakout quando il prezzo supera la banda esterna con un buffer di sicurezza.

**Indicatori:**
- `q80` = rolling 80° percentile del close su 30 barre 1H (finestra scorrevole)
- `q20` = rolling 20° percentile del close su 30 barre 1H

**Regole di Entry:**
```
LONG  : close[i] > q80[i] × 1.01     # breakout sopra banda superiore (+1% buffer)
SHORT : close[i] < q20[i] / 1.01     # breakout sotto banda inferiore (-1% buffer)
```

**Requisiti tecnici:**
- Warmup minimo: 32 barre 1H (per stabilizzare il quantile rolling)
- `q20 > 0` (evita divisioni su prezzi nulli)
- `ATR4H[i] > 0`

**Razionale:** Il breakout della banda q80/q20 identifica espansioni di volatilità direzionale. Il buffer del 1% riduce i falsi breakout. Il filtro HMM (Layer 2) elimina i breakout contro-trend in regime laterale o opposto.

---

### S07 — OU Mean Reversion

**Logica:** Mean-reversion calibrata su processo Ornstein-Uhlenbeck. Entra quando il prezzo è statisticamente "tirato" rispetto al suo equilibrio stimato, ma nella direzione del trend intermedio (SMA30).

**Indicatori:**
- `ou_z` = z-score OU su finestra rolling 30 barre di `log(close)`
- `sma30` = Simple Moving Average 30 barre del close

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
- `ATR4H[i] > 0`

**Razionale:** L'OU z-score modella la tendenza del prezzo a tornare verso un valore di equilibrio stimato localmente. La condizione SMA30 fa sì che si entri solo in dip all'interno di un trend (non contro-trend puri).

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
n_components  = 3        # BULL, SIDEWAYS, BEAR
covariance_type = "full"
n_iter        = 500
random_state  = 42
```

**Labeling automatico degli stati:**
- Calcola il rendimento medio (`log_ret` medio) per ogni stato
- Stato con media più alta → **BULL**
- Stato con media più bassa → **BEAR**
- Stato intermedio → **SIDEWAYS**

**Distribuzione osservata (full-sample 2020-2026):**
| Stato | N barre 4H | Mean log-ret 4H |
|-------|-----------|----------------|
| BULL | 5,313 (37.8%) | +0.039% |
| BEAR | 1,519 (10.8%) | -0.070% |
| SIDEWAYS | 7,226 (51.4%) | +0.018% |

**Regola di filtro:**
```
BULL regime  → accetta solo segnali LONG
BEAR regime  → accetta solo segnali SHORT
SIDEWAYS     → scarta tutti i segnali
```

### Protocollo Causale (zero lookahead bias)

In walk-forward e in live, l'HMM è **sempre riaddestrato su dati passati**:

```
  IS window (6 mesi 4H) ──→ fit_hmm(X_is) ──→ hmm_w
                                                  │
  OOS (14-day tail + 2 mesi) ─→ hmm_w.predict() → regime[t]
                                                  │
  Signal at t ──→ accept/reject based on regime[t]
```

> Il tail di 14 giorni prima del periodo OOS viene incluso per stabilizzare le probabilità di stato iniziali del modello HMM al boundary IS/OOS.

---

## Layer 3 — Sizing e TP/SL (4H ATR)

### ATR a 4H

TP e SL sono calibrati sull'ATR a 4 ore, non sull'ATR 1H. L'ATR 4H viene mappato sulla timeline 1H via forward-fill:

```python
atr4h_mapped = df4h["atr_14"].reindex(df1h.index, method="ffill")
ATR4H = where(atr4h_raw > 0, atr4h_raw, ATR1H)  # fallback su ATR1H
```

**Motivazione:** ATR4H ≈ 4× ATR1H (~1.6% vs ~0.4%), che abbassa il break-even fee-adjusted da ~47-68% (con ATR1H) a ~51-52% (con ATR4H), rendendo le fee relativamente meno distruttive rispetto all'edge.

### Formula TP/SL

```
TP_px = entry_price + direction × tp_frac × ATR4H
SL_px = entry_price − direction × sl_frac × ATR4H
```

dove `direction = +1` per long, `-1` per short.

### Parametri ottimizzati (IS scan)

| Strategia | tp_frac | sl_frac | R:R | WR IS | BE_fee | ExpPnL(adj)/trade |
|-----------|---------|---------|-----|-------|--------|-------------------|
| S08 Quantile Channel | **1.0** | **1.0** | 1:1 | 52.6% | 51.6% | +0.051% |
| S07 OU Mean Reversion | **1.0** | **1.0** | 1:1 | 54.1% | 51.8% | +0.101% |

> R:R simmetrico 1:1 è risultato ottimale per entrambe le strategie con HMM filter. Con TP > SL si abbassa il WR richiesto ma aumenta la volatilità per trade.

### Risk per trade

```python
sl_distance = sl_frac × ATR4H          # distanza in punti prezzo
qty = min(
    equity × RISK_PCT / sl_distance,    # sizing basato su rischio 1% del capitale
    equity × MAX_LEV / entry_price      # cap leva massima 5×
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

### Break-even fee-adjusted

```
BE_fee = (avg_SL% + FEE_RT%) / (avg_TP% + avg_SL%)
```

Con tp_frac=sl_frac=1.0 e ATR4H~1.6%:
- avg_TP% ≈ avg_SL% ≈ 1.6%  
- BE_fee = (1.6 + 0.08) / (1.6 + 1.6) ≈ **52.5%**

WR effettivo S08=54.5%, S07=55.5% → entrambe sopra break-even.

---

## Metodologia di Validazione

### Information Coefficient (IC)

```
IC = Spearman(signal_direction, signed_forward_return)
horizon = 16 barre 1H = 16 ore
```

- `signal_direction = +1` (long) / `-1` (short)
- `signed_forward_return = direction × (close[t+16] - close[t]) / close[t]`
- Validità: `IC > 0` con `p < 0.05`

| Strategia | IC BASE (no filter) | IC HMM filter | Δ |
|-----------|--------------------|--------------|----|
| S08 Quantile Channel | +0.081 | **+0.137** | +0.056 |
| S07 OU Mean Reversion | +0.011 | **+0.093** | +0.082 |

### Walk-Forward Out-of-Sample

```
IS window  : 6 mesi
OOS window : 2 mesi
Step       : 2 mesi
Periodo    : 2020-01 → 2026-05  (~35 finestre)
```

**Sequenza per finestra WFO:**
1. Fit HMM su IS 4H bars
2. IS scan: griglia 4×4 (tp_frac × sl_frac) → seleziona best ExpPnL(adj)
3. Predict regime su OOS con HMM riaddestrato (no lookahead)
4. Run backtest OOS con parametri IS migliori
5. Concatena tutti i periodi OOS → equity curve totale

### Monte Carlo (N=5,000 simulazioni)

Ogni simulazione: permutazione casuale dell'ordine dei trade OOS, ricalcolo equity.

- **P(profit)**: % simulazioni con return finale > 0
- **P(ruin)**: % simulazioni con equity < 5% del capitale iniziale

---

## Protocollo Live Trading

### Setup iniziale

1. Scarica 4H OHLCV degli ultimi 12+ mesi
2. Fit HMM su tutti i 4H bars disponibili → ottieni `BULL_state`, `BEAR_state`
3. Scarica dati 1H degli ultimi 6 mesi
4. Esegui IS scan su dati 1H con HMM filter → ottieni `(tp_frac, sl_frac)` ottimali
5. Configura bot con i parametri risultanti

### Ottimizzazione TP/SL semestrale (obbligatoria)

> ⚠️ **Ogni 6 mesi** va eseguita una nuova ottimizzazione IS scan per aggiornare `tp_frac` e `sl_frac`.

**Procedura:**
```
1. Raccogli 1H bars degli ultimi 6 mesi in produzione
2. Applica HMM filter (usa modello corrente o riaddestra)
3. Esegui is_scan() sulla griglia TP_FRAC × SL_FRAC
4. Seleziona la coppia con ExpPnL(adj) massimo
5. Aggiorna i parametri del bot
6. Logga: data ottimizzazione, n_trade IS, parametri scelti, ExpPnL(adj) IS
```

**Calendario suggerito:** Gennaio e Luglio (fine/inizio semestre).

**Trigger anticipato:** Se nel periodo di 30 giorni consecutivi il WR live scende sotto `BE_fee - 3pp`, eseguire ottimizzazione fuori ciclo.

### Riaddestramento HMM

L'HMM va riaddestrato con maggiore frequenza per catturare cambiamenti di regime:

```
Frequenza consigliata: mensile (ogni 4 settimane)
Input: ultimi 12 mesi di 4H bars
Procedura: fit_hmm(X_12m) → aggiorna BULL_state, BEAR_state
```

> Il modello HMM può "ruotare" i label degli stati tra un fit e l'altro: verificare sempre che il BULL state identificato abbia `mean_log_ret > 0` e il BEAR state `mean_log_ret < 0`.

### Monitoraggio continuo

| Metrica | Soglia di allerta | Azione |
|---------|-----------------|--------|
| WR rolling 30gg | < BE_fee − 3pp | Ottimizzazione TP/SL anticipata |
| WR rolling 30gg | < BE_fee − 6pp | Stop trading, revisione strategia |
| % trade SIDEWAYS filtrati | > 80% in 30gg | Verifica parametri HMM |
| MaxDD live | > 40% | Riduzione size del 50% |
| MaxDD live | > 55% | Stop trading |

---

## Parametri di Configurazione (Summary)

```python
# ─── Dati ───────────────────────────────────────────────────────────────
SYMBOL         = "BTCUSDT"
TIMEFRAME_1H   = "1h"
TIMEFRAME_4H   = "4h"
START_YEAR     = 2020

# ─── Capitale e rischio ──────────────────────────────────────────────────
INIT_CAP       = 100_000        # USD (o unità di conto)
RISK_PCT       = 0.01           # 1% del capitale per trade
MAX_LEV        = 5.0            # leva massima
MAX_HOLD       = 96             # barre 1H (= 4 giorni)

# ─── Fee ─────────────────────────────────────────────────────────────────
FEE_PER_SIDE   = 0.0004         # 0.04% taker (Binance/Bybit)
FEE_RT         = 0.0008         # 0.08% round-trip

# ─── HMM ─────────────────────────────────────────────────────────────────
HMM_STATES     = 3
HMM_COVARIANCE = "full"
HMM_ITER       = 500
HMM_SEED       = 42
HMM_VOL_WINDOW = 10             # barre 4H per rolling vol features
HMM_RETRAIN_MONTHS = 1          # riaddestramento mensile
HMM_HISTORY_MONTHS = 12         # 12 mesi di dati 4H per fit

# ─── TP/SL (aggiornati ogni 6 mesi via IS scan) ──────────────────────────
# S08 Quantile Channel
S08_TP_FRAC    = 1.0            # × ATR4H
S08_SL_FRAC    = 1.0            # × ATR4H

# S07 OU Mean Reversion
S07_TP_FRAC    = 1.0            # × ATR4H
S07_SL_FRAC    = 1.0            # × ATR4H

# ─── Griglia IS scan (ottimizzazione semestrale) ──────────────────────────
TP_FRAC_GRID   = [1.0, 2.0, 3.0, 5.0]
SL_FRAC_GRID   = [0.25, 0.5, 0.75, 1.0]

# ─── IC validation ───────────────────────────────────────────────────────
IC_HORIZON     = 16             # barre 1H
IC_MIN         = 0.0            # soglia minima IC
IC_P_MAX       = 0.05           # soglia p-value

# ─── WFO ─────────────────────────────────────────────────────────────────
WF_TRAIN_M     = 6              # mesi IS
WF_OOS_M       = 2              # mesi OOS
WF_STEP_M      = 2              # passo

# ─── Monte Carlo ─────────────────────────────────────────────────────────
MC_SIMS        = 5_000
MC_RUIN_THRESH = 0.05           # equity < 5% = ruin
```

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

## Performance Summary (WFO OOS — Causal HMM)

### S08 — Quantile Channel

| Metrica | Valore |
|---------|--------|
| OOS Return (6.4 anni) | +214.3% |
| Max Drawdown | -63.9% |
| Win Rate OOS | 54.5% |
| Break-even (fee-adj) | 51.6% |
| WR surplus vs BE_fee | +2.9 pp |
| N trade OOS | 2,828 |
| Binomial p-value | < 0.0001 |
| P(profit) MC | 98.5% |
| P(ruin) MC | 0.0% |
| IC (HMM filtered) | +0.137 (p≈0) |
| Parametri TP/SL | 1.0×ATR4H / 1.0×ATR4H |

### S07 — OU Mean Reversion

| Metrica | Valore |
|---------|--------|
| OOS Return (6.4 anni) | +78.1% |
| Max Drawdown | **-30.2%** ← miglior profilo DD |
| Win Rate OOS | 55.5% |
| Break-even (fee-adj) | ~51.8% |
| WR surplus vs BE_fee | +3.7 pp |
| N trade OOS | 1,165 |
| Binomial p-value | 0.0001 |
| P(profit) MC | 95.5% |
| P(ruin) MC | 0.0% |
| IC (HMM filtered) | +0.093 (p=0.0014) |
| Parametri TP/SL | 1.0×ATR4H / 1.0×ATR4H |

---

## Avvertenze e Limiti

1. **Regime shift**: Il modello HMM cattura regimi storici. Un cambiamento strutturale di mercato (es. regolamentazione cripto, ETF spot massicci) potrebbe rendere i regimi non stazionari.

2. **Slippage**: Il backtest assume esecuzione al prezzo di close della barra di entry. In live, entry su 1H close introduce slippage reale → usare limit order sul prossimo open o accettare 0.01-0.02% di slippage addizionale.

3. **Liquidità**: BTCUSDT Perpetual su Binance/Bybit ha liquidità sufficiente per size fino a $5M. Per posizioni > $500k valutare VWAP entry.

4. **S07 campione ridotto**: 1,165 trade OOS in 6.4 anni = ~182 trade/anno. Intervallo di confidenza del WR più ampio rispetto a S08. Monitorare con attenzione nei primi 6 mesi live.

5. **Correlazione S07-S08**: Le due strategie sono parzialmente correlate (entrambe usano ATR4H e HMM). Tenere risk cumulato sotto il 2% del capitale se si tradano in simultanea.

---

*Documento generato da pipeline di validazione quantitativa — BTCUSDT 2020-2026*  
*Ultimo aggiornamento: 2026-07-02*
