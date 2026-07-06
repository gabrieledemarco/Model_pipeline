# BTCUSDT Perpetual — ML RandomForest 8h Strategy (Candidata a Paper Trading)

**Simbolo:** BTCUSDT Perpetual (Binance)
**Timeframe base:** 1H (feature multi-timeframe da 5m, 15m, 1H, Daily)
**Periodo di validazione (backtest):** Gennaio 2020 – Giugno 2026
**Stato:** Validata su WFO + holdout genuino 2025-2026 + Monte Carlo + DSR.
**⚠️ NON validata su slippage realistico — vedi "Avvertenze critiche" prima di allocare capitale reale.**
**Prossimo passo raccomandato:** Paper trading live per misurare lo slippage reale (vedi sezione dedicata).

---

## 0. Perché questa strategia e non le precedenti

Nella stessa sessione di ricerca sono state testate e **scartate** (fallite sull'holdout 2025-2026 genuino):
la strategia ADP mean-reversion/pullback (edge concentrato nel 2021), un modello foundation zero-shot (Chronos-Bolt, nessun edge), un classificatore ML su indicatori classici (RSI/CCI/%B/MACD/Stoch/MFI/Choppiness, edge che collassa fuori sample), un sistema multi-timeframe rule-based a 48 combinazioni (holdout pp=0.534, rumore).

Questa è la **prima strategia della sessione** che supera simultaneamente: stabilità per-anno, holdout 2025-2026 mai toccato in fase di fit/selection, Monte Carlo i.i.d. **e** block-bootstrap, e correzione DSR per selection bias multipla. Per questo è la prima a essere documentata come candidata a un test in paper trading — non ancora a capitale reale.

---

## 1. Architettura del Sistema

```
[Dati]     H1 (base) + M5 + M15 + Daily OHLCV, storico continuo
              │
[Feature]  23 feature causali: OHLCV(6) + Pivot MTF(4 TF × 3 = 12) + HMM regime(5)
              │
[Modello]  RandomForestClassifier, walk-forward (6m fit / 2m produzione), 1 per orizzonte
              │
[Segnale]  P(ritorno log a +8h > 0) → soglia di confidenza → long/short/flat
              │
[Trade]    Entry a mercato sulla barra del segnale, uscita A TEMPO FISSO a +8h,
           stop di sicurezza largo (4×ATR) solo anti-crash
              │
[Sizing]   Rischio fisso in $ (1% capitale iniziale), cap di leva 10×
```

---

## 2. Dati richiesti

| Timeframe | Uso | Fonte |
|-----------|-----|-------|
| 1H | Base: target, OHLCV feature, entry/exit, ATR | Binance Vision klines |
| 5m | Pivot MTF (trend + distanza da pivot) | Binance Vision klines |
| 15m | Pivot MTF | Binance Vision klines |
| 1D | Pivot MTF (feature #1 in importanza per tutti i modelli) | Binance Vision klines |

Storico minimo per operare: **12+ mesi di 1H** (per il primo fit IS da 6 mesi + margine), ma il backtest usa storico dal 2020-01-01 per il pivot detection e l'HMM training iniziale. In produzione, scaricare tutto lo storico disponibile per coerenza con il modello validato.

---

## 3. Feature Engineering (23 feature, tutte causali)

### 3.1 OHLCV (6 feature, su base 1H)

```python
ret_1h    = log(close[t] / close[t-1])
ret_4h    = log(close[t] / close[t-4])
ret_24h   = log(close[t] / close[t-24])
atr_pct   = ATR_14[t] / close[t]                      # ATR% standard, no shift extra qui
vol_ratio = volume[t] / rolling_mean(volume, 20)[t]
range_pct = (high[t] - low[t]) / close[t]
```

### 3.2 Pivot Multi-Timeframe (12 feature: 3 per timeframe × 4 timeframe)

Per ciascun timeframe TF ∈ {5m, 15m, 1H, Daily} con parametro fractal `left_right`:

| TF | left_right (barre) |
|----|---------------------|
| 5m | 12 |
| 15m | 8 |
| 1H | 6 |
| Daily | 3 |

**Pivot detection (fractal, causale):** un massimo/minimo alla barra `i` del TF è "pivot" se è il massimo/minimo assoluto su `[i-left_right, i+left_right]`. È **confermato/utilizzabile solo a partire dalla barra `i + right`** (lag di conferma) — mai prima, altrimenti si introduce lookahead.

**Trend state (Dow theory, causale):** tracciando la sequenza di pivot confermati, ad ogni barra si mantiene:
- `trend_state` ∈ {+1 (HH/HL), -1 (LH/LL), 0 (indeterminato)}
- `target_high` = livello dell'ultimo pivot high **non ancora rotto** (NaN se rotto o assente)
- `target_low` = livello dell'ultimo pivot low **non ancora rotto**

**Allineamento sul base 1H:** propagazione causale via `merge_asof(..., direction="backward")` — ogni barra 1H eredita lo stato pivot del TF più recente il cui pivot è già confermato a quel timestamp (mai dati futuri).

**Feature finali per timeframe** (`tf` ∈ {m5, m15, h1, d1}):
```python
{tf}_trend      = trend_state allineato su 1H
{tf}_dist_high  = clip((target_high - close) / ATR_tf, 0, 10)   # 10 se NaN (nessun target)
{tf}_dist_low   = clip((close - target_low)  / ATR_tf, 0, 10)   # 10 se NaN
```
`ATR_tf` = media mobile 14 periodi di `(high-low)` **sul timeframe nativo del pivot**, poi allineata su 1H con lo stesso merge_asof causale.

> **Finding chiave dalla feature importance (POC v2):** `d1_dist_high`/`d1_dist_low` (distanza dal pivot Daily) è la feature #1 o #2 in **tutti e tre** i modelli testati (LogReg, GBoost, RandomForest). `ret_24h`, `atr_pct` e le feature HMM seguono. Le feature M5/M15 **non compaiono mai in top-10** — ma rimuoverle ha comunque **peggiorato** il win rate reale (50.8%→47.9% holdout vs 52-54% con il set completo): non-importanza individuale ≠ sicurezza di rimozione (probabile effetto di interazione/random-subspace in RandomForest). **Mantenere tutte e 23 le feature.**

### 3.3 Regime HMM (5 feature)

```python
N_HMM_STATES = 3   # BULL / SIDEWAYS / BEAR (labeling automatico per mean log-ret)
HMM_FEATURE_NAMES = ["hmm_state", "hmm_prob_bear", "hmm_prob_side", "hmm_prob_bull", "hmm_duration"]
```
- Feature di input all'HMM: `[log_return, log_vol_24h]` su base 1H.
- **Fit SOLO sulla finestra IS** (walk-forward), poi `predict` in avanti sulla finestra OOS/produzione — mai rifittato su dati che includano il futuro rispetto al punto di predizione.
- `hmm_duration` = numero di barre consecutive nello stato corrente.

---

## 4. Target e Modello

### 4.1 Target

```python
horizon = 8   # ore — il migliore dei 3 testati (2h, 4h, 8h)
fwd_log_ret[t] = log(close[t+horizon]) - log(close[t])
target[t] = 1 if fwd_log_ret[t] > 0 else 0
```

Il modello predice **il segno** del rendimento a 8 ore, non una soglia di magnitudo — questo è il motivo per cui l'uscita deve essere a tempo fisso (vedi §5).

### 4.2 Modello

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
)
```
Preceduto da `StandardScaler` **fittato solo su IS**, applicato a IS e OOS.

`RandomForest` è stato selezionato come il più robusto tra 3 modelli testati (LogReg, HistGradientBoosting, RandomForest): unico a restare positivo su **tutti e 8** gli orizzonti testati nel POC v2 in holdout genuino.

### 4.3 Walk-Forward (fit e refresh del modello)

```
IS (train)  : 6 mesi
OOS (uso)   : 2 mesi
Step        : 2 mesi   (= cadenza di re-training in produzione)
```

Procedura per ogni finestra:
1. Fit HMM (3 stati) su IS → genera feature HMM per IS e OOS.
2. Fit `StandardScaler` su IS.
3. Fit `RandomForestClassifier` su IS (feature scalate + target).
4. Predici `predict_proba` su OOS (2 mesi) — questi sono gli output usati per generare segnali/trade in quel periodo.
5. Ogni finestra è **indipendente**: nessuno stato è condiviso tra finestre, quindi il modello "corrente" in produzione è sempre quello fittato sugli ultimi 6 mesi disponibili.

---

## 5. Regola di Trading

### 5.1 Segnale

```python
THRESHOLD = 0.55
p = model.predict_proba(X_t)[1]     # P(fwd_log_ret_8h > 0)

if p > THRESHOLD:        direction = +1   # long
elif p < 1 - THRESHOLD:  direction = -1   # short
else:                     direction = 0   # flat, nessun trade
```

### 5.2 Cooldown

```python
COOLDOWN_HOURS = 4   # dopo un segnale eseguito, ignora nuovi segnali per 4 barre 1H
```
Evita posizioni sovrapposte/ridondanti sullo stesso segnale grezzo.

### 5.3 Entry

Entry a mercato al `close` della barra 1H in cui il segnale è generato (`ep = close[t]`).

### 5.4 Exit — a tempo fisso (CRITICO, non TP/SL classico)

```python
max_hold = horizon   # 8 barre 1H

# Uscita normale: al close della barra t + max_hold
exit_price = close[t + max_hold]

# UNICA eccezione: stop di sicurezza anti-crash, largo, quasi mai attivo
safety_sl = entry_price - direction * SAFETY_SL_ATR_MULT * ATR_14[t]
SAFETY_SL_ATR_MULT = 4.0
# Se durante l'holding high/low tocca il safety_sl PRIMA di t+max_hold → esci lì
```

> **Perché non un TP/SL classico (es. 2×ATR / 1×ATR):** verificato con una simulazione indipendente, model-free, che entro una finestra di poche ore uno stop vicino (1×ATR) viene toccato dal **puro rumore** ~3× più spesso di un target lontano (2×ATR) — un'asimmetria strutturale di first-passage-time che nessuno skill direzionale modesto (52-54%) può compensare. La prima versione testata con TP/SL classico dava P&L fortemente negativo nonostante win rate teoricamente sufficiente. L'unica uscita fedele a cosa il modello realmente predice (il segno del rendimento a h ore) è l'uscita a tempo fisso a h.

Nel backtest validato (orizzonte 8h), il safety-stop attiva solo nel **6.8%** dei trade, con perdita per-trade paragonabile (non peggiore) alla peggior uscita a tempo — non introduce code di rischio nascoste.

---

## 6. Position Sizing

```python
RISK_PCT = 0.01           # 1% del capitale INIZIALE per trade (non % del capitale corrente)
MAX_LEV  = 10.0            # cap di leva massima
FEE      = 0.0004          # 0.04% per lato (taker)

risk       = INIT_CAP * RISK_PCT
stop_dist  = abs(entry_price - safety_sl)
units      = min(risk / stop_dist, MAX_LEV * INIT_CAP / entry_price)
notional   = units * entry_price
pnl_dollar = units * (exit_price - entry_price) * direction - FEE * 2 * notional
```

**Perché rischio fisso in $ e non % del capitale corrente:** isola l'edge grezzo dagli effetti di compounding/variance-drag — lezione appresa nel sistema MTF trend-scan precedente, dove il sizing fixed-fractional su variance di payoff elevata portava a "geometric ruin" anche con edge positivo.

**Distribuzione di leva osservata nel backtest (orizzonte 8h):** mediana 0.33×, p95 0.79×, massimo 2.79× — **mai** vicino al cap di 10×. Il cap è quindi un puro safety-net, non un vincolo binding in condizioni normali.

---

## 7. Risultati di Validazione (backtest, orizzonte 8h)

| Metrica | Full-sample (2020-2026) | Holdout genuino 2025-2026 |
|---------|--------------------------|----------------------------|
| N trade | 8,312 | 1,770 |
| Win rate | 52.7% | 51.6% |
| Return totale | **+213.5%** | **+47.4%** |
| Max Drawdown | -12.4% | -12.5% |
| MC i.i.d. — P(profit) / P(ruin) | 1.000 / 0.000 | 0.989 / 0.000 |
| MC block-bootstrap — P(profit) / P(ruin) | 1.000 / 0.000 | 0.974 / 0.000 |
| Exit: time / safety-SL | 93.2% / 6.8% | — |

**Breakdown per anno (full-sample):** positivo 6 anni su 7 (2020 +6.6%→ 2026 -7.1% parziale, solo 378 trade); nessun anno con perdita catastrofica isolata.

**DSR (Deflated Sharpe Ratio), famiglia N=3 orizzonti testati (2h/4h/8h):**

| Orizzonte | Sharpe_hat (full) | DSR (full) | Sharpe_hat (holdout) | DSR (holdout) |
|-----------|---------------------|------------|------------------------|-----------------|
| 2h | -4.996 | 0.000 | -2.679 | 0.000 |
| 4h | +0.075 | 0.000 | -0.291 | 0.000 |
| **8h** | **+4.481** | **1.000** | **+2.146** | **0.937** |

L'orizzonte 8h non è "il migliore di 3 per fortuna" — supera nettamente la correzione per selection bias su una famiglia stretta e genuinamente comparabile.

---

## 8. Avvertenze Critiche — leggere prima di procedere

### 8.1 Sensibilità allo slippage (il rischio principale)

L'edge è **molto più sottile** di quanto suggerisca il return headline. Test con slippage avverso aggiuntivo (oltre alla commissione 0.04%/lato già modellata) applicato sia in entry che in exit:

| Slippage extra | Full-sample Ret% | Holdout Ret% |
|-----------------|-------------------|---------------|
| 0 bps | +213.5% | +47.4% |
| 2 bps | +86.8% | +15.3% |
| **5 bps** | **-103.1%** | **-32.8%** |
| 10 bps | -419.8% | -113.0% |

**L'edge si azzera e inverte tra 2 e 5 bps di slippage aggiuntivo.** Questo NON è uno scenario estremo per BTCUSDT su Binance in condizioni di bassa liquidità, news event o esecuzione a mercato in size. Prima di allocare capitale reale è **obbligatorio** misurare lo slippage realmente ottenibile — da qui la raccomandazione di procedere con un test in **paper trading** (§9), che permette di misurare lo slippage effettivo di un motore di esecuzione reale senza rischiare capitale.

### 8.2 Altri limiti

- **Campione holdout**: 1,770 trade in ~18 mesi (2025-2026) — intervallo di confidenza del win rate non trascurabile; il 2026 nel dato è parziale.
- **Regime shift**: HMM e RandomForest sono addestrati su dati storici 2020-2026 (inclusi eventi come crollo LUNA/FTX, halving 2024). Un cambiamento strutturale di mercato (ETF flows, nuova regolamentazione) può alterare la stazionarietà del segnale.
- **Nessun controllo di funding rate / costi di carry** per posizioni perpetual multi-ora — da aggiungere nel modello di costo del paper trading.
- **Feature pivot Daily = feature #1**: la strategia dipende fortemente da un solo pivot giornaliero: un bug nel calcolo/allineamento di questa feature invaliderebbe silenziosamente gran parte dell'edge. Testare con attenzione extra.

---

## 9. Protocollo Paper Trading

Obiettivo primario del paper trading: **misurare lo slippage reale** e verificare che l'edge sopravviva (§8.1), non solo confermare il segnale.

### 9.1 Setup iniziale

1. Scaricare storico completo 1H + 5m + 15m + Daily per BTCUSDT (Binance Vision o API live), da almeno 2020-01-01 per coerenza con il modello validato.
2. Calcolare le 23 feature causali (§3) sull'intero storico disponibile.
3. Fit dell'ultima finestra IS (ultimi 6 mesi di dati disponibili) → ottenere il modello RandomForest "corrente" + scaler + modello HMM correnti.
4. Avviare il loop di produzione (9.2) usando questo modello fino al prossimo refresh (9.4).

### 9.2 Loop di produzione (ad ogni nuova barra 1H chiusa)

```python
# 1. Aggiorna feature causali con l'ultima barra 1H appena chiusa
#    (attenzione: i pivot 5m/15m/Daily richiedono il lag di conferma right —
#     una barra "chiusa ora" su 1H potrebbe NON aver ancora confermato
#     un pivot 5m/15m/Daily formatosi pochi minuti/ore prima)
features_t = compute_features(t)

# 2. Regime HMM: usa il modello HMM fittato nell'ultimo refresh, predict-only
hmm_feats_t = predict_hmm_features(current_hmm_model, current_sorted_idx, df_up_to(t))

# 3. Proba dal modello RandomForest corrente (scaler fittato nell'ultimo refresh)
X_t = scaler.transform([features_t + hmm_feats_t])
p = current_rf_model.predict_proba(X_t)[0, 1]

# 4. Applica soglia + cooldown (§5.1, §5.2)
# 5. Se segnale attivo: apri posizione PAPER a mercato (o limit, per stimare slippage),
#    registra timestamp, entry_price REALE (non teorico), direction, safety_sl, exit_time = t+8h
# 6. Alla scadenza (t+8h) o al touch del safety_sl: chiudi la posizione paper,
#    registra exit_price REALE
```

### 9.3 Cosa registrare per ogni trade paper (obbligatorio)

| Campo | Motivo |
|-------|--------|
| `entry_price_teorico` (close della barra segnale) vs `entry_price_reale` (fill paper) | **Misura diretta dello slippage effettivo** — il dato mancante di tutta questa validazione |
| `exit_price_teorico` vs `exit_price_reale` | idem in uscita |
| orario esatto di invio ordine vs fill | latenza di esecuzione |
| direzione, proba del modello, orizzonte | per audit del segnale |
| se uscita = time o safety-SL | per confrontare con il 93.2%/6.8% del backtest |

### 9.4 Refresh del modello (cadenza fissa)

```
Frequenza     : ogni 2 mesi (= WF_STEP_M, coerente col backtest)
Procedura     : re-fit HMM + scaler + RandomForest sugli ultimi 6 mesi di dati
Validazione   : verificare che il nuovo modello, testato sugli ultimi 2 mesi
                appena trascorsi (che diventano "OOS" per quel refresh),
                dia risultati in linea con i range storici (§7) prima di
                metterlo in produzione
```

### 9.5 Durata minima e criteri go/no-go verso capitale reale

**Durata minima raccomandata: 3-6 mesi di paper trading** (≈220-440 trade attesi a questo tasso di frequenza), sufficienti per un primo confronto statistico col backtest.

| Condizione | Esito |
|------------|-------|
| Slippage medio realizzato < 2 bps/lato **e** win rate paper entro ±3pp dal 52-53% storico | ✅ Procedere a un allocazione reale ridotta (es. 5-10% del capitale target) |
| Slippage medio 2-5 bps/lato | ⚠️ Edge marginale — rivalutare frequenza/size dei trade, considerare limit order invece di market, o orizzonti diversi da 8h |
| Slippage medio > 5 bps/lato **o** win rate paper sotto 50% per 2+ mesi consecutivi | ❌ Non procedere a capitale reale — l'edge misurato nel backtest non è monetizzabile con l'esecuzione disponibile |
| Safety-stop attiva > 15% dei trade (vs 6.8% atteso) | ⚠️ Segnale di volatilità realizzata diversa dal periodo di backtest — investigare prima di continuare |

### 9.6 Monitoraggio continuo durante il paper trading

| Metrica | Frequenza check | Soglia di attenzione |
|---------|------------------|------------------------|
| Slippage medio per trade | Settimanale | > 3 bps/lato |
| Win rate rolling (ultimi 60 trade) | Settimanale | < 48% |
| Return cumulato vs backtest atteso per lo stesso periodo | Mensile | Divergenza > 2× la dispersione Monte Carlo (§7) |
| % trade su safety-SL | Mensile | > 15% |
| Drift della feature `d1_dist_high`/`d1_dist_low` (verifica bug silenzioso) | Ad ogni refresh modello | Valori fuori range storico osservato |

---

## 10. Parametri di Configurazione (riferimento)

```python
# ─── Simbolo e timeframe ────────────────────────────────────────────────────
SYMBOL         = "BTCUSDT"
TF_BASE        = "1h"
TF_MTF         = ["5m", "15m", "1h", "1d"]

# ─── Capitale e rischio ──────────────────────────────────────────────────────
INIT_CAP       = 100_000
RISK_PCT       = 0.01
MAX_LEV        = 10.0
FEE_PER_SIDE   = 0.0004

# ─── Pivot MTF (left=right, barre native del TF) ────────────────────────────
PIVOT_LR       = {"5m": 12, "15m": 8, "1h": 6, "1d": 3}

# ─── HMM ─────────────────────────────────────────────────────────────────────
HMM_STATES     = 3
HMM_SEED       = 42

# ─── Modello ──────────────────────────────────────────────────────────────
RF_N_ESTIMATORS   = 200
RF_MAX_DEPTH      = 5
RF_MIN_LEAF       = 50
RF_SEED           = 42

# ─── Target / orizzonte ─────────────────────────────────────────────────────
HORIZON        = 8               # ore — validato come migliore su 2h/4h/8h

# ─── Segnale ─────────────────────────────────────────────────────────────────
THRESHOLD      = 0.55
COOLDOWN_HOURS = 4

# ─── Exit ─────────────────────────────────────────────────────────────────
SAFETY_SL_ATR_MULT = 4.0          # anti-crash, non un target di uscita primario

# ─── Walk-Forward (fit/refresh) ──────────────────────────────────────────────
WF_TRAIN_M     = 6
WF_OOS_M       = 2
WF_STEP_M      = 2                # = cadenza di refresh in produzione

# ─── Monte Carlo (per revalidazioni periodiche) ─────────────────────────────
MC_SIMS        = 5_000
```

---

## 11. Dipendenze Software

```
python        >= 3.10
pandas        >= 2.0
numpy         >= 1.24
scikit-learn  >= 1.3     # RandomForestClassifier, StandardScaler
hmmlearn      == 0.3.3   # GaussianHMM
scipy         >= 1.10    # DSR / norm
```

## 12. Codice sorgente di riferimento

| File | Contenuto |
|------|-----------|
| `src/strategy/mtf_swing.py` | Pivot detection causale, trend state, allineamento MTF (`align_htf_to_ltf`) |
| `src/strategy/hmm_regime.py` | Fit/predict HMM 3-stati causale |
| `src/strategy/monte_carlo.py` | Monte Carlo i.i.d./block-bootstrap, Deflated Sharpe Ratio |
| `create_ml_mtf_pivot_hmm_poc_v2_report.py` | POC classificazione + feature importance + confronto modelli/orizzonti |
| `create_ml_trading_strategy_backtest_report.py` | Backtest strategia completa (entry/exit/sizing/fee), WFO + holdout + MC |
| `create_ml_trading_strategy_refinement_report.py` | Slippage sensitivity, DSR family, diagnostica leva/safety-stop |
| `reports/ml_trading_strategy_backtest.md` | Output numerico completo del backtest |
| `reports/ml_trading_strategy_refinement.md` | Output numerico completo dei controlli di rifinitura |

---

*Documento generato dalla pipeline di validazione quantitativa — BTCUSDT 2020-2026.*
*Stato: candidata a paper trading, non a capitale reale, in attesa di misura dello slippage effettivo.*
*Ultimo aggiornamento: 2026-07-06*
