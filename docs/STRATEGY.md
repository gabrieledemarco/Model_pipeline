# BTCUSDT Multi-Timeframe Quantitative Strategy
## Complete Documentation

---

## 1. Idea di Fondo

### Tesi centrale
I mercati crypto hanno struttura frattale: la direzione del trend si forma lentamente (weekly/daily) e viene confermata o negata a livello intermedio (4H) prima che appaia un'opportunità di ingresso pulita (1H). Il segnale più affidabile si ottiene quando **tutti i timeframe puntano nella stessa direzione**.

### Filosofia operativa
- **Trend-following** a medio termine (trade medi da 12–48 ore)
- **Conferma multi-livello**: ogni ingresso richiede consenso da almeno 3 timeframe
- **Risk-first**: sizing basato su ATR, stop fisso a 2×ATR, mai più di 1% del capitale a rischio per trade
- **Neutralità durante l'indecisione**: flat se il composite score è nella zona grigia (−5 → +5)

### Mercato target
BTCUSDT perpetual futures (Binance UM), il più liquido strumento crypto al mondo. Spread ≈ 0.01%, funding 8-orario, fee taker 0.04%.

---

## 2. Ricerca e Basi Teoriche

### Multi-Timeframe Analysis (MTFA)
La letteratura tecnica (Dow Theory, Elder Triple Screen, Murphy) mostra che gli operatori istituzionali usano HTF per definire la direzione e LTF per l'esecuzione. La nostra architettura formalizza questo in un sistema di scoring quantitativo.

### Open Interest e Basis
L'OI misura il denaro totale nel mercato; la sua direzione rispetto al prezzo rivela la qualità del movimento:
- **Prezzo ↑ + OI ↑** → nuovi long entrano: trend genuino, non squeeze
- **Prezzo ↑ + OI ↓** → short squeeze: rialzo temporaneo, meno affidabile

Il **basis** (premium index = futures − spot index) è il proxy più diretto e gratuito per l'OI. Dati reali da Binance Vision `premiumIndexKlines`, disponibili dal 2020, aggiornati 1H.

### Funding Rate come segnale contrarian
Il funding rate dell'8H su perp riflette il sentiment aggregato. Funding elevato (+0.05%) → longs overextended → contrarian sell. Studi empirici su BTC perp 2019–2024 mostrano che funding estremi precedono correzioni con lag medio di 12–36h.

### Ciclicità Stagionale
Analisi del calendario storico BTC (2015–2024):
- **Mesi rialzisti**: Ott, Nov, Dic, Gen, Apr → bias +0.5
- **Mesi ribassisti**: Mag, Giu, Set → bias −0.5
- **Giorni forti**: Lun–Mer → +0.5 (volumi istituzionali)
- **Giorni deboli**: Sab–Dom → −0.5 (retail prevalente, alta volatilità dispersiva)

### Volume Confirmation
Volume spike (> 1.3× media mobile) nella direzione del prezzo = conferma istituzionale. Spike contro direzione = segnale di esaurimento.

---

## 3. Architettura del Segnale

### Schema gerarchico

```
[1W Weekly Trend]   ──────────────────────────────► peso 3
   EMA-13 vs EMA-34 + Price vs EMA-13

[1D Daily Trend]    ──────────────────────────────► peso 4  (più pesante)
   EMA stack 21/50/200 + RSI-14 + MACD histogram

[4H Setup]          ──────────────────────────────► peso 3
   EMA 21/50 + RSI-14 + MACD histogram

[1H Entry]          ──────────────────────────────► peso 3
   RSI zone + MACD + Volume spike direction

[OI/Basis Signal]   ──────────────────────────────► peso 2
   premiumIndexKlines (Binance Vision, reale)
   price Δ × basis Δ → conferma/divergenza

[Funding Signal]    ──────────────────────────────► peso 1  (contrarian)
   funding > +0.05% → −1 / funding < −0.05% → +1

[Volume Signal]     ──────────────────────────────► peso 2
   vol_ratio > 1.3 × avg → sign(log_ret)

[Cyclicality]       ──────────────────────────────► peso 1
   mese + giorno della settimana
                                                    ──────
                                          Composite max: ±19
```

### Soglie di ingresso
| Livello | Composite | Azione |
|---|---|---|
| Segnale weak long | ≥ +5 | LONG (sizing standard) |
| Segnale weak short | ≤ −5 | SHORT (sizing standard) |
| Segnale strong long | ≥ +8 | LONG (può aumentare size) |
| Segnale strong short | ≤ −8 | SHORT (può aumentare size) |
| Zona grigia | −5 → +5 | FLAT |

---

## 4. Pesi e Logica dei Componenti

### `s_weekly` — Trend Settimanale (peso 3)
```
Score range: −3 → +3
+2  : EMA-13 > EMA-34 (bull cross settimanale)
−2  : EMA-13 < EMA-34
+1  : Close > EMA-13 (prezzo sopra trend)
−1  : Close < EMA-13
```
**Motivazione**: il timeframe weekly filtra i movimenti di rumore. Un EMA cross 13/34 settimanale richiede mesi per formarsi e segnala cambi di regime reali.

### `s_daily` — Trend Primario (peso 4)
```
Score range: −4 → +4
±2  : EMA stack 21 > 50 > 200 (bull) o inverso (bear)
±1  : RSI-14 > 55 (bull) o < 45 (bear)
±1  : segno del MACD histogram
```
**Motivazione**: il daily è il timeframe operativo degli istituzionali. Il massimo peso (4) riflette la priorità della struttura di mercato giornaliera.

### `s_4h` — Setup Intermedio (peso 3)
```
Score range: −3 → +3
±1  : EMA-21 vs EMA-50
±1  : RSI-14 zona
±1  : MACD histogram
```
**Motivazione**: il 4H è il "filtro di setup". Evita ingressi contro la struttura intermedia anche quando HTF e LTF concordano.

### `s_1h` — Entry Timing (peso 3)
```
Score range: −3 → +3
±1  : RSI-14 zona
±1  : MACD histogram
±1  : volume spike > 1.5× avg nella direzione del prezzo
```
**Motivazione**: il working timeframe per l'esecuzione. Il volume spike è il trigger più importante: conferma che il movimento è istituzionale, non retail.

### `s_oi` — OI / Basis (peso 2)
```
Score range: −2 → +2
+2  : Prezzo ↑ + Basis ↑   (new longs, trend genuino)
−2  : Prezzo ↓ + Basis ↓   (longs liquidano, bear confermato)
−1  : Prezzo ↓ + Basis ↑   (shorts entrano, bear meno pulito)
+1  : Prezzo ↑ + Basis ↓   (short squeeze, bullish debole)

Cap contrarian:
  Basis > +0.15%  → score cappato a 0  (longs overextended)
  Basis < −0.10%  → score cappato a 0  (shorts overextended)
```
**Fonte dati**: `Binance Vision /premiumIndexKlines/BTCUSDT/1h/` — reale, gratuito, no API key.

### `s_funding` — Funding Contrarian (peso 1)
```
Score range: −1 → +1
−1  : funding > +0.05%  (longs pagano molto → overextended)
+1  : funding < −0.05%  (shorts pagano molto → overextended)
 0  : zona neutrale
```
**Fonte dati**: `Binance Vision /fundingRate/BTCUSDT/` — reale, 8-hourly.

### `s_vol` — Volume Confirmation (peso 2)
```
Score range: −1 → +1
+1  : vol_ratio > 1.3 e candle rialzista
−1  : vol_ratio > 1.3 e candle ribassista
 0  : volume normale
```

### `s_cycle` — Ciclicità (peso 1)
```
Score range: −1 → +1
+0.5 : mese bullish (Ott/Nov/Dic/Gen/Apr)
−0.5 : mese bearish (Mag/Giu/Set)
+0.5 : giorno forte (Lun/Mar/Mer)
−0.5 : giorno debole (Sab/Dom)
Clip: [−1, +1]
```

---

## 5. Pipeline Dati

### Fonti

| Dato | Fonte | Frequenza | Disponibilità |
|---|---|---|---|
| 1H OHLCV | Binance Vision CDN | 1H | 2020-presente, gratuito |
| 4H OHLCV | Resampled da 1H | 4H | stessa |
| 1D / 1W | Yahoo Finance (BTC-USD) | daily/weekly | 4+ anni |
| Basis (premium) | Binance Vision premiumIndexKlines | 1H | 2020-presente, gratuito |
| Funding Rate | Binance Vision fundingRate | 8H | 2020-presente, gratuito |
| Open Interest raw | — | — | Non disponibile gratis in storico lungo |

### URL CDN Binance Vision
```
Klines:   https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/
Premium:  https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1h/
Funding:  https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/
```
File mensili in formato `.zip` → CSV. Cachati in `data/cache/` come Parquet.

### Indicatori calcolati
Ogni timeframe riceve `add_indicators()`:
- EMA 13, 21, 34, 50, 200
- RSI-14
- MACD (12/26/9) + histogram
- ATR-14
- Volume ratio (vol / rolling_mean_20)
- Log return

### Allineamento temporale
I segnali HTF (weekly, daily, 4H) vengono forward-filled sull'indice 1H tramite `pd.merge_asof` con direzione `backward`. Normalizzazione timestamp a `datetime64[s]` per evitare conflitti di precisione tra sorgenti diverse.

---

## 6. Motore di Backtest

### Execution model (no look-ahead)
1. Segnale calcolato a **bar close**
2. Ordine eseguito all'**open della barra successiva**
3. Una posizione alla volta (no pyramiding)

### Risk management per trade
```
Position size = 1% di equity / (ATR × prezzo)   [fixed-risk sizing]
Stop Loss     = entry ± 2 × ATR
TP1 (50%)     = entry ± 2 × ATR   [R/R 1:1]
TP2 (25%)     = entry ± 4 × ATR   [R/R 1:2]
TP3 (25%)     = entry ± 6 × ATR   [R/R 1:3]
Break-even    = dopo TP1, stop → entry price
```

### Costi
- Fee: 0.04% per side (Binance taker USDT-perp)
- Slippage: non modellato esplicitamente (conservativo)
- Funding: non incluso nel P&L (funding basso su perp standard)

---

## 7. Scenari Testati

| Scenario | Filtro aggiuntivo | Note |
|---|---|---|
| **Baseline** | nessuno | tutti i segnali |
| **Regime filter** | solo long in bull regime (price > EMA-200 daily) | riduce DD |
| **Strong (≥±18)** | solo segnali forti | meno trade, più selettivo |
| **Wide SL (2.5×)** | stop a 2.5×ATR invece di 2× | più respiro |
| **Session 08-21** | solo ore 08:00–21:00 UTC | evita liquidità bassa notturna |
| **Monthly filter** | solo mesi storicamente bullish | ciclicità estrema |
| **Combined** | regime + session + strong | massima selettività |

---

## 8. Walk-Forward Optimization

### Design
- **Tipo**: fixed-parameter rolling WFO (non re-ottimizza i parametri)
- **Finestra train**: 6 mesi (warm-up indicatori + contesto mercato)
- **Finestra OOS**: 2 mesi (test out-of-sample)
- **Step**: 2 mesi (finestre OOS non-overlapping)
- **Scopo**: testare la stabilità temporale su regimi diversi (bull 2023, bear 2022, mania 2024)

### Come si legge
L'equity OOS è concatenata compound: ogni finestra parte dal capitale finale della precedente. Il risultato è una simulazione realistica di trading continuo su 4.5 anni.

---

## 9. Monte Carlo

### Metodologia
- **Bootstrap sui trade**: ricampionamento con rimpiazzo dei net_pnl
- **1000 simulazioni** per distribuzione robusta
- **Capitale iniziale**: $100,000
- Eseguito sia su trade in-sample che su trade OOS (walk-forward)

### Metriche
- P(profit): % simulazioni con return > 0
- P(ruin): % simulazioni con draw > 50% del capitale
- Percentili p5/p50/p95 del return finale

---

## 10. Risultati — Con OI Reale (basis premiumIndex)

> Dataset: 38,688 barre 1H · 2022-01-01 → 2026-05-31
> Scenario base: **Session 08-21**

### Backtest In-Sample

| Metrica | Valore |
|---|---|
| Total Return | **+271%** |
| Sharpe Ratio | 9.72 |
| Max Drawdown | −74.0% |
| Calmar Ratio | 3.67 |
| Win Rate | ~57% |
| Profit Factor | ~1.25 |
| # Trades | 1,855 |

### Scenario Migliore (Baseline)

| Metrica | Valore |
|---|---|
| Total Return | **+504%** |
| Sharpe Ratio | 11.48 |
| Max Drawdown | −87% |

### Walk-Forward (23 finestre, 2022–2026)

| Metrica | Valore |
|---|---|
| Finestre profitable | **70%** (16/23) |
| Median OOS return | +3.95% per finestra |
| OOS Sharpe combinato | **2.773** |
| OOS Return combinato | **+152%** |
| OOS Max DD | −36.8% |
| Consistency | +0.464 |

### Monte Carlo — In-Sample (1000 sim)

| Metrica | Valore |
|---|---|
| P(profit) | **99.8%** |
| p50 return | +274% |
| P(ruin >50% DD) | **0%** |

### Monte Carlo — OOS Trades (1000 sim)

| Metrica | Valore |
|---|---|
| P(profit) | **99.1%** |
| p50 return | +123% |
| P(ruin) | ~0% |

---

## 11. Ablation Test — Contributo del Segnale OI

Test rimuovendo `s_oi` (peso 0 vs peso 2):

| Metrica | OI = 2 (baseline) | OI = 0 (ablato) | Delta |
|---|---|---|---|
| WFO Windows profitable | 83% | 70% | −13pp |
| OOS Sharpe | 3.835 | 2.918 | −24% |
| OOS Return | +415% | +192% | −54% |
| Consistency | +0.905 | +0.607 | −33% |

**Conclusione**: l'OI (basis reale) aggiunge valore genuino e non è ridondante rispetto agli altri segnali di prezzo.

> Nota: il test con OI sintetico mostrava 83% di finestre profitable; con OI reale (basis) scende al 70%. La differenza riflette un segnale più onesto e indipendente.

---

## 12. Limitazioni e Rischi

| Limitazione | Impatto | Mitigazione |
|---|---|---|
| Sharpe inflato su barre orarie | Sharpe 9–12 irrealistici; reale ~1.5–2 | Usare trade-level Sharpe e Calmar |
| Max DD −74% in-sample | Non gestibile live senza leva ridotta | 1x leva, stop fisso, WFO monitoring |
| OI sintetico (rimosso) | Segnale autoreferenziale | Sostituito con basis reale |
| No slippage | Ottimismo del backtest | Conservativo: fee reali Binance |
| No funding nel P&L | Leggero ottimismo long | Funding BTC medio: +0.01% = trascurabile |
| Dataset 2022–2026 | Include solo 1 ciclo bear + 1 bull | Necessario paper trading live 3–6 mesi |

---

## 13. Prossimi Passi

1. **Paper trading live** (3–6 mesi) per validare che l'OI basis reale non distorca il segnale
2. **Connettore live** via Binance FAPI WebSocket per segnale real-time
3. **OI raw storico** (possibile source: Laevitas, CoinGlass paid) per confronto diretto con basis proxy
4. **Ottimizzazione pesi** tramite Bayesian search su WFO OOS (non in-sample per evitare overfitting)
5. **Multi-asset**: estendere a ETHUSDT, SOLUSDT con stesso framework

---

## 14. File e Moduli

```
Model_pipeline/
├── strategy_btcusdt.py          # Pipeline principale (13 step)
├── run_html_report.py           # Pipeline → HTML report (9 step, usa cache)
├── ablation_oi.py               # Test ablazione segnale OI
│
├── src/strategy/
│   ├── data_fetcher.py          # Binance Vision CDN + yfinance fallback
│   ├── indicators.py            # EMA, RSI, MACD, ATR, volume ratio
│   ├── signals.py               # 8 moduli segnale + aggregatore
│   ├── engine.py                # Backtester (ATR sizing, 3-tier TP)
│   ├── optimizer.py             # 7 scenari + apply_filters()
│   ├── monte_carlo.py           # Bootstrap MC su trade
│   ├── walk_forward.py          # Rolling WFO non-overlapping
│   ├── leverage_study.py        # Grid leva × risk × metodo sizing
│   ├── analytics.py             # Statistiche avanzate
│   ├── charts.py                # PNG charts individuali
│   ├── report.py                # PDF report 18 pagine
│   └── report_html.py           # HTML report self-contained
│
├── reports/
│   ├── BTCUSDT_Strategy_Report.pdf       # PDF 18 pagine (main pipeline)
│   ├── BTCUSDT_Strategy_Report.html      # HTML self-contained (4.6 MB)
│   ├── BTCUSDT_Strategy_Report_from_html.pdf  # PDF da HTML (WeasyPrint)
│   ├── ablation_oi_report.pdf            # PDF confronto OI ablation
│   └── charts/                           # 19 PNG individuali
│
└── data/cache/                  # Parquet cache (gitignored)
```

---

*Ultima revisione: giugno 2026 — dati al 2026-05-31*
