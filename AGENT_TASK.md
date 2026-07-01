# Agent Task: SQLite Persistence Layer for Paper Trading

## Contesto per l'agent

Questo file è il briefing completo per un agente Claude che deve implementare uno strato di persistenza SQLite sul sistema di paper trading già presente nel repository. Leggi questo file prima di qualsiasi altro, poi segui l'ordine di lettura indicato nella sezione 2.

---

## 1. Cos'è questo progetto

Strategia quantitativa multi-timeframe su BTCUSDT perpetual futures (Binance). L'approccio è puramente rule-based (no ML): combina segnali da 10 timeframe e indicatori (weekly, daily, 4H, 1H, 15m, 1m + OI, funding, volume, ciclicità) in un **composite score** pesato. Un backtest event-driven su dati storici 2022–2026 valida la strategia; il walk-forward (WFO) ne testa la robustezza OOS.

**Stato del progetto al momento di questo task:**

| Componente | File chiave | Stato |
|---|---|---|
| Fetcher dati storici | `src/strategy/data_fetcher.py` | ✅ completo |
| Indicatori tecnici | `src/strategy/indicators.py` | ✅ completo |
| Matrice segnali | `src/strategy/signals.py` | ✅ completo (10 segnali + 15m + 1m) |
| Backtest engine | `src/strategy/engine.py` | ✅ completo (4 risk controls) |
| Walk-forward | `src/strategy/walk_forward.py` | ✅ completo |
| Optimizer / scenari | `src/strategy/optimizer.py` | ✅ completo |
| Ablation 15m+1m | `ablation_15m_1m.py` | ✅ eseguito, PDF in reports/ |
| Drawdown study | `drawdown_study.py` | ✅ eseguito, PDF in reports/ |
| Report HTML | `create_html_report.py` | ✅ eseguito, HTML in reports/ |
| **Paper trading live** | `src/live/` + `run_paper_trading.py` | ✅ codice scritto, **NON ancora integrato con DB** |
| **SQLite persistence** | `src/live/db.py` | ❌ **DA IMPLEMENTARE** |

---

## 2. Ordine di lettura consigliato

Leggi i file in questo ordine per capire il progetto:

```
1.  src/strategy/engine.py          # backtest engine: capire Trade, sizing, TP/SL logic
2.  src/strategy/signals.py         # WEIGHTS, build_signal_matrix, _align
3.  src/live/paper_trader.py        # PaperTrader: on_bar(), _try_entry(), _check_exits()
4.  src/live/bootstrap.py           # come vengono caricati i dati storici + REST
5.  src/live/feed.py                # WebSocket feed: KlineBar, stream_klines()
6.  src/live/runner.py              # LiveRunner: bootstrap → feed loop → signal → traders
7.  run_paper_trading.py            # entry point, STRATEGY_CATALOGUE (4 strategie)
```

Non è necessario leggere tutti gli altri file per questo task.

---

## 3. Cosa è stato implementato nel paper trading (da un agente precedente)

### Architettura `src/live/`

```
src/live/
  __init__.py
  bootstrap.py     ← carica parquet cache + appende via Binance REST
  feed.py          ← WebSocket asincrono: btcusdt@kline_{1h,15m,1m}
  paper_trader.py  ← state machine: posizione, TP parziali, risk controls
  runner.py        ← orchestratore async: bootstrap → loop → segnali → traders
```

### Flusso dati (run_paper_trading.py → runner.py)

```
startup
  └─ LiveRunner.bootstrap()
       ├─ bootstrap.py: carica 1H da parquet cache (~38k barre) + REST recente
       ├─ bootstrap.py: carica 15M da cache (~154k barre) + REST
       ├─ bootstrap.py: carica 1M solo REST ultimi 500 bars
       └─ bootstrap.py: fetch funding rate + premium index (OI proxy)

per ogni closed KlineBar (WebSocket)
  ├─ interval == "1m"  → append to _buf_1m  (max 250 barre)
  ├─ interval == "15m" → append to _buf_15m (max 600 barre)
  └─ interval == "1h"  → _on_1h_close():
       ├─ add_indicators(_buf_1h)
       ├─ resample 1H → 4H, 1D, 1W  + add_indicators su ognuno
       ├─ add_indicators(_buf_15m)  → per s_15m
       ├─ add_indicators(_buf_1m)   → per s_1m
       ├─ build_signal_matrix(tf_data, oi_df, funding, premium_1h, df_15m, df_1m)
       ├─ apply_filters(signals, SCENARIOS["Session 08-21"])
       ├─ per ogni PaperTrader:
       │    trader.on_bar(ts, open, high, low, close, atr, rvol, signal, composite)
       │    → ritorna list[str] con eventi (ENTER / EXIT / TP1)
       ├─ stampa dashboard ogni 4H
       └─ salva state JSON ogni 4H  (data/paper_trading_state.json)
```

### PaperTrader: logica esatta

`on_bar(ts, open_px, high_px, low_px, close_px, atr, rvol, signal, composite)`:

1. Se `_pos is not None`: `_check_exits(ts, open, high, low, close)` — stesso ordine del backtest (stop prima, poi TP3>TP2>TP1)
2. Se `_pos is None` e `_pending_dir != 0`: `_try_entry(ts, open_px, atr, rvol, composite)`
3. Reset `_pending_dir = 0`
4. Se `_pos is None` e `signal != 0`: `_pending_dir = signal` (si eseguirà al prossimo bar)
5. `_mark_equity(close_px)` → appende a `_equity_history`

`_try_entry()` implementa tutti i 4 risk controls:
- `min_score`: salta se `|composite| < min_score`
- `dd_halt_pct`: pausa totale se DD > 1.5×soglia, half-size se DD > soglia
- `vol_target`: scala `RISK_PCT` inversamente a `rvol_20` (clip 0.25–3.0)
- `max_notional_pct`: cap sulla notional come % dell'equity

### 4 strategie pre-configurate (in `run_paper_trading.py`)

| Chiave | Label | max_notional | vol_target | dd_halt | min_score |
|---|---|---|---|---|---|
| `max_return` | Max Return (no controls) | 100% | off | off | off |
| `min_dd` | Min DD (Cap10+Vol15+CB15) | 10% | 15% | 15% | off |
| `balanced` | Balanced (Cap20+Vol20+CB15) | 20% | 20% | 15% | off |
| `conservative` | Conservative (+score≥8) | 20% | 20% | 15% | 8.0 |

---

## 4. Task: Implementare SQLite Persistence

### Obiettivo

Ogni volta che il paper trader processa una candela 1H, i dati devono essere salvati su un DB SQLite locale in `data/paper_trading.db`. Il DB deve permettere di:
- Ricostruire l'equity curve di ogni sessione
- Analizzare tutti i trade chiusi (PnL, durata, reason)
- Confrontare le performance tra strategie nella stessa sessione
- Analizzare la distribuzione dei segnali nel tempo
- Riprendere una sessione dopo un riavvio (append, non sovrascrittura)

### File da CREARE

#### `src/live/db.py` — Database layer

Schema completo:

```sql
-- Una riga per ogni avvio del runner (possono esserci più sessioni nel tempo)
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,          -- ISO UTC
    scenario        TEXT NOT NULL,          -- "Session 08-21"
    strategy_key    TEXT NOT NULL,          -- "balanced", "min_dd", ecc.
    label           TEXT NOT NULL,          -- label estesa
    initial_capital REAL NOT NULL,
    max_notional_pct REAL,
    vol_target      REAL,
    dd_halt_pct     REAL,
    min_score       REAL
);

-- Snapshot equity ogni bar 1H per ogni sessione/trader
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    ts          TEXT NOT NULL,
    equity      REAL NOT NULL,
    cash        REAL NOT NULL,
    pnl_pct     REAL NOT NULL,
    current_dd  REAL NOT NULL,             -- % da picco (negativo)
    in_position INTEGER NOT NULL           -- 0/1
);

-- Segnali calcolati su ogni bar 1H chiuso (condivisi tra tutte le sessioni)
CREATE TABLE IF NOT EXISTS bar_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL UNIQUE,      -- ISO UTC del bar open
    open        REAL, high REAL, low REAL, close REAL,
    atr         REAL,
    rvol        REAL,
    composite   REAL,
    signal      INTEGER,                   -- -1 / 0 / +1 (pre-filter)
    s_weekly    REAL, s_daily  REAL, s_4h      REAL,
    s_1h        REAL, s_oi     REAL, s_funding REAL,
    s_vol       REAL, s_cycle  REAL,
    s_15m       REAL, s_1m     REAL
);

-- Trade chiusi
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    entry_ts    TEXT NOT NULL,
    exit_ts     TEXT NOT NULL,
    direction   INTEGER NOT NULL,          -- +1 long / -1 short
    entry_px    REAL NOT NULL,
    exit_px     REAL NOT NULL,
    size_btc    REAL NOT NULL,
    notional    REAL NOT NULL,             -- entry_px * size_btc
    net_pnl     REAL NOT NULL,
    exit_reason TEXT NOT NULL,             -- "stop_loss" / "tp1" / "tp2" / "tp3"
    duration_h  INTEGER,
    tp1_hit     INTEGER,                   -- 0/1
    tp2_hit     INTEGER,                   -- 0/1
    score       REAL                       -- composite score at entry
);

-- Metriche aggregate (calcolate ogni 4H o ogni 24H)
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    ts              TEXT NOT NULL,
    total_return_pct REAL,
    sharpe_approx   REAL,                  -- da equity_snapshots degli ultimi N bar
    max_dd_pct      REAL,
    n_trades        INTEGER,
    win_rate        REAL
);
```

Funzioni da esporre nel modulo:

```python
def get_connection(db_path: str | Path = "data/paper_trading.db") -> sqlite3.Connection
def init_db(conn) -> None                          # CREATE TABLE IF NOT EXISTS
def create_session(conn, strategy_key, label, initial_capital, **risk_params) -> int  # ritorna session_id
def save_bar_signal(conn, ts, ohlcv: dict, signals_row: pd.Series, atr, rvol) -> None
def save_equity_snapshot(conn, session_id, ts, trader: PaperTrader) -> None
def save_trade(conn, session_id, trade: CompletedTrade) -> None
def save_performance_snapshot(conn, session_id, ts, trader: PaperTrader) -> None
def load_session_equity(conn, session_id) -> pd.DataFrame    # per plot
def load_session_trades(conn, session_id) -> pd.DataFrame    # per analisi
def load_all_sessions(conn) -> pd.DataFrame                  # overview
```

Dettagli implementativi importanti:
- Usa `sqlite3` dalla stdlib (nessuna dipendenza extra)
- Connessione con `check_same_thread=False` (chiamata da thread asyncio)
- Usa `conn.execute("PRAGMA journal_mode=WAL")` per performance async
- Ogni `INSERT` deve essere seguito da `conn.commit()` o wrappato in `with conn:`
- `save_bar_signal` usa `INSERT OR IGNORE` su `ts` (colonna UNIQUE) → no duplicati su riavvio
- `save_trade` viene chiamato dal `PaperTrader` SOLO quando un trade si chiude completamente

### File da MODIFICARE

#### `src/live/paper_trader.py`

Aggiungi hook opzionale per il salvataggio su DB. Il PaperTrader NON deve dipendere direttamente da `db.py` (separation of concerns) — invece espone una callback:

```python
# Nel __init__:
self.on_trade_closed: callable | None = None   # callback(trade: CompletedTrade) -> None

# In _exit_full(), dopo aver fatto append a self.trades:
if self.on_trade_closed is not None:
    self.on_trade_closed(self.trades[-1])
```

Stessa cosa per equity snapshots — il runner può già leggere `trader._equity_history` ma è meglio aggiungere:

```python
self.on_equity_update: callable | None = None  # callback(ts, equity, cash, dd) -> None
# chiamata alla fine di on_bar(), dopo _mark_equity()
```

#### `src/live/runner.py`

In `LiveRunner.__init__()`:
```python
from src.live.db import get_connection, init_db, create_session, ...

# Se db_path è fornito:
self._conn = get_connection(db_path)
init_db(self._conn)
```

In `bootstrap()` o `run()`, dopo aver creato i trader:
```python
for key, trader in self.traders.items():
    cfg = ...   # dai parametri del trader
    session_id = create_session(
        self._conn, key, trader.label, trader.initial_capital,
        max_notional_pct=trader.max_notional,
        vol_target=trader.vol_target,
        dd_halt_pct=trader.dd_halt_pct,
        min_score=trader.min_score,
    )
    self._session_ids[key] = session_id
    
    # Wiring callbacks
    trader.on_trade_closed = lambda t, sid=session_id: save_trade(self._conn, sid, t)
    trader.on_equity_update = lambda ts, eq, cash, dd, sid=session_id: \
        save_equity_snapshot_raw(self._conn, sid, ts, eq, cash, dd, ...)
```

In `_on_1h_close()`, dopo il calcolo dei segnali e prima del loop sui trader:
```python
save_bar_signal(self._conn, ts, bar_ohlcv, row_sig, atr_val, rvol_val)
```

In `_print_dashboard()` o in un metodo separato chiamato ogni 24H:
```python
for key, trader in self.traders.items():
    save_performance_snapshot(self._conn, self._session_ids[key], ts, trader)
```

#### `run_paper_trading.py`

Aggiungi argomento `--db-path`:
```python
p.add_argument("--db-path", default="data/paper_trading.db",
               help="Path al file SQLite (default: data/paper_trading.db)")
```

Passa `db_path` a `LiveRunner`:
```python
runner = LiveRunner(traders, verbose=verbose, db_path=args.db_path)
```

### File da CREARE (opzionale ma utile)

#### `query_results.py` — CLI per interrogare il DB

```
Usage:
  python query_results.py sessions               # lista tutte le sessioni
  python query_results.py equity <session_id>    # equity curve in ASCII
  python query_results.py trades <session_id>    # ultimi N trade
  python query_results.py compare                # confronto metriche tutte le sessioni
  python query_results.py export <session_id>    # CSV dei trade
```

---

## 5. Step-by-step per l'agent

Esegui i passi in questo ordine. Non saltare la fase di lettura.

### Fase 1 — Lettura e comprensione

```
1. Leggi src/live/paper_trader.py  → capisci PaperTrader, CompletedTrade, _exit_full
2. Leggi src/live/runner.py        → capisci _on_1h_close, dove vengono chiamati i trader
3. Leggi run_paper_trading.py      → capisci argparse e STRATEGY_CATALOGUE
```

### Fase 2 — Crea `src/live/db.py`

Implementa tutte le funzioni descritte nella sezione 4.
Testa il modulo in isolamento:
```python
# test rapido
from src.live.db import get_connection, init_db, create_session
conn = get_connection("data/test_paper.db")
init_db(conn)
sid = create_session(conn, "balanced", "Balanced Test", 100000.0,
                     max_notional_pct=0.20, vol_target=0.20,
                     dd_halt_pct=0.15, min_score=None)
print(f"Session ID: {sid}")
conn.close()
import os; os.remove("data/test_paper.db")
```

### Fase 3 — Modifica `src/live/paper_trader.py`

Aggiungi i due callback `on_trade_closed` e `on_equity_update` come descritto. Non rompere l'interfaccia esistente — sono opzionali (default `None`).

### Fase 4 — Modifica `src/live/runner.py`

Integra il DB nel runner. Aggiungi `db_path: str | Path | None = None` al `__init__`. Se `db_path is None`, il DB è disabilitato (compatibilità backward). Crea le sessioni nel `bootstrap()` dopo aver confermato i trader.

### Fase 5 — Modifica `run_paper_trading.py`

Aggiungi `--db-path` all'argparse, passalo a `LiveRunner`.

### Fase 6 — (Opzionale) Crea `query_results.py`

### Fase 7 — Verifica

```bash
# 1. Syntax check
python -m py_compile src/live/db.py
python -m py_compile src/live/paper_trader.py
python -m py_compile src/live/runner.py
python -m py_compile run_paper_trading.py

# 2. Unit test db.py
python -c "
from src.live.db import get_connection, init_db, create_session, load_all_sessions
import pandas as pd, os
conn = get_connection('data/test.db')
init_db(conn)
sid = create_session(conn, 'balanced', 'Test', 100000.0,
                     max_notional_pct=0.20, vol_target=0.20,
                     dd_halt_pct=0.15, min_score=None)
print('Sessions:', load_all_sessions(conn))
conn.close()
os.remove('data/test.db')
print('DB test OK')
"

# 3. Unit test paper_trader callbacks
python -c "
from src.live.paper_trader import PaperTrader
from datetime import datetime
trades_saved = []
t = PaperTrader('test', 100_000, max_notional_pct=0.20)
t.on_trade_closed = lambda trade: trades_saved.append(trade)
# bar 1: signal
t.on_bar(datetime(2025,1,1,8), 90000,91000,89000,90500, 1500,0.35, 1, 7.0)
# bar 2: entry + stop hit (low < sl)
t.on_bar(datetime(2025,1,1,9), 90100, 90200, 84000, 85000, 1500,0.35, 0, 6.0)
print(f'Trade saved: {len(trades_saved)} — reason={trades_saved[0].exit_reason if trades_saved else \"none\"}')
print('Callback test OK')
"
```

### Fase 8 — Commit

```bash
git add src/live/db.py src/live/paper_trader.py src/live/runner.py run_paper_trading.py
# se creato: git add query_results.py
git commit -m "feat: add SQLite persistence layer for paper trading results"
git push -u origin claude/btcusdt-quant-strategy-y6hft2
```

---

## 6. Note importanti per l'agent

- **Non modificare** la logica di trading in `paper_trader.py` — solo aggiungere i callback
- **Non modificare** `engine.py`, `signals.py`, `indicators.py` — sono componenti stabili
- Il DB deve supportare **append** tra sessioni diverse: se il runner viene riavviato, crea una nuova sessione (nuovo `session_id`) ma usa `INSERT OR IGNORE` per `bar_signals` (evita duplicati sui timestamp già presenti)
- Il campo `notional` in `trades` è `entry_px * size_btc`
- `sharpe_approx` in `performance_snapshots` può essere calcolato con `mean(returns) / std(returns) * sqrt(8760)` sulle ultime N equity snapshots (8760 = ore/anno)
- La connessione SQLite deve essere aperta **una sola volta** nel runner e tenuta aperta per tutta la sessione (non aprire/chiudere ad ogni write)
- In ambienti async (asyncio), wrappa le chiamate SQLite con `asyncio.to_thread()` se causano blocking — ma per questo volume di dati (1 write/ora per strategia) non è necessario

---

## 7. Risultati attesi dal report HTML (per contesto)

Dalle simulazioni storiche 2022–2026 (IS):

| Strategia | Return | Sharpe | Max DD | Calmar | Trades |
|---|---|---|---|---|---|
| Baseline (8 segnali) | +266% | 2.45 | -75.4% | 3.53 | 1794 |
| Extended (+15m+1m) | +299% | 2.48 | -75.4% | 3.97 | 1836 |
| balanced (Cap20+Vol20+CB15) | +37% | 1.19 | -27.7% | 1.33 | 1531 |
| min_dd (Cap10+Vol15+CB15) | +28% | 1.04 | -21.8% | 1.26 | 1836 |
| conservative (+score≥8) | +37% | 1.19 | -27.4% | 1.35 | 1491 |

Walk-forward OOS (23 finestre da 2 mesi):

| Strategia | OOS Return | OOS Sharpe | OOS Max DD | Win% Windows |
|---|---|---|---|---|
| Baseline | +173% | 2.87 | -36.8% | 74% |
| Extended | +191% | 2.82 | -31.6% | 74% |
| balanced | +46% | 1.17 | -17.0% | 74% |
