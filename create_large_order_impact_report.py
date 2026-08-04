#!/usr/bin/env python3
"""
create_large_order_impact_report.py
======================================
Event study sul market impact di "grandi ordini": esiste potere
predittivo a breve termine nel piazzamento di un grande ordine? Come
influenza i movimenti di prezzo successivi (impatto permanente vs
temporaneo)?

Questa è un'analisi DIAGNOSTICA/di ricerca, non un backtest di strategia:
nessuna fee, nessun sizing, nessuno stop — solo il return grezzo forward,
condizionato all'evento, con test di significatività, esattamente lo
strumento giusto per rispondere alla domanda "c'è un segnale qui?" prima
di costruire qualunque regola di trading sopra.

Proxy del "grande ordine" (limite dichiarato dei dati): Binance klines non
espone il tape trade-by-trade né l'order book — l'informazione più vicina
disponibile è OHLCV + taker_buy_base + n_trades per barra 1 minuto.
  avg_trade_size[i] = volume[i] / n_trades[i]
è un proxy MIGLIORE del semplice "volume alto" (già testato e bocciato in
`candle_diagnostics.md`, effetto <2.2bps): una barra con volume alto E
poche transazioni è compatibile con pochi trade grandi; una barra con
volume alto e MOLTE transazioni è invece tanti trade piccoli — il volume
di barra da solo confonde questi due casi.
  delta[i] = 2*taker_buy_base[i] - volume[i]
è il flusso aggressivo netto della barra (proxy standard di order-flow da
klines) — usato per classificare l'evento come "grande BUY" (delta>0) o
"grande SELL" (delta<0), invece del colore candela (già mostrato non
predittivo di per sé).

Metodologia:
  1. EVENTO: avg_trade_size[i] nel percentile top P di una finestra
     rolling causale di WIN_BIG=1440 barre (24h) — due soglie, P=99%
     (top 1%) e P=99.9% (top 0.1%), per un check dose-response (se
     l'effetto è reale, dovrebbe essere più forte per eventi più estremi).
  2. DIREZIONE: BUY se delta[i]>0, SELL se delta[i]<0.
  3. FORWARD RETURN: per ogni orizzonte h in [1,5,15,30,60,120,240,480]
     minuti, fwd_ret[i,h] = (open[i+1+h]-open[i+1])/open[i+1] — causale,
     misurato da subito dopo la barra evento (già chiusa) in poi.
  4. SIGNIFICATIVITÀ: Welch t-test tra il gruppo evento e il gruppo di
     controllo (tutte le altre barre, stessa metrica fwd_ret) per ogni
     orizzonte — non un singolo numero puntuale ma una vera verifica
     statistica.
  5. DECOMPOSIZIONE temporaneo/permanente: si confronta la traiettoria
     del mean forward return attraverso gli orizzonti — se decade verso
     zero (o si inverte), è impatto TEMPORANEO (consumo di liquidità); se
     tiene/cresce, è impatto PERMANENTE (informativo).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_binance_vision_taker_flow

SEP = "═" * 78
START_YEAR = 2020
WIN_BIG = 1440                    # 24h rolling window per il percentile causale
PCTL_TIERS = [0.99, 0.999]        # top 1%, top 0.1%
HORIZONS = [1, 5, 15, 30, 60, 120, 240, 480]   # minuti

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Large-Order Impact Study — potere predittivo del piazzamento di grandi ordini")
w(SEP)
w("\nAnalisi diagnostica (event study), non backtest: nessuna fee/sizing.")
w("Proxy 'grande ordine' = avg_trade_size (volume/n_trades) in percentile alto")
w("di finestra rolling 24h causale. Direzione = segno del delta (flusso")
w("aggressivo netto). Forward return + Welch t-test per 8 orizzonti, 2 soglie.")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 1m + taker flow (taker_buy_base, n_trades) …")
df1m = fetch_binance_vision_taker_flow("1m", start_year=START_YEAR, start_month=1,
                                        workers=6, verbose=False)
IDX = df1m.index
N = len(df1m)
print(f"  1m: {N:,} bars  ({IDX[0].date()} → {IDX[-1].date()})  (loaded in {time.time()-t0:.0f}s)")

OP = df1m["open"].values.astype(float)
VOL = df1m["volume"].values.astype(float)
TBB = df1m["taker_buy_base"].values.astype(float)
NTR = np.maximum(df1m["n_trades"].values.astype(float), 1.0)

AVG_SIZE = VOL / NTR
DELTA = 2.0 * TBB - VOL

print("[EVENT] Computing rolling causal percentile of avg_trade_size …")
t1 = time.time()
size_pctl = pd.Series(AVG_SIZE).rolling(WIN_BIG, min_periods=WIN_BIG).rank(pct=True).values
print(f"  done in {time.time()-t1:.0f}s")

MAX_H = max(HORIZONS)
valid_base = np.isfinite(size_pctl) & (np.arange(N) < N - MAX_H - 2) & (np.arange(N) >= WIN_BIG)
direction = np.where(DELTA > 0, 1, np.where(DELTA < 0, -1, 0))

# ── Forward returns per ogni orizzonte, precomputati una volta ───────────
# fwd_ret[i] = (OP[i+1+h]-OP[i+1])/OP[i+1]  (causale: entry = open barra i+1)
print("[FWD] Precomputing forward returns per horizon …")
fwd = {}
for h in HORIZONS:
    ep = np.full(N, np.nan)
    xp = np.full(N, np.nan)
    ep[: N - h - 1] = OP[1: N - h]
    xp[: N - h - 1] = OP[1 + h: N]
    fwd[h] = np.where(ep > 0, (xp - ep) / ep, np.nan)

# ── Event study per tier di percentile e direzione ────────────────────────
summary_rows = []

for pctl in PCTL_TIERS:
    is_event = valid_base & (size_pctl >= pctl) & (direction != 0)
    n_events = int(is_event.sum())
    w(f"\n{SEP}")
    w(f"SOGLIA: avg_trade_size nel top {(1-pctl)*100:.2f}% (finestra 24h)  —  n eventi totali = {n_events:,} "
      f"({n_events/N:.3%} delle barre)")
    w(SEP)

    for label, dsign in [("Grande BUY (delta>0)", 1), ("Grande SELL (delta<0)", -1)]:
        mask_ev = is_event & (direction == dsign)
        n_ev = int(mask_ev.sum())
        mask_ctrl = valid_base & ~is_event
        w(f"\n  {label}  —  n={n_ev:,}")
        w(f"    {'Hold':>6}  {'n':>9}  {'mean fwd%':>11}  {'95% CI':>22}  {'t-stat':>8}  {'p-value':>9}  {'ctrl mean%':>11}")
        for h in HORIZONS:
            fh = fwd[h]
            ev_vals = fh[mask_ev]
            ev_vals = ev_vals[np.isfinite(ev_vals)]
            ctrl_vals = fh[mask_ctrl]
            ctrl_vals = ctrl_vals[np.isfinite(ctrl_vals)]
            if len(ev_vals) < 20:
                continue
            mean_ev = ev_vals.mean() * 100
            sem_ev = ev_vals.std(ddof=1) / np.sqrt(len(ev_vals)) * 100
            ci_lo, ci_hi = mean_ev - 1.96 * sem_ev, mean_ev + 1.96 * sem_ev
            tstat, pval = st.ttest_ind(ev_vals, ctrl_vals, equal_var=False)
            mean_ctrl = ctrl_vals.mean() * 100
            w(f"    {h:>4}m  {len(ev_vals):>9,}  {mean_ev:>+10.4f}%  "
              f"[{ci_lo:>+7.4f}%,{ci_hi:>+7.4f}%]  {tstat:>8.2f}  {pval:>9.4f}  {mean_ctrl:>+10.4f}%")
            summary_rows.append(dict(pctl=pctl, label=label, h=h, n=len(ev_vals),
                                      mean_ev=mean_ev, mean_ctrl=mean_ctrl, tstat=tstat, pval=pval))

# ── Riepilogo: decomposizione temporaneo/permanente ───────────────────────
w(f"\n{SEP}")
w("RIEPILOGO — traiettoria del forward return per orizzonte (temporaneo vs permanente)")
w(SEP)
for pctl in PCTL_TIERS:
    w(f"\n  Soglia top {(1-pctl)*100:.2f}%:")
    w(f"    {'Direzione':<24} " + "".join(f"{h:>9}m" for h in HORIZONS))
    for label in ["Grande BUY (delta>0)", "Grande SELL (delta<0)"]:
        rows = [r for r in summary_rows if r["pctl"] == pctl and r["label"] == label]
        rows.sort(key=lambda r: r["h"])
        line = f"    {label:<24} " + "".join(f"{r['mean_ev']:>+9.4f}" for r in rows)
        w(line)
    w(f"    {'-> significativo (p<0.05)':<24} ", )
    for label in ["Grande BUY (delta>0)", "Grande SELL (delta<0)"]:
        rows = [r for r in summary_rows if r["pctl"] == pctl and r["label"] == label]
        rows.sort(key=lambda r: r["h"])
        flags = "".join(f"{'   sig*  ' if r['pval'] < 0.05 else '   n.s.  '}" for r in rows)
        w(f"      {label:<22} {flags}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/large_order_impact.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Large-Order Impact Study — potere predittivo del piazzamento di grandi ordini\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
