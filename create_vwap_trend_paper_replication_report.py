#!/usr/bin/env python3
"""
create_vwap_trend_paper_replication_report.py
=================================================
Replica della strategia descritta in:

  Zarattini, C. & Aziz, A. (2023). "Volume Weighted Average Price (VWAP):
  The Holy Grail for Day Trading Systems." SSRN.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351

Ricerca condotta via web search (SSRN blocca il fetch diretto del PDF —
403 — quindi la metodologia è ricostruita da fonti secondarie che citano
il paper: Concretum Group — il sito degli stessi autori — e Bear Bull
Traders, entrambe fonti dirette/autorevoli).

REGOLA DEL PAPER (letterale, la più semplice possibile — TREND-FOLLOWING,
non mean-reversion, l'opposto di tutte le varianti VWAP testate finora in
questa sessione):
  "enters long positions when price trades above the VWAP and opens short
   positions when price moves below it"

Risultati originali (QQQ, 2018-01-02 → 2023-09-28, $25k iniziali):
  Return totale: +671%   MaxDD: -9.4%   Sharpe: 2.1
  (confronto buy&hold QQQ nello stesso periodo: +126%, MaxDD -37%, Sharpe 0.7)
  Con TQQQ (3x leva): +8,242% totale, +116%/anno medio

AVVERTENZA ESPLICITA riportata dalle fonti secondarie: gli stessi autori
definiscono questo "not a complete trading system yet" — nessuno
stop-loss, take-profit, o position sizing è specificato nel materiale
disponibile. Questa replica quindi implementa la regola letterale (nessun
filtro, nessuno stop) COME PRIMA VARIANTE, e aggiunge un safety-stop largo
come SECONDA VARIANTE (coerente con la buona pratica di risk management
usata in tutta questa sessione), per vedere quanto la mancanza di gestione
del rischio dichiarata dagli autori pesi sul risultato.

Replica su BTCUSDT 1H (a differenza del paper che usa QQQ — timeframe
intraday non specificato nelle fonti, orario di mercato azionario
USA — qui si usa la stessa architettura VWAP di sessione giornaliera
(reset UTC 00:00) già validata nelle altre 5 varianti mean-reversion di
questa sessione):

  segnale[i] = +1 se close[i] > VWAP[i]   (LONG)
             = -1 se close[i] < VWAP[i]   (SHORT)
  Entry/reversal: quando il segnale cambia rispetto alla barra precedente,
    si chiude la posizione corrente (se esiste) e se ne apre una nuova
    nella nuova direzione, ESECUZIONE CAUSALE all'apertura della barra
    successiva (mai al close della barra segnale)
  Force-close a fine giornata UTC (il VWAP si resetta il giorno dopo —
    stessa convenzione "day trading" delle altre varianti VWAP di sessione)

Due varianti testate:
  A) REGOLA LETTERALE: nessuno stop, nessun filtro — fedele al paper come
     descritto dalle fonti disponibili
  B) + SAFETY-STOP largo (3×ATR_1H, fisso) — per colmare la lacuna di risk
     management che gli stessi autori riconoscono nel materiale citato

Validazione (più rigorosa di quella nel paper originale, che non riporta
walk-forward né holdout): per-anno dal primo run, holdout 2025-2026
genuino, Monte Carlo i.i.d.+block, fee reali Bybit derivatives (taker,
dato che un sistema "flip on cross" richiede esecuzione immediata per non
perdere il segnale — non è un livello di prezzo noto in anticipo come nel
caso mean-reversion, quindi non è candidato naturale per ordini limit).
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

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
NOTIONAL_FRAC = 1.0          # per la variante A (nessuno stop -> notional fisso)
FEE_TAKER = 0.00055          # Bybit derivatives taker reale (vedi conversazione)
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
SAFETY_SL_ATR_MULT = 3.0     # variante B
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Replica paper: VWAP Trend-Following (Zarattini & Aziz, SSRN 2023) su BTCUSDT")
w(SEP)
w("\nPaper: 'Volume Weighted Average Price (VWAP): The Holy Grail for Day")
w("Trading Systems' — SSRN abstract_id=4631351")
w("Regola originale (QQQ 2018-2023): long sopra VWAP, short sotto VWAP.")
w("Risultati dichiarati: +671% ret, MaxDD -9.4%, Sharpe 2.1 (no walk-forward,")
w("no holdout dichiarati nelle fonti disponibili; PDF SSRN non fetchabile,")
w("metodologia ricostruita da Concretum Group — sito degli stessi autori —")
w("e Bear Bull Traders).")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 1H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H = len(df1h)
print(f"  1H: {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})  "
      f"(loaded in {time.time()-t0:.0f}s)")

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
OP = df1h["open"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
ATR = np.where(df1h["atr_14"].values > 0, df1h["atr_14"].values, 1.0)

# ── VWAP di sessione (reset giornaliero UTC), cumulativo, causale ─────────
dates = IDX1H.normalize().values
day_change = np.r_[True, dates[1:] != dates[:-1]]
day_id = np.cumsum(day_change) - 1

tp = (HI + LO + CL) / 3.0
tmp = pd.DataFrame({"day_id": day_id, "pv": tp * VOL, "vol": VOL})
g = tmp.groupby("day_id")
cum_pv = g["pv"].cumsum().values
cum_v = g["vol"].cumsum().values
bar_in_day = g.cumcount().values

VWAP = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
hour_arr = IDX1H.hour.values
bars_to_dayend = 23 - hour_arr

print(f"[VWAP] valid bars: {np.isfinite(VWAP).sum():,} / {N1H:,}")

# ── Segnale: sopra/sotto VWAP (letterale dal paper) ───────────────────────
signal = np.where(np.isfinite(VWAP), np.sign(CL - VWAP), 0).astype(int)


def build_trend_events():
    """Genera eventi di reversal: entry alla prima barra dopo un cambio di
    segnale, execution causale (open della barra successiva), force-close
    a fine giornata UTC."""
    evs = []
    prev_sig = 0
    for i in range(2, N1H - 1):
        if not np.isfinite(VWAP[i]) or bar_in_day[i] < 2:
            prev_sig = 0
            continue
        sig = signal[i]
        if sig != 0 and sig != prev_sig and bars_to_dayend[i] >= 1:
            entry_i = i + 1
            if entry_i < N1H:
                evs.append(dict(i=i, entry_i=entry_i, d=sig, ep=OP[entry_i]))
        if sig != 0:
            prev_sig = sig
    return evs


ALL_EVENTS = build_trend_events()
print(f"[EVENTS] {len(ALL_EVENTS)} reversal trade (long/short flip su VWAP)")


def run_bt(evs, use_safety_stop, slippage_pct=0.0):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_sl = n_time = 0
    for ev in evs:
        i, entry_i, d, ep = ev["i"], ev["entry_i"], ev["d"], ev["ep"]
        hold = min(bars_to_dayend[i], N1H - 1 - entry_i)
        if hold < 1:
            continue
        exit_i = entry_i + hold
        out = "time"; exit_price = OP[exit_i] if exit_i < N1H else CL[N1H - 1]

        if use_safety_stop:
            sl = ep - d * SAFETY_SL_ATR_MULT * ATR[i]
            for k in range(1, hold + 1):
                j = entry_i + k - 1
                if j >= N1H: break
                hk, lk = HI[j], LO[j]
                if d == 1 and lk <= sl: out = "sl"; exit_price = sl; break
                if d == -1 and hk >= sl: out = "sl"; exit_price = sl; break
            stop_dist = abs(ep - sl)
            if stop_dist <= 0: continue
            risk = INIT_CAP * RISK_PCT
            units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
        else:
            units = INIT_CAP * NOTIONAL_FRAC / ep

        if out == "sl": n_sl += 1
        else: n_time += 1

        notional = units * ep
        fill_ep = ep * (1 + d * slippage_pct)
        fill_xp = exit_price * (1 - d * slippage_pct)
        pnl = units * (fill_xp - fill_ep) * d - FEE_TAKER * 2 * notional
        cap += pnl
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl > 0)
        net_pnls.append(pnl)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_sl=n_sl, n_time=n_time)


MC_MAX_TRADES = 1_500   # subsample for MC when n is huge — run_monte_carlo's per-trade
                         # Python loop is O(n_sims x n_trades), doesn't scale to ~12k trades
MC_N_SIMS = 1_000

def _maybe_subsample(pnls, seed=42):
    if len(pnls) <= MC_MAX_TRADES:
        return pnls
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pnls), size=MC_MAX_TRADES, replace=False)
    return [pnls[i] for i in idx]

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    sub = _maybe_subsample(pnls)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": sub}), INIT_CAP, MC_N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=15):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    sub = _maybe_subsample(pnls)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": sub}), INIT_CAP, MC_N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


holdout_events = [e for e in ALL_EVENTS if IDX1H[e["entry_i"]] >= CUTOFF]
w(f"\n  Holdout genuino 2025-2026: {len(holdout_events)} trade")

for label, use_ss in [("A) Regola letterale (nessuno stop)", False),
                       ("B) + safety-stop 3xATR", True)]:
    w(f"\n{SEP}")
    w(f"{label}")
    w(SEP)

    res = run_bt(ALL_EVENTS, use_ss)
    n = res["n"]
    pval = st.binomtest(int(round(res["wr"] * n)), n, 0.5, alternative="greater").pvalue if n else 1.0
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL-SAMPLE: n={n}  wr={res['wr']:.1%} (p={pval:.4f} vs 50%)  "
      f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    if use_ss:
        w(f"    Exit: safety-SL={res['n_sl']}  time(fine giornata)={res['n_time']}")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    year_evs: dict[int, list] = {}
    for e in ALL_EVENTS:
        year_evs.setdefault(IDX1H[e["entry_i"]].year, []).append(e)
    for yr in sorted(year_evs):
        yevs = year_evs[yr]
        if len(yevs) < 5: continue
        yres = run_bt(yevs, use_ss)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    hres = run_bt(holdout_events, use_ss)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    w(f"\n  Slippage sensitivity:")
    w(f"    {'Slippage':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
    for bps in SLIPPAGE_TESTS_BPS:
        slip = bps / 10_000.0
        for scope_name, evs in [("full-sample", ALL_EVENTS), ("holdout", holdout_events)]:
            r = run_bt(evs, use_ss, slippage_pct=slip)
            m = mc_summary(r["net_pnls"])
            w(f"    {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
              f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vwap_trend_paper_replication.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Replica paper VWAP Trend-Following (Zarattini & Aziz, SSRN 2023) su BTCUSDT\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
