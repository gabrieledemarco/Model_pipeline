#!/usr/bin/env python3
"""
create_vwap_bounce_report.py
===============================
Variante "bounce" della replica del paper VWAP trend-following (Zarattini
& Aziz — vedi create_vwap_trend_paper_replication_report.py), che aveva
fallito nettamente su BTCUSDT per overtrading: la regola letterale
("ribalta posizione ad ogni attraversamento del VWAP") produceva ~4,6
trade/giorno, puro whipsaw, negativo in ogni anno 2020-2026.

Diagnosi (confermata dall'utente): la regola corretta non è "entra ad
OGNI incrocio", ma "entra quando il prezzo RIMBALZA sul VWAP" — un trade
di continuazione del trend, non un flip continuo:

  1. CONTESTO DI TREND (causale, prima del tocco): il prezzo deve essere
     stato chiaramente da un lato del VWAP qualche barra prima —
     close[i-LOOKBACK] a distanza >= MIN_SEP_ATR × ATR dal VWAP di allora.
     bias = +1 (rialzista) se sopra, -1 (ribassista) se sotto.
  2. TOUCH: la barra i tocca il VWAP intrabar (low[i] <= VWAP[i] <= high[i])
     — il prezzo è tornato a testare il livello.
  3. REJECTION: la barra i chiude di nuovo dalla parte del bias (non lo
     attraversa in modo decisivo) — close[i] sul lato del bias rispetto
     a VWAP[i]. Questo è il "rimbalzo".
  4. ENTRY causale: open della barra i+1, direzione = bias.

Exit: stop ATR fisso (il rimbalzo fallisce se il prezzo rompe comunque il
VWAP in modo deciso) + force-close a fine giornata UTC (stessa convenzione
"sessione VWAP giornaliera" di tutta questa famiglia di strategie).
Cooldown: nessuna nuova entry finché la posizione corrente non è chiusa
(trade sequenziali, non sovrapposti).

Un solo set di parametri testato letteralmente (nessun IS-scan/grid — è
un primo test dell'idea, non un'ottimizzazione), fee taker reali Bybit
(0.055%/lato — l'entry richiede reazione alla chiusura della barra di
rejection, non è un livello noto in anticipo come nel caso mean-reversion,
quindi non è candidato naturale per ordini limit).

Validazione: per-anno dal primo run, holdout 2025-2026 genuino, Monte
Carlo i.i.d.+block, slippage sensitivity.
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
FEE_TAKER = 0.00055          # Bybit derivatives taker reale
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

LOOKBACK = 5                 # barre indietro per stabilire il contesto di trend
MIN_SEP_ATR = 0.5            # separazione minima dal VWAP (in ATR) per un trend "vero"
SL_ATR_MULT = 1.5            # stop fisso in ATR
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("VWAP Bounce (rimbalzo in direzione del trend) — Validation Pipeline")
w(SEP)
w("\nDiagnosi del fallimento della replica letterale (flip ad ogni cross):")
w("overtrading, ~4.6 trade/giorno, negativo ogni anno. Questa variante entra")
w("solo quando il prezzo RITORNA sul VWAP e RIMBALZA nella direzione del")
w("trend già stabilito, non ad ogni attraversamento.")

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


def build_bounce_events():
    evs = []
    last_exit = -1
    for i in range(LOOKBACK + 2, N1H - 1):
        if i <= last_exit:
            continue
        if not np.isfinite(VWAP[i]) or bar_in_day[i] < 2 or ATR[i] <= 0:
            continue

        ref = i - LOOKBACK
        if not np.isfinite(VWAP[ref]) or bar_in_day[ref] < 2 or ATR[ref] <= 0:
            continue
        sep = CL[ref] - VWAP[ref]
        if abs(sep) < MIN_SEP_ATR * ATR[ref]:
            continue
        bias = 1 if sep > 0 else -1

        touch = LO[i] <= VWAP[i] <= HI[i]
        if not touch:
            continue

        rejection = (bias == 1 and CL[i] > VWAP[i]) or (bias == -1 and CL[i] < VWAP[i])
        if not rejection:
            continue

        if bars_to_dayend[i] < 1:
            continue

        entry_i = i + 1
        ep = OP[entry_i]
        sl = ep - bias * SL_ATR_MULT * ATR[i]
        hold = min(bars_to_dayend[i], N1H - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(i=i, entry_i=entry_i, d=bias, ep=ep, sl=sl, hold=hold))
        last_exit = entry_i + hold
    return evs


ALL_EVENTS = build_bounce_events()
print(f"[EVENTS] {len(ALL_EVENTS)} bounce trade (sequenziali, non sovrapposti)")


def run_bt(evs, slippage_pct=0.0):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_sl = n_time = 0
    for ev in evs:
        d, ep, sl, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["entry_i"], ev["hold"]
        out = "time"; exit_price = OP[min(entry_i + hold, N1H - 1)]
        for k in range(hold):
            j = entry_i + k
            if j >= N1H: break
            hk, lk = HI[j], LO[j]
            if d == 1 and lk <= sl: out = "sl"; exit_price = sl; break
            if d == -1 and hk >= sl: out = "sl"; exit_price = sl; break
        if out == "sl": n_sl += 1
        else: n_time += 1

        stop_dist = abs(ep - sl)
        if stop_dist <= 0: continue
        risk = INIT_CAP * RISK_PCT
        units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
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


def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=15):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


holdout_events = [e for e in ALL_EVENTS if IDX1H[e["entry_i"]] >= CUTOFF]

res = run_bt(ALL_EVENTS)
n = res["n"]
pval = st.binomtest(int(round(res["wr"] * n)), n, 0.5, alternative="greater").pvalue if n else 1.0
mc = mc_summary(res["net_pnls"])
mc_blk = mc_block_summary(res["net_pnls"])
w(f"\n{SEP}")
w("RISULTATI")
w(SEP)
w(f"\n  FULL-SAMPLE: n={n}  wr={res['wr']:.1%} (p={pval:.4f} vs 50%)  "
  f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
w(f"    Exit: SL={res['n_sl']}  time(fine giornata)={res['n_time']}")
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
    yres = run_bt(yevs)
    w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

hres = run_bt(holdout_events)
hmc = mc_summary(hres["net_pnls"])
hmc_blk = mc_block_summary(hres["net_pnls"])
w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
  f"mdd={hres['mdd']:.1f}%")
w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

w(f"\n{SEP}")
w("Slippage sensitivity")
w(SEP)
w(f"\n  {'Slippage':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_TESTS_BPS:
    slip = bps / 10_000.0
    for scope_name, evs in [("full-sample", ALL_EVENTS), ("holdout", holdout_events)]:
        r = run_bt(evs, slippage_pct=slip)
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vwap_bounce.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# VWAP Bounce (rimbalzo in direzione del trend) — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
