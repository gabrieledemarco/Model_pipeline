#!/usr/bin/env python3
"""
create_vwap_bounce_v2_report.py
===================================
v2 — ottimizzazione dello stop della strategia VWAP bounce (v1: entry
quando il prezzo rimbalza sul VWAP in direzione del trend già stabilito).

v1 con stop fisso a 1.5×ATR aveva fallito (WR 35.6%, peggio della regola
letterale) con una diagnostica chiara: il 50.5% delle uscite erano via
stop-loss, con un'uscita a tempo aperta fino a fine giornata (molte ore) a
dare al rumore ampio tempo per toccare uno stop relativamente vicino —
stessa asimmetria di first-passage-time vista più volte in questa
sessione. Non era chiaro se il problema fosse la tesi del rimbalzo o la
costruzione dello stop.

v2 testa una griglia di moltiplicatori ATR per lo stop — [1.0, 1.5, 2.0,
2.5, 3.0, 4.0] — con backtest COMPLETO per ciascuno (non un IS-scan
causale per finestra: qui è un unico ruleset fisso su tutta la storia,
quindi si tratta di N=6 backtest completi comparabili, non di selezione
causale per-finestra). Per correggere la selection bias del "migliore
della griglia" si applica DSR family N=6 — stessa disciplina già usata
per i grid RR di NY-ORB-VP e Asia-sweep.

Entry logic (identica a v1, indipendente dallo stop — quindi la lista di
eventi è costruita UNA SOLA VOLTA e riusata per tutti i moltiplicatori):
  1. CONTESTO DI TREND (causale, prima del tocco): close[i-LOOKBACK] a
     distanza >= MIN_SEP_ATR × ATR dal VWAP di allora. bias = +1/-1.
  2. TOUCH: low[i] <= VWAP[i] <= high[i].
  3. REJECTION: close[i] sul lato del bias rispetto a VWAP[i].
  4. ENTRY causale: open della barra i+1, direzione = bias.

Exit: stop ATR (moltiplicatore scansionato) + force-close a fine giornata
UTC. Trade sequenziali, non sovrapposti. Fee taker reali Bybit.

Validazione: per-anno dal primo run, holdout 2025-2026 genuino, Monte
Carlo i.i.d.+block, DSR family N=6, slippage sensitivity sul migliore.
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
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

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
SL_ATR_MULT_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("VWAP Bounce v2 — ottimizzazione dello stop ATR — Validation Pipeline")
w(SEP)
w("\nv1 (stop fisso 1.5xATR) aveva WR=35.6%, peggio della regola letterale,")
w("con 50.5% delle uscite via stop-loss — sospetto di first-passage-time.")
w("v2 scansiona il moltiplicatore ATR dello stop [1.0-4.0] per isolare se")
w("il problema sia la tesi del rimbalzo o la costruzione dello stop.")

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
        hold = min(bars_to_dayend[i], N1H - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(i=i, entry_i=entry_i, d=bias, ep=ep, atr=ATR[i], hold=hold))
        last_exit = entry_i + hold
    return evs


ALL_EVENTS = build_bounce_events()
print(f"[EVENTS] {len(ALL_EVENTS)} bounce trade (sequenziali, non sovrapposti)")


def run_bt(evs, sl_atr_mult, slippage_pct=0.0):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_sl = n_time = 0
    for ev in evs:
        d, ep, entry_i, hold = ev["d"], ev["ep"], ev["entry_i"], ev["hold"]
        sl = ep - d * sl_atr_mult * ev["atr"]
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
w(f"\n  Holdout genuino 2025-2026: {len(holdout_events)} trade")

dsr_full = []
dsr_holdout = []

for sl_mult in SL_ATR_MULT_GRID:
    w(f"\n{SEP}")
    w(f"SL = {sl_mult:.2f} × ATR")
    w(SEP)

    res = run_bt(ALL_EVENTS, sl_mult)
    n = res["n"]
    pval = st.binomtest(int(round(res["wr"] * n)), n, 0.5, alternative="greater").pvalue if n else 1.0
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
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
        yres = run_bt(yevs, sl_mult)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    hres = run_bt(holdout_events, sl_mult)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    dsr_full.append(dict(sl_mult=sl_mult, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_holdout.append(dict(sl_mult=sl_mult, net_pnls=hres["net_pnls"], ret=hres["ret"]))

# ── DSR family (6 moltiplicatori ATR) ─────────────────────────────────────
w(f"\n{SEP}")
w("DSR — famiglia N=6 (moltiplicatori ATR dello stop)")
w(SEP)
deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
w(f"\n  Full-sample:")
w(f"  {'SL xATR':>8}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_full:
    w(f"  {r['sl_mult']:>7.2f}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
w(f"\n  Holdout 2025-2026:")
w(f"  {'SL xATR':>8}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_holdout:
    w(f"  {r['sl_mult']:>7.2f}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

# ── Slippage sensitivity sul migliore per DSR ─────────────────────────────
best_sl = max(dsr_full, key=lambda r: r["dsr"])["sl_mult"]
w(f"\n{SEP}")
w(f"Slippage sensitivity — SL migliore per DSR = {best_sl:.2f}×ATR")
w(SEP)
w(f"\n  {'Slippage':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_TESTS_BPS:
    slip = bps / 10_000.0
    for scope_name, evs in [("full-sample", ALL_EVENTS), ("holdout", holdout_events)]:
        r = run_bt(evs, best_sl, slippage_pct=slip)
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vwap_bounce_v2.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# VWAP Bounce v2 — ottimizzazione dello stop ATR — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
