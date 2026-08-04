#!/usr/bin/env python3
"""
create_wyckoff_bybit_validation_report.py
============================================
Ri-validazione di Wyckoff Spring/Upthrust (1H, `src/strategy/wyckoff.py`,
già documentata in `docs/wyckoff_strategy_technical_spec.md`) con la
pipeline rigorosa di QUESTA sessione — la spec esistente usava
`src/strategy/engine.py` con FEE=0.04%/lato (stile Binance spot), non le
frizioni reali Bybit derivatives (taker 0.055% + slippage 0.015%/lato =
0.14% round-trip) usate per ogni altra strategia validata in sessione.
Nessun'altra assunzione cambia: stesso segnale causale (spring/upthrust
su range accumulazione/distribuzione, entry alla barra successiva),
stesso motore di rilevamento (`build_wyckoff_signals`).

Requisito nuovo: sostituire le 2 strategie a timeframe 1D (Carver
Breakout pool, TSMOM-sign pool) — escluse perché il nuovo vincolo
richiede SOLO timeframe intraday. Wyckoff è 1H, intraday per
costruzione: è un candidato naturale da ri-controllare prima di
sviluppare segnali nuovi da zero.

Costruzione (causale, invariata da `src/strategy/wyckoff.py`):
  1. Range: ADX<25 AND ampiezza range<10%, boundaries shiftate di 1 barra.
  2. Bias volumetrico: OBV vs EMA21 (accumulation/distribution).
  3. Spring/Upthrust: penetrazione + rientro nel range su volume alto.
  4. Entry: apertura della barra successiva alla conferma (già causale
     nel segnale — `signal[i]` implica info nota alla chiusura di i-1).
  5. Stop: ATR_SL × ATR_14 alla barra segnale; Target: RR × rischio,
     griglia RR con DSR family (stesso schema di `ict_fade_standalone`).
  6. Uscita a tempo: MAX_HOLD_BARS se né stop né target vengono toccati.

Frizioni Bybit obbligatorie. Validazione: per-anno, holdout 2025-2026
genuino, Monte Carlo i.i.d.+block, DSR family sul grid RR, slippage-stress.
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
from src.strategy.wyckoff import build_wyckoff_signals
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

ATR_SL = 1.5          # stop = ATR_SL * ATR_14 alla barra segnale
MAX_HOLD_BARS = 48    # 2 giorni su 1H
RR_GRID = [1.5, 2.0, 3.0]
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Wyckoff Spring/Upthrust (1H) — Ri-validazione Bybit-fee-aware")
w(SEP)
w("\nStessa costruzione causale di src/strategy/wyckoff.py, ma con frizioni")
w("Bybit reali (0.055%+0.015%/lato) al posto del FEE=0.04% dell'engine")
w("generico — pipeline identica alle altre strategie validate in sessione.")

t0 = time.time()
print("\n[DATA] Loading 1H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"].copy())
IDX = df1h.index
N = len(df1h)
OP = df1h["open"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
CL = df1h["close"].values.astype(float)
ATR = df1h["atr_14"].values.astype(float)
print(f"  1H: {N:,} bars  ({IDX[0]} → {IDX[-1]})  (loaded in {time.time()-t0:.0f}s)")

print("[SIGNAL] Building Wyckoff spring/upthrust signals …")
sig_df = build_wyckoff_signals(df1h)
signal = sig_df["signal"].values

n_spring = int((signal == 1).sum())
n_upthrust = int((signal == -1).sum())
print(f"  {n_spring} spring (long) + {n_upthrust} upthrust (short) trigger grezzi")


def build_events():
    evs = []
    last_exit = -1
    for i in range(2, N - 1):
        if signal[i] == 0: continue
        if i <= last_exit: continue
        if np.isnan(ATR[i]) or ATR[i] <= 0: continue
        d = int(signal[i])
        ep = OP[i]
        sl = ep - d * ATR_SL * ATR[i]
        risk = abs(ep - sl)
        if risk <= 0: continue
        hold = min(MAX_HOLD_BARS, N - 1 - i)
        if hold < 1: continue
        evs.append(dict(entry_i=i, d=d, ep=ep, sl=sl, risk=risk, hold=hold))
        last_exit = i + hold
    return evs


ALL_EVENTS = build_events()
print(f"[EVENTS] {len(ALL_EVENTS)} trade Wyckoff (sequenziali, non sovrapposti)")


def run_bt(evs, rr, extra_slippage_pct=0.0):
    slip_pct = SLIPPAGE_BASE + extra_slippage_pct
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0
    for ev in evs:
        d, ep, sl, risk, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"]
        target = ep + d * rr * risk
        out = "time"; exit_price = None
        for k in range(hold):
            j = entry_i + k
            if j >= N: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl: out = "sl"; exit_price = sl; break
            if hit_tp: out = "tp"; exit_price = target; break
        if exit_price is None:
            exit_price = CL[min(entry_i + hold, N - 1)]
        if out == "tp": n_tp += 1
        elif out == "sl": n_sl += 1
        else: n_time += 1
        if risk <= 0: continue
        r = INIT_CAP * RISK_PCT
        units = min(r / risk, MAX_LEV * INIT_CAP / ep)
        fill_ep = ep * (1 + d * slip_pct)
        fill_xp = exit_price * (1 - d * slip_pct)
        pnl = units * (fill_xp - fill_ep) * d - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        cap += pnl
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl > 0)
        net_pnls.append(pnl)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_tp=n_tp, n_sl=n_sl, n_time=n_time)


def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=10):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


holdout_events = [e for e in ALL_EVENTS if IDX[e["entry_i"]] >= CUTOFF]
w(f"\n  Holdout genuino 2025-2026: {len(holdout_events)} trade")

dsr_full = []
dsr_holdout = []
for rr in RR_GRID:
    w(f"\n{SEP}")
    w(f"RR = {rr:.1f}")
    w(SEP)

    res = run_bt(ALL_EVENTS, rr)
    n = res["n"]
    pval = st.binomtest(int(round(res["wr"] * n)), n, 0.5, alternative="greater").pvalue if n else 1.0
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL-SAMPLE: n={n}  wr={res['wr']:.1%} (p={pval:.4f} vs 50%)  "
      f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    year_evs: dict[int, list] = {}
    for e in ALL_EVENTS:
        year_evs.setdefault(IDX[e["entry_i"]].year, []).append(e)
    for yr in sorted(year_evs):
        yevs = year_evs[yr]
        if len(yevs) < 5: continue
        yres = run_bt(yevs, rr)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    hres = run_bt(holdout_events, rr)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    dsr_full.append(dict(rr=rr, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_holdout.append(dict(rr=rr, net_pnls=hres["net_pnls"], ret=hres["ret"]))

w(f"\n{SEP}")
w(f"DSR — famiglia N={len(RR_GRID)} (griglia RR)")
w(SEP)
deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
w(f"\n  Full-sample:")
w(f"  {'RR':>6}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_full:
    w(f"  {r['rr']:>5.1f}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
w(f"\n  Holdout 2025-2026:")
w(f"  {'RR':>6}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_holdout:
    w(f"  {r['rr']:>5.1f}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

holdout_dsr_by_rr = {r["rr"]: r["dsr"] for r in dsr_holdout}
holdout_sharpe_by_rr = {r["rr"]: r["sharpe_hat"] for r in dsr_holdout}
best_rr = max(dsr_full, key=lambda r: (r["dsr"], holdout_dsr_by_rr[r["rr"]], holdout_sharpe_by_rr[r["rr"]]))["rr"]
w(f"\n{SEP}")
w(f"Slippage-stress — RR migliore per DSR full+holdout = {best_rr:.1f}")
w(SEP)
w(f"\n  {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_STRESS_BPS:
    extra = bps / 10_000.0
    for scope_name, evs in [("full-sample", ALL_EVENTS), ("holdout", holdout_events)]:
        r = run_bt(evs, best_rr, extra_slippage_pct=extra)
        w(f"  {bps:>9}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  {r['wr']:>5.1%}  "
          f"{mc_summary(r['net_pnls'])['p_profit']:>7.3f}  {mc_summary(r['net_pnls'])['p_ruin']:>7.3f}")

w(f"\n{SEP}")
w("[DONE]")
w(SEP)

Path("reports").mkdir(exist_ok=True)
with open("reports/wyckoff_bybit_validation.md", "w") as f:
    f.write("# Wyckoff Spring/Upthrust (1H) — Ri-validazione Bybit-fee-aware\n\n")
    f.write("```\n" + "\n".join(report_lines) + "\n```\n")
print(f"\n[DONE] reports/wyckoff_bybit_validation.md   (total runtime {time.time()-t0:.0f}s)")
