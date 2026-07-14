#!/usr/bin/env python3
"""
create_asia_sweep_mss_report.py
==================================
Nuova strategia proposta dall'utente: liquidity sweep del range Asia +
market structure shift + entry su FVG/Order Block, target all'estremo
opposto del range, sottoposta alla pipeline di validazione rigorosa.

Sequenza (tutto su barre 1H, causale):
  1. RANGE ASIA: sessione 00:00-07:00 UTC (stessa convenzione già usata nel
     modello Power-of-3 di create_ict_suite_report.py). asia_hi/asia_lo.
  2. SWEEP: prima barra (07:00 UTC -> fine giornata) che spazza un estremo
     e chiude di nuovo dentro il range:
       bullish sweep: low < asia_lo  AND  close > asia_lo  -> bias LONG
       bearish sweep: high > asia_hi AND  close < asia_hi  -> bias SHORT
     (se una barra spazza entrambi gli estremi: ambiguo, si scarta il giorno)
  3. MARKET STRUCTURE SHIFT (MSS): rottura in chiusura dell'ultimo pivot
     fractal confermato (left=right=3, src/strategy/mtf_swing.py) nella
     direzione del sweep — per un bias LONG, la prima barra che chiude sopra
     l'ultimo pivot high confermato al momento dello sweep; simmetrico per
     SHORT. Il pivot è per costruzione già noto/confermato al momento dello
     sweep (nessun lookahead).
  4. ENTRY su FVG o Order Block: dopo l'MSS, si cerca — sulla gamba
     d'impulso dallo sweep all'MSS (+ una piccola estensione) — un Fair
     Value Gap (gap a 3 candele, stessa definizione di MODEL 1 in
     create_ict_suite_report.py) o un Order Block (ultima candela contraria
     prima di un impulso ≥ swing_atr×ATR, stessa definizione di MODEL 2).
     L'entry è il primo ritorno di prezzo nella zona FVG/OB, MAI prima
     della barra di conferma MSS. Si prende il trigger che scatta per primo
     tra FVG e OB.
  5. TARGET = estremo opposto del range Asia originale (fisso).
  6. STOP dinamico per mantenere RR fisso rispetto al target:
     stop_dist = target_dist / RR, testato per RR = 1.0, 2.0, 3.0 (SL
     "1:1, 1:2, 1:3 rispetto a TP" — richiesta esplicita dell'utente).

Un solo trade al giorno (la prima sequenza completa valida). Sizing a
rischio-dollaro fisso, fee 0.04%/lato — stessa convenzione della pipeline.

Validazione: per-anno dal primo run, holdout 2025-2026 genuino, Monte Carlo
i.i.d.+block, DSR family N=3 sui 3 RR, slippage sensitivity sul RR migliore.
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
from src.strategy.mtf_swing import causal_trend_state
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0004
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

ASIA_START_H, ASIA_END_H = 0, 7           # UTC, stessa convenzione del PO3 esistente
PIVOT_LR = 3                                # left=right per il pivot di riferimento MSS
MSS_MAX_BARS = 24                           # entro 24h dallo sweep deve arrivare l'MSS
FVG_OB_SEARCH_EXT = 6                       # barre oltre l'MSS in cui può formarsi FVG/OB
FVG_OB_MAX_AGE = 24                         # barre max di attesa per il retest della zona
MIN_ZONE_ATR_FRAC = 0.05                    # filtro dimensione minima zona FVG/OB
SWING_ATR_OB = 1.5                          # impulso minimo per un Order Block valido
MAX_HOLD = 48                               # barre 1H dall'entry (2 giorni)
RR_TESTS = [1.0, 2.0, 3.0]
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Asia Range Sweep + MSS + FVG/OB Entry — Validation Pipeline")
w(SEP)

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

OP = df1h["open"].values.astype(float)
CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
ATR = np.where(df1h["atr_14"].values > 0, df1h["atr_14"].values, 1.0)
hour_arr = IDX1H.hour.values
date_arr = IDX1H.normalize()

trend = causal_trend_state(CL, HI, LO, PIVOT_LR, PIVOT_LR)
LAST_PH = trend["last_pivot_high"].values
LAST_PL = trend["last_pivot_low"].values


# ── FVG / Order Block detectors (stessa definizione di create_ict_suite_report.py) ─
def find_fvg_entry(direction, leg_start, leg_end, mss_i, cutoff_i):
    """Cerca FVG con candela centrale in [leg_start, leg_end]; ritorna il
    primo trigger di entry (>= mss_i) trovato in ordine temporale, o None."""
    best = None
    for i in range(max(leg_start, 1), min(leg_end, N1H - 2)):
        atr = ATR[i]
        if atr <= 0: continue
        if direction == 1 and HI[i - 1] < LO[i + 1]:
            fbot, ftop = HI[i - 1], LO[i + 1]
            fsz = ftop - fbot
            if fsz < MIN_ZONE_ATR_FRAC * atr: continue
            start_j = max(i + 2, mss_i + 1)
            for j in range(start_j, min(i + 2 + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] < fbot: break
                if LO[j] <= ftop:
                    trig = (j, min(CL[j], ftop))
                    if best is None or trig[0] < best[0]: best = trig
                    break
        if direction == -1 and LO[i - 1] > HI[i + 1]:
            ftop, fbot = LO[i - 1], HI[i + 1]
            fsz = ftop - fbot
            if fsz < MIN_ZONE_ATR_FRAC * atr: continue
            start_j = max(i + 2, mss_i + 1)
            for j in range(start_j, min(i + 2 + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] > ftop: break
                if HI[j] >= fbot:
                    trig = (j, max(CL[j], fbot))
                    if best is None or trig[0] < best[0]: best = trig
                    break
    return best


def find_ob_entry(direction, leg_start, leg_end, mss_i, cutoff_i):
    best = None
    for i in range(max(leg_start, 0), min(leg_end, N1H - 8)):
        atr = ATR[i]
        if atr <= 0: continue
        if direction == 1 and OP[i] > CL[i]:                 # candela ribassista = OB rialzista
            ob_hi, ob_lo = HI[i], LO[i]
            if ob_hi - ob_lo < 1e-9: continue
            swing_end = -1; swing_start = CL[i]
            for s in range(i + 1, min(i + 8, N1H)):
                if CL[s] - swing_start >= SWING_ATR_OB * atr:
                    swing_end = s; break
            if swing_end < 0: continue
            start_j = max(swing_end + 1, mss_i + 1)
            for j in range(start_j, min(i + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] < ob_lo: break
                if LO[j] <= ob_hi and HI[j] >= ob_lo:
                    trig = (j, max(CL[j], ob_lo))
                    if best is None or trig[0] < best[0]: best = trig
                    break
        if direction == -1 and OP[i] < CL[i]:                 # candela rialzista = OB ribassista
            ob_hi, ob_lo = HI[i], LO[i]
            if ob_hi - ob_lo < 1e-9: continue
            swing_end = -1; swing_start = CL[i]
            for s in range(i + 1, min(i + 8, N1H)):
                if swing_start - CL[s] >= SWING_ATR_OB * atr:
                    swing_end = s; break
            if swing_end < 0: continue
            start_j = max(swing_end + 1, mss_i + 1)
            for j in range(start_j, min(i + FVG_OB_MAX_AGE, cutoff_i)):
                if CL[j] > ob_hi: break
                if HI[j] >= ob_lo and LO[j] <= ob_hi:
                    trig = (j, min(CL[j], ob_hi))
                    if best is None or trig[0] < best[0]: best = trig
                    break
    return best


# ── Event construction ──────────────────────────────────────────────────
print("\n[EVENTS] Scanning Asia range -> sweep -> MSS -> FVG/OB entry, day by day …")
t1 = time.time()

all_days = pd.date_range(IDX1H[0].normalize(), IDX1H[-1].normalize(), freq="D")
events = []
n_no_range = n_no_sweep = n_no_mss = n_no_entry = n_bad_target = 0

for day in all_days:
    asia_mask = (date_arr == day) & (hour_arr >= ASIA_START_H) & (hour_arr < ASIA_END_H)
    asia_idx = np.where(asia_mask)[0]
    if len(asia_idx) < 4:
        n_no_range += 1; continue
    asia_hi = float(HI[asia_idx].max())
    asia_lo = float(LO[asia_idx].min())
    if asia_hi <= asia_lo:
        n_no_range += 1; continue

    post_mask = (date_arr == day) & (hour_arr >= ASIA_END_H)
    post_idx = np.where(post_mask)[0]
    if len(post_idx) == 0:
        n_no_sweep += 1; continue

    sweep_i = -1; direction = 0
    for k in post_idx:
        bull = (LO[k] < asia_lo) and (CL[k] > asia_lo)
        bear = (HI[k] > asia_hi) and (CL[k] < asia_hi)
        if bull and bear:
            break   # ambiguo
        if bull:
            sweep_i = k; direction = 1; break
        if bear:
            sweep_i = k; direction = -1; break
    if sweep_i < 0:
        n_no_sweep += 1; continue

    ref_level = LAST_PH[sweep_i] if direction == 1 else LAST_PL[sweep_i]
    if np.isnan(ref_level):
        n_no_mss += 1; continue

    mss_i = -1
    for j in range(sweep_i + 1, min(sweep_i + 1 + MSS_MAX_BARS, N1H)):
        if direction == 1 and CL[j] > ref_level:
            mss_i = j; break
        if direction == -1 and CL[j] < ref_level:
            mss_i = j; break
    if mss_i < 0:
        n_no_mss += 1; continue

    leg_start = sweep_i
    leg_end = min(mss_i + FVG_OB_SEARCH_EXT, N1H - 2)
    cutoff_i = min(mss_i + 1 + FVG_OB_MAX_AGE, N1H)

    trig_fvg = find_fvg_entry(direction, leg_start, leg_end, mss_i, cutoff_i)
    trig_ob = find_ob_entry(direction, leg_start, leg_end, mss_i, cutoff_i)
    candidates = [t for t in (trig_fvg, trig_ob) if t is not None]
    if not candidates:
        n_no_entry += 1; continue
    entry_i, entry_price = min(candidates, key=lambda t: t[0])

    target = asia_hi if direction == 1 else asia_lo
    if (direction == 1 and target <= entry_price) or (direction == -1 and target >= entry_price):
        n_bad_target += 1; continue
    target_dist = abs(target - entry_price)
    if target_dist < 0.05 * ATR[entry_i]:
        n_bad_target += 1; continue

    events.append(dict(day=day, direction=direction, entry_i=entry_i, entry_price=entry_price,
                        target=target, target_dist=target_dist, sweep_i=sweep_i, mss_i=mss_i))

print(f"  done in {time.time()-t1:.0f}s")
w(f"\n[EVENTS] {len(all_days)} giorni scansionati -> {len(events)} trade candidati")
w(f"  scartati: no-range={n_no_range}  no-sweep={n_no_sweep}  no-mss={n_no_mss}  "
  f"no-entry(fvg/ob)={n_no_entry}  bad-target={n_bad_target}")


# ── Backtest ─────────────────────────────────────────────────────────────
def run_bt(evs, rr, slippage_pct=0.0):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0
    for ev in evs:
        d = ev["direction"]; ep = ev["entry_price"]; target = ev["target"]
        stop_dist = ev["target_dist"] / rr
        sl = ep - d * stop_dist
        i0 = ev["entry_i"]
        out = "time"; exit_price = None
        for k in range(1, MAX_HOLD + 1):
            j = i0 + k
            if j >= N1H: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl:
                out = "sl"; exit_price = sl; break
            if hit_tp:
                out = "tp"; exit_price = target; break
        if exit_price is None:
            j = min(i0 + MAX_HOLD, N1H - 1)
            exit_price = CL[j]
        if out == "tp": n_tp += 1
        elif out == "sl": n_sl += 1
        else: n_time += 1

        if stop_dist <= 0: continue
        risk = INIT_CAP * RISK_PCT
        units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
        notional = units * ep
        fill_ep = ep * (1 + d * slippage_pct)
        fill_xp = exit_price * (1 - d * slippage_pct)
        pnl = units * (fill_xp - fill_ep) * d - FEE * 2 * notional
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


holdout_events = [e for e in events if IDX1H[e["entry_i"]] >= CUTOFF]
w(f"\n  Holdout genuino 2025-2026: {len(holdout_events)} trade")

dsr_full = []
dsr_holdout = []

for rr in RR_TESTS:
    w(f"\n{SEP}")
    w(f"RR = {rr:.1f} : 1   (stop = target_dist / {rr:.1f})")
    w(SEP)

    res = run_bt(events, rr)
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    be_wr = 1.0 / (1.0 + rr)
    pval = (st.binomtest(int(round(res["wr"] * res["n"])), res["n"], be_wr,
                          alternative="greater").pvalue if res["n"] else 1.0)

    w(f"\n  FULL-SAMPLE: n={res['n']}  wr={res['wr']:.1%} (BE_teorico={be_wr:.1%}, p={pval:.4f})  "
      f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    year_evs: dict[int, list] = {}
    for e in events:
        year_evs.setdefault(e["day"].year, []).append(e)
    for yr in sorted(year_evs):
        yevs = year_evs[yr]
        if len(yevs) < 5: continue
        yres = run_bt(yevs, rr)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    hres = run_bt(holdout_events, rr)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    dsr_full.append(dict(rr=rr, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_holdout.append(dict(rr=rr, net_pnls=hres["net_pnls"], ret=hres["ret"]))

# ── DSR family (3 RR) ────────────────────────────────────────────────────
w(f"\n{SEP}")
w("DSR — famiglia N=3 (RR 1:1, 2:1, 3:1)")
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

# ── Slippage sensitivity on the best RR ─────────────────────────────────
best_rr = max(dsr_full, key=lambda r: r["dsr"])["rr"]
w(f"\n{SEP}")
w(f"Slippage sensitivity — RR migliore per DSR = {best_rr:.1f}:1")
w(SEP)
w(f"\n  {'Slippage':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MDD%':>7}  "
  f"{'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_TESTS_BPS:
    slip = bps / 10_000.0
    for scope_name, evs in [("full-sample", events), ("holdout", holdout_events)]:
        r = run_bt(evs, best_rr, slippage_pct=slip)
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {r['mdd']:>6.1f}%  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/asia_sweep_mss.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Asia Range Sweep + MSS + FVG/OB Entry — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
