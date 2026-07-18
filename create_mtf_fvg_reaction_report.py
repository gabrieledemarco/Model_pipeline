#!/usr/bin/env python3
"""
create_mtf_fvg_reaction_report.py
====================================
Replica letterale della regola ("Here's how you win", claim social-media
non attribuita, no track record verificabile allegato):

  1. Aspetta che il prezzo entri in un FVG 4H
  2. Aspetta una reazione AGGRESSIVA fuori dal FVG 4H, sul 15m
  3. Aspetta che si formi un NUOVO FVG sul 15m (dentro/dopo la reazione)
  4. Entra su quel nuovo FVG
  5. Target 2 o 3 RR

Implementazione causale, stato-macchina sequenziale per ogni FVG 4H:

  FVG (definizione ICT standard, 3 candele): bullish se HI[i-1]<LO[i+1],
  bearish se LO[i-1]>HI[i+1] — riusa la stessa definizione di
  `create_ict_suite_report.py`. Filtro dimensione minima:
  FVG_MIN_ATR_FRAC × ATR del rispettivo timeframe.

  STEP 1 (touch, 15m dentro l'FVG 4H): dal momento in cui l'FVG 4H è
  confermato (chiusura della 3a candela) fino a invalidazione (chiusura
  4H oltre il bordo lontano) o staleness massima, cerca sul 15m la prima
  barra che tocca la zona [fbot,ftop].

  STEP 2 (reazione aggressiva, 15m): dal touch, entro
  MAX_TOUCH_TO_REACT_BARS barre 15m, cerca una candela "displacement"
  nella direzione di rimbalzo attesa (range >= REACT_DISPLACEMENT_MULT ×
  range medio delle 3 barre precedenti — stessa soglia grounded-in-ricerca
  usata per l'MSS in Asia-sweep v4 di questa sessione).

  STEP 3 (nuovo FVG 15m): dalla candela di reazione, entro
  MAX_REACT_TO_NEWFVG_BARS barre, cerca un nuovo FVG 15m nella stessa
  direzione della reazione.

  STEP 4 (entry): dalla formazione del nuovo FVG, entro
  MAX_NEWFVG_TO_RETEST_BARS barre, cerca il primo ritorno del prezzo
  dentro la zona del nuovo FVG — entry lì (ordine limite, fill al bordo
  della zona, stessa convenzione di `create_ict_suite_report.py`).

  STOP: oltre l'estremo (min/max) del leg di reazione (dal touch alla
  formazione del nuovo FVG) — l'origine strutturale del movimento.
  TARGET: RR × rischio, RR testato in {2.0, 3.0} come dichiarato dalla
  fonte, con correzione DSR (famiglia N=2).

Un solo trade per FVG 4H (il primo pattern completo trovato); niente
overlap. Fee/slippage Bybit obbligatorie di questa sessione (taker
0.055% + slippage 0.015%/lato = 0.14% round-trip), nessun look-ahead
(segnale su barra chiusa, esecuzione al tocco della zona = ordine limite,
o alla successiva se serve conferma di chiusura).
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
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

FVG_MIN_ATR_FRAC = 0.05
MAX_FVG_STALE_DAYS = 20            # staleness massima di un FVG 4H mai toccato
REACT_DISPLACEMENT_MULT = 2.0      # soglia displacement, stessa di Asia-sweep v4
MAX_TOUCH_TO_REACT_BARS = 16       # 4h su 15m
MAX_REACT_TO_NEWFVG_BARS = 12      # 3h su 15m
MAX_NEWFVG_TO_RETEST_BARS = 16     # 4h su 15m
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 96                 # 24h su 15m, cap di sicurezza
RR_GRID = [2.0, 3.0]
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("MTF FVG Reaction (4H FVG -> reazione 15m -> nuovo FVG 15m -> entry) — Validation Pipeline")
w(SEP)
w("\nReplica letterale di una regola pubblicata sui social (non attribuita,")
w("nessun track record allegato): 4H FVG come zona, reazione aggressiva +")
w("nuovo FVG sul 15m come timing di ingresso, target 2-3 RR.")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 4H + 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
df15 = add_indicators(raw["15M"])
IDX4H = df4h.index; N4H = len(df4h)
IDX15 = df15.index; N15 = len(df15)
print(f"  4H:  {N4H:,} bars  ({IDX4H[0].date()} → {IDX4H[-1].date()})")
print(f"  15M: {N15:,} bars  ({IDX15[0].date()} → {IDX15[-1].date()})  (loaded in {time.time()-t0:.0f}s)")

OP4, CL4, HI4, LO4 = (df4h[c].values.astype(float) for c in ("open", "close", "high", "low"))
ATR4 = np.where(df4h["atr_14"].values > 0, df4h["atr_14"].values, 1.0)

OP, CL, HI, LO = (df15[c].values.astype(float) for c in ("open", "close", "high", "low"))
ATR15 = np.where(df15["atr_14"].values > 0, df15["atr_14"].values, 1.0)
rng15 = HI - LO
avg_rng3 = pd.Series(rng15).rolling(3).mean().shift(1).values   # media mobile CAUSALE (barre precedenti)

IDX15_vals = IDX15.values


def bar15_at_or_after(ts):
    pos = np.searchsorted(IDX15_vals, np.datetime64(ts), side="left")
    return pos if pos < N15 else None


# ── STEP 0: 4H FVG list con finestra di validità causale ──────────────────
print("[FVG-4H] Detecting 4H FVGs …")
fvgs_4h = []
for i in range(1, N4H - 1):
    atr = ATR4[i]
    if atr <= 0: continue
    if HI4[i - 1] < LO4[i + 1]:
        fbot, ftop = HI4[i - 1], LO4[i + 1]
        if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
        direction = "bull"
    elif LO4[i - 1] > HI4[i + 1]:
        ftop, fbot = LO4[i - 1], HI4[i + 1]
        if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
        direction = "bear"
    else:
        continue
    # invalidazione: prima barra 4H (dopo conferma) la cui chiusura supera il
    # bordo lontano, cercata entro un tetto di staleness (evita scansioni
    # O(N) per FVG mai invalidati fino a fine storia)
    invalid_4h_i = None
    search_end = min(N4H, i + 2 + MAX_FVG_STALE_DAYS * 6)   # 6 barre 4H/giorno
    for j in range(i + 2, search_end):
        if direction == "bull" and CL4[j] < fbot:
            invalid_4h_i = j; break
        if direction == "bear" and CL4[j] > ftop:
            invalid_4h_i = j; break
    fvgs_4h.append(dict(i=i, direction=direction, top=ftop, bot=fbot, invalid_4h_i=invalid_4h_i))
print(f"  {len(fvgs_4h)} FVG 4H rilevati")


# ── State machine per ogni FVG 4H ─────────────────────────────────────────
def find_pattern(fvg):
    start_ts = IDX4H[fvg["i"] + 2]
    start15 = bar15_at_or_after(start_ts)
    if start15 is None:
        return None
    if fvg["invalid_4h_i"] is not None:
        end_ts = IDX4H[fvg["invalid_4h_i"]]
        end15 = bar15_at_or_after(end_ts)
        end15 = end15 if end15 is not None else N15
    else:
        end15 = min(N15, start15 + MAX_FVG_STALE_DAYS * 96)

    ftop, fbot, direction = fvg["top"], fvg["bot"], fvg["direction"]

    # STEP 1: primo tocco 15m dentro la zona 4H
    touch_i = None
    for t in range(start15, min(end15, N15)):
        if LO[t] <= ftop and HI[t] >= fbot:
            touch_i = t
            break
    if touch_i is None:
        return None

    bias = 1 if direction == "bull" else -1

    # STEP 2: reazione aggressiva nella direzione del bias, entro la finestra
    react_i = None
    for j in range(touch_i + 1, min(touch_i + 1 + MAX_TOUCH_TO_REACT_BARS, N15 - 1)):
        if np.isnan(avg_rng3[j]) or avg_rng3[j] <= 0: continue
        is_bull_candle = CL[j] > OP[j]
        is_bear_candle = CL[j] < OP[j]
        displaced = rng15[j] >= REACT_DISPLACEMENT_MULT * avg_rng3[j]
        if not displaced: continue
        if bias == 1 and is_bull_candle:
            react_i = j; break
        if bias == -1 and is_bear_candle:
            react_i = j; break
    if react_i is None:
        return None

    # STEP 3: nuovo FVG 15m, stessa direzione, entro la finestra dalla reazione
    new_fvg = None
    for m in range(max(react_i - 1, touch_i + 1), min(react_i + MAX_REACT_TO_NEWFVG_BARS, N15 - 1)):
        atr = ATR15[m]
        if atr <= 0: continue
        if bias == 1 and HI[m - 1] < LO[m + 1]:
            nbot, ntop = HI[m - 1], LO[m + 1]
            if (ntop - nbot) < FVG_MIN_ATR_FRAC * atr: continue
            new_fvg = dict(m=m, top=ntop, bot=nbot)
            break
        if bias == -1 and LO[m - 1] > HI[m + 1]:
            ntop, nbot = LO[m - 1], HI[m + 1]
            if (ntop - nbot) < FVG_MIN_ATR_FRAC * atr: continue
            new_fvg = dict(m=m, top=ntop, bot=nbot)
            break
    if new_fvg is None:
        return None

    # STEP 4: retest del nuovo FVG 15m
    m = new_fvg["m"]; ntop, nbot = new_fvg["top"], new_fvg["bot"]
    retest_i = None
    for r in range(m + 2, min(m + 2 + MAX_NEWFVG_TO_RETEST_BARS, N15 - 1)):
        if LO[r] <= ntop and HI[r] >= nbot:
            retest_i = r
            break
    if retest_i is None:
        return None

    if bias == 1:
        ep = min(CL[retest_i], ntop)
        swing = LO[touch_i:m + 1].min()
        sl = swing - STOP_BUFFER_ATR * ATR15[retest_i]
    else:
        ep = max(CL[retest_i], nbot)
        swing = HI[touch_i:m + 1].max()
        sl = swing + STOP_BUFFER_ATR * ATR15[retest_i]

    risk = abs(ep - sl)
    if risk <= 0:
        return None
    hold = min(MAX_HOLD_BARS, N15 - 1 - retest_i)
    if hold < 1:
        return None
    return dict(entry_i=retest_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold, fvg4h_i=fvg["i"])


print("[PATTERN] Scanning sequential state machine per FVG 4H …")
t1 = time.time()
ALL_EVENTS = []
n_no_touch = n_no_react = n_no_newfvg = n_no_retest = 0
for fvg in fvgs_4h:
    ev = find_pattern(fvg)
    if ev is not None:
        ALL_EVENTS.append(ev)
print(f"  done in {time.time()-t1:.0f}s")
w(f"\n[EVENTS] {len(fvgs_4h)} FVG 4H -> {len(ALL_EVENTS)} pattern completi (1 trade/FVG 4H max)")


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
            if j >= N15: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl: out = "sl"; exit_price = sl; break
            if hit_tp: out = "tp"; exit_price = target; break
        if exit_price is None:
            exit_price = CL[min(entry_i + hold, N15 - 1)]
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


holdout_events = [e for e in ALL_EVENTS if IDX15[e["entry_i"]] >= CUTOFF]
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
        year_evs.setdefault(IDX15[e["entry_i"]].year, []).append(e)
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

best_rr = max(dsr_full, key=lambda r: r["dsr"])["rr"]
w(f"\n{SEP}")
w(f"Slippage-stress — RR migliore per DSR = {best_rr:.1f}")
w(SEP)
w(f"\n  {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_STRESS_BPS:
    extra = bps / 10_000.0
    for scope_name, evs in [("full-sample", ALL_EVENTS), ("holdout", holdout_events)]:
        r = run_bt(evs, best_rr, extra_slippage_pct=extra)
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/mtf_fvg_reaction.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# MTF FVG Reaction (4H FVG -> reazione 15m -> nuovo FVG 15m) — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
