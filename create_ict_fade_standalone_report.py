#!/usr/bin/env python3
"""
create_ict_fade_standalone_report.py
=======================================
Prima strategia costruita combinando gli oggetti ICT/SMC studiati: FADE
dei trigger 15M (Sweep, FVG, Order Block, Breaker Block, Inversion FVG,
Power of 3) che si formano SENZA una zona 4H (demand/supply, Order Block
o FVG) attiva a supporto — il segnale "standalone" che in
`htf_ltf_confluence.md` mostrava il forward return più forte e più
statisticamente robusto di tutta la sessione (fino a -0.43% a 20 barre
nella direzione implicita del trigger, un ordine di grandezza sopra la
frizione 0.14% round-trip) — qui tradato in modo diretto: si prende il
lato OPPOSTO del trigger quando non c'è supporto strutturale HTF.

Costruzione (causale):
  1. Rileva le 6 zone 4H (Order Block + FVG, demand/supply) con finestra
     attiva mappata sul 15M (identico a `htf_ltf_confluence.md`).
  2. Rileva i 6 trigger 15M (identico a `htf_ltf_confluence.md`).
  3. STANDALONE = trigger la cui barra NON è dentro nessuna zona 4H
     attiva della stessa direzione.
  4. FADE: direzione del trade = OPPOSTA alla direzione implicita del
     trigger standalone.
  5. Pool di tutti e 6 i tipi di trigger, evento singolo per barra
     (nessun overlap — gating sequenziale, un solo trade alla volta).
  6. STOP strutturale: minimo/massimo (nel verso del fade) sulle ultime
     SWING_LOOKBACK barre prima dell'entry, + piccolo buffer ATR — stessa
     costruzione già validata in `vp_vwap_confluence.md`.
  7. TARGET: RR × rischio, griglia [1.5, 2.0, 3.0] con DSR family.
  8. ENTRY causale: apertura della barra successiva alla conferma del
     trigger.

Frizioni Bybit obbligatorie (taker 0.055% + slippage 0.015%/lato = 0.14%
round-trip). Validazione: per-anno, holdout 2025-2026 genuino, Monte
Carlo i.i.d.+block, DSR family sul grid RR, slippage-stress.
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
from src.strategy.mtf_swing import find_pivots
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
OB_LOCAL_WIN = 3
OB_MAX_CONFIRM_BARS = 20
SWEEP_MAX_REJECT_BARS = 3
BREAKER_MAX_WAIT_BARS = 60
ACC_BARS = 8
ACC_RANGE_ATR_MULT = 1.5
MANIP_MAX_BARS = 10
ZONE_MAX_AGE_BARS_4H = 60

SWING_LOOKBACK = 10
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 96
RR_GRID = [1.5, 2.0, 3.0]
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Fade dei trigger ICT standalone (senza supporto 4H) — Validation Pipeline")
w(SEP)
w("\nCombina i 6 oggetti (Sweep/FVG/OB/Breaker/IFVG/PO3): quando un trigger 15M")
w("si forma SENZA una zona demand/supply 4H attiva, si prende il lato OPPOSTO")
w("(fade) — segnale con l'effetto più forte trovato in questa sessione.")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 4H + 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
df15 = add_indicators(raw["15M"])
IDX4H = df4h.index; N4H = len(df4h)
IDX = df15.index; N = len(df15)
print(f"  4H:  {N4H:,} bars   15M: {N:,} bars  (loaded in {time.time()-t0:.0f}s)")

CL4, HI4, LO4, OP4 = (df4h[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR4 = np.where(df4h["atr_14"].values > 0, df4h["atr_14"].values, np.nan)
CL, HI, LO, OP = (df15[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR15raw = df15["atr_14"].values
ATR = np.where(ATR15raw > 0, ATR15raw, np.nan)

IDX15_vals = IDX.values
def bar15_at_or_after(ts):
    pos = np.searchsorted(IDX15_vals, np.datetime64(ts), side="left")
    return pos if pos < N else None


# ── Detectors (identici a htf_ltf_confluence.py) ──────────────────────────
def detect_fvgs(HIx, LOx, ATRx):
    n = len(HIx); events = []; zones = []
    for i in range(1, n - 1):
        atr = ATRx[i]
        if atr <= 0 or np.isnan(atr): continue
        if HIx[i - 1] < LOx[i + 1]:
            fbot, ftop = HIx[i - 1], LOx[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=1)); zones.append(dict(idx=i + 1, dir=1, lo=fbot, hi=ftop))
        elif LOx[i - 1] > HIx[i + 1]:
            ftop, fbot = LOx[i - 1], HIx[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=-1)); zones.append(dict(idx=i + 1, dir=-1, lo=fbot, hi=ftop))
    return events, zones


def detect_order_blocks(OPx, CLx):
    n = len(CLx); events = []; zones = []
    for i in range(OB_LOCAL_WIN, n - OB_MAX_CONFIRM_BARS - 1):
        if CLx[i] < OPx[i]:
            if CLx[i] == CLx[i - OB_LOCAL_WIN + 1:i + 1].min():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CLx[j] > OPx[i]:
                        events.append(dict(idx=j, dir=1))
                        zones.append(dict(idx=j, dir=1, lo=min(OPx[i], CLx[i]), hi=max(OPx[i], CLx[i])))
                        break
        elif CLx[i] > OPx[i]:
            if CLx[i] == CLx[i - OB_LOCAL_WIN + 1:i + 1].max():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CLx[j] < OPx[i]:
                        events.append(dict(idx=j, dir=-1))
                        zones.append(dict(idx=j, dir=-1, lo=min(OPx[i], CLx[i]), hi=max(OPx[i], CLx[i])))
                        break
    return events, zones


def detect_sweeps(HIx, LOx, CLx, pivots):
    n = len(CLx); events = []
    confirmed_highs, confirmed_lows = [], []; p_idx = 0
    for i in range(n):
        while p_idx < len(pivots) and pivots[p_idx]["confirm_idx"] == i:
            piv = pivots[p_idx]
            (confirmed_highs if piv["kind"] == 1 else confirmed_lows).append(piv["price"])
            p_idx += 1
        if confirmed_highs and HIx[i] > confirmed_highs[-1]:
            level = confirmed_highs[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CLx[i + k] < level: events.append(dict(idx=i + k, dir=-1)); break
        if confirmed_lows and LOx[i] < confirmed_lows[-1]:
            level = confirmed_lows[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CLx[i + k] > level: events.append(dict(idx=i + k, dir=1)); break
    return events


def detect_breakers(zones_ob, HIx, LOx, CLx):
    n = len(CLx); events = []
    for z in zones_ob:
        j0 = z["idx"]; lo, hi = z["lo"], z["hi"]
        invalid_i = None
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1 and CLx[k] < lo: invalid_i = k; break
            if z["dir"] == -1 and CLx[k] > hi: invalid_i = k; break
        if invalid_i is None: continue
        for k in range(invalid_i + 1, min(invalid_i + 1 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1 and LOx[k] <= hi and HIx[k] >= lo and CLx[k] < lo:
                events.append(dict(idx=k, dir=-1)); break
            if z["dir"] == -1 and HIx[k] >= lo and LOx[k] <= hi and CLx[k] > hi:
                events.append(dict(idx=k, dir=1)); break
    return events


def detect_ifvgs(zones_fvg, CLx):
    n = len(CLx); events = []
    for z in zones_fvg:
        j0 = z["idx"]; lo, hi, d = z["lo"], z["hi"], z["dir"]
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if d == 1 and CLx[k] < lo: events.append(dict(idx=k, dir=-1)); break
            if d == -1 and CLx[k] > hi: events.append(dict(idx=k, dir=1)); break
    return events


def detect_amd(HIx, LOx, CLx, ATRx):
    n = len(CLx); events = []
    for i in range(ACC_BARS, n - MANIP_MAX_BARS - 1):
        atr = ATRx[i]
        if atr <= 0 or np.isnan(atr): continue
        seg_hi = HIx[i - ACC_BARS:i].max(); seg_lo = LOx[i - ACC_BARS:i].min()
        if (seg_hi - seg_lo) > ACC_RANGE_ATR_MULT * atr: continue
        for k in range(i, min(i + MANIP_MAX_BARS, n)):
            if HIx[k] > seg_hi:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CLx[m] < seg_hi: events.append(dict(idx=m, dir=-1)); break
                break
            if LOx[k] < seg_lo:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CLx[m] > seg_lo: events.append(dict(idx=m, dir=1)); break
                break
    return events


print("[HTF] Detecting 4H demand/supply zones …")
_, ob_zones_4h = detect_order_blocks(OP4, CL4)
_, fvg_zones_4h = detect_fvgs(HI4, LO4, ATR4)
raw_zones_4h = ob_zones_4h + fvg_zones_4h

htf_zones_15 = []
for z in raw_zones_4h:
    j0 = z["idx"]
    if j0 >= N4H: continue
    invalid_i = None
    for k in range(j0, min(j0 + ZONE_MAX_AGE_BARS_4H, N4H)):
        if z["dir"] == 1 and CL4[k] < z["lo"]: invalid_i = k; break
        if z["dir"] == -1 and CL4[k] > z["hi"]: invalid_i = k; break
    end_4h = invalid_i if invalid_i is not None else min(N4H - 1, j0 + ZONE_MAX_AGE_BARS_4H)
    start15 = bar15_at_or_after(IDX4H[j0])
    end15 = bar15_at_or_after(IDX4H[end_4h])
    if start15 is None: continue
    if end15 is None: end15 = N
    if end15 <= start15: continue
    htf_zones_15.append(dict(dir=z["dir"], start=start15, end=end15))
print(f"  {len(htf_zones_15)} zone 4H mappate su 15M")

htf_by_dir = {1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == 1]),
              -1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == -1])}
starts_1 = np.array([s for s, e in htf_by_dir[1]]); ends_1 = np.array([e for s, e in htf_by_dir[1]])
starts_m1 = np.array([s for s, e in htf_by_dir[-1]]); ends_m1 = np.array([e for s, e in htf_by_dir[-1]])


def in_active_zone(idx, d):
    starts, ends = (starts_1, ends_1) if d == 1 else (starts_m1, ends_m1)
    if len(starts) == 0: return False
    pos = np.searchsorted(starts, idx, side="right") - 1
    while pos >= 0:
        if starts[pos] <= idx <= ends[pos]: return True
        if idx - starts[pos] > 4 * ZONE_MAX_AGE_BARS_4H: break
        pos -= 1
    return False


print("[LTF] Detecting 15M triggers …")
pivots15 = find_pivots(HI, LO, 1, 1)
sweeps = detect_sweeps(HI, LO, CL, pivots15)
fvgs, _ = detect_fvgs(HI, LO, ATR)
obs, ob_zones15 = detect_order_blocks(OP, CL)
_, fvg_zones15 = detect_fvgs(HI, LO, ATR)
breakers = detect_breakers(ob_zones15, HI, LO, CL)
ifvgs = detect_ifvgs(fvg_zones15, CL)
amd = detect_amd(HI, LO, CL, ATR)

ALL_TRIGGERS = sweeps + fvgs + obs + breakers + ifvgs + amd
standalone = [e for e in ALL_TRIGGERS if not in_active_zone(e["idx"], e["dir"])]
standalone.sort(key=lambda e: e["idx"])
print(f"  {len(ALL_TRIGGERS)} trigger totali -> {len(standalone)} standalone (senza zona 4H)")

# ── Costruzione eventi FADE (direzione opposta), sequenziali, stop strutturale ──
def build_fade_events():
    evs = []
    last_exit = -1
    for e in standalone:
        i = e["idx"]
        if i <= last_exit: continue
        if i < SWING_LOOKBACK + 2 or i >= N - 1: continue
        if np.isnan(ATR[i]) or ATR[i] <= 0: continue
        bias = -e["dir"]   # FADE: direzione opposta al trigger
        if bias == 1:
            swing = LO[i - SWING_LOOKBACK:i + 1].min()
            sl = swing - STOP_BUFFER_ATR * ATR[i]
        else:
            swing = HI[i - SWING_LOOKBACK:i + 1].max()
            sl = swing + STOP_BUFFER_ATR * ATR[i]
        entry_i = i + 1
        ep = OP[entry_i]
        risk = abs(ep - sl)
        if risk <= 0: continue
        hold = min(MAX_HOLD_BARS, N - 1 - entry_i)
        if hold < 1: continue
        evs.append(dict(entry_i=entry_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold))
        last_exit = entry_i + hold
    return evs


ALL_EVENTS = build_fade_events()
print(f"[EVENTS] {len(ALL_EVENTS)} trade fade (sequenziali, non sovrapposti)")


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
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/ict_fade_standalone.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Fade dei trigger ICT standalone (senza supporto 4H) — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
