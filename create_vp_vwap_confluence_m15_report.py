#!/usr/bin/env python3
"""
create_vp_vwap_confluence_m15_report.py
=========================================
Volume Profile + VWAP Confluence su M15 invece di 1H (richiesta esplicita
dell'utente: "testa su 15 min"). Stessa logica del report base
(`create_vp_vwap_confluence_report.py`, RR=3.0 il migliore per DSR lì
trovato): rolling daily POC del Volume Profile confluente col VWAP di
sessione, pullback in trend-context, rejection, stop al minimo/massimo
strutturale, target RR-graded.

Le soglie espresse in NUMERO DI BARRE sono riscalate ×4 (SCALE = 60/15)
per conservare lo stesso significato in tempo reale calibrato su 1H
(stessa disciplina già usata per Asia-sweep v3 M15):
  LOOKBACK        5   ->  20   barre (contesto di trend, stesse ore)
  SWING_LOOKBACK 10   ->  40   barre (minimo/massimo strutturale, stesse ore)
  MAX_HOLD_BARS  48   -> 192   barre (stesso cap di 2 giorni)
  bar_in_day>=2 h ->  bar_in_day>=8  barre (stesse 2h minime di sessione)
LOOKBACK_DAYS (20 giorni civili) e N_BINS (50) NON si riscalano: sono
espressi in giorni/bin, non in barre.
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

from src.strategy.data_fetcher import fetch_binance_vision_klines
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

BAR_MIN = 15
SCALE = 60 // BAR_MIN                    # 4

LOOKBACK = 5 * SCALE                     # 20 barre M15 = 5h (come 5 barre 1H)
MIN_SEP_ATR = 0.5
LOOKBACK_DAYS = 20                       # giorni civili, non riscalato
N_BINS = 50                              # bin di prezzo, non riscalato
CONF_ATR_MULT = 0.5
SWING_LOOKBACK = 10 * SCALE              # 40 barre M15 = 10h
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 48 * SCALE               # 192 barre M15 = 2 giorni
MIN_BARS_IN_DAY = 2 * SCALE              # 8 barre M15 = 2h minime di sessione
RR_GRID = [1.0, 1.5, 2.0, 3.0]
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Volume Profile + VWAP Confluence — M15 — Validation Pipeline")
w(SEP)
w(f"\nStessa logica del report 1H base (RR=3.0 migliore per DSR: full DSR=0.884,")
w(f"holdout DSR=0.498). Qui su barre M15, soglie in-barra riscalate ×{SCALE} per")
w(f"conservare lo stesso significato in tempo reale (LOOKBACK={LOOKBACK}b={LOOKBACK//SCALE}h, "
  f"SWING_LOOKBACK={SWING_LOOKBACK}b={SWING_LOOKBACK//SCALE}h, MAX_HOLD={MAX_HOLD_BARS}b={MAX_HOLD_BARS//SCALE//24}gg).")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print(f"\n[DATA] Loading {BAR_MIN}m …")
dfM = add_indicators(fetch_binance_vision_klines(
    f"{BAR_MIN}m", start_year=START_YEAR, start_month=1, workers=6, verbose=False))
IDXM = dfM.index
NM = len(dfM)
print(f"  {BAR_MIN}m: {NM:,} bars  ({IDXM[0].date()} → {IDXM[-1].date()})  "
      f"(loaded in {time.time()-t0:.0f}s)")

CL = dfM["close"].values.astype(float)
HI = dfM["high"].values.astype(float)
LO = dfM["low"].values.astype(float)
OP = dfM["open"].values.astype(float)
VOL = dfM["volume"].values.astype(float)
ATR = np.where(dfM["atr_14"].values > 0, dfM["atr_14"].values, 1.0)

# ── VWAP di sessione (reset giornaliero UTC), cumulativo, causale ─────────
dates = IDXM.normalize().values
day_change = np.r_[True, dates[1:] != dates[:-1]]
day_id = np.cumsum(day_change) - 1
day_start_idx = np.where(day_change)[0]
n_days = len(day_start_idx)

tp = (HI + LO + CL) / 3.0
tmp = pd.DataFrame({"day_id": day_id, "pv": tp * VOL, "vol": VOL})
g = tmp.groupby("day_id")
cum_pv = g["pv"].cumsum().values
cum_v = g["vol"].cumsum().values
bar_in_day = g.cumcount().values

VWAP = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
print(f"[VWAP] valid bars: {np.isfinite(VWAP).sum():,} / {NM:,}")

# ── Volume Profile / POC giornaliero rolling, causale ─────────────────────
print(f"[VP] Computing rolling daily POC (lookback={LOOKBACK_DAYS}d, bins={N_BINS}) …")
t1 = time.time()
daily_poc = np.full(n_days, np.nan)
MIN_WIN_BARS = 100 * SCALE
for d in range(n_days):
    win_start_day = d - LOOKBACK_DAYS
    if win_start_day < 0:
        continue
    lo_bar = day_start_idx[win_start_day]
    hi_bar = day_start_idx[d]
    if hi_bar - lo_bar < MIN_WIN_BARS:
        continue
    seg_tp = tp[lo_bar:hi_bar]
    seg_vol = VOL[lo_bar:hi_bar]
    seg_lo = LO[lo_bar:hi_bar].min()
    seg_hi = HI[lo_bar:hi_bar].max()
    if seg_hi <= seg_lo:
        continue
    hist, edges = np.histogram(seg_tp, bins=N_BINS, range=(seg_lo, seg_hi), weights=seg_vol)
    poc_idx = int(np.argmax(hist))
    daily_poc[d] = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0

POC = daily_poc[day_id]
print(f"  valid POC bars: {np.isfinite(POC).sum():,} / {NM:,}  (in {time.time()-t1:.0f}s)")


def build_events():
    evs = []
    last_exit = -1
    start_i = max(LOOKBACK, SWING_LOOKBACK) + 2
    for i in range(start_i, NM - 1):
        if i <= last_exit:
            continue
        if not np.isfinite(VWAP[i]) or not np.isfinite(POC[i]) or bar_in_day[i] < MIN_BARS_IN_DAY or ATR[i] <= 0:
            continue

        ref = i - LOOKBACK
        if not np.isfinite(VWAP[ref]) or ATR[ref] <= 0:
            continue
        sep = CL[ref] - VWAP[ref]
        if abs(sep) < MIN_SEP_ATR * ATR[ref]:
            continue
        bias = 1 if sep > 0 else -1

        if abs(POC[i] - VWAP[i]) > CONF_ATR_MULT * ATR[i]:
            continue

        zone_lo = min(POC[i], VWAP[i])
        zone_hi = max(POC[i], VWAP[i])
        touch = LO[i] <= zone_hi and HI[i] >= zone_lo
        if not touch:
            continue

        zone_mid = (POC[i] + VWAP[i]) / 2.0
        rejection = (bias == 1 and CL[i] > zone_mid) or (bias == -1 and CL[i] < zone_mid)
        if not rejection:
            continue

        if bias == 1:
            swing = LO[i - SWING_LOOKBACK:i + 1].min()
            sl = swing - STOP_BUFFER_ATR * ATR[i]
        else:
            swing = HI[i - SWING_LOOKBACK:i + 1].max()
            sl = swing + STOP_BUFFER_ATR * ATR[i]

        entry_i = i + 1
        ep = OP[entry_i]
        risk = abs(ep - sl)
        if risk <= 0:
            continue
        hold = min(MAX_HOLD_BARS, NM - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(i=i, entry_i=entry_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold))
        last_exit = entry_i + hold
    return evs


ALL_EVENTS = build_events()
print(f"[EVENTS] {len(ALL_EVENTS)} trade di confluenza POC/VWAP (sequenziali, non sovrapposti)")


def run_bt(evs, rr, slippage_pct=0.0):
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
            if j >= NM: break
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
            j = min(entry_i + hold, NM - 1)
            exit_price = CL[j]
        if out == "tp": n_tp += 1
        elif out == "sl": n_sl += 1
        else: n_time += 1

        if risk <= 0: continue
        r = INIT_CAP * RISK_PCT
        units = min(r / risk, MAX_LEV * INIT_CAP / ep)
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
                n_tp=n_tp, n_sl=n_sl, n_time=n_time)


def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=10):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


holdout_events = [e for e in ALL_EVENTS if IDXM[e["entry_i"]] >= CUTOFF]
w(f"\n  Holdout genuino 2025-2026: {len(holdout_events)} trade")

dsr_full = []
dsr_holdout = []

for rr in RR_GRID:
    w(f"\n{SEP}")
    w(f"RR = {rr:.2f}")
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
        year_evs.setdefault(IDXM[e["entry_i"]].year, []).append(e)
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

# ── DSR family (griglia RR) ────────────────────────────────────────────────
w(f"\n{SEP}")
w(f"DSR — famiglia N={len(RR_GRID)} (griglia RR)")
w(SEP)
deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
w(f"\n  Full-sample:")
w(f"  {'RR':>6}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_full:
    w(f"  {r['rr']:>5.2f}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
w(f"\n  Holdout 2025-2026:")
w(f"  {'RR':>6}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_holdout:
    w(f"  {r['rr']:>5.2f}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

# ── Slippage sensitivity sul migliore per DSR ─────────────────────────────
best_rr = max(dsr_full, key=lambda r: r["dsr"])["rr"]
w(f"\n{SEP}")
w(f"Slippage sensitivity — RR migliore per DSR = {best_rr:.2f}")
w(SEP)
w(f"\n  {'Slippage':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_TESTS_BPS:
    slip = bps / 10_000.0
    for scope_name, evs in [("full-sample", ALL_EVENTS), ("holdout", holdout_events)]:
        r = run_bt(evs, best_rr, slippage_pct=slip)
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vp_vwap_confluence_m15.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Volume Profile + VWAP Confluence — M15 — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
