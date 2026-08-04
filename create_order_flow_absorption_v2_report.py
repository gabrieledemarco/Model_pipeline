#!/usr/bin/env python3
"""
create_order_flow_absorption_v2_report.py
=============================================
v2 — diagnosi del fallimento di `create_order_flow_absorption_report.py`:
lo stop era posizionato all'estremo della barra di assorbimento stessa,
ma quella barra è STATA SELEZIONATA per avere range basso rispetto al
volume -> il suo estremo è, per costruzione, insolitamente vicino
all'entry. Stessa asimmetria di first-passage-time vista più volte in
questa sessione (NY-ORB-VP, VWAP MR v2, VWAP Bounce v1/v2): SL-hit rate
saliva dal 52.0% al 65.6% man mano che il RR (e quindi l'attesa) cresceva.

v2 sostituisce lo stop con un multiplo ATR (scansionato, RR fissato a
2.0 — valore intermedio della griglia v1) per isolare se il problema sia
la tesi di assorbimento/CVD in sé o solo la costruzione dello stop.
Entry logic IDENTICA a v1 (trend-context + tocco POC + tape footprint
large-print/absorption + CVD locale + rejection); solo lo stop cambia,
quindi la lista eventi è costruita UNA SOLA VOLTA e riusata per tutta la
griglia (stesso pattern di `create_vwap_bounce_v2_report.py`).
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
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE_TAKER = 0.00055
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

BAR_MIN = 15
SCALE = 60 // BAR_MIN

LOOKBACK = 5 * SCALE
MIN_SEP_ATR = 0.5
LOOKBACK_DAYS = 20
N_BINS = 50
CONF_ATR_MULT = 0.5
MIN_BARS_IN_DAY = 2 * SCALE

ROLL_WIN = 5 * 24 * SCALE
P_ABSORPTION = 0.90
P_SIZE = 0.75
CVD_K = 3
FIXED_RR = 2.0
SL_ATR_MULT_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MAX_HOLD_BARS = 48 * SCALE
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Order-Flow: Large-Order Absorption v2 — stop ATR scan — Validation Pipeline")
w(SEP)
w("\nv1 aveva SL-hit rate crescente (52.0%->65.6%) con RR, sintomo di uno")
w("stop troppo vicino perché ancorato all'estremo di una barra selezionata")
w("apposta per range basso. v2 scansiona lo stop ATR con RR fisso=2.0.")

t0 = time.time()
print(f"\n[DATA] Loading {BAR_MIN}m + taker flow …")
dfM = fetch_binance_vision_taker_flow(f"{BAR_MIN}m", start_year=START_YEAR, start_month=1,
                                       workers=6, verbose=False)
dfM = add_indicators(dfM)
IDXM = dfM.index
NM = len(dfM)
print(f"  {BAR_MIN}m: {NM:,} bars  ({IDXM[0].date()} → {IDXM[-1].date()})  "
      f"(loaded in {time.time()-t0:.0f}s)")

CL = dfM["close"].values.astype(float)
HI = dfM["high"].values.astype(float)
LO = dfM["low"].values.astype(float)
OP = dfM["open"].values.astype(float)
VOL = dfM["volume"].values.astype(float)
TBB = dfM["taker_buy_base"].values.astype(float)
NTR = np.maximum(dfM["n_trades"].values.astype(float), 1.0)
ATR = np.where(dfM["atr_14"].values > 0, dfM["atr_14"].values, 1.0)

DELTA = 2.0 * TBB - VOL
AVG_SIZE = VOL / NTR
RNG = HI - LO
ABSORPTION_SCORE = VOL / np.maximum(RNG / ATR, 1e-9)

print(f"[TAPE] Computing rolling percentiles (window={ROLL_WIN} bars) …")
abs_pctl = pd.Series(ABSORPTION_SCORE, index=IDXM).rolling(ROLL_WIN, min_periods=ROLL_WIN).rank(pct=True).values
size_pctl = pd.Series(AVG_SIZE, index=IDXM).rolling(ROLL_WIN, min_periods=ROLL_WIN).rank(pct=True).values
cvd_local = pd.Series(DELTA, index=IDXM).rolling(CVD_K, min_periods=CVD_K).sum().values

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

print(f"[VP] Computing rolling daily POC (lookback={LOOKBACK_DAYS}d, bins={N_BINS}) …")
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


def build_events():
    evs = []
    last_exit = -1
    start_i = max(LOOKBACK, ROLL_WIN) + 2
    for i in range(start_i, NM - 1):
        if i <= last_exit:
            continue
        if not np.isfinite(VWAP[i]) or not np.isfinite(POC[i]) or bar_in_day[i] < MIN_BARS_IN_DAY or ATR[i] <= 0:
            continue
        if not np.isfinite(abs_pctl[i]) or not np.isfinite(size_pctl[i]) or not np.isfinite(cvd_local[i]):
            continue

        ref = i - LOOKBACK
        if not np.isfinite(VWAP[ref]) or ATR[ref] <= 0:
            continue
        sep = CL[ref] - VWAP[ref]
        if abs(sep) < MIN_SEP_ATR * ATR[ref]:
            continue
        bias = 1 if sep > 0 else -1

        touch = LO[i] <= POC[i] + CONF_ATR_MULT * ATR[i] and HI[i] >= POC[i] - CONF_ATR_MULT * ATR[i]
        if not touch:
            continue

        if not (abs_pctl[i] >= P_ABSORPTION and size_pctl[i] >= P_SIZE):
            continue

        if cvd_local[i] * bias <= 0:
            continue

        rejection = (bias == 1 and CL[i] > POC[i]) or (bias == -1 and CL[i] < POC[i])
        if not rejection:
            continue

        entry_i = i + 1
        ep = OP[entry_i]
        hold = min(MAX_HOLD_BARS, NM - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(i=i, entry_i=entry_i, d=bias, ep=ep, atr=ATR[i], hold=hold))
        last_exit = entry_i + hold
    return evs


ALL_EVENTS = build_events()
print(f"[EVENTS] {len(ALL_EVENTS)} trade (sequenziali, non sovrapposti)")


def run_bt(evs, sl_atr_mult, slippage_pct=0.0):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0
    for ev in evs:
        d, ep, entry_i, hold = ev["d"], ev["ep"], ev["entry_i"], ev["hold"]
        risk = sl_atr_mult * ev["atr"]
        sl = ep - d * risk
        target = ep + d * FIXED_RR * risk
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

for sl_mult in SL_ATR_MULT_GRID:
    w(f"\n{SEP}")
    w(f"SL = {sl_mult:.2f} × ATR  (RR fisso = {FIXED_RR})")
    w(SEP)

    res = run_bt(ALL_EVENTS, sl_mult)
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

w(f"\n{SEP}")
w(f"DSR — famiglia N={len(SL_ATR_MULT_GRID)} (moltiplicatori ATR dello stop)")
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
out_path = Path("reports/order_flow_absorption_v2.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Order-Flow: Large-Order Absorption v2 — stop ATR scan — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
