#!/usr/bin/env python3
"""
create_ny_orb_vp_absorption_report.py
========================================
Nuova strategia proposta dall'utente, sottoposta alla pipeline di
validazione rigorosa della sessione (WFO/per-anno, holdout genuino
2025-2026, Monte Carlo i.i.d.+block, DSR, slippage sensitivity).

Regola di trading:
  1. RANGE: primi 30 minuti della sessione NY (13:30-14:00 UTC, convenzione
     "NY cash/killzone" gia' usata altrove nel repo — nessun aggiustamento
     DST, come nel resto della pipeline).
  2. BREAKOUT: prima barra 5m (14:00-20:00 UTC) che chiude sopra il massimo
     del range (LONG) o sotto il minimo (SHORT).
  3. VOLUME PROFILE: calcolato sulle barre 1m del range (30 barre), bucket
     di prezzo (20 bin), volume di ogni barra distribuito proporzionalmente
     alla sovrapposizione [low,high] della barra coi bin. POC = bin con
     volume totale massimo.
  4. ABSORPTION ENTRY: dopo il breakout, si attende che il prezzo ritorni
     sul POC (barra 5m con low <= POC <= high) entro fine giornata
     (22:00 UTC). Entry = POC, stessa direzione del breakout (continuazione
     dopo il "retest" della zona a piu' alto volume).
  5. TARGET = prossimo livello di liquidita': pivot 1H causale non ancora
     rotto (target_high/target_low da src/strategy/mtf_swing.py,
     left=right=6), nella direzione del trade. Nessun target valido -> no
     trade quel giorno.
  6. STOP dinamico per mantenere RR fisso: stop_dist = target_dist / RR,
     testato per RR = 1.0 e RR = 2.0 (richiesta esplicita dell'utente).
  7. Uscita: TP, SL, o time-stop a max_hold=8h (96 barre 5m) — stessa
     convenzione della strategia ML RF 8h gia' validata in questa sessione.
     Se SL e TP sono toccati nella stessa barra, si assume SL (conservativo).

Un solo trade al giorno (max). Sizing a rischio-dollaro fisso, fee
0.04%/lato — stessa convenzione di tutta la pipeline.
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

# ── Parametri strategia (fissi, non ottimizzati sul dataset) ───────────────
NY_OPEN_H, NY_OPEN_M = 13, 30      # inizio range, UTC
RANGE_MIN = 30                      # minuti del range di apertura
VP_BINS = 20                        # bin del volume profile
BREAKOUT_CUTOFF_H = 20              # entro le 20:00 UTC deve avvenire il breakout
ABSORPTION_CUTOFF_H = 22            # entro le 22:00 UTC deve avvenire il ritorno al POC
MAX_HOLD_5M = 96                    # 96 barre 5m = 8h, come la strategia ML RF validata
PIVOT_LR_1H = 6                     # left=right per i pivot 1H (target di liquidita')
RR_TESTS = [1.0, 2.0]
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("NY ORB + Volume Profile Absorption — Validation Pipeline")
w(SEP)

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 1m (volume profile) + 5m (execution) + 1H (ATR/pivot targets) …")
df_1m = fetch_binance_vision_klines("1m", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
df_5m = fetch_binance_vision_klines("5m", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
df_1h_raw = fetch_binance_vision_klines("1h", start_year=START_YEAR, start_month=1,
                                         workers=6, verbose=False)
df1h = add_indicators(df_1h_raw)
print(f"  1m: {len(df_1m):,}   5m: {len(df_5m):,}   1H: {len(df1h):,}  "
      f"(loaded in {time.time()-t0:.0f}s)")

IDX1H = df1h.index
IDX1H_vals = IDX1H.values
ATR1H = df1h["atr_14"].values
trend1h = causal_trend_state(df1h["close"].values.astype(float),
                              df1h["high"].values.astype(float),
                              df1h["low"].values.astype(float),
                              PIVOT_LR_1H, PIVOT_LR_1H)
TARGET_HIGH = trend1h["target_high"].values
TARGET_LOW = trend1h["target_low"].values

IDX5M = df_5m.index
O5, H5, L5, C5 = (df_5m["open"].values.astype(float), df_5m["high"].values.astype(float),
                   df_5m["low"].values.astype(float), df_5m["close"].values.astype(float))
N5 = len(df_5m)

IDX1M = df_1m.index


# ── Volume Profile / POC ────────────────────────────────────────────────
def compute_poc(lows: np.ndarray, highs: np.ndarray, vols: np.ndarray,
                 range_lo: float, range_hi: float, n_bins: int) -> float:
    """POC = price bin (of n_bins between range_lo/range_hi) with the most
    volume, distributing each bar's volume proportionally to how much of
    its [low,high] overlaps each bin."""
    if range_hi <= range_lo:
        return np.nan
    edges = np.linspace(range_lo, range_hi, n_bins + 1)
    vol_bins = np.zeros(n_bins)
    for lo, hi, v in zip(lows, highs, vols):
        if v <= 0:
            continue
        if hi <= lo:
            idx = np.clip(np.searchsorted(edges, lo) - 1, 0, n_bins - 1)
            vol_bins[idx] += v
            continue
        bar_range = hi - lo
        lo_idx = max(0, np.searchsorted(edges, lo, side="right") - 1)
        hi_idx = min(n_bins - 1, np.searchsorted(edges, hi, side="left"))
        for i in range(lo_idx, hi_idx + 1):
            e_lo, e_hi = edges[i], edges[i + 1]
            overlap = max(0.0, min(hi, e_hi) - max(lo, e_lo))
            if overlap > 0:
                vol_bins[i] += v * overlap / bar_range
    poc_idx = int(np.argmax(vol_bins))
    return float((edges[poc_idx] + edges[poc_idx + 1]) / 2)


# ── Event construction: one candidate trade per calendar day ───────────────
print("\n[EVENTS] Scanning NY opening range -> breakout -> POC absorption, day by day …")
t1 = time.time()

day0 = IDX1M[0].normalize()
dayN = IDX1M[-1].normalize()
all_days = pd.date_range(day0, dayN, freq="D")

events = []
n_no_range = n_no_breakout = n_no_absorption = n_no_target = 0

for day in all_days:
    range_start = day + pd.Timedelta(hours=NY_OPEN_H, minutes=NY_OPEN_M)
    range_end = range_start + pd.Timedelta(minutes=RANGE_MIN)
    breakout_cutoff = day + pd.Timedelta(hours=BREAKOUT_CUTOFF_H)
    absorption_cutoff = day + pd.Timedelta(hours=ABSORPTION_CUTOFF_H)

    win_1m = df_1m.loc[range_start:range_end - pd.Timedelta(minutes=1)]
    if len(win_1m) < RANGE_MIN * 0.8:      # tolerate a few missing minutes
        n_no_range += 1
        continue
    range_hi = float(win_1m["high"].max())
    range_lo = float(win_1m["low"].min())
    if range_hi <= range_lo:
        n_no_range += 1
        continue

    poc = compute_poc(win_1m["low"].values.astype(float), win_1m["high"].values.astype(float),
                       win_1m["volume"].values.astype(float), range_lo, range_hi, VP_BINS)
    if np.isnan(poc):
        n_no_range += 1
        continue

    # breakout search on 5m bars [range_end, breakout_cutoff)
    seg = df_5m.loc[range_end:breakout_cutoff - pd.Timedelta(minutes=5)]
    if seg.empty:
        n_no_breakout += 1
        continue
    seg_c = seg["close"].values.astype(float)
    long_mask = seg_c > range_hi
    short_mask = seg_c < range_lo
    first_long = np.argmax(long_mask) if long_mask.any() else -1
    first_short = np.argmax(short_mask) if short_mask.any() else -1
    if first_long < 0 and first_short < 0:
        n_no_breakout += 1
        continue
    if first_long >= 0 and (first_short < 0 or first_long <= first_short):
        breakout_i = first_long
        direction = 1
    else:
        breakout_i = first_short
        direction = -1
    breakout_time = seg.index[breakout_i]

    # absorption search on 5m bars (breakout_time, absorption_cutoff)
    seg2 = df_5m.loc[breakout_time + pd.Timedelta(minutes=5):
                      absorption_cutoff - pd.Timedelta(minutes=5)]
    if seg2.empty:
        n_no_absorption += 1
        continue
    touch_mask = (seg2["low"].values.astype(float) <= poc) & (seg2["high"].values.astype(float) >= poc)
    if not touch_mask.any():
        n_no_absorption += 1
        continue
    entry_i = int(np.argmax(touch_mask))
    entry_time = seg2.index[entry_i]
    entry_price = poc

    # target = nearest unbroken 1H pivot, as of the most recent CLOSED 1H bar
    h1_idx = np.searchsorted(IDX1H_vals, np.datetime64(entry_time), side="right") - 1
    if h1_idx < 0:
        n_no_target += 1
        continue
    target = TARGET_HIGH[h1_idx] if direction == 1 else TARGET_LOW[h1_idx]
    atr_at_entry = ATR1H[h1_idx]
    if np.isnan(target) or atr_at_entry <= 0:
        n_no_target += 1
        continue
    if (direction == 1 and target <= entry_price) or (direction == -1 and target >= entry_price):
        n_no_target += 1
        continue
    target_dist = abs(target - entry_price)
    # floor to avoid degenerate near-zero stop distances
    target_dist = max(target_dist, 0.05 * atr_at_entry)

    entry_5m_idx = IDX5M.searchsorted(entry_time)
    events.append(dict(day=day, direction=direction, entry_time=entry_time,
                        entry_price=entry_price, target=target, target_dist=target_dist,
                        entry_5m_idx=entry_5m_idx, poc=poc, range_hi=range_hi, range_lo=range_lo))

print(f"  done in {time.time()-t1:.0f}s")
w(f"\n[EVENTS] {len(all_days)} giorni scansionati -> {len(events)} trade candidati")
w(f"  scartati: no-range={n_no_range}  no-breakout={n_no_breakout}  "
  f"no-absorption={n_no_absorption}  no-target={n_no_target}")


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
        i0 = ev["entry_5m_idx"]
        out = "time"
        exit_price = None
        for k in range(1, MAX_HOLD_5M + 1):
            j = i0 + k
            if j >= N5: break
            hk, lk = H5[j], L5[j]
            if d == 1:
                hit_sl = lk <= sl
                hit_tp = hk >= target
            else:
                hit_sl = hk >= sl
                hit_tp = lk <= target
            if hit_sl:            # SL-first assumption when both touched same bar
                out = "sl"; exit_price = sl; break
            if hit_tp:
                out = "tp"; exit_price = target; break
        if exit_price is None:
            j = min(i0 + MAX_HOLD_5M, N5 - 1)
            exit_price = C5[j]
        if out == "tp": n_tp += 1
        elif out == "sl": n_sl += 1
        else: n_time += 1

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


holdout_events = [e for e in events if e["entry_time"] >= CUTOFF]
w(f"\n  Holdout genuino 2025-2026: {len(holdout_events)} trade")

dsr_full = []
dsr_holdout = []
per_rr_results = {}

for rr in RR_TESTS:
    w(f"\n{SEP}")
    w(f"RR = {rr:.1f} : 1   (stop = target_dist / {rr:.1f})")
    w(SEP)

    res = run_bt(events, rr)
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    be_wr = 1.0 / (1.0 + rr)
    pval = st.binomtest(int(res["wr"] * res["n"]), res["n"], be_wr, alternative="greater").pvalue if res["n"] else 1.0

    w(f"\n  FULL-SAMPLE: n={res['n']}  wr={res['wr']:.1%} (BE_teorico={be_wr:.1%}, "
      f"p={pval:.4f})  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    year_evs: dict[int, list] = {}
    for e in events:
        year_evs.setdefault(e["entry_time"].year, []).append(e)
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

    per_rr_results[rr] = dict(full=res, holdout=hres)
    dsr_full.append(dict(rr=rr, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_holdout.append(dict(rr=rr, net_pnls=hres["net_pnls"], ret=hres["ret"]))

# ── DSR family (RR=1 vs RR=2) ───────────────────────────────────────────
w(f"\n{SEP}")
w("DSR — famiglia N=2 (RR 1:1 vs RR 2:1)")
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

# ── Slippage sensitivity on the best RR (by full-sample DSR) ──────────────
best_rr = max(dsr_full, key=lambda r: r["dsr"])["rr"]
w(f"\n{SEP}")
w(f"Slippage sensitivity — RR migliore per DSR = {best_rr:.1f}:1")
w(SEP)
w(f"\n  {'Slippage':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MDD%':>7}  "
  f"{'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_TESTS_BPS:
    slip = bps / 10_000.0
    for scope_name, evs in [("full-sample", events), ("holdout", holdout_events)]:
        res = run_bt(evs, best_rr, slippage_pct=slip)
        mc = mc_summary(res["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {res['n']:>6}  {res['ret']:>+7.1f}%  "
          f"{res['wr']:>5.1%}  {res['mdd']:>6.1f}%  {mc['p_profit']:>6.3f}  {mc['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/ny_orb_vp_absorption.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# NY ORB + Volume Profile Absorption — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
