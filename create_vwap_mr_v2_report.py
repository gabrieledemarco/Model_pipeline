#!/usr/bin/env python3
"""
create_vwap_mr_v2_report.py
==============================
Fix della strategia VWAP Mean-Reversion (create_vwap_mr_report.py) dopo la
diagnosi: il win rate era alto e stabilissimo (61-66% ogni anno 2020-2026 —
un edge direzionale reale) ma lo stop fisso a 2×ATR era ~1.83× più largo
della tipica distanza di reversione al VWAP, spingendo il win rate di
breakeven (~64.7%) sopra quello osservato (62.9%) — perdita strutturale da
rapporto rischio/rendimento male impostato, non da assenza di segnale.

FIX: lo stop non è più un multiplo fisso di ATR, ma è derivato dalla STESSA
scala del target (come nella strategia NY-ORB-VP):

  target_dist = |entry_price - VWAP[entry]|      (= |z_entry| × sigma_vwap)
  stop_dist   = target_dist / RR
  SL = entry - direction × stop_dist

testato per RR = 1.0, 1.5, 2.0 (RR=1.5 è il valore intermedio suggerito
dalla diagnosi: breakeven teorico ≈ 1/(1+1.5) = 40%, ben sotto il 62-66%
di win rate osservato nella v1).

Tutto il resto dell'architettura è invariato rispetto a v1: VWAP di sessione
(reset giornaliero UTC) + z-score, HMM regime gate causale (SIDEWAYS_ONLY
vs NO_FILTER), IS-scan causale (Z_ENTRY × filtro) per ogni finestra WFO,
target = ritorno al VWAP corrente, time-stop a fine giornata UTC.

Poiché qui RR è un asse esterno testato come 3 backtest completi separati
(non riselezionato causalmente finestra per finestra come Z_ENTRY/filtro),
si applica DSR family N=3 sui 3 RR per correggere la selection bias del
"migliore dei 3" — stessa disciplina usata per gli orizzonti ML e per la
strategia NY-ORB-VP.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, trade_level_sharpe,
    deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0004
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2

Z_ENTRY_GRID = [1.0, 1.5, 2.0]
FILTER_VARIANTS = ["SIDEWAYS_ONLY", "NO_FILTER"]
RR_GRID = [1.0, 1.5, 2.0]
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("VWAP Mean-Reversion v2 — stop a RR proporzionale al target (fix v1)")
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

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
ATR = np.where(df1h["atr_14"].values > 0, df1h["atr_14"].values, 1.0)

# ── VWAP di sessione (reset giornaliero UTC), cumulativo, causale ─────────
dates = IDX1H.normalize().values
day_change = np.r_[True, dates[1:] != dates[:-1]]
day_id = np.cumsum(day_change) - 1

tp = (HI + LO + CL) / 3.0
tmp = pd.DataFrame({"day_id": day_id, "pv": tp * VOL, "pv2": tp * tp * VOL, "vol": VOL})
g = tmp.groupby("day_id")
cum_pv = g["pv"].cumsum().values
cum_pv2 = g["pv2"].cumsum().values
cum_v = g["vol"].cumsum().values
bar_in_day = g.cumcount().values

VWAP = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
var_vwap = np.maximum(cum_pv2 / np.maximum(cum_v, 1e-9) - VWAP ** 2, 0.0)
SIGMA = np.sqrt(var_vwap)
Z = np.where((SIGMA > 1e-9) & (bar_in_day >= 2), (CL - VWAP) / np.maximum(SIGMA, 1e-9), np.nan)

hour_arr = IDX1H.hour.values
bars_to_dayend = 23 - hour_arr

print(f"[VWAP] valid z-score bars: {np.isfinite(Z).sum():,} / {N1H:,}")


# ── Event / backtest helpers ───────────────────────────────────────────────
def build_events(idx_range, z_entry, filter_variant, hmm_state):
    evs = []
    for i in idx_range:
        if not np.isfinite(Z[i]) or bars_to_dayend[i] < 1 or ATR[i] <= 0:
            continue
        if Z[i] < -z_entry:
            d = 1
        elif Z[i] > z_entry:
            d = -1
        else:
            continue
        if filter_variant == "SIDEWAYS_ONLY":
            if hmm_state[i] != 1:
                continue
        target_dist = max(abs(CL[i] - VWAP[i]), 0.0002 * CL[i])   # tiny safety floor
        evs.append(dict(i=i, d=d, ep=CL[i], target_dist=target_dist))
    return evs


def run_bt(evs, rr, slippage_pct=0.0):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0
    for ev in evs:
        i, d, ep, target_dist = ev["i"], ev["d"], ev["ep"], ev["target_dist"]
        stop_dist = target_dist / rr
        sl = ep - d * stop_dist
        hold = min(bars_to_dayend[i], N1H - 1 - i)
        out = "time"; exit_price = CL[min(i + hold, N1H - 1)]
        for k in range(1, hold + 1):
            j = i + k
            hk, lk, vwj = HI[j], LO[j], VWAP[j]
            if d == 1:
                if lk <= sl: out = "sl"; exit_price = sl; break
                if not np.isnan(vwj) and hk >= vwj >= lk:
                    out = "tp"; exit_price = vwj; break
            else:
                if hk >= sl: out = "sl"; exit_price = sl; break
                if not np.isnan(vwj) and hk >= vwj >= lk:
                    out = "tp"; exit_price = vwj; break
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

def mc_block_summary(pnls, block_size=15):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


def wf_dates(idx):
    t0_ = idx[0]; windows = []
    while True:
        tr_s = t0_; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > idx[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0_ += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
print(f"[WFO] {len(WF_WINDOWS)} windows")


def run_wfo_for_rr(rr):
    """Full causal WFO (HMM fit + IS-scan Z_ENTRY x filter, Sharpe-selected on
    IS data only) for a given fixed RR. Returns list of OOS trade events."""
    all_oos_evs = []
    selection_log = []
    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_is = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 500 or len(idx_oos) < 50:
            continue

        model, sorted_idx = fit_hmm(df1h.iloc[idx_is], n_states=3, random_state=42)
        hmm_is = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_is])
        hmm_oos = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_oos])

        state_is_full = np.full(N1H, np.nan)
        state_is_full[idx_is] = hmm_is["hmm_state"].values
        state_oos_full = np.full(N1H, np.nan)
        state_oos_full[idx_oos] = hmm_oos["hmm_state"].values

        best = None
        for z_entry in Z_ENTRY_GRID:
            for filt in FILTER_VARIANTS:
                is_evs = build_events(idx_is, z_entry, filt, state_is_full)
                if len(is_evs) < 15:
                    continue
                is_res = run_bt(is_evs, rr)
                sharpe = trade_level_sharpe(is_res["net_pnls"])
                if best is None or sharpe > best["sharpe"]:
                    best = dict(z_entry=z_entry, filt=filt, sharpe=sharpe, n_is=len(is_evs))

        if best is None:
            continue
        selection_log.append(dict(window_start=tr_s, **best))
        oos_evs = build_events(idx_oos, best["z_entry"], best["filt"], state_oos_full)
        all_oos_evs.extend(oos_evs)
    return all_oos_evs, selection_log


# ── Run WFO for each RR ────────────────────────────────────────────────────
print("\n[WFO] Fitting HMM + IS-scan per window, per RR …")
t1 = time.time()
rr_data = {}
for rr in RR_GRID:
    evs, sel_log = run_wfo_for_rr(rr)
    rr_data[rr] = dict(evs=evs, sel_log=sel_log)
    print(f"  RR={rr}: {len(evs)} OOS trade totali")
print(f"  done in {time.time()-t1:.0f}s")

dsr_full = []
dsr_holdout = []

for rr in RR_GRID:
    all_oos_evs = rr_data[rr]["evs"]
    sel_df = pd.DataFrame(rr_data[rr]["sel_log"])

    w(f"\n{SEP}")
    w(f"RR = {rr:.1f} : 1   (stop = target_dist / {rr:.1f})")
    w(SEP)
    w(f"  Z_ENTRY più scelto: {sel_df['z_entry'].value_counts().to_dict() if not sel_df.empty else {}}")
    w(f"  Filtro più scelto  : {sel_df['filt'].value_counts().to_dict() if not sel_df.empty else {}}")

    res = run_bt(all_oos_evs, rr)
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    be_wr = 1.0 / (1.0 + rr)
    w(f"\n  FULL-SAMPLE: n={res['n']}  wr={res['wr']:.1%} (BE_teorico={be_wr:.1%})  "
      f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    year_evs: dict[int, list] = {}
    for e in all_oos_evs:
        year_evs.setdefault(IDX1H[e["i"]].year, []).append(e)
    for yr in sorted(year_evs):
        yevs = year_evs[yr]
        if len(yevs) < 5: continue
        yres = run_bt(yevs, rr)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    holdout_evs = [e for e in all_oos_evs if IDX1H[e["i"]] >= CUTOFF]
    hres = run_bt(holdout_evs, rr)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    rr_data[rr]["holdout_evs"] = holdout_evs
    dsr_full.append(dict(rr=rr, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_holdout.append(dict(rr=rr, net_pnls=hres["net_pnls"], ret=hres["ret"]))

# ── DSR family across RR variants ──────────────────────────────────────────
w(f"\n{SEP}")
w("DSR — famiglia N=3 (RR 1:1, 1.5:1, 2:1)")
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
    for scope_name, evs in [("full-sample", rr_data[best_rr]["evs"]),
                             ("holdout", rr_data[best_rr]["holdout_evs"])]:
        r = run_bt(evs, best_rr, slippage_pct=slip)
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {r['mdd']:>6.1f}%  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vwap_mr_v2.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# VWAP Mean-Reversion v2 — RR-proportional stop\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
