#!/usr/bin/env python3
"""
create_vwap_mr_v1_makerfee_report.py
=======================================
Ri-verifica della strategia VWAP Mean-Reversion v1 (target=tocco VWAP,
stop largo fisso 2xATR — l'unica costruzione, tra le 5 provate, che
preserva il vero win rate 61-66%) con la struttura di fee REALE di Bybit
per i derivati (fornita dall'utente), invece dell'assunzione generica
"0.04%/lato taker" usata in tutta la sessione:

  Bybit Derivatives (Futures/Perpetuals): maker 0.0200%, taker 0.0550%

Motivazione: sia l'entry (soglia Z-score nota in anticipo) sia l'uscita
(tocco di un livello VWAP noto in anticipo) sono segnali "a livello di
prezzo", non reazioni immediate — esattamente il tipo di segnale
eseguibile con ordini limit (maker) invece che ordini a mercato (taker).
Se eseguibile in maker su entrambi i lati, la fee round-trip scende da
0.08% (assunzione originale) a 0.04% — la metà.

Vengono testati 4 scenari di fee (stessa pipeline WFO causale, IS-scan
Z_ENTRY×filtro, per ciascuno — la selezione stessa può cambiare con la
fee, quindi si rifà l'intero walk-forward per scenario, non solo un
ricalcolo del P&L):

  1. Taker reale Bybit     : 0.0550% + 0.0550% = 0.1100% round-trip
  2. Assunzione originale  : 0.0400% + 0.0400% = 0.0800% round-trip (per confronto)
  3. Mista (entry a mercato, exit a limit): 0.0550% + 0.0200% = 0.0750%
  4. Piena maker (limit su entrambi i lati): 0.0200% + 0.0200% = 0.0400%

Tutto il resto è identico a create_vwap_mr_report.py (v1).
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
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2

Z_ENTRY_GRID = [1.0, 1.5, 2.0]
FILTER_VARIANTS = ["SIDEWAYS_ONLY", "NO_FILTER"]
SL_ATR_MULT = 2.0

FEE_SCENARIOS = [
    ("Taker reale Bybit (0.055%+0.055%)",   0.00055, 0.00055),
    ("Assunzione originale sessione (0.04%+0.04%)", 0.00040, 0.00040),
    ("Mista: entry taker / exit maker (0.055%+0.02%)", 0.00055, 0.00020),
    ("Piena maker: limit su entrambi i lati (0.02%+0.02%)", 0.00020, 0.00020),
]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("VWAP Mean-Reversion v1 — re-check con fee reali Bybit derivatives")
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
        evs.append(dict(i=i, d=d, ep=CL[i]))
    return evs


def run_bt(evs, fee_entry_pct, fee_exit_pct, slippage_pct=0.0):
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0
    for ev in evs:
        i, d, ep = ev["i"], ev["d"], ev["ep"]
        sl = ep - d * SL_ATR_MULT * ATR[i]
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

        stop_dist = abs(ep - sl)
        if stop_dist <= 0: continue
        risk = INIT_CAP * RISK_PCT
        units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
        notional_entry = units * ep
        notional_exit = units * exit_price
        fill_ep = ep * (1 + d * slippage_pct)
        fill_xp = exit_price * (1 - d * slippage_pct)
        pnl = (units * (fill_xp - fill_ep) * d
               - fee_entry_pct * notional_entry - fee_exit_pct * notional_exit)
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


def run_wfo(fee_entry_pct, fee_exit_pct):
    """Full causal WFO (HMM fit + IS-scan Z_ENTRY x filter, Sharpe-selected
    on IS data under THIS fee scenario) — rerun per scenario since the fee
    can shift which combo the IS-scan prefers, not just rescale P&L after
    the fact."""
    all_oos_evs = []
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
                is_res = run_bt(is_evs, fee_entry_pct, fee_exit_pct)
                sharpe = trade_level_sharpe(is_res["net_pnls"])
                if best is None or sharpe > best["sharpe"]:
                    best = dict(z_entry=z_entry, filt=filt, sharpe=sharpe, n_is=len(is_evs))

        if best is None:
            continue
        oos_evs = build_events(idx_oos, best["z_entry"], best["filt"], state_oos_full)
        all_oos_evs.extend(oos_evs)
    return all_oos_evs


# ── Run all 4 fee scenarios ─────────────────────────────────────────────
print("\n[RUN] Full causal WFO per fee scenario …")
for label, fee_entry, fee_exit in FEE_SCENARIOS:
    t1 = time.time()
    all_oos_evs = run_wfo(fee_entry, fee_exit)
    res = run_bt(all_oos_evs, fee_entry, fee_exit)
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])

    w(f"\n{SEP}")
    w(f"Scenario: {label}  (round-trip = {(fee_entry+fee_exit)*100:.3f}%)")
    w(SEP)
    w(f"\n  FULL-SAMPLE: n={res['n']}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    year_evs: dict[int, list] = {}
    for e in all_oos_evs:
        year_evs.setdefault(IDX1H[e["i"]].year, []).append(e)
    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    for yr in sorted(year_evs):
        yevs = year_evs[yr]
        if len(yevs) < 5: continue
        yres = run_bt(yevs, fee_entry, fee_exit)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    holdout_evs = [e for e in all_oos_evs if IDX1H[e["i"]] >= CUTOFF]
    hres = run_bt(holdout_evs, fee_entry, fee_exit)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")
    print(f"  ({label}: done in {time.time()-t1:.0f}s)")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vwap_mr_v1_makerfee.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# VWAP Mean-Reversion v1 — re-check con fee reali Bybit derivatives\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
