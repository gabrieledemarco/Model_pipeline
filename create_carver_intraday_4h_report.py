#!/usr/bin/env python3
"""
create_carver_intraday_4h_report.py
======================================
Nuovo requisito di sessione: le strategie live devono usare timeframe
INTRADAY — Carver Breakout pool e TSMOM-sign pool (validate su 1D,
DSR=1.000 full-sample+holdout in `reports/carver_rules.md` e
`reports/tsmom_strategy.md`) sono state ESCLUSE dal set live per questo
motivo, non perché invalidate nel merito.

Questo script porta ESATTAMENTE la stessa infrastruttura (calibrazione
causale del forecast scalar, vol-targeting a target 20% annuo, costi sul
turnover) dal timeframe 1D al **4H** (intraday, 6 barre/giorno) — stesso
edge economico, stessa filosofia "mai una sola velocità" (pool di
lookback), lookback riscalati in barre 4H per preservare la stessa
finestra di CALENDARIO (es. canale Donchian 10 giorni = 60 barre 4H).

Domanda aperta (non scontata): il ribilanciamento 6× più frequente
(ogni 4H invece che ogni giorno) paga il turnover 6× più spesso — la
domanda è se l'edge sopravvive a questo aumento di frizione.

Frizioni Bybit obbligatorie sul turnover (0.055%+0.015%=0.07%/lato).
Validazione: per-anno, holdout 2025-2026 genuino, Monte Carlo
i.i.d.+block, DSR family sui 3 candidati (Breakout 4H, TSMOM-sign 4H,
pool combinato), slippage-stress.
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
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
TARGET_ANNUAL_VOL = 0.20
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
FRICTION = FEE_TAKER + SLIPPAGE_BASE
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

BARS_PER_DAY_4H = 6
VOL_SPAN_4H = 25 * BARS_PER_DAY_4H          # equivalente calendario di span=25 giorni
FORECAST_CAP = 20.0
FORECAST_TARGET_ABS = 10.0
# Lookback in giorni di calendario -> barre 4H (stessa finestra temporale del report 1D)
BREAKOUT_NS_DAYS = [10, 20, 40, 80, 160, 320]
TSMOM_LOOKBACKS_DAYS = [30, 60, 90, 120, 252]
BREAKOUT_NS = [d * BARS_PER_DAY_4H for d in BREAKOUT_NS_DAYS]
TSMOM_LOOKBACKS = [d * BARS_PER_DAY_4H for d in TSMOM_LOOKBACKS_DAYS]
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Carver Breakout & TSMOM-sign — porting INTRADAY a 4H su BTCUSDT")
w(SEP)
w("\nStessa infrastruttura di calibrazione/vol-targeting già validata su 1D,")
w("qui a 4H (intraday) — lookback riscalati in barre per preservare la")
w("stessa finestra di calendario. Costi sul turnover, ribilanciamento 4H.")

t0 = time.time()
print("\n[DATA] Loading 4H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df4h = raw["4H"]
IDX = df4h.index
N = len(df4h)
CL = df4h["close"].values.astype(float)
print(f"  4H: {N:,} bars  ({IDX[0]} → {IDX[-1]})  (loaded in {time.time()-t0:.0f}s)")

price = pd.Series(CL, index=IDX)
bar_ret = price.diff()
price_vol = bar_ret.ewm(span=VOL_SPAN_4H, min_periods=VOL_SPAN_4H).std()
annualized_price_vol = (price_vol * np.sqrt(365 * BARS_PER_DAY_4H)).values
price_vol_v = price_vol.values


def calibrate_forecast(raw_arr):
    raw_s = pd.Series(raw_arr)
    abs_expanding_mean = raw_s.abs().expanding(min_periods=60 * BARS_PER_DAY_4H).mean()
    scalar = FORECAST_TARGET_ABS / abs_expanding_mean.replace(0, np.nan)
    fcst = (raw_s * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
    return fcst.values


# ── Breakout (Donchian) ────────────────────────────────────────────────────
print("[BREAKOUT-4H] Computing 6 N variants + pool …")
breakout_forecasts = {}
for Nb in BREAKOUT_NS:
    roll_max = price.rolling(Nb, min_periods=Nb).max()
    roll_min = price.rolling(Nb, min_periods=Nb).min()
    mid = (roll_max + roll_min) / 2.0
    half_range = (roll_max - roll_min) / 2.0
    raw_b = np.where(half_range.values > 0, 40.0 * (price.values - mid.values) / half_range.values, np.nan)
    raw_b_s = pd.Series(raw_b).ewm(span=max(Nb // 4, 2), min_periods=max(Nb // 4, 2)).mean()
    breakout_forecasts[f"BRK({Nb})"] = raw_b_s.clip(-FORECAST_CAP, FORECAST_CAP).values
breakout_pool = np.nanmean(np.column_stack(list(breakout_forecasts.values())), axis=1)

# ── TSMOM-sign ───────────────────────────────────────────────────────────
print("[TSMOM-4H] Computing 5 lookback sign variants + pool …")
tsmom_forecasts = []
for L in TSMOM_LOOKBACKS:
    past_price = price.shift(L)
    raw_sign = np.sign(price.values - past_price.values)
    tsmom_forecasts.append(calibrate_forecast(raw_sign))
tsmom_sign_pool = np.nanmean(np.column_stack(tsmom_forecasts), axis=1)

grand_pool = np.nanmean(np.column_stack([breakout_pool, tsmom_sign_pool]), axis=1)

CANDIDATES = {
    "Breakout pool 4H": breakout_pool,
    "TSMOM-sign pool 4H": tsmom_sign_pool,
    "Grand pool (BRK+TSMOM) 4H": grand_pool,
}


def run_bt(forecast, fee=FRICTION, slippage_extra=0.0):
    n = len(forecast)
    valid = np.isfinite(forecast) & (annualized_price_vol > 0)
    raw_units = np.where(valid, (forecast / FORECAST_TARGET_ABS) * INIT_CAP * TARGET_ANNUAL_VOL /
                          np.where(annualized_price_vol > 0, annualized_price_vol, np.nan), 0.0)
    max_units = MAX_LEV * INIT_CAP / CL
    units = np.clip(raw_units, -max_units, max_units)
    units = np.where(valid, units, 0.0)

    tot_fee = fee + slippage_extra
    pnl_series = np.zeros(n)
    cost_series = np.zeros(n)
    prev_units = 0.0
    for t in range(1, n):
        cost_series[t] = tot_fee * abs(units[t] - prev_units) * CL[t]
        pnl_series[t] = prev_units * (CL[t] - CL[t - 1])
        prev_units = units[t]
    net_bar = pnl_series - cost_series
    net_bar[0] = 0.0

    cap = INIT_CAP + np.cumsum(net_bar)
    peak = np.maximum.accumulate(np.concatenate([[INIT_CAP], cap]))[1:]
    mdd = ((cap - peak) / peak).min() if n > 1 else 0.0
    ret_pct = (cap[-1] / INIT_CAP - 1) * 100 if n > 1 else 0.0
    net_pnls_list = net_bar[1:].tolist()
    wins = sum(1 for x in net_pnls_list if x > 0)
    wr = wins / len(net_pnls_list) if net_pnls_list else 0.0
    return dict(n=len(net_pnls_list), wr=wr, ret=ret_pct, mdd=mdd * 100,
                net_pnls=net_pnls_list, dates=IDX[1:], cap=cap)


def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=15):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


dsr_full = []
dsr_holdout = []
for name, fcst in CANDIDATES.items():
    w(f"\n{SEP}")
    w(name)
    w(SEP)

    res = run_bt(fcst)
    idx_dates = IDX[1:]
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL-SAMPLE: n_bars={res['n']}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>7}  {'Ret%':>8}  {'WR':>6}")
    yrs = pd.Series(idx_dates.year, index=range(len(idx_dates)))
    for yr in sorted(yrs.unique()):
        mask = (yrs == yr).values
        pnls_yr = [p for p, m in zip(res["net_pnls"], mask) if m]
        if len(pnls_yr) < 30: continue
        ret_yr = sum(pnls_yr) / INIT_CAP * 100
        wr_yr = sum(1 for x in pnls_yr if x > 0) / len(pnls_yr)
        w(f"    {yr:>6}  {len(pnls_yr):>7}  {ret_yr:>+7.1f}%  {wr_yr:>5.1%}")

    hold_mask = idx_dates >= CUTOFF
    pnls_h = [p for p, m in zip(res["net_pnls"], hold_mask) if m]
    ret_h = sum(pnls_h) / INIT_CAP * 100
    wr_h = sum(1 for x in pnls_h if x > 0) / len(pnls_h) if pnls_h else 0.0
    hmc = mc_summary(pnls_h)
    hmc_blk = mc_block_summary(pnls_h)
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={len(pnls_h)}  wr={wr_h:.1%}  ret={ret_h:+.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    dsr_full.append(dict(name=name, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_holdout.append(dict(name=name, net_pnls=pnls_h, ret=ret_h))

w(f"\n{SEP}")
w(f"DSR — famiglia N={len(CANDIDATES)}")
w(SEP)
deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
w(f"\n  Full-sample:")
w(f"  {'Candidate':>28}  {'n':>7}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_full:
    w(f"  {r['name']:>28}  {len(r['net_pnls']):>7}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
w(f"\n  Holdout 2025-2026:")
w(f"  {'Candidate':>28}  {'n':>7}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_holdout:
    w(f"  {r['name']:>28}  {len(r['net_pnls']):>7}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

holdout_dsr_by_name = {r["name"]: r["dsr"] for r in dsr_holdout}
holdout_sharpe_by_name = {r["name"]: r["sharpe_hat"] for r in dsr_holdout}
best_name = max(dsr_full, key=lambda r: (r["dsr"], holdout_dsr_by_name[r["name"]], holdout_sharpe_by_name[r["name"]]))["name"]
w(f"\n{SEP}")
w(f"Slippage-stress — migliore per DSR = {best_name}")
w(SEP)
best_fcst = CANDIDATES[best_name]
idx_dates = IDX[1:]
hold_mask = idx_dates >= CUTOFF
w(f"\n  {'Extra slip':>10}  {'Scope':>10}  {'n':>7}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_STRESS_BPS:
    extra = bps / 10_000.0
    r = run_bt(best_fcst, slippage_extra=extra)
    mc_f = mc_summary(r["net_pnls"])
    w(f"  {bps:>9}bps  {'full-sample':>10}  {r['n']:>7}  {r['ret']:>+7.1f}%  {r['wr']:>5.1%}  "
      f"{mc_f['p_profit']:>7.3f}  {mc_f['p_ruin']:>7.3f}")
    pnls_h = [p for p, m in zip(r["net_pnls"], hold_mask) if m]
    ret_h = sum(pnls_h) / INIT_CAP * 100
    wr_h = sum(1 for x in pnls_h if x > 0) / len(pnls_h) if pnls_h else 0.0
    mc_h = mc_summary(pnls_h)
    w(f"  {bps:>9}bps  {'holdout':>10}  {len(pnls_h):>7}  {ret_h:>+7.1f}%  {wr_h:>5.1%}  "
      f"{mc_h['p_profit']:>7.3f}  {mc_h['p_ruin']:>7.3f}")

w(f"\n{SEP}")
w("[DONE]")
w(SEP)

Path("reports").mkdir(exist_ok=True)
with open("reports/carver_intraday_4h.md", "w") as f:
    f.write("# Carver Breakout & TSMOM-sign — porting intraday a 4H\n\n")
    f.write("```\n" + "\n".join(report_lines) + "\n```\n")
print(f"\n[DONE] reports/carver_intraday_4h.md   (total runtime {time.time()-t0:.0f}s)")
