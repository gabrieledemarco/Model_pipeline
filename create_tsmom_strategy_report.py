#!/usr/bin/env python3
"""
create_tsmom_strategy_report.py
===================================
Time-Series Momentum (TSMOM) — la costruzione classica di Moskowitz, Ooi
& Pedersen ("Time Series Momentum", J. Financial Economics 2012):
posizione = SEGNO del rendimento passato su un lookback L, sizing
vol-targeted separato dal segnale stesso. Distinta sia da EWMAC
(crossover di medie mobili) sia da Breakout (canale Donchian) già
testati in questa sessione — usa il segno del rendimento totale
realizzato, non una media mobile o un canale.

Stesso identico framework infrastrutturale di create_carver_rules_report.py
(forecast continui, vol-targeting, mark-to-market giornaliero, frizioni
sul turnover) — riuso deliberato per evitare di reintrodurre bug già
risolti nella formula di sizing.

── TSMOM-sign (canonica) ────────────────────────────────────────────────
  raw[t] = sign(price[t] - price[t-L])   per L in [30,60,90,120,252] gg
  forecast[t] = calibrato a target=10 (banale per un segnale ±1: risulta
  in un forecast a gradini ±10) — pool = media delle 5 velocità.

── TSMOM-magnitude (variante continua, stessa filosofia vol_adj di EWMAC) ──
  raw[t] = (price[t]-price[t-L]) / price_vol[t]   — non solo il segno ma
  l'ampiezza del momentum relativa alla volatilità, calibrato allo stesso
  modo di EWMAC. Pool = media delle 5 velocità.

── Combined ──────────────────────────────────────────────────────────────
  media di TSMOM-sign pool e TSMOM-magnitude pool.

Validazione: per-anno, holdout 2025-2026 genuino, Monte Carlo
i.i.d.+block, DSR family sui 3 candidati finali, slippage-stress.
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

VOL_SPAN = 25
FORECAST_CAP = 20.0
FORECAST_TARGET_ABS = 10.0
TSMOM_LOOKBACKS = [30, 60, 90, 120, 252]
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Time-Series Momentum (Moskowitz/Ooi/Pedersen 2012) su BTCUSDT — Validation Pipeline")
w(SEP)
w("\nCostruzione classica: posizione = segno del rendimento passato su")
w("lookback L, vol-targeting separato. Pool su 5 lookback (30-252 giorni).")
w("Stesso framework infrastrutturale (forecast continui, mark-to-market,")
w("frizioni sul turnover) già validato per Carver Breakout pool.")

t0 = time.time()
print("\n[DATA] Loading 1D …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1d = raw["1D"]
IDX = df1d.index
N = len(df1d)
CL = df1d["close"].values.astype(float)
print(f"  1D: {N:,} bars  ({IDX[0].date()} → {IDX[-1].date()})  (loaded in {time.time()-t0:.0f}s)")

price = pd.Series(CL, index=IDX)
daily_ret = price.diff()
price_vol = daily_ret.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
annualized_price_vol = (price_vol * np.sqrt(365)).values
price_vol_v = price_vol.values


def calibrate_forecast(raw_arr):
    raw_s = pd.Series(raw_arr)
    abs_expanding_mean = raw_s.abs().expanding(min_periods=60).mean()
    scalar = FORECAST_TARGET_ABS / abs_expanding_mean.replace(0, np.nan)
    fcst = (raw_s * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
    return fcst.values


# ── TSMOM-sign ────────────────────────────────────────────────────────────
print("[TSMOM-sign] Computing 5 lookback variants + pool …")
sign_forecasts = {}
for L in TSMOM_LOOKBACKS:
    past_price = price.shift(L)
    raw_sign = np.sign(price.values - past_price.values)
    raw_sign = np.where(np.isfinite(past_price.values), raw_sign, np.nan)
    sign_forecasts[f"TSMOM-sign({L})"] = calibrate_forecast(raw_sign)
tsmom_sign_pool = np.nanmean(np.column_stack(list(sign_forecasts.values())), axis=1)

# ── TSMOM-magnitude ───────────────────────────────────────────────────────
print("[TSMOM-magnitude] Computing 5 lookback variants + pool …")
mag_forecasts = {}
for L in TSMOM_LOOKBACKS:
    past_price = price.shift(L)
    raw_mag = (price.values - past_price.values) / np.where(price_vol_v > 0, price_vol_v, np.nan)
    raw_mag = np.where(np.isfinite(past_price.values), raw_mag, np.nan)
    mag_forecasts[f"TSMOM-mag({L})"] = calibrate_forecast(raw_mag)
tsmom_mag_pool = np.nanmean(np.column_stack(list(mag_forecasts.values())), axis=1)

# ── Combined ───────────────────────────────────────────────────────────────
combined = np.nanmean(np.column_stack([tsmom_sign_pool, tsmom_mag_pool]), axis=1)

CANDIDATES = {
    "TSMOM-sign pool": tsmom_sign_pool,
    "TSMOM-magnitude pool": tsmom_mag_pool,
    "Combined (sign+mag)": combined,
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
    net_daily = pnl_series - cost_series
    net_daily[0] = 0.0

    cap = INIT_CAP + np.cumsum(net_daily)
    peak = np.maximum.accumulate(np.concatenate([[INIT_CAP], cap]))[1:]
    mdd = ((cap - peak) / peak).min() if n > 1 else 0.0
    ret_pct = (cap[-1] / INIT_CAP - 1) * 100 if n > 1 else 0.0
    net_pnls_list = net_daily[1:].tolist()
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


# ── Diagnostica: singole velocità ─────────────────────────────────────────
w(f"\n{SEP}")
w("Diagnostica — singoli lookback TSMOM-sign e TSMOM-magnitude (mai tradati da soli)")
w(SEP)
w(f"\n  {'Rule':>20}  {'Ret full%':>10}  {'WR':>6}  {'MDD%':>7}")
for name, fcst in list(sign_forecasts.items()) + list(mag_forecasts.items()):
    r = run_bt(fcst)
    w(f"  {name:>20}  {r['ret']:>+9.1f}%  {r['wr']:>5.1%}  {r['mdd']:>6.1f}%")

# ── Candidati finali: full-sample + per-anno + holdout ────────────────────
dsr_full = []
dsr_holdout = []
for name, fcst in CANDIDATES.items():
    w(f"\n{SEP}")
    w(f"{name}")
    w(SEP)
    res = run_bt(fcst)
    mask_full = np.isfinite(fcst)
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL-SAMPLE: n={res['n']}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"      {'Year':>6}  {'Ret%':>8}  {'WR':>6}")
    dates = res["dates"]; pnls = np.array(res["net_pnls"])
    years = pd.DatetimeIndex(dates).year
    for yr in sorted(set(years)):
        m = years == yr
        ypnls = pnls[m]
        if len(ypnls) < 30: continue
        yret = ypnls.sum() / INIT_CAP * 100
        ywr = (ypnls > 0).mean()
        w(f"      {yr:>6}  {yret:>+7.1f}%  {ywr:>5.1%}")

    holdout_mask = pd.DatetimeIndex(dates) >= CUTOFF
    h_pnls = pnls[holdout_mask].tolist()
    h_n = len(h_pnls)
    h_ret = sum(h_pnls) / INIT_CAP * 100 if h_n else 0.0
    h_wr = sum(1 for p in h_pnls if p > 0) / h_n if h_n else 0.0
    hmc = mc_summary(h_pnls); hmc_blk = mc_block_summary(h_pnls)
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={h_n}  wr={h_wr:.1%}  ret={h_ret:+.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    dsr_full.append(dict(name=name, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_holdout.append(dict(name=name, net_pnls=h_pnls, ret=h_ret))

w(f"\n{SEP}")
w(f"DSR — famiglia N={len(CANDIDATES)} (3 candidati finali)")
w(SEP)
deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
w(f"\n  Full-sample:")
w(f"  {'Candidate':>26}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_full:
    w(f"  {r['name']:>26}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
w(f"\n  Holdout 2025-2026:")
w(f"  {'Candidate':>26}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_holdout:
    w(f"  {r['name']:>26}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

holdout_dsr_by_name = {r["name"]: r["dsr"] for r in dsr_holdout}
holdout_sharpe_by_name = {r["name"]: r["sharpe_hat"] for r in dsr_holdout}
best_name = max(dsr_full, key=lambda r: (r["dsr"], holdout_dsr_by_name[r["name"]],
                                          holdout_sharpe_by_name[r["name"]]))["name"]
w(f"\n{SEP}")
w(f"Slippage-stress — migliore per DSR = {best_name}")
w(SEP)
best_fcst = CANDIDATES[best_name]
best_res_full = run_bt(best_fcst)
best_dates = best_res_full["dates"]
holdout_mask = pd.DatetimeIndex(best_dates) >= CUTOFF
w(f"\n  {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_STRESS_BPS:
    extra = bps / 10_000.0
    r = run_bt(best_fcst, slippage_extra=extra)
    pnls_arr = np.array(r["net_pnls"])
    m = mc_summary(r["net_pnls"])
    w(f"  {bps:>7}bps  {'full-sample':>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  {r['wr']:>5.1%}  "
      f"{m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")
    h_pnls2 = pnls_arr[holdout_mask].tolist()
    h_ret2 = sum(h_pnls2) / INIT_CAP * 100 if h_pnls2 else 0.0
    h_wr2 = sum(1 for p in h_pnls2 if p > 0) / len(h_pnls2) if h_pnls2 else 0.0
    hm2 = mc_summary(h_pnls2)
    w(f"  {bps:>7}bps  {'holdout':>10}  {len(h_pnls2):>6}  {h_ret2:>+7.1f}%  {h_wr2:>5.1%}  "
      f"{hm2['p_profit']:>6.3f}  {hm2['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/tsmom_strategy.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Time-Series Momentum (TSMOM) su BTCUSDT — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
