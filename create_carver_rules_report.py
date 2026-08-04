#!/usr/bin/env python3
"""
create_carver_rules_report.py
================================
Test delle regole di trading FONDAMENTALI del framework sistematico di
Rob Carver (qoppac.blogspot.com, autore di "Systematic Trading" e
"Leveraged Trading") su BTCUSDT — non un singolo post del blog (che copre
10+ anni, in gran parte metodologia costruita SOPRA queste regole), ma i
building block che l'intero corpus assume: EWMAC (trend), Breakout
(canale Donchian), Carry (qui: funding rate del perpetual, l'analogo
crypto del "roll yield" dei futures).

A differenza di ogni altra strategia testata in questa sessione, queste
NON sono regole di entry/exit discrete con stop/target — sono FORECAST
CONTINUI che ridimensionano una posizione ogni giorno (vol-targeting),
esattamente come nel framework originale. Backtest mark-to-market
giornaliero, non trade-based.

── EWMAC (trend) ───────────────────────────────────────────────────────
  raw[t]  = EWMA(price, fast) - EWMA(price, slow)
  vol_adj[t] = raw[t] / price_vol[t]           (price_vol = EWMA stdev
                                                 dei delta di prezzo, span=25)
  forecast[t] = vol_adj[t] * scalar[t]          (scalar calibrato
                                                 causalmente: media mobile
                                                 espandente di |vol_adj|
                                                 fino a t, target=10 —
                                                 stessa filosofia del
                                                 "forecast scalar" del
                                                 framework, senza
                                                 assumere le costanti
                                                 pubblicate nel libro)
  cap a ±20. 6 velocità standard: (2,8) (4,16) (8,32) (16,64) (32,128)
  (64,256), più il POOL (media delle 6) — Carver non tradizionale mai una
  sola velocità.

── Breakout (canale Donchian) ──────────────────────────────────────────
  mid[t]  = (max_N(price) + min_N(price)) / 2   (canale N barre, causale)
  raw[t]  = 40 * (price[t]-mid[t]) / (0.5*(max_N-min_N))
  forecast[t] = EWMA(raw, span=N/4), cap ±20
  N standard: 10, 20, 40, 80, 160, 320, più il POOL.

── Carry (funding rate BTCUSDT perpetual) ──────────────────────────────
  Una posizione che RICEVE funding guadagna carry: funding>0 (i long
  pagano gli short) -> lo short riceve -> forecast negativo (short-bias)
  quando il funding è positivo, e viceversa.
  forecast[t] = -normalizza(EWMA(funding, 7g)) , stessa calibrazione
  scalar/cap del EWMAC.

Sizing (vol-targeting, stessa filosofia del framework, capitale fisso non
compounding, coerente con tutti gli altri script di questa sessione):
  units[t] = (forecast[t]/10) * (INIT_CAP * TARGET_ANNUAL_VOL) / annualized_price_vol[t]
  cap a MAX_LEV × INIT_CAP / price[t]

Costi: frizione Bybit obbligatoria (taker 0.055% + slippage 0.015% =
0.07%/lato) applicata sul TURNOVER (|Δunits|×price), non per-trade — è
così che un sistema a posizione continua paga i costi.

Validazione: per-anno (sui 4 candidati finali: EWMAC pool, Breakout pool,
Carry, Grand pool = media dei 3), holdout 2025-2026 genuino, Monte Carlo
i.i.d.+block, DSR family sui 4 candidati, slippage-stress. Le singole
velocità sono mostrate solo come tabella diagnostica (non ri-testate
separatamente con DSR — nessuno le tradirebbe da sole).
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

from src.strategy.data_fetcher import fetch_extended_data, fetch_binance_vision_funding
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
EWMAC_SPEEDS = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
BREAKOUT_NS = [10, 20, 40, 80, 160, 320]
CARRY_SMOOTH_DAYS = 7
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Carver Systematic Rules (EWMAC / Breakout / Carry) su BTCUSDT — Validation Pipeline")
w(SEP)
w("\nBuilding block del framework di qoppac.blogspot.com: forecast continui,")
w("vol-targeting, mai una sola velocità (pool). Backtest mark-to-market")
w("giornaliero, frizioni Bybit obbligatorie sul turnover.")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 1D + funding rate …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1d = raw["1D"]
IDX = df1d.index
N = len(df1d)
CL = df1d["close"].values.astype(float)
print(f"  1D: {N:,} bars  ({IDX[0].date()} → {IDX[-1].date()})")

funding = fetch_binance_vision_funding(start_year=START_YEAR, start_month=1)
funding_daily = funding.resample("1D").mean().reindex(IDX, method="ffill")
print(f"  Funding: {funding.notna().sum():,} righe 8h  ({funding.index[0].date() if len(funding) else 'n/a'} → "
      f"{funding.index[-1].date() if len(funding) else 'n/a'})  (loaded in {time.time()-t0:.0f}s)")

price = pd.Series(CL, index=IDX)
daily_ret = price.diff()
price_vol = daily_ret.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
annualized_price_vol = (price_vol * np.sqrt(365)).values
price_vol_v = price_vol.values


def calibrate_forecast(raw_arr):
    """Scalar causale: media espandente di |raw| fino a t, target=10; cap ±20."""
    raw_s = pd.Series(raw_arr)
    abs_expanding_mean = raw_s.abs().expanding(min_periods=60).mean()
    scalar = FORECAST_TARGET_ABS / abs_expanding_mean.replace(0, np.nan)
    fcst = (raw_s * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
    return fcst.values


# ── EWMAC ────────────────────────────────────────────────────────────────
print("[EWMAC] Computing 6 speed variants + pool …")
ewmac_forecasts = {}
for fast, slow in EWMAC_SPEEDS:
    raw_e = (price.ewm(span=fast, min_periods=fast).mean()
             - price.ewm(span=slow, min_periods=slow).mean()).values
    vol_adj = np.where(price_vol_v > 0, raw_e / price_vol_v, np.nan)
    ewmac_forecasts[f"EWMAC({fast},{slow})"] = calibrate_forecast(vol_adj)
ewmac_pool = np.nanmean(np.column_stack(list(ewmac_forecasts.values())), axis=1)

# ── Breakout (Donchian) ────────────────────────────────────────────────────
print("[BREAKOUT] Computing 6 N variants + pool …")
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

# ── Carry (funding) ─────────────────────────────────────────────────────
print("[CARRY] Computing funding-based carry forecast …")
funding_smooth = funding_daily.ewm(span=CARRY_SMOOTH_DAYS, min_periods=CARRY_SMOOTH_DAYS).mean()
carry_raw = -funding_smooth.values   # short quando il funding è positivo (riceve carry)
carry_forecast = calibrate_forecast(np.where(np.isfinite(carry_raw), carry_raw, np.nan))

# ── Grand pool ───────────────────────────────────────────────────────────
grand_pool = np.nanmean(np.column_stack([ewmac_pool, breakout_pool, carry_forecast]), axis=1)

CANDIDATES = {
    "EWMAC pool": ewmac_pool,
    "Breakout pool": breakout_pool,
    "Carry (funding)": carry_forecast,
    "Grand pool (EWMAC+BRK+Carry)": grand_pool,
}


def run_bt(forecast, fee=FRICTION, slippage_extra=0.0):
    n = len(forecast)
    units = np.full(n, np.nan)
    valid = np.isfinite(forecast) & (annualized_price_vol > 0)
    # annualized_price_vol è già in $/unità/anno (dev.std. annualizzata del
    # prezzo in dollari) -> Capitale*TargetVol ($) / annualized_price_vol
    # ($/unità) = UNITÀ direttamente, nessuna ulteriore divisione per CL.
    raw_units = np.where(valid, (forecast / FORECAST_TARGET_ABS) * INIT_CAP * TARGET_ANNUAL_VOL /
                          np.where(annualized_price_vol > 0, annualized_price_vol, np.nan), 0.0)
    max_units = MAX_LEV * INIT_CAP / CL
    units = np.clip(raw_units, -max_units, max_units)
    units = np.where(valid, units, 0.0)

    tot_fee = fee + slippage_extra
    # pnl[t] = units[t-1]*(CL[t]-CL[t-1]) - costo di transizione a units[t] (pagato al ribilanciamento)
    pnl_series = np.zeros(n)
    cost_series = np.zeros(n)
    prev_units = 0.0
    for t in range(1, n):
        cost_series[t] = tot_fee * abs(units[t] - prev_units) * CL[t]
        pnl_series[t] = prev_units * (CL[t] - CL[t - 1]) if t >= 1 else 0.0
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


# ── Tabella diagnostica: singole velocità (nessun DSR, solo diagnostica) ──
w(f"\n{SEP}")
w("Diagnostica — singole velocità EWMAC e Breakout (mai tradate da sole)")
w(SEP)
w(f"\n  {'Rule':>16}  {'Ret full%':>10}  {'WR':>6}  {'MDD%':>7}")
for name, fcst in list(ewmac_forecasts.items()) + list(breakout_forecasts.items()):
    r = run_bt(fcst)
    w(f"  {name:>16}  {r['ret']:>+9.1f}%  {r['wr']:>5.1%}  {r['mdd']:>6.1f}%")

# ── 4 candidati finali: full pipeline ──────────────────────────────────────
dsr_full = []
dsr_holdout = []
for label, fcst in CANDIDATES.items():
    w(f"\n{SEP}")
    w(f"{label}")
    w(SEP)

    r = run_bt(fcst)
    mc = mc_summary(r["net_pnls"])
    mc_blk = mc_block_summary(r["net_pnls"])
    w(f"\n  FULL-SAMPLE: n_days={r['n']}  wr={r['wr']:.1%}  ret={r['ret']:+.1f}%  mdd={r['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    yr_of = np.array([d.year for d in r["dates"]])
    for yr in sorted(set(yr_of)):
        mask = yr_of == yr
        pnl_y = [r["net_pnls"][i] for i in range(len(mask)) if mask[i]]
        if len(pnl_y) < 30: continue
        cap_y = INIT_CAP + np.cumsum(pnl_y)
        wins_y = sum(1 for x in pnl_y if x > 0)
        w(f"    {yr:>6}  {len(pnl_y):>6}  {(cap_y[-1]/INIT_CAP-1)*100:>+7.1f}%  {wins_y/len(pnl_y):>5.1%}")

    hold_mask = np.array([d >= CUTOFF for d in r["dates"]])
    hold_pnls = [r["net_pnls"][i] for i in range(len(hold_mask)) if hold_mask[i]]
    hcap = INIT_CAP + np.cumsum(hold_pnls) if hold_pnls else np.array([INIT_CAP])
    hwins = sum(1 for x in hold_pnls if x > 0)
    hwr = hwins / len(hold_pnls) if hold_pnls else 0.0
    hmc = mc_summary(hold_pnls)
    hmc_blk = mc_block_summary(hold_pnls)
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={len(hold_pnls)}  wr={hwr:.1%}  "
      f"ret={(hcap[-1]/INIT_CAP-1)*100:+.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    dsr_full.append(dict(label=label, net_pnls=r["net_pnls"], ret=r["ret"]))
    dsr_holdout.append(dict(label=label, net_pnls=hold_pnls, ret=(hcap[-1]/INIT_CAP-1)*100))

w(f"\n{SEP}")
w(f"DSR — famiglia N={len(CANDIDATES)} (4 candidati finali)")
w(SEP)
deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
w(f"\n  Full-sample:")
w(f"  {'Rule':>30}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_full:
    w(f"  {r['label']:>30}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
w(f"\n  Holdout 2025-2026:")
w(f"  {'Rule':>30}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_holdout:
    w(f"  {r['label']:>30}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

best_label = max(dsr_full, key=lambda r: r["dsr"])["label"]
best_fcst = CANDIDATES[best_label]
w(f"\n{SEP}")
w(f"Slippage-stress — migliore per DSR = {best_label}")
w(SEP)
w(f"\n  {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_STRESS_BPS:
    extra = bps / 10_000.0
    r_full = run_bt(best_fcst, slippage_extra=extra)
    r_hold_pnls = [r_full["net_pnls"][i] for i in range(len(r_full["dates"])) if r_full["dates"][i] >= CUTOFF]
    for scope_name, pnls_scope in [("full-sample", r_full["net_pnls"]), ("holdout", r_hold_pnls)]:
        if not pnls_scope: continue
        cap_s = INIT_CAP + np.cumsum(pnls_scope)
        wins_s = sum(1 for x in pnls_scope if x > 0)
        m = mc_summary(pnls_scope)
        w(f"  {bps:>7}bps  {scope_name:>10}  {len(pnls_scope):>6}  {(cap_s[-1]/INIT_CAP-1)*100:>+7.1f}%  "
          f"{wins_s/len(pnls_scope):>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/carver_rules.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Carver Systematic Rules (EWMAC / Breakout / Carry) su BTCUSDT — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
