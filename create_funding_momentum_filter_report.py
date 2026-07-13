#!/usr/bin/env python3
"""
create_funding_momentum_filter_report.py
════════════════════════════════════════════════════════════════════════════
Funding-Rate Momentum Filter — BTCUSDT 1H, 2022-01 -> 2026-06.

Idea C from analyze_btcusdt_edge.py's finding #2 — flagged LOW confidence
at proposal time: top-5%-funding periods are followed by continuation
(fwd_3d=+2.35%, contradicting the textbook "fade crowded longs" prior), but
the effective independent sample is small (extreme-funding periods cluster
into a handful of serially-correlated bull-run episodes, not 247 independent
events despite n=247 8h-settlements).

Base strategy: the SMC 1H single-timeframe structural trend engine that
create_vol_regime_dual_mode_report.py's ablation validated independently
(V3 SMC Always-On: n=629, +145.8%, MaxDD -6.1%, DSR=0.998) — reused here
unmodified as C1, so this script directly tests whether a funding filter
helps or hurts an ALREADY-VALIDATED baseline, same ablation discipline as
the V1-V4/V5/V6 CVD-sweep study.

  C1 Baseline (no funding filter)      : identical engine to the validated V3
  C2 Avoid Fighting Extreme Funding    : skip longs when funding in bottom
                                          decile, skip shorts when funding in
                                          top decile (don't trade against an
                                          extreme-funding regime)
  C3 Require Funding Alignment         : only take longs when funding is in
                                          the top quintile, shorts when in the
                                          bottom quintile (require the momentum
                                          effect as confirmation, not just
                                          avoid its opposite) — expected to
                                          cut trade count hard; if n<30 that's
                                          the honest answer to "is this
                                          strong enough to require", not a bug.

Funding percentile is a CAUSAL rolling rank (trailing 180 settlements ≈ 60
days, min_periods=60) — a full-sample rank would leak the future funding
distribution into a signal available "live" only from trailing data.
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

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_binance_vision_funding
from src.strategy.indicators import atr as atr_fn
from src.strategy.smc import compute_smc_features, smc_trend_signal
from src.strategy.mtf_swing import causal_trend_state
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR, START_MONTH = 2022, 1
END_YEAR, END_MONTH = 2026, 6
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0006
MAX_LEV = 1.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 2_000
DSR_THRESHOLD = 0.95

TREND_SWING_LEN, TREND_INTERNAL_LEN = 30, 5
SL_BUFFER_ATR = 0.15
MIN_RR = 1.2
FALLBACK_SL_ATR = 1.0
FALLBACK_TP_ATR = 2.0
TIME_STOP_BARS = 96
COOLDOWN_BARS = 4

FUNDING_PCTILE_WINDOW = 180   # ~60 days of 8h settlements, causal rolling rank
AVOID_DECILE = 0.10
REQUIRE_QUINTILE = 0.20

report_lines: list[str] = []
def w(line: str = "") -> None:
    print(line)
    report_lines.append(line)

t_start = time.time()
w(SEP)
w("BTCUSDT — Funding-Rate Momentum Filter — Strategy Research")
w(SEP)

print("\n[DATA] Loading 1H OHLCV + funding rate...")
df1h = fetch_binance_vision_klines("1h", START_YEAR, START_MONTH, END_YEAR, END_MONTH, workers=6)
df1h["atr_14"] = atr_fn(df1h["high"], df1h["low"], df1h["close"], 14)
N1 = len(df1h)
IDX1 = df1h.index
HI1 = df1h["high"].values.astype(float)
LO1 = df1h["low"].values.astype(float)
CL1 = df1h["close"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0, df1h["atr_14"].shift(1).values, 1.0)

funding_8h = fetch_binance_vision_funding(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
funding_pctile_8h = funding_8h.rolling(FUNDING_PCTILE_WINDOW, min_periods=60).apply(
    lambda x: (x <= x.iloc[-1]).mean(), raw=False)
# Funding is known AT its settlement timestamp — plain ffill onto 1H is
# causally correct (unlike the daily-profile/vol-regime day+1 trick, which
# exists because THOSE aggregates aren't complete until the day ends).
FUND_PCTILE1 = funding_pctile_8h.reindex(IDX1, method="ffill").values
print(f"  {N1:,} 1H bars, {len(funding_8h):,} funding settlements")

print("\n[TREND ENGINE] SMC structural signals (1H, same engine validated as V3 in "
      "create_vol_regime_dual_mode_report.py)...")
smc1h = compute_smc_features(df1h, swing_len=TREND_SWING_LEN, internal_len=TREND_INTERNAL_LEN, prefix="smc")
bias1h = smc_trend_signal(smc1h, prefix="smc").values.astype(int)
zone_bull1h = ((smc1h["smc_ob_bull_in"].values == 1) | (smc1h["smc_in_discount"].values == 1))
zone_bear1h = ((smc1h["smc_ob_bear_in"].values == 1) | (smc1h["smc_in_premium"].values == 1))
struct1h = causal_trend_state(CL1, HI1, LO1, left=8, right=8)
sl_basis_low = struct1h["last_pivot_low"].values
sl_basis_high = struct1h["last_pivot_high"].values
tp_basis_high = struct1h["target_high"].values
tp_basis_low = struct1h["target_low"].values
TREND_SIGNAL1 = np.where((bias1h == 1) & zone_bull1h, 1, np.where((bias1h == -1) & zone_bear1h, -1, 0))
print(f"  Trend signal bars: long={int((TREND_SIGNAL1==1).sum())}  short={int((TREND_SIGNAL1==-1).sum())}")


def simulate(mode: str) -> tuple[pd.DataFrame, float, float]:
    cap = INIT_CAP
    peak = cap
    mdd = 0.0
    trades: list[dict] = []
    in_pos = False
    dirn = 0
    entry_px = sl = tp = qty = 0.0
    entry_i = 0
    hold_end = 0
    last_exit = -COOLDOWN_BARS

    for i in range(N1):
        if in_pos:
            hit = None
            if dirn == 1:
                if LO1[i] <= sl: hit = "SL"
                elif HI1[i] >= tp: hit = "TP"
            else:
                if HI1[i] >= sl: hit = "SL"
                elif LO1[i] <= tp: hit = "TP"
            if hit is None and i >= hold_end:
                hit = "TIME"
            if hit is not None:
                exit_px = sl if hit == "SL" else (tp if hit == "TP" else CL1[i])
                notional = qty * entry_px
                pnl = qty * (exit_px - entry_px) * dirn - FEE * 2 * notional
                cap += pnl
                peak = max(peak, cap)
                mdd = min(mdd, (cap - peak) / peak if peak > 0 else 0.0)
                trades.append(dict(entry_time=IDX1[entry_i], exit_time=IDX1[i], direction=dirn,
                                    entry_price=entry_px, exit_price=exit_px, exit_reason=hit,
                                    net_pnl=pnl, equity=cap))
                in_pos = False
                last_exit = i
            continue

        if i - last_exit < COOLDOWN_BARS:
            continue
        d = int(TREND_SIGNAL1[i])
        if d == 0:
            continue

        fp = FUND_PCTILE1[i]
        if mode == "avoid" and not np.isnan(fp):
            if d == 1 and fp < AVOID_DECILE:
                continue
            if d == -1 and fp > (1 - AVOID_DECILE):
                continue
        if mode == "require":
            if np.isnan(fp):
                continue
            if d == 1 and fp < (1 - REQUIRE_QUINTILE):
                continue
            if d == -1 and fp > REQUIRE_QUINTILE:
                continue

        atrv = ATR1[i]
        if atrv <= 0:
            continue
        px = CL1[i]
        sl_level = sl_basis_low[i] if d == 1 else sl_basis_high[i]
        tp_level = tp_basis_high[i] if d == 1 else tp_basis_low[i]
        slp = (sl_level - d * SL_BUFFER_ATR * atrv) if not np.isnan(sl_level) else (px - d * FALLBACK_SL_ATR * atrv)
        tpp = tp_level if not np.isnan(tp_level) else (px + d * FALLBACK_TP_ATR * atrv)
        risk = abs(px - slp)
        if risk <= 0:
            continue
        if abs(tpp - px) / risk < MIN_RR:
            tpp = px + d * max(MIN_RR * risk, FALLBACK_TP_ATR * atrv)

        risk_usd = INIT_CAP * RISK_PCT
        q = min(risk_usd / risk, cap * MAX_LEV / px)
        if q <= 0:
            continue
        dirn, entry_px, sl, tp, qty, entry_i = d, px, slp, tpp, q, i
        hold_end = i + TIME_STOP_BARS
        in_pos = True

    trades_df = pd.DataFrame(trades, columns=[
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "exit_reason", "net_pnl", "equity",
    ])
    return trades_df, cap, mdd


print("\n[BACKTEST] Running 3 variants (baseline / avoid-extreme / require-alignment)...")
c1_trades, c1_cap, c1_mdd = simulate("baseline")
c2_trades, c2_cap, c2_mdd = simulate("avoid")
c3_trades, c3_cap, c3_mdd = simulate("require")

VARIANTS = {
    "C1 Baseline (no funding filter)": dict(trades=c1_trades, cap=c1_cap, mdd=c1_mdd),
    "C2 Avoid Fighting Extreme Funding": dict(trades=c2_trades, cap=c2_cap, mdd=c2_mdd),
    "C3 Require Funding Alignment": dict(trades=c3_trades, cap=c3_cap, mdd=c3_mdd),
}
for name, v in VARIANTS.items():
    n = len(v["trades"])
    ret = (v["cap"] / INIT_CAP - 1) * 100
    wr = (v["trades"]["net_pnl"] > 0).mean() * 100 if n else 0.0
    w(f"  {name:<34}  n={n:>5}  ret={ret:>+7.1f}%  mdd={v['mdd']*100:>6.1f}%  wr={wr:>5.1f}%")

print("\n[VALIDATION] Monte Carlo + holdout + DSR...")
def mc_pair(trades_df):
    if len(trades_df) < 5:
        return {}, {}
    return run_monte_carlo(trades_df, INIT_CAP, N_SIMS), run_monte_carlo_block(trades_df, INIT_CAP, N_SIMS, block_size=20)

dsr_family = []
for name, v in VARIANTS.items():
    trades = v["trades"]
    mc, mc_blk = mc_pair(trades)
    holdout = trades[trades["entry_time"] >= CUTOFF] if len(trades) else trades
    v["mc"] = mc
    v["holdout_trades"] = holdout
    dsr_family.append({"variant": name, "net_pnls": trades["net_pnl"].tolist() if len(trades) else []})

deflated_sharpe_ratio_family(dsr_family, pnls_key="net_pnls")
for d in dsr_family:
    VARIANTS[d["variant"]]["sharpe_hat"] = d["sharpe_hat"]
    VARIANTS[d["variant"]]["dsr"] = d["dsr"]

w(f"\n  Deflated Sharpe Ratio (family N={len(dsr_family)}, threshold={DSR_THRESHOLD}):")
for name, v in VARIANTS.items():
    w(f"    {name:<34}  sharpe_hat={v['sharpe_hat']:>+6.3f}  DSR={v['dsr']:>5.3f}  "
      f"{'PASS' if v['dsr'] >= DSR_THRESHOLD else 'FAIL'}")

for name, v in VARIANTS.items():
    ret = (v["cap"] / INIT_CAP - 1)
    p_ruin = v["mc"].get("p_ruin", 1.0) if v["mc"] else 1.0
    n_trades = len(v["trades"])
    holdout_ret = ((v["holdout_trades"]["net_pnl"].sum() / INIT_CAP) if len(v["holdout_trades"]) else -1.0)
    v["verdict"] = (n_trades >= 30 and ret > 0 and p_ruin < 0.10 and v["dsr"] >= DSR_THRESHOLD and holdout_ret > 0)
    w(f"  {name:<34}  n>=30:{n_trades>=30}  ret>0:{ret>0}  P(ruin)<10%:{p_ruin<0.10}  "
      f"DSR>=0.95:{v['dsr']>=DSR_THRESHOLD}  holdout>0:{holdout_ret>0}  -> "
      f"{'VALIDATA' if v['verdict'] else 'NON VALIDATA'}")

print(f"\n[DONE] runtime: {time.time()-t_start:.0f}s")
Path("reports").mkdir(exist_ok=True)
Path("reports/funding_momentum_filter_research.md").write_text("\n".join(report_lines))
print("[SAVED] reports/funding_momentum_filter_research.md")
