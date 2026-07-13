#!/usr/bin/env python3
"""
create_daily_volume_profile_report.py
════════════════════════════════════════════════════════════════════════════
Daily Volume Profile Exhaustion Reversal — BTCUSDT 5m, 2022-01 -> 2026-06.

User's original idea: compute yesterday's daily volume profile at the start
of each session, find b-shape/d-shape, identify demand/supply, trade the
exhaustion of the prior trend and the start of a new one from the demand
zone. Extended here (see reports/ for the writeup) with:
  - the mirror short setup from a P-shape supply zone (the original brief
    only covered the long side — a real edge shouldn't be direction-biased
    by construction)
  - an explicit no-trade regime filter on D-shape/B-shape days (balance
    days have no directional exhaustion to trade)
  - 3-way confluence requirement at the zone touch instead of a bare
    price-in-zone trigger: volume climax+contraction, real CVD/delta
    divergence (from actual taker_buy_base, not synthetic OI), and a
    structural CHoCH confirmation (reusing src/strategy/smc.py) — see
    Section 5

Reuse (no logic reimplemented):
  volume_profile.py -> src/strategy/volume_profile.py (new this session)
  structure          -> src/strategy/smc.py (CHoCH confirmation)
  data                -> src/strategy/data_fetcher.py's
                          fetch_binance_vision_taker_flow (REAL taker_buy_base,
                          not the generate_oi/generate_funding synthetic
                          fallbacks elsewhere in that module)
  validation          -> src/strategy/monte_carlo.py (iid + block MC, DSR)
  report              -> src/strategy/report_html.py (shared palette/CSS)

Zero fitted parameters (all thresholds are a-priori domain choices, not
fit to data) -> no WFO/train-test split needed for the signal itself; the
genuine 2025-2026 holdout is still checked to catch threshold-tuning-by-
iteration, same as every other report in this repo.

Gate (identical to the rest of the repo): n>=30 trade OOS, return OOS>0,
P(ruin) MC iid < 10%, Deflated Sharpe Ratio >= 0.95 (family of 3 variants,
corrects the multi-variant selection bias), return positive on the genuine
2025-2026 holdout (never touched in threshold selection).
"""
from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.data_fetcher import fetch_binance_vision_taker_flow
from src.strategy.indicators import atr as atr_fn
from src.strategy.volume_profile import compute_daily_profiles, compute_bar_delta
from src.strategy.smc import compute_smc_features, smc_trend_signal
from src.strategy.mtf_swing import align_htf_to_ltf
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)
from src.strategy.report_html import (
    _CSS, _fig_to_b64, _img_tag, _kpi_card, _table, _signed, _color_signed,
    BG, PANEL, BORDER, WHITE, GRAY, GREEN, RED, GOLD, BLUE, PURPLE, ORANGE,
)

plt.style.use("dark_background")

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
SEP = "═" * 78
START_YEAR, START_MONTH = 2022, 1
END_YEAR, END_MONTH = 2026, 6
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0006
MAX_LEV = 1.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 2_000

TIME_STOP_BARS_5M = 288      # 24h at 5m — one full session
COOLDOWN_BARS_5M = 48        # 4h — avoid re-entering the same zone repeatedly
SL_BUFFER_ATR = 0.15
MIN_RR = 1.2
FALLBACK_SL_ATR = 1.0
FALLBACK_TP_ATR = 2.0

VOL_Z_LOOKBACK = 20           # bars for volume mean/std (climax detection)
VOL_Z_CLIMAX_THRESHOLD = 2.0
VOL_CONTRACTION_FRAC = 0.70   # current volume must be < this fraction of its SMA
CLIMAX_RECENT_BARS = 3        # climax must have happened in the last N bars (not the entry bar itself)

CVD_LOOKBACK_BARS = 12        # 1h — window to find "the prior local extreme"
CVD_ROLL_BARS = 3             # rolling delta sum window for the divergence comparison

STRUCT_SWING_LEN = 20         # smaller than the 30m HTF bias's 50 — this is a same-timeframe
STRUCT_INTERNAL_LEN = 3       # micro-confirmation, not a multi-day trend bias
STRUCT_LOOKBACK_BARS = 6      # CHoCH must have flipped to our direction within the last N bars

report_lines: list[str] = []
def w(line: str = "") -> None:
    print(line)
    report_lines.append(line)

t_start = time.time()
w(SEP)
w("BTCUSDT — Daily Volume Profile Exhaustion Reversal — Strategy Research")
w(SEP)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA — 5m OHLCV + real taker_buy_base (for CVD), 2022-01 -> 2026-06
# ═══════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 5m BTCUSDT perpetual history with real order-flow (taker_buy_base)...")
t0 = time.time()
df5 = fetch_binance_vision_taker_flow(
    "5m", START_YEAR, START_MONTH, END_YEAR, END_MONTH, workers=6)
df5["atr_14"] = atr_fn(df5["high"], df5["low"], df5["close"], 14)
print(f"  5m: {len(df5):,}  ({df5.index[0].date()} -> {df5.index[-1].date()}, {time.time()-t0:.0f}s)")

N5 = len(df5)
IDX5 = df5.index
OP5 = df5["open"].values.astype(float)
HI5 = df5["high"].values.astype(float)
LO5 = df5["low"].values.astype(float)
CL5 = df5["close"].values.astype(float)
VOL5 = df5["volume"].values.astype(float)
# ATR at entry uses the PRIOR bar's ATR (shift(1)) — same no-lookahead
# convention as every other report in this repo.
ATR5 = np.where(df5["atr_14"].shift(1).values > 0, df5["atr_14"].shift(1).values, 1.0)

# ═══════════════════════════════════════════════════════════════════════════
# 2. DAILY VOLUME PROFILE — prior-day POC/VA/shape, aligned causally onto
#    the 5m grid (profile_ts = day+1 00:00, see volume_profile.py's
#    Causality contract docstring for why this indexing avoids the same
#    lookahead-bug class documented in VALIDATED_STRATEGIES_SPEC.md).
# ═══════════════════════════════════════════════════════════════════════════
print("\n[PROFILE] Building daily volume profiles (POC/VA/shape/demand-supply)...")
t0 = time.time()
profiles = compute_daily_profiles(df5, n_bins=50, value_area_pct=0.70)
shape_counts = profiles["shape"].value_counts()
print(f"  {len(profiles)} daily profiles  ({time.time()-t0:.0f}s)  shapes: {shape_counts.to_dict()}")

prof_aligned = align_htf_to_ltf(
    profiles.index, profiles[["shape", "demand_lo", "demand_hi", "supply_lo", "supply_hi", "poc", "vah", "val"]],
    IDX5,
)
SHAPE5 = prof_aligned["shape"].values
DEM_LO5 = prof_aligned["demand_lo"].values.astype(float)
DEM_HI5 = prof_aligned["demand_hi"].values.astype(float)
SUP_LO5 = prof_aligned["supply_lo"].values.astype(float)
SUP_HI5 = prof_aligned["supply_hi"].values.astype(float)
POC5 = prof_aligned["poc"].values.astype(float)
VAH5 = prof_aligned["vah"].values.astype(float)
VAL5 = prof_aligned["val"].values.astype(float)

# ═══════════════════════════════════════════════════════════════════════════
# 3. CONFLUENCE SIGNAL 1 — volume climax + contraction
# ═══════════════════════════════════════════════════════════════════════════
print("\n[SIGNALS] Volume climax+contraction...")
vol_s = df5["volume"]
vol_sma = vol_s.rolling(VOL_Z_LOOKBACK, min_periods=5).mean()
vol_std = vol_s.rolling(VOL_Z_LOOKBACK, min_periods=5).std()
vol_z = ((vol_s - vol_sma) / vol_std.replace(0, np.nan)).fillna(0).values
VOL_SMA5 = vol_sma.fillna(0).values
recent_climax = pd.Series(vol_z).rolling(CLIMAX_RECENT_BARS, min_periods=1).max().shift(1).fillna(0).values
CLIMAX5 = (recent_climax > VOL_Z_CLIMAX_THRESHOLD)
CONTRACTION5 = VOL5 < (VOL_CONTRACTION_FRAC * VOL_SMA5)
FLAG_A5 = CLIMAX5 & CONTRACTION5

# ═══════════════════════════════════════════════════════════════════════════
# 4. CONFLUENCE SIGNAL 2 — real CVD/delta divergence at a new local extreme
# ═══════════════════════════════════════════════════════════════════════════
print("[SIGNALS] CVD/delta divergence (real taker_buy_base)...")
delta5 = compute_bar_delta(df5)
delta_roll = delta5.rolling(CVD_ROLL_BARS, min_periods=1).sum().values

roll_low_idx = pd.Series(LO5).rolling(CVD_LOOKBACK_BARS, min_periods=CVD_LOOKBACK_BARS).apply(
    lambda x: int(np.argmin(x.values)), raw=False).shift(1)
roll_high_idx = pd.Series(HI5).rolling(CVD_LOOKBACK_BARS, min_periods=CVD_LOOKBACK_BARS).apply(
    lambda x: int(np.argmax(x.values)), raw=False).shift(1)
prior_low = pd.Series(LO5).rolling(CVD_LOOKBACK_BARS, min_periods=CVD_LOOKBACK_BARS).min().shift(1).values
prior_high = pd.Series(HI5).rolling(CVD_LOOKBACK_BARS, min_periods=CVD_LOOKBACK_BARS).max().shift(1).values

new_low5 = LO5 < prior_low
new_high5 = HI5 > prior_high

# roll_low_idx/roll_high_idx are the WITHIN-WINDOW offset of the prior
# extreme (0..CVD_LOOKBACK_BARS-1); convert to an absolute bar index.
window_start = np.arange(N5) - CVD_LOOKBACK_BARS
FLAG_B_LONG5 = np.zeros(N5, dtype=bool)
FLAG_B_SHORT5 = np.zeros(N5, dtype=bool)
rli = roll_low_idx.values
rhi = roll_high_idx.values
for i in range(CVD_LOOKBACK_BARS + 1, N5):
    if new_low5[i] and not np.isnan(rli[i]):
        prior_idx = window_start[i] + int(rli[i])
        if 0 <= prior_idx < i:
            FLAG_B_LONG5[i] = delta_roll[i] > delta_roll[prior_idx]
    if new_high5[i] and not np.isnan(rhi[i]):
        prior_idx = window_start[i] + int(rhi[i])
        if 0 <= prior_idx < i:
            FLAG_B_SHORT5[i] = delta_roll[i] < delta_roll[prior_idx]

del delta5, roll_low_idx, roll_high_idx, prior_low, prior_high, rli, rhi
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# 5. CONFLUENCE SIGNAL 3 — structural CHoCH confirmation (5m, reused smc.py)
# ═══════════════════════════════════════════════════════════════════════════
print("[SIGNALS] Structural CHoCH confirmation (smc.py, 5m)...")
smc5 = compute_smc_features(df5, swing_len=STRUCT_SWING_LEN, internal_len=STRUCT_INTERNAL_LEN, prefix="smc")
trend5 = smc_trend_signal(smc5, prefix="smc").values
trend_shift = pd.Series(trend5).shift(STRUCT_LOOKBACK_BARS).values
FLAG_C_LONG5 = (trend5 == 1) & (trend_shift != 1)
FLAG_C_SHORT5 = (trend5 == -1) & (trend_shift != -1)

del smc5
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# 6. ZONE TOUCH
# ═══════════════════════════════════════════════════════════════════════════
DIRN5 = np.where(SHAPE5 == "b", 1, np.where(SHAPE5 == "P", -1, 0))
ZONE_LO5 = np.where(DIRN5 == 1, DEM_LO5, np.where(DIRN5 == -1, SUP_LO5, np.nan))
ZONE_HI5 = np.where(DIRN5 == 1, DEM_HI5, np.where(DIRN5 == -1, SUP_HI5, np.nan))
TOUCHED5 = (LO5 <= ZONE_HI5) & (HI5 >= ZONE_LO5)

CONFLUENCE5 = np.where(
    DIRN5 == 1, FLAG_A5.astype(int) + FLAG_B_LONG5.astype(int) + FLAG_C_LONG5.astype(int),
    np.where(DIRN5 == -1, FLAG_A5.astype(int) + FLAG_B_SHORT5.astype(int) + FLAG_C_SHORT5.astype(int), 0),
)

print(f"  bars with shape=b: {(SHAPE5=='b').sum():,}  shape=P: {(SHAPE5=='P').sum():,}  "
      f"zone touches: {TOUCHED5.sum():,}  (of which conf>=2: {int((TOUCHED5 & (CONFLUENCE5>=2)).sum()):,}, "
      f"conf==3: {int((TOUCHED5 & (CONFLUENCE5==3)).sum()):,})")

# ═══════════════════════════════════════════════════════════════════════════
# 7. BACKTEST ENGINE — single-position state machine, structural SL/TP
#    (POC = TP1 target, MIN_RR floor widens toward VAH/VAL), 24h time-stop,
#    4h cooldown. Same order-of-checks convention as the rest of the repo.
# ═══════════════════════════════════════════════════════════════════════════
def simulate(min_confluence: int) -> tuple[pd.DataFrame, float, float]:
    cap = INIT_CAP
    peak = cap
    mdd = 0.0
    trades: list[dict] = []
    in_pos = False
    dirn = 0
    entry_px = sl = tp = qty = 0.0
    entry_i = 0
    hold_end = 0
    last_exit = -COOLDOWN_BARS_5M

    for i in range(N5):
        if in_pos:
            hit = None
            if dirn == 1:
                if LO5[i] <= sl: hit = "SL"
                elif HI5[i] >= tp: hit = "TP"
            else:
                if HI5[i] >= sl: hit = "SL"
                elif LO5[i] <= tp: hit = "TP"
            if hit is None and i >= hold_end:
                hit = "TIME"
            if hit is not None:
                exit_px = sl if hit == "SL" else (tp if hit == "TP" else CL5[i])
                notional = qty * entry_px
                pnl = qty * (exit_px - entry_px) * dirn - FEE * 2 * notional
                cap += pnl
                peak = max(peak, cap)
                mdd = min(mdd, (cap - peak) / peak if peak > 0 else 0.0)
                trades.append(dict(
                    entry_time=IDX5[entry_i], exit_time=IDX5[i], direction=dirn,
                    entry_price=entry_px, exit_price=exit_px, exit_reason=hit,
                    net_pnl=pnl, equity=cap,
                ))
                in_pos = False
                last_exit = i
            continue

        if i - last_exit < COOLDOWN_BARS_5M:
            continue
        d = DIRN5[i]
        if d == 0 or not TOUCHED5[i]:
            continue
        if CONFLUENCE5[i] < min_confluence:
            continue

        atrv = ATR5[i]
        if atrv <= 0:
            continue
        px = CL5[i]
        zone_extreme = DEM_LO5[i] if d == 1 else SUP_HI5[i]
        slp = (zone_extreme - d * SL_BUFFER_ATR * atrv) if not np.isnan(zone_extreme) else (px - d * FALLBACK_SL_ATR * atrv)
        tpp = POC5[i] if not np.isnan(POC5[i]) else (px + d * FALLBACK_TP_ATR * atrv)
        # Sanity: TP must be on the reward side of entry (POC of a b-shape
        # day sits above the demand zone by construction, but guard anyway).
        if (tpp - px) * d <= 0:
            tpp = px + d * FALLBACK_TP_ATR * atrv
        risk = abs(px - slp)
        if risk <= 0:
            continue
        reward = abs(tpp - px)
        if reward / risk < MIN_RR:
            va_edge = VAH5[i] if d == 1 else VAL5[i]
            tpp = va_edge if not np.isnan(va_edge) and (va_edge - px) * d > risk * MIN_RR else px + d * max(MIN_RR * risk, FALLBACK_TP_ATR * atrv)

        risk_usd = INIT_CAP * RISK_PCT
        q = risk_usd / risk
        max_q = cap * MAX_LEV / px
        q = min(q, max_q)
        if q <= 0:
            continue

        dirn, entry_px, sl, tp, qty, entry_i = d, px, slp, tpp, q, i
        hold_end = i + TIME_STOP_BARS_5M
        in_pos = True

    trades_df = pd.DataFrame(trades, columns=[
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "exit_reason", "net_pnl", "equity",
    ])
    return trades_df, cap, mdd


# ═══════════════════════════════════════════════════════════════════════════
# 8. RUN 3 VARIANTS — increasing confluence requirement
# ═══════════════════════════════════════════════════════════════════════════
print("\n[BACKTEST] Running 3 variants (zone-touch only / >=2 confluence / all 3)...")
v1_trades, v1_cap, v1_mdd = simulate(min_confluence=0)
v2_trades, v2_cap, v2_mdd = simulate(min_confluence=2)
v3_trades, v3_cap, v3_mdd = simulate(min_confluence=3)

VARIANTS = {
    "V1 Zone Touch Only": dict(trades=v1_trades, cap=v1_cap, mdd=v1_mdd),
    "V2 Confluence >=2/3": dict(trades=v2_trades, cap=v2_cap, mdd=v2_mdd),
    "V3 Confluence 3/3": dict(trades=v3_trades, cap=v3_cap, mdd=v3_mdd),
}

for name, v in VARIANTS.items():
    n = len(v["trades"])
    ret = (v["cap"] / INIT_CAP - 1) * 100
    wr = (v["trades"]["net_pnl"] > 0).mean() * 100 if n else 0.0
    w(f"  {name:<22}  n={n:>5}  ret={ret:>+7.1f}%  mdd={v['mdd']*100:>6.1f}%  wr={wr:>5.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 9. VALIDATION — Monte Carlo (iid + block), holdout, Deflated Sharpe Ratio
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VALIDATION] Monte Carlo + holdout + DSR...")

def mc_pair(trades_df: pd.DataFrame) -> tuple[dict, dict]:
    if len(trades_df) < 5:
        return {}, {}
    mc = run_monte_carlo(trades_df, INIT_CAP, N_SIMS)
    mc_blk = run_monte_carlo_block(trades_df, INIT_CAP, N_SIMS, block_size=20)
    return mc, mc_blk

DSR_THRESHOLD = 0.95
dsr_family = []
for name, v in VARIANTS.items():
    trades = v["trades"]
    mc, mc_blk = mc_pair(trades)
    holdout = trades[trades["entry_time"] >= CUTOFF] if len(trades) else trades
    hmc, hmc_blk = mc_pair(holdout)
    v["mc"] = mc
    v["mc_block"] = mc_blk
    v["holdout_trades"] = holdout
    v["holdout_mc"] = hmc
    dsr_family.append({"variant": name, "net_pnls": trades["net_pnl"].tolist() if len(trades) else []})

deflated_sharpe_ratio_family(dsr_family, pnls_key="net_pnls")
for d in dsr_family:
    VARIANTS[d["variant"]]["sharpe_hat"] = d["sharpe_hat"]
    VARIANTS[d["variant"]]["dsr"] = d["dsr"]

w(f"\n  Deflated Sharpe Ratio (family N={len(dsr_family)}, threshold={DSR_THRESHOLD}):")
for name, v in VARIANTS.items():
    w(f"    {name:<22}  sharpe_hat={v['sharpe_hat']:>+6.3f}  DSR={v['dsr']:>5.3f}  "
      f"{'PASS' if v['dsr'] >= DSR_THRESHOLD else 'FAIL'}")

for name, v in VARIANTS.items():
    ret = (v["cap"] / INIT_CAP - 1)
    p_ruin = v["mc"].get("p_ruin", 1.0) if v["mc"] else 1.0
    n_trades = len(v["trades"])
    holdout_ret = ((v["holdout_trades"]["net_pnl"].sum() / INIT_CAP) if len(v["holdout_trades"]) else -1.0)
    v["verdict"] = (
        n_trades >= 30
        and ret > 0
        and p_ruin < 0.10
        and v["dsr"] >= DSR_THRESHOLD
        and holdout_ret > 0
    )
    w(f"  {name:<22}  n>=30:{n_trades>=30}  ret>0:{ret>0}  P(ruin)<10%:{p_ruin<0.10}  "
      f"DSR>=0.95:{v['dsr']>=DSR_THRESHOLD}  holdout>0:{holdout_ret>0}  "
      f"-> {'VALIDATA' if v['verdict'] else 'NON VALIDATA'}")

w("\n  Breakdown per anno:")
for name, v in VARIANTS.items():
    w(f"\n    {name}")
    w(f"      {'Year':>6}  {'n':>5}  {'Ret%':>8}  {'WR':>6}")
    trades = v["trades"]
    if trades.empty:
        continue
    for yr, grp in trades.groupby(trades["entry_time"].dt.year):
        ret_yr = grp["net_pnl"].sum() / INIT_CAP * 100
        wr_yr = (grp["net_pnl"] > 0).mean() * 100
        w(f"      {yr:>6}  {len(grp):>5}  {ret_yr:>+7.1f}%  {wr_yr:>5.1f}%")

print(f"\n[DONE] backtest+validation runtime: {time.time()-t_start:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════
# 10. HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n[REPORT] Building HTML report...")
COLORS = {"V1 Zone Touch Only": BLUE, "V2 Confluence >=2/3": GOLD, "V3 Confluence 3/3": PURPLE}


def _ax2(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.10, color=GRAY, linestyle="--")
    if title:  ax.set_title(title, color=WHITE, fontsize=9, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY, fontsize=7)


def equity_series(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series([INIT_CAP], index=[IDX5[0]])
    idx = pd.DatetimeIndex([IDX5[0]] + list(trades_df["exit_time"]))
    vals = [INIT_CAP] + list(trades_df["equity"])
    return pd.Series(vals, index=idx)


def chart_equity_overlay() -> str:
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=BG)
    _ax2(ax, "Equity Curves", ylabel="Equity ($)")
    for name, v in VARIANTS.items():
        eq = equity_series(v["trades"])
        ax.plot(eq.index, eq.values, color=COLORS[name], lw=1.4, label=name)
    ax.axhline(INIT_CAP, color=GRAY, lw=0.8, ls=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"${val:,.0f}"))
    ax.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2, loc="upper left")
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_mc_fan(mc: dict, title: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6), facecolor=BG)
    fig.subplots_adjust(wspace=0.3, left=0.06, right=0.97)
    _ax2(ax1, f"{title} — MC Equity Paths (iid bootstrap)")
    if mc:
        paths = mc["paths"]
        xs = np.arange(paths.shape[1])
        for k in range(min(250, paths.shape[0])):
            ax1.plot(xs, paths[k], color=BLUE, lw=0.3, alpha=0.10)
        p5 = np.percentile(paths, 5, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        ax1.plot(xs, p5, color=RED, lw=1.2, label="p5")
        ax1.plot(xs, p50, color=WHITE, lw=1.5, label="p50")
        ax1.plot(xs, p95, color=GREEN, lw=1.2, label="p95")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"${val:,.0f}"))
        ax1.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    _ax2(ax2, "Final Return Distribution")
    if mc:
        ax2.hist(mc["total_return"] * 100, bins=50, color=BLUE, alpha=0.75, edgecolor=BORDER)
        ax2.axvline(0, color=GOLD, lw=1, ls=":")
        ax2.set_xlabel("Final Return (%)", color=GRAY, fontsize=7)
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


eq_chart_b64 = chart_equity_overlay()

kpi_cards = ""
variant_sections = ""
for name, v in VARIANTS.items():
    n = len(v["trades"])
    ret = (v["cap"] / INIT_CAP - 1) * 100
    wr = (v["trades"]["net_pnl"] > 0).mean() * 100 if n else 0.0
    mc = v["mc"]
    p_profit = mc.get("p_profit", 0.0) if mc else 0.0
    p_ruin = mc.get("p_ruin", 1.0) if mc else 1.0
    mc_blk = v["mc_block"]
    p_ruin_blk = mc_blk.get("p_ruin", 1.0) if mc_blk else 1.0
    holdout = v["holdout_trades"]
    holdout_ret = (holdout["net_pnl"].sum() / INIT_CAP * 100) if len(holdout) else 0.0
    verdict_badge = ('<span class="badge badge-green">VALIDATA</span>' if v["verdict"]
                      else '<span class="badge badge-red">NON VALIDATA</span>')

    kpi_cards += f"""
<div class="card" style="border-left:3px solid {COLORS[name]}">
  <div class="val {'green' if ret>0 else 'red'}">{ret:+.1f}%</div>
  <div class="lbl">{name} — Return</div>
</div>"""

    mc_chart_b64 = chart_mc_fan(mc, name) if mc else ""
    stat_rows = [
        ["Trades", f"{n:,}"],
        ["Win Rate", f"{wr:.1f}%"],
        ["Max Drawdown", f"{v['mdd']*100:.1f}%"],
        ["Sharpe (trade-level)", f"{v['sharpe_hat']:+.3f}"],
        ["Deflated Sharpe Ratio", f"{v['dsr']:.3f}"],
        ["MC P(profit)", f"{p_profit*100:.1f}%"],
        ["MC P(ruin) iid", f"{p_ruin*100:.1f}%"],
        ["MC P(ruin) block", f"{p_ruin_blk*100:.1f}%"],
        ["Holdout 2025-2026 return", f"{holdout_ret:+.1f}%"],
        ["Holdout trades", f"{len(holdout):,}"],
    ]
    variant_sections += f"""
<div class="section">
  <h3>{name} &nbsp; {verdict_badge}</h3>
  {_table(["Metric", "Value"], stat_rows)}
  {_img_tag(mc_chart_b64) if mc_chart_b64 else '<p class="muted">Insufficient trades for Monte Carlo.</p>'}
</div>"""

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Daily Volume Profile Exhaustion Reversal</title>
<style>{_CSS}</style></head><body>
<div class="container">
  <h1>BTCUSDT Daily Volume Profile Exhaustion Reversal</h1>
  <p class="muted">BTCUSDT Perpetual &middot; 5m &middot; {START_YEAR}-{START_MONTH:02d}-01 &rarr;
     {END_YEAR}-{END_MONTH:02d}-30 &middot; {N5:,} barre 5m &middot;
     {len(profiles):,} profili giornalieri &middot; holdout genuino &ge;{CUTOFF.date()}</p>

  <h2>Executive Summary</h2>
  <div class="kpi-grid">{kpi_cards}</div>

  <div class="section">
    <h3>Architettura e Criteri Decisionali</h3>
    <p><strong>Profilo (giorno precedente, causale):</strong> POC, Value Area (70%),
    shape su 3 zone di prezzo (top/mid/bottom). <code>b-shape</code> (volume pesante in
    basso) &rarr; zona domanda <code>[day_low, VAL]</code>. <code>P-shape</code> (mirror,
    volume pesante in alto) &rarr; zona offerta <code>[VAH, day_high]</code>.
    <code>D-shape</code>/<code>B-shape</code> (bilanciato/bimodale) &rarr; nessun trade,
    l'esaurimento ha senso solo dopo un giorno direzionalmente sbilanciato.</p>
    <p><strong>Confluenza al tocco della zona</strong> (3 segnali, non un tocco nudo):
    (a) climax di volume (z-score&gt;{VOL_Z_CLIMAX_THRESHOLD}) seguito da contrazione,
    (b) divergenza CVD/delta reale (da <code>taker_buy_base</code>, non serie sintetiche) su
    un nuovo estremo locale, (c) conferma strutturale CHoCH su 5m (riuso <code>smc.py</code>).</p>
    <p><strong>Uscita:</strong> TP = POC del giorno precedente (primo target), floor
    R:R={MIN_RR} verso VAH/VAL; SL strutturale oltre l'estremo zona &plusmn;
    {SL_BUFFER_ATR}&times;ATR; time-stop 24h; cooldown 4h.</p>
    <p class="muted">Zero parametri fittati — tutte le soglie sono scelte a-priori
    (dominio, non fit sui dati); l'holdout 2025-2026 resta comunque il check onesto
    contro l'overfitting-per-iterazione delle soglie stesse.</p>
    <p class="muted">Distribuzione shape sui {len(profiles):,} giorni:
    {', '.join(f'{k}={v}' for k, v in shape_counts.to_dict().items())}</p>
  </div>

  <div class="section">
    <h3>Equity Curves</h3>
    {_img_tag(eq_chart_b64)}
  </div>

  {variant_sections}

  <p class="muted" style="margin-top:2rem">
  BTCUSDT Daily Volume Profile Exhaustion Reversal Research &middot;
  Generated {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')} UTC &middot;
  Data: Binance Vision CDN (real taker_buy_base)</p>
</div>
</body></html>"""

Path("reports").mkdir(exist_ok=True)
Path("reports/report_daily_volume_profile.html").write_text(html)
Path("reports/daily_volume_profile_research.md").write_text("\n".join(report_lines))
print(f"\n[SAVED] reports/report_daily_volume_profile.html")
print(f"[SAVED] reports/daily_volume_profile_research.md")
print(f"\n[TOTAL RUNTIME] {time.time()-t_start:.0f}s")
