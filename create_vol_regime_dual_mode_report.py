#!/usr/bin/env python3
"""
create_vol_regime_dual_mode_report.py
════════════════════════════════════════════════════════════════════════════
Volatility-Regime-Gated Dual-Mode — BTCUSDT 1H, 2022-01 -> 2026-06.

Grounded in analyze_btcusdt_edge.py's finding #4: daily ATR% regime is
extremely persistent day-over-day (P(HIGH tomorrow|HIGH today)=93.5%,
P(LOW|LOW)=94.1%, near-zero LOW<->HIGH flips) — the cleanest, least
tail-fragile signal found in the EDA (unlike the funding-extreme finding,
which rests on a handful of serially-correlated episodes).

Hypothesis: route capital between two ALREADY-VALIDATED building blocks by
which regime is persistent, instead of running either one unconditionally:
  LOW-vol persistent  -> mean-reversion (src/strategy/ou_mean_reversion.py,
                          the same OU z-score engine validated as S07, here
                          WITHOUT its HMM 4H gate — the vol-regime gate
                          replaces it)
  HIGH-vol persistent -> trend-following (src/strategy/smc.py's structural
                          CHoCH bias + order-block/discount-premium zone
                          entry, the same mechanism validated as V1 LTF SMC,
                          collapsed here to a single 1H timeframe)
  MID (transitional, only 88% persistent vs 93-94%) -> flat

3-way honest ablation (repo convention, e.g. the V1/V2/V3 SMC study and the
V4/V5/V6 CVD ablations): does the vol-regime gate actually help each leg,
or would the leg do just as well unconditionally?
  V1 Dual-Mode Gated  : as designed above
  V2 OU Always-On     : same OU engine, no vol-gate (any regime, no HMM)
  V3 SMC Always-On    : same SMC engine, no vol-gate (any regime)

Zero fitted parameters (OU refits its own OLS every bar by construction,
same as S07's live engine; SMC structure has none) -> no WFO split needed;
genuine 2025-2026 holdout still checked per repo convention.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.data_fetcher import fetch_binance_vision_klines
from src.strategy.indicators import atr as atr_fn
from src.strategy.ou_mean_reversion import build_ou_signals
from src.strategy.smc import compute_smc_features, smc_trend_signal
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)
from src.strategy.report_html import (
    _CSS, _fig_to_b64, _img_tag, _table,
    BG, PANEL, BORDER, WHITE, GRAY, GREEN, RED, GOLD, BLUE, PURPLE,
)

plt.style.use("dark_background")

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

VOL_PCTILE_WINDOW_DAYS = 90   # rolling window for the daily ATR% percentile rank
VOL_LOW_MAX = 0.33
VOL_HIGH_MIN = 0.67

# Mean-reversion leg (OU, no HMM gate) — same TP/SL convention as S07
# (Layer 3 of docs/VALIDATED_STRATEGIES_SPEC.md) but ATR1H (no ATR4H
# available without the HMM's 4H fetch this experiment deliberately drops).
MR_TP_FRAC, MR_SL_FRAC = 1.0, 1.0
MR_TIME_STOP_BARS = 48    # 2 days

# Trend leg (SMC structural, single-TF 1H) — same mechanics as V1 LTF SMC
# (SL_BUFFER_ATR/MIN_RR/fallbacks) collapsed to one timeframe.
TREND_SWING_LEN, TREND_INTERNAL_LEN = 30, 5
SL_BUFFER_ATR = 0.15
MIN_RR = 1.2
FALLBACK_SL_ATR = 1.0
FALLBACK_TP_ATR = 2.0
TREND_TIME_STOP_BARS = 96   # 4 days

COOLDOWN_BARS = 4

report_lines: list[str] = []
def w(line: str = "") -> None:
    print(line)
    report_lines.append(line)

t_start = time.time()
w(SEP)
w("BTCUSDT — Volatility-Regime-Gated Dual-Mode — Strategy Research")
w(SEP)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H BTCUSDT perpetual history...")
df1h = fetch_binance_vision_klines("1h", START_YEAR, START_MONTH, END_YEAR, END_MONTH, workers=6)
df1h["atr_14"] = atr_fn(df1h["high"], df1h["low"], df1h["close"], 14)
N1 = len(df1h)
IDX1 = df1h.index
HI1 = df1h["high"].values.astype(float)
LO1 = df1h["low"].values.astype(float)
CL1 = df1h["close"].values.astype(float)
ATR1 = np.where(df1h["atr_14"].shift(1).values > 0, df1h["atr_14"].shift(1).values, 1.0)
print(f"  {N1:,} bars, {IDX1[0]} -> {IDX1[-1]}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. VOLATILITY REGIME — daily ATR%, rolling percentile rank, causal
#    (profile_ts = day+1, same causality trick as volume_profile.py, so a
#    1H bar during day D+1 only ever sees day D's completed classification)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[REGIME] Classifying daily volatility regime (rolling ATR% percentile)...")
daily = df1h.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
daily_atr_pct = atr_fn(daily["high"], daily["low"], daily["close"], 14) / daily["close"]
daily_pctile = daily_atr_pct.rolling(VOL_PCTILE_WINDOW_DAYS, min_periods=30).apply(
    lambda x: (x <= x.iloc[-1]).mean(), raw=False)
bucket = pd.Series(
    np.select([daily_pctile <= VOL_LOW_MAX, daily_pctile >= VOL_HIGH_MIN], ["LOW", "HIGH"], default="MID"),
    index=daily.index,
)
regime_df = pd.DataFrame({"bucket": bucket}, index=daily.index + pd.Timedelta(days=1))
# Adding a plain Timedelta upcasts the DatetimeIndex's time unit (ms/us/ns
# mismatch) — cast back to IDX1's dtype so merge_asof's matching-dtype
# requirement doesn't choke (same fix as volume_profile.py's profile_ts).
regime_df.index = regime_df.index.astype(IDX1.dtype)
regime_aligned = align_htf_to_ltf(regime_df.index, regime_df, IDX1)
REGIME1 = regime_aligned["bucket"].fillna("MID").values
print(f"  regime distribution (1H bars): {pd.Series(REGIME1).value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. MEAN-REVERSION LEG — OU z-score (src/strategy/ou_mean_reversion.py)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[MR LEG] Building OU mean-reversion signals...")
ou_sig = build_ou_signals(df1h, window=30)
OU_SIGNAL1 = ou_sig["signal"].values.astype(int)
print(f"  OU signal bars: long={int((OU_SIGNAL1==1).sum())}  short={int((OU_SIGNAL1==-1).sum())}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. TREND LEG — SMC structural CHoCH bias + zone entry, single-TF 1H
# ═══════════════════════════════════════════════════════════════════════════
print("[TREND LEG] Building SMC structural signals (1H)...")
smc1h = compute_smc_features(df1h, swing_len=TREND_SWING_LEN, internal_len=TREND_INTERNAL_LEN, prefix="smc")
bias1h = smc_trend_signal(smc1h, prefix="smc").values.astype(int)
zone_bull1h = ((smc1h["smc_ob_bull_in"].values == 1) | (smc1h["smc_in_discount"].values == 1))
zone_bear1h = ((smc1h["smc_ob_bear_in"].values == 1) | (smc1h["smc_in_premium"].values == 1))

struct1h = causal_trend_state(CL1, HI1, LO1, left=8, right=8)
sl_basis_low = struct1h["last_pivot_low"].values
sl_basis_high = struct1h["last_pivot_high"].values
tp_basis_high = struct1h["target_high"].values
tp_basis_low = struct1h["target_low"].values

TREND_SIGNAL1 = np.where(
    (bias1h == 1) & zone_bull1h, 1, np.where((bias1h == -1) & zone_bear1h, -1, 0),
)
print(f"  Trend signal bars: long={int((TREND_SIGNAL1==1).sum())}  short={int((TREND_SIGNAL1==-1).sum())}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def simulate(mode: str) -> tuple[pd.DataFrame, float, float]:
    """mode: 'dual' (regime-gated), 'ou_always', 'smc_always'."""
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
                trades.append(dict(
                    entry_time=IDX1[entry_i], exit_time=IDX1[i], direction=dirn,
                    entry_price=entry_px, exit_price=exit_px, exit_reason=hit,
                    net_pnl=pnl, equity=cap, leg="",
                ))
                in_pos = False
                last_exit = i
            continue

        if i - last_exit < COOLDOWN_BARS:
            continue

        atrv = ATR1[i]
        if atrv <= 0:
            continue
        px = CL1[i]
        regime = REGIME1[i]

        use_mr = (mode == "ou_always") or (mode == "dual" and regime == "LOW")
        use_trend = (mode == "smc_always") or (mode == "dual" and regime == "HIGH")

        d = 0
        leg = ""
        if use_mr and OU_SIGNAL1[i] != 0:
            d = int(OU_SIGNAL1[i])
            leg = "MR"
            slp = px - d * MR_SL_FRAC * atrv
            tpp = px + d * MR_TP_FRAC * atrv
            hold_end_local = i + MR_TIME_STOP_BARS
        elif use_trend and TREND_SIGNAL1[i] != 0:
            d = int(TREND_SIGNAL1[i])
            leg = "TREND"
            sl_level = sl_basis_low[i] if d == 1 else sl_basis_high[i]
            tp_level = tp_basis_high[i] if d == 1 else tp_basis_low[i]
            slp = (sl_level - d * SL_BUFFER_ATR * atrv) if not np.isnan(sl_level) else (px - d * FALLBACK_SL_ATR * atrv)
            tpp = tp_level if not np.isnan(tp_level) else (px + d * FALLBACK_TP_ATR * atrv)
            risk_chk = abs(px - slp)
            if risk_chk > 0 and abs(tpp - px) / risk_chk < MIN_RR:
                tpp = px + d * max(MIN_RR * risk_chk, FALLBACK_TP_ATR * atrv)
            hold_end_local = i + TREND_TIME_STOP_BARS
        else:
            continue

        risk = abs(px - slp)
        if risk <= 0:
            continue
        risk_usd = INIT_CAP * RISK_PCT
        q = risk_usd / risk
        max_q = cap * MAX_LEV / px
        q = min(q, max_q)
        if q <= 0:
            continue

        dirn, entry_px, sl, tp, qty, entry_i = d, px, slp, tpp, q, i
        hold_end = hold_end_local
        in_pos = True

    trades_df = pd.DataFrame(trades, columns=[
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "exit_reason", "net_pnl", "equity", "leg",
    ])
    return trades_df, cap, mdd


print("\n[BACKTEST] Running 3 variants (dual-mode gated / OU always-on / SMC always-on)...")
v1_trades, v1_cap, v1_mdd = simulate("dual")
v2_trades, v2_cap, v2_mdd = simulate("ou_always")
v3_trades, v3_cap, v3_mdd = simulate("smc_always")

VARIANTS = {
    "V1 Dual-Mode Gated": dict(trades=v1_trades, cap=v1_cap, mdd=v1_mdd),
    "V2 OU Always-On": dict(trades=v2_trades, cap=v2_cap, mdd=v2_mdd),
    "V3 SMC Always-On": dict(trades=v3_trades, cap=v3_cap, mdd=v3_mdd),
}

for name, v in VARIANTS.items():
    n = len(v["trades"])
    ret = (v["cap"] / INIT_CAP - 1) * 100
    wr = (v["trades"]["net_pnl"] > 0).mean() * 100 if n else 0.0
    w(f"  {name:<22}  n={n:>5}  ret={ret:>+7.1f}%  mdd={v['mdd']*100:>6.1f}%  wr={wr:>5.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 6. VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VALIDATION] Monte Carlo + holdout + DSR...")

def mc_pair(trades_df: pd.DataFrame) -> tuple[dict, dict]:
    if len(trades_df) < 5:
        return {}, {}
    mc = run_monte_carlo(trades_df, INIT_CAP, N_SIMS)
    mc_blk = run_monte_carlo_block(trades_df, INIT_CAP, N_SIMS, block_size=20)
    return mc, mc_blk

dsr_family = []
for name, v in VARIANTS.items():
    trades = v["trades"]
    mc, mc_blk = mc_pair(trades)
    holdout = trades[trades["entry_time"] >= CUTOFF] if len(trades) else trades
    v["mc"] = mc
    v["mc_block"] = mc_blk
    v["holdout_trades"] = holdout
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
    v["verdict"] = (n_trades >= 30 and ret > 0 and p_ruin < 0.10 and v["dsr"] >= DSR_THRESHOLD and holdout_ret > 0)
    w(f"  {name:<22}  n>=30:{n_trades>=30}  ret>0:{ret>0}  P(ruin)<10%:{p_ruin<0.10}  "
      f"DSR>=0.95:{v['dsr']>=DSR_THRESHOLD}  holdout>0:{holdout_ret>0}  -> "
      f"{'VALIDATA' if v['verdict'] else 'NON VALIDATA'}")

w("\n  Breakdown per anno:")
for name, v in VARIANTS.items():
    w(f"\n    {name}")
    trades = v["trades"]
    if trades.empty:
        continue
    for yr, grp in trades.groupby(trades["entry_time"].dt.year):
        ret_yr = grp["net_pnl"].sum() / INIT_CAP * 100
        wr_yr = (grp["net_pnl"] > 0).mean() * 100
        w(f"      {yr:>6}  n={len(grp):>5}  ret={ret_yr:>+7.1f}%  wr={wr_yr:>5.1f}%")

if len(v1_trades):
    w("\n  V1 Dual-Mode — breakdown by leg:")
    for leg, grp in v1_trades.groupby("leg"):
        ret_leg = grp["net_pnl"].sum() / INIT_CAP * 100
        wr_leg = (grp["net_pnl"] > 0).mean() * 100
        w(f"    {leg:<6}  n={len(grp):>5}  ret={ret_leg:>+7.1f}%  wr={wr_leg:>5.1f}%")

print(f"\n[DONE] runtime: {time.time()-t_start:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════
# 7. HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n[REPORT] Building HTML report...")
COLORS = {"V1 Dual-Mode Gated": GOLD, "V2 OU Always-On": BLUE, "V3 SMC Always-On": PURPLE}

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
        return pd.Series([INIT_CAP], index=[IDX1[0]])
    idx = pd.DatetimeIndex([IDX1[0]] + list(trades_df["exit_time"]))
    vals = [INIT_CAP] + list(trades_df["equity"])
    return pd.Series(vals, index=idx)

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
eq_chart_b64 = _fig_to_b64(fig)

kpi_cards = ""
variant_sections = ""
for name, v in VARIANTS.items():
    n = len(v["trades"])
    ret = (v["cap"] / INIT_CAP - 1) * 100
    wr = (v["trades"]["net_pnl"] > 0).mean() * 100 if n else 0.0
    mc = v["mc"]
    p_profit = mc.get("p_profit", 0.0) if mc else 0.0
    p_ruin = mc.get("p_ruin", 1.0) if mc else 1.0
    holdout = v["holdout_trades"]
    holdout_ret = (holdout["net_pnl"].sum() / INIT_CAP * 100) if len(holdout) else 0.0
    verdict_badge = ('<span class="badge badge-green">VALIDATA</span>' if v["verdict"]
                      else '<span class="badge badge-red">NON VALIDATA</span>')
    kpi_cards += f"""
<div class="card" style="border-left:3px solid {COLORS[name]}">
  <div class="val {'green' if ret>0 else 'red'}">{ret:+.1f}%</div>
  <div class="lbl">{name} — Return</div>
</div>"""
    stat_rows = [
        ["Trades", f"{n:,}"], ["Win Rate", f"{wr:.1f}%"], ["Max Drawdown", f"{v['mdd']*100:.1f}%"],
        ["Sharpe (trade-level)", f"{v['sharpe_hat']:+.3f}"], ["Deflated Sharpe Ratio", f"{v['dsr']:.3f}"],
        ["MC P(profit)", f"{p_profit*100:.1f}%"], ["MC P(ruin)", f"{p_ruin*100:.1f}%"],
        ["Holdout 2025-2026 return", f"{holdout_ret:+.1f}%"], ["Holdout trades", f"{len(holdout):,}"],
    ]
    variant_sections += f"""
<div class="section"><h3>{name} &nbsp; {verdict_badge}</h3>{_table(["Metric", "Value"], stat_rows)}</div>"""

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Volatility-Regime-Gated Dual-Mode</title>
<style>{_CSS}</style></head><body>
<div class="container">
  <h1>BTCUSDT Volatility-Regime-Gated Dual-Mode</h1>
  <p class="muted">BTCUSDT Perpetual &middot; 1H &middot; {START_YEAR}-{START_MONTH:02d} &rarr;
     {END_YEAR}-{END_MONTH:02d} &middot; {N1:,} barre 1H &middot; holdout genuino &ge;{CUTOFF.date()}</p>
  <h2>Executive Summary</h2>
  <div class="kpi-grid">{kpi_cards}</div>
  <div class="section">
    <h3>Razionale</h3>
    <p>Basato su analyze_btcusdt_edge.py: persistenza regime volatilit&agrave; giornaliera
    93-94% (P(HIGH domani|HIGH oggi), P(LOW domani|LOW oggi)) &mdash; il segnale pi&ugrave;
    pulito trovato nell'EDA. LOW-vol persistente &rarr; mean-reversion (OU z-score, motore
    di S07 senza il gate HMM 4H). HIGH-vol persistente &rarr; trend-following (struttura
    SMC CHoCH + zona, motore di V1, single-TF 1H). MID &rarr; flat.</p>
    <p class="muted">Zero parametri fittati (OU rifitta la propria OLS ogni barra, SMC
    è puramente strutturale) &mdash; nessuno split WFO necessario per il segnale stesso;
    holdout 2025-2026 comunque verificato.</p>
  </div>
  <div class="section"><h3>Equity Curves</h3>{_img_tag(eq_chart_b64)}</div>
  {variant_sections}
  <p class="muted" style="margin-top:2rem">Generated {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')} UTC</p>
</div></body></html>"""

Path("reports").mkdir(exist_ok=True)
Path("reports/report_vol_regime_dual_mode.html").write_text(html)
Path("reports/vol_regime_dual_mode_research.md").write_text("\n".join(report_lines))
print("\n[SAVED] reports/report_vol_regime_dual_mode.html")
print(f"[TOTAL RUNTIME] {time.time()-t_start:.0f}s")
