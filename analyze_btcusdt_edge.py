#!/usr/bin/env python3
"""
analyze_btcusdt_edge.py — exploratory data analysis, NOT a strategy backtest.

Purpose: find where BTCUSDT actually has statistically exploitable structure
before designing another strategy, instead of starting from an a-priori
trading idea (which is what led to the Daily Volume Profile strategy's
honest failure — the base mean-reversion premise had no edge in the data).

Four questions, each a well-known candidate crypto-perp edge, checked
directly against 2022-2026 data before building anything on top of them:
  1. Momentum vs mean-reversion by horizon, and how that flips by realized-
     vol regime and by HMM trend regime (informs: should the next strategy
     be trend-following or mean-reverting, and when).
  2. Funding rate extremes -> forward returns (crowded positioning /
     liquidation-cascade setup — classic perp-specific edge, untested here
     until now despite real funding data being available in the repo).
  3. Session / hour-of-day effects (already partially exploited by ICT
     Silver Bullet's NY AM/PM windows — checked here with raw stats instead
     of assumed).
  4. Volatility regime persistence (does a high-vol day predict a high-vol
     tomorrow? informs whether a vol-regime filter is worth adding).
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_binance_vision_funding
from src.strategy.indicators import atr as atr_fn
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features

pd.set_option("display.width", 140)

print("=" * 78)
print("BTCUSDT Edge Discovery — 2022-01 -> 2026-06, 1H")
print("=" * 78)

df1h = fetch_binance_vision_klines("1h", 2022, 1, 2026, 6, workers=6)
close = df1h["close"]
ret1h = np.log(close / close.shift(1))
df1h["atr_14"] = atr_fn(df1h["high"], df1h["low"], df1h["close"], 14)
df1h["atr_pct"] = df1h["atr_14"] / close
print(f"\n{len(df1h):,} bars, {df1h.index[0]} -> {df1h.index[-1]}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. MOMENTUM vs MEAN-REVERSION BY HORIZON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("1. RETURN AUTOCORRELATION BY HORIZON")
print("=" * 78)
print("(sign-correlation of past-N-bar return vs next-N-bar return — >0 = trend/momentum, <0 = mean-reversion)")

def sign_autocorr(ret: pd.Series, horizon: int) -> tuple[float, float]:
    past = ret.rolling(horizon).sum()
    fwd = ret.rolling(horizon).sum().shift(-horizon)
    df = pd.DataFrame({"past": past, "fwd": fwd}).dropna()
    corr = df["past"].corr(df["fwd"])
    sign_corr = np.sign(df["past"]).corr(np.sign(df["fwd"]))
    return corr, sign_corr

print(f"\n  {'Horizon':>10}  {'PearsonCorr':>12}  {'SignCorr':>10}  {'n':>8}")
for h, label in [(1, "1h"), (4, "4h"), (8, "8h"), (24, "24h (1d)"), (72, "72h (3d)")]:
    c, sc = sign_autocorr(ret1h, h)
    print(f"  {label:>10}  {c:>+12.4f}  {sc:>+10.4f}  {len(df1h)-2*h:>8,}")

# Conditional on trailing realized-vol regime (median split of rolling 7d ATR%)
print("\n  Same, split by trailing volatility regime (ATR% percentile, 7d rolling):")
vol_pctile = df1h["atr_pct"].rolling(500, min_periods=100).apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)
high_vol = vol_pctile > 0.5
for h, label in [(4, "4h"), (24, "24h")]:
    past = ret1h.rolling(h).sum()
    fwd = ret1h.rolling(h).sum().shift(-h)
    df = pd.DataFrame({"past": past, "fwd": fwd, "high_vol": high_vol}).dropna()
    for regime, sub in df.groupby("high_vol"):
        label_r = "HIGH-vol" if regime else "LOW-vol"
        sc = np.sign(sub["past"]).corr(np.sign(sub["fwd"]))
        print(f"    {label:>6} | {label_r:>8}  sign_corr={sc:>+.4f}  n={len(sub):,}")

# Conditional on HMM regime (fit ONCE on full history — descriptive only,
# not a tradeable signal; walk-forward refit is what the live strategies do)
print("\n  Same, split by HMM regime (BULL/BEAR/SIDEWAYS, fit full-sample, descriptive only):")
model, sorted_idx = fit_hmm(df1h, n_states=3, random_state=42)
hmm_feats = predict_hmm_features(model, sorted_idx, df1h)
regime_label = pd.Series(
    np.select(
        [hmm_feats["hmm_state"] == 2, hmm_feats["hmm_state"] == 0],
        ["BULL", "BEAR"], default="SIDEWAYS",
    ), index=df1h.index,
)
for h, label in [(4, "4h"), (24, "24h")]:
    past = ret1h.rolling(h).sum()
    fwd = ret1h.rolling(h).sum().shift(-h)
    df = pd.DataFrame({"past": past, "fwd": fwd, "regime": regime_label}).dropna()
    for regime, sub in df.groupby("regime"):
        sc = np.sign(sub["past"]).corr(np.sign(sub["fwd"]))
        mean_fwd = sub["fwd"].mean() * 100
        print(f"    {label:>6} | {regime:>9}  sign_corr={sc:>+.4f}  mean_fwd_ret={mean_fwd:>+.3f}%  n={len(sub):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. FUNDING RATE EXTREMES -> FORWARD RETURNS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("2. FUNDING RATE EXTREMES -> FORWARD RETURNS (event study)")
print("=" * 78)
funding_8h = fetch_binance_vision_funding(2022, 1, 2026, 6)
print(f"  {len(funding_8h):,} funding settlements ({funding_8h.index[0]} -> {funding_8h.index[-1]})")

funding_pctile = funding_8h.rank(pct=True)
fwd_ret_1d = np.log(close.reindex(funding_8h.index, method="ffill").shift(-24) /
                     close.reindex(funding_8h.index, method="ffill"))
fwd_ret_3d = np.log(close.reindex(funding_8h.index, method="ffill").shift(-72) /
                     close.reindex(funding_8h.index, method="ffill"))
fdf = pd.DataFrame({"funding": funding_8h, "pctile": funding_pctile,
                     "fwd_1d": fwd_ret_1d, "fwd_3d": fwd_ret_3d}).dropna()

buckets = [(0, 0.05, "bottom 5% (most negative)"), (0.05, 0.20, "5-20%"),
           (0.20, 0.80, "20-80% (normal)"), (0.80, 0.95, "80-95%"),
           (0.95, 1.0, "top 5% (most positive)")]
print(f"\n  {'Bucket':>28}  {'n':>6}  {'mean_funding':>13}  {'fwd_1d%':>9}  {'fwd_3d%':>9}")
for lo, hi, label in buckets:
    sub = fdf[(fdf["pctile"] >= lo) & (fdf["pctile"] < hi if hi < 1.0 else fdf["pctile"] <= hi)]
    print(f"  {label:>28}  {len(sub):>6}  {sub['funding'].mean()*100:>12.4f}%  "
          f"{sub['fwd_1d'].mean()*100:>+8.3f}%  {sub['fwd_3d'].mean()*100:>+8.3f}%")

corr_1d = fdf["funding"].corr(fdf["fwd_1d"])
corr_3d = fdf["funding"].corr(fdf["fwd_3d"])
print(f"\n  Correlation funding vs fwd_1d: {corr_1d:+.4f}   funding vs fwd_3d: {corr_3d:+.4f}")
print("  (negative correlation = contrarian edge: extreme positive funding -> price falls, and vice versa)")

# ═══════════════════════════════════════════════════════════════════════════
# 3. SESSION / HOUR-OF-DAY EFFECTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("3. HOUR-OF-DAY EFFECTS (UTC)")
print("=" * 78)
hod = pd.DataFrame({"ret": ret1h, "hour": df1h.index.hour, "abs_ret": ret1h.abs()}).dropna()
hod_stats = hod.groupby("hour").agg(mean_ret=("ret", "mean"), vol=("abs_ret", "mean"),
                                     pos_rate=("ret", lambda x: (x > 0).mean()))
hod_stats["mean_ret_pct"] = hod_stats["mean_ret"] * 100
hod_stats["vol_pct"] = hod_stats["vol"] * 100
print(f"\n  {'UTC Hour':>9}  {'mean_ret%':>10}  {'avg|ret|%':>10}  {'pos_rate':>9}")
for h, row in hod_stats.iterrows():
    marker = "  <-- NY AM (ICT)" if h in (13, 14) else ("  <-- NY PM (ICT)" if h in (19, 20) else "")
    print(f"  {h:>9}  {row['mean_ret_pct']:>+10.4f}  {row['vol_pct']:>10.4f}  {row['pos_rate']:>9.3f}{marker}")

print("\n  Day-of-week effect:")
dow = pd.DataFrame({"ret": ret1h, "dow": df1h.index.day_name()}).dropna()
dow_stats = dow.groupby("dow").agg(mean_ret=("ret", "mean"), vol=("ret", "std"))
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for d in dow_order:
    if d in dow_stats.index:
        r = dow_stats.loc[d]
        print(f"    {d:<10}  mean_ret={r['mean_ret']*100:>+.4f}%  std={r['vol']*100:.4f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 4. VOLATILITY REGIME PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("4. VOLATILITY REGIME PERSISTENCE (daily ATR% percentile, day-over-day)")
print("=" * 78)
daily = df1h.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
daily_atr_pct = (atr_fn(daily["high"], daily["low"], daily["close"], 14) / daily["close"])
daily_pctile = daily_atr_pct.rank(pct=True)
today_bucket = pd.cut(daily_pctile, [0, 0.33, 0.67, 1.0], labels=["LOW", "MID", "HIGH"])
tomorrow_bucket = today_bucket.shift(-1)
trans = pd.crosstab(today_bucket, tomorrow_bucket, normalize="index")
print("\n  Transition matrix P(tomorrow_bucket | today_bucket):")
print(trans.round(3).to_string())

print("\n" + "=" * 78)
print("DONE")
print("=" * 78)
