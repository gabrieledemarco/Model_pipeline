"""
Statistical analysis of signals and trades for the BTCUSDT strategy.

Provides:
  • Alpha-decay analysis (IC at multiple horizons)
  • Signal-score quantile → forward-return mapping
  • Cyclicality: monthly / day-of-week / hour-of-day seasonality
  • FFT cycle detection on daily closes
  • Trade statistics: regime, session, MAE/MFE distributions
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq


# ─────────────────────────────────────────────────────────────────────────────
# Forward return construction
# ─────────────────────────────────────────────────────────────────────────────

def forward_returns(df: pd.DataFrame,
                    horizons: List[int] = [1, 2, 4, 8, 12, 24]) -> pd.DataFrame:
    """Log-returns at h bars ahead, indexed to *df*."""
    log_c = np.log(df["close"])
    out   = {}
    for h in horizons:
        out[f"fwd_{h}h"] = log_c.shift(-h) - log_c
    return pd.DataFrame(out, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-decay
# ─────────────────────────────────────────────────────────────────────────────

def alpha_decay(signals: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal direction and horizon, compute:
      mean forward return, std, t-stat, p-value, IC (Spearman with composite).
    """
    rows = []
    score = signals["composite"]

    for col in fwd.columns:
        h = int(col.replace("fwd_", "").replace("h", ""))
        fwd_h = fwd[col].dropna()

        for sig_val, label in [(1, "Long"), (-1, "Short"), (0, "Neutral")]:
            mask = (signals["signal"] == sig_val) & fwd_h.notna()
            sub  = fwd_h[mask]
            if len(sub) < 5:
                continue
            t_s, p_v = stats.ttest_1samp(sub, 0) if len(sub) > 3 else (0, 1)
            common = score.index.intersection(sub.index)
            ic = 0.0
            if len(common) > 10:
                sp_r, _ = stats.spearmanr(score.loc[common].values,
                                          sub.loc[common].values)
                ic = float(sp_r) if np.isfinite(sp_r) else 0.0

            rows.append({
                "horizon_h":   h,
                "signal":      label,
                "mean_ret":    float(sub.mean()),
                "std_ret":     float(sub.std()),
                "t_stat":      float(t_s),
                "p_value":     float(p_v),
                "IC_spearman": ic,
                "n_obs":       len(sub),
            })

    return pd.DataFrame(rows).sort_values(["signal", "horizon_h"])


# ─────────────────────────────────────────────────────────────────────────────
# Score quantile → returns
# ─────────────────────────────────────────────────────────────────────────────

def score_quintile_returns(signals: pd.DataFrame,
                           fwd: pd.DataFrame,
                           col: str = "fwd_4h",
                           n_bins: int = 10) -> pd.DataFrame:
    """Average forward return per composite-score decile."""
    merged = pd.DataFrame({
        "score": signals["composite"],
        "fwd":   fwd[col],
    }).dropna()

    merged["bin"] = pd.qcut(merged["score"], n_bins, labels=False,
                            duplicates="drop")
    gb = merged.groupby("bin")
    return (gb.agg(score_mean=("score", "mean"),
                   avg_fwd_ret=("fwd", "mean"),
                   n_obs=("fwd", "count"))
              .reset_index())


# ─────────────────────────────────────────────────────────────────────────────
# Cyclicality / seasonality
# ─────────────────────────────────────────────────────────────────────────────

def monthly_seasonality(price: pd.Series) -> pd.DataFrame:
    """Average monthly log-return (from daily close series)."""
    ret  = np.log(price).diff().dropna()
    df   = pd.DataFrame({"ret": ret.values,
                          "month": ret.index.month,
                          "year":  ret.index.year},
                         index=ret.index)
    monthly = df.resample("ME")["ret"].sum()
    m_df    = pd.DataFrame({"ret": monthly.values,
                              "month": monthly.index.month})
    agg = m_df.groupby("month")["ret"].agg(["mean", "std", "count"])
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    agg.index = [names[m] for m in agg.index]
    return agg.rename(columns={"mean": "avg_ret",
                                "std":  "std_ret",
                                "count":"n_months"})


def dow_seasonality(price: pd.Series) -> pd.DataFrame:
    """Average daily log-return by day of week."""
    ret = np.log(price).diff().dropna()
    df  = pd.DataFrame({"ret": ret.values, "dow": ret.index.dayofweek})
    agg = df.groupby("dow")["ret"].agg(["mean", "std", "count"])
    names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    agg.index = [names[d] for d in agg.index]
    return agg.rename(columns={"mean":"avg_ret","std":"std_ret","count":"n_days"})


def hour_seasonality(price_1h: pd.Series) -> pd.DataFrame:
    """Average hourly log-return by hour-of-day (UTC)."""
    ret = np.log(price_1h).diff().dropna()
    df  = pd.DataFrame({"ret": ret.values, "hour": ret.index.hour})
    agg = df.groupby("hour")["ret"].agg(["mean", "std", "count"])
    return agg.rename(columns={"mean":"avg_ret","std":"std_ret","count":"n_bars"})


# ─────────────────────────────────────────────────────────────────────────────
# FFT cycle detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_cycles(price: pd.Series, min_period: int = 4,
                  max_period: int = 200) -> pd.DataFrame:
    """
    Detect dominant cycles in the log-price series via FFT.
    Returns cycles (period in bars, amplitude) sorted by amplitude desc.
    """
    lp = np.log(price.dropna().values)
    # Linear detrend
    x  = np.arange(len(lp))
    lp = lp - np.poly1d(np.polyfit(x, lp, 1))(x)

    N  = len(lp)
    yf = fft(lp)
    xf = fftfreq(N, d=1)      # cycles-per-bar

    pos     = xf > 0
    freqs   = xf[pos]
    amps    = 2.0 / N * np.abs(yf[pos])
    periods = 1.0 / freqs

    valid = (periods >= min_period) & (periods <= max_period)
    out   = pd.DataFrame({
        "period_bars": periods[valid],
        "amplitude":   amps[valid],
        "frequency":   freqs[valid],
    }).sort_values("amplitude", ascending=False).head(15).reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Trade-level breakdown
# ─────────────────────────────────────────────────────────────────────────────

def regime_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Win-rate and avg PnL by market regime."""
    if trades.empty or "regime" not in trades.columns:
        return pd.DataFrame()
    rows = []
    for regime, grp in trades.groupby("regime"):
        n = len(grp); w = int((grp["net_pnl"] > 0).sum())
        rows.append({"regime": regime, "n_trades": n,
                     "win_rate": w / n if n else 0,
                     "avg_pnl": grp["net_pnl"].mean(),
                     "total_pnl": grp["net_pnl"].sum()})
    return pd.DataFrame(rows).set_index("regime")


def session_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Win-rate and avg PnL by trading session (UTC hours)."""
    if trades.empty or "entry_ts" not in trades.columns:
        return pd.DataFrame()

    def _session(ts):
        h = pd.Timestamp(ts).hour
        if 0 <= h < 8:   return "Asian (00-08)"
        if 8 <= h < 13:  return "London (08-13)"
        if 13 <= h < 21: return "NY (13-21)"
        return "Late (21-00)"

    t = trades[["entry_ts", "net_pnl", "duration_h"]].copy()
    t["regime"] = t["entry_ts"].apply(_session)
    return regime_stats(t)


def score_vs_outcome(trades: pd.DataFrame) -> pd.DataFrame:
    """Group trades by score quintile, report win-rate and avg net PnL."""
    if trades.empty or "score" not in trades.columns:
        return pd.DataFrame()
    t = trades.copy()
    t["score_bin"] = pd.qcut(t["score"], 5, labels=False, duplicates="drop")
    def agg_fn(g):
        n = len(g); w = (g["net_pnl"] > 0).sum()
        return pd.Series({"n": n, "win_rate": w/n if n else 0,
                          "avg_pnl": g["net_pnl"].mean(),
                          "score_mean": g["score"].mean()})
    return t.groupby("score_bin").apply(agg_fn, include_groups=False).reset_index()


def compute_rolling_metrics(equity: pd.Series,
                             window_h: int = 168) -> pd.DataFrame:
    """Rolling Sharpe (7-day window) and rolling win-rate (if trades given)."""
    ret = equity.pct_change().dropna()
    roll_sharpe = (ret.rolling(window_h).mean() /
                   ret.rolling(window_h).std()) * np.sqrt(24 * 365)
    return pd.DataFrame({"rolling_sharpe": roll_sharpe}, index=equity.index)


def monthly_pnl(equity: pd.Series) -> pd.DataFrame:
    """
    Monthly P&L pivoted as year × month matrix, values in %.
    Suitable for heat-map plotting.
    """
    ret = equity.resample("ME").last().pct_change().dropna()
    df  = pd.DataFrame({
        "year":  ret.index.year,
        "month": ret.index.month,
        "ret":   ret.values,
    })
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df["month_name"] = df["month"].map(names)
    pivot = df.pivot(index="year", columns="month_name", values="ret")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    return pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
