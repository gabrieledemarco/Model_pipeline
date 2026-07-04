"""
Signal generation for the S&P 500 sector-rotation strategy.

Three well-documented, non-overfit risk premia are combined:
  1. Relative momentum  — rank sector ETFs by composite 3/6/12-month return,
     hold the top-K equally weighted (Jegadeesh & Titman 1993).
  2. Absolute trend filter — Faber's 10-month SMA timing rule on SPY: fully
     invested only while price is above its trailing 10-month average,
     otherwise rotate to a cash/T-bill proxy (Faber 2007, Antonacci 2014).
  3. Volatility targeting — scale invested exposure down (never up, this is
     an unlevered long-only portfolio) when trailing realised vol of the
     selected basket exceeds the target, capping tail drawdowns.

All decisions at execution date `t` use data available strictly through
`t - 1` trading day — no lookahead.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

MOM_LOOKBACKS: Tuple[int, ...] = (63, 126, 252)   # ~3M, 6M, 12M trading days
TREND_MONTHS  = 10                                 # Faber 10-month SMA regime filter
TREND_BAND    = 0.02                               # hysteresis band around the SMA (2%)
VOL_WINDOW    = 20                                 # trading days for realised vol
VOL_TARGET    = 0.18                               # annualised vol target (~sector basket avg vol)
MAX_LEVERAGE  = 1.15                               # modest cap on the vol-target scalar
TOP_K         = 4
TRADING_DAYS  = 252


def month_end_series(price: pd.Series) -> pd.Series:
    """Resample a daily price series to month-end closes."""
    return price.resample("ME").last()


def trend_regime(spy_daily: pd.Series, decision_date: pd.Timestamp,
                  trend_months: int = TREND_MONTHS,
                  prev_regime: bool = True,
                  band: float = TREND_BAND) -> bool:
    """
    True = risk-on. Uses Faber's 10-month-SMA rule on SPY month-end closes,
    with a +/-`band` hysteresis around the SMA to damp single-month whipsaws:
    once risk-on, stays on until price falls `band` below the SMA; once
    risk-off, stays off until price rises `band` above the SMA.
    """
    me = month_end_series(spy_daily.loc[:decision_date])
    if len(me) < trend_months + 1:
        return True   # insufficient history yet -> default risk-on
    sma = me.rolling(trend_months).mean()
    px, avg = float(me.iloc[-1]), float(sma.iloc[-1])
    if prev_regime:
        return px >= avg * (1 - band)
    return px >= avg * (1 + band)


def momentum_scores(panel: pd.DataFrame, decision_date: pd.Timestamp,
                     lookbacks: Tuple[int, ...] = MOM_LOOKBACKS) -> pd.Series:
    """
    Composite momentum score (average of trailing N-day returns) as of
    decision_date. Sectors without enough history yet (e.g. XLC, XLRE in
    their early years) are returned as NaN and excluded by the caller.
    """
    hist = panel.loc[:decision_date]
    if len(hist) <= max(lookbacks):
        return pd.Series(index=panel.columns, dtype=float)
    scores = [hist.iloc[-1] / hist.iloc[-lb - 1] - 1.0 for lb in lookbacks]
    return sum(scores) / len(scores)


def realized_vol(returns: pd.Series, decision_date: pd.Timestamp,
                  window: int = VOL_WINDOW, default: float = VOL_TARGET) -> float:
    hist = returns.loc[:decision_date].tail(window)
    if len(hist) < window // 2:
        return default
    return float(hist.std() * np.sqrt(TRADING_DAYS))


def build_rebalance_plan(
    panel: pd.DataFrame,
    sectors: List[str],
    *,
    top_k: int = TOP_K,
    use_trend_filter: bool = True,
    use_vol_target: bool = True,
    vol_target: float = VOL_TARGET,
    max_leverage: float = MAX_LEVERAGE,
    mom_lookbacks: Tuple[int, ...] = MOM_LOOKBACKS,
    trend_months: int = TREND_MONTHS,
    trend_band: float = TREND_BAND,
) -> pd.DataFrame:
    """
    Compute target portfolio weights at every monthly rebalance execution date
    (the first trading day of each calendar month present in `panel`).

    `max_leverage` caps the vol-target scalar above 1.0 (e.g. 1.2 allows up
    to 20% leverage, financed/rebated at the cash rate, when realised vol is
    well below target); default 1.0 keeps the book fully unlevered.

    Returns a DataFrame indexed by execution date with columns
    [sectors..., 'CASH', 'regime', 'vol_scalar', 'selected'].
    """
    idx = panel.index
    months = idx.to_series().groupby([idx.year, idx.month]).min()
    exec_dates = pd.DatetimeIndex(sorted(months.values))

    warmup = max(mom_lookbacks) + 5
    rows = []
    prev_regime = True
    for exec_date in exec_dates:
        loc = idx.get_loc(exec_date)
        if loc < warmup:
            continue
        decision_date = idx[loc - 1]

        scores = momentum_scores(panel[sectors], decision_date, mom_lookbacks).dropna()
        if scores.empty:
            continue
        ranked = scores.sort_values(ascending=False)
        picks = list(ranked.index[:top_k])

        risk_on = trend_regime(panel["SPY"], decision_date, trend_months,
                                prev_regime=prev_regime, band=trend_band) \
                  if use_trend_filter else True
        prev_regime = risk_on

        if use_vol_target:
            basket_ret = panel[picks].pct_change().mean(axis=1)
            rv = realized_vol(basket_ret, decision_date, default=vol_target)
            vol_scalar = float(np.clip(vol_target / max(rv, 1e-6), 0.0, max_leverage))
        else:
            vol_scalar = 1.0

        invested = vol_scalar if risk_on else 0.0
        row = {s: 0.0 for s in sectors}
        for s in picks:
            row[s] = invested / top_k
        row["CASH"] = 1.0 - invested
        row["regime"] = "risk-on" if risk_on else "risk-off"
        row["vol_scalar"] = vol_scalar
        row["selected"] = ",".join(picks) if risk_on else ""
        row["date"] = exec_date
        rows.append(row)

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
