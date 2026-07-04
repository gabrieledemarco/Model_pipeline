"""
Portfolio backtest engine for the S&P 500 sector-rotation strategy.

Execution model
────────────────
• Target weights are decided using data available strictly through the
  trading day *before* the execution date (see signals.build_rebalance_plan)
  — no lookahead.
• Rebalancing executes at the *closing* adjusted price of the execution date
  itself — the standard convention in monthly tactical-allocation research
  (Faber 2007, Antonacci 2014).
• Between rebalances, weights are NOT reset daily — positions drift with
  each asset's own return (buy-and-hold within the month).
• A one-way transaction cost (`cost_bps`) is applied to traded notional
  (turnover) at every rebalance.
• Every asset holding period between two consecutive rebalances is recorded
  as a `Trade` (including CASH) for trade-level analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

INIT_CAP     = 100_000.0
COST_BPS     = 0.0005     # 5 bps one-way cost on turnover notional
TRADING_DAYS = 252


@dataclass
class Trade:
    asset:       str
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    entry_price: float
    exit_price:  float
    weight:      float
    shares:      float
    notional:    float
    net_pnl:     float
    duration_d:  int
    regime:      str


def run_portfolio_backtest(
    panel: pd.DataFrame,
    cash_nav: pd.Series,
    plan: pd.DataFrame,
    sectors: List[str],
    initial_capital: float = INIT_CAP,
    cost_bps: float = COST_BPS,
    whole_shares: bool = False,
    commission_per_leg: float = 0.0,
) -> dict:
    """
    Simulate the monthly sector-rotation portfolio over `panel`'s date range.

    `whole_shares` and `commission_per_leg` model small-account frictions that
    are invisible at institutional size: most EU retail brokers do not offer
    fractional ETF shares, and many charge a flat fee per order rather than a
    pure bps spread. Set `whole_shares=True` to floor every position to an
    integer share count (uninvested residual falls back to CASH) and
    `commission_per_leg` to a flat currency cost applied to every non-zero
    turnover leg at each rebalance, to see the drag this adds at small capital.

    Returns dict with keys: equity, drawdown, trades, kpis
    """
    if plan.empty:
        idx = panel.index
        equity = pd.Series(initial_capital, index=idx)
        dd = pd.Series(0.0, index=idx)
        return {"equity": equity, "drawdown": dd, "trades": pd.DataFrame(),
                "kpis": compute_kpis(equity, dd, pd.DataFrame(), initial_capital)}

    assets = sectors + ["CASH"]
    # Later-inception sectors (e.g. XLC, XLRE) carry leading NaNs; they are
    # never allocated a weight before the signal layer deems them eligible,
    # but 0-weight * NaN-price would poison the equity sum, so zero-fill.
    prices = panel[sectors].fillna(0.0).copy()
    prices["CASH"] = cash_nav.reindex(panel.index).ffill().bfill()
    idx = panel.index

    exec_set = set(plan.index)
    shares = {a: 0.0 for a in assets}
    equity_arr = np.full(len(idx), np.nan)
    trades: List[Trade] = []
    open_positions: dict = {}
    prev_weights = {a: 0.0 for a in assets}
    started = False

    for i, dt in enumerate(idx):
        price_row = prices.iloc[i]

        if dt in exec_set:
            total_value = (sum(shares[a] * price_row[a] for a in assets)
                            if started else initial_capital)

            target_weights = {a: float(plan.loc[dt].get(a, 0.0)) for a in assets}
            turnover = sum(abs(target_weights[a] - prev_weights[a]) for a in assets)
            total_value -= total_value * turnover * cost_bps

            n_legs_traded = sum(1 for a in assets
                                 if abs(target_weights[a] - prev_weights[a]) > 1e-9 and a != "CASH")
            total_value -= n_legs_traded * commission_per_leg

            # close out previous holding period -> trade records
            for a, pos in open_positions.items():
                exit_price = price_row[a]
                pnl = pos["shares"] * (exit_price - pos["entry_price"])
                trades.append(Trade(
                    asset=a, entry_ts=pos["entry_ts"], exit_ts=dt,
                    entry_price=pos["entry_price"], exit_price=exit_price,
                    weight=pos["weight"], shares=pos["shares"],
                    notional=pos["shares"] * pos["entry_price"], net_pnl=pnl,
                    duration_d=(dt - pos["entry_ts"]).days, regime=pos["regime"],
                ))
            open_positions = {}

            regime = str(plan.loc[dt].get("regime", ""))
            uninvested = 0.0
            for a in assets:
                if a == "CASH":
                    continue
                w = target_weights[a]
                raw_shares = (total_value * w) / price_row[a] if price_row[a] > 0 else 0.0
                if whole_shares:
                    shares[a] = float(np.floor(raw_shares))
                    uninvested += raw_shares * price_row[a] - shares[a] * price_row[a]
                else:
                    shares[a] = raw_shares
                if w > 1e-9 and shares[a] > 0:
                    open_positions[a] = dict(entry_ts=dt, entry_price=price_row[a],
                                              shares=shares[a], weight=w, regime=regime)
            cash_w = target_weights["CASH"]
            cash_value = total_value * cash_w + uninvested
            shares["CASH"] = cash_value / price_row["CASH"] if price_row["CASH"] > 0 else 0.0
            if cash_value > 1e-9:
                open_positions["CASH"] = dict(entry_ts=dt, entry_price=price_row["CASH"],
                                               shares=shares["CASH"], weight=cash_w, regime=regime)
            prev_weights = target_weights
            started = True

        equity_arr[i] = (sum(shares[a] * price_row[a] for a in assets)
                          if started else initial_capital)

    # close remaining open positions at the final bar
    last_dt = idx[-1]
    last_price_row = prices.iloc[-1]
    for a, pos in open_positions.items():
        if pos["entry_ts"] == last_dt:
            continue
        exit_price = last_price_row[a]
        pnl = pos["shares"] * (exit_price - pos["entry_price"])
        trades.append(Trade(
            asset=a, entry_ts=pos["entry_ts"], exit_ts=last_dt,
            entry_price=pos["entry_price"], exit_price=exit_price,
            weight=pos["weight"], shares=pos["shares"],
            notional=pos["shares"] * pos["entry_price"], net_pnl=pnl,
            duration_d=(last_dt - pos["entry_ts"]).days, regime=pos["regime"],
        ))

    equity = pd.Series(equity_arr, index=idx)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    trades_df = (pd.DataFrame([t.__dict__ for t in trades]).sort_values("entry_ts")
                 .reset_index(drop=True) if trades else pd.DataFrame())

    return {
        "equity": equity,
        "drawdown": drawdown,
        "trades": trades_df,
        "kpis": compute_kpis(equity, drawdown, trades_df, initial_capital),
    }


def compute_kpis(equity: pd.Series, drawdown: pd.Series,
                  trades_df: pd.DataFrame, init_cap: float) -> dict:
    """Daily-bar performance metrics (252 trading days/year)."""
    equity = equity.dropna()
    if equity.empty:
        return dict(total_return=0.0, final_equity=init_cap, cagr=0.0, sharpe=0.0,
                     sortino=0.0, calmar=0.0, max_drawdown=0.0, n_trades=0,
                     win_rate=0, profit_factor=0, expectancy=0, avg_duration_d=0)

    final = float(equity.iloc[-1])
    tot_ret = final / init_cap - 1.0
    n_years = max(len(equity) / TRADING_DAYS, 1e-6)
    cagr = (final / init_cap) ** (1.0 / n_years) - 1.0

    day_ret = equity.pct_change().dropna()
    ann_vol = float(day_ret.std() * np.sqrt(TRADING_DAYS))
    ann_ret = float(day_ret.mean() * TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    neg = day_ret[day_ret < 0]
    sortino = (ann_ret / (neg.std() * np.sqrt(TRADING_DAYS))
               if len(neg) > 1 and neg.std() > 0 else 0.0)

    max_dd = float(drawdown.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    risky = trades_df[trades_df["asset"] != "CASH"] if (not trades_df.empty and
             "asset" in trades_df.columns) else trades_df

    if risky is None or len(risky) == 0:
        return dict(total_return=tot_ret, final_equity=final, cagr=cagr,
                    sharpe=sharpe, sortino=sortino, calmar=calmar,
                    max_drawdown=max_dd, n_trades=0, win_rate=0,
                    avg_win=0, avg_loss=0, profit_factor=0, expectancy=0,
                    avg_duration_d=0)

    wins = risky["net_pnl"] > 0
    n = len(risky)
    n_w = int(wins.sum())
    n_l = n - n_w
    wr = n_w / n

    avg_w = float(risky.loc[wins, "net_pnl"].mean()) if n_w else 0.0
    avg_l = float(risky.loc[~wins, "net_pnl"].mean()) if n_l else 0.0
    pf = (avg_w * n_w / (-avg_l * n_l)) if (n_l > 0 and avg_l < 0) else 0.0
    exp = wr * avg_w + (1 - wr) * avg_l
    avg_dur = float(risky["duration_d"].mean())

    return dict(
        total_return=tot_ret, final_equity=final, cagr=cagr,
        sharpe=sharpe, sortino=sortino, calmar=calmar, max_drawdown=max_dd,
        n_trades=n, n_wins=n_w, n_losses=n_l, win_rate=wr,
        avg_win=avg_w, avg_loss=avg_l, profit_factor=pf, expectancy=exp,
        avg_duration_d=avg_dur,
    )


def monthly_return_trades(equity: pd.Series) -> pd.DataFrame:
    """
    Convert a daily equity curve into a "trade" log of calendar-month returns
    (entry_ts / exit_ts / net_pnl), i.e. one synthetic round-trip per month.

    This is the unit fed to src.strategy.monte_carlo.run_monte_carlo, which
    bootstraps *independent, sequential* trade returns and compounds them —
    appropriate here because our real positions overlap in time (concurrently
    holding several sectors + cash), so individual position P&L is not a
    valid Monte-Carlo resampling unit for the *portfolio's* path. Monthly
    portfolio returns, on the other hand, are sequential and non-overlapping.
    """
    equity = equity.dropna()
    monthly = equity.resample("ME").last()
    rets = monthly.pct_change()
    rets.iloc[0] = monthly.iloc[0] / equity.iloc[0] - 1.0

    rows = []
    prev_eq = float(equity.iloc[0])
    for dt, r in rets.items():
        pnl = prev_eq * r
        rows.append({"entry_ts": dt, "exit_ts": dt, "net_pnl": pnl})
        prev_eq = prev_eq + pnl
    return pd.DataFrame(rows)
