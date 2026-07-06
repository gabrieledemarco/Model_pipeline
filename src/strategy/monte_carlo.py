"""
Monte Carlo simulation for the BTCUSDT strategy.

Methodology: bootstrap resampling of realized trade returns (expressed as
fractions of equity-at-entry, applied compoundingly) to estimate the
distribution of:
  - Final equity
  - Maximum drawdown
  - Total return
  - Sharpe ratio (trade-level, not time-series)
  - Probability of ruin  (equity < 50 % of initial)
  - Probability of profit (equity > initial)

This is a trade-level simulation, not a bar-level one, so it captures the
edge's statistical uncertainty without depending on the specific order of
market conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trade_pct_returns(trades_df: pd.DataFrame,
                       initial_capital: float) -> np.ndarray:
    """
    Replay the original trade sequence once to get equity at each trade entry,
    then express each trade's net PnL as a fraction of that equity.
    This preserves compounding for the bootstrap resampling.
    """
    eq = float(initial_capital)
    pcts = []
    for pnl in trades_df["net_pnl"].to_numpy():
        pct = pnl / eq if eq > 0 else 0.0
        pcts.append(pct)
        eq = max(eq + pnl, 1e-6)
    return np.array(pcts)


def _sim_path(pct_returns: np.ndarray, initial_capital: float) -> np.ndarray:
    """Build an equity path from fractional trade returns (compounded)."""
    n = len(pct_returns)
    eq = np.empty(n + 1)
    eq[0] = initial_capital
    for t, r in enumerate(pct_returns):
        eq[t + 1] = max(eq[t] * (1.0 + r), 0.0)
    return eq


def _max_dd(eq: np.ndarray) -> float:
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / np.where(running_max > 0, running_max, 1)
    return float(dd.min())


def _sharpe(pct_returns: np.ndarray) -> float:
    if len(pct_returns) < 2:
        return 0.0
    std = pct_returns.std()
    if std <= 0:
        return 0.0
    return float(pct_returns.mean() / std * np.sqrt(len(pct_returns)))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    trades_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    n_sims: int = 1_000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap Monte Carlo on realized trade P&L.

    Parameters
    ----------
    trades_df       : DataFrame with at least a 'net_pnl' column.
    initial_capital : Starting equity for each simulated path.
    n_sims          : Number of bootstrap resamples.
    seed            : RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        paths        : ndarray (n_sims, n_trades+1)  equity paths
        final_equity : ndarray (n_sims,)
        max_drawdown : ndarray (n_sims,)   negative fractions
        total_return : ndarray (n_sims,)   fractions
        sharpe       : ndarray (n_sims,)   trade-level Sharpe
        summary      : DataFrame  percentile table
        p_ruin       : float  P(final_equity < 0.5 × initial)
        p_profit     : float  P(final_equity > initial)
        n_trades     : int
        n_sims       : int
        initial_capital : float
    """
    if trades_df is None or trades_df.empty or "net_pnl" not in trades_df.columns:
        return {}

    rng = np.random.default_rng(seed)
    base_pcts = _trade_pct_returns(trades_df, initial_capital)
    n_trades  = len(base_pcts)

    paths        = np.zeros((n_sims, n_trades + 1))
    final_equity = np.zeros(n_sims)
    max_drawdown = np.zeros(n_sims)
    sharpes      = np.zeros(n_sims)

    for s in range(n_sims):
        sampled = rng.choice(base_pcts, size=n_trades, replace=True)
        eq = _sim_path(sampled, initial_capital)
        paths[s]        = eq
        final_equity[s] = eq[-1]
        max_drawdown[s] = _max_dd(eq)
        sharpes[s]      = _sharpe(sampled)

    total_return = (final_equity - initial_capital) / initial_capital

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    summary = pd.DataFrame({
        "percentile":   pcts,
        "final_equity": np.percentile(final_equity, pcts),
        "total_return%": np.percentile(total_return * 100, pcts),
        "max_drawdown%": np.percentile(max_drawdown * 100, pcts),
        "sharpe":        np.percentile(sharpes, pcts),
    })

    return {
        "paths":           paths,
        "final_equity":    final_equity,
        "max_drawdown":    max_drawdown,
        "total_return":    total_return,
        "sharpe":          sharpes,
        "summary":         summary,
        "p_ruin":          float((final_equity < 0.5 * initial_capital).mean()),
        "p_profit":        float((final_equity > initial_capital).mean()),
        "n_trades":        n_trades,
        "n_sims":          n_sims,
        "initial_capital": initial_capital,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
#
# Corrects the observed Sharpe ratio for two distinct biases:
#   1. Non-normality of trade returns (skewness/kurtosis)   → PSR
#   2. Selection bias from testing N variants and reporting the best one → SR0
#
# Reference: "The Deflated Sharpe Ratio: Correcting for Selection Bias,
# Backtest Overfitting and Non-Normality", Bailey & López de Prado, JPM 2014.
# ─────────────────────────────────────────────────────────────────────────────

_EULER_MASCHERONI = 0.5772156649015329


def trade_level_sharpe(pnls) -> float:
    """Sharpe ratio computed directly on a trade P&L sequence (not %-returns)."""
    arr = np.asarray(pnls, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = arr.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(len(arr)))


def _skew_kurt(pnls) -> tuple:
    """Sample skewness (γ3) and excess-kurtosis+3 (γ4, Pearson convention)."""
    arr = np.asarray(pnls, dtype=float)
    n = len(arr)
    if n < 3:
        return 0.0, 3.0
    std = arr.std(ddof=1)
    if std <= 0:
        return 0.0, 3.0
    z = (arr - arr.mean()) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))          # Pearson kurtosis (normal = 3.0)
    return skew, kurt


def expected_max_sharpe(sr_std_across_trials: float, n_trials: int) -> float:
    """
    SR0 = expected maximum Sharpe ratio observable by chance alone when
    N independent (or weakly correlated) strategies with true skill = 0
    are backtested, given the empirical cross-sectional dispersion of
    their Sharpe ratios.

    This is the "how good does the best-of-N look purely by luck" benchmark
    that the deflated Sharpe ratio tests the winner against.
    """
    if n_trials <= 1 or sr_std_across_trials <= 0:
        return 0.0
    from scipy.stats import norm
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std_across_trials *
                 ((1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(
    sr_observed: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    PSR(SR*) = Φ( (SR_hat − SR*) · sqrt(T−1) / sqrt(1 − γ3·SR_hat + (γ4−1)/4·SR_hat²) )

    Probability that the *true* Sharpe ratio exceeds sr_benchmark, given the
    observed Sharpe, sample size, and the distribution's skew/kurtosis
    (fat tails and negative skew — both common in stop-loss/take-profit
    trade return distributions — inflate the variance of the Sharpe
    estimator and are penalised here).
    """
    if n_obs < 2:
        return 0.0
    from scipy.stats import norm
    denom = 1 - skew * sr_observed + ((kurt - 1) / 4) * sr_observed ** 2
    if denom <= 0:
        return 0.0
    stat = (sr_observed - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(norm.cdf(stat))


def deflated_sharpe_ratio_family(results: list) -> list:
    """
    Apply the Deflated Sharpe Ratio correction across a family of N backtested
    variants (e.g. every combination scored in a grid/parameter search).

    Parameters
    ----------
    results : list of dict, each with at least:
        "net_pnls" : list/array of trade P&Ls (dollar or ATR-multiple units)

    Returns
    -------
    Same list, with each dict augmented in-place with:
        "sharpe_hat" : trade-level Sharpe ratio of that variant
        "dsr"        : Deflated Sharpe Ratio ∈ [0, 1]
                       (probability true Sharpe > 0, net of selection bias
                        across all `len(results)` trials and non-normality)

    Notes
    -----
    Must be called once, after the full family of trials has been scored —
    SR0 (the "best-of-N by luck alone" benchmark) depends on the empirical
    cross-sectional dispersion of Sharpe ratios across the *entire* family,
    so it cannot be computed one variant at a time during the scan.
    """
    scored = [r for r in results if len(r.get("net_pnls", [])) >= 5]
    n_trials = len(results)
    if n_trials < 2 or not scored:
        for r in results:
            r["sharpe_hat"] = trade_level_sharpe(r.get("net_pnls", []))
            r["dsr"] = 0.0
        return results

    for r in results:
        r["sharpe_hat"] = trade_level_sharpe(r.get("net_pnls", []))

    sr_values = np.array([r["sharpe_hat"] for r in scored])
    sr_std = float(sr_values.std(ddof=1)) if len(sr_values) > 1 else 0.0
    sr0 = expected_max_sharpe(sr_std, n_trials)

    for r in results:
        pnls = r.get("net_pnls", [])
        n_obs = len(pnls)
        if n_obs < 5:
            r["dsr"] = 0.0
            continue
        skew, kurt = _skew_kurt(pnls)
        r["dsr"] = probabilistic_sharpe_ratio(
            r["sharpe_hat"], sr0, n_obs, skew, kurt
        )
    return results


def mc_summary_table(mc_store: dict) -> pd.DataFrame:
    """
    Aggregate MC stats for multiple scenarios into a single comparison table.

    Parameters
    ----------
    mc_store : {scenario_name: run_monte_carlo result dict}

    Returns
    -------
    DataFrame indexed by scenario name.
    """
    rows = []
    for name, mc in mc_store.items():
        if not mc:
            continue
        fe = mc["final_equity"]
        dd = mc["max_drawdown"] * 100
        tr = mc["total_return"] * 100
        rows.append({
            "Scenario":           name,
            "Median Ret (%)":     round(float(np.percentile(tr, 50)), 2),
            "p5 Ret (%)":         round(float(np.percentile(tr,  5)), 2),
            "p95 Ret (%)":        round(float(np.percentile(tr, 95)), 2),
            "Median MDD (%)":     round(float(np.percentile(dd, 50)), 2),
            "Worst MDD 5% (%)":   round(float(np.percentile(dd,  5)), 2),
            "Median Sharpe":      round(float(np.percentile(mc["sharpe"], 50)), 3),
            "P(Profit) (%)":      round(mc["p_profit"] * 100, 1),
            "P(Ruin) (%)":        round(mc["p_ruin"]   * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Scenario") if rows else pd.DataFrame()
