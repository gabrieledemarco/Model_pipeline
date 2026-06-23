"""
Natural Evolution Strategy (NES) for signal weight optimisation.

Learns per-window optimal weights w ∈ R^n_signals that replace the fixed
WEIGHTS dict in signals.py.  Maximises a fast vectorised return proxy on
training data, then applies the learned weights to generate OOS signals.

Reference: Salimans et al. "Evolution Strategies as a Scalable Alternative
to Reinforcement Learning" (OpenAI, 2017).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered signal component columns (must match signals.py WEIGHTS order)
SIGNAL_COLS = [
    "s_weekly", "s_daily", "s_4h", "s_1h",
    "s_oi", "s_funding", "s_vol", "s_cycle", "s_15m",
]

# Initial weights mirroring signals.py WEIGHTS (s_1m excluded — always 0 on 1H)
INIT_WEIGHTS = np.array([3, 4, 3, 3, 2, 1, 2, 1, 2], dtype=float)

# Default threshold matching signals.py LONG_THRESH / SHORT_THRESH
DEFAULT_THRESHOLD = 5.0


def _build_composite(signals_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted sum of signal components."""
    avail = [c for c in SIGNAL_COLS if c in signals_df.columns]
    w = weights[: len(avail)]
    return pd.Series(
        (signals_df[avail].values * w).sum(axis=1),
        index=signals_df.index,
    )


def _fast_fitness(
    signals_df: pd.DataFrame,
    df_1h: pd.DataFrame,
    weights: np.ndarray,
    threshold: float,
    session_hours: tuple,
) -> float:
    """
    Vectorised return proxy (no SL/TP, no fees).
    Fitness = sum of next-bar log-returns when in the signalled direction.
    Fast enough for NES population evaluation.
    """
    composite = _build_composite(signals_df, weights)
    in_sess   = (signals_df.index.hour >= session_hours[0]) & (signals_df.index.hour < session_hours[1])

    sig = pd.Series(0, index=signals_df.index)
    sig[(composite >= threshold)  & in_sess] =  1
    sig[(composite <= -threshold) & in_sess] = -1

    if (sig != 0).sum() < 3:
        return -1.0

    close   = df_1h["close"].reindex(signals_df.index).ffill()
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    ret     = (sig.shift(1).fillna(0) * log_ret).sum()
    return float(ret)


def build_evo_signals(
    signals_df: pd.DataFrame,
    weights: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    session_hours: tuple = (8, 21),
) -> pd.DataFrame:
    """Apply learned weights to produce a signal DataFrame for run_backtest()."""
    composite = _build_composite(signals_df, weights)
    in_sess   = (signals_df.index.hour >= session_hours[0]) & (signals_df.index.hour < session_hours[1])

    sig = pd.Series(0, index=signals_df.index)
    sig[(composite >= threshold)  & in_sess] =  1
    sig[(composite <= -threshold) & in_sess] = -1

    return pd.DataFrame({"signal": sig.astype(int), "composite": composite}, index=signals_df.index)


def fit_nes(
    signals_train: pd.DataFrame,
    df_train: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    session_hours: tuple = (8, 21),
    pop_size: int = 15,
    n_iter: int = 30,
    sigma: float = 0.3,
    lr: float = 0.05,
) -> np.ndarray:
    """
    Fit NES on training window.  Returns optimal weight vector w ∈ R^9.

    Parameters
    ----------
    signals_train : signal component DataFrame for training bars.
    df_train      : 1H OHLCV for training bars (needs 'close').
    pop_size      : population size per iteration.
    n_iter        : optimisation iterations.
    sigma         : noise std for perturbations.
    lr            : NES learning rate.
    """
    weights = INIT_WEIGHTS.copy().astype(float)

    for _ in range(n_iter):
        noise   = np.random.randn(pop_size, len(weights))
        rewards = np.zeros(pop_size)
        for k in range(pop_size):
            w_k       = weights + sigma * noise[k]
            rewards[k] = _fast_fitness(signals_train, df_train, w_k, threshold, session_hours)

        r_std = rewards.std()
        if r_std > 1e-8:
            rewards_norm = (rewards - rewards.mean()) / r_std
        else:
            rewards_norm = np.zeros(pop_size)

        # NES update
        gradient = (rewards_norm[:, None] * noise).sum(axis=0)
        weights += lr / (pop_size * sigma) * gradient

    return weights


def walk_forward_evo(
    df_1h: pd.DataFrame,
    signals_full: pd.DataFrame,
    wf_windows: list,
    threshold: float = DEFAULT_THRESHOLD,
    session_hours: tuple = (8, 21),
    pop_size: int = 15,
    n_iter: int = 30,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list, list]:
    """
    Walk-forward Evolution Strategy.

    For each WF window:
      1. Fit NES on training bars.
      2. Apply learned weights to OOS bars.

    Returns
    -------
    evo_signals : full-period signal DataFrame (OOS only, rest = 0)
    window_stats: list of per-window dicts
    all_weights : list of fitted weight vectors (one per window)
    """
    index = df_1h.index
    evo_sig = pd.DataFrame({"signal": 0, "composite": 0.0}, index=index)
    window_stats: list = []
    all_weights: list  = []

    if verbose:
        print(f"  Mode: NES (pop={pop_size}, iter={n_iter})  |  {len(wf_windows)} windows")

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(wf_windows):
        tr_mask  = (index >= tr_s) & (index < tr_e)
        oos_mask = (index >= oo_s) & (index < oo_e)

        sig_tr   = signals_full[tr_mask]
        sig_oos  = signals_full[oos_mask]
        df_tr    = df_1h[tr_mask]

        if tr_mask.sum() < 200 or oos_mask.sum() < 10:
            continue

        weights = fit_nes(sig_tr, df_tr, threshold, session_hours, pop_size, n_iter)
        oos_out = build_evo_signals(sig_oos, weights, threshold, session_hours)

        evo_sig.loc[oos_mask, "signal"]    = oos_out["signal"].values
        evo_sig.loc[oos_mask, "composite"] = oos_out["composite"].values

        n_sig = int((oos_out["signal"] != 0).sum())
        all_weights.append(weights)

        if verbose:
            wstr = ", ".join(f"{c}:{w:.2f}" for c, w in
                             zip(SIGNAL_COLS, weights.round(2)))
            print(f"    Win {i+1:2d} [{oo_s.date()}→{oo_e.date()}]: "
                  f"n_sig={n_sig}  weights=[{wstr}]")

        window_stats.append({
            "window":    i + 1,
            "oos_start": str(oo_s.date()),
            "oos_end":   str(oo_e.date()),
            "n_sig":     n_sig,
            "weights":   weights.tolist(),
        })

    return evo_sig, window_stats, all_weights
