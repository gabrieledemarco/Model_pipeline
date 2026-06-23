"""
Hidden Markov Model regime detector for BTCUSDT 1H data.

3 latent states ordered by mean log-return:
    0 = bear   (negative drift, high vol)
    1 = sideways (near-zero drift, moderate vol)
    2 = bull   (positive drift, variable vol)

Observation vector per bar: [log_return, log_vol_24h]
Model: GaussianHMM with diagonal covariance.

Usage in walk-forward gate:
    Fit on TRAINING bars only → predict on OOS bars.
    This avoids any look-ahead: the OOS model applies transition dynamics
    learned entirely from past data.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm as _hmm

warnings.filterwarnings("ignore", module="hmmlearn")

N_HMM_STATES = 3

HMM_FEATURE_NAMES = [
    "hmm_state",       # 0=bear 1=side 2=bull (float)
    "hmm_prob_bear",   # posterior P(bear)
    "hmm_prob_side",   # posterior P(sideways)
    "hmm_prob_bull",   # posterior P(bull)
    "hmm_duration",    # log1p(bars since last state change)
]


def _obs(df: pd.DataFrame) -> np.ndarray:
    """Build (n, 2) observation matrix: log_return + log_vol_24h."""
    close   = df["close"].values.astype(float)
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)), prepend=0.0)
    log_ret[0] = 0.0

    vol24 = (pd.Series(log_ret)
             .rolling(24, min_periods=4)
             .std()
             .bfill()
             .fillna(0.0)
             .values)
    log_vol = np.log(np.maximum(vol24, 1e-8))
    return np.column_stack([log_ret, log_vol])


def fit_hmm(
    df_train: pd.DataFrame,
    n_states: int = N_HMM_STATES,
    random_state: int = 42,
) -> tuple:
    """
    Fit GaussianHMM on training bars.

    Returns
    -------
    model      : fitted GaussianHMM
    sorted_idx : ndarray mapping semantic label → raw HMM state index
                 sorted_idx[0] = raw state with lowest mean log_return (bear)
                 sorted_idx[1] = mid (sideways)
                 sorted_idx[2] = highest mean log_return (bull)
    """
    X = _obs(df_train)
    model = _hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=300,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)
    # Sort states by mean log_return ascending → bear, side, bull
    sorted_idx = np.argsort(model.means_[:, 0])
    return model, sorted_idx


def predict_hmm_features(
    model,
    sorted_idx: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict HMM regime features for a DataFrame using an already-fitted model.

    Parameters
    ----------
    model      : GaussianHMM fitted on training data
    sorted_idx : mapping from fit_hmm()
    df         : 1H OHLCV DataFrame (train or OOS slice)

    Returns DataFrame with HMM_FEATURE_NAMES columns, indexed like df.
    """
    X = _obs(df)
    raw_states = model.predict(X)
    posteriors = model.predict_proba(X)   # (n, n_states)

    n_states = model.n_components
    # raw_state → semantic label
    raw_to_sem = np.empty(n_states, dtype=int)
    for sem, raw in enumerate(sorted_idx):
        raw_to_sem[raw] = sem
    semantic = raw_to_sem[raw_states]

    # Reorder posteriors to [bear, side, bull]
    reordered = posteriors[:, sorted_idx]   # (n, n_states)

    # Bars since last state change (in raw state space, then log-scaled)
    dur = np.ones(len(raw_states), dtype=float)
    for i in range(1, len(raw_states)):
        dur[i] = dur[i - 1] + 1 if raw_states[i] == raw_states[i - 1] else 1.0

    out = pd.DataFrame(index=df.index)
    out["hmm_state"]    = semantic.astype(float)
    out["hmm_prob_bear"] = reordered[:, 0]
    out["hmm_prob_side"] = reordered[:, 1]
    out["hmm_prob_bull"] = reordered[:, 2]
    out["hmm_duration"]  = np.log1p(dur)
    return out
