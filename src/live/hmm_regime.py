"""
HMM Regime Gate (Layer 2) — monthly-refreshed 3-state GaussianHMM on 4H data.

Used by src/live/ou_live.py (S07 OU Mean Reversion) to gate 1H entry signals:
accept LONG only in a BULL regime, SHORT only in a BEAR regime, reject
everything in SIDEWAYS. See docs/VALIDATED_STRATEGIES_SPEC.md, Layer 2.

Fitting logic mirrors create_mtf_hmm_report.py's `_hmm_features()` /
`fit_hmm()` / `_label_states()` (lines ~106-124 at time of writing) exactly:
    features   = [log_ret, rolling_vol(10-bar)] on 4H closes
    model      = GaussianHMM(n_components=3, covariance_type="full",
                              n_iter=500, random_state=42)
    BULL state = highest mean log_ret; BEAR state = lowest mean log_ret
    (SIDEWAYS is implicitly "neither" — not tracked as its own variable).

Refit cadence: the HMM is refit only when the cached model on disk is older
than `max_age_days` (default 30 ≈ HMM_RETRAIN_MONTHS=1 month from the spec),
NOT on every signal computation — that would be wasteful and would make the
regime label unstable between adjacent bars for no benefit (the spec's
causal-refit protocol is about walk-forward validation honesty, not about
refitting every live tick).
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.live.data_live import fetch_4h_bars_extended, fetch_tf_data
from src.strategy.indicators import add_indicators

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

HMM_STATES = 3
HMM_ITER = 500
HMM_SEED = 42
VOL_WINDOW = 10          # bars for rolling vol in HMM features (4H)
HMM_HISTORY_LIMIT = 2200  # ~12 months of 4H bars (HMM_HISTORY_MONTHS=12)


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"hmm_regime_{symbol}.pkl"


def _hmm_features(close_series: pd.Series) -> Tuple[np.ndarray, pd.Series]:
    """Return (X, log_ret) for a close price series.

    Mirrors create_mtf_hmm_report.py's `_hmm_features()` exactly — feature
    matrix is [log_ret, rolling_10bar_vol]. Shared by both the fitting path
    (get_regime_model) and the live-prediction path (current_regime) so the
    formula is never duplicated / allowed to drift between the two.
    """
    lr = np.log(close_series / close_series.shift(1)).fillna(0)
    vol = lr.rolling(VOL_WINDOW, min_periods=2).std().fillna(0)
    X = np.column_stack([lr.values, vol.values])
    return X, lr


def _label_states(states: np.ndarray, log_ret: pd.Series, n: int = HMM_STATES) -> Tuple[int, int]:
    """Return (BULL_state, BEAR_state) labelled by per-state mean log_ret."""
    mean_r = {
        s: log_ret.values[states == s].mean()
        for s in range(n) if (states == s).any()
    }
    sorted_s = sorted(mean_r, key=mean_r.get)
    return sorted_s[-1], sorted_s[0]  # BULL, BEAR


def _fit_hmm(X: np.ndarray) -> GaussianHMM:
    m = GaussianHMM(
        n_components=HMM_STATES, covariance_type="full",
        n_iter=HMM_ITER, random_state=HMM_SEED,
    )
    m.fit(X)
    return m


def get_regime_model(
    symbol: str = "BTCUSDT", max_age_days: int = 30,
) -> Tuple[GaussianHMM, int, int]:
    """Return (hmm_model, bull_state, bear_state).

    Loads a cached fitted model from disk if it's younger than `max_age_days`;
    otherwise fetches ~12 months of 4H OHLCV, fits a fresh GaussianHMM,
    caches it, and returns the new one.
    """
    cache_file = _cache_path(symbol)

    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            fitted_at = datetime.fromisoformat(cached["fitted_at"])
            if fitted_at.tzinfo is None:
                fitted_at = fitted_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - fitted_at).total_seconds() / 86400.0
            if age_days < max_age_days:
                return cached["model"], cached["bull_state"], cached["bear_state"]
        except Exception:
            pass  # corrupt/unreadable cache — fall through to refit

    df_4h_raw = fetch_4h_bars_extended(symbol, limit=HMM_HISTORY_LIMIT)
    X, log_ret = _hmm_features(df_4h_raw["close"])
    model = _fit_hmm(X)
    states = model.predict(X)
    bull_state, bear_state = _label_states(states, log_ret, model.n_components)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "bull_state": bull_state,
        "bear_state": bear_state,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = cache_file.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    tmp.replace(cache_file)

    return model, bull_state, bear_state


def current_regime(
    hmm_model: GaussianHMM, bull_state: int, bear_state: int,
    symbol: str = "BTCUSDT",
) -> Tuple[str, float, float]:
    """Return (regime_label, close, atr_1h_equiv) for the most recently
    CLOSED 4H bar.

    Fetches a short recent 4H window (`fetch_tf_data(symbol)["4H"]`, ~125
    bars resampled from the existing 500x1H fetch). Computes the SAME
    [log_ret, rolling_vol(10)] features on that tail via `_hmm_features()`
    (the same helper `get_regime_model()`'s fitting path uses), calls
    `hmm_model.predict()`, and returns the state label of the LAST
    *fully closed* row — mapped via bull_state/bear_state. Also returns
    that same bar's atr_14 (from add_indicators()) for use as ATR4H in
    sizing.

    Uses `.iloc[-2]`, NOT `.iloc[-1]`: the 4H series is resampled from a
    live 1H fetch whose own last bar is still forming (same reason every
    other live signal function in this codebase reads `.iloc[-2]`), so the
    resampled 4H series' last row is itself frequently a still-forming bar
    too (e.g. at 11:11 UTC the resample's last 4H bucket covers
    08:00-12:00, which hasn't closed yet). Using `.iloc[-1]` here would
    silently reintroduce, in the live path, the same class of intra-bar
    lookahead this whole strategy's offline validation had to be corrected
    for (see docs/VALIDATED_STRATEGIES_SPEC.md's correction notice).
    """
    df_4h = add_indicators(fetch_tf_data(symbol)["4H"])

    X, _ = _hmm_features(df_4h["close"])
    states = hmm_model.predict(X)
    last_state = int(states[-2])

    if last_state == bull_state:
        regime = "BULL"
    elif last_state == bear_state:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"

    close = float(df_4h["close"].iloc[-2])
    atr_1h_equiv = float(df_4h["atr_14"].iloc[-2])

    return regime, close, atr_1h_equiv
