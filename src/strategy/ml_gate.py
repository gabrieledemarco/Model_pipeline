"""
ML gate model for BTCUSDT strategy.

Architecture
────────────
1. Walk-forward splits (identical windows to WFO optimizer):
     train 6 months → OOS 2 months, step 2 months, 23 windows
2. Per window:
   a. Extract trades from the composite-signal backtest on training data.
   b. Label each training trade: 1 = profitable (net_pnl > 0), 0 = loss.
   c. Build feature vector at the entry bar of each trade.
   d. Train LightGBM binary classifier.
   e. On OOS: for every bar where composite fires a signal, predict
      P(profitable | features).  If P ≥ gate_threshold → allow trade.
3. Gate: composite signal × ML approval → gated signal column.
4. Run gated backtest on stitched OOS periods.

No data leaks because:
  • Features are computed only from data available at bar T.
  • Trade labels use bars T+1 … T+n (future), but only in the training set
    which is strictly before the OOS window.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from .engine import run_backtest, INIT_CAP
from .ml_features import build_feature_matrix

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward parameters
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_MONTHS = 6
OOS_MONTHS   = 2
STEP_MONTHS  = 2      # must equal OOS_MONTHS for contiguous OOS coverage

LGBM_PARAMS = {
    "objective":       "binary",
    "metric":          "binary_logloss",
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       4,
    "num_leaves":      15,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "class_weight":    "balanced",
    "verbose":         -1,
    "n_jobs":          -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wf_windows(index: pd.DatetimeIndex) -> List[Tuple]:
    """
    Return list of (train_start, train_end, oos_start, oos_end) as Timestamps.
    Mirrors walk_forward.py logic.
    """
    start = index[0]
    end   = index[-1]

    windows = []
    cur = start
    while True:
        train_end = cur + pd.DateOffset(months=TRAIN_MONTHS)
        oos_start = train_end
        oos_end   = oos_start + pd.DateOffset(months=OOS_MONTHS)
        if oos_end > end:
            break
        windows.append((cur, train_end, oos_start, oos_end))
        cur = cur + pd.DateOffset(months=STEP_MONTHS)

    return windows


def _extract_trade_labels(
    df_1h:   pd.DataFrame,
    signals: pd.DataFrame,
    feat_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run backtest on df_1h/signals slice, extract the bar index of each entry,
    return (feature_rows, labels).

    label = 1 if net_pnl > 0 else 0
    """
    bt = run_backtest(df_1h, signals)
    trades = bt["trades"]

    if trades.empty:
        return pd.DataFrame(), np.array([])

    # Match entry_ts to a row in feat_df
    entry_times  = pd.DatetimeIndex(trades["entry_ts"])
    feat_idx_set = feat_df.index

    rows, labels = [], []
    for idx, trade in trades.iterrows():
        ts = trade["entry_ts"]
        if ts in feat_idx_set:
            rows.append(feat_df.loc[ts])
            labels.append(1 if trade["net_pnl"] > 0 else 0)

    if not rows:
        return pd.DataFrame(), np.array([])

    X = pd.DataFrame(rows)
    y = np.array(labels, dtype=int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward ML gate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLGateResult:
    oos_prob:       pd.Series         # prob(profitable) for every signal bar
    gated_signal:   pd.Series         # +1 / -1 / 0 after ML gate
    window_stats:   List[dict] = field(default_factory=list)
    feature_names:  List[str]  = field(default_factory=list)
    importances:    Optional[pd.DataFrame] = None


def walk_forward_ml_gate(
    df_1h:           pd.DataFrame,
    signals:         pd.DataFrame,
    feat_df:         pd.DataFrame,
    gate_threshold:  float = 0.55,
    verbose:         bool  = True,
) -> MLGateResult:
    """
    Walk-forward ML gate training and OOS prediction.

    Parameters
    ----------
    df_1h           : 1H OHLCV + indicators (full history)
    signals         : composite signal matrix (full history, session-filtered)
    feat_df         : feature matrix from ml_features.build_feature_matrix()
    gate_threshold  : minimum P(profitable) to allow a trade (default 0.55)
    verbose         : print per-window stats

    Returns
    -------
    MLGateResult with full OOS gated signal series
    """
    index    = df_1h.index
    windows  = _wf_windows(index)
    feat_cols = list(feat_df.columns)

    all_probs  = pd.Series(np.nan, index=index)
    importances_acc: Dict[str, list] = {c: [] for c in feat_cols}
    window_stats: List[dict] = []

    if verbose:
        print(f"  Walk-forward ML: {len(windows)} windows "
              f"({TRAIN_MONTHS}m train / {OOS_MONTHS}m OOS)")

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        # ── Slice data ────────────────────────────────────────────────────────
        tr_mask   = (index >= tr_s) & (index < tr_e)
        oos_mask  = (index >= oo_s) & (index < oo_e)

        df_tr     = df_1h[tr_mask]
        sig_tr    = signals[tr_mask]
        feat_tr   = feat_df[tr_mask]

        df_oos    = df_1h[oos_mask]
        sig_oos   = signals[oos_mask]
        feat_oos  = feat_df[oos_mask]

        if len(df_tr) < 200 or len(df_oos) < 10:
            continue

        # ── Build training labels ─────────────────────────────────────────────
        X_tr, y_tr = _extract_trade_labels(df_tr, sig_tr, feat_tr)

        n_pos = int(y_tr.sum()) if len(y_tr) else 0
        n_neg = len(y_tr) - n_pos

        if len(X_tr) < 20 or n_pos < 5 or n_neg < 5:
            if verbose:
                print(f"    Win {i+1:2d}: skip (only {len(X_tr)} trades, "
                      f"{n_pos}+/{n_neg}-)")
            continue

        # ── Train LightGBM ────────────────────────────────────────────────────
        sw = compute_sample_weight("balanced", y_tr)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr[feat_cols], y_tr, sample_weight=sw)

        # ── OOS prediction on signal bars only ────────────────────────────────
        sig_bars = sig_oos["signal"] != 0
        if sig_bars.sum() > 0:
            X_oos  = feat_oos[sig_bars][feat_cols]
            probs  = model.predict_proba(X_oos)[:, 1]
            all_probs.loc[X_oos.index] = probs

        # ── Accumulate feature importances ────────────────────────────────────
        imp = dict(zip(feat_cols, model.feature_importances_))
        for c in feat_cols:
            importances_acc[c].append(imp.get(c, 0))

        # ── Window stats ──────────────────────────────────────────────────────
        n_sig_oos  = int(sig_bars.sum())
        n_gated    = int((all_probs.loc[sig_oos.index].dropna() >= gate_threshold).sum())
        train_acc  = float((model.predict(X_tr[feat_cols]) == y_tr).mean())

        window_stats.append({
            "window":      i + 1,
            "oos_start":   str(oo_s.date()),
            "oos_end":     str(oo_e.date()),
            "n_train_tr":  len(y_tr),
            "n_pos_tr":    n_pos,
            "n_neg_tr":    n_neg,
            "train_acc":   round(train_acc * 100, 1),
            "n_sig_oos":   n_sig_oos,
            "n_gated":     n_gated,
        })

        if verbose:
            pct = n_gated / n_sig_oos * 100 if n_sig_oos else 0
            print(f"    Win {i+1:2d} [{oo_s.date()} → {oo_e.date()}]: "
                  f"train={len(y_tr)} ({n_pos}+/{n_neg}-)  "
                  f"acc={train_acc*100:.0f}%  "
                  f"OOS signals={n_sig_oos} → gated={n_gated} ({pct:.0f}%)")

    # ── Build gated signal ────────────────────────────────────────────────────
    # Bars with a composite signal but no ML prediction: treat as blocked
    base_signal   = signals["signal"].copy()
    gated_signal  = pd.Series(0, index=index)

    has_sig  = base_signal != 0
    has_prob = all_probs.notna()

    # Allow trade only where both composite fires AND ML probability ≥ threshold
    allow = has_sig & has_prob & (all_probs >= gate_threshold)
    gated_signal[allow] = base_signal[allow]

    # Feature importance summary
    imp_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": [np.mean(importances_acc[c]) for c in feat_cols],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return MLGateResult(
        oos_prob      = all_probs,
        gated_signal  = gated_signal,
        window_stats  = window_stats,
        feature_names = feat_cols,
        importances   = imp_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run gated backtest given a result
# ─────────────────────────────────────────────────────────────────────────────

def run_gated_backtest(
    df_1h:      pd.DataFrame,
    signals:    pd.DataFrame,
    gate_result: MLGateResult,
) -> dict:
    """Replace the signal column in *signals* with the gated signal and backtest."""
    sig_gated = signals.copy()
    sig_gated["signal"] = gate_result.gated_signal.reindex(signals.index).fillna(0).astype(int)
    return run_backtest(df_1h, sig_gated)
