"""
ML gate model v2 — fixes overfitting, adds regression target, progressive feature selection.
ML gate model v3 — bar-level direction gate (30× more training data).

Improvements over v1
────────────────────
1. Early stopping with chronological val split (last 20% of training window)
   → eliminates 100% train accuracy overfitting
2. Regression variant: predicts net_pnl / notional (continuous)
   → more informative target than binary win/loss
3. Progressive feature selection: after first 5 windows accumulate OOS
   feature importance and restrict to top-N for the remaining windows
   → better feature/sample ratio (~160 samples, 20 features vs 60)

v3 addition: bar-level direction gate
──────────────────────────────────────
Root cause of v2 failure: ~200 trades per 6m window is insufficient for
LightGBM to learn per-trade profitability from 60 features.

Solution: train on ALL 1H bars (~6,000 per window) with target:
    y = 1 if close[T + forward_bars] > close[T] else 0

Gate:
    LONG  signal fires only if P(up) ≥ gate_threshold
    SHORT signal fires only if P(up) ≤ (1 − gate_threshold)

Walk-forward: 6m train / 2m OOS / 23 windows.
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
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_MONTHS = 6
OOS_MONTHS   = 2
STEP_MONTHS  = 2

# Binary classifier: smaller, regularised, uses early stopping
LGBM_CLF = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "n_estimators":     500,        # capped by early stopping
    "learning_rate":    0.03,
    "max_depth":        3,
    "num_leaves":       7,
    "min_child_samples": 30,
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "reg_alpha":        1.0,
    "reg_lambda":       2.0,
    "class_weight":     "balanced",
    "verbose":          -1,
    "n_jobs":           -1,
}

# Regression: predict net_pnl / notional
LGBM_REG = {
    "objective":        "regression",
    "metric":           "rmse",
    "n_estimators":     500,
    "learning_rate":    0.03,
    "max_depth":        3,
    "num_leaves":       7,
    "min_child_samples": 20,
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "reg_alpha":        1.0,
    "reg_lambda":       2.0,
    "verbose":          -1,
    "n_jobs":           -1,
}

WARMUP_WINDOWS = 5      # windows before feature selection kicks in
TOP_N_FEATURES = 20     # features to keep after selection

# Bar-level direction model: slightly larger (more data → more capacity)
LGBM_BAR = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "n_estimators":      500,
    "learning_rate":     0.03,
    "max_depth":         4,
    "num_leaves":        15,
    "min_child_samples": 80,     # ~1.3% of 6k training bars
    "subsample":         0.7,
    "colsample_bytree":  0.6,
    "reg_alpha":         0.5,
    "reg_lambda":        1.0,
    "verbose":           -1,
    "n_jobs":            -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wf_windows(index: pd.DatetimeIndex) -> List[Tuple]:
    start, end = index[0], index[-1]
    windows, cur = [], start
    while True:
        tr_end = cur + pd.DateOffset(months=TRAIN_MONTHS)
        oo_s   = tr_end
        oo_e   = oo_s + pd.DateOffset(months=OOS_MONTHS)
        if oo_e > end:
            break
        windows.append((cur, tr_end, oo_s, oo_e))
        cur = cur + pd.DateOffset(months=STEP_MONTHS)
    return windows


def _extract_labels(
    df_1h: pd.DataFrame,
    signals: pd.DataFrame,
    feat_df: pd.DataFrame,
    regression: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run backtest on slice, return (feature_rows_at_entry, labels).
    Binary label : 1 if net_pnl > 0 else 0
    Regression   : net_pnl / notional (fraction return)
    """
    bt = run_backtest(df_1h, signals)
    trades = bt["trades"]
    if trades.empty:
        return pd.DataFrame(), np.array([])

    feat_idx_set = set(feat_df.index)
    rows, labels = [], []
    for _, t in trades.iterrows():
        ts = t["entry_ts"]
        if ts in feat_idx_set:
            rows.append(feat_df.loc[ts])
            if regression:
                lbl = float(t["net_pnl"] / t["notional"]) if t["notional"] > 0 else 0.0
            else:
                lbl = 1 if t["net_pnl"] > 0 else 0
            labels.append(lbl)

    if not rows:
        return pd.DataFrame(), np.array([])
    return pd.DataFrame(rows), np.array(labels)


def _fit_model(X_tr, y_tr, feat_cols, regression=False):
    """
    Fit LGBM with chronological val split for early stopping.
    Returns (model, best_iteration, val_metric).
    """
    n = len(X_tr)
    val_size = max(15, n // 5)
    split = n - val_size

    X_fit = X_tr.iloc[:split][feat_cols]
    y_fit = y_tr[:split]
    X_val = X_tr.iloc[split:][feat_cols]
    y_val = y_tr[split:]

    if regression:
        params = {**LGBM_REG}
        model  = lgb.LGBMRegressor(**params)
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        val_score = float(np.sqrt(np.mean((model.predict(X_val) - y_val)**2)))
    else:
        params = {**LGBM_CLF}
        sw     = compute_sample_weight("balanced", y_fit)
        model  = lgb.LGBMClassifier(**params)
        model.fit(
            X_fit, y_fit,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        val_probs  = model.predict_proba(X_val)[:, 1]
        val_preds  = (val_probs >= 0.5).astype(int)
        val_score  = float((val_preds == y_val).mean())

    train_acc = float((model.predict(X_tr[feat_cols]) == y_tr).mean()) if not regression else float(val_score)
    return model, train_acc, val_score


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLGateResult:
    oos_pred:       pd.Series           # probability (binary) or predicted return (reg)
    gated_signal:   pd.Series
    window_stats:   List[dict] = field(default_factory=list)
    feature_names:  List[str]  = field(default_factory=list)
    importances:    Optional[pd.DataFrame] = None


# ─────────────────────────────────────────────────────────────────────────────
# Binary gate with early stopping
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_binary_gate(
    df_1h:          pd.DataFrame,
    signals:        pd.DataFrame,
    feat_df:        pd.DataFrame,
    gate_threshold: float = 0.55,
    use_feat_sel:   bool  = True,
    verbose:        bool  = True,
    hmm_df:         Optional[pd.DataFrame] = None,
) -> MLGateResult:
    """
    Binary gate with early stopping and optional progressive feature selection.

    Parameters
    ----------
    use_feat_sel : after WARMUP_WINDOWS windows, restrict to top-N features
                   by accumulated OOS importance
    hmm_df       : if provided, fit a GaussianHMM per window on training bars
                   and append 5 HMM regime features to feat_tr / feat_oos.
                   HMM is fitted on TRAINING data only (no lookahead).
    """
    from .hmm_regime import fit_hmm, predict_hmm_features, HMM_FEATURE_NAMES

    index     = df_1h.index
    windows   = _wf_windows(index)
    all_cols  = list(feat_df.columns)
    if hmm_df is not None:
        all_cols = all_cols + HMM_FEATURE_NAMES
    feat_cols = all_cols[:]          # starts with all features

    all_probs = pd.Series(np.nan, index=index)
    imp_acc: Dict[str, list] = {c: [] for c in all_cols}
    window_stats: List[dict] = []

    if verbose:
        mode = "binary+early-stop" + ("+feat-sel" if use_feat_sel else "")
        if hmm_df is not None:
            mode += "+HMM"
        print(f"  Mode: {mode}  |  {len(windows)} windows")

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        tr_mask  = (index >= tr_s) & (index < tr_e)
        oos_mask = (index >= oo_s) & (index < oo_e)

        df_tr   = df_1h[tr_mask];    sig_tr  = signals[tr_mask];  feat_tr  = feat_df[tr_mask]
        df_oos  = df_1h[oos_mask];   sig_oos = signals[oos_mask]; feat_oos = feat_df[oos_mask]

        # Per-window HMM: fit on training bars only, predict train + OOS
        if hmm_df is not None:
            try:
                hmm_model, sorted_idx = fit_hmm(hmm_df[tr_mask])
                hmm_tr  = predict_hmm_features(hmm_model, sorted_idx, hmm_df[tr_mask])
                hmm_oos = predict_hmm_features(hmm_model, sorted_idx, hmm_df[oos_mask])
                feat_tr  = pd.concat([feat_tr,  hmm_tr],  axis=1)
                feat_oos = pd.concat([feat_oos, hmm_oos], axis=1)
            except Exception:
                # HMM failed for this window — pad with zeros and continue
                zeros_tr  = pd.DataFrame(0.0, index=feat_tr.index,  columns=HMM_FEATURE_NAMES)
                zeros_oos = pd.DataFrame(0.0, index=feat_oos.index, columns=HMM_FEATURE_NAMES)
                feat_tr  = pd.concat([feat_tr,  zeros_tr],  axis=1)
                feat_oos = pd.concat([feat_oos, zeros_oos], axis=1)

        if len(df_tr) < 200 or len(df_oos) < 10:
            continue

        # Progressive feature selection
        if use_feat_sel and i >= WARMUP_WINDOWS and imp_acc:
            mean_imp = {c: np.mean(v) for c, v in imp_acc.items() if v}
            feat_cols = sorted(mean_imp, key=mean_imp.get, reverse=True)[:TOP_N_FEATURES]

        X_tr, y_tr = _extract_labels(df_tr, sig_tr, feat_tr, regression=False)
        n_pos = int(y_tr.sum()) if len(y_tr) else 0
        n_neg = len(y_tr) - n_pos

        if len(X_tr) < 20 or n_pos < 5 or n_neg < 5:
            if verbose:
                print(f"    Win {i+1:2d}: skip (trades={len(X_tr)}, {n_pos}+/{n_neg}-)")
            continue

        model, train_acc, val_score = _fit_model(X_tr, y_tr, feat_cols, regression=False)
        best_iter = getattr(model, "best_iteration_", -1)

        # OOS prediction on signal bars
        sig_bars = sig_oos["signal"] != 0
        if sig_bars.sum() > 0:
            probs = model.predict_proba(feat_oos[sig_bars][feat_cols])[:, 1]
            all_probs.loc[feat_oos[sig_bars].index] = probs

        # Accumulate importance
        imp = dict(zip(feat_cols, model.feature_importances_))
        for c in all_cols:
            if c in imp:
                imp_acc[c].append(imp[c])

        n_sig = int(sig_bars.sum())
        n_gated = int((all_probs.loc[sig_oos.index].dropna() >= gate_threshold).sum())
        window_stats.append({
            "window": i + 1, "oos_start": str(oo_s.date()), "oos_end": str(oo_e.date()),
            "n_train": len(y_tr), "n_pos": n_pos, "n_neg": n_neg,
            "val_acc": round(val_score * 100, 1),
            "best_iter": best_iter,
            "n_feat": len(feat_cols),
            "n_sig_oos": n_sig, "n_gated": n_gated,
        })

        if verbose:
            pct = n_gated / n_sig * 100 if n_sig else 0
            print(f"    Win {i+1:2d} [{oo_s.date()}→{oo_e.date()}]: "
                  f"train={len(y_tr)}({n_pos}+/{n_neg}-)  "
                  f"val_acc={val_score*100:.0f}%  iter={best_iter}  "
                  f"feat={len(feat_cols)}  "
                  f"oos={n_sig}→{n_gated}({pct:.0f}%)")

    base_signal  = signals["signal"].copy()
    gated_signal = pd.Series(0, index=index)
    has_sig  = base_signal != 0
    has_prob = all_probs.notna()
    allow    = has_sig & has_prob & (all_probs >= gate_threshold)
    gated_signal[allow] = base_signal[allow]

    imp_df = pd.DataFrame({
        "feature": all_cols,
        "importance": [np.mean(imp_acc[c]) if imp_acc[c] else 0.0 for c in all_cols],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return MLGateResult(
        oos_pred     = all_probs,
        gated_signal = gated_signal,
        window_stats = window_stats,
        feature_names= all_cols,
        importances  = imp_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regression gate
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_regression_gate(
    df_1h:           pd.DataFrame,
    signals:         pd.DataFrame,
    feat_df:         pd.DataFrame,
    min_return_pct:  float = 0.0,
    use_feat_sel:    bool  = True,
    verbose:         bool  = True,
) -> MLGateResult:
    """
    Regression gate: predict net_pnl / notional.
    Allow trade if predicted return ≥ min_return_pct.

    Parameters
    ----------
    min_return_pct : minimum predicted trade return to allow (e.g. 0.005 = 0.5%)
    """
    index     = df_1h.index
    windows   = _wf_windows(index)
    all_cols  = list(feat_df.columns)
    feat_cols = all_cols[:]

    all_pred  = pd.Series(np.nan, index=index)
    imp_acc: Dict[str, list] = {c: [] for c in all_cols}
    window_stats: List[dict] = []

    if verbose:
        mode = "regression+early-stop" + ("+feat-sel" if use_feat_sel else "")
        print(f"  Mode: {mode}  |  {len(windows)} windows")

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        tr_mask  = (index >= tr_s) & (index < tr_e)
        oos_mask = (index >= oo_s) & (index < oo_e)

        df_tr   = df_1h[tr_mask];    sig_tr  = signals[tr_mask];  feat_tr  = feat_df[tr_mask]
        df_oos  = df_1h[oos_mask];   sig_oos = signals[oos_mask]; feat_oos = feat_df[oos_mask]

        if len(df_tr) < 200 or len(df_oos) < 10:
            continue

        if use_feat_sel and i >= WARMUP_WINDOWS and imp_acc:
            mean_imp  = {c: np.mean(v) for c, v in imp_acc.items() if v}
            feat_cols = sorted(mean_imp, key=mean_imp.get, reverse=True)[:TOP_N_FEATURES]

        X_tr, y_tr = _extract_labels(df_tr, sig_tr, feat_tr, regression=True)
        if len(X_tr) < 20:
            if verbose:
                print(f"    Win {i+1:2d}: skip (trades={len(X_tr)})")
            continue

        model, _, val_rmse = _fit_model(X_tr, y_tr, feat_cols, regression=True)
        best_iter = getattr(model, "best_iteration_", -1)

        sig_bars = sig_oos["signal"] != 0
        if sig_bars.sum() > 0:
            preds = model.predict(feat_oos[sig_bars][feat_cols])
            all_pred.loc[feat_oos[sig_bars].index] = preds

        imp = dict(zip(feat_cols, model.feature_importances_))
        for c in all_cols:
            if c in imp:
                imp_acc[c].append(imp[c])

        n_sig   = int(sig_bars.sum())
        oos_preds_present = all_pred.loc[sig_oos.index].dropna()
        n_gated = int((oos_preds_present >= min_return_pct).sum())

        window_stats.append({
            "window": i + 1, "oos_start": str(oo_s.date()), "oos_end": str(oo_e.date()),
            "n_train": len(y_tr),
            "train_mean_ret": round(float(y_tr.mean()) * 100, 2),
            "val_rmse": round(val_rmse * 100, 3),
            "best_iter": best_iter,
            "n_feat": len(feat_cols),
            "n_sig_oos": n_sig, "n_gated": n_gated,
        })

        if verbose:
            pct = n_gated / n_sig * 100 if n_sig else 0
            print(f"    Win {i+1:2d} [{oo_s.date()}→{oo_e.date()}]: "
                  f"train={len(y_tr)}(μret={y_tr.mean()*100:.2f}%)  "
                  f"val_rmse={val_rmse*100:.3f}%  iter={best_iter}  "
                  f"feat={len(feat_cols)}  "
                  f"oos={n_sig}→{n_gated}({pct:.0f}%)")

    base_signal  = signals["signal"].copy()
    gated_signal = pd.Series(0, index=index)
    has_sig  = base_signal != 0
    has_pred = all_pred.notna()
    allow    = has_sig & has_pred & (all_pred >= min_return_pct)
    gated_signal[allow] = base_signal[allow]

    imp_df = pd.DataFrame({
        "feature":    all_cols,
        "importance": [np.mean(imp_acc[c]) if imp_acc[c] else 0.0 for c in all_cols],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return MLGateResult(
        oos_pred     = all_pred,
        gated_signal = gated_signal,
        window_stats = window_stats,
        feature_names= all_cols,
        importances  = imp_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bar-level direction gate (v3) — 30× more training samples than trade gate
# ─────────────────────────────────────────────────────────────────────────────

def _extract_bar_labels(
    df_1h: pd.DataFrame,
    feat_df: pd.DataFrame,
    forward_bars: int = 4,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Label every bar: y=1 if close[T + forward_bars] > close[T].
    Drops the last `forward_bars` rows (no future label available within slice).
    """
    close = df_1h["close"].reindex(feat_df.index)
    fwd_close = close.shift(-forward_bars)
    y = (fwd_close > close).astype(int)
    valid = ~fwd_close.isna()
    return feat_df[valid], y[valid].values


def walk_forward_bar_level_gate(
    df_1h:          pd.DataFrame,
    signals:        pd.DataFrame,
    feat_df:        pd.DataFrame,
    gate_threshold: float = 0.55,
    forward_bars:   int   = 4,
    use_feat_sel:   bool  = True,
    verbose:        bool  = True,
) -> MLGateResult:
    """
    Bar-level direction gate (v3).

    Trains LightGBM on ALL 1H bars (~6k per window vs ~200 trades from v1/v2).
    Target: y = 1 if close[T + forward_bars] > close[T] (price goes up).

    At inference, gating is direction-aware:
        LONG  signal allowed if P(up) ≥ gate_threshold
        SHORT signal allowed if P(up) ≤ (1 − gate_threshold)

    Parameters
    ----------
    gate_threshold : minimum directional confidence to pass the gate (both sides)
    forward_bars   : lookahead horizon for label (default 4 = 4 hours)
    use_feat_sel   : enable progressive feature selection after WARMUP_WINDOWS
    """
    index     = df_1h.index
    windows   = _wf_windows(index)
    all_cols  = list(feat_df.columns)
    feat_cols = all_cols[:]

    all_probs = pd.Series(np.nan, index=index)   # P(price goes up)
    imp_acc: Dict[str, list] = {c: [] for c in all_cols}
    window_stats: List[dict] = []

    if verbose:
        mode = f"bar-level(fwd={forward_bars}h)+early-stop" + ("+feat-sel" if use_feat_sel else "")
        print(f"  Mode: {mode}  |  {len(windows)} windows")

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        tr_mask  = (index >= tr_s) & (index < tr_e)
        oos_mask = (index >= oo_s) & (index < oo_e)

        df_tr   = df_1h[tr_mask];   feat_tr  = feat_df[tr_mask]
        df_oos  = df_1h[oos_mask];  sig_oos  = signals[oos_mask]; feat_oos = feat_df[oos_mask]

        if len(df_tr) < 200 or len(df_oos) < 10:
            continue

        # Progressive feature selection
        if use_feat_sel and i >= WARMUP_WINDOWS and imp_acc:
            mean_imp  = {c: np.mean(v) for c, v in imp_acc.items() if v}
            feat_cols = sorted(mean_imp, key=mean_imp.get, reverse=True)[:TOP_N_FEATURES]

        X_tr, y_tr = _extract_bar_labels(df_tr, feat_tr, forward_bars=forward_bars)
        n_up = int(y_tr.sum())
        n_dn = len(y_tr) - n_up

        if len(X_tr) < 100 or n_up < 20 or n_dn < 20:
            if verbose:
                print(f"    Win {i+1:2d}: skip (bars={len(X_tr)}, {n_up}up/{n_dn}dn)")
            continue

        # Chronological val split (last 20%, min 100 bars)
        n = len(X_tr)
        val_size = max(100, n // 5)
        split    = n - val_size

        X_fit, y_fit = X_tr.iloc[:split][feat_cols], y_tr[:split]
        X_val, y_val = X_tr.iloc[split:][feat_cols], y_tr[split:]

        model = lgb.LGBMClassifier(**{**LGBM_BAR})
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        best_iter = getattr(model, "best_iteration_", -1)

        val_probs = model.predict_proba(X_val[feat_cols])[:, 1]
        val_acc   = float(((val_probs >= 0.5).astype(int) == y_val).mean())

        # Predict P(up) on ALL OOS bars
        oos_probs = model.predict_proba(feat_oos[feat_cols])[:, 1]
        all_probs.loc[feat_oos.index] = oos_probs

        # Accumulate feature importance
        imp = dict(zip(feat_cols, model.feature_importances_))
        for c in all_cols:
            if c in imp:
                imp_acc[c].append(imp[c])

        # Count how many OOS signals pass the gate
        sig_vals   = sig_oos["signal"]
        oos_prob_s = pd.Series(oos_probs, index=feat_oos.index)
        long_mask  = sig_vals == 1
        short_mask = sig_vals == -1
        n_long     = int(long_mask.sum())
        n_short    = int(short_mask.sum())
        n_long_g   = int((oos_prob_s[long_mask]  >= gate_threshold).sum()) if n_long  > 0 else 0
        n_short_g  = int((oos_prob_s[short_mask] <= (1 - gate_threshold)).sum()) if n_short > 0 else 0
        n_sig      = n_long + n_short
        n_gated    = n_long_g + n_short_g

        window_stats.append({
            "window":      i + 1,
            "oos_start":   str(oo_s.date()), "oos_end": str(oo_e.date()),
            "n_train_bars": len(y_tr), "n_up": n_up, "n_dn": n_dn,
            "val_acc":     round(val_acc * 100, 1),
            "best_iter":   best_iter,
            "n_feat":      len(feat_cols),
            "n_sig_oos":   n_sig, "n_gated": n_gated,
        })

        if verbose:
            pct = n_gated / n_sig * 100 if n_sig else 0
            print(f"    Win {i+1:2d} [{oo_s.date()}→{oo_e.date()}]: "
                  f"train={len(y_tr)}({n_up}up/{n_dn}dn)  "
                  f"val_acc={val_acc*100:.0f}%  iter={best_iter}  "
                  f"feat={len(feat_cols)}  "
                  f"oos={n_sig}→{n_gated}({pct:.0f}%)")

    # Build directionally-gated signal
    base_signal  = signals["signal"].copy()
    gated_signal = pd.Series(0, index=index)
    has_prob     = all_probs.notna()

    long_pass  = (base_signal == 1)  & has_prob & (all_probs >= gate_threshold)
    short_pass = (base_signal == -1) & has_prob & (all_probs <= (1 - gate_threshold))
    gated_signal[long_pass]  = 1
    gated_signal[short_pass] = -1

    imp_df = pd.DataFrame({
        "feature":    all_cols,
        "importance": [np.mean(imp_acc[c]) if imp_acc[c] else 0.0 for c in all_cols],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return MLGateResult(
        oos_pred     = all_probs,
        gated_signal = gated_signal,
        window_stats = window_stats,
        feature_names= all_cols,
        importances  = imp_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared: run gated backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_gated_backtest(
    df_1h:       pd.DataFrame,
    signals:     pd.DataFrame,
    gate_result: MLGateResult,
) -> dict:
    sig = signals.copy()
    sig["signal"] = gate_result.gated_signal.reindex(signals.index).fillna(0).astype(int)
    return run_backtest(df_1h, sig)
