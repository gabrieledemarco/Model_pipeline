"""
ML gate model v2 — fixes overfitting, adds regression target, progressive feature selection.

Improvements over v1
────────────────────
1. Early stopping with chronological val split (last 20% of training window)
   → eliminates 100% train accuracy overfitting
2. Regression variant: predicts net_pnl / notional (continuous)
   → more informative target than binary win/loss
3. Progressive feature selection: after first 5 windows accumulate OOS
   feature importance and restrict to top-N for the remaining windows
   → better feature/sample ratio (~160 samples, 20 features vs 60)

Architecture (unchanged from v1)
──────────────────────────────────
Walk-forward: 6m train / 2m OOS / 23 windows.
Gate: signal fires only when ML P(profitable) ≥ threshold
      OR predicted return ≥ min_return_pct (regression).
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
) -> MLGateResult:
    """
    Binary gate with early stopping and optional progressive feature selection.

    Parameters
    ----------
    use_feat_sel : after WARMUP_WINDOWS windows, restrict to top-N features
                   by accumulated OOS importance
    """
    index     = df_1h.index
    windows   = _wf_windows(index)
    all_cols  = list(feat_df.columns)
    feat_cols = all_cols[:]          # starts with all features

    all_probs = pd.Series(np.nan, index=index)
    imp_acc: Dict[str, list] = {c: [] for c in all_cols}
    window_stats: List[dict] = []

    if verbose:
        mode = "binary+early-stop" + ("+feat-sel" if use_feat_sel else "")
        print(f"  Mode: {mode}  |  {len(windows)} windows")

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        tr_mask  = (index >= tr_s) & (index < tr_e)
        oos_mask = (index >= oo_s) & (index < oo_e)

        df_tr   = df_1h[tr_mask];    sig_tr  = signals[tr_mask];  feat_tr  = feat_df[tr_mask]
        df_oos  = df_1h[oos_mask];   sig_oos = signals[oos_mask]; feat_oos = feat_df[oos_mask]

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
