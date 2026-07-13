"""
Live signal computation for the ML RandomForest 8h strategy —
docs/ML_RF_8H_STRATEGY_SPEC.md.

Mirrors compute_ict_signal()'s / compute_ou_signal()'s contract: fetch fresh
multi-timeframe OHLCV from Binance production FAPI, run the shared feature
pipeline (src/live/ml_rf_features.py), predict via the persisted RandomForest
model, and return the 6-tuple required by LiveTrader's pluggable `signal_fn`.

⚠️ Status per the spec: this strategy has NOT been validated against
realistic slippage (the backtest edge flips negative between 2-5bps of extra
slippage). It is deployed here in the same testnet/no-real-capital mode as
the other 3 strategies, which doubles as the spec's own recommended first
step ("paper trading to measure real slippage") before any real allocation
— see docs/ML_RF_8H_STRATEGY_SPEC.md section 9.

Exit rule: NOT a classic TP/SL. The model predicts the *sign* of the 8h
forward return, so the only faithful exit is a fixed time-stop at entry+8h.
LiveTrader's engine already supports this generically via `time_stop_hours`
(checked before any TP/SL logic, see LiveTrader.check_position) — reused
here by setting `exit_mode`-equivalent fields as "single_tp" with `tp` placed
far enough away (many hundreds of ATR) that it can never realistically
trigger before the time-stop fires first, and `sl` as the spec's wide
4xATR safety-stop (crash protection only, fires on ~7% of trades per the
backtest). No new exit_mode was needed in trader.py.

max_leverage=1.0 (not the spec's MAX_LEV=10.0): the exchange account itself
is hard-configured to 1x leverage (LiveTrader.run() calls
client.set_leverage(1)), so a higher sizing cap here would just reproduce
the ICT/S07 bug fixed earlier (Bybit error 110007, orders rejected for
exceeding available margin) — see git history on src/live/ict_live.py /
src/live/ou_live.py for that incident. Sizing also reuses current-equity
risk (`_size_from_stop`) rather than the spec's fixed-initial-capital
formula, a deliberate simplification: the fixed-$ design is about isolating
the edge from compounding effects for research purposes, which matters less
for a testnet slippage-measurement deployment than staying consistent with
every other live strategy's sizing path.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.live.data_live import fetch_tf_data, fetch_15m_bars, fetch_1h_bars_extended, fetch_5m_bars_extended
from src.live.ml_rf_features import build_features, FEATURE_NAMES
from src.live.strategy_id import slugify_strategy_id
from src.strategy.indicators import add_indicators

SCENARIO = "ML RandomForest 8h"
EXCHANGE = "bybit"
STRATEGY_ID = slugify_strategy_id(SCENARIO, EXCHANGE)

ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "logs" / "strategies" / STRATEGY_ID / "model"
TRADES_CSV = ROOT / "logs" / "strategies" / STRATEGY_ID / "trades.csv"

MODEL_MAX_AGE_DAYS = 60   # WF_STEP_M=2 months refresh cadence per the spec

# Inference-time fetch sizes: smaller than training's (only need enough
# lookback for the causal pivot/rolling state to be correctly warmed up,
# not the full 6-month training window — see src/live/ml_rf_features.py).
FETCH_1H_BARS = 3000    # ≈ 125 days
FETCH_15M_BARS = 8000   # ≈ 83 days
FETCH_5M_BARS = 20000   # ≈ 69 days

HORIZON_HOURS = 8.0
THRESHOLD = 0.55
COOLDOWN_HOURS = 4.0
SAFETY_SL_ATR_MULT = 4.0
MIN_SL_ATR_FLOOR = 0.0
MAX_LEVERAGE = 1.0   # see module docstring — NOT the spec's 10.0


def _model_is_stale() -> bool:
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text())
    trained_at = datetime.fromisoformat(meta["trained_at"])
    age_days = (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400
    return age_days > MODEL_MAX_AGE_DAYS


def _ensure_model_fresh() -> None:
    if _model_is_stale():
        from scripts.train_ml_rf_8h import fit_and_persist_model
        fit_and_persist_model()


def _load_model():
    hmm_model = joblib.load(MODEL_DIR / "hmm_model.joblib")
    hmm_sorted_idx = joblib.load(MODEL_DIR / "hmm_sorted_idx.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    rf = joblib.load(MODEL_DIR / "rf.joblib")
    return hmm_model, hmm_sorted_idx, scaler, rf


def _hours_since_last_entry() -> Optional[float]:
    """Read this strategy's own trades.csv for the last ENTRY timestamp.
    Returns None if no prior entry (no cooldown to apply)."""
    if not TRADES_CSV.exists():
        return None
    try:
        df = pd.read_csv(TRADES_CSV)
    except Exception:
        return None
    entries = df[df["event"] == "ENTRY"]
    if entries.empty:
        return None
    last_ts = pd.Timestamp(entries.iloc[-1]["timestamp"])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return (now - last_ts).total_seconds() / 3600


def compute_ml_rf_signal(
    symbol: str = "BTCUSDT",
) -> Tuple[int, float, float, float, Optional[dict], Optional[dict]]:
    """
    Fetch live 1H+15M+5M+1D data, predict P(fwd_log_ret_8h > 0) via the
    persisted RandomForest model, and return the signal for the latest
    completed 1H bar.

    Returns
    -------
    signal    : int   – +1 long / -1 short / 0 flat
    composite : float – (p - 0.5) x 100 on a signal bar, 0.0 flat
    atr       : float – ATR_1H (used for the safety-stop sizing)
    last_price: float – close price of the latest completed 1H bar
    exit_plan : dict | None – see module docstring re: time-stop-via-single_tp
    diagnostics: dict | None – {"reason": str, "metrics": {...}}
    """
    _ensure_model_fresh()
    hmm_model, hmm_sorted_idx, scaler, rf = _load_model()

    df_1h = add_indicators(fetch_1h_bars_extended(limit=FETCH_1H_BARS))
    df_15m = fetch_15m_bars(limit=FETCH_15M_BARS)
    df_5m = fetch_5m_bars_extended(limit=FETCH_5M_BARS)
    df_1d = fetch_tf_data(symbol)["1D"]

    feats = build_features(df_1h, df_5m, df_15m, df_1d, hmm_model, hmm_sorted_idx)

    # Last *completed* 1H bar — same iloc[-2] convention as every other
    # live signal module (the freshly-fetched last row, [-1], is still forming).
    row = feats.iloc[-2]
    close = float(df_1h["close"].iloc[-2])
    atr_1h = float(df_1h["atr_14"].clip(lower=1.0).iloc[-2])

    X = row[FEATURE_NAMES].values.reshape(1, -1).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    p = float(rf.predict_proba(scaler.transform(X))[0, 1])

    if p > THRESHOLD:
        raw_signal = 1
    elif p < (1 - THRESHOLD):
        raw_signal = -1
    else:
        raw_signal = 0

    cooldown_h = _hours_since_last_entry()
    in_cooldown = cooldown_h is not None and cooldown_h < COOLDOWN_HOURS
    signal = 0 if in_cooldown else raw_signal
    composite = (p - 0.5) * 100 if signal != 0 else 0.0

    regime_label = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}.get(int(row["hmm_state"]), "?")
    metrics = {
        "p_up": p,
        "hmm_regime": regime_label,
        "hmm_prob_bull": float(row["hmm_prob_bull"]),
        "hmm_prob_bear": float(row["hmm_prob_bear"]),
        "cooldown_hours_remaining": (
            max(0.0, COOLDOWN_HOURS - cooldown_h) if cooldown_h is not None else 0.0
        ),
    }

    if signal == 0:
        if in_cooldown:
            reason = (f"p_up={p:.2f} would signal {'LONG' if raw_signal>0 else 'SHORT' if raw_signal<0 else 'flat'} "
                       f"but in cooldown ({cooldown_h:.1f}h < {COOLDOWN_HOURS}h since last entry)")
        else:
            reason = f"p_up={p:.2f} within [{1-THRESHOLD:.2f}, {THRESHOLD:.2f}] — no signal"
        diagnostics = {"reason": reason, "metrics": metrics}
        return 0, 0.0, atr_1h, close, None, diagnostics

    direction = signal
    sl_px = close - direction * SAFETY_SL_ATR_MULT * atr_1h
    tp_px = close + direction * 1000.0 * atr_1h   # unreachable — time-stop is the real exit

    exit_plan = {
        "sl": sl_px,
        "tp": tp_px,
        "min_sl_atr_floor": MIN_SL_ATR_FLOOR,
        "max_leverage": MAX_LEVERAGE,
        "time_stop_hours": HORIZON_HOURS,
    }
    reason = (f"p_up={p:.2f} {'>' if direction>0 else '<'} "
              f"{THRESHOLD if direction>0 else 1-THRESHOLD:.2f} — "
              f"{'LONG' if direction>0 else 'SHORT'} entry, exit @ +{HORIZON_HOURS:.0f}h")
    diagnostics = {"reason": reason, "metrics": metrics}

    return direction, composite, atr_1h, close, exit_plan, diagnostics
