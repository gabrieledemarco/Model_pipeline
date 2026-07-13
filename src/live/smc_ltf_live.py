"""
Live signal computation for the LTF SMC/MTF strategies — V1 Structure
Baseline and V2 + Regime + Sweep — reports/report_ltf_smc_mtf.html /
reports/ltf_smc_mtf_research.md.

Only V1 and V2 are validated (DSR >= 0.95 and positive genuine 2025-2026
holdout return, see the report). V3 + Expected-Return Gate FAILED both
gates (DSR=0.000, holdout=-5.4%) and is deliberately NOT implemented here.

Mirrors compute_ou_signal()'s / compute_ml_rf_signal()'s contract: fetch
fresh multi-timeframe OHLCV from Binance production FAPI, run the SAME
vectorized feature functions the backtest report used (src/strategy/smc.py,
src/strategy/mtf_swing.py, src/strategy/hmm_regime.py — zero reimplemented
logic), and return the 6-tuple required by LiveTrader's pluggable
`signal_fn` (signal, composite, atr, close, exit_plan, diagnostics) — see
src/live/trader.py's LiveTrader.__init__ docstring for the exact contract.

3-timeframe architecture (create_ltf_smc_mtf_report.py, section headers
[STRUCTURE]/[LIQUIDITY]):
  30m  BIAS      : smc_trend_signal — locked until the next confirmed CHoCH.
  15m  ENTRY     : price inside an order block aligned with the bias, or in
                   the discount (long) / premium (short) half of the last
                   swing. SL/TP from the nearest structural pivots.
  5m   CONFIRM   : liquidity sweep (V2 only) — wick beyond the recent
                   extreme with a close-back reclaim, in the bias direction.

V2 additionally gates on a 15m HMM regime model (V1 has zero fitted
parameters, per the report, and needs none). The HMM here is refit
periodically and cached to disk, same staleness-check pattern as
src/live/ml_rf_live.py's RandomForest model (MODEL_MAX_AGE_DAYS mirrors that
module's WF_STEP_M=2-month retrain cadence).

⚠️ Same caveat as every other strategy in this repo: the backtest models no
realistic slippage. Deployed here in testnet/no-real-capital paper mode only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.live.data_live import (
    fetch_15m_bars, fetch_30m_bars_extended, fetch_5m_bars_extended,
)
from src.live.strategy_id import slugify_strategy_id
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import align_htf_to_ltf, causal_trend_state
from src.strategy.smc import compute_smc_features, smc_trend_signal

SCENARIO_V1 = "SMC LTF V1 Structure Baseline"
SCENARIO_V2 = "SMC LTF V2 Regime+Sweep"
EXCHANGE = "bybit"
STRATEGY_ID_V1 = slugify_strategy_id(SCENARIO_V1, EXCHANGE)
STRATEGY_ID_V2 = slugify_strategy_id(SCENARIO_V2, EXCHANGE)

ROOT = Path(__file__).parent.parent.parent
TRADES_CSV_V1 = ROOT / "logs" / "strategies" / STRATEGY_ID_V1 / "trades.csv"
TRADES_CSV_V2 = ROOT / "logs" / "strategies" / STRATEGY_ID_V2 / "trades.csv"
MODEL_DIR_V2 = ROOT / "logs" / "strategies" / STRATEGY_ID_V2 / "model"

# Fetch sizes for LIVE INFERENCE (not training) — only need enough lookback
# for each causal state (pivots, zones, sweeps) to be correctly warmed up.
FETCH_30M_BARS = 3000    # ≈ 62 days
FETCH_15M_BARS = 8000    # ≈ 83 days
FETCH_5M_BARS = 1500     # ≈ 5.2 days — sweep lookback only needs ~20 bars

# 15m HMM regime model (V2 only) — separate training-size fetch, only used
# inside the cache-miss branch. WF_TRAIN_M=6 months in the report.
FETCH_15M_TRAIN_BARS = 20000   # ≈ 208 days
MODEL_MAX_AGE_DAYS = 60         # WF_STEP_M=2 months, same convention as ml_rf_live

# Validated production defaults — reports/report_ltf_smc_mtf.html / create_ltf_smc_mtf_report.py
TIME_STOP_HOURS = 12.0    # TIME_STOP_BARS=48 15m bars
COOLDOWN_HOURS = 1.0      # COOLDOWN_BARS=4 15m bars
SL_BUFFER_ATR = 0.15
MIN_RR = 1.2
FALLBACK_SL_ATR = 1.0
FALLBACK_TP_ATR = 2.0
SWEEP_LOOKBACK_BARS_5M = 6
SWEEP_EXTREME_LOOKBACK_5M = 20
MIN_SL_ATR_FLOOR = 0.25   # sizing safety floor — same value as src/live/ou_live.py
MAX_LEVERAGE = 1.0        # matches the exchange account's fixed 1x leverage config


def _hours_since_last_entry(trades_csv: Path) -> Optional[float]:
    """Read this strategy's own trades.csv for the last ENTRY timestamp.
    Same pattern as src/live/ml_rf_live.py._hours_since_last_entry."""
    if not trades_csv.exists():
        return None
    try:
        df = pd.read_csv(trades_csv)
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


def _compute_structure(symbol: str = "BTCUSDT") -> dict:
    """Fetch 30m/15m/5m OHLCV and compute every V1/V2 input for the latest
    *completed* 15m bar (iloc[-2] — the freshly-fetched last row is still
    forming, same convention as every other live signal function here).

    Reuses the exact vectorized functions create_ltf_smc_mtf_report.py's
    simulate() consumes (src/strategy/smc.py, src/strategy/mtf_swing.py) —
    no logic reimplemented, so live inference stays byte-identical to what
    was validated.
    """
    df30 = add_indicators(fetch_30m_bars_extended(symbol, limit=FETCH_30M_BARS))
    df15 = add_indicators(fetch_15m_bars(symbol, limit=FETCH_15M_BARS))
    df5 = fetch_5m_bars_extended(symbol, limit=FETCH_5M_BARS)

    smc30 = compute_smc_features(df30, swing_len=50, internal_len=5, prefix="smc")
    smc15 = compute_smc_features(df15, swing_len=50, internal_len=5, prefix="smc")

    bias_30m = smc_trend_signal(smc30, prefix="smc")
    bias_df = align_htf_to_ltf(
        df30.index, pd.DataFrame({"bias": bias_30m.values}, index=df30.index), df15.index,
    )
    bias15 = bias_df["bias"].fillna(0).astype(int).values

    CL15 = df15["close"].values.astype(float)
    HI15 = df15["high"].values.astype(float)
    LO15 = df15["low"].values.astype(float)
    ATR15 = np.where(df15["atr_14"].shift(1).values > 0, df15["atr_14"].shift(1).values, 1.0)

    struct15 = causal_trend_state(CL15, HI15, LO15, left=8, right=8)
    struct15.index = df15.index

    zone_bull15 = ((smc15["smc_ob_bull_in"].values == 1) | (smc15["smc_in_discount"].values == 1)).astype(int)
    zone_bear15 = ((smc15["smc_ob_bear_in"].values == 1) | (smc15["smc_in_premium"].values == 1)).astype(int)

    recent_low_5m = df5["low"].rolling(SWEEP_EXTREME_LOOKBACK_5M).min().shift(1)
    recent_high_5m = df5["high"].rolling(SWEEP_EXTREME_LOOKBACK_5M).max().shift(1)
    sweep_bull_5m = ((df5["low"] < recent_low_5m) & (df5["close"] > recent_low_5m)).astype(int)
    sweep_bear_5m = ((df5["high"] > recent_high_5m) & (df5["close"] < recent_high_5m)).astype(int)
    sweep_bull_roll = sweep_bull_5m.rolling(SWEEP_LOOKBACK_BARS_5M, min_periods=1).max()
    sweep_bear_roll = sweep_bear_5m.rolling(SWEEP_LOOKBACK_BARS_5M, min_periods=1).max()
    sweep_bull15 = sweep_bull_roll.reindex(df15.index, method="ffill").fillna(0).astype(int).values
    sweep_bear15 = sweep_bear_roll.reindex(df15.index, method="ffill").fillna(0).astype(int).values

    i = len(df15) - 2  # last completed 15m bar
    return dict(
        df15=df15, i=i,
        bias=int(bias15[i]),
        zone_bull=bool(zone_bull15[i]), zone_bear=bool(zone_bear15[i]),
        sweep_bull=bool(sweep_bull15[i]), sweep_bear=bool(sweep_bear15[i]),
        atr=float(ATR15[i]), close=float(CL15[i]),
        sl_basis_low=float(struct15["last_pivot_low"].iloc[i]) if pd.notna(struct15["last_pivot_low"].iloc[i]) else float("nan"),
        sl_basis_high=float(struct15["last_pivot_high"].iloc[i]) if pd.notna(struct15["last_pivot_high"].iloc[i]) else float("nan"),
        tp_basis_high=float(struct15["target_high"].iloc[i]) if pd.notna(struct15["target_high"].iloc[i]) else float("nan"),
        tp_basis_low=float(struct15["target_low"].iloc[i]) if pd.notna(struct15["target_low"].iloc[i]) else float("nan"),
    )


def _build_exit_plan(direction: int, px: float, atrv: float, s: dict) -> dict:
    """Structural SL/TP from the nearest pivots + MIN_RR floor — exact
    formula from create_ltf_smc_mtf_report.py's simulate()."""
    sl_level = s["sl_basis_low"] if direction == 1 else s["sl_basis_high"]
    tp_level = s["tp_basis_high"] if direction == 1 else s["tp_basis_low"]
    slp = (sl_level - direction * SL_BUFFER_ATR * atrv) if sl_level == sl_level else (px - direction * FALLBACK_SL_ATR * atrv)
    tpp = tp_level if tp_level == tp_level else (px + direction * FALLBACK_TP_ATR * atrv)
    risk = abs(px - slp)
    reward = abs(tpp - px)
    if risk > 0 and reward / risk < MIN_RR:
        tpp = px + direction * max(MIN_RR * risk, FALLBACK_TP_ATR * atrv)
    return {
        "sl": slp, "tp": tpp,
        "min_sl_atr_floor": MIN_SL_ATR_FLOOR,
        "max_leverage": MAX_LEVERAGE,
        "time_stop_hours": TIME_STOP_HOURS,
    }


def compute_smc_v1_signal(
    symbol: str = "BTCUSDT",
) -> Tuple[int, float, float, float, Optional[dict], Optional[dict]]:
    """V1 Structure Baseline — pure structural rule, zero fitted parameters.
    docs verdict: VALIDATA (OOS +485.6%, MaxDD -7.0%, DSR 1.000, holdout
    2025-2026 +196.6%)."""
    s = _compute_structure(symbol)
    b = s["bias"]

    hrs = _hours_since_last_entry(TRADES_CSV_V1)
    if hrs is not None and hrs < COOLDOWN_HOURS:
        return 0, 0.0, s["atr"], s["close"], None, {
            "reason": f"cooldown ({hrs:.2f}h < {COOLDOWN_HOURS}h since last entry)",
            "metrics": {"bias": b},
        }

    if b == 0:
        return 0, 0.0, s["atr"], s["close"], None, {"reason": "no structural bias (30m)", "metrics": {}}

    zone_ok = s["zone_bull"] if b == 1 else s["zone_bear"]
    if not zone_ok:
        return 0, 0.0, s["atr"], s["close"], None, {
            "reason": f"bias={b:+d} but price not in aligned OB/discount-premium zone (15m)",
            "metrics": {"bias": b},
        }

    exit_plan = _build_exit_plan(b, s["close"], s["atr"], s)
    reason = f"bias={b:+d}, in aligned zone (15m) — {'LONG' if b == 1 else 'SHORT'} entry"
    return b, 100.0 * b, s["atr"], s["close"], exit_plan, {"reason": reason, "metrics": {"bias": b}}


def _hmm_is_stale() -> bool:
    meta_path = MODEL_DIR_V2 / "meta.json"
    if not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text())
    trained_at = datetime.fromisoformat(meta["trained_at"])
    age_days = (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400
    return age_days > MODEL_MAX_AGE_DAYS


def _refit_hmm(symbol: str) -> None:
    df15_train = add_indicators(fetch_15m_bars(symbol, limit=FETCH_15M_TRAIN_BARS))
    model, sorted_idx = fit_hmm(df15_train, n_states=3, random_state=42)
    MODEL_DIR_V2.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR_V2 / "hmm_model.joblib")
    joblib.dump(sorted_idx, MODEL_DIR_V2 / "hmm_sorted_idx.joblib")
    (MODEL_DIR_V2 / "meta.json").write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_bars": len(df15_train),
    }))


def _load_hmm():
    model = joblib.load(MODEL_DIR_V2 / "hmm_model.joblib")
    sorted_idx = joblib.load(MODEL_DIR_V2 / "hmm_sorted_idx.joblib")
    return model, sorted_idx


def compute_smc_v2_signal(
    symbol: str = "BTCUSDT",
) -> Tuple[int, float, float, float, Optional[dict], Optional[dict]]:
    """V2 + Regime + Sweep — V1 + 15m HMM regime agreement + 5m liquidity
    sweep confirmation. docs verdict: VALIDATA (OOS +79.8%, MaxDD -8.3%,
    DSR 1.000, holdout 2025-2026 +41.5%)."""
    s = _compute_structure(symbol)
    b = s["bias"]

    hrs = _hours_since_last_entry(TRADES_CSV_V2)
    if hrs is not None and hrs < COOLDOWN_HOURS:
        return 0, 0.0, s["atr"], s["close"], None, {
            "reason": f"cooldown ({hrs:.2f}h < {COOLDOWN_HOURS}h since last entry)",
            "metrics": {"bias": b},
        }

    if b == 0:
        return 0, 0.0, s["atr"], s["close"], None, {"reason": "no structural bias (30m)", "metrics": {}}

    zone_ok = s["zone_bull"] if b == 1 else s["zone_bear"]
    if not zone_ok:
        return 0, 0.0, s["atr"], s["close"], None, {
            "reason": f"bias={b:+d} but price not in aligned OB/discount-premium zone (15m)",
            "metrics": {"bias": b},
        }

    if _hmm_is_stale():
        _refit_hmm(symbol)
    hmm_model, sorted_idx = _load_hmm()
    hmm_feats = predict_hmm_features(hmm_model, sorted_idx, s["df15"])
    hmm_bull = float(hmm_feats["hmm_prob_bull"].iloc[s["i"]])
    hmm_bear = float(hmm_feats["hmm_prob_bear"].iloc[s["i"]])
    regime_ok = (hmm_bull > 0.5) if b == 1 else (hmm_bear > 0.5)
    sweep_ok = s["sweep_bull"] if b == 1 else s["sweep_bear"]
    if not (regime_ok and sweep_ok):
        return 0, 0.0, s["atr"], s["close"], None, {
            "reason": f"bias={b:+d} in zone but regime_ok={regime_ok} sweep_ok={sweep_ok} — need both",
            "metrics": {"bias": b, "hmm_prob_bull": hmm_bull, "hmm_prob_bear": hmm_bear},
        }

    exit_plan = _build_exit_plan(b, s["close"], s["atr"], s)
    reason = f"bias={b:+d}, zone+regime+sweep aligned (15m/5m) — {'LONG' if b == 1 else 'SHORT'} entry"
    return b, 100.0 * b, s["atr"], s["close"], exit_plan, {
        "reason": reason,
        "metrics": {"bias": b, "hmm_prob_bull": hmm_bull, "hmm_prob_bear": hmm_bear},
    }
