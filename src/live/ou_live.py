"""
Live signal computation for S07 OU Mean Reversion — BTCUSDT 1H, HMM-gated.

Mirrors compute_wyckoff_signal()'s / compute_ict_signal()'s contract: fetch
fresh OHLCV from Binance production FAPI, run the shared indicator pipeline,
run the OU z-score signal builder (src/strategy/ou_mean_reversion.py), gate
it through the monthly-refreshed HMM regime model (src/live/hmm_regime.py),
and return the 6-tuple required by LiveTrader's pluggable `signal_fn`
(signal, composite, atr, close, exit_plan, diagnostics) — see
src/live/trader.py's LiveTrader.__init__ docstring for the exact contract.

S07 is the ONLY validated strategy of the S07/S08 pair investigated in
docs/VALIDATED_STRATEGIES_SPEC.md — S08 Quantile Channel does NOT survive the
lookahead-bias correction documented there and is explicitly not deployed.

Per the spec's 3-layer architecture:
  Layer 1 (this module + src/strategy/ou_mean_reversion.py): 1H OU z-score
           entry signal.
  Layer 2 (src/live/hmm_regime.py): 4H HMM regime gate — accept LONG only in
           BULL, SHORT only in BEAR, reject everything in SIDEWAYS.
  Layer 3 (this module): TP/SL sized off 4H ATR (tp_frac=sl_frac=1.0×ATR4H,
           confirmed optimal for S07 post lookahead-bias fix), MAX_HOLD=96
           1H bars (=96h) time-stop, MAX_LEV=5.0× sizing cap.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from src.live.data_live import fetch_tf_data
from src.live.hmm_regime import get_regime_model, current_regime
from src.strategy.indicators import add_indicators
from src.strategy.ou_mean_reversion import build_ou_signals

# Validated production defaults — docs/VALIDATED_STRATEGIES_SPEC.md, Layer 3.
TP_FRAC = 1.0             # x ATR4H
SL_FRAC = 1.0             # x ATR4H
MIN_SL_ATR_FLOOR = 0.25   # MIN_SL_ATR — sizing floor, inert at sl_frac=1.0
MAX_LEVERAGE = 5.0        # MAX_LEV
TIME_STOP_HOURS = 96.0    # MAX_HOLD=96 1H bars = 96 hours


def compute_ou_signal(
    symbol: str = "BTCUSDT",
) -> Tuple[int, float, float, float, Optional[dict], Optional[dict]]:
    """
    Fetch live 1H data, run the OU mean-reversion signal pipeline, gate it
    through the HMM regime filter, and return the signal for the latest
    completed 1H bar.

    Returns
    -------
    signal     : int         – +1 long / -1 short / 0 flat (post HMM gate)
    composite  : float       – -z (OU composite score) on a signal bar, 0.0 flat
    atr        : float       – ATR4H (4H ATR — used for sizing/TP/SL, NOT ATR1H)
    last_price : float       – close price of the latest completed 1H bar
    exit_plan  : dict | None – None when flat, else {"sl", "tp",
                               "min_sl_atr_floor", "max_leverage",
                               "time_stop_hours"} per LiveTrader's contract.
    diagnostics: dict | None – {"reason": str, "metrics": {...}}
    """
    # 1. Fetch 1H bars, add indicators.
    df_1h = add_indicators(fetch_tf_data(symbol)["1H"])

    # 2. Build OU signals; use the last *completed* bar (iloc[-2] — the
    #    freshly-fetched last row, [-1], is still forming), same convention
    #    as compute_live_signal() / compute_wyckoff_signal().
    sig_df = build_ou_signals(df_1h)
    row = sig_df.iloc[-2]

    raw_signal = int(row["signal"])
    raw_composite = float(row["composite"])
    z = float(row["z"]) if row["z"] == row["z"] else float("nan")  # NaN-safe
    sma30 = float(row["sma30"]) if row["sma30"] == row["sma30"] else float("nan")
    close = float(df_1h["close"].iloc[-2])

    # 3. HMM regime model — cached, only refit when the cache is stale
    #    (monthly, per docs/VALIDATED_STRATEGIES_SPEC.md HMM_RETRAIN_MONTHS=1).
    hmm_model, bull_state, bear_state = get_regime_model(symbol)

    # 4. Current regime (most recently closed 4H bar) + ATR4H for sizing.
    regime, _regime_close, atr4h = current_regime(hmm_model, bull_state, bear_state, symbol)

    # 5. Gate the raw OU signal through the HMM regime.
    if raw_signal > 0 and regime != "BULL":
        signal = 0
    elif raw_signal < 0 and regime != "BEAR":
        signal = 0
    else:
        signal = raw_signal

    composite = raw_composite if signal != 0 else 0.0
    atr = atr4h  # Layer 3 sizing/TP/SL always uses ATR4H, not the 1H ATR

    # ── Diagnostics ──────────────────────────────────────────────────────────
    z_is_nan = math.isnan(z)
    metrics = {
        "z": None if z_is_nan else float(z),
        "sma30": None if math.isnan(sma30) else float(sma30),
        "regime": str(regime),
        "raw_signal": int(raw_signal),
        "atr4h": float(atr4h),
    }

    if z_is_nan:
        reason = (
            "insufficient history for OU fit this bar (beta>=0 or "
            "degenerate window) — no signal"
        )
    elif signal != 0:
        dir_label = "LONG" if signal > 0 else "SHORT"
        thresh = "< -1.0" if z < -1.0 else "> +1.0"
        reason = f"z={z:.2f} {thresh}, regime={regime} — {dir_label} accepted"
    elif raw_signal != 0:
        dir_label = "LONG" if raw_signal > 0 else "SHORT"
        reason = (
            f"OU signal={dir_label} (z={z:.2f}) but regime={regime} — "
            f"rejected by HMM gate"
        )
    else:
        reason = (
            f"z={z:.2f} within [-1.0,+1.0] or against SMA30 trend — no signal"
        )

    diagnostics = {"reason": reason, "metrics": metrics}

    # 6. Exit plan (Layer 3 — TP/SL sized off ATR4H).
    exit_plan = None
    if signal != 0:
        direction = signal
        tp_px = close + direction * TP_FRAC * atr4h
        sl_px = close - direction * SL_FRAC * atr4h
        exit_plan = {
            "sl": sl_px,
            "tp": tp_px,
            "min_sl_atr_floor": MIN_SL_ATR_FLOOR,
            "max_leverage": MAX_LEVERAGE,
            "time_stop_hours": TIME_STOP_HOURS,
        }

    return signal, composite, atr, close, exit_plan, diagnostics
