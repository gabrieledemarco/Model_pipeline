"""
Live signal computation for the Wyckoff Spring/Upthrust strategy — BTCUSDT 1H.

Mirrors compute_live_signal()'s contract in src/live/data_live.py: fetch fresh
OHLCV from Binance production FAPI, run the shared indicator pipeline, run the
Wyckoff signal-matrix builder (src/strategy/wyckoff.py), and return the signal
for the last *completed* 1H bar (index [-2], since [-1] is the still-forming
current bar in a freshly-fetched dataset).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.live.data_live import fetch_tf_data
from src.strategy.indicators import add_indicators
from src.strategy.wyckoff import build_wyckoff_signals

# Same parameters passed to build_wyckoff_signals() below — kept as module
# constants so the local range/regime recompute (for diagnostics) uses
# exactly the same formula/config the signal builder saw.
N_RANGE          = 24
ADX_MAX          = 25.0
RANGE_WIDTH_MAX  = 0.10


def compute_wyckoff_signal(
    symbol: str = "BTCUSDT",
) -> Tuple[int, float, float, float, Optional[dict]]:
    """
    Fetch live 1H data, run the Wyckoff Spring/Upthrust signal pipeline, and
    return the signal for the latest completed 1H bar.

    Returns
    -------
    signal    : int   – +1 long / -1 short / 0 flat
    composite : float – raw composite score (± penetration depth × vol_ratio × 2)
    atr_14    : float – ATR-14 for position sizing
    last_price: float – close price of latest completed bar
    diagnostics: dict | None – {"reason": str, "metrics": {...}} explaining
                 the current signal/regime for the dashboard's "why did this
                 strategy do what it did" view.
    """
    # 1. Fetch OHLCV (reuse the existing multi-TF fetcher; only 1H is needed here)
    df_1h = fetch_tf_data(symbol)["1H"]

    # 2. Add indicators (same function used in backtest)
    df_1h = add_indicators(df_1h)

    # 3. Build Wyckoff spring/upthrust signals.
    # Parameters below are the validated production defaults from
    # docs/wyckoff_strategy_technical_spec.md (section 3 / 7 — 1H optimal config).
    sig_df = build_wyckoff_signals(
        df_1h,
        n_range=N_RANGE,
        adx_max=ADX_MAX,
        range_width_max=RANGE_WIDTH_MAX,
        vol_threshold=1.3,
        session_hours=(8, 21),
    )

    # Use signal from bar[-2] (last *completed* bar before current live bar)
    last_signal    = int(sig_df["signal"].iloc[-2])
    last_composite = float(sig_df["composite"].iloc[-2])
    last_atr       = float(df_1h["atr_14"].iloc[-2])
    last_close     = float(df_1h["close"].iloc[-2])

    # ── Diagnostics ──────────────────────────────────────────────────────────
    # Recompute range boundaries / regime locally with the exact same formula
    # build_wyckoff_signals() uses internally, so diagnostics reflect exactly
    # what the signal builder saw for this bar.
    hi = df_1h["high"]
    lo = df_1h["low"]
    min_periods = max(N_RANGE // 2, 5)
    range_high = hi.shift(1).rolling(N_RANGE, min_periods=min_periods).max()
    range_low  = lo.shift(1).rolling(N_RANGE, min_periods=min_periods).min()
    range_width = (range_high - range_low) / range_low.clip(lower=1.0)

    adx_col = df_1h["adx"] if "adx" in df_1h.columns else None
    adx_val = float(adx_col.iloc[-2]) if adx_col is not None else float("nan")

    is_ranging = bool(
        (adx_col.iloc[-2] < ADX_MAX
         and range_width.iloc[-2] < RANGE_WIDTH_MAX
         and not np.isnan(range_high.iloc[-2]))
        if adx_col is not None
        else (range_width.iloc[-2] < RANGE_WIDTH_MAX and not np.isnan(range_high.iloc[-2]))
    )

    range_high_val = float(range_high.iloc[-2]) if not np.isnan(range_high.iloc[-2]) else float("nan")
    range_low_val  = float(range_low.iloc[-2])  if not np.isnan(range_low.iloc[-2])  else float("nan")

    if "obv_trend" in df_1h.columns:
        obv_trend_val = float(np.sign(df_1h["obv_trend"].iloc[-2]))
    elif "obv" in df_1h.columns and "obv_ema21" in df_1h.columns:
        obv_trend_val = float(np.sign(df_1h["obv"].iloc[-2] - df_1h["obv_ema21"].iloc[-2]))
    else:
        obv_trend_val = 0.0
    obv_trend_int = int(obv_trend_val)

    vol_ratio_val = (
        float(df_1h["vol_ratio"].iloc[-2]) if "vol_ratio" in df_1h.columns else 1.0
    )

    regime_val = str(sig_df["regime"].iloc[-2])

    metrics = {
        "adx": float(adx_val),
        "is_ranging": bool(is_ranging),
        "range_high": float(range_high_val),
        "range_low": float(range_low_val),
        "obv_trend": int(obv_trend_int),
        "vol_ratio": float(vol_ratio_val),
        "regime": str(regime_val),
    }

    if last_signal != 0:
        reason = (
            f"Spring/Upthrust detected: {regime_val}, ADX={adx_val:.1f}, "
            f"vol_ratio={vol_ratio_val:.2f}x — "
            f"{'LONG' if last_signal > 0 else 'SHORT'} entry"
        )
    elif is_ranging:
        bias = (
            "accumulation" if obv_trend_int > 0
            else "distribution" if obv_trend_int < 0
            else "neutral"
        )
        reason = (
            f"Ranging (ADX={adx_val:.1f}<{ADX_MAX:.0f}), {bias} bias — "
            f"no spring/upthrust this bar"
        )
    else:
        reason = (
            f"Trending (ADX={adx_val:.1f}≥{ADX_MAX:.0f}) — Wyckoff "
            f"requires a ranging regime, no signal"
        )

    diagnostics = {"reason": reason, "metrics": metrics}

    return last_signal, last_composite, last_atr, last_close, diagnostics
