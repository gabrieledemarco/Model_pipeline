"""
Live signal computation for the Wyckoff Spring/Upthrust strategy — BTCUSDT 1H.

Mirrors compute_live_signal()'s contract in src/live/data_live.py: fetch fresh
OHLCV from Binance production FAPI, run the shared indicator pipeline, run the
Wyckoff signal-matrix builder (src/strategy/wyckoff.py), and return the signal
for the last *completed* 1H bar (index [-2], since [-1] is the still-forming
current bar in a freshly-fetched dataset).
"""
from __future__ import annotations

from typing import Tuple

from src.live.data_live import fetch_tf_data
from src.strategy.indicators import add_indicators
from src.strategy.wyckoff import build_wyckoff_signals


def compute_wyckoff_signal(symbol: str = "BTCUSDT") -> Tuple[int, float, float, float]:
    """
    Fetch live 1H data, run the Wyckoff Spring/Upthrust signal pipeline, and
    return the signal for the latest completed 1H bar.

    Returns
    -------
    signal    : int   – +1 long / -1 short / 0 flat
    composite : float – raw composite score (± penetration depth × vol_ratio × 2)
    atr_14    : float – ATR-14 for position sizing
    last_price: float – close price of latest completed bar
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
        n_range=24,
        adx_max=25.0,
        range_width_max=0.10,
        vol_threshold=1.3,
        session_hours=(8, 21),
    )

    # Use signal from bar[-2] (last *completed* bar before current live bar)
    last_signal    = int(sig_df["signal"].iloc[-2])
    last_composite = float(sig_df["composite"].iloc[-2])
    last_atr       = float(df_1h["atr_14"].iloc[-2])
    last_close     = float(df_1h["close"].iloc[-2])

    return last_signal, last_composite, last_atr, last_close
