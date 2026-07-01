"""
Live signal computation for the ICT Silver Bullet (NY AM+PM) strategy —
BTCUSDT 15M.

Mirrors compute_live_signal()'s / compute_wyckoff_signal()'s contract: fetch
fresh OHLCV from Binance production FAPI, run the shared indicator pipeline,
run the Silver Bullet signal builder (src/strategy/ict_silver_bullet.py), and
return the 5-tuple required by LiveTrader's pluggable `signal_fn`
(signal, composite, atr, close, exit_plan) — see src/live/trader.py's
LiveTrader.__init__ docstring for the exact contract. Unlike the composite-
score and Wyckoff strategies, Silver Bullet has its own validated single
TP/SL + time-stop exit rule (tp=3x FVG width, sl=0.5x ATR_1H beyond the FVG
edge, 8h max hold) instead of the standard ATR ladder, so `exit_plan` is
populated whenever a signal fires.

NOTE — fee discrepancy: the ICT Silver Bullet backtest (create_ict_suite_report.py,
docs/ICT_SILVER_BULLET_STRATEGY.md) was validated using FEE=0.0004 (4bps,
Binance taker). LiveTrader applies its own engine-level FEE=0.0006 (6bps,
Bitget) uniformly in `_exit_full`/`_partial_close` regardless of strategy —
that constant is out of scope for this module (fixed, shared engine
behavior). Live P&L for this strategy will therefore be marginally more
fee-conservative than the validated backtest.
"""
from __future__ import annotations

from typing import Optional, Tuple

from src.live.data_live import fetch_tf_data, fetch_15m_bars
from src.strategy.indicators import add_indicators
from src.strategy.ict_silver_bullet import build_silver_bullet_signals

# Validated production defaults — see docs/ICT_SILVER_BULLET_STRATEGY.md
# section 8 ("Implementazione — File e Parametri").
MIN_FVG_ATR_FRAC = 0.05
MAX_AGE          = 16     # 15M bars = 4h re-entry window
MIN_SL_ATR_FLOOR = 0.5    # sizing floor: sl_dist >= 0.5 x ATR_1H
MAX_LEVERAGE     = 5.0    # notional <= 5x equity
TP_MULT          = 3.0    # tp = entry +/- 3.0 x ref_size (FVG width)
SL_ATR_MULT      = 0.5    # sl = ref_lo/ref_hi -/+ 0.5 x ATR_1H
TIME_STOP_HOURS  = 8.0    # 32 bars of 15M


def compute_ict_signal(symbol: str = "BTCUSDT") -> Tuple[int, float, float, float, Optional[dict]]:
    """
    Fetch live 1H + 15M data, run the ICT Silver Bullet FVG-in-killzone
    signal pipeline, and return the signal for the latest completed 15M bar.

    Returns
    -------
    signal    : int         – +1 long / -1 short / 0 flat
    composite : float       – 1.0 on a signal bar, 0.0 when flat
    atr_1h    : float       – ATR_1H mapped onto the signal bar (position sizing)
    last_price: float       – close price of the latest completed 15M bar
    exit_plan : dict | None – None when flat, else {"sl", "tp",
                              "min_sl_atr_floor", "max_leverage",
                              "time_stop_hours"} per LiveTrader's contract.
    """
    # 1. Fetch 1H bars (for ATR_1H context) and 15M bars (signal timeframe).
    df_1h  = add_indicators(fetch_tf_data(symbol)["1H"])
    df_15m = add_indicators(fetch_15m_bars(symbol))

    # 2. Build Silver Bullet signals (NY AM+PM combined killzone, default params).
    sig_df = build_silver_bullet_signals(
        df_15m, df_1h,
        min_fvg_atr_frac=MIN_FVG_ATR_FRAC,
        max_age=MAX_AGE,
    )

    # 3. Last *completed* 15M bar — mirrors the .iloc[-2] convention used by
    # compute_live_signal() / compute_wyckoff_signal() (the freshly fetched
    # last bar, [-1], is still forming).
    row = sig_df.iloc[-2]
    signal    = int(row["signal"])
    composite = float(row["composite"])
    close     = float(df_15m["close"].iloc[-2])
    atr_1h    = float(row["atr_1h"]) if signal != 0 else float(
        df_1h["atr_14"].clip(lower=1.0).iloc[-2]
    )

    if signal == 0:
        return 0, 0.0, atr_1h, close, None

    direction = signal
    ref_lo   = float(row["ref_lo"])
    ref_hi   = float(row["ref_hi"])
    ref_size = float(row["ref_size"])

    tp_px = close + direction * TP_MULT * ref_size
    sl_px = (ref_lo - SL_ATR_MULT * atr_1h) if direction == 1 else (ref_hi + SL_ATR_MULT * atr_1h)

    exit_plan = {
        "sl": sl_px,
        "tp": tp_px,
        "min_sl_atr_floor": MIN_SL_ATR_FLOOR,
        "max_leverage": MAX_LEVERAGE,
        "time_stop_hours": TIME_STOP_HOURS,
    }

    return direction, composite, atr_1h, close, exit_plan
