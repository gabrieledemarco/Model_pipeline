"""
TradingView Webhook → Binance Executor
=======================================
Receives JSON alerts from TradingView (via webhook) and executes
BTCUSDT trades on Binance using CCXT.

Requirements
------------
pip install fastapi uvicorn ccxt python-dotenv

Environment variables (.env or shell)
--------------------------------------
BINANCE_API_KEY      = your Binance API key
BINANCE_API_SECRET   = your Binance API secret
WEBHOOK_SECRET       = a random secret string you set in TradingView alert URL
USE_TESTNET          = "true"  (omit or "false" for live)
RISK_PCT             = "1.0"   (% of equity per trade, default 1 %)
ATR_SL_MULT          = "2.0"   (ATR multiplier for SL, default 2.0)

Deploy
------
uvicorn webhook_executor:app --host 0.0.0.0 --port 8000

TradingView alert URL
---------------------
https://<your-server-ip>:8000/webhook?secret=<WEBHOOK_SECRET>

Alert message body (set in TradingView)
----------------------------------------
{"action":"buy","symbol":"BTCUSDT","score":{{plot("Composite")}},"atr":{{plot("ATR")}}}
{"action":"sell","symbol":"BTCUSDT","score":{{plot("Composite")}},"atr":{{plot("ATR")}}}
{"action":"close","symbol":"BTCUSDT"}
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import ccxt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("executor")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY",    "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
WH_SECRET  = os.getenv("WEBHOOK_SECRET",     "change_me")
TESTNET    = os.getenv("USE_TESTNET",        "true").lower() == "true"
RISK_PCT   = float(os.getenv("RISK_PCT",   "1.0"))
ATR_SL     = float(os.getenv("ATR_SL_MULT","2.0"))
SYMBOL     = "BTC/USDT"
MARKET     = "BTC/USDT:USDT"   # CCXT unified perpetual symbol

# ── Exchange init ──────────────────────────────────────────────────────────────
exchange: ccxt.binance | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global exchange
    exchange = ccxt.binance({
        "apiKey":    API_KEY,
        "secret":    API_SECRET,
        "options":   {"defaultType": "future"},
        "enableRateLimit": True,
    })
    if TESTNET:
        exchange.set_sandbox_mode(True)
        log.info("TESTNET mode active — no real trades")
    else:
        log.warning("LIVE mode — real funds at risk")

    try:
        balance = exchange.fetch_balance()
        usdt = balance["USDT"]["free"]
        log.info(f"Connected to Binance | USDT free: {usdt:.2f}")
    except Exception as e:
        log.error(f"Binance connection failed: {e}")

    yield

    log.info("Shutting down executor")

app = FastAPI(title="MTF Webhook Executor", lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _equity() -> float:
    bal = exchange.fetch_balance()
    return float(bal["USDT"]["total"])


def _current_position() -> dict:
    """Returns the current open position (or None)."""
    positions = exchange.fetch_positions([MARKET])
    for p in positions:
        if abs(float(p["contracts"] or 0)) > 0:
            return p
    return {}


def _atr_from_payload(payload: dict) -> float:
    """Extract ATR from the alert payload, or fetch last ATR from OHLCV."""
    if "atr" in payload and float(payload["atr"]) > 0:
        return float(payload["atr"])
    # Fallback: compute ATR-14 from 1H candles
    ohlcv = exchange.fetch_ohlcv(MARKET, "1h", limit=30)
    highs  = [x[2] for x in ohlcv]
    lows   = [x[3] for x in ohlcv]
    closes = [x[4] for x in ohlcv]
    trs = []
    for i in range(1, len(ohlcv)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i]  - closes[i-1]))
        trs.append(tr)
    return sum(trs[-14:]) / 14


def _qty(side: str, atr: float) -> float:
    """Risk-based position size in BTC."""
    eq        = _equity()
    risk_usd  = eq * RISK_PCT / 100.0
    sl_dist   = atr * ATR_SL
    ticker    = exchange.fetch_ticker(MARKET)
    price     = float(ticker["last"])
    qty_btc   = risk_usd / sl_dist
    # Binance min notional ~5 USDT, min qty 0.001 BTC
    qty_btc   = max(round(qty_btc, 3), 0.001)
    max_qty   = eq * 0.95 / price   # never more than 95 % of equity
    return min(qty_btc, max_qty)


def _sl_tp_prices(side: str, entry: float, atr: float) -> dict:
    sl_dist = atr * ATR_SL
    if side == "buy":
        return {
            "sl":  round(entry - sl_dist, 2),
            "tp1": round(entry + atr * 2.0, 2),
            "tp2": round(entry + atr * 4.0, 2),
            "tp3": round(entry + atr * 6.0, 2),
        }
    else:
        return {
            "sl":  round(entry + sl_dist, 2),
            "tp1": round(entry - atr * 2.0, 2),
            "tp2": round(entry - atr * 4.0, 2),
            "tp3": round(entry - atr * 6.0, 2),
        }


# ── Webhook endpoint ─────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(
    request: Request,
    secret: str = Query(..., description="Must match WEBHOOK_SECRET"),
):
    if secret != WH_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    action = str(payload.get("action", "")).lower()
    symbol = str(payload.get("symbol", "BTCUSDT"))
    score  = float(payload.get("score", 0))
    log.info(f"Alert received: action={action} symbol={symbol} score={score}")

    if not exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialised")

    result = {}

    # ── CLOSE any open position ──────────────────────────────────────────────
    if action == "close":
        pos = _current_position()
        if pos:
            side_close = "sell" if float(pos["contracts"]) > 0 else "buy"
            qty = abs(float(pos["contracts"]))
            order = exchange.create_market_order(MARKET, side_close, qty,
                                                 params={"reduceOnly": True})
            log.info(f"Closed position: {order['id']}")
            result = {"status": "closed", "order": order["id"]}
        else:
            result = {"status": "no_position"}

    # ── LONG or SHORT entry ───────────────────────────────────────────────────
    elif action in ("buy", "sell"):
        pos = _current_position()
        if pos and abs(float(pos.get("contracts", 0))) > 0:
            log.info("Already in a position – signal ignored")
            result = {"status": "skipped", "reason": "position_exists"}
        else:
            atr = _atr_from_payload(payload)
            qty = _qty(action, atr)

            order = exchange.create_market_order(MARKET, action, qty)
            entry_price = float(order.get("average") or
                                exchange.fetch_ticker(MARKET)["last"])
            levels = _sl_tp_prices(action, entry_price, atr)

            log.info(
                f"{'LONG' if action=='buy' else 'SHORT'}  qty={qty} BTC  "
                f"entry={entry_price:.2f}  SL={levels['sl']}  "
                f"TP1={levels['tp1']}  TP2={levels['tp2']}  TP3={levels['tp3']}"
            )

            # Place stop-loss as a stop-market order
            sl_side = "sell" if action == "buy" else "buy"
            sl_order = exchange.create_order(
                MARKET, "stop_market", sl_side, qty,
                params={
                    "stopPrice":   levels["sl"],
                    "reduceOnly":  True,
                    "closePosition": True,
                },
            )

            result = {
                "status":       "opened",
                "direction":    "long" if action == "buy" else "short",
                "entry":        entry_price,
                "qty":          qty,
                "sl":           levels["sl"],
                "tp1":          levels["tp1"],
                "tp2":          levels["tp2"],
                "tp3":          levels["tp3"],
                "order_id":     order["id"],
                "sl_order_id":  sl_order["id"],
                "score":        score,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            }

            # NOTE: TP partial closes must be managed separately.
            # Options:
            # a) Set TP1 limit order for 50% of qty → after fill, update SL to BE
            # b) Use a separate monitoring loop (see manage_tps() below)
            # For simplicity this executor places only the SL; TP orders are
            # handled by the strategy's subsequent TradingView alerts (TP1/TP2/TP3).

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    log.info(f"Response: {result}")
    return JSONResponse(result)


@app.get("/status")
async def status():
    """Health check + current position."""
    try:
        pos = _current_position()
        eq  = _equity()
        return {
            "equity":   eq,
            "position": pos,
            "testnet":  TESTNET,
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webhook_executor:app", host="0.0.0.0", port=8000, reload=False)
