"""
Binance FAPI Testnet REST client.

Order execution → testnet.binancefuture.com  (requires API keys)
Market data    → fapi.binance.com            (no keys needed)
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional
from urllib.parse import urlencode

import requests

TESTNET_BASE = "https://testnet.binancefuture.com"
PROD_BASE    = "https://fapi.binance.com"

SYMBOL        = "BTCUSDT"
PRICE_PREC    = 1    # round to $0.1
QTY_PREC      = 3   # 0.001 BTC minimum step


def _round_price(p: float) -> float:
    return round(p, PRICE_PREC)


def _round_qty(q: float) -> float:
    return round(q, QTY_PREC)


class BinanceTestnetClient:
    """Signed REST client for Binance FAPI testnet."""

    def __init__(self, api_key: str, api_secret: str, recv_window: int = 5000):
        self.api_key     = api_key
        self.api_secret  = api_secret
        self.recv_window = recv_window
        self._session    = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": api_key})

    # ── Signing ───────────────────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = self.recv_window
        qs  = urlencode(params)
        sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    # ── Low-level request helpers ─────────────────────────────────────────────

    def _get(self, path: str, params: dict = None, signed: bool = True) -> dict | list:
        p = dict(params or {})
        if signed:
            p = self._sign(p)
        r = self._session.get(f"{TESTNET_BASE}{path}", params=p, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: dict) -> dict:
        p = self._sign(dict(params))
        r = self._session.post(f"{TESTNET_BASE}{path}", params=p, timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, params: dict) -> dict:
        p = self._sign(dict(params))
        r = self._session.delete(f"{TESTNET_BASE}{path}", params=p, timeout=10)
        r.raise_for_status()
        return r.json()

    # ── Account / position ────────────────────────────────────────────────────

    def get_account(self) -> dict:
        return self._get("/fapi/v2/account")

    def get_available_balance(self) -> float:
        """Available USDT cross-margin balance."""
        acc = self.get_account()
        for a in acc.get("assets", []):
            if a["asset"] == "USDT":
                return float(a["availableBalance"])
        return 0.0

    def get_position_info(self, symbol: str = SYMBOL) -> dict:
        """Return position dict for symbol (positionAmt, entryPrice, etc.)."""
        acc = self.get_account()
        for p in acc.get("positions", []):
            if p["symbol"] == symbol:
                return p
        return {}

    def get_position_size(self, symbol: str = SYMBOL) -> float:
        """Signed position size (+ long, - short, 0 flat)."""
        pos = self.get_position_info(symbol)
        return float(pos.get("positionAmt", 0.0))

    # ── Market data (production endpoint, no signature needed) ────────────────

    def get_mark_price(self, symbol: str = SYMBOL) -> float:
        r = requests.get(f"{PROD_BASE}/fapi/v1/premiumIndex",
                         params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return float(r.json()["markPrice"])

    def get_server_time(self) -> int:
        r = requests.get(f"{TESTNET_BASE}/fapi/v1/time", timeout=10)
        r.raise_for_status()
        return int(r.json()["serverTime"])

    # ── Order management ──────────────────────────────────────────────────────

    def place_market(self, symbol: str, side: str, quantity: float) -> dict:
        """Place MARKET entry order. side: 'BUY' or 'SELL'."""
        return self._post("/fapi/v1/order", {
            "symbol":   symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": _round_qty(quantity),
        })

    def place_reduce_market(self, symbol: str, side: str, quantity: float) -> dict:
        """Partial close via reduceOnly MARKET."""
        return self._post("/fapi/v1/order", {
            "symbol":    symbol,
            "side":      side,
            "type":      "MARKET",
            "quantity":  _round_qty(quantity),
            "reduceOnly": "true",
        })

    def place_stop_market(self, symbol: str, side: str,
                          stop_price: float, quantity: float) -> dict:
        """STOP_MARKET reduceOnly for stop-loss."""
        return self._post("/fapi/v1/order", {
            "symbol":    symbol,
            "side":      side,
            "type":      "STOP_MARKET",
            "stopPrice": _round_price(stop_price),
            "quantity":  _round_qty(quantity),
            "reduceOnly": "true",
        })

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._delete("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def cancel_all_orders(self, symbol: str = SYMBOL) -> dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    def get_open_orders(self, symbol: str = SYMBOL) -> list:
        return self._get("/fapi/v1/openOrders", {"symbol": symbol})

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def set_leverage(self, symbol: str = SYMBOL, leverage: int = 1) -> dict:
        return self._post("/fapi/v1/leverage",
                          {"symbol": symbol, "leverage": leverage})

    def set_isolated_margin(self, symbol: str = SYMBOL) -> None:
        """Set ISOLATED margin type. Ignores error if already set."""
        try:
            self._post("/fapi/v1/marginType",
                       {"symbol": symbol, "marginType": "ISOLATED"})
        except requests.HTTPError as e:
            if "already" not in str(e).lower():
                raise

    def ping(self) -> bool:
        """Verify connectivity to testnet."""
        try:
            r = self._session.get(f"{TESTNET_BASE}/fapi/v1/ping", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
