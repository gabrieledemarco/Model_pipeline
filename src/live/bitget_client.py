"""
Bitget USDT-Futures client (Simulated Trading / Testnet).

Authentication: HMAC-SHA256 + Base64, headers ACCESS-KEY / ACCESS-SIGN /
ACCESS-TIMESTAMP / ACCESS-PASSPHRASE.
Base URL: https://api.bitget.com  (same endpoint for both live and simulated keys)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Optional
from urllib.parse import urlencode

import requests

BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN  = "USDT"
SYMBOL       = "BTCUSDT"

PRICE_PREC   = 1    # $0.1 tick
QTY_PREC     = 3    # 0.001 BTC minimum step


def _rp(p: float) -> str:
    return str(round(p, PRICE_PREC))


def _rq(q: float) -> str:
    return str(round(q, QTY_PREC))


class BitgetClient:
    """Signed REST client for Bitget Mix V2 (USDT-FUTURES)."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self._session   = requests.Session()

    # ── Signing ───────────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        pre_hash  = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            pre_hash.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time() * 1000))
        return {
            "ACCESS-KEY":        self.api_key,
            "ACCESS-SIGN":       self._sign(ts, method, path, body),
            "ACCESS-TIMESTAMP":  ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type":      "application/json",
            "locale":            "en-US",
        }

    # ── Low-level request helpers ─────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> dict:
        qs       = ("?" + urlencode(params)) if params else ""
        full     = path + qs
        headers  = self._headers("GET", full)
        r        = self._session.get(BASE_URL + full, headers=headers, timeout=10)
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != "00000":
            raise RuntimeError(f"Bitget error {resp.get('code')}: {resp.get('msg')}")
        return resp

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        headers  = self._headers("POST", path, body_str)
        r        = self._session.post(BASE_URL + path, data=body_str, headers=headers, timeout=10)
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != "00000":
            raise RuntimeError(f"Bitget error {resp.get('code')}: {resp.get('msg')}")
        return resp

    # ── Account / position ────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """Single account info for USDT-FUTURES / USDT margin."""
        return self._get("/api/v2/mix/account/account", {
            "productType": PRODUCT_TYPE,
            "marginCoin":  MARGIN_COIN,
            "symbol":      SYMBOL,
        })["data"]

    def get_available_balance(self) -> float:
        acc = self.get_account()
        return float(acc.get("available", 0.0))

    def get_position(self) -> dict:
        """Return current position info (may be empty when flat)."""
        resp = self._get("/api/v2/mix/position/single-position", {
            "productType": PRODUCT_TYPE,
            "symbol":      SYMBOL,
            "marginCoin":  MARGIN_COIN,
        })
        data = resp.get("data", [])
        if not data:
            return {}
        return data[0] if isinstance(data, list) else data

    def get_position_size(self) -> float:
        """Signed position size: +BTC for long, -BTC for short, 0 if flat."""
        pos = self.get_position()
        if not pos:
            return 0.0
        total = float(pos.get("total", 0.0))
        side  = pos.get("holdSide", "")
        return total if side == "long" else -total if side == "short" else 0.0

    # ── Market data (public, no auth) ─────────────────────────────────────────

    def get_mark_price(self) -> float:
        resp = requests.get(
            f"{BASE_URL}/api/v2/mix/market/ticker",
            params={"productType": PRODUCT_TYPE, "symbol": SYMBOL},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "00000":
            raise RuntimeError(f"Bitget error: {data.get('msg')}")
        ticker = data["data"]
        if isinstance(ticker, list):
            ticker = ticker[0]
        return float(ticker["markPrice"])

    # ── Set position mode ─────────────────────────────────────────────────────

    def set_one_way_mode(self) -> None:
        """Set to one-way (non-hedge) position mode."""
        try:
            self._post("/api/v2/mix/account/set-position-mode", {
                "productType": PRODUCT_TYPE,
                "posMode":     "one_way_mode",
            })
        except Exception:
            pass  # Already set

    def set_leverage(self, leverage: int = 1) -> None:
        try:
            self._post("/api/v2/mix/account/set-leverage", {
                "symbol":      SYMBOL,
                "productType": PRODUCT_TYPE,
                "marginCoin":  MARGIN_COIN,
                "leverage":    str(leverage),
            })
        except Exception:
            pass

    def set_margin_mode(self, mode: str = "isolated") -> None:
        try:
            self._post("/api/v2/mix/account/set-margin-mode", {
                "symbol":      SYMBOL,
                "productType": PRODUCT_TYPE,
                "marginCoin":  MARGIN_COIN,
                "marginMode":  mode,
            })
        except Exception:
            pass

    # ── Order placement ───────────────────────────────────────────────────────

    def _order_side(self, direction: int, open_: bool) -> tuple[str, str]:
        """Return (side, tradeSide) for one-way mode."""
        if direction == 1:
            return ("buy",  "open") if open_ else ("sell", "close")
        else:
            return ("sell", "open") if open_ else ("buy",  "close")

    def place_market_entry(self, direction: int, quantity: float) -> dict:
        """Open new position with MARKET order."""
        side, trade_side = self._order_side(direction, open_=True)
        return self._post("/api/v2/mix/order/place-order", {
            "symbol":      SYMBOL,
            "productType": PRODUCT_TYPE,
            "marginMode":  "isolated",
            "marginCoin":  MARGIN_COIN,
            "size":        _rq(quantity),
            "side":        side,
            "tradeSide":   trade_side,
            "orderType":   "market",
            "force":       "ioc",
        })["data"]

    def place_market_close(self, direction: int, quantity: float) -> dict:
        """Close (partial or full) position with MARKET order."""
        side, trade_side = self._order_side(direction, open_=False)
        return self._post("/api/v2/mix/order/place-order", {
            "symbol":      SYMBOL,
            "productType": PRODUCT_TYPE,
            "marginMode":  "isolated",
            "marginCoin":  MARGIN_COIN,
            "size":        _rq(quantity),
            "side":        side,
            "tradeSide":   trade_side,
            "orderType":   "market",
            "force":       "ioc",
            "reduceOnly":  "YES",
        })["data"]

    def place_stop_loss(self, direction: int,
                        stop_price: float, quantity: float) -> Optional[str]:
        """
        Place TPSL plan order as stop-loss (mark-price trigger, market exit).
        Returns orderId or None on failure.
        """
        hold_side = "long" if direction == 1 else "short"
        try:
            resp = self._post("/api/v2/mix/order/place-tpsl-order", {
                "symbol":       SYMBOL,
                "productType":  PRODUCT_TYPE,
                "marginCoin":   MARGIN_COIN,
                "planType":     "loss_plan",
                "triggerPrice": _rp(stop_price),
                "triggerType":  "mark_price",
                "executePrice": "0",     # 0 = market execution at trigger
                "holdSide":     hold_side,
                "size":         _rq(quantity),
            })
            return str(resp["data"].get("orderId", ""))
        except Exception as e:
            return None

    def cancel_tpsl_order(self, order_id: str) -> None:
        """Cancel a specific TPSL plan order."""
        try:
            self._post("/api/v2/mix/order/cancel-plan-order", {
                "symbol":      SYMBOL,
                "productType": PRODUCT_TYPE,
                "marginCoin":  MARGIN_COIN,
                "orderId":     order_id,
            })
        except Exception:
            pass

    def cancel_all_tpsl(self) -> None:
        """Cancel all TPSL plan orders for this symbol."""
        try:
            self._post("/api/v2/mix/order/cancel-all-trigger-order", {
                "productType": PRODUCT_TYPE,
                "symbol":      SYMBOL,
                "planType":    "loss_plan",
            })
        except Exception:
            pass

    # ── Connectivity check ────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            r = requests.get(f"{BASE_URL}/api/v2/public/time", timeout=5)
            data = r.json()
            return data.get("code") == "00000"
        except Exception:
            return False
