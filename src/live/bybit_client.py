"""
Bybit USDT-Perpetual client (V5 API, testnet).

Authentication: HMAC-SHA256 (hex digest), headers X-BAPI-API-KEY /
X-BAPI-SIGN / X-BAPI-TIMESTAMP / X-BAPI-RECV-WINDOW.
Base URL: https://api-testnet.bybit.com
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Optional
from urllib.parse import urlencode

import requests

BASE_URL    = "https://api-demo.bybit.com"
CATEGORY    = "linear"     # USDT-margined perpetuals
SYMBOL      = "BTCUSDT"
RECV_WINDOW = "5000"

PRICE_PREC  = 1    # $0.1 tick
QTY_PREC    = 3    # 0.001 BTC


def _rp(p: float) -> str:
    return str(round(p, PRICE_PREC))


def _rq(q: float) -> str:
    return str(round(q, QTY_PREC))


class BybitClient:
    """Signed REST client for Bybit Linear Perps V5 (testnet).

    Method signatures are intentionally identical to BitgetClient so
    LiveTrader can swap exchanges with no changes.
    """

    def __init__(self, api_key: str, api_secret: str, passphrase: str = ""):
        self.api_key    = api_key
        self.api_secret = api_secret
        # passphrase not used by Bybit; accepted for interface parity
        self._session   = requests.Session()

    # ── Signing ───────────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, params_str: str) -> str:
        pre_hash = timestamp + self.api_key + RECV_WINDOW + params_str
        return hmac.new(
            self.api_secret.encode("utf-8"),
            pre_hash.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

    def _headers(self, timestamp: str, signature: str) -> dict:
        return {
            "X-BAPI-API-KEY":     self.api_key,
            "X-BAPI-SIGN":        signature,
            "X-BAPI-TIMESTAMP":   timestamp,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "Content-Type":       "application/json",
        }

    # ── Low-level request helpers ─────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> dict:
        params  = params or {}
        qs      = urlencode(params)
        ts      = str(int(time.time() * 1000))
        sig     = self._sign(ts, qs)
        url     = BASE_URL + path + ("?" + qs if qs else "")
        r       = self._session.get(url, headers=self._headers(ts, sig), timeout=10)
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {resp.get('retCode')}: {resp.get('retMsg')}")
        return resp

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        ts       = str(int(time.time() * 1000))
        sig      = self._sign(ts, body_str)
        r        = self._session.post(
            BASE_URL + path, data=body_str,
            headers=self._headers(ts, sig), timeout=10,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {resp.get('retCode')}: {resp.get('retMsg')}")
        return resp

    # ── Account / position ────────────────────────────────────────────────────

    def get_available_balance(self) -> float:
        """Fetch available USDT balance for UTA (Unified) or CONTRACT account."""
        try:
            resp   = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            wallet = resp["result"]["list"][0]
            # Try coin-level first (populated when there's a balance)
            for coin in wallet.get("coin", []):
                if coin.get("coin") == "USDT":
                    val = coin.get("availableToWithdraw") or coin.get("walletBalance", 0)
                    return float(val)
            # Fall back to account-level total (UTA aggregated)
            top = float(wallet.get("totalAvailableBalance", 0) or 0)
            if top > 0:
                return top
        except Exception:
            pass
        return 10_000.0

    def get_position(self) -> dict:
        resp  = self._get("/v5/position/list", {"category": CATEGORY, "symbol": SYMBOL})
        items = resp["result"].get("list", [])
        return items[0] if items else {}

    def get_position_size(self) -> float:
        pos  = self.get_position()
        if not pos:
            return 0.0
        size = float(pos.get("size", 0.0))
        side = pos.get("side", "None")
        return size if side == "Buy" else -size if side == "Sell" else 0.0

    # ── Market data (public, no auth) ─────────────────────────────────────────

    def get_mark_price(self) -> float:
        r = requests.get(
            f"{BASE_URL}/v5/market/tickers",
            params={"category": CATEGORY, "symbol": SYMBOL},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data.get('retMsg')}")
        return float(data["result"]["list"][0]["markPrice"])

    # ── Account configuration ─────────────────────────────────────────────────

    def set_one_way_mode(self) -> None:
        """Switch to one-way position mode (mode=0)."""
        try:
            self._post("/v5/position/switch-mode", {
                "category": CATEGORY,
                "symbol":   SYMBOL,
                "mode":     0,
            })
        except Exception:
            pass  # Already set or not needed

    def set_leverage(self, leverage: int = 1) -> None:
        try:
            self._post("/v5/position/set-leverage", {
                "category":     CATEGORY,
                "symbol":       SYMBOL,
                "buyLeverage":  str(leverage),
                "sellLeverage": str(leverage),
            })
        except Exception:
            pass

    def set_margin_mode(self, mode: str = "isolated") -> None:
        """No-op stub — Bybit margin mode is set per-position at order time."""
        pass

    # ── Order placement ───────────────────────────────────────────────────────

    def place_market_entry(self, direction: int, quantity: float) -> dict:
        """Open new position with MARKET order (one-way mode, positionIdx=0)."""
        resp = self._post("/v5/order/create", {
            "category":    CATEGORY,
            "symbol":      SYMBOL,
            "side":        "Buy" if direction == 1 else "Sell",
            "orderType":   "Market",
            "qty":         _rq(quantity),
            "positionIdx": 0,
        })
        return resp["result"]

    def place_market_close(self, direction: int, quantity: float) -> dict:
        """Close (partial or full) position with MARKET order."""
        resp = self._post("/v5/order/create", {
            "category":    CATEGORY,
            "symbol":      SYMBOL,
            "side":        "Sell" if direction == 1 else "Buy",
            "orderType":   "Market",
            "qty":         _rq(quantity),
            "positionIdx": 0,
            "reduceOnly":  True,
        })
        return resp["result"]

    def place_stop_loss(self, direction: int,
                        stop_price: float, quantity: float) -> Optional[str]:
        """
        Place a conditional stop-loss order (MarkPrice trigger, Market exit).
        direction=+1 LONG: triggers when price FALLS below stop_price → triggerDirection=2
        direction=-1 SHORT: triggers when price RISES above stop_price → triggerDirection=1
        """
        try:
            resp = self._post("/v5/order/create", {
                "category":        CATEGORY,
                "symbol":          SYMBOL,
                "side":            "Sell" if direction == 1 else "Buy",
                "orderType":       "Market",
                "qty":             _rq(quantity),
                "triggerPrice":    _rp(stop_price),
                "triggerBy":       "MarkPrice",
                "triggerDirection": 2 if direction == 1 else 1,
                "positionIdx":     0,
                "reduceOnly":      True,
            })
            return str(resp["result"].get("orderId", ""))
        except Exception:
            return None

    def cancel_tpsl_order(self, order_id: str) -> None:
        """Cancel a specific stop order by orderId."""
        try:
            self._post("/v5/order/cancel", {
                "category": CATEGORY,
                "symbol":   SYMBOL,
                "orderId":  order_id,
            })
        except Exception:
            pass

    def cancel_all_tpsl(self) -> None:
        """Cancel all open stop orders for this symbol."""
        try:
            self._post("/v5/order/cancel-all", {
                "category":    CATEGORY,
                "symbol":      SYMBOL,
                "orderFilter": "StopOrder",
            })
        except Exception:
            pass

    # ── Connectivity check ────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            r    = requests.get(f"{BASE_URL}/v5/market/time", timeout=5)
            data = r.json()
            return data.get("retCode") == 0
        except Exception:
            return False
