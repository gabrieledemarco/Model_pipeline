"""
Shared helper to derive a filesystem-safe strategy identifier.

Used to namespace per-strategy log directories (logs/strategies/{strategy_id}/)
so multiple LiveTrader processes can run concurrently on the same VPS without
colliding on shared files (log, trades.csv, position.json, meta.json).
"""
from __future__ import annotations

import re


def slugify_strategy_id(scenario: str, exchange: str) -> str:
    """
    Build a filesystem-safe strategy_id slug from scenario name + exchange.

    Rule: lowercase, keep only [a-z0-9_], collapse repeated underscores,
    strip leading/trailing underscores. Any non-alnum character (spaces,
    parentheses, ≥, ±, etc.) becomes an underscore.

    Example: "Strong (≥±18)" + "bybit" -> "strong_18_bybit"
    """
    raw = f"{scenario}_{exchange}".lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug
