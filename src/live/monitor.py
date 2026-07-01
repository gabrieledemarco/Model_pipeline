"""
src/live/monitor.py
────────────────────
Lightweight async HTTP server for the paper-trading dashboard.

Endpoints
─────────
GET /          → HTML dashboard (auto-refreshes every 30 s)
GET /api/state → JSON snapshot of all traders
GET /health    → 200 OK  (Render health check)

Usage (internal — called from run_paper_trading.py)::

    from src.live.monitor import start_monitor
    await start_monitor(traders, port=int(os.environ.get("PORT", 8080)))
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict

from aiohttp import web

log = logging.getLogger("live.monitor")

# ── Dashboard HTML template ───────────────────────────────────────────────────

_CSS = """
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:20px}
h1{font-size:1.4rem;color:#fff;border-bottom:2px solid #2196f3;padding-bottom:8px}
h2{font-size:1rem;color:#90caf9;margin-top:24px}
.meta{color:#546e7a;font-size:.8rem;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-top:12px}
.card{background:#12151f;border:1px solid #1e2130;border-radius:8px;padding:16px}
.card h3{margin:0 0 10px;font-size:.95rem;color:#fff}
.label{color:#546e7a;font-size:.78rem;text-transform:uppercase;margin-top:8px}
.value{font-size:1.1rem;font-weight:600;margin-top:2px}
.pos{color:#4caf50}.neg{color:#f44336}.flat{color:#90a4ae}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:600;margin-bottom:8px}
.badge-long{background:#1b5e20;color:#a5d6a7}
.badge-short{background:#b71c1c;color:#ffcdd2}
.badge-flat{background:#1e2130;color:#90a4ae}
table{border-collapse:collapse;width:100%;font-size:.8rem;margin-top:8px}
th{color:#90caf9;text-align:left;padding:4px 8px;border-bottom:1px solid #1e2130}
td{padding:4px 8px;border-bottom:1px solid #1e2130}
.sig-row{background:#0d2a0d}.sig-neg{background:#2a0d0d}
"""

_REFRESH = 30  # seconds


def _pnl_class(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "flat")


def _pos_badge(t) -> str:
    if t.in_position and t._pos is not None:
        side = "LONG" if t._pos.direction == 1 else "SHORT"
        cls  = "badge-long" if t._pos.direction == 1 else "badge-short"
        return f'<span class="badge {cls}">{side} @ {t._pos.entry_px:,.0f}</span>'
    return '<span class="badge badge-flat">FLAT</span>'


def _trader_card(t) -> str:
    pnl  = t.pnl_pct
    dd   = t.current_dd
    wr   = (sum(1 for x in t.trades if x.net_pnl > 0) / len(t.trades) * 100
            if t.trades else 0.0)

    recent = "".join(
        f'<tr class="{"sig-row" if x.net_pnl > 0 else "sig-neg"}">'
        f'<td>{"L" if x.direction==1 else "S"}</td>'
        f'<td>{x.exit_ts[:10]}</td>'
        f'<td class="{_pnl_class(x.net_pnl)}">{x.net_pnl:+,.0f}</td>'
        f'<td>{x.exit_reason}</td>'
        f'</tr>'
        for x in reversed(t.trades[-5:])
    ) if t.trades else '<tr><td colspan="4" style="color:#546e7a">no trades yet</td></tr>'

    return f"""
<div class="card">
  <h3>{t.label}</h3>
  {_pos_badge(t)}
  <div class="label">Equity</div>
  <div class="value">{t.equity:,.0f} USDT</div>
  <div class="label">P&amp;L</div>
  <div class="value {_pnl_class(pnl)}">{pnl:+.2f}%</div>
  <div class="label">Current DD</div>
  <div class="value {_pnl_class(-dd)}">{dd:.2f}%</div>
  <div class="label">Trades / Win-rate</div>
  <div class="value">{len(t.trades)} / {wr:.0f}%</div>
  <div class="label">Recent trades</div>
  <table><thead><tr><th></th><th>Date</th><th>PnL</th><th>Reason</th></tr></thead>
  <tbody>{recent}</tbody></table>
</div>"""


def _build_html(traders: dict, last_signal: dict) -> str:
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cards    = "".join(_trader_card(t) for t in traders.values())
    sig      = last_signal.get("signal", 0)
    comp     = last_signal.get("composite", 0.0)
    ts       = last_signal.get("ts", "—")
    atr      = last_signal.get("atr", 0.0)
    sig_cls  = "pos" if sig > 0 else ("neg" if sig < 0 else "flat")
    sig_str  = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(sig, "—")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="{_REFRESH}">
<title>BTCUSDT Paper Trading</title>
<style>{_CSS}</style>
</head>
<body>
<h1>BTCUSDT Paper Trading — Live Dashboard</h1>
<p class="meta">
  Updated: {now} · Auto-refresh every {_REFRESH}s ·
  Last bar: {ts} ·
  Signal: <span class="{sig_cls}">{sig_str}</span> ·
  Score: {comp:+.1f} · ATR: {atr:,.0f}
</p>
<div class="grid">{cards}</div>
</body>
</html>"""


def _build_json(traders: dict, last_signal: dict) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "last_signal": last_signal,
        "traders": {k: t.to_dict() for k, t in traders.items()},
    }


# ── aiohttp request handlers ──────────────────────────────────────────────────

def make_app(traders: dict, last_signal: dict) -> web.Application:
    """
    Create the aiohttp Application.

    *traders* and *last_signal* are dicts mutated by the paper-trading loop;
    handlers read them on every request (no locking needed for reads in asyncio).
    """
    app = web.Application()

    async def health(request):
        return web.Response(text="OK")

    async def state(request):
        payload = json.dumps(_build_json(traders, last_signal), default=str)
        return web.Response(text=payload, content_type="application/json")

    async def dashboard(request):
        html = _build_html(traders, last_signal)
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/health", health)
    app.router.add_get("/api/state", state)
    app.router.add_get("/", dashboard)
    return app


async def start_monitor(traders: dict, last_signal: dict, port: int = 8080) -> None:
    """
    Start the HTTP monitor server (non-blocking — runs in the background).

    Parameters
    ----------
    traders      : dict[str, PaperTrader] — shared reference, updated live
    last_signal  : dict mutated by LiveRunner with the most-recent bar data
    port         : TCP port to bind (default 8080; Render sets $PORT)
    """
    app    = make_app(traders, last_signal)
    runner = web.AppRunner(app)
    await runner.setup()
    site   = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("Monitor listening on http://0.0.0.0:%d", port)
    print(f"  [monitor] Dashboard → http://0.0.0.0:{port}/")
