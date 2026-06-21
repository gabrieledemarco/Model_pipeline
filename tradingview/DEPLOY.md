# TradingView → Binance Live Trading Pipeline

## Architecture

```
TradingView Chart (Pine Script)
         │  alert fires on signal
         ▼
TradingView Webhook  (Pro+ required)
         │  POST JSON  https://your-vps:8000/webhook?secret=XXX
         ▼
webhook_executor.py  (FastAPI on VPS / cloud)
         │  CCXT
         ▼
Binance Futures API  (BTCUSDT perpetual)
```

---

## Step 1 – TradingView Setup

1. Open **BINANCE:BTCUSDT.P** on a **1H** chart.
2. Open Pine Editor → paste `BTCUSDT_MTF_Strategy.pine` → **Save + Add to chart**.
3. In the Strategy Tester, verify the results broadly match:
   - Win rate ~65 %, Sharpe ~1.5 (Session filter enabled).
4. Create an alert on the indicator:
   - **Condition**: `MTF Long Entry` or `MTF Short Entry`
   - **Trigger**: Once per bar close
   - **Notifications**: Webhook URL → `https://your-vps:8000/webhook?secret=YOUR_SECRET`
   - **Message body**:
     ```json
     {"action":"buy","symbol":"BTCUSDT","score":{{plot("Composite")}},"atr":{{plot("ATR")}}}
     ```
     (use `"sell"` for the short alert)

> ⚠️  TradingView Pro+ ($59.95/mo) is required for webhook alerts.
> Free and Pro users can use email alerts instead and process them manually.

---

## Step 2 – VPS / Server Setup

Any Linux VPS (e.g. DigitalOcean $6/mo, AWS t3.micro, Hetzner CX11).

```bash
git clone https://github.com/gabrieledemarco/model_pipeline.git
cd model_pipeline/tradingview

pip install fastapi uvicorn ccxt python-dotenv

cp .env.example .env          # fill in API keys
uvicorn webhook_executor:app --host 0.0.0.0 --port 8000
```

Create `.env`:
```
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
WEBHOOK_SECRET=choose_a_random_string
USE_TESTNET=true        # set false for live
RISK_PCT=1.0            # 1 % of equity per trade
ATR_SL_MULT=2.0
```

Open port 8000 in firewall, or put Nginx in front with TLS.

---

## Step 3 – Binance API Setup

1. Binance → Account → API Management → Create API Key
2. Permissions needed: **Enable Futures** only (no spot, no withdrawals)
3. Whitelist your VPS IP for extra security
4. Use **testnet** first: https://testnet.binancefuture.com

---

## Step 4 – Test the Pipeline

```bash
# Simulate a long signal
curl -X POST "http://localhost:8000/webhook?secret=YOUR_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"action":"buy","symbol":"BTCUSDT","score":15.5,"atr":1200}'

# Check current position
curl http://localhost:8000/status
```

---

## TP Management

The executor places the SL stop-market order automatically.
For TP partial closes, two options:

**Option A – Additional TradingView alerts** (simplest):
- Create separate alerts for TP1/TP2/TP3 conditions in Pine Script.
- Send `{"action":"close_partial","pct":50}` to a `/close_partial` endpoint.

**Option B – Server-side monitoring loop**:
- After entry, spawn a background task that polls price every minute.
- Closes 50% at TP1, 25% at TP2, 25% at TP3.
- Moves SL to BE after TP1.

---

## Alternative: No-Code with 3Commas

If you don't want to run a server:

1. Create a **3Commas** account → connect Binance API.
2. Create a **Simple Bot** for BTCUSDT.
3. In TradingView alerts, use the 3Commas webhook URL format:
   ```json
   {"message_type":"bot","bot_id":12345,"email_token":"xxx","action":"start_bot"}
   ```
4. 3Commas handles order placement, TP, and SL automatically.

> ⚠️  3Commas cannot replicate the 3-tier partial close exactly —
> it supports a single TP. Use the Python executor for full fidelity.

---

## Risk Checklist (before going live)

- [ ] Run on testnet for at least 2 weeks
- [ ] Verify webhook latency < 2 s
- [ ] Confirm SL orders appear on Binance after each entry
- [ ] Test the `/status` endpoint from TradingView alert (dummy signal)
- [ ] Set a max daily loss circuit-breaker in Binance Risk Management settings
- [ ] Never risk more than 1–2 % per trade until live performance is verified
