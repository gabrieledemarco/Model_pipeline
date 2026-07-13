#!/usr/bin/env bash
# Launch a live_trader.py strategy instance in the background, detached from
# the current shell, so it keeps running after you disconnect.
#
# Usage:
#   scripts/start_strategy.sh <EXCHANGE> "<SCENARIO>" [STRATEGY_ID] [CAPITAL] [DRY_RUN]
#
# Examples:
#   scripts/start_strategy.sh bybit "Strong (>=+-18)"
#   scripts/start_strategy.sh bitget "Combined" combined_bitget 10000 true
#
# Required env vars (exchange-dependent), export before calling this script:
#   Bybit:  BYBIT_TESTNET_API_KEY, BYBIT_TESTNET_API_SECRET
#   Bitget: BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE
#
# Multiple strategies can be started concurrently: each run gets its own
# logs/strategies/{strategy_id}/ directory (auto-derived from scenario+exchange
# unless STRATEGY_ID is given), so trades/logs/position never collide.
set -euo pipefail

EXCHANGE="${1:?Usage: $0 <exchange> <scenario> [strategy_id] [capital] [dry_run]}"
SCENARIO="${2:?Usage: $0 <exchange> <scenario> [strategy_id] [capital] [dry_run]}"
STRATEGY_ID="${3:-}"
CAPITAL="${4:-}"
DRY_RUN="${5:-false}"

cd "$(dirname "$0")/.."

ARGS=(--exchange "$EXCHANGE" --scenario "$SCENARIO")
[[ -n "$STRATEGY_ID" ]] && ARGS+=(--strategy-id "$STRATEGY_ID")
[[ -n "$CAPITAL" ]] && ARGS+=(--capital "$CAPITAL")
[[ "$DRY_RUN" == "true" ]] && ARGS+=(--dry-run)

nohup .venv/bin/python3 live_trader.py "${ARGS[@]}" > /dev/null 2>&1 &
echo "Started PID $! -- exchange=$EXCHANGE scenario=$SCENARIO strategy_id=${STRATEGY_ID:-<auto>}"
