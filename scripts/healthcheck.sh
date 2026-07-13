#!/usr/bin/env bash
# Hourly health check for the 7 live_trader.py strategy processes.
# Reports status for each strategy and restarts any that are dead.
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a

STRATEGIES_ROOT="logs/strategies"
declare -A LAUNCH_ARGS=(
  [strong_18_bybit]="--exchange bybit --log-level INFO"
  [wyckoff_spring_upthrust_bybit]="--exchange bybit --strategy-type wyckoff --log-level INFO"
  [ict_silver_bullet_ny_am_pm_bybit]="--exchange bybit --strategy-type ict --log-level INFO"
  [s07_ou_mean_reversion_hmm_gated_bybit]="--exchange bybit --strategy-type ou_hmm --log-level INFO"
  [ml_randomforest_8h_bybit]="--exchange bybit --strategy-type ml_rf_8h --log-level INFO"
  [smc_ltf_v1_structure_baseline_bybit]="--exchange bybit --strategy-type smc_v1 --log-level INFO"
  [smc_ltf_v2_regime_sweep_bybit]="--exchange bybit --strategy-type smc_v2 --log-level INFO"
)

NOW=$(date -u +%s)
echo "=== Healthcheck $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

for sid in "${!LAUNCH_ARGS[@]}"; do
  meta="$STRATEGIES_ROOT/$sid/meta.json"
  log="$STRATEGIES_ROOT/$sid/live_trader.log"
  pid=""
  [ -f "$meta" ] && pid=$(python3 -c "import json;print(json.load(open('$meta')).get('pid',''))" 2>/dev/null)

  alive="no"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o cmd= 2>/dev/null | grep -q "live_trader.py"; then
    alive="yes"
  fi

  last_heartbeat_age="n/a"
  last_signal_line=""
  critical_last_hour=0
  if [ -f "$log" ]; then
    last_ts=$(grep -oE "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" "$log" | tail -1)
    if [ -n "$last_ts" ]; then
      last_epoch=$(date -u -d "$last_ts" +%s 2>/dev/null || echo "")
      [ -n "$last_epoch" ] && last_heartbeat_age="$((NOW - last_epoch))s"
    fi
    last_signal_line=$(grep "Signal=" "$log" | tail -1)
    critical_last_hour=$(awk -v cutoff="$(date -u -d '@'$((NOW-3600)) '+%Y-%m-%d %H:%M:%S')" \
      '$0 > cutoff' "$log" 2>/dev/null | grep -c "CRITICAL" || true)
  fi

  echo "--- $sid ---"
  echo "  pid=$pid alive=$alive last_log_age=$last_heartbeat_age critical_last_1h=$critical_last_hour"
  [ -n "$last_signal_line" ] && echo "  ultimo segnale: $last_signal_line"

  if [ "$alive" = "no" ]; then
    echo "  RIAVVIO in corso..."
    if [ "$sid" = "s07_ou_mean_reversion_hmm_gated_bybit" ]; then
      nohup .venv/bin/python3 live_trader.py ${LAUNCH_ARGS[$sid]} >> /dev/null 2>&1 < /dev/null &
    else
      nohup .venv/bin/python3 live_trader.py ${LAUNCH_ARGS[$sid]} >> "$STRATEGIES_ROOT/$sid/live_stdout.log" 2>&1 < /dev/null &
    fi
    disown
    sleep 3
    echo "  nuovo pid: $!"
  fi
done

echo "=== fine healthcheck ==="
