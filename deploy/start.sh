#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Aligner Trading — Production Startup Script (Docker entrypoint)            ║
# ║  Runs: (1) Trading Engine  (2) PWA Dashboard                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -e

export PYTHONPATH=/app
export TZ=Asia/Kolkata
DASHBOARD_PORT=${DASHBOARD_PORT:-8510}

echo "$(date) | Starting Aligner Trading System..."
echo "$(date) | Capital: ${TRADING_CAPITAL:-30000}"
echo "$(date) | Index: ${DEFAULT_INDEX:-NIFTY}"
echo "$(date) | Dashboard port: $DASHBOARD_PORT"

# ── Start PWA Dashboard (FastAPI + WebSocket) ──
echo "$(date) | Starting dashboard on port $DASHBOARD_PORT..."
python -m uvicorn dashboard.app:app \
    --host 0.0.0.0 \
    --port "$DASHBOARD_PORT" \
    --log-level warning \
    &
DASHBOARD_PID=$!
echo "$(date) | Dashboard PID: $DASHBOARD_PID"

# Wait for dashboard
sleep 5
if curl -sf "http://localhost:$DASHBOARD_PORT/health" > /dev/null 2>&1; then
    echo "$(date) | Dashboard ready at http://0.0.0.0:$DASHBOARD_PORT"
else
    echo "$(date) | WARNING: Dashboard health check failed, continuing anyway"
fi

# ── Start Trading Engine ──
echo "$(date) | Starting autonomous trading engine..."
mkdir -p /app/logs
python run_autonomous.py 2>&1 | tee -a /app/logs/engine_stdout.log &
ENGINE_PID=$!
echo "$(date) | Engine PID: $ENGINE_PID"

# ── Health Monitor Loop ──
echo "$(date) | All services running. Monitoring..."

while true; do
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        echo "$(date) | ENGINE DIED — restarting in 10s..."
        sleep 10
        python run_autonomous.py 2>&1 | tee -a /app/logs/engine_stdout.log &
        ENGINE_PID=$!
        echo "$(date) | Engine restarted with PID: $ENGINE_PID"
    fi

    if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
        echo "$(date) | DASHBOARD DIED — restarting..."
        python -m uvicorn dashboard.app:app --host 0.0.0.0 --port "$DASHBOARD_PORT" --log-level warning &
        DASHBOARD_PID=$!
        echo "$(date) | Dashboard restarted with PID: $DASHBOARD_PID"
    fi

    sleep 30
done
