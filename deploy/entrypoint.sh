#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Entrypoint: Start Rust execution engine, then Python strategy layer        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

echo "══════════════════════════════════════════════════════════════"
echo "  Indian Options Trading System - Starting Up"
echo "  Timestamp: $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
echo "  Mode: ${PAPER_TRADING:-true}"
echo "══════════════════════════════════════════════════════════════"

# Validate static IP binding (SEBI compliance)
if [ -n "${STATIC_IP:-}" ]; then
    echo "[INFO] Validating static IP binding: ${STATIC_IP}"
    CURRENT_IP=$(curl -s --connect-timeout 5 https://checkip.amazonaws.com/ || echo "unknown")
    if [ "$CURRENT_IP" != "$STATIC_IP" ]; then
        echo "[WARN] Current public IP ($CURRENT_IP) does not match configured STATIC_IP ($STATIC_IP)"
        echo "[WARN] SEBI compliance requires API calls from a registered static IP"
    else
        echo "[OK] Static IP verified: $STATIC_IP"
    fi
else
    echo "[WARN] STATIC_IP not set — SEBI requires a static IP for algo trading API access"
fi

# Verify required environment variables
REQUIRED_VARS=("BROKER_API_KEY" "BROKER_API_SECRET" "BROKER_TOTP_SECRET" "BROKER_USER_ID")
MISSING=0
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR:-}" ]; then
        echo "[WARN] Required env var $VAR is not set"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ] && [ "${PAPER_TRADING:-true}" != "true" ]; then
    echo "[ERROR] Missing $MISSING required environment variables for live trading"
    echo "[ERROR] Set PAPER_TRADING=true or provide all broker credentials"
    exit 1
fi

# Create log directory
mkdir -p /app/logs

# Start Rust execution engine in background
echo "[INFO] Starting Rust execution engine on port ${GRPC_PORT:-50051}..."
/app/bin/execution_engine \
    --grpc-host "${GRPC_HOST:-127.0.0.1}" \
    --grpc-port "${GRPC_PORT:-50051}" \
    --log-level "${LOG_LEVEL:-info}" \
    2>&1 | tee -a /app/logs/execution_engine.log &

RUST_PID=$!
echo "[INFO] Execution engine started (PID: $RUST_PID)"

# Wait for gRPC server to be ready
echo "[INFO] Waiting for gRPC server..."
MAX_RETRIES=30
RETRY=0
while ! curl -s "http://127.0.0.1:${GRPC_PORT:-50051}" >/dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
        echo "[WARN] gRPC server readiness check timed out — proceeding anyway"
        break
    fi
    sleep 1
done

# Start Python strategy layer
echo "[INFO] Starting Python strategy layer..."
exec python /app/main.py \
    --grpc-host "${GRPC_HOST:-127.0.0.1}" \
    --grpc-port "${GRPC_PORT:-50051}" \
    2>&1 | tee -a /app/logs/strategy.log
