# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Indian Options Algorithmic Trading System                                  ║
# ║  Multi-stage build: Rust execution engine + Python strategy layer           ║
# ║  Target: AWS EC2 M8g (Graviton4) in ap-south-1 (Mumbai)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: Build Rust Execution Engine ─────────────────────────────────────
FROM rust:1.77-bookworm AS rust-builder

WORKDIR /build

# Install protobuf compiler for gRPC
RUN apt-get update && apt-get install -y protobuf-compiler libprotobuf-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy Rust project files
COPY execution_engine/Cargo.toml execution_engine/Cargo.lock* ./execution_engine/
COPY execution_engine/build.rs ./execution_engine/
COPY execution_engine/src/ ./execution_engine/src/
COPY proto/ ./proto/

# Build release binary
WORKDIR /build/execution_engine
RUN cargo build --release

# ── Stage 2: Python Runtime ─────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS python-base

# System dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gRPC tools and compile proto
COPY proto/ ./proto/
RUN python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    ./proto/trading.proto

# ── Stage 3: Final Runtime Image ────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    curl \
    iproute2 \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r trader && useradd -r -g trader -m -s /bin/bash trader

WORKDIR /app

# Copy Rust binary from builder
COPY --from=rust-builder /build/execution_engine/target/release/execution_engine /app/bin/execution_engine

# Copy Python environment from python-base
COPY --from=python-base /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=python-base /usr/local/bin/ /usr/local/bin/

# Copy compiled proto
COPY --from=python-base /app/proto/ /app/proto/

# Copy application code
COPY config/ ./config/
COPY strategy/ ./strategy/
COPY risk_management/ ./risk_management/
COPY backtesting/ ./backtesting/
COPY broker/ ./broker/
COPY main.py .

# Create required directories
RUN mkdir -p /app/logs /app/data/cache /app/models && \
    chown -R trader:trader /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV GRPC_HOST=127.0.0.1
ENV GRPC_PORT=50051
ENV LOG_LEVEL=INFO
ENV LOG_FILE=/app/logs/trading.log
ENV PAPER_TRADING=true
ENV AWS_REGION=ap-south-1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 50051 8080

# Switch to non-root user
USER trader

# Entry point: start Rust engine, then Python strategy
COPY deploy/entrypoint.sh /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
