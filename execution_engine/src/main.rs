mod grpc_server;
mod order_router;
mod rate_limiter;
mod websocket;

pub mod trading {
    tonic::include_proto!("trading");
}

use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};
use tonic::transport::Server;
use tracing::{info, error};
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

use crate::grpc_server::{AppState, ExecutionEngineService};
use crate::order_router::OrderRouter;
use crate::rate_limiter::RateLimiter;
use crate::trading::execution_engine_server::ExecutionEngineServer;
use crate::websocket::WebSocketManager;

/// Default gRPC listen address.
const GRPC_ADDR: &str = "0.0.0.0:50051";

/// Broker WebSocket URL (override via BROKER_WS_URL env var).
const DEFAULT_WS_URL: &str = "wss://ws.broker.example.com/ws";

/// Broker REST URL for state reconciliation (override via BROKER_REST_URL env var).
const DEFAULT_REST_URL: &str = "https://api.broker.example.com";

/// Tick broadcast channel capacity.
const TICK_CHANNEL_CAPACITY: usize = 4096;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ─── Logging ──────────────────────────────────────────────────────────────
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new("execution_engine=debug,tower=info,tonic=info")
        }))
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .with_timer(fmt::time::SystemTime::default()),
        )
        .init();

    info!("Execution engine starting");

    // ─── Configuration ────────────────────────────────────────────────────────
    let grpc_addr = std::env::var("GRPC_ADDR")
        .unwrap_or_else(|_| GRPC_ADDR.to_string())
        .parse()?;

    let ws_url = std::env::var("BROKER_WS_URL")
        .unwrap_or_else(|_| DEFAULT_WS_URL.to_string());

    let rest_url = std::env::var("BROKER_REST_URL")
        .unwrap_or_else(|_| DEFAULT_REST_URL.to_string());

    // ─── Shared components ────────────────────────────────────────────────────
    let (tick_sender, _) = broadcast::channel::<trading::TickData>(TICK_CHANNEL_CAPACITY);
    let rate_limiter = RateLimiter::new();
    let order_router = OrderRouter::new(rate_limiter);

    let app_state = Arc::new(AppState {
        order_router,
        tick_sender: tick_sender.clone(),
        kill_switch: RwLock::new(false),
        portfolio: RwLock::new(trading::PortfolioState {
            positions: vec![],
            total_mtm: 0.0,
            total_capital: 0.0,
            used_margin: 0.0,
            available_margin: 0.0,
            timestamp_ms: 0,
        }),
        risk_status: RwLock::new(trading::RiskStatus {
            daily_pnl: 0.0,
            daily_pnl_pct: 0.0,
            max_loss_threshold_pct: -2.0,
            kill_switch_active: false,
            open_positions: 0,
            pending_orders: 0,
        }),
    });

    // ─── WebSocket manager ────────────────────────────────────────────────────
    let ws_manager = WebSocketManager::new(ws_url.clone(), rest_url.clone(), tick_sender);

    let ws_handle = tokio::spawn(async move {
        ws_manager.run().await;
    });

    info!(
        ws_url = %ws_url,
        rest_url = %rest_url,
        "WebSocket manager started"
    );

    // ─── gRPC server ──────────────────────────────────────────────────────────
    let grpc_service = ExecutionEngineService::new(app_state);
    let grpc_server = ExecutionEngineServer::new(grpc_service);

    info!(addr = %grpc_addr, "gRPC server listening");

    let grpc_handle = tokio::spawn(async move {
        if let Err(e) = Server::builder()
            .add_service(grpc_server)
            .serve(grpc_addr)
            .await
        {
            error!(error = %e, "gRPC server error");
        }
    });

    // ─── Wait for tasks ───────────────────────────────────────────────────────
    tokio::select! {
        _ = grpc_handle => {
            error!("gRPC server exited unexpectedly");
        }
        _ = ws_handle => {
            error!("WebSocket manager exited unexpectedly");
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received Ctrl+C, shutting down gracefully");
        }
    }

    info!("Execution engine shut down");
    Ok(())
}
