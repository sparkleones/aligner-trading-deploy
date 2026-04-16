use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{broadcast, RwLock};
use tokio_stream::{wrappers::BroadcastStream, Stream, StreamExt};
use tonic::{Request, Response, Status};
use tracing::{error, info, warn, instrument};

use crate::order_router::OrderRouter;
use crate::trading;
use crate::trading::execution_engine_server::ExecutionEngine;

/// Shared application state accessible from all gRPC handlers.
pub struct AppState {
    pub order_router: OrderRouter,
    pub tick_sender: broadcast::Sender<trading::TickData>,
    /// Kill switch: when true, all new orders are rejected.
    pub kill_switch: RwLock<bool>,
    /// Current portfolio state (updated by WebSocket tick handler / reconciliation).
    pub portfolio: RwLock<trading::PortfolioState>,
    /// Current risk metrics.
    pub risk_status: RwLock<trading::RiskStatus>,
}

/// The gRPC service implementation.
pub struct ExecutionEngineService {
    state: Arc<AppState>,
}

impl ExecutionEngineService {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ExecutionEngine for ExecutionEngineService {
    // ─── SubmitOrder ──────────────────────────────────────────────────────────

    #[instrument(skip(self, request), fields(
        request_id = %request.get_ref().request_id,
        symbol = %request.get_ref().symbol,
    ))]
    async fn submit_order(
        &self,
        request: Request<trading::OrderRequest>,
    ) -> Result<Response<trading::OrderResponse>, Status> {
        let start = Instant::now();
        let order = request.into_inner();

        info!(
            request_id = %order.request_id,
            symbol = %order.symbol,
            side = order.side,
            quantity = order.quantity,
            order_type = order.order_type,
            price = order.price,
            "SubmitOrder RPC received"
        );

        // Check kill switch
        if *self.state.kill_switch.read().await {
            warn!(
                request_id = %order.request_id,
                "Order rejected — kill switch is active"
            );
            let now_ms = chrono::Utc::now().timestamp_millis();
            return Ok(Response::new(trading::OrderResponse {
                request_id: order.request_id,
                order_id: String::new(),
                success: false,
                message: "Kill switch is active. All orders are rejected.".to_string(),
                timestamp_ms: now_ms,
                latency_ms: start.elapsed().as_millis() as f64,
            }));
        }

        // No delta provided over gRPC for now; could be added as a field
        match self.state.order_router.submit_order(&order, None).await {
            Ok(mut response) => {
                response.latency_ms = start.elapsed().as_millis() as f64;
                info!(
                    request_id = %response.request_id,
                    order_id = %response.order_id,
                    latency_ms = response.latency_ms,
                    "SubmitOrder completed"
                );
                Ok(Response::new(response))
            }
            Err(e) => {
                error!(
                    request_id = %order.request_id,
                    error = %e,
                    latency_ms = start.elapsed().as_millis() as f64,
                    "SubmitOrder failed"
                );
                Err(Status::internal(format!("Order submission failed: {e}")))
            }
        }
    }

    // ─── CancelOrder ──────────────────────────────────────────────────────────

    #[instrument(skip(self, request), fields(order_id = %request.get_ref().order_id))]
    async fn cancel_order(
        &self,
        request: Request<trading::CancelRequest>,
    ) -> Result<Response<trading::OrderResponse>, Status> {
        let start = Instant::now();
        let cancel_req = request.into_inner();

        info!(order_id = %cancel_req.order_id, "CancelOrder RPC received");

        match self
            .state
            .order_router
            .cancel_order(&cancel_req.order_id)
            .await
        {
            Ok(mut response) => {
                response.latency_ms = start.elapsed().as_millis() as f64;
                info!(
                    order_id = %cancel_req.order_id,
                    latency_ms = response.latency_ms,
                    "CancelOrder completed"
                );
                Ok(Response::new(response))
            }
            Err(e) => {
                error!(
                    order_id = %cancel_req.order_id,
                    error = %e,
                    "CancelOrder failed"
                );
                Err(Status::internal(format!("Cancel failed: {e}")))
            }
        }
    }

    // ─── SquareOffAll ─────────────────────────────────────────────────────────

    #[instrument(skip(self, request))]
    async fn square_off_all(
        &self,
        request: Request<trading::SquareOffRequest>,
    ) -> Result<Response<trading::SquareOffResponse>, Status> {
        let start = Instant::now();
        let sq_req = request.into_inner();

        warn!(reason = %sq_req.reason, "SquareOffAll RPC received — EMERGENCY");

        // Activate kill switch
        {
            let mut ks = self.state.kill_switch.write().await;
            *ks = true;
        }
        info!("Kill switch activated");

        // Update risk status
        {
            let mut risk = self.state.risk_status.write().await;
            risk.kill_switch_active = true;
        }

        match self
            .state
            .order_router
            .square_off_all(&sq_req.reason)
            .await
        {
            Ok(response) => {
                info!(
                    orders_cancelled = response.orders_cancelled,
                    positions_closed = response.positions_closed,
                    latency_ms = start.elapsed().as_millis() as f64,
                    "SquareOffAll completed"
                );
                Ok(Response::new(response))
            }
            Err(e) => {
                error!(error = %e, "SquareOffAll failed");
                Err(Status::internal(format!("Square-off failed: {e}")))
            }
        }
    }

    // ─── StreamTicks ──────────────────────────────────────────────────────────

    type StreamTicksStream =
        Pin<Box<dyn Stream<Item = Result<trading::TickData, Status>> + Send + 'static>>;

    #[instrument(skip(self, request))]
    async fn stream_ticks(
        &self,
        request: Request<trading::TickSubscription>,
    ) -> Result<Response<Self::StreamTicksStream>, Status> {
        let subscription = request.into_inner();
        let symbols: std::collections::HashSet<String> =
            subscription.symbols.into_iter().collect();

        info!(
            symbols = ?symbols,
            "StreamTicks subscription started"
        );

        let rx = self.state.tick_sender.subscribe();
        let stream = BroadcastStream::new(rx).filter_map(move |result| {
            match result {
                Ok(tick) => {
                    // Filter by subscribed symbols (empty = all)
                    if symbols.is_empty() || symbols.contains(&tick.symbol) {
                        Some(Ok(tick))
                    } else {
                        None
                    }
                }
                Err(e) => {
                    // Lagged — some ticks were dropped
                    tracing::warn!(error = %e, "Tick stream lagged");
                    None
                }
            }
        });

        Ok(Response::new(Box::pin(stream)))
    }

    // ─── StreamBars ───────────────────────────────────────────────────────────

    type StreamBarsStream =
        Pin<Box<dyn Stream<Item = Result<trading::OhlcvBar, Status>> + Send + 'static>>;

    #[instrument(skip(self, _request))]
    async fn stream_bars(
        &self,
        _request: Request<trading::BarSubscription>,
    ) -> Result<Response<Self::StreamBarsStream>, Status> {
        // Bar aggregation would be implemented here; for now return an empty stream
        // that stays open. In production, a separate bar aggregator would push
        // OHLCVBar messages into a broadcast channel.
        info!("StreamBars subscription started (placeholder)");

        let (_tx, rx) = broadcast::channel::<trading::OhlcvBar>(256);
        let stream = BroadcastStream::new(rx).filter_map(|result| match result {
            Ok(bar) => Some(Ok(bar)),
            Err(_) => None,
        });

        Ok(Response::new(Box::pin(stream)))
    }

    // ─── GetPortfolio ─────────────────────────────────────────────────────────

    #[instrument(skip(self, _request))]
    async fn get_portfolio(
        &self,
        _request: Request<trading::Empty>,
    ) -> Result<Response<trading::PortfolioState>, Status> {
        let start = Instant::now();
        let portfolio = self.state.portfolio.read().await.clone();

        info!(
            positions = portfolio.positions.len(),
            total_mtm = portfolio.total_mtm,
            latency_us = start.elapsed().as_micros() as u64,
            "GetPortfolio response"
        );

        Ok(Response::new(portfolio))
    }

    // ─── GetRiskStatus ────────────────────────────────────────────────────────

    #[instrument(skip(self, _request))]
    async fn get_risk_status(
        &self,
        _request: Request<trading::Empty>,
    ) -> Result<Response<trading::RiskStatus>, Status> {
        let start = Instant::now();
        let risk = self.state.risk_status.read().await.clone();

        info!(
            daily_pnl = risk.daily_pnl,
            kill_switch = risk.kill_switch_active,
            open_positions = risk.open_positions,
            latency_us = start.elapsed().as_micros() as u64,
            "GetRiskStatus response"
        );

        Ok(Response::new(risk))
    }
}
